"""Volumetric refiner (issue #41): every view fused into one voxel grid, refined in 3D.

The per-view refiners (the energy, `learned.py`) see a thin object and the wall behind
it as image neighbours. Here all pixels of all views are lifted, x = o + D d along Deja
View's pinhole rays, into one grid over the scene, as MoGe-3 (Wang et al. 2026) refines
in a sparse voxel shell, Pixal3D (2026) puts pixel features into a pixel-aligned volume
and DFusion (Niu et al. 2022) fuses depth maps before a 3D U-Net. Per cell the grid
holds the log point count, the mean confidence and colour, the spread of its points (the
views' layering, in cell units) and, with semantics, the histogram of its points'
classes (NYU40, from SAM 3's per-view masks or Hypersim's instances). A 3D U-Net turns
it into features; every pixel samples them at its point (trilinear), adds its own
colour, confidence, offset in the cell, ray direction and class, and a small head
predicts r: the refined depth is D exp(r). The head's last layer starts at zero, so an
untrained refiner returns its input bit for bit.

Target: the ground truth's points, mapped into Deja View's frame by the Umeyama
similarity of Deja View's points onto them (the evaluation's alignment), are projected
onto each pixel's ray, t* = (x_gt - o) . d / |d|^2, and r* = log(t* / D); pixels without
ground truth or with |r*| > `MAX_RESIDUAL` are left out. Loss: L1 on r. `volume-refine`
trains every model of the config on the Hypersim train split and writes
`<model>_<views>.npz` for the prediction scenes, scored by `scene_eval` like a backbone.

Keep head (issue #57, config `keep`): the voxel density filter (E45) wins by deleting
points, which this refiner cannot. With `keep` the head has a second output, a keep
logit per pixel, trained by binary cross-entropy to tell whether the pixel's remaining
error |r* - r| (r = 0 without the residual) lies in the lower `quantile` of the step's
pixels; pixels with ground truth but no valid r* (|r*| > `MAX_RESIDUAL`, behind the
camera) count as errors. The cut is a quantile, not a length, so the label needs no
unit and matches the evaluation's per-scene cut. The logit is written per pixel as
`<model>_keep_<views>.npz`, and `scene_eval`'s `keep_by` ranks by it instead of instance
voxel density, at the same kept fraction. The held-out step kept is then the one whose
kept pixels (top `quantile` by logit) have the least mean error.

Distance head (issue #57, E57c, `keep.target: distance`): E57a's labels are an error
quantile, spatially smooth, and its cut opened holes. Here the second output regresses
what the filter ranks by: the distance from the pixel's point to the ground-truth
surface, the nearest point of the ground-truth cloud in Deja View's frame (the
evaluation's accuracy before its own alignment), in units of the scene's median depth.
L1 loss, so the output is a conditional median and heavy tails (a point behind a wall)
do not dominate. Every pixel has such a distance, also those without ground-truth depth,
so no pixel needs the "count as an error" rule. The score written for `scene_eval` is
the negated prediction, so the points of largest predicted distance are the ones the cut
drops.
"""

from __future__ import annotations

import argparse
import dataclasses
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import Tensor, nn
from torch.nn import functional as F  # noqa: N812
from torch.utils.tensorboard import SummaryWriter

from src.analysis.scene_eval import git_hash
from src.device import best_device
from src.evaluation.labeled import lift, nearest_distances, umeyama
from src.geometry import Geometry
from src.loaders.ground_truth import GroundTruth
from src.loaders.hypersim.config import load_config
from src.loaders.hypersim.scene import NYU40
from src.loaders.views import Views
from src.refinement.multiview import scene_from_geometry
from src.segmentation.base import Segmentation

CLASSES = len(NYU40)  # index 0: no class
MAX_RESIDUAL = 0.5  # |log(t* / D)| above this is a misregistered pixel, not a target
SEMANTICS = ("none", "sam3", "gt")
KEEP_TARGETS = ("label", "distance")  # E57a: error-quantile labels; E57c: distance
CHUNK = 500_000  # pixels per forward pass at prediction


@dataclasses.dataclass(frozen=True)
class KeepHead:
    """The keep score's training (issue #57)."""

    quantile: float  # kept fraction: the label's cut, and the reported kept error
    residual: bool  # also train the depth residual; else the depth stays the backbone's
    weight: float  # of the keep loss (cross-entropy or distance L1) next to the L1 on r
    target: str = "label"  # one of KEEP_TARGETS


@dataclasses.dataclass(frozen=True)
class VolumeConfig:
    """Data, models and training settings (`configs/e41b_volume.yaml`)."""

    root: Path
    views: str
    backbone: str
    segmentation: str
    train: list[str]  # scene folders of `train_splits`; the last `held_out` only report
    held_out: int  # 0: report on the train scenes themselves (the overfit rungs)
    predict: list[str]
    models: dict[str, str]  # name -> semantics: none, sam3 or gt
    resolution: int  # cells along the scene box's longest side
    widths: list[int]  # U-Net channels per level
    head: int
    steps: int
    batch_points: int  # pixels per step
    kept_points: int  # pixels per held-out scene the error is reported on
    view_subset: list[
        int
    ]  # each step's grid sees a random number of views in this range
    amp: bool  # U-Net in bfloat16 on CUDA
    lr: float
    seeds: list[int]  # one model per seed; with several, names end in -s<seed>
    keep: KeepHead | None = None  # a keep score next to the residual (issue #57)
    val: list[str] = dataclasses.field(default_factory=list)  # curves only, never fit

    @classmethod
    def load(cls, path: Path) -> VolumeConfig:
        """Read the config; train and prediction scenes come from Hypersim splits."""
        raw = yaml.safe_load(path.read_text())
        split = load_config(Path(raw.pop("split")))
        if "train_scenes" in raw:  # the overfit rungs name their scenes
            assert "train_splits" not in raw, "train_scenes replaces train_splits"
            raw["train"] = raw.pop("train_scenes")
        else:
            train = split.scenes(raw.pop("train_splits"))
            raw["train"] = [f"{s}_{c}" for s, c in train.items()]
        val = split.scenes(raw.pop("val_splits")) if "val_splits" in raw else {}
        raw["val"] = [f"{s}_{c}" for s, c in val.items()]
        predict = split.scenes(raw.pop("predict_splits"))
        raw["predict"] = [f"{s}_{c}" for s, c in predict.items()]
        raw["root"] = Path(raw["root"])
        assert all(s in SEMANTICS for s in raw["models"].values()), raw["models"]
        if "keep" in raw:
            raw["keep"] = KeepHead(**raw["keep"])
            assert 0 < raw["keep"].quantile < 1, raw["keep"]
            assert raw["keep"].target in KEEP_TARGETS, raw["keep"]
            if raw["keep"].target == "distance":
                assert not raw["keep"].residual, raw["keep"]
        return cls(**raw)


@dataclasses.dataclass(frozen=True)
class ScenePoints:
    """One scene's pixels, flat (N = V H W): points in the grid's frame, features."""

    points: Tensor  # (N, 3) in the normalized frame
    rays: Tensor  # (N, 3) unit ray directions
    color: Tensor  # (N, 3) in [0, 1]
    log_conf: Tensor  # (N,) mean 0 per scene
    classes: Tensor  # (N,) NYU40 index, 0 none
    lo: Tensor  # (3,) grid corner
    cell: float  # cell size in the normalized frame
    dims: tuple[int, int, int]  # cells along x, y, z
    patches: Tensor | None = None  # (V, C, h, w) DINOv2 features per view, if any
    image: tuple[int, int] = (0, 0)  # H, W of the views `patches` belong to


def compact(pts: ScenePoints) -> ScenePoints:
    """Return the scene in small dtypes for the CPU: uint8 colour, float16 rays."""
    return dataclasses.replace(
        pts,
        rays=pts.rays.half(),
        color=(pts.color * 255).round().to(torch.uint8),
        log_conf=pts.log_conf.half(),
        classes=pts.classes.to(torch.uint8),
    )


def expand(pts: ScenePoints, sel: Tensor, device: torch.device) -> ScenePoints:
    """Return the selected pixels of a compact scene as float32 tensors on device."""
    return dataclasses.replace(
        pts,
        points=pts.points[sel].to(device),
        rays=pts.rays[sel].to(device).float(),
        color=pts.color[sel].to(device).float() / 255,
        log_conf=pts.log_conf[sel].to(device).float(),
        classes=pts.classes[sel].to(device).long(),
        lo=pts.lo.to(device),
    )


def view_pixels(views: np.ndarray, pixels: int) -> Tensor:
    """Flat indices of every pixel of the given views (view-major order)."""
    return torch.cat([torch.arange(v * pixels, (v + 1) * pixels) for v in views])


def pixel_classes(ids: np.ndarray, classes: dict[int, str]) -> np.ndarray:
    """NYU40 index (V, H, W) of every pixel's instance; 0 where none or unknown."""
    lut = {i: NYU40.index(c) for i, c in classes.items() if c in NYU40}
    out = np.zeros(ids.shape, dtype=np.int64)
    for i, c in lut.items():
        out[ids == i] = c
    return out


def grid_box(
    points: np.ndarray,
    resolution: int,
    *,
    percentile: float = 1.0,
    pad: float = 0.05,
    cell: float | None = None,
) -> tuple[np.ndarray, float, tuple[int, int, int]]:
    """Corner, cell, cells per axis (x8) of the padded p..100-p box."""
    assert 0 <= percentile < 100 - percentile and pad >= 0, (percentile, pad)
    lo, hi = np.percentile(points, [percentile, 100 - percentile], axis=0)
    margin = pad * (hi - lo)
    lo, hi = lo - margin, hi + margin
    cell = float((hi - lo).max() / resolution) if cell is None else cell
    dims = tuple(
        int(8 * np.ceil((h - low) / cell / 8)) for low, h in zip(lo, hi, strict=True)
    )
    return lo, cell, dims


def scene_points(  # noqa: PLR0913, PLR0917 - the box's extent is one more grid setting
    geometry: Geometry,
    images: np.ndarray,
    ids: np.ndarray,
    classes: dict[int, str],
    resolution: int,
    box: tuple[float, float] = (1.0, 0.05),
) -> tuple[ScenePoints, np.ndarray, np.ndarray, float]:
    """Flat pixels in the grid's frame; also the raw rays (N, 6), depth and scale.

    The cell is the padded 1-99 % box's longest side / `resolution`; the grid spans
    the `box` = (percentile, pad) box at that cell (E69f: (0.1, 0.05) keeps the seen
    walls the 1-99 % box clips, without coarsening the cell).
    """
    scene = scene_from_geometry(geometry)
    rays = scene.rays.reshape(-1, 6).double().numpy()
    depth = geometry.depth.reshape(-1).astype(np.float64)
    scale = float(np.median(depth))
    x = (rays[:, :3] + depth[:, None] * rays[:, 3:]) / scale
    cell = grid_box(x, resolution)[1]
    lo, cell, dims = grid_box(x, resolution, percentile=box[0], pad=box[1], cell=cell)
    log_conf = np.log(geometry.confidence.reshape(-1).astype(np.float64))
    direction = rays[:, 3:] / np.linalg.norm(rays[:, 3:], axis=1, keepdims=True)
    pts = ScenePoints(
        torch.from_numpy(x).float(),
        torch.from_numpy(direction).float(),
        torch.from_numpy(images.reshape(-1, 3) / 255).float(),
        torch.from_numpy(log_conf - log_conf.mean()).float(),
        torch.from_numpy(pixel_classes(ids, classes).reshape(-1)),
        torch.from_numpy(lo).float(),
        cell,
        dims,
    )
    return pts, rays, depth, scale


def gt_alignment(
    rays: np.ndarray, depth: np.ndarray, views: Views, gt: GroundTruth
) -> tuple[np.ndarray, np.ndarray, tuple[float, np.ndarray, np.ndarray]]:
    """GT points (N, 3), where the GT has one (N,), Umeyama (s, R, t) onto them.

    The similarity maps Deja View's points onto the GT pixels' points (the
    evaluation's alignment): gt = s R dv + t.
    """
    gt_points = np.stack(
        [
            lift(d.astype(np.float64), k.astype(np.float64), c2w)
            for d, k, c2w in zip(
                gt.depth, views.intrinsics, views.cam_to_world, strict=True
            )
        ]
    ).reshape(-1, 3)
    has = gt.depth.reshape(-1) > 0
    pred = rays[:, :3] + depth[:, None] * rays[:, 3:]
    return gt_points, has, umeyama(pred[has], gt_points[has])


def gt_in_frame(
    rays: np.ndarray, depth: np.ndarray, views: Views, gt: GroundTruth
) -> tuple[np.ndarray, np.ndarray]:
    """GT points (N, 3) in Deja View's frame (Umeyama onto them) and where GT is."""
    gt_points, has, (s, rotation, t) = gt_alignment(rays, depth, views, gt)
    return (gt_points - t) @ rotation / s, has


def target_residual(
    rays: np.ndarray, depth: np.ndarray, views: Views, gt: GroundTruth
) -> tuple[np.ndarray, np.ndarray]:
    """Return r* = log(t* / D) per pixel (N,) and where it is valid (Umeyama)."""
    in_dv, has = gt_in_frame(rays, depth, views, gt)
    d = rays[:, 3:]
    t_star = ((in_dv - rays[:, :3]) * d).sum(1) / (d * d).sum(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.log(t_star / depth)
    valid = has & (t_star > 0) & (np.abs(r) <= MAX_RESIDUAL)
    return np.where(valid, r, 0).astype(np.float32), valid


def distance_target(
    rays: np.ndarray, depth: np.ndarray, views: Views, gt: GroundTruth, scale: float
) -> np.ndarray:
    """Distance (N,) of every pixel's point to the GT cloud, in units of `scale`."""
    assert depth.ndim == 1 and rays.shape == (len(depth), 6) and scale > 0, rays.shape
    in_dv, has = gt_in_frame(rays, depth, views, gt)
    points = rays[:, :3] + depth[:, None] * rays[:, 3:]
    out = nearest_distances(in_dv[has] / scale, points / scale).astype(np.float32)
    assert out.shape == depth.shape and np.isfinite(out).all(), out.shape
    return out


def cell_index(pts: ScenePoints) -> Tensor:
    """Flat z-major index (N,) of each point's grid cell, clamped into the grid."""
    nx, ny, _ = pts.dims
    idx3 = ((pts.points - pts.lo) / pts.cell).floor().long()
    idx3 = torch.stack([idx3[:, k].clamp(0, n - 1) for k, n in enumerate(pts.dims)], 1)
    return (idx3[:, 2] * ny + idx3[:, 1]) * nx + idx3[:, 0]


def voxelize_features(
    pts: ScenePoints, patches: Tensor, size: tuple[int, int]
) -> Tensor:
    """Grid (C, Z, Y, X) of the mean feature of the pixels in each cell (0 where none).

    patches (V, C, h, w): a feature per patch of each view (`backbone/dinov2.py`);
    pts holds the V * H * W pixels view-major, size = (H, W). Each pixel's feature is
    the patch grid upsampled bilinearly to the pixel (patch centres at (i + .5) H / h);
    a cell's feature is the mean over all its pixels, from every view (Pixal3D, 2026).
    """
    height, width = size
    views, channels = patches.shape[:2]
    assert len(pts.points) == views * height * width, (len(pts.points), patches.shape)
    nx, ny, nz = pts.dims
    flat = cell_index(pts)
    sums = torch.zeros(nx * ny * nz, channels, device=pts.points.device)
    for v in range(views):
        up = (
            F.interpolate(
                patches[v : v + 1].to(pts.points.device).float(),
                size=size,
                mode="bilinear",
                align_corners=False,
            )[0]
            .reshape(channels, -1)
            .T
        )
        sums.index_add_(0, flat[v * height * width : (v + 1) * height * width], up)
    count = torch.zeros(nx * ny * nz, device=sums.device).index_add_(
        0, flat, torch.ones(len(flat), device=sums.device)
    )
    grid = (sums / count.clamp(min=1)[:, None]).T.reshape(channels, nz, ny, nx)
    assert grid.isfinite().all()
    return grid


def voxelize(pts: ScenePoints, *, semantics: bool) -> Tensor:
    """Grid (C, Z, Y, X): log count, conf, colour, spread, class histogram, features.

    The feature channels (`voxelize_features`) are appended where the scene has patches.
    """
    nx, ny, nz = pts.dims
    flat = cell_index(pts)
    cells = nx * ny * nz
    local = (pts.points - pts.lo) / pts.cell
    values = torch.cat([pts.log_conf[:, None], pts.color, local, local**2], 1)
    dev = pts.points.device
    ones = torch.ones(len(flat), device=dev)
    count = torch.zeros(cells, device=dev).index_add_(0, flat, ones)
    sums = torch.zeros(cells, values.shape[1], device=dev).index_add_(0, flat, values)
    mean = sums / count.clamp(min=1)[:, None]
    spread = (mean[:, 7:10] - mean[:, 4:7] ** 2).clamp(min=0).sqrt()
    feats = [count.log1p()[:, None], mean[:, :4], spread]
    if semantics:
        hist = torch.zeros(cells * CLASSES, device=dev).index_add_(
            0, flat * CLASSES + pts.classes, ones
        )
        feats.append(hist.view(cells, CLASSES) / count.clamp(min=1)[:, None])
    grid = torch.cat(feats, 1).T.reshape(-1, nz, ny, nx)
    if pts.patches is not None:
        grid = torch.cat([grid, voxelize_features(pts, pts.patches, pts.image)])
    assert grid.isfinite().all()
    return grid


def block(c_in: int, c_out: int) -> nn.Sequential:
    """Two 3x3x3 convolutions with GroupNorm and GELU."""
    return nn.Sequential(
        nn.Conv3d(c_in, c_out, 3, padding=1), nn.GroupNorm(8, c_out), nn.GELU(),
        nn.Conv3d(c_out, c_out, 3, padding=1), nn.GroupNorm(8, c_out), nn.GELU(),
    )  # fmt: skip


class UNet3d(nn.Module):
    """3D U-Net on a grid (C, Z, Y, X): one block per width, average-pool down."""

    def __init__(self, grid_channels: int, widths: list[int]) -> None:
        """Widths per level; the last output has `widths[0]` channels at full size."""
        super().__init__()
        self.down = nn.ModuleList()
        c = grid_channels
        for w in widths:
            self.down.append(block(c, w))
            c = w
        self.up = nn.ModuleList(
            block(widths[i] + widths[i + 1], widths[i]) for i in range(len(widths) - 1)
        )

    def pyramid(self, grid: Tensor, *, amp: bool = False) -> list[Tensor]:
        """Return the decoder features (C_i, Z_i, Y_i, X_i), full size first.

        grid: (C_in, Z, Y, X); the sizes halve per level; bfloat16 with amp.
        """
        if amp and grid.is_cuda:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                return [level.float() for level in self.pyramid(grid)]
        skips, x = [], grid[None]
        for i, enc in enumerate(self.down):
            x = enc(x if i == 0 else F.avg_pool3d(x, 2))
            skips.append(x)
        levels = [x[0]]
        for i in reversed(range(len(self.up))):
            x = F.interpolate(
                x, size=skips[i].shape[2:], mode="trilinear", align_corners=False
            )
            x = self.up[i](torch.cat([x, skips[i]], 1))
            levels.append(x[0])
        return levels[::-1]

    def features(self, grid: Tensor, *, amp: bool = False) -> Tensor:
        """U-Net features (C, Z, Y, X) of a grid (C_in, Z, Y, X); bfloat16 with amp."""
        return self.pyramid(grid, amp=amp)[0]


class VolumeRefiner(UNet3d):
    """3D U-Net on the grid, then a per-pixel head on the sampled features."""

    def __init__(
        self,
        grid_channels: int,
        widths: list[int],
        head: int,
        *,
        semantics: bool,
        outputs: int = 1,
    ) -> None:
        """Widths per U-Net level; outputs r (and the keep logit), zero at the start."""
        super().__init__(grid_channels, widths)
        self.embed = nn.Embedding(CLASSES, 8) if semantics else None
        point = 3 + 1 + 3 + 3 + (8 if semantics else 0)
        self.head = nn.Sequential(
            nn.Linear(widths[0] + point, head),
            nn.GELU(),
            nn.Linear(head, head),
            nn.GELU(),
            nn.Linear(head, outputs),
        )
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    @property
    def semantics(self) -> bool:
        """Whether the refiner sees classes: grid histogram and per-pixel embedding."""
        return self.embed is not None

    @property
    def outputs(self) -> int:
        """Head outputs: 1 (r) or 2 (r and the keep logit)."""
        return self.head[-1].out_features

    def forward(self, feats: Tensor, pts: ScenePoints, sel: Tensor) -> Tensor:
        """Residual r (n,) of the selected pixels from U-Net features (C, Z, Y, X)."""
        return self.heads(feats, pts, sel)[:, 0]

    def heads(self, feats: Tensor, pts: ScenePoints, sel: Tensor) -> Tensor:
        """Head outputs (n, outputs) of the selected pixels: r, then the keep logit."""
        p = pts.points[sel]
        extent = torch.tensor(pts.dims, dtype=p.dtype, device=p.device) * pts.cell
        coords = 2 * (p - pts.lo) / extent - 1
        sampled = (
            F.grid_sample(
                feats[None],
                coords.view(1, -1, 1, 1, 3),
                align_corners=False,
                padding_mode="border",
            )
            .view(feats.shape[0], -1)
            .T
        )
        local = (p - pts.lo) / pts.cell
        own = [
            sampled,
            pts.color[sel],
            pts.log_conf[sel, None],
            local - local.floor() - 0.5,
            pts.rays[sel],
        ]
        if self.embed is not None:
            own.append(self.embed(pts.classes[sel]))
        return self.head(torch.cat(own, 1))


def to_device(pts: ScenePoints, device: torch.device) -> ScenePoints:
    """Move every tensor of the scene to the device."""
    return dataclasses.replace(pts, **{
        f.name: getattr(pts, f.name).to(device)
        for f in dataclasses.fields(pts) if isinstance(getattr(pts, f.name), Tensor)
    })  # fmt: skip


def load_scene(
    cfg: VolumeConfig, scene: str, semantics: str
) -> tuple[ScenePoints, Geometry, np.ndarray, np.ndarray, np.ndarray | None]:
    """One scene's pixels, geometry, r*, its valid mask (N,) and the distance target."""
    folder = cfg.root / scene
    views = Views.load(folder / f"{cfg.views}.npz")
    geometry = Geometry.load(folder / f"{cfg.backbone}_{cfg.views}.npz")
    gt = GroundTruth.load(folder / f"gt_{cfg.views}.npz")
    if semantics == "gt":
        ids, classes = gt.instance_ids, gt.classes
    elif semantics == "sam3":
        masks = Segmentation.load(folder / f"{cfg.segmentation}_{cfg.views}.npz")
        ids, classes = masks.instance_ids, masks.classes
    else:
        ids, classes = np.full(geometry.depth.shape, -1), {}
    pts, rays, depth, scale = scene_points(
        geometry, views.images, ids, classes, cfg.resolution
    )
    r, valid = target_residual(rays, depth, views, gt)
    distance = None
    if cfg.keep is not None and cfg.keep.target == "distance":
        distance = distance_target(rays, depth, views, gt, scale)
    return pts, geometry, r, valid, distance


@dataclasses.dataclass(frozen=True)
class TrainScene:
    """A training scene on the CPU: compact pixels, r* (float16), valid, held pick."""

    pts: ScenePoints
    r: Tensor
    valid: Tensor
    views: int
    pixels: int  # per view
    held: Tensor  # pixels the held-out error is reported on
    has: Tensor | None = None  # (N,) pixels with a keep target; only with a keep head
    distance: Tensor | None = None  # (N,) distance to the GT surface; target "distance"


def load_scenes(
    cfg: VolumeConfig, scenes: list[str], semantics: str
) -> tuple[list[TrainScene], list[str]]:
    """Every scene, compact; skipped: no pinhole rays or missing inputs."""
    rng = np.random.default_rng(0)
    data, skipped = [], []
    for scene in scenes:
        try:
            pts, geometry, r, valid, distance = load_scene(cfg, scene, semantics)
        except AssertionError as err:  # fit_pinhole: Deja View's rays are no pinhole
            skipped.append(f"{scene}: pinhole fit failed ({err})")
            print(f"{scene}: skipped, pinhole fit failed ({err})", flush=True)  # noqa: T201
            continue
        except FileNotFoundError as err:  # a scene whose download or caching failed
            skipped.append(f"{scene}: missing {Path(str(err.filename)).name}")
            print(f"{scene}: skipped, missing input ({err})", flush=True)  # noqa: T201
            continue
        views, h, w = geometry.depth.shape
        has = None
        if distance is not None:  # every pixel has a distance
            has = np.ones(len(r), dtype=bool)
        elif cfg.keep is not None:  # the keep head also learns from the invalid pixels
            folder = cfg.root / scene
            has = GroundTruth.load(folder / f"gt_{cfg.views}.npz").depth.reshape(-1) > 0
        pool = valid if has is None else has
        keep = rng.choice(
            np.flatnonzero(pool), size=min(cfg.kept_points, int(pool.sum()))
        )
        truth = None if has is None else torch.from_numpy(has)
        target = None if distance is None else torch.from_numpy(distance)
        data.append(TrainScene(compact(pts), torch.from_numpy(r).half(),
                               torch.from_numpy(valid), views, h * w,
                               torch.from_numpy(np.sort(keep)), truth,
                               target))  # fmt: skip
    return data, skipped


def keep_labels(err: Tensor, quantile: float) -> Tensor:
    """1 for the round(quantile n) pixels of least error (n,), else 0; inf: an error."""
    assert err.ndim == 1 and 0 < quantile < 1, (err.shape, quantile)
    k = max(1, round(quantile * len(err)))
    rank = torch.empty_like(err, dtype=torch.long)
    rank[err.argsort(stable=True)] = torch.arange(len(err), device=err.device)
    return (rank < k).float()


def keep_loss(
    model: VolumeRefiner,
    feats: Tensor,
    pts: ScenePoints,
    pick: Tensor,
    target: tuple[Tensor, Tensor, KeepHead],
) -> Tensor:
    """L1 on r (valid pixels, if the residual trains) plus the keep cross-entropy."""
    r_star, valid, keep = target
    out = model.heads(feats, pts, pick)
    r = out[:, 0] if keep.residual else torch.zeros_like(r_star)
    err = torch.where(valid, (r.detach() - r_star).abs(), torch.inf)
    bce = F.binary_cross_entropy_with_logits(out[:, 1], keep_labels(err, keep.quantile))
    l1 = (r - r_star).abs()[valid].mean() if keep.residual else 0.0
    return l1 + keep.weight * bce


def distance_loss(
    model: VolumeRefiner,
    feats: Tensor,
    pts: ScenePoints,
    pick: Tensor,
    target: tuple[Tensor, KeepHead],
) -> Tensor:
    """L1 between the predicted and the true distance to the GT surface."""
    d_star, keep = target
    out = model.heads(feats, pts, pick)[:, 1]
    return keep.weight * (out - d_star).abs().mean()


def step_loss(
    model: VolumeRefiner,
    scene: TrainScene,
    cfg: VolumeConfig,
    rng: np.random.Generator,
    device: torch.device,
) -> Tensor:
    """L1 on r of random valid pixels, the grid built from a random subset of views."""
    k = int(rng.integers(cfg.view_subset[0], min(cfg.view_subset[1], scene.views) + 1))
    sel = view_pixels(
        np.sort(rng.choice(scene.views, size=k, replace=False)), scene.pixels
    )
    pts = expand(scene.pts, sel, device)
    pool = scene.valid if scene.has is None else scene.has
    candidates = torch.nonzero(pool[sel])[:, 0]
    pick = candidates[
        torch.from_numpy(rng.integers(len(candidates), size=cfg.batch_points))
    ]
    feats = model.features(voxelize(pts, semantics=model.semantics), amp=cfg.amp)
    r_star = scene.r[sel][pick].to(device).float()
    if cfg.keep is not None and cfg.keep.target == "distance":
        assert scene.distance is not None
        d_star = scene.distance[sel][pick].to(device)
        return distance_loss(model, feats, pts, pick.to(device), (d_star, cfg.keep))
    if cfg.keep is not None:
        valid = scene.valid[sel][pick].to(device)
        return keep_loss(model, feats, pts, pick.to(device), (r_star, valid, cfg.keep))
    pred = model(feats, pts, pick.to(device))
    return (pred - r_star).abs().mean()


def selection_key(cfg: VolumeConfig) -> str:
    """Return the held-out quantity the kept model minimizes."""
    if cfg.keep is None:
        return "abs_log_after"
    return "distance_l1" if cfg.keep.target == "distance" else "kept_error"


def train(  # noqa: PLR0913 - one flat entry point; a settings object would only rename these
    cfg: VolumeConfig,
    data: list[TrainScene],
    val: list[TrainScene],
    device: torch.device,
    *,
    semantics: bool,
    seed: int,
    writer: SummaryWriter,
) -> tuple[VolumeRefiner, list[dict]]:
    """Fit one refiner, one train scene per step; keep the least held-out error state.

    E41a's held-out error rose after the first few hundred steps (35 train scenes), so
    the returned model is the one at the evaluation step where the held-out train scenes
    fit best; no dev or val scene takes part in the choice. TensorBoard (`writer`) gets
    the loss of every step and, at every evaluation step, the held-out and the `val`
    scenes' errors (issue #57's ladder: curves only, `val` never trains or chooses).
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    held_out = cfg.held_out or len(data)
    fit, held = (data if cfg.held_out == 0 else data[: -cfg.held_out]), data[-held_out:]
    channels = grid_channels(semantics=semantics)
    model = VolumeRefiner(
        channels, cfg.widths, cfg.head, semantics=semantics, outputs=outputs(cfg)
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, cfg.lr, total_steps=cfg.steps)
    log: list[dict] = []
    t0 = time.perf_counter()
    best, best_state = float("inf"), None
    for step in range(cfg.steps + 1):
        if step % (cfg.steps // 20) == 0:
            err = held_out_error(model, held, cfg, device)
            log.append({"step": step, "seconds": time.perf_counter() - t0, **err})
            for key, value in err.items():
                writer.add_scalar(f"held_out/{key}", value, step)
            if val:
                for key, value in held_out_error(model, val, cfg, device).items():
                    writer.add_scalar(f"val/{key}", value, step)
            print(f"step {step}: {log[-1]}", flush=True)  # noqa: T201
            metric = err[selection_key(cfg)]
            if metric < best:
                best = metric
                best_state = {
                    k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                }
                log[-1]["best"] = True
        if step == cfg.steps:
            break
        model.train()
        value = step_loss(model, fit[rng.integers(len(fit))], cfg, rng, device)
        opt.zero_grad()
        value.backward()
        opt.step()
        sched.step()
        writer.add_scalar("train/loss", value.item(), step)
    assert best_state is not None
    model.load_state_dict(best_state)
    return model, log


def held_out_error(
    model: VolumeRefiner,
    held: list[TrainScene],
    cfg: VolumeConfig,
    device: torch.device,
) -> dict[str, float]:
    """Mean |r* - r| before (r = 0) and after on the held-out scenes, all views.

    With a keep head also the mean error of the kept pixels, the top `quantile` by
    score (`kept_error`), against all pixels (`all_error`); an invalid pixel's error is
    `MAX_RESIDUAL`. With the distance target the error is the true distance, and
    `distance_l1` and the rank correlation `spearman` of the prediction with it are
    reported too.
    """
    model.eval()
    out = dict.fromkeys(["abs_log_before", "abs_log_after"], 0.0)
    with torch.no_grad():
        for scene in held:
            pts = expand(scene.pts, torch.arange(len(scene.r)), device)
            feats = model.features(
                voxelize(pts, semantics=model.semantics), amp=cfg.amp
            )
            heads = model.heads(feats, pts, scene.held.to(device)).cpu()
            valid = scene.valid[scene.held]
            r = scene.r[scene.held].float()
            pred = heads[:, 0]
            out["abs_log_before"] += float(r[valid].abs().mean()) / len(held)
            out["abs_log_after"] += float((pred - r)[valid].abs().mean()) / len(held)
            if cfg.keep is not None:
                if cfg.keep.target == "distance":
                    assert scene.distance is not None
                    err = scene.distance[scene.held]
                    scores = {
                        "distance_l1": (heads[:, 1] - err).abs().mean(),
                        "spearman": spearman(heads[:, 1], err),
                    }
                else:
                    used = pred if cfg.keep.residual else torch.zeros_like(pred)
                    err = torch.where(valid, (used - r).abs(), MAX_RESIDUAL)
                    scores = {}
                k = max(1, round(cfg.keep.quantile * len(err)))
                score = keep_score(cfg.keep, heads)
                order = score.argsort(descending=True, stable=True)
                scores |= {"kept_error": err[order[:k]].mean(), "all_error": err.mean()}
                for key, value in scores.items():
                    out[key] = out.get(key, 0.0) + float(value) / len(held)
    return out


def keep_score(keep: KeepHead, heads: Tensor) -> Tensor:
    """Per pixel, higher keeps: the keep logit, or the negated predicted distance."""
    return -heads[:, 1] if keep.target == "distance" else heads[:, 1]


def spearman(a: Tensor, b: Tensor) -> Tensor:
    """Rank correlation of two vectors; 0 when `a` is constant (an untrained head)."""
    assert a.shape == b.shape and a.ndim == 1, (a.shape, b.shape)
    if a.max() == a.min():
        return a.new_zeros(())
    ranks = torch.stack([x.argsort(stable=True).argsort(stable=True) for x in (a, b)])
    return torch.corrcoef(ranks.float())[0, 1]


def refine(
    model: VolumeRefiner,
    pts: ScenePoints,
    geometry: Geometry,
    device: torch.device,
    *,
    amp: bool = False,
) -> Geometry:
    """Return the backbone geometry with depth D exp(r); all else kept."""
    r = predict_heads(model, pts, device, amp=amp)[:, 0]
    return with_residual(geometry, r)


def predict_heads(
    model: VolumeRefiner, pts: ScenePoints, device: torch.device, *, amp: bool = False
) -> Tensor:
    """Head outputs (N, outputs) of every pixel of the scene, on the CPU."""
    model.eval()
    with torch.no_grad():
        on = expand(compact(pts), torch.arange(len(pts.points)), device)
        feats = model.features(voxelize(on, semantics=model.semantics), amp=amp)
        n = len(pts.points)
        return torch.cat(
            [
                model.heads(
                    feats, on, torch.arange(i, min(i + CHUNK, n), device=device)
                ).cpu()
                for i in range(0, n, CHUNK)
            ]
        )


def with_residual(geometry: Geometry, r: Tensor) -> Geometry:
    """Return the geometry with depth D exp(r), r flat (N,); all else kept."""
    depth = (geometry.depth * np.exp(r.numpy().reshape(geometry.depth.shape))).astype(
        np.float32
    )
    assert np.all(np.isfinite(depth))
    assert np.all(depth > 0)
    return dataclasses.replace(geometry, depth=depth)


def write_prediction(
    model: VolumeRefiner,
    cfg: VolumeConfig,
    scene: tuple[str, str],
    name: str,
    device: torch.device,
) -> None:
    """Write `<name>_<views>.npz` of a (scene, semantics), and the keep logit if any."""
    folder, semantics = cfg.root / scene[0], scene[1]
    pts, geometry, _, _, _ = load_scene(cfg, scene[0], semantics)
    if cfg.keep is None:
        refined = refine(model, pts, geometry, device, amp=cfg.amp)
    else:
        heads = predict_heads(model, pts, device, amp=cfg.amp)
        r = heads[:, 0] if cfg.keep.residual else torch.zeros(len(heads))
        refined = with_residual(geometry, r)
        score = keep_score(cfg.keep, heads).numpy().reshape(geometry.depth.shape)
        np.savez_compressed(
            folder / f"{name}_keep_{cfg.views}.npz", keep=score.astype(np.float32)
        )
    refined.save(folder / f"{name}_{cfg.views}.npz")


def outputs(cfg: VolumeConfig) -> int:
    """Head outputs: r, and the keep logit with a keep head."""
    return 1 if cfg.keep is None else 2


def grid_channels(*, semantics: bool, features: int = 0) -> int:
    """Channels `voxelize` writes: count, conf, colour, spread, classes, features."""
    return 8 + (CLASSES if semantics else 0) + features


def main() -> None:
    """Train every model of the config per seed, save it, write its depth for `predict`.

    The train scenes are loaded once per class source and shared by its seeds. `--out`
    resumes a run: a model already saved there is loaded instead of trained, so a job
    that died (out of memory, a reboot) is restarted without losing finished models. On
    CUDA the process first takes `--reserve-gb` of GPU memory and frees it into
    PyTorch's cache, which keeps it, so a job started later on the shared GPU cannot
    take the memory this one needs. `--only` restricts the run to the named models
    (with -s<seed>), one SLURM task each, each with its own `--out`.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/e41b_volume.yaml"))
    parser.add_argument("--out", type=Path, help="run folder to create or resume")
    parser.add_argument("--reserve-gb", type=float, default=6.0)
    parser.add_argument("--only", nargs="*", help="model names (with -s<seed>)")
    args = parser.parse_args()
    cfg = VolumeConfig.load(args.config)
    device = best_device()
    if device.type == "cuda" and args.reserve_gb > 0:
        torch.empty(int(args.reserve_gb * 2**30), dtype=torch.uint8, device=device)
    out = args.out or Path("runs") / f"volume-{time.strftime('%Y%m%d-%H%M%S')}"
    out.mkdir(parents=True, exist_ok=True)
    run_yaml = out / "run.yaml"
    if run_yaml.exists():
        record = yaml.safe_load(run_yaml.read_text())
        record.setdefault("resumed_at", []).append(git_hash())
    else:
        record = {"config": yaml.safe_load(args.config.read_text()), "git": git_hash()}
    for base, semantics in cfg.models.items():
        sem = semantics != "none"
        names = {s: f"{base}-s{s}" if len(cfg.seeds) > 1 else base for s in cfg.seeds}
        names = {s: n for s, n in names.items() if not args.only or n in args.only}
        todo = [
            s
            for s, n in names.items()
            if not ((out / f"{n}.pt").exists() and n in record)
        ]
        data, val = [], []
        if todo:
            data, skipped = load_scenes(cfg, cfg.train, semantics)
            val, skipped_val = load_scenes(cfg, cfg.val, semantics)
            record.setdefault("skipped", {})[base] = skipped + skipped_val
        for seed, name in names.items():
            saved = out / f"{name}.pt"
            channels = grid_channels(semantics=sem)
            if seed in todo:
                t0 = time.perf_counter()
                with SummaryWriter(out / "tensorboard" / name) as writer:
                    model, log = train(
                        cfg, data, val, device, semantics=sem, seed=seed, writer=writer
                    )
                record[name] = {"seconds": time.perf_counter() - t0, "held_out": log}
                torch.save(model.state_dict(), saved)
                run_yaml.write_text(yaml.safe_dump(record, sort_keys=False))
            else:
                model = VolumeRefiner(
                    channels, cfg.widths, cfg.head, semantics=sem, outputs=outputs(cfg)
                ).to(device)
                model.load_state_dict(torch.load(saved, map_location=device))
                print(f"{name}: loaded from {saved}")  # noqa: T201
            for scene in cfg.predict:
                write_prediction(model, cfg, (scene, semantics), name, device)
            print(f"{name}: wrote {len(cfg.predict)} scenes", flush=True)  # noqa: T201
        del data, val
    run_yaml.write_text(yaml.safe_dump(record, sort_keys=False))
    print(f"wrote {out}")  # noqa: T201


if __name__ == "__main__":
    main()
