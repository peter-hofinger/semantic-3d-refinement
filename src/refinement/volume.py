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

from src.analysis.scene_eval import git_hash
from src.device import best_device
from src.evaluation.labeled import lift, umeyama
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
CHUNK = 500_000  # pixels per forward pass at prediction


@dataclasses.dataclass(frozen=True)
class VolumeConfig:
    """Data, models and training settings (`configs/e41b_volume.yaml`)."""

    root: Path
    views: str
    backbone: str
    segmentation: str
    train: list[str]  # scene folders of `train_splits`; the last `held_out` only report
    held_out: int
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

    @classmethod
    def load(cls, path: Path) -> VolumeConfig:
        """Read the config; train and prediction scenes come from Hypersim splits."""
        raw = yaml.safe_load(path.read_text())
        split = load_config(Path(raw.pop("split")))
        train = split.scenes(raw.pop("train_splits"))
        raw["train"] = [f"{s}_{c}" for s, c in train.items()]
        predict = split.scenes(raw.pop("predict_splits"))
        raw["predict"] = [f"{s}_{c}" for s, c in predict.items()]
        raw["root"] = Path(raw["root"])
        assert all(s in SEMANTICS for s in raw["models"].values()), raw["models"]
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
    points: np.ndarray, resolution: int
) -> tuple[np.ndarray, float, tuple[int, int, int]]:
    """Corner, cell size and cells per axis (x8) of the padded 1-99 % box."""
    lo, hi = np.percentile(points, [1, 99], axis=0)
    pad = 0.05 * (hi - lo)
    lo, hi = lo - pad, hi + pad
    cell = float((hi - lo).max() / resolution)
    dims = tuple(
        int(8 * np.ceil((h - low) / cell / 8)) for low, h in zip(lo, hi, strict=True)
    )
    return lo, cell, dims


def scene_points(
    geometry: Geometry,
    images: np.ndarray,
    ids: np.ndarray,
    classes: dict[int, str],
    resolution: int,
) -> tuple[ScenePoints, np.ndarray, np.ndarray, float]:
    """Flat pixels in the grid's frame; also the raw rays (N, 6), depth and scale."""
    scene = scene_from_geometry(geometry)
    rays = scene.rays.reshape(-1, 6).double().numpy()
    depth = geometry.depth.reshape(-1).astype(np.float64)
    scale = float(np.median(depth))
    x = (rays[:, :3] + depth[:, None] * rays[:, 3:]) / scale
    lo, cell, dims = grid_box(x, resolution)
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


def target_residual(
    rays: np.ndarray, depth: np.ndarray, views: Views, gt: GroundTruth
) -> tuple[np.ndarray, np.ndarray]:
    """Return r* = log(t* / D) per pixel (N,) and where it is valid (Umeyama)."""
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
    s, rotation, t = umeyama(pred[has], gt_points[has])
    in_dv = (gt_points - t) @ rotation / s
    d = rays[:, 3:]
    t_star = ((in_dv - rays[:, :3]) * d).sum(1) / (d * d).sum(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.log(t_star / depth)
    valid = has & (t_star > 0) & (np.abs(r) <= MAX_RESIDUAL)
    return np.where(valid, r, 0).astype(np.float32), valid


def voxelize(pts: ScenePoints, *, semantics: bool) -> Tensor:
    """Grid (C, Z, Y, X): log count, conf, colour, spread, class histogram."""
    nx, ny, nz = pts.dims
    idx3 = ((pts.points - pts.lo) / pts.cell).floor().long()
    idx3 = torch.stack([idx3[:, k].clamp(0, n - 1) for k, n in enumerate(pts.dims)], 1)
    flat = (idx3[:, 2] * ny + idx3[:, 1]) * nx + idx3[:, 0]
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
    assert grid.isfinite().all()
    return grid


def block(c_in: int, c_out: int) -> nn.Sequential:
    """Two 3x3x3 convolutions with GroupNorm and GELU."""
    return nn.Sequential(
        nn.Conv3d(c_in, c_out, 3, padding=1), nn.GroupNorm(8, c_out), nn.GELU(),
        nn.Conv3d(c_out, c_out, 3, padding=1), nn.GroupNorm(8, c_out), nn.GELU(),
    )  # fmt: skip


class VolumeRefiner(nn.Module):
    """3D U-Net on the grid, then a per-pixel head on the sampled features."""

    def __init__(
        self, grid_channels: int, widths: list[int], head: int, *, semantics: bool
    ) -> None:
        """Widths per U-Net level; the head's last layer starts at zero (identity)."""
        super().__init__()
        self.down = nn.ModuleList()
        c = grid_channels
        for w in widths:
            self.down.append(block(c, w))
            c = w
        self.up = nn.ModuleList(
            block(widths[i] + widths[i + 1], widths[i]) for i in range(len(widths) - 1)
        )
        self.embed = nn.Embedding(CLASSES, 8) if semantics else None
        point = 3 + 1 + 3 + 3 + (8 if semantics else 0)
        self.head = nn.Sequential(
            nn.Linear(widths[0] + point, head),
            nn.GELU(),
            nn.Linear(head, head),
            nn.GELU(),
            nn.Linear(head, 1),
        )
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    @property
    def semantics(self) -> bool:
        """Whether the refiner sees classes: grid histogram and per-pixel embedding."""
        return self.embed is not None

    def features(self, grid: Tensor, *, amp: bool = False) -> Tensor:
        """U-Net features (C, Z, Y, X) of a grid (C_in, Z, Y, X); bfloat16 with amp."""
        if amp and grid.is_cuda:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                return self.features(grid).float()
        skips, x = [], grid[None]
        for i, enc in enumerate(self.down):
            x = enc(x if i == 0 else F.avg_pool3d(x, 2))
            skips.append(x)
        for i in reversed(range(len(self.up))):
            x = F.interpolate(
                x, size=skips[i].shape[2:], mode="trilinear", align_corners=False
            )
            x = self.up[i](torch.cat([x, skips[i]], 1))
        return x[0]

    def forward(self, feats: Tensor, pts: ScenePoints, sel: Tensor) -> Tensor:
        """Residual r (n,) of the selected pixels from U-Net features (C, Z, Y, X)."""
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
        return self.head(torch.cat(own, 1))[:, 0]


def to_device(pts: ScenePoints, device: torch.device) -> ScenePoints:
    """Move every tensor of the scene to the device."""
    return dataclasses.replace(pts, **{
        f.name: getattr(pts, f.name).to(device)
        for f in dataclasses.fields(pts) if isinstance(getattr(pts, f.name), Tensor)
    })  # fmt: skip


def load_scene(
    cfg: VolumeConfig, scene: str, semantics: str
) -> tuple[ScenePoints, Geometry, np.ndarray, np.ndarray]:
    """One scene's pixels and geometry, with r* and its valid mask (N,)."""
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
    pts, rays, depth, _ = scene_points(
        geometry, views.images, ids, classes, cfg.resolution
    )
    r, valid = target_residual(rays, depth, views, gt)
    return pts, geometry, r, valid


@dataclasses.dataclass(frozen=True)
class TrainScene:
    """A training scene on the CPU: compact pixels, r* (float16), valid, held pick."""

    pts: ScenePoints
    r: Tensor
    valid: Tensor
    views: int
    pixels: int  # per view
    held: Tensor  # pixels the held-out error is reported on


def load_train(cfg: VolumeConfig, semantics: str) -> tuple[list[TrainScene], list[str]]:
    """Every train scene, compact; skipped: no pinhole rays or missing inputs."""
    rng = np.random.default_rng(0)
    data, skipped = [], []
    for scene in cfg.train:
        try:
            pts, geometry, r, valid = load_scene(cfg, scene, semantics)
        except AssertionError as err:  # fit_pinhole: Deja View's rays are no pinhole
            skipped.append(f"{scene}: pinhole fit failed ({err})")
            print(f"{scene}: skipped, pinhole fit failed ({err})", flush=True)  # noqa: T201
            continue
        except FileNotFoundError as err:  # a scene whose download or caching failed
            skipped.append(f"{scene}: missing {Path(str(err.filename)).name}")
            print(f"{scene}: skipped, missing input ({err})", flush=True)  # noqa: T201
            continue
        views, h, w = geometry.depth.shape
        keep = rng.choice(
            np.flatnonzero(valid), size=min(cfg.kept_points, int(valid.sum()))
        )
        data.append(TrainScene(compact(pts), torch.from_numpy(r).half(),
                               torch.from_numpy(valid), views, h * w,
                               torch.from_numpy(np.sort(keep))))  # fmt: skip
    return data, skipped


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
    candidates = torch.nonzero(scene.valid[sel])[:, 0]
    pick = candidates[
        torch.from_numpy(rng.integers(len(candidates), size=cfg.batch_points))
    ]
    feats = model.features(voxelize(pts, semantics=model.semantics), amp=cfg.amp)
    pred = model(feats, pts, pick.to(device))
    return (pred - scene.r[sel][pick].to(device).float()).abs().mean()


def train(
    cfg: VolumeConfig,
    data: list[TrainScene],
    device: torch.device,
    *,
    semantics: bool,
    seed: int,
) -> tuple[VolumeRefiner, list[dict]]:
    """Fit one refiner, one train scene per step; log the held-out error."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    fit, held = data[: -cfg.held_out], data[-cfg.held_out :]
    channels = grid_channels(semantics=semantics)
    model = VolumeRefiner(channels, cfg.widths, cfg.head, semantics=semantics).to(
        device
    )
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, cfg.lr, total_steps=cfg.steps)
    log: list[dict] = []
    t0 = time.perf_counter()
    for step in range(cfg.steps + 1):
        if step % (cfg.steps // 20) == 0:
            err = held_out_error(model, held, cfg, device)
            log.append({"step": step, "seconds": time.perf_counter() - t0, **err})
            print(f"step {step}: {log[-1]}", flush=True)  # noqa: T201
        if step == cfg.steps:
            break
        model.train()
        value = step_loss(model, fit[rng.integers(len(fit))], cfg, rng, device)
        opt.zero_grad()
        value.backward()
        opt.step()
        sched.step()
    return model, log


def held_out_error(
    model: VolumeRefiner,
    held: list[TrainScene],
    cfg: VolumeConfig,
    device: torch.device,
) -> dict[str, float]:
    """Mean |r* - r| before (r = 0) and after on the held-out scenes, all views."""
    model.eval()
    before = after = 0.0
    with torch.no_grad():
        for scene in held:
            pts = expand(scene.pts, torch.arange(len(scene.r)), device)
            feats = model.features(
                voxelize(pts, semantics=model.semantics), amp=cfg.amp
            )
            pred = model(feats, pts, scene.held.to(device)).cpu()
            r = scene.r[scene.held].float()
            before += float(r.abs().mean()) / len(held)
            after += float((pred - r).abs().mean()) / len(held)
    return {"abs_log_before": before, "abs_log_after": after}


def refine(
    model: VolumeRefiner,
    pts: ScenePoints,
    geometry: Geometry,
    device: torch.device,
    *,
    amp: bool = False,
) -> Geometry:
    """Return the backbone geometry with depth D exp(r); all else kept."""
    model.eval()
    with torch.no_grad():
        on = expand(compact(pts), torch.arange(len(pts.points)), device)
        feats = model.features(voxelize(on, semantics=model.semantics), amp=amp)
        n = len(pts.points)
        r = torch.cat(
            [
                model(
                    feats, on, torch.arange(i, min(i + CHUNK, n), device=device)
                ).cpu()
                for i in range(0, n, CHUNK)
            ]
        )
    depth = (geometry.depth * np.exp(r.numpy().reshape(geometry.depth.shape))).astype(
        np.float32
    )
    assert np.all(np.isfinite(depth))
    assert np.all(depth > 0)
    return dataclasses.replace(geometry, depth=depth)


def grid_channels(*, semantics: bool) -> int:
    """Channels `voxelize` writes: count, conf, colour, spread (+ class histogram)."""
    return 8 + (CLASSES if semantics else 0)


def main() -> None:
    """Train every model of the config per seed, save it, write its depth for `predict`.

    The train scenes are loaded once per class source and shared by its seeds. `--out`
    resumes a run: a model already saved there is loaded instead of trained, so a job
    that died (out of memory, a reboot) is restarted without losing finished models. On
    CUDA the process first takes `--reserve-gb` of GPU memory and frees it into
    PyTorch's cache, which keeps it, so a job started later on the shared GPU cannot
    take the memory this one needs.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/e41b_volume.yaml"))
    parser.add_argument("--out", type=Path, help="run folder to create or resume")
    parser.add_argument("--reserve-gb", type=float, default=6.0)
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
        todo = [
            s
            for s, n in names.items()
            if not ((out / f"{n}.pt").exists() and n in record)
        ]
        data = []
        if todo:
            data, skipped = load_train(cfg, semantics)
            record.setdefault("skipped", {})[base] = skipped
        for seed, name in names.items():
            saved = out / f"{name}.pt"
            channels = grid_channels(semantics=sem)
            if seed in todo:
                t0 = time.perf_counter()
                model, log = train(cfg, data, device, semantics=sem, seed=seed)
                record[name] = {"seconds": time.perf_counter() - t0, "held_out": log}
                torch.save(model.state_dict(), saved)
                run_yaml.write_text(yaml.safe_dump(record, sort_keys=False))
            else:
                model = VolumeRefiner(channels, cfg.widths, cfg.head, semantics=sem)
                model = model.to(device)
                model.load_state_dict(torch.load(saved, map_location=device))
                print(f"{name}: loaded from {saved}")  # noqa: T201
            for scene in cfg.predict:
                pts, geometry, _, _ = load_scene(cfg, scene, semantics)
                refined = refine(model, pts, geometry, device, amp=cfg.amp)
                refined.save(cfg.root / scene / f"{name}_{cfg.views}.npz")
            print(f"{name}: wrote {len(cfg.predict)} scenes", flush=True)  # noqa: T201
        del data
    run_yaml.write_text(yaml.safe_dump(record, sort_keys=False))
    print(f"wrote {out}")  # noqa: T201


if __name__ == "__main__":
    main()
