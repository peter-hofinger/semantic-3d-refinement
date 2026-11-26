"""E69f: close E69e's rung 0. Caveats, candidate rules, the grid box (issue #69).

`diagnose`: C1, every input pixel's distance between Déjà View's aligned point and its GT
point, and of those off by more than `FAR`, how many are bright and how deep each says the
surface is; C2, in-box unseen coverage of candidates dilated around GT20's own points
against those around Déjà View's (same box, reach 1-4); C3, where the in-box unseen points
outside reach 4 lie: distance to Déjà View and whether any input view's frustum holds them.

`coverage`: candidate rules on one Pareto table. A rule is a mask (Z, Y, X) of candidate
voxels; its in-box unseen coverage, the Chamfer of its oracle (the GT99 children of split
4, the mean of their points) in-box and with Déjà View's out-of-box points passed through,
its size, its candidates per 32-cell crop and its children's positive / negative ratio.
Carving (space carving; SG-NN's known-free space, Dai et al. 2020) removes a voxel whose
centre lies, in some view, in front of Déjà View's depth at its pixel by more than one cell:

    free(x) = OR_v [ inside_v(x) and 0 < z_v(x) < D_v(pixel_v(x)) - cell ]

Two-level (coarse to fine; TRELLIS.2 §3.3 generates the occupancy first): coarse voxels of 2^3
fine ones dilated by a coarse reach; the fine candidates are the coarse voxels that hold
GT (what a perfect coarse stage passes on), so the cost is both levels' voxels.

`box`: why Déjà View's out-of-box points matter (their distance to GT20, the face they leave
through) and boxes from per-axis percentiles p, 100 - p with a relative pad, at 192 cells
and at the 1-99 % box's cell. Seen clipping here is at a 1 cm output voxel for every box,
so that the box, not the child size, is what changes.

`lock`: the six-scene rung 0 again with the locked box (`--config configs/e69f.yaml`):
both baselines, the box's clipping and every rule's coverage and oracle.

`uv run python -m experiments.scripts.e69f_rung0 diagnose|coverage|box|lock` writes
`experiments/results/e69f_<command>.txt`; `coverage` also keeps its numbers in
`e69f_coverage.json`, from which `table` rewrites the Pareto table alone.
"""

import argparse
import dataclasses
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F  # noqa: N812

from experiments.scripts.e69c_oracle import Frame, children_cloud
from experiments.scripts.e69d_all_frames_gt import seen_by, voxel_centroids
from experiments.scripts.e69e_rung0 import (
    CM,
    SEEN_TOLERANCE,
    covered,
    score,
    to_grid_frame,
    to_world,
)
from src.evaluation.labeled import nearest_distances
from src.geometry import Geometry
from src.loaders.ground_truth import GroundTruth
from src.loaders.views import Views
from src.refinement.generator import (
    GeneratorConfig,
    box_of,
    child_targets,
    load_rung,
    load_scene,
)
from src.refinement.generator import (
    dilate as grow,
)
from src.refinement.multiview import Scene, points_from_rays, scene_from_geometry
from src.refinement.volume import grid_box, gt_alignment, scene_points, voxelize

SCENES = (
    "ai_006_007_cam_00",
    "ai_010_007_cam_00",
    "ai_044_001_cam_00",
    "ai_004_003_cam_00",
    "ai_009_009_cam_00",
    "ai_031_004_cam_00",
)
SPLIT = 4
FAR = 0.5  # metres: C1's "far off" pixel
BRIGHT = 0.9  # C1's luminance of a blown-out pixel
CROP = 32  # cells of a training crop
CROPS = 200  # crops sampled for the candidates per crop
CELL_REACHES = (4, 6, 8, 10, 12)
PHYSICAL_CM = (25, 50, 75, 100)
CARVED_REACHES = (6, 8, 12)
COARSE_REACHES = (3, 4, 6)
PERCENTILES = (1.0, 0.5, 0.1, 0.0)
PADS = (0.05, 0.10)
MAX_CELLS = 384  # longest side of a grid whose cell is kept


@dataclasses.dataclass
class Loaded:
    """One scene in the world frame and in the grid's, with what every part needs."""

    name: str
    views: Views
    gt20: GroundTruth
    gt_world: np.ndarray  # (P, 3) every input pixel's GT point (P = all pixels)
    has: np.ndarray  # (P,) the pixel has GT
    dv: np.ndarray  # (P, 3) every pixel's Déjà View point, world
    dv_norm: np.ndarray  # (P, 3) the same in the grid's normalized frame
    scene: Scene  # Déjà View's pinhole scene, its own frame
    similarity: tuple[float, np.ndarray, np.ndarray]
    scale: float
    lo: np.ndarray
    cell: float
    dims: tuple[int, int, int]
    grid: torch.Tensor
    truth: np.ndarray  # GT99v, world
    seen: np.ndarray
    truth_cells: np.ndarray
    inside: np.ndarray  # GT99v points inside the box
    dv_in: np.ndarray  # pixels inside the box
    frame: Frame

    @property
    def child_m(self) -> float:
        """Side of a child of split 4 in metres."""
        return self.cell * self.scale * self.similarity[0] / SPLIT


def load(cfg: GeneratorConfig, name: str) -> Loaded:
    """Load one scene as E69e's checks do."""
    folder = cfg.root / name
    views = Views.load(folder / f"{cfg.views}.npz")
    geometry = Geometry.load(folder / f"{cfg.backbone}_{cfg.views}.npz")
    gt20 = GroundTruth.load(folder / f"gt_{cfg.views}.npz")
    pts, _, _, scale = load_scene(cfg, name, "none")
    _, rays, depth, _ = scene_points(
        geometry,
        views.images,
        np.full(geometry.depth.shape, -1),
        {},
        cfg.resolution,
        box_of(cfg),
    )
    gt_world, has, similarity = gt_alignment(rays, depth, views, gt20)
    truth = np.load(folder / f"gt99v1cm_{cfg.views}.npz")["points"]
    scene = scene_from_geometry(geometry)
    pixels = points_from_rays(scene.depth.double(), scene.rays.double()).numpy()
    pixels = pixels.reshape(-1, 3)
    lo = pts.lo.double().numpy()
    truth_cells = (to_grid_frame(truth, similarity, scale) - lo) / pts.cell
    box = np.ones(pts.dims[::-1], dtype=bool)
    dv_norm = pixels / scale
    return Loaded(
        name,
        views,
        gt20,
        gt_world,
        has,
        to_world(pixels, similarity),
        dv_norm,
        scene,
        similarity,
        scale,
        lo,
        pts.cell,
        pts.dims,
        voxelize(pts, semantics=False),
        truth,
        seen_by(truth, views, gt20, SEEN_TOLERANCE),
        truth_cells,
        covered(truth_cells, box),
        covered((dv_norm - lo) / pts.cell, box),
        Frame(pts.lo, pts.cell, scale),
    )


def dilate(mask: np.ndarray, reach: int) -> np.ndarray:
    """Voxels (Z, Y, X) within `reach` cells (Chebyshev) of a True one (numpy)."""
    return grow(torch.from_numpy(mask), reach).numpy()


def occupancy(cells: np.ndarray, dims: tuple[int, int, int]) -> np.ndarray:
    """Mask (Z, Y, X) of the voxels holding a point (N, 3) given in cells x y z."""
    index = np.floor(cells).astype(np.int64)
    keep = ((index >= 0) & (index < np.array(dims))).all(1)
    mask = np.zeros(dims[::-1], dtype=bool)
    x, y, z = index[keep].T
    mask[z, y, x] = True
    return mask


def free_space(
    s: Loaded,
) -> np.ndarray:
    """Mask (Z, Y, X) of the scene's voxels some view sees in front of its depth."""
    return carve(
        s.lo * s.scale,
        s.cell * s.scale,
        s.dims,
        s.scene.depth.double().numpy(),
        s.scene.k.double().numpy(),
        s.scene.rotation.double().numpy(),
        s.scene.center.double().numpy(),
    )


def carve(
    lo: np.ndarray,
    cell: float,
    dims: tuple[int, int, int],
    depth: np.ndarray,
    k: np.ndarray,
    rotation: np.ndarray,
    center: np.ndarray,
) -> np.ndarray:
    """Mask (Z, Y, X): voxel centre in front of a view's depth by more than one cell.

    Grid corner `lo` and `cell` in the cameras' frame; depth (V, H, W) along the
    camera's z; k (V, 3, 3) with pixel centres at integers; rotation (V, 3, 3) and
    center (V, 3) camera to world.
    """
    assert depth.ndim == 3, depth.shape
    z, y, x = np.indices(dims[::-1]).reshape(3, -1)
    centre = lo + (np.stack([x, y, z], 1) + 0.5) * cell
    free = np.zeros(len(centre), dtype=bool)
    _, h, w = depth.shape
    for v in range(len(depth)):
        cam = (centre - center[v]) @ rotation[v]
        front = cam[:, 2] > 1e-6
        u = np.full(len(cam), -1)
        r = np.full(len(cam), -1)
        u[front] = np.rint(k[v, 0, 0] * cam[front, 0] / cam[front, 2] + k[v, 0, 2])
        r[front] = np.rint(k[v, 1, 1] * cam[front, 1] / cam[front, 2] + k[v, 1, 2])
        ok = front & (u >= 0) & (u < w) & (r >= 0) & (r < h)
        free[ok] |= cam[ok, 2] < depth[v, r[ok], u[ok]] - cell
    return free.reshape(dims[::-1])


def check_projection(s: Loaded) -> float:
    """Largest |z - depth| (relative) of view 0's own pixels projected back: 0 = consistent."""
    pixels = points_from_rays(s.scene.depth[:1].double(), s.scene.rays[:1].double())
    cam = (pixels.reshape(-1, 3).numpy() - s.scene.center[0].double().numpy()) @ (
        s.scene.rotation[0].double().numpy()
    )
    depth = s.scene.depth[0].double().numpy().reshape(-1)
    return float(np.abs(cam[:, 2] / depth - 1).max())


def in_any_frustum(points: np.ndarray, views: Views) -> np.ndarray:
    """Whether each world point (N, 3) projects inside at least one input view."""
    out = np.zeros(len(points), dtype=bool)
    _, h, w, _ = views.images.shape
    for k, c2w in zip(views.intrinsics, views.cam_to_world, strict=True):
        cam = (points - c2w[:3, 3]) @ c2w[:3, :3]
        z = cam[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            u = k[0, 0] * cam[:, 0] / z + k[0, 2]
            v = k[1, 1] * cam[:, 1] / z + k[1, 2]
        out |= (z > 0) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
    return out


def diagnose(s: Loaded) -> list[str]:
    """C1-C3 for one scene."""
    lines = [f"\n== {s.name} (cell {CM * s.cell * s.scale * s.similarity[0]:.2f} cm)"]
    off = np.linalg.norm(s.dv - s.gt_world, axis=1)[s.has]
    images = s.views.images.reshape(-1, 3).astype(np.float64)
    luminance = (images / (255.0 if images.max() > 1.5 else 1.0)).mean(1)[s.has]
    gt_depth = s.gt20.depth.reshape(-1)[s.has]
    dv_depth = s.scene.depth.numpy().reshape(-1)[s.has] * s.similarity[0]
    far = off > FAR
    lines.append(
        f"C1 pixels with GT {s.has.sum():,}; off by > {CM * FAR:.0f} cm: "
        f"{100 * far.mean():.2f} %; bright among them {100 * (luminance[far] > BRIGHT).mean():.1f} % "
        f"(among all {100 * (luminance > BRIGHT).mean():.1f} %); "
        f"median depth GT / Déjà View: far {np.median(gt_depth[far]) if far.any() else np.nan:.2f} / "
        f"{np.median(dv_depth[far]) if far.any() else np.nan:.2f} m, "
        f"rest {np.median(gt_depth[~far]):.2f} / {np.median(dv_depth[~far]):.2f} m"
    )
    unseen_in = ~s.seen & s.inside
    gt20_cells = (
        to_grid_frame(s.gt_world[s.has], s.similarity, s.scale) - s.lo
    ) / s.cell
    gt_occ, dv_occ = occupancy(gt20_cells, s.dims), (s.grid[0] > 0).numpy()
    row = []
    for r in (1, 2, 3, 4):
        cover_dv = covered(s.truth_cells, dilate(dv_occ, r))[unseen_in].mean()
        cover_gt = covered(s.truth_cells, dilate(gt_occ, r))[unseen_in].mean()
        row.append(f"r{r} {100 * cover_dv:5.1f} / {100 * cover_gt:5.1f}")
    lines.append(
        "C2 in-box unseen coverage, Déjà View / GT20 candidates: " + ", ".join(row)
    )
    missed = unseen_in & ~covered(s.truth_cells, dilate(dv_occ, 4))
    if missed.any():
        d = nearest_distances(s.dv, s.truth[missed])
        cell_m = s.cell * s.scale * s.similarity[0]
        q = np.percentile(d, [50, 80, 95])
        frustum = in_any_frustum(s.truth[missed], s.views)
        d20 = nearest_distances(s.gt_world[s.has], s.truth[missed])
        lines.append(
            f"C3 in-box unseen outside reach 4: {missed.sum():,} points "
            f"({100 * missed.sum() / unseen_in.sum():.1f} %); distance to Déjà View "
            f"median / p80 / p95 {CM * q[0]:.0f} / {CM * q[1]:.0f} / {CM * q[2]:.0f} cm = "
            f"{q[0] / cell_m:.1f} / {q[1] / cell_m:.1f} / {q[2] / cell_m:.1f} cells; "
            f"to GT20 median {CM * np.median(d20):.0f} cm; "
            f"in an input frustum (occluded) {100 * frustum.mean():.1f} %"
        )
    return lines


def far_figure(s: Loaded, path: Path) -> None:
    """Draw the 4 views with most far-off pixels: image (far in red), GT, Déjà View depth."""
    import matplotlib as mpl  # noqa: PLC0415 - only this debug figure needs it

    mpl.use("Agg")
    from matplotlib import pyplot as plt  # noqa: PLC0415 - as above

    n, h, w, _ = s.views.images.shape
    off = np.linalg.norm(s.dv - s.gt_world, axis=1).reshape(n, h, w)
    far = (off > FAR) & s.has.reshape(n, h, w)
    pick = np.argsort(far.reshape(n, -1).mean(1))[::-1][:4]
    fig, axes = plt.subplots(3, 4, figsize=(16, 9))
    top = float(np.percentile(s.gt20.depth[s.gt20.depth > 0], 99))
    for col, v in enumerate(pick):
        image = s.views.images[v].astype(np.float64)
        image = image / (255.0 if image.max() > 1.5 else 1.0)
        image[far[v]] = 0.6 * image[far[v]] + 0.4 * np.array([1.0, 0.0, 0.0])
        axes[0, col].imshow(image)
        axes[0, col].set_title(
            f"view {v}: {100 * far[v].mean():.1f} % off > {CM * FAR:.0f} cm (red)"
        )
        for row, (d, label) in enumerate(
            (
                (s.gt20.depth[v], "GT depth"),
                (s.scene.depth[v].numpy() * s.similarity[0], "Déjà View depth"),
            ),
            1,
        ):
            im = axes[row, col].imshow(d, vmin=0, vmax=top, cmap="viridis")
            axes[row, col].set_title(f"{label}, m")
        for a in axes[:, col]:
            a.axis("off")
    fig.colorbar(
        im, ax=axes[1:].ravel().tolist(), shrink=0.6, label="depth, m (shared scale)"
    )
    fig.suptitle(
        f"{s.name}: where Déjà View's point is > {CM * FAR:.0f} cm from its pixel's GT point"
    )
    fig.savefig(path, dpi=80)
    plt.close(fig)


def crop_counts(mask: np.ndarray, occ: np.ndarray, rng: np.random.Generator) -> float:
    """Mean candidates in a CROP-cell window centred on a random occupied voxel."""
    sites = np.argwhere(occ)
    total = 0
    for z, y, x in sites[rng.integers(len(sites), size=CROPS)]:
        lo = np.maximum(np.array([z, y, x]) - CROP // 2, 0)
        hi = lo + CROP
        total += mask[lo[0] : hi[0], lo[1] : hi[1], lo[2] : hi[2]].sum()
    return total / CROPS


def oracle(s: Loaded, mask: np.ndarray) -> tuple[np.ndarray, float]:
    """Return the GT99 children of split 4 of the candidates (world), pos / neg ratio."""
    voxel = torch.nonzero(torch.from_numpy(mask)).flip(1)
    occ, vertex = child_targets(
        voxel, torch.from_numpy(s.truth_cells).float(), SPLIT, s.dims
    )
    cloud = to_world(
        children_cloud(voxel, occ, vertex, SPLIT, s.frame).astype(np.float64),
        s.similarity,
    )
    return cloud, float(occ.sum() / max((~occ).sum(), 1))


def rules(s: Loaded) -> list[tuple[str, np.ndarray, int, int]]:
    """Every candidate rule of the plan: name, fine mask, voxels it costs, its reach."""
    occ = (s.grid[0] > 0).numpy()
    cell_cm = CM * s.cell * s.scale * s.similarity[0]
    out = [(f"R{r}", m := dilate(occ, r), int(m.sum()), r) for r in CELL_REACHES]
    for d in PHYSICAL_CM:
        r = int(np.ceil(d / cell_cm))
        out.append((f"P{d}cm", m := dilate(occ, r), int(m.sum()), r))
    free = free_space(s)
    for r in CARVED_REACHES:
        m = (dilate(occ, r) & ~free) | occ
        out.append((f"C{r}", m, int(m.sum()), r))
    coarse_occ = (
        F.max_pool3d(torch.from_numpy(occ).float()[None, None], 2)[0, 0].numpy() > 0
    )
    coarse_gt = occupancy(s.truth_cells / 2, coarse_occ.shape[::-1])
    for r in COARSE_REACHES:
        coarse = dilate(coarse_occ, r)
        fine = np.repeat(np.repeat(np.repeat(coarse & coarse_gt, 2, 0), 2, 1), 2, 2)
        out.append((f"H{r}(coarse)", fine, int(coarse.sum() + fine.sum()), r))
    return out


def coverage(s: Loaded, rng: np.random.Generator) -> dict[str, dict[str, float]]:
    """Return the Pareto numbers of every rule on one scene."""
    unseen_in = ~s.seen & s.inside
    out_dv = s.dv[~s.dv_in]
    passed = voxel_centroids(out_dv, s.child_m) if len(out_dv) else out_dv
    occ = (s.grid[0] > 0).numpy()
    out = {}
    for name, mask, cost, reach in rules(s):
        t = time.time()
        cloud, ratio = oracle(s, mask)
        boxed = score(cloud, s.truth[s.inside], s.seen[s.inside], voxel=False)
        full = score(np.concatenate([cloud, passed]), s.truth, s.seen, voxel=False)
        out[name] = {
            "cover %": 100 * covered(s.truth_cells, mask)[unseen_in].mean(),
            "oracle in-box": boxed["chamfer"],
            "oracle pass": full["chamfer"],
            "voxels k": cost / 1e3,
            "per crop k": crop_counts(mask, occ, rng) / 1e3,
            "pos/neg %": 100 * ratio,
            "reach": reach,
        }
        print(f"{s.name} {name} {out[name]} {time.time() - t:.0f} s", flush=True)
    return out


def boxes(s: Loaded) -> list[str]:
    """Why out-of-box points matter, and every box of the plan on one scene."""
    out_pts = ~s.dv_in
    lines = [f"\n== {s.name}"]
    cells = (s.dv_norm - s.lo) / s.cell
    hi = np.array(s.dims)
    below, above = np.maximum(-cells, 0), np.maximum(cells - hi, 0)
    beyond = np.concatenate([below, above], 1)[out_pts]
    face = np.bincount(beyond.argmax(1), minlength=6) / max(len(beyond), 1)
    keep = s.has & out_pts
    d20 = np.linalg.norm(s.dv[keep] - s.gt_world[keep], axis=1)
    lines.append(
        f"out-of-box Déjà View pixels {out_pts.sum():,} ({100 * out_pts.mean():.2f} %); "
        f"own-pixel error median {CM * np.median(d20):.1f} cm, < 5 cm {100 * (d20 < 0.05).mean():.1f} %; "
        "face x- y- z- x+ y+ z+ (Déjà View's frame) "
        + " ".join(f"{100 * f:.0f}" for f in face)
    )
    full = score(s.dv, s.truth, s.seen, side=0.01)
    norm_truth = to_grid_frame(s.truth, s.similarity, s.scale)
    m = s.similarity[0] * s.scale
    for p in PERCENTILES:
        for pad in PADS:
            for keep_cell in (False, True):
                lo, cell, dims = grid_box(
                    s.dv_norm,
                    192,
                    percentile=p,
                    pad=pad,
                    cell=s.cell if keep_cell else None,
                )
                if keep_cell and max(dims) > MAX_CELLS:
                    lines.append(
                        f"p {p} pad {pad} cell kept: {max(dims)} cells, too many"
                    )
                    continue
                top = lo + np.array(dims) * cell
                dv_in = ((s.dv_norm >= lo) & (s.dv_norm < top)).all(1)
                gt_in = ((norm_truth >= lo) & (norm_truth < top)).all(1)
                inner = score(s.dv[dv_in], s.truth, s.seen, side=0.01)
                lines.append(
                    f"p {p:4.1f} pad {pad:.2f} {'cell kept' if keep_cell else '192 cells'}: "
                    f"cells {max(dims)}, child {CM * cell * m / SPLIT:.2f} cm; out-of-box GT "
                    f"{100 * (~gt_in).mean():5.2f} %, Déjà View {100 * (~dv_in).mean():.3f} %; "
                    f"seen clipping {inner['comp_seen'] - full['comp_seen']:.3f} cm"
                )
    return lines


def baselines(s: Loaded) -> str:
    """One scene's box and both baselines at the child's size (E69e's definitions)."""
    full = score(s.dv, s.truth, s.seen, side=s.child_m)
    inner = score(s.dv[s.dv_in], s.truth, s.seen, side=s.child_m)
    boxed = score(s.dv[s.dv_in], s.truth[s.inside], s.seen[s.inside], side=s.child_m)
    unseen = 100 * (~s.seen & s.inside).sum() / s.inside.sum()
    return (
        f"{s.name}: cells {max(s.dims)}, child {CM * s.child_m:.2f} cm; in-box unseen "
        f"{unseen:.1f} %; out-of-box GT {100 * (~s.inside).mean():.2f} %, Déjà View "
        f"{100 * (~s.dv_in).mean():.3f} %; seen clipping "
        f"{inner['comp_seen'] - full['comp_seen']:.3f} cm; in-box baseline "
        f"{boxed['chamfer']:.3f} cm, pass-through baseline {full['chamfer']:.3f} cm"
    )


def main() -> None:
    """Run one part over the six scenes and write its table."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=["diagnose", "coverage", "box", "table", "lock"]
    )
    parser.add_argument("--scenes", nargs="+", default=list(SCENES))
    parser.add_argument("--config", type=Path, default=Path("configs/e69c.yaml"))
    args = parser.parse_args()
    cfg = load_rung(args.config, "overfit1").cfg
    t0 = time.time()
    lines = [f"E69f {args.command}: split {SPLIT}, seen tolerance 2 cm / 2 %"]
    table: dict[str, dict[str, dict[str, float]]] = {}
    numbers = Path("experiments/results", f"e69f_{args.command}.json")
    if args.command == "table":
        numbers = numbers.with_name("e69f_coverage.json")
        table = json.loads(numbers.read_text())
        args.command, args.scenes = "coverage", []
    for name in args.scenes:
        s = load(cfg, name)
        assert check_projection(s) < 1e-4, check_projection(s)
        if args.command == "diagnose":
            lines += diagnose(s)
            far_figure(s, Path("experiments/figures", f"e69f_far_{name}.jpg"))
        elif args.command == "box":
            lines += boxes(s)
        elif args.command == "lock":
            lines.append(baselines(s))
            table[name] = coverage(s, np.random.default_rng(0))
        else:
            table[name] = coverage(s, np.random.default_rng(0))
        print("\n".join(lines[-3:]), flush=True)
    if table:
        numbers.write_text(json.dumps(table, indent=1))
        lines += pareto(table)
    lines.append(f"\ntotal {time.time() - t0:.0f} s")
    out = Path("experiments/results", f"e69f_{args.command}.txt")
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def pareto(table: dict[str, dict[str, dict[str, float]]]) -> list[str]:
    """Per rule: coverage median / mean / min, the other numbers' medians, per scene."""
    scenes = list(table)
    names = list(table[scenes[0]])
    keys = list(table[scenes[0]][names[0]])
    lines = [
        (
            "\nrule            cover med / mean / min   oracle in-box med  pass med  "
            "voxels k med  per crop k med  pos/neg % med"
        )
    ]
    for n in names:
        c = np.array([table[s][n]["cover %"] for s in scenes])
        med = {k: np.median([table[s][n][k] for s in scenes]) for k in keys}
        lines.append(
            f"{n:14s} {np.median(c):6.1f} / {c.mean():5.1f} / {c.min():5.1f}"
            f"   {med['oracle in-box']:8.3f}  {med['oracle pass']:8.3f}"
            f"  {med['voxels k']:10.0f}  {med['per crop k']:12.2f}  {med['pos/neg %']:10.2f}"
        )
    lines.append(
        "\nper scene: cover % | oracle in-box | oracle pass | voxels k | per crop k | pos/neg %"
    )
    lines.extend(
        f"{n:14s} "
        + "  ".join(
            f"{s[:10]} " + "/".join(f"{table[s][n][k]:.2f}" for k in keys)
            for s in scenes
        )
        for n in names
    )
    return lines


if __name__ == "__main__":
    main()
