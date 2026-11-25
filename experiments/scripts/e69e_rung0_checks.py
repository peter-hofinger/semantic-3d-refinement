"""E69e rung-0 checks: G0a's gap as quantization or bug, and G0c inside Déjà View's box.

Bug check: `dv-ovoxel` (the children of split k holding a Déjà View point, at their mean, from
`generator.child_targets`) is recomputed by an independent path, `child_means`: each point's
child is floor(k c) of its grid cell coordinate c, the points are grouped by it and averaged.
The two sets must agree to float32 precision. Matched scorer: the outputs are voxelized at the
child's size (cell / k in metres) instead of rung 0's 1 cm, against the same GT99v. G0c inside
the box: coverage and unseen completeness over the unseen GT99v points inside the grid box
(Déjà View's 1-99 % box, padded 5 %) and outside it, separately. Out-of-box: the Déjà View
points outside the box, which `child_targets` drops, restored to `dv-ovoxel` (voxelized at the
child's size) and removed from Déjà View; completeness split by the box.

`uv run python -m experiments.scripts.e69e_rung0_checks` writes
`experiments/results/e69e_rung0_checks_<scene>.txt`.
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from experiments.scripts.e69c_oracle import Frame, children_cloud
from experiments.scripts.e69d_all_frames_gt import seen_by, voxel_centroids
from experiments.scripts.e69e_rung0 import (
    CM,
    REACHES,
    SEEN_TOLERANCE,
    covered,
    line,
    score,
    to_grid_frame,
    to_world,
)
from src.analysis.filters import instance_voxel_density, keep_top, ranks
from src.evaluation.labeled import nearest_distances
from src.geometry import Geometry
from src.loaders.ground_truth import GroundTruth
from src.loaders.views import Views
from src.refinement.generator import (
    candidate_mask,
    child_targets,
    load_rung,
    load_scene,
)
from src.refinement.multiview import points_from_rays, scene_from_geometry
from src.refinement.volume import gt_alignment, scene_points, voxelize

SPLIT = 4
DENSITY_Q = 0.7  # rung 0's best on the bedroom


def child_means(
    cells: np.ndarray, split: int, dims: tuple[int, int, int]
) -> tuple[np.ndarray, int]:
    """Mean (M, 3), in cells, of the points (N, 3) in each child of the grid; points outside."""
    assert cells.ndim == 2, cells.shape
    assert cells.shape[1] == 3, cells.shape
    assert split >= 1, split
    fine = np.floor(cells * split).astype(np.int64)
    inside = ((fine >= 0) & (fine < np.array(dims) * split)).all(1)
    _, group = np.unique(fine[inside], axis=0, return_inverse=True)
    group = group.ravel()
    count = np.bincount(group)
    means = np.stack(
        [np.bincount(group, weights=cells[inside, a]) / count for a in range(3)], 1
    )
    assert np.isfinite(means).all()
    return means, int((~inside).sum())


def same_set(a: np.ndarray, b: np.ndarray) -> float:
    """Largest distance from a point of either set to its nearest in the other."""
    return float(max(nearest_distances(a, b).max(), nearest_distances(b, a).max()))


def main() -> None:
    """Run both checks on the scene and write the table."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", default="ai_006_007_cam_00")
    parser.add_argument("--config", type=Path, default=Path("configs/e69c.yaml"))
    args = parser.parse_args()
    cfg = load_rung(args.config, "overfit1").cfg
    folder = cfg.root / args.scene
    t0 = time.time()

    views = Views.load(folder / f"{cfg.views}.npz")
    geometry = Geometry.load(folder / f"{cfg.backbone}_{cfg.views}.npz")
    gt20 = GroundTruth.load(folder / f"gt_{cfg.views}.npz")
    pts, _, _, scale = load_scene(cfg, args.scene, "none")
    _, rays, depth, _ = scene_points(
        geometry, views.images, np.full(geometry.depth.shape, -1), {}, cfg.resolution
    )
    _, _, similarity = gt_alignment(rays, depth, views, gt20)
    truth = np.load(folder / f"gt99v1cm_{cfg.views}.npz")["points"]
    seen = seen_by(truth, views, gt20, SEEN_TOLERANCE)
    frame = Frame(pts.lo, pts.cell, scale)
    grid = voxelize(pts, semantics=False)
    child_m = pts.cell * scale * similarity[0] / SPLIT
    lines = [
        f"E69e rung-0 checks, {args.scene}: split {SPLIT}, child {CM * child_m:.3f} cm"
    ]

    dv_cells = (pts.points - pts.lo) / pts.cell
    voxel = torch.nonzero(candidate_mask(grid, 1)).flip(1)
    occ, vertex = child_targets(voxel, dv_cells, SPLIT, pts.dims)
    dv_ovoxel = to_world(
        children_cloud(voxel, occ, vertex, SPLIT, frame).astype(np.float64), similarity
    )
    means, outside = child_means(dv_cells.double().numpy(), SPLIT, pts.dims)
    independent = to_world(
        (pts.lo.double().numpy() + means * pts.cell) * scale, similarity
    )
    gap = same_set(dv_ovoxel, independent)
    lines += [
        "\nbug check: dv-ovoxel (child_targets) against the independent child means",
        (
            f"  points {len(dv_ovoxel):,} / {len(independent):,}; largest nearest distance "
            f"{CM * gap:.2e} cm; Déjà View points outside the grid box {outside:,} "
            f"of {len(dv_cells):,} ({100 * outside / len(dv_cells):.2f} %)"
        ),
    ]

    dv_scene = scene_from_geometry(geometry)
    pixels = points_from_rays(dv_scene.depth.double(), dv_scene.rays.double()).numpy()
    dv = to_world(pixels.reshape(-1, 3), similarity)
    count = instance_voxel_density(
        pixels,
        dv_scene.depth.numpy(),
        dv_scene.k.numpy(),
        np.full(geometry.depth.shape, -1),
    )
    clouds = {
        "Déjà View": dv,
        f"density q {DENSITY_Q}": dv[keep_top(ranks(count), DENSITY_Q).reshape(-1)],
        f"dv-ovoxel s{SPLIT}": dv_ovoxel,
    }
    for name, side in (("1 cm (rung 0)", 0.01), (f"{CM * child_m:.2f} cm", child_m)):
        lines.append(f"\noutput voxelized at {name}, against GT99v (1 cm); cm")
        for label, cloud in clouds.items():
            lines.append(line(label, score(cloud, truth, seen, side=side)))
    lines.append("\nunvoxelized; cm")
    lines.append(
        line(f"dv-ovoxel s{SPLIT}", score(dv_ovoxel, truth, seen, voxel=False))
    )
    lines.append(
        line("child means (independent)", score(independent, truth, seen, voxel=False))
    )

    box = np.ones(grid.shape[1:], dtype=bool)
    lo = pts.lo.double().numpy()
    truth_cells = (to_grid_frame(truth, similarity, scale) - lo) / pts.cell
    inside = covered(truth_cells, box)
    dv_in = covered((to_grid_frame(dv, similarity, scale) - lo) / pts.cell, box)
    assert (~dv_in).sum() == outside, ((~dv_in).sum(), outside)
    dropped = voxel_centroids(dv[~dv_in], child_m)
    restored = np.concatenate([dv_ovoxel, dropped])
    lines += [
        "\nout-of-box points: restored to dv-ovoxel, removed from Déjà View; cm",
        line(
            f"Déjà View, voxelized at {CM * child_m:.2f} cm",
            score(dv, truth, seen, side=child_m),
        ),
        line(f"dv-ovoxel s{SPLIT}", score(dv_ovoxel, truth, seen, voxel=False)),
        line(
            f"dv-ovoxel s{SPLIT} + dropped", score(restored, truth, seen, voxel=False)
        ),
        line("Déjà View inside the box", score(dv[dv_in], truth, seen, side=child_m)),
        (
            "  completeness over the GT99v points inside / outside the grid box "
            f"({inside.sum():,} / {(~inside).sum():,}):"
        ),
    ]
    for label, cloud in (
        ("Déjà View", voxel_centroids(dv, child_m)),
        (f"dv-ovoxel s{SPLIT}", dv_ovoxel),
        (f"dv-ovoxel s{SPLIT} + dropped", restored),
    ):
        comp = nearest_distances(cloud, truth)
        lines.append(
            f"  {label:30s} inside {CM * comp[inside].mean():6.3f}, "
            f"outside {CM * comp[~inside].mean():7.3f} cm"
        )
    unseen_in, unseen_out = ~seen & inside, ~seen & ~inside
    lines += [
        "\nG0c inside the grid box",
        (
            f"  unseen GT99v points: inside {unseen_in.sum():,}, outside {unseen_out.sum():,} "
            f"({100 * unseen_out.sum() / (~seen).sum():.1f} % of the unseen)"
        ),
    ]
    gt99_cells = torch.from_numpy(truth_cells).float()
    for r in REACHES:
        mask = candidate_mask(grid, r)
        cover = covered(truth_cells, mask.numpy())
        voxel = torch.nonzero(mask).flip(1)
        occ, vertex = child_targets(voxel, gt99_cells, SPLIT, pts.dims)
        oracle = to_world(
            children_cloud(voxel, occ, vertex, SPLIT, frame).astype(np.float64),
            similarity,
        )
        comp = nearest_distances(oracle, truth)
        lines.append(
            f"  reach {r}: covers {100 * cover[unseen_in].mean():5.1f} % of the unseen inside; "
            f"oracle GT99 s{SPLIT} unseen completeness inside {CM * comp[unseen_in].mean():6.2f}, "
            f"outside {CM * comp[unseen_out].mean():6.2f} cm"
        )
    comp = nearest_distances(dv, truth)
    lines.append(
        f"  Déjà View unseen completeness inside {CM * comp[unseen_in].mean():6.2f}, "
        f"outside {CM * comp[unseen_out].mean():6.2f} cm"
    )
    lines.append(f"\ntotal {time.time() - t0:.0f} s")

    out = Path("experiments/results", f"e69e_rung0_checks_{args.scene}.txt")
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
