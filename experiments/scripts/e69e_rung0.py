"""E69e rung 0: the O-Voxel references against the 99-frame ground truth, without training.

Every cloud is mapped from Déjà View's frame to the ground truth's by the protocol's one
Umeyama onto GT20 (gt = s R dv + t) and scored against GT99v (E69d's 99 frames, one point per
occupied 1 cm cube). The output is voxelized the same way first, so neither side's density
weights the means; the raw output's scores are kept next to it. Completeness is split over
the GT99v points that the 20 input views see and those they do not (E69d's test, 2 cm / 2 %).

References, the generator's grid and children (`generator.child_targets`): `dv-ovoxel-s<k>`
(the children holding a Déjà View point, at those points' mean), `oracle-ovoxel-<gt>-s<k>-r<r>`
(every child of the candidate voxels within `r` cells of a Déjà View voxel that holds a point
of GT20 or GT99, at their mean: what zero loss produces), the plain voxel density filter at q.
Coverage: the share of GT99v's seen and unseen points whose voxel is a candidate at reach r.
Halves: the GT99v box is cut at the median of its longest horizontal axis (world z is up);
A trains in E69e rung 3c, B is held out, and every score is also taken inside B alone.

`uv run python -m experiments.scripts.e69e_rung0` writes
`experiments/results/e69e_rung0_<scene>.txt` and `experiments/figures/e69e_rung0_<scene>.jpg`.
"""

import argparse
import time
from pathlib import Path

import matplotlib as mpl
import numpy as np
import torch

mpl.use("Agg")
import matplotlib.pyplot as plt

from experiments.scripts.e69c_oracle import Frame, children_cloud
from experiments.scripts.e69d_all_frames_gt import seen_by, voxel_centroids
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

CM = 100.0
SIDE = 0.01  # metres, the voxel of GT99v and of every scored output
SEEN_TOLERANCE = (0.02, 0.02)  # E69d's headline
SPLITS = (2, 4)
REACHES = (1, 2, 3, 4)
QUANTILES = (0.5, 0.7, 0.9)
UP = 2  # Hypersim's world z


def to_grid_frame(
    world: np.ndarray, similarity: tuple[float, np.ndarray, np.ndarray], scale: float
) -> np.ndarray:
    """Map world points (N, 3) into the grid's normalized frame: R^T (p - t) / (s scale)."""
    s, rotation, t = similarity
    assert world.ndim == 2, world.shape
    assert world.shape[1] == 3, world.shape
    assert s > 0, s
    assert scale > 0, scale
    return (world - t) @ rotation / (s * scale)


def to_world(
    dv: np.ndarray, similarity: tuple[float, np.ndarray, np.ndarray]
) -> np.ndarray:
    """Map points (N, 3) of Déjà View's frame to the world: s R p + t."""
    s, rotation, t = similarity
    return s * dv @ rotation.T + t


def covered(cells: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Whether each point (N, 3), in cells x y z, lies in a True voxel of mask (Z, Y, X)."""
    assert cells.ndim == 2, cells.shape
    assert cells.shape[1] == 3, cells.shape
    assert mask.ndim == 3, mask.shape
    index = np.floor(cells).astype(np.int64)
    dims = np.array(mask.shape[::-1])
    inside = ((index >= 0) & (index < dims)).all(1)
    out = np.zeros(len(cells), dtype=bool)
    x, y, z = index[inside].T
    out[inside] = mask[z, y, x]
    return out


def halves(points: np.ndarray, up: int = UP) -> tuple[int, float]:
    """Return the longest horizontal axis of the points' box and its median."""
    assert points.ndim == 2, points.shape
    assert points.shape[1] == 3, points.shape
    flat = [a for a in range(3) if a != up]
    extent = points[:, flat].max(0) - points[:, flat].min(0)
    axis = flat[int(np.argmax(extent))]
    return axis, float(np.median(points[:, axis]))


def score(
    cloud: np.ndarray,
    truth: np.ndarray,
    seen: np.ndarray,
    *,
    voxel: bool = True,
    region: tuple[int, float] | None = None,
    side: float = SIDE,
) -> dict[str, float]:
    """Accuracy, completeness (seen / unseen), Chamfer in cm of a world cloud against GT99v.

    `region` (axis, cut) keeps the points of both with coordinate >= cut, half B; the
    output is voxelized at `side` metres unless `voxel` is False.
    """
    assert len(truth) == len(seen), (len(truth), len(seen))
    out = voxel_centroids(cloud, side) if voxel else cloud
    if region is not None:
        axis, cut = region
        out = out[out[:, axis] >= cut]
        keep = truth[:, axis] >= cut
        truth, seen = truth[keep], seen[keep]
    acc = nearest_distances(truth, out)
    comp = nearest_distances(out, truth)
    return {
        "points": float(len(out)),
        "accuracy": CM * acc.mean(),
        "completeness": CM * comp.mean(),
        "comp_seen": CM * comp[seen].mean(),
        "comp_unseen": CM * comp[~seen].mean() if (~seen).any() else float("nan"),
        "chamfer": CM * (acc.mean() + comp.mean()) / 2,
    }


def line(name: str, s: dict[str, float]) -> str:
    """One row of the table."""
    return (
        f"{name:34s} {s['points'] / 1e6:6.2f} M  acc {s['accuracy']:6.3f}  "
        f"comp {s['completeness']:6.3f} (seen {s['comp_seen']:6.3f}, unseen "
        f"{s['comp_unseen']:7.3f})  Chamfer {s['chamfer']:6.3f}"
    )


def figure(
    truth: np.ndarray,
    seen: np.ndarray,
    cover: dict[int, np.ndarray],
    region: tuple[int, float],
    out: Path,
    title: str,
) -> None:
    """Top and side views of GT99v: seen grey, unseen covered at each reach or not."""
    rng = np.random.default_rng(0)
    sample = rng.random(len(truth)) < 0.15
    axis, cut = region
    other = 1 - axis if axis < 2 else 0
    reaches = sorted(cover)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.2))
    for ax, (u, v, name) in zip(
        axes, [(axis, other, "top"), (axis, UP, "side")], strict=True
    ):
        s = sample & seen
        ax.scatter(
            truth[s, u], truth[s, v], s=0.2, c="0.8", rasterized=True, label="seen"
        )
        colours = plt.cm.viridis(np.linspace(0.1, 0.9, len(reaches)))
        done = np.zeros(len(truth), dtype=bool)
        for r, c in zip(reaches, colours, strict=True):
            new = sample & ~seen & cover[r] & ~done
            ax.scatter(
                truth[new, u],
                truth[new, v],
                s=0.6,
                color=c,
                rasterized=True,
                label=f"unseen, candidate at reach {r}",
            )
            done |= cover[r]
        miss = sample & ~seen & ~done
        ax.scatter(
            truth[miss, u],
            truth[miss, v],
            s=0.6,
            c="crimson",
            rasterized=True,
            label="unseen, no candidate",
        )
        ax.axvline(cut, color="k", lw=1, ls="--")
        ax.text(cut, ax.get_ylim()[1], "  B (held out) →", va="top", fontsize=9)
        ax.set_xlabel(f"world {'xyz'[u]} (m)")
        ax.set_ylabel(f"world {'xyz'[v]} (m)")
        ax.set_aspect("equal")
        ax.set_title(f"{name} view", fontsize=10)
    axes[0].legend(markerscale=12, fontsize=8, loc="lower left")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


def main() -> None:
    """Score every rung-0 reference of the scene and write the table and figure."""
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
    pts, gt20_grid, has, scale = load_scene(cfg, args.scene, "none")
    _, rays, depth, _ = scene_points(
        geometry, views.images, np.full(geometry.depth.shape, -1), {}, cfg.resolution
    )
    _, _, similarity = gt_alignment(rays, depth, views, gt20)
    truth = np.load(folder / f"gt99v1cm_{cfg.views}.npz")["points"]
    seen = seen_by(truth, views, gt20, SEEN_TOLERANCE)
    region = halves(truth)
    in_b = truth[:, region[0]] >= region[1]
    lines = [
        (
            f"E69e rung 0, {args.scene}: GT99v {len(truth):,} points (1 cm), unseen "
            f"{100 * (~seen).mean():.1f} %; alignment scale {similarity[0]:.4f}"
        ),
        (
            f"halves: world {'xyz'[region[0]]} cut at {region[1]:.3f} m; "
            f"A {(~in_b).sum():,} points, unseen {100 * (~seen[~in_b]).mean():.1f} %; "
            f"B {in_b.sum():,} points, unseen {100 * (~seen[in_b]).mean():.1f} %"
        ),
        (
            f"grid: {cfg.resolution} cells on the longest side, cell "
            f"{CM * pts.cell * scale * similarity[0]:.2f} cm, dims {pts.dims}"
        ),
    ]

    frame = Frame(pts.lo, pts.cell, scale)
    grid = voxelize(pts, semantics=False)
    truth_cells = (
        torch.from_numpy(to_grid_frame(truth, similarity, scale)).float() - pts.lo
    ) / pts.cell
    gt20_cells = (gt20_grid[has] - pts.lo) / pts.cell
    dv_cells = (pts.points - pts.lo) / pts.cell
    lines.append("\ncoverage of GT99v by the candidate voxels (share of points)")
    masks, cover = {}, {}
    for r in REACHES:
        masks[r] = candidate_mask(grid, r)
        cover[r] = covered(truth_cells.numpy(), masks[r].numpy())
        lines.append(
            f"  reach {r}: seen {100 * cover[r][seen].mean():5.1f} %, unseen "
            f"{100 * cover[r][~seen].mean():5.1f} %, candidates {int(masks[r].sum()):,} voxels"
        )
    inside = covered(truth_cells.numpy(), np.ones(grid.shape[1:], dtype=bool))
    lines.append(
        f"  inside the grid at all: seen {100 * inside[seen].mean():.1f} %, unseen "
        f"{100 * inside[~seen].mean():.1f} %"
    )

    clouds: dict[str, np.ndarray] = {}
    dv_scene = scene_from_geometry(geometry)
    pixels = points_from_rays(dv_scene.depth.double(), dv_scene.rays.double()).numpy()
    clouds["Déjà View, every pixel"] = to_world(pixels.reshape(-1, 3), similarity)
    clouds["Déjà View, pixels with GT"] = to_world(
        pixels.reshape(-1, 3)[has.numpy()], similarity
    )
    count = instance_voxel_density(
        pixels,
        dv_scene.depth.numpy(),
        dv_scene.k.numpy(),
        np.full(geometry.depth.shape, -1),
    )
    for q in QUANTILES:
        keep = keep_top(ranks(count), q).reshape(-1)
        clouds[f"density q {q}"] = clouds["Déjà View, every pixel"][keep]
    for split in SPLITS:
        voxel = torch.nonzero(masks[1]).flip(1)
        occ, vertex = child_targets(voxel, dv_cells, split, pts.dims)
        clouds[f"dv-ovoxel s{split}"] = children_cloud(voxel, occ, vertex, split, frame)
        for r in REACHES:
            voxel = torch.nonzero(masks[r]).flip(1)
            for name, cells in (("gt20", gt20_cells), ("gt99", truth_cells)):
                occ, vertex = child_targets(voxel, cells, split, pts.dims)
                clouds[f"oracle-ovoxel {name} s{split} r{r}"] = children_cloud(
                    voxel, occ, vertex, split, frame
                )
    for name in list(clouds):
        if name.startswith(("dv-ovoxel", "oracle-ovoxel")):
            clouds[name] = to_world(clouds[name].astype(np.float64), similarity)

    for title, kwargs in (
        ("whole room, output voxelized at 1 cm (the headline)", {}),
        ("whole room, raw output", {"voxel": False}),
        ("half B only, output voxelized at 1 cm", {"region": region}),
    ):
        lines.append(f"\n{title}; cm")
        for name, cloud in clouds.items():
            lines.append(line(name, score(cloud, truth, seen, **kwargs)))
    lines.append(f"\ntotal {time.time() - t0:.0f} s")

    stem = f"e69e_rung0_{args.scene}"
    Path("experiments/results", f"{stem}.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    figure(
        truth,
        seen,
        cover,
        region,
        Path("experiments/figures", f"{stem}.jpg"),
        f"{args.scene}, GT99v: grey = seen by the 20 input views; unseen coloured by the "
        "smallest reach whose candidate voxels hold it, red = none up to 4",
    )


if __name__ == "__main__":
    main()
