"""E69c rung 1: look at the inputs and the fixed targets of the bedroom before any training.

A 4-cell slab (seen along y, x to the right, z up, in cells of the generator's grid) of the
bedroom `ai_006_007`: the grid channels (log point count, mean colour), the first three
PCA channels of the back-projected DINOv2 volume, the occupancy targets of the children of
the candidate voxels, the target vertices on top of the GT points, and Deja View's points
against the GT's. Also prints the shapes, ranges and counts the figure rests on.

uv run python -m experiments.scripts.e69c_data configs/e69c.yaml
"""

import sys
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.refinement.generator import (
    candidate_mask,
    child_corner,
    child_targets,
    for_arm,
    load_rung,
    load_scene,
)
from src.refinement.volume import voxelize

BEDROOM = "ai_006_007_cam_00"
SLAB = 4  # cells along y


def slab_image(channels: torch.Tensor, count: torch.Tensor) -> np.ndarray:
    """Return the count-weighted mean over the slab's y of channels (C, Z, 4, X) as (Z, X, C)."""
    weight = count[None].clamp(min=0)
    total = weight.sum(2).clamp(min=1e-6)
    return ((channels * weight).sum(2) / total).permute(1, 2, 0).numpy()


def in_slab(points: np.ndarray, rows: slice) -> np.ndarray:
    """Return which of the points (n, 3), in cells, lie in the slab's y range."""
    return (points[:, 1] >= rows.start) & (points[:, 1] < rows.stop)


def scale(image: np.ndarray, occupied: np.ndarray) -> np.ndarray:
    """Stretch each channel to [0, 1] between the 2nd and 98th percentile of the occupied cells."""
    lo, hi = np.percentile(image[occupied], [2, 98], axis=0)
    return np.clip((image - lo) / np.maximum(hi - lo, 1e-6), 0, 1)


def main() -> None:
    """Draw `experiments/figures/e69c_data.jpg` and print what it shows."""
    cfg = for_arm(load_rung(Path(sys.argv[1]), "overfit1"), "ovoxel-dino")
    pts, gt, has, _ = load_scene(cfg, BEDROOM, "none")
    grid = voxelize(pts, semantics=False)
    dv = ((pts.points - pts.lo) / pts.cell).numpy()
    truth = ((gt[has] - pts.lo) / pts.cell).numpy()
    y0 = int(np.median(dv[:, 1]))
    rows = slice(y0 - SLAB // 2, y0 + SLAB // 2)
    count = torch.expm1(grid[0, :, rows, :])
    occupied = (count.sum(1) > 0).numpy()
    print(
        f"grid {tuple(grid.shape)}, y slab {rows}, {int(occupied.sum())} occupied columns"
    )
    print(
        f"features: {grid[8:].shape[0]} channels, range {float(grid[8:].min()):.2f} to {float(grid[8:].max()):.2f}"
    )
    colour = scale(slab_image(grid[2:5, :, rows, :], count), occupied)
    pca = scale(slab_image(grid[8:11, :, rows, :], count), occupied)

    voxel = torch.nonzero(candidate_mask(grid, cfg.reach)).flip(1)
    occ, vertex = child_targets(voxel, torch.from_numpy(truth), cfg.split, pts.dims)
    corner = (voxel.float()[:, None] + child_corner(cfg.split)).numpy()
    inside = (voxel[:, 1] >= rows.start) & (voxel[:, 1] < rows.stop)
    print(
        f"{len(voxel)} candidate voxels, {int(occ.sum())} occupied children of "
        f"{occ.numel()}; vertex targets in [{float(vertex[occ].min()):.2f}, {float(vertex[occ].max()):.2f}]"
    )
    centre = corner + 0.5 / cfg.split
    mine = inside.numpy()
    target = (corner + vertex.numpy() / cfg.split)[mine][occ.numpy()[mine]]

    fig, axes = plt.subplots(2, 3, figsize=(18, 11.5), constrained_layout=True)
    extent = (0, grid.shape[3], 0, grid.shape[1])
    for ax, image, title in (
        (
            axes[0, 0],
            np.log1p(count.sum(1).numpy()),
            "log point count, summed over the slab",
        ),
        (axes[0, 1], colour, "mean colour of the points"),
        (
            axes[0, 2],
            pca,
            "DINOv2 PCA channels 1-3, back-projected and averaged per voxel",
        ),
    ):
        ax.imshow(
            image,
            origin="lower",
            extent=extent,
            cmap="gray" if image.ndim == 2 else None,
        )
        ax.set_title(title)
    kept = occ.numpy()[mine]
    flat = centre[mine].reshape(-1, 3)
    axes[1, 0].scatter(
        flat[~kept.reshape(-1), 0],
        flat[~kept.reshape(-1), 2],
        s=2,
        c="0.8",
        rasterized=True,
    )
    axes[1, 0].scatter(
        flat[kept.reshape(-1), 0],
        flat[kept.reshape(-1), 2],
        s=2,
        c="tab:red",
        rasterized=True,
    )
    axes[1, 0].set_title(
        f"children of the candidate voxels: occupied (red), empty (grey), split {cfg.split}"
    )
    axes[1, 1].scatter(
        truth[in_slab(truth, rows), 0],
        truth[in_slab(truth, rows), 2],
        s=0.05,
        c="tab:blue",
        alpha=0.4,
        rasterized=True,
    )
    axes[1, 1].scatter(target[:, 0], target[:, 2], s=1.5, c="tab:red", rasterized=True)
    axes[1, 1].set_title("target vertices (red) on the GT points (blue)")
    axes[1, 2].scatter(
        truth[in_slab(truth, rows), 0],
        truth[in_slab(truth, rows), 2],
        s=0.05,
        c="tab:orange",
        alpha=0.4,
        rasterized=True,
    )
    axes[1, 2].scatter(
        dv[in_slab(dv, rows), 0],
        dv[in_slab(dv, rows), 2],
        s=0.05,
        c="tab:blue",
        alpha=0.4,
        rasterized=True,
    )
    axes[1, 2].set_title("Déjà View's points (blue) and the GT's (orange)")
    for ax in axes.flat:
        ax.set_aspect("equal")
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        ax.set_xlabel("x (cells, 7.2 cm median)")
        ax.set_ylabel("z (cells)")
    fig.suptitle(
        "E69c rung 1, bedroom ai_006_007: a 4-cell slab seen along y; the inputs on top, "
        "the fixed targets below",
        fontsize=12,
    )
    fig.savefig("experiments/figures/e69c_data.jpg", dpi=100)


if __name__ == "__main__":
    main()
