"""E69a rung 1: look at the generator's inputs and targets, and measure what a move can reach.

A generator moves a point by at most `offset` cells per axis, so the share of ground-truth
points that have an input point within `offset` is the largest completeness it can reach,
and the share of input points with a ground-truth point within `offset` the largest
accuracy. Both are measured per scene (all 20 views, 192 cells, in Deja View's frame, the
cell of each scene's own grid) for offsets of 0.5 to 3 cells, with the distribution of
each pixel's distance to its own ground-truth point; and the bedroom's inputs and
targets are drawn as a 4-cell slab.

uv run python experiments/scripts/e69a_data.py configs/e69a.yaml overfit3 dev val
"""

import sys
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree

from src.loaders.hypersim.config import load_config
from src.refinement.generator import load_rung, load_scene

OFFSETS = (0.5, 1.0, 2.0, 3.0)
CAP = 300_000
BEDROOM = "ai_006_007_cam_00"


def reach(a: np.ndarray, b: np.ndarray, d: float, rng: np.random.Generator) -> float:
    """Share of the points a (n, 3) that have a point of b (m, 3) within d on every axis."""
    a = a[rng.choice(len(a), min(len(a), CAP), replace=False)]
    dist, _ = cKDTree(b, balanced_tree=False, compact_nodes=False).query(
        a, p=np.inf, distance_upper_bound=d, workers=-1
    )
    return float(np.isfinite(dist).mean())


def main() -> None:
    """Print the reachability table of the named splits and draw the bedroom."""
    config, rung, *splits = sys.argv[1:]
    cfg = load_rung(Path(config), rung).cfg
    scenes = load_config(Path("configs/hypersim.yaml")).scenes(splits)
    rng = np.random.default_rng(0)
    head = "scene | cell (norm.) | median / p90 / p99 error (cells) | "
    head += " | ".join(f"acc@{d:g} comp@{d:g}" for d in OFFSETS)
    lines = [head]
    for scene, cam in scenes.items():
        name = f"{scene}_{cam}"
        pts, gt, has, _ = load_scene(cfg, name, "none")
        dv, truth = pts.points[has].numpy(), gt[has].numpy()
        cell = pts.cell
        err = np.linalg.norm(dv - truth, axis=1) / cell
        row = f"{name} | {cell:.4f} | {np.median(err):.2f} / {np.percentile(err, 90):.2f} / {np.percentile(err, 99):.2f}"
        for d in OFFSETS:
            acc = reach(dv, truth, d * cell, rng)
            comp = reach(truth, dv, d * cell, rng)
            row += f" | {acc:.3f} {comp:.3f}"
        lines.append(row)
        print(row, flush=True)
        if name == BEDROOM:
            draw(dv, truth, cell, err)
    Path("experiments/results/e69a_reach.txt").write_text("\n".join(lines) + "\n")


def draw(dv: np.ndarray, truth: np.ndarray, cell: float, err: np.ndarray) -> None:
    """Draw a 4-cell slab of the bedroom's inputs and targets, and the error histogram."""
    y0 = np.median(dv[:, 1])
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.6), constrained_layout=True)
    for ax, (title, cloud, colour) in zip(
        axes[:3],
        [
            ("Déjà View's points", dv, "tab:blue"),
            ("ground truth", truth, "tab:orange"),
            (None, None, None),
        ],
        strict=True,
    ):
        for c, col in (
            ((dv, "tab:blue"), (truth, "tab:orange"))
            if cloud is None
            else ((cloud, colour),)
        ):
            slab = np.abs(c[:, 1] - y0) < 2 * cell
            ax.scatter(
                c[slab, 0] / cell,
                c[slab, 2] / cell,
                s=0.05,
                c=col,
                alpha=0.5,
                rasterized=True,
            )
        ax.set_title(title or "both: blue input, orange target")
        ax.set_aspect("equal")
        ax.set_xlabel("x (cells)")
        ax.set_ylabel("z (cells)")
    axes[3].hist(np.minimum(err, 12), bins=120, color="tab:blue")
    for d in OFFSETS[1:]:
        axes[3].axvline(d, color="k", lw=0.8)
    axes[3].set_title("pixel's distance to its own GT point; lines at 1, 2, 3 cells")
    axes[3].set_xlabel("cells (clipped at 12)")
    axes[3].set_ylabel("pixels")
    fig.suptitle(
        "E69a rung 1, bedroom ai_006_007: a 4-cell slab of the inputs and targets, seen along y",
        fontsize=11,
    )
    fig.savefig("experiments/figures/e69a_data.jpg", dpi=110)


if __name__ == "__main__":
    main()
