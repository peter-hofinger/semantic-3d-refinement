"""E69b: does Deja View put a point in every voxel the ground truth occupies, unthresholded.

Both clouds (all pixels of 20 views; the GT points of the same pixels) are put on one voxel grid of 96,
192 or 384 cells along the longest side. Per scene: recall = share of GT-occupied voxels that hold a
Deja View point (exactly, and with a point within one voxel on every axis), precision = the reverse,
for Deja View's points kept by confidence (all, top 75%, top 50%), and for the GT voxels with no point
the distance to the nearest Deja View voxel (displaced evidence, or a hole).

uv run python experiments/scripts/e69b_voxel_coverage.py runs/e69b experiments/figures/e69b_coverage.jpg dev val
"""

import json
import sys
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree

from src.geometry import Geometry
from src.loaders.ground_truth import GroundTruth
from src.loaders.hypersim.config import load_config
from src.loaders.views import Views
from src.refinement.volume import gt_alignment, scene_points

ROOT = Path("data/processed/hypersim")
VIEWS = "views_n20_uniform_504"
BACKBONE = "dejaview_k16"
RESOLUTIONS = (96, 192, 384)
KEEPS = (1.0, 0.75, 0.5)
BEDROOM = "ai_006_007_cam_00"
STEPS = np.array(
    [(i, j, k) for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)]
)


def voxels(points: np.ndarray, lo: np.ndarray, cell: float) -> np.ndarray:
    """Return the unique integer voxel indices (M, 3) of the points."""
    return np.unique(np.floor((points - lo) / cell).astype(np.int64), axis=0)


def keys(index: np.ndarray, dims: np.ndarray) -> np.ndarray:
    """Flat key of each voxel index (M, 3)."""
    return (index[:, 0] * dims[1] + index[:, 1]) * dims[2] + index[:, 2]


def within_one(a: np.ndarray, b: np.ndarray, dims: np.ndarray) -> np.ndarray:
    """Per voxel of a (M, 3): whether a voxel of b is within one voxel on every axis."""
    held = np.sort(keys(b, dims))
    hit = np.zeros(len(a), bool)
    for step in STEPS:
        near = a + step
        ok = ((near >= 0) & (near < dims)).all(1)
        k = keys(np.clip(near, 0, dims - 1), dims)
        pos = np.minimum(np.searchsorted(held, k), len(held) - 1)
        hit |= ok & (held[pos] == k)
    return hit


def scene_metrics(
    dv: np.ndarray, conf: np.ndarray, gt: np.ndarray, size: int
) -> tuple[dict, dict]:
    """Per keep the recall and precision; also the voxel classes of the finest-keep grid."""
    lo = np.minimum(dv.min(0), gt.min(0))
    hi = np.maximum(dv.max(0), gt.max(0))
    cell = float((hi - lo).max() / size)
    dims = np.ceil((hi - lo) / cell).astype(np.int64) + 1
    g = voxels(gt, lo, cell)
    out, detail = {}, {}
    for keep in KEEPS:
        sel = conf >= np.quantile(conf, 1 - keep)
        d = voxels(dv[sel], lo, cell)
        gk, dk = keys(g, dims), keys(d, dims)
        exact_g = np.isin(gk, dk)
        exact_d = np.isin(dk, gk)
        near_g, near_d = within_one(g, d, dims), within_one(d, g, dims)
        missed = g[~exact_g]
        gap = cKDTree(d).query(missed)[0] if len(missed) and len(d) else np.zeros(0)
        out[keep] = {
            "recall": float(exact_g.mean()),
            "recall1": float(near_g.mean()),
            "precision": float(exact_d.mean()),
            "precision1": float(near_d.mean()),
            "missed_median_cells": float(np.median(gap)) if len(gap) else 0.0,
            "missed_far": float((gap > 3).mean()) if len(gap) else 0.0,
            "gt_voxels": len(g),
            "dv_voxels": len(d),
        }
        detail[keep] = (g, exact_g, near_g, d, near_d, cell, lo)
    return out, detail


def load(name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Deja View's points, their confidence, the GT points (frame of Deja View) and the cell in cm."""
    folder = ROOT / name
    views = Views.load(folder / f"{VIEWS}.npz")
    geometry = Geometry.load(folder / f"{BACKBONE}_{VIEWS}.npz")
    truth = GroundTruth.load(folder / f"gt_{VIEWS}.npz")
    pts, rays, depth, scale = scene_points(
        geometry, views.images, np.full(geometry.depth.shape, -1), {}, 192
    )
    gt_points, has, (s, rotation, t) = gt_alignment(rays, depth, views, truth)
    gt = ((gt_points - t) @ rotation / s / scale)[has]
    return pts.points.numpy(), pts.log_conf.numpy(), gt, scale * s * 100


def draw(detail: dict, means: dict, path: Path) -> None:
    """Bedroom slab at full and half confidence, and the mean recall and precision per keep."""
    fig, axes = plt.subplots(
        1, 3, figsize=(18, 5.2), gridspec_kw={"width_ratios": [1, 1, 0.9]}
    )
    for ax, keep in zip(axes[:2], (1.0, 0.5), strict=True):
        g, exact_g, near_g, d, near_d, _, _ = detail[keep]
        y0 = int(np.median(d[:, 1]))
        slab_g, slab_d = np.abs(g[:, 1] - y0) < 2, np.abs(d[:, 1] - y0) < 2
        spurious = slab_d & ~near_d
        ax.scatter(
            d[spurious, 0],
            d[spurious, 2],
            s=2,
            c="tab:blue",
            label="Déjà View voxel far from any GT",
        )
        for mask, colour, label in (
            (exact_g, "tab:green", "GT voxel with a point"),
            (~exact_g & near_g, "tab:orange", "GT voxel, point one voxel away"),
            (~near_g, "tab:red", "GT voxel, no point within one voxel"),
        ):
            m = slab_g & mask
            ax.scatter(g[m, 0], g[m, 2], s=2, c=colour, label=label)
        ax.set_aspect("equal")
        ax.set_title(f"bedroom, Déjà View points kept: top {keep:.0%} by confidence")
        ax.set_xlabel("x (voxels of 192)")
        ax.set_ylabel("z (voxels of 192)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=9, markerscale=5)
    x = np.arange(len(KEEPS))
    for k, (key, label, colour) in enumerate(
        [
            ("recall", "recall, exact", "tab:green"),
            ("recall1", "recall, within 1 voxel", "tab:olive"),
            ("precision", "precision, exact", "tab:blue"),
            ("precision1", "precision, within 1 voxel", "tab:cyan"),
        ]
    ):
        axes[2].bar(
            x + (k - 1.5) * 0.2,
            [means[keep][key] for keep in KEEPS],
            0.2,
            color=colour,
            label=label,
        )
    axes[2].set_xticks(x, [f"top {k:.0%}" for k in KEEPS])
    axes[2].set_ylim(0, 1.05)
    axes[2].set_ylabel("mean over 14 dev and val scenes, 192 cells (7 cm)")
    axes[2].set_title("voxel recall and precision by confidence cut")
    axes[2].legend(fontsize=8, loc="lower left")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(path, dpi=110)


def main() -> None:
    """Measure the named splits, write the table and the figure."""
    out, figure, *splits = sys.argv[1:]
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    scenes = load_config(Path("configs/hypersim.yaml")).scenes(splits)
    results: dict = {}
    detail: dict = {}
    for scene, cam in scenes.items():
        name = f"{scene}_{cam}"
        dv, conf, gt, cm = load(name)
        results[name] = {}
        for size in RESOLUTIONS:
            metrics, parts = scene_metrics(dv, conf, gt, size)
            results[name][size] = {"cell_cm": float(parts[1.0][5]) * cm, **metrics}
            if name == BEDROOM and size == 192:
                detail = parts
        print(
            name,
            {s: round(results[name][s][1.0]["recall"], 3) for s in RESOLUTIONS},
            flush=True,
        )
    (out_dir / "coverage.json").write_text(json.dumps(results, indent=1))
    means = {
        keep: {
            m: float(np.mean([results[n][192][keep][m] for n in results]))
            for m in ("recall", "recall1", "precision", "precision1")
        }
        for keep in KEEPS
    }
    draw(detail, means, Path(figure))


if __name__ == "__main__":
    main()
