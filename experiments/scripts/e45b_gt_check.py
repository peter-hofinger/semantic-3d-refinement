"""E45b Tanks and Temples loader: check the ground truth before any score is computed.

Per prepared scene (`tanksandtemples-prepare`):
1. Coverage: share of pixels with GT depth, and its depth range.
2. Edges: share of GT pixels within EDGE_PX of a GT depth edge, by E35's step rule
   and by the plane rule of src/analysis/edges.py (Hypersim: 13.5%, KITTI-360: 44%).
3. Multi-view consistency (poses, the assumed focal length and the rendering
   together): view i's GT lifted and projected into view j; where it lands on a GT
   pixel, the relative depth difference: the share within CONSISTENT (the same
   surface; the rest is hidden from j, the camera circles the object) and the median
   difference among those.
4. See-through: a GT pixel of view j is contradicted by view i when a GT point of i
   lands on it more than CONSISTENT in front of its depth: a surface i saw should
   have hidden what j's pixel shows (a hole in the scan seen from j, or a part
   thinner than a pixel). Share of GT pixels contradicted by >= CONTRADICTING views.
Figure: FIGURE_VIEWS of each scene: image, GT depth, plane-rule edges over the image,
and the number of contradicting views. Tanks and Temples: Knapitsch et al., SIGGRAPH
2017 (www.tanksandtemples.org).

uv run python experiments/scripts/e45b_gt_check.py > experiments/results/e45b_gt_check.txt
"""

import argparse
import itertools
from pathlib import Path

import matplotlib as mpl
import numpy as np

mpl.use("Agg")
import matplotlib.pyplot as plt
from e35_error_split import depth_edges as step_edges
from e45a_gt_check import CONSISTENT, consistency, near_edge

from src.analysis.edges import depth_edges
from src.loaders.ground_truth import GroundTruth
from src.loaders.tanksandtemples.prepare import output_path
from src.loaders.tanksandtemples.scene import load_config
from src.loaders.views import Views

CONTRADICTING = 2  # views that must contradict a pixel for it to count as see-through
FIGURE_VIEWS = (0, 5, 10, 15)  # indices of the views shown per scene
OUT = Path("experiments/figures/e45b_gt_check.jpg")


def contradictions(views: Views, gt: GroundTruth) -> np.ndarray:
    """Per GT pixel (V, H, W), the number of other views with a GT point in front."""
    n, h, w = gt.depth.shape
    k = views.intrinsics.astype(np.float64)
    pose = views.cam_to_world.astype(np.float64)
    cols, rows = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
    count = np.zeros((n, h, w), dtype=np.int32)
    for i, j in itertools.permutations(range(n), 2):
        valid = gt.depth[i] > 0
        z = gt.depth[i][valid].astype(np.float64)
        x = (cols[valid] - k[i, 0, 2]) / k[i, 0, 0] * z
        y = (rows[valid] - k[i, 1, 2]) / k[i, 1, 1] * z
        world = np.stack([x, y, z], axis=1) @ pose[i, :3, :3].T + pose[i, :3, 3]
        cam = (world - pose[j, :3, 3]) @ pose[j, :3, :3]
        front = cam[:, 2] > 0
        zj = np.where(front, cam[:, 2], 1.0)
        c = np.floor(k[j, 0, 0] * cam[:, 0] / zj + k[j, 0, 2]).astype(int)
        r = np.floor(k[j, 1, 1] * cam[:, 1] / zj + k[j, 1, 2]).astype(int)
        inside = front & (c >= 0) & (c < w) & (r >= 0) & (r < h)
        c, r, zj = c[inside], r[inside], zj[inside]
        hit = np.zeros((h, w), dtype=bool)
        ahead = (gt.depth[j][r, c] > 0) & (zj < (1 - CONSISTENT) * gt.depth[j][r, c])
        hit[r[ahead], c[ahead]] = True
        count[j] += hit
    return count


def figure(shown: list[tuple[str, Views, GroundTruth, np.ndarray]]) -> None:
    """Image, GT depth, plane-rule edges and contradicting views per shown view."""
    rows = [(s, v, g, c, i) for s, v, g, c in shown for i in FIGURE_VIEWS]
    fig, axes = plt.subplots(len(rows), 4, figsize=(17, 2.45 * len(rows)))
    for ax, (scene, views, gt, count, i) in zip(axes, rows, strict=True):
        depth, valid = gt.depth[i], gt.depth[i] > 0
        ax[0].imshow(views.images[i])
        ax[0].set_ylabel(f"{scene} {views.names[i]}", fontsize=8)
        low, high = np.percentile(gt.depth[gt.depth > 0], [1, 99])
        shade = ax[1].imshow(
            np.where(valid, depth, np.nan),
            cmap="turbo",
            norm=mpl.colors.LogNorm(low, high),
        )
        fig.colorbar(shade, ax=ax[1], fraction=0.03, pad=0.01, label="m")
        overlay = views.images[i].copy()
        overlay[depth_edges(depth, valid)] = (255, 0, 0)
        ax[2].imshow(overlay)
        votes = ax[3].imshow(
            np.where(valid, count[i], np.nan), cmap="magma", vmin=0, vmax=6
        )
        fig.colorbar(votes, ax=ax[3], fraction=0.03, pad=0.01, label="views")
        for a in ax:
            a.set_xticks([])
            a.set_yticks([])
    titles = (
        "image (504 x 280)",
        "GT depth from the laser scan (log scale; white = none)",
        "GT depth edges, plane rule",
        "views whose GT lies > 5% in front of this pixel's",
    )
    for a, title in zip(axes[0], titles, strict=True):
        a.set_title(title, fontsize=9)
    fig.suptitle(
        "E45b: Tanks and Temples ground truth rendered from the laser scans. Look at:"
        " depth and edges should sit on the image's outlines; bright pixels on the"
        " right are where the scan is seen through.",
        fontsize=10,
        y=0.893,
        va="bottom",
    )
    fig.savefig(OUT, dpi=100, bbox_inches="tight", pil_kwargs={"quality": 85})


def main() -> None:
    """Print the checks for every prepared scene and draw the figure."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cfg", type=Path, default=Path("configs/tanksandtemples.yaml")
    )
    args = parser.parse_args()
    cfg = load_config(args.cfg)
    shown = []
    print("scene (group): GT px | depth m 1%-99% | px <= 3 px of an edge, step rule,")
    print(
        "  plane rule | view i in j: share within 5% (the rest is hidden in j), their"
    )
    print("  median relative difference |")
    print(f"  GT px contradicted by >= 1, >= {CONTRADICTING} views")
    for scene, group in cfg.scenes.items():
        path = output_path(cfg, scene)
        if not path.exists():
            print(f"{scene} ({group}): not prepared")
            continue
        views = Views.load(path)
        gt = GroundTruth.load(path.with_name(f"gt_{path.stem}.npz"))
        valid = gt.depth > 0
        near = [near_edge(gt.depth, valid, rule) for rule in (step_edges, depth_edges)]
        diff = consistency(views, gt)
        count = contradictions(views, gt)
        low, high = np.percentile(gt.depth[valid], [1, 99])
        print(
            f"{scene} ({group}): {valid.mean():.3f} | {low:.2f}-{high:.2f} |"
            f" {near[0].sum() / valid.sum():.3f}, {near[1].sum() / valid.sum():.3f} |"
            f" {(diff < CONSISTENT).mean():.3f}, {np.median(diff[diff < CONSISTENT]):.4f} |"
            f" {(count[valid] >= 1).mean():.3f},"
            f" {(count[valid] >= CONTRADICTING).mean():.3f}"
        )
        shown.append((scene, views, gt, count))
    if shown:
        figure(shown)
        print(f"\nfigure: {OUT}")


if __name__ == "__main__":
    main()
