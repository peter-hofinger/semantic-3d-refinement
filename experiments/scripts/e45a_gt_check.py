"""E45a KITTI-360 loader: check the ground truth before any score is computed on it.

Per prepared scene (`kitti360-prepare`):
1. Coverage: share of pixels with GT depth, and its depth range.
2. Edges: share of GT pixels within EDGE_PX of a GT depth edge, by E35's step rule
   and by the plane rule of src/analysis/edges.py.
3. Multi-view consistency (poses, intrinsics and the projection together): lift view
   i's GT, project it into view j; where it lands on a GT pixel of the same instance,
   the relative depth difference: its median and the share within CONSISTENT.
4. Classes: share of the GT pixels per class, and the share of each class's pixels
   within EDGE_PX of a plane-rule edge, over the listed scenes.
Figure: FIGURE_VIEWS views of each of --figure's scenes: image, GT depth, GT class,
and the GT depth edges of both rules over the image. KITTI-360 data: CC BY-NC-SA 3.0
(www.cvlibs.net/datasets/kitti-360), Liao et al., TPAMI 2022.

uv run python experiments/scripts/e45a_gt_check.py > experiments/results/e45a_gt_check.txt
"""

import argparse
import itertools
from collections import Counter
from collections.abc import Callable
from pathlib import Path

import matplotlib as mpl
import numpy as np

mpl.use("Agg")
import matplotlib.pyplot as plt
from e35_error_split import depth_edges as step_edges
from scipy import ndimage

from src.analysis.edges import depth_edges
from src.loaders.ground_truth import GroundTruth
from src.loaders.kitti360.prepare import output_path
from src.loaders.kitti360.scene import CLASSES, load_config
from src.loaders.views import Views

EDGE_PX = 3  # px from a GT depth edge that count as "at the edge"
CONSISTENT = 0.05  # relative depth difference that counts as the same surface
FIGURE_VIEWS = (0, 13)  # indices of the views shown per figure scene
OUT = Path("experiments/figures/e45a_gt_check.jpg")


def consistency(views: Views, gt: GroundTruth) -> np.ndarray:
    """Relative depth differences of view i's GT seen in view j, same instance."""
    n, h, w = gt.depth.shape
    k = views.intrinsics.astype(np.float64)
    pose = views.cam_to_world.astype(np.float64)
    cols, rows = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
    out = []
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
        ids = gt.instance_ids[i][valid][inside]
        seen = (gt.depth[j][r, c] > 0) & (gt.instance_ids[j][r, c] == ids)
        out.append(np.abs(gt.depth[j][r, c][seen] - zj[seen]) / zj[seen])
    return np.concatenate(out)


def near_edge(depth: np.ndarray, valid: np.ndarray, rule: Callable) -> np.ndarray:
    """Mark the valid pixels (V, H, W) within EDGE_PX of an edge of their view."""
    near = np.zeros(depth.shape, dtype=bool)
    for i, (d, v) in enumerate(zip(depth, valid, strict=True)):
        edge = rule(d, v)
        if edge.any():
            near[i] = v & (ndimage.distance_transform_edt(~edge) <= EDGE_PX)
    return near


def class_image(gt: GroundTruth, view: int) -> np.ndarray:
    """Class id (H, W) of a view's GT pixels, -1 without a class."""
    index = {name: c for c, name in CLASSES.items()}
    out = np.full(gt.instance_ids[view].shape, -1)
    for i in np.unique(gt.instance_ids[view]):
        if i >= 0:
            out[gt.instance_ids[view] == i] = index[gt.classes[int(i)]]
    return out


def figure(shown: list[tuple[str, Views, GroundTruth]]) -> None:
    """Image, GT depth, GT class and both rules' edges for FIGURE_VIEWS of each scene."""
    rows = [(s, v, g, i) for s, v, g in shown for i in FIGURE_VIEWS]
    fig, axes = plt.subplots(len(rows), 5, figsize=(22, 1.75 * len(rows)))
    far = max(np.percentile(g.depth[g.depth > 0], 99) for _, _, g in shown)
    for ax, (scene, views, gt, i) in zip(axes, rows, strict=True):
        depth, valid = gt.depth[i], gt.depth[i] > 0
        ax[0].imshow(views.images[i])
        ax[0].set_ylabel(f"{scene[:4]} {views.names[i][-4:]}", fontsize=8)
        shade = ax[1].imshow(
            np.where(valid, depth, np.nan),
            cmap="turbo",
            norm=mpl.colors.LogNorm(3, far),
        )
        classes = class_image(gt, i)
        ax[2].imshow(
            np.where(classes >= 0, classes, np.nan),
            cmap="tab20",
            vmin=0,
            vmax=40,
            interpolation="nearest",
        )
        for a, rule in ((ax[3], step_edges), (ax[4], depth_edges)):
            overlay = views.images[i].copy()
            overlay[rule(depth, valid)] = (255, 0, 0)
            a.imshow(overlay)
        for a in ax:
            a.set_xticks([])
            a.set_yticks([])
    titles = (
        "image (cropped about the principal point, 504 x 182)",
        "GT depth (m, log scale; white = none)",
        "GT class (one colour per class)",
        "GT depth edges, step rule (E35)",
        "GT depth edges, plane rule",
    )
    for a, title in zip(axes[0], titles, strict=True):
        a.set_title(title, fontsize=9)
    fig.colorbar(shade, ax=axes[:, 1], fraction=0.03, pad=0.01, label="m")
    fig.suptitle(
        "E45a: KITTI-360 ground truth from the accumulated LiDAR. Look at: the plane"
        " rule's edges should follow object outlines and leave the far road free.",
        fontsize=10,
    )
    fig.savefig(OUT, dpi=110, bbox_inches="tight", pil_kwargs={"quality": 85})


def main() -> None:
    """Print the checks for every prepared scene and draw the figure."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=Path, default=Path("configs/kitti360.yaml"))
    parser.add_argument(
        "--figure",
        nargs="*",
        default=["0003_0000000274_0000000401", "0010_0000000353_0000000557"],
    )
    args = parser.parse_args()
    cfg = load_config(args.cfg)
    pixels: Counter = Counter()
    at_edge: Counter = Counter()
    shown = []
    print("scene (group): GT px | depth m 1%-99% | px <= 3 px of an edge, step rule,")
    print("  plane rule | view i in j: median relative difference, share within 5%")
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
        low, high = np.percentile(gt.depth[valid], [1, 99])
        print(
            f"{scene} ({group}): {valid.mean():.3f} | {low:.1f}-{high:.1f} |"
            f" {near[0].sum() / valid.sum():.3f}, {near[1].sum() / valid.sum():.3f} |"
            f" {np.median(diff):.4f},"
            f" {(diff < CONSISTENT).mean():.3f}"
        )
        for mask, counter in ((valid, pixels), (near[1], at_edge)):
            ids, counts = np.unique(gt.instance_ids[mask], return_counts=True)
            for i, n in zip(ids.tolist(), counts.tolist(), strict=True):
                counter[gt.classes.get(i, "unlabeled")] += n
        if scene in args.figure:
            shown.append((scene, views, gt))
    total = sum(pixels.values())
    print(
        "\nclass: share of the GT pixels | share of its pixels within 3 px of an edge"
    )
    print("  (plane rule), over the prepared scenes")
    for name, n in pixels.most_common():
        print(f"  {name}: {n / total:.4f} | {at_edge[name] / n:.3f}")
    if shown:
        figure(shown)
        print(f"\nfigure: {OUT}")


if __name__ == "__main__":
    main()
