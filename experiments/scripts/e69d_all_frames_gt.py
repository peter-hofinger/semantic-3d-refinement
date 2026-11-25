"""E69d: how much of a scene do the 20 input views miss? An all-frames ground truth (issue #69).

Every score so far uses GT20, the ground truth of the 20 views Déjà View sees, so a correct
point on a surface those views miss counts as an accuracy error. Here all public frames of
the trajectory are lifted exactly as `prepare.build` lifts the 20 (504 px, nearest
sampling, no-mesh pixels dropped) into GT99, and both targets are voxelized to one point
per occupied cube of side v (the mean of its points), so overlapping views stop counting a
surface several times. A GT point is *seen* if it is visible in at least one input view:

    seen(x) = OR_v [ inside_v(x) and z_v(x) > 0 and |z_v(x) - D_v(pixel_v(x))| <= max(t, r z_v(x)) ]

with z_v the depth of x in view v, D_v that view's GT depth (0 = no surface, sees nothing),
t an absolute and r a relative tolerance. Déjà View (unrefined, pinhole rays) is aligned once
by Umeyama on its pixels onto GT20, the protocol's alignment, and scored with that one
alignment against every target; its completeness is also split over the seen and the unseen
points. A pixel of a non-input frame is unseen if its lifted point is, which the figure shows.

`uv run python experiments/scripts/e69d_all_frames_gt.py` (downloads nothing; run
`src.loaders.hypersim.download` for the scene first) writes
`experiments/results/e69d_<scene>.txt`, `experiments/figures/e69d_<scene>_unseen.jpg` and the
cache `gt99v1cm_<views>.npz` (GT99 at the first voxel side, world frame) that E69e trains on.
"""

import argparse
import time
from pathlib import Path

import matplotlib as mpl
import numpy as np

mpl.use("Agg")
import matplotlib.pyplot as plt

from src.analysis.filters import voxel_index
from src.evaluation.labeled import (
    align_output,
    ground_truth_cloud,
    lift,
    nearest_distances,
)
from src.geometry import Geometry
from src.loaders.frames import resize_view
from src.loaders.ground_truth import GroundTruth
from src.loaders.hypersim.config import load_config
from src.loaders.hypersim.prepare import frame_ground_truth, sample_nearest
from src.loaders.hypersim.scene import HypersimScene
from src.loaders.views import Views
from src.refinement.multiview import points_from_rays, scene_from_geometry

VIEWS = "views_n20_uniform_504"
BACKBONE = "dejaview_k16"
CM = 100.0
SIDES = (0.01, 0.02)  # voxel side in metres; the first is the headline
TOLERANCES = (
    (0.01, 0.01),
    (0.02, 0.02),
    (0.04, 0.04),
)  # (t in metres, r); 2nd headline
SHOWN = 4  # non-input frames in the figure, those with the largest unseen share


def voxel_centroids(points: np.ndarray, side: float) -> np.ndarray:
    """One point (M, 3) per occupied cube of the given side: the mean of its points."""
    assert points.ndim == 2, points.shape
    assert points.shape[1] == 3, points.shape
    assert len(points), points.shape
    index = voxel_index(points, side)
    count = np.bincount(index)
    out = np.stack(
        [np.bincount(index, weights=points[:, a]) / count for a in range(3)], axis=1
    )
    assert np.isfinite(out).all()
    assert len(out) == index.max() + 1
    return out


def visible(
    points: np.ndarray,
    k: np.ndarray,
    cam_to_world: np.ndarray,
    depth: np.ndarray,
    tolerance: tuple[float, float],
) -> np.ndarray:
    """Whether each point (N, 3) lies on the surface a view's GT depth (H, W) sees (module)."""
    assert points.ndim == 2, points.shape
    assert points.shape[1] == 3, points.shape
    assert depth.ndim == 2, depth.shape
    assert k.shape == (3, 3), k.shape
    assert cam_to_world.shape == (4, 4), cam_to_world.shape
    h, w = depth.shape
    cam = (points - cam_to_world[:3, 3]) @ cam_to_world[:3, :3]
    z = cam[:, 2]
    front = z > 0
    uv = cam[:, :2] / np.where(front, z, 1.0)[:, None] * k[[0, 1], [0, 1]] + k[:2, 2]
    col, row = np.floor(uv[:, 0]).astype(np.int64), np.floor(uv[:, 1]).astype(np.int64)
    inside = front & (col >= 0) & (col < w) & (row >= 0) & (row < h)
    surface = np.zeros(len(points))
    surface[inside] = depth[row[inside], col[inside]]
    seen = inside & (surface > 0)
    return seen & (np.abs(z - surface) <= np.maximum(tolerance[0], tolerance[1] * z))


def seen_by(
    points: np.ndarray, views: Views, gt: GroundTruth, tolerance: tuple[float, float]
) -> np.ndarray:
    """Whether each point (N, 3) is visible in at least one of the views."""
    out = np.zeros(len(points), dtype=bool)
    for k, c2w, depth in zip(
        views.intrinsics, views.cam_to_world, gt.depth, strict=True
    ):
        out |= visible(
            points, k.astype(np.float64), c2w.astype(np.float64), depth, tolerance
        )
    return out


def all_frames(scene: HypersimScene, size: tuple[int, int]) -> list[dict]:
    """Every public frame as `prepare.build` makes a view: name, image, K, pose, z-depth."""
    k, _ = scene.intrinsics()
    out = []
    for frame in scene.frames():
        image, k_scaled = resize_view(scene.load_image(frame.frame_id), k, size)
        _, depth, _ = frame_ground_truth(scene.load_labels(frame.frame_id), k)
        out.append(
            {
                "name": frame.name,
                "image": image,
                "k": k_scaled,
                "cam_to_world": frame.cam_to_world,
                "depth": sample_nearest(depth, size).astype(np.float32),
            }
        )
    return out


def scores(pred: np.ndarray, truth: np.ndarray, seen: np.ndarray | None = None) -> dict:
    """Accuracy, completeness and Chamfer in cm; completeness also over seen / unseen truth."""
    acc = nearest_distances(truth, pred)
    comp = nearest_distances(pred, truth)
    out = {
        "accuracy": CM * acc.mean(),
        "completeness": CM * comp.mean(),
        "chamfer": CM * (acc.mean() + comp.mean()) / 2,
    }
    if seen is not None:
        out["comp_seen"] = CM * comp[seen].mean()
        out["comp_unseen"] = CM * comp[~seen].mean() if (~seen).any() else float("nan")
    return out


def figure(frames: list[dict], unseen: list[np.ndarray], out: Path, title: str) -> None:
    """Draw the non-input frames with the most unseen pixels, those pixels in red."""
    share = [
        u.sum() / max(1, (f["depth"] > 0).sum())
        for f, u in zip(frames, unseen, strict=True)
    ]
    order = np.argsort(share)[::-1][:SHOWN]
    fig, axes = plt.subplots(1, len(order), figsize=(4.2 * len(order), 3.6))
    for ax, i in zip(np.atleast_1d(axes), order, strict=True):
        image = frames[i]["image"].astype(np.float32) / 255
        image[unseen[i]] = 0.35 * image[unseen[i]] + 0.65 * np.array([0.85, 0.1, 0.1])
        ax.imshow(image)
        ax.set_title(f"{frames[i]['name']}: {100 * share[i]:.1f} % unseen", fontsize=10)
        ax.axis("off")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


def main() -> None:
    """Build GT99, classify seen / unseen, score Déjà View against every target."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", default="ai_006_007")
    parser.add_argument("--camera", default="cam_00")
    parser.add_argument("--config", type=Path, default=Path("configs/hypersim.yaml"))
    args = parser.parse_args()
    cfg = load_config(args.config)
    scene = HypersimScene(cfg.data_root, cfg.metadata_root, args.scene, args.camera)
    folder = cfg.output_root / scene.scene_id
    views = Views.load(folder / f"{VIEWS}.npz")
    gt20 = GroundTruth.load(folder / f"gt_{VIEWS}.npz")
    lines = [
        f"E69d, {scene.scene_id}: GT20 = the 20 input views, GT99 = every public frame"
    ]
    t0 = time.time()

    size = gt20.depth.shape[2], gt20.depth.shape[1]
    frames = all_frames(scene, size)
    by_name = {f["name"]: f for f in frames}
    for i, name in enumerate(views.names):
        f = by_name[str(name)]
        assert np.array_equal(f["depth"], gt20.depth[i]), name
        assert np.allclose(f["k"], views.intrinsics[i], rtol=1e-5), name
        assert np.allclose(f["cam_to_world"], views.cam_to_world[i], atol=1e-4), name
    lines.append(
        f"frames: {len(frames)}, of which the 20 input views (depth identical to the cache: yes); "
        f"lifted in {time.time() - t0:.0f} s"
    )

    truth20 = ground_truth_cloud(
        gt20.depth,
        gt20.instance_ids,
        gt20.classes,
        views.intrinsics,
        views.cam_to_world,
    )
    gt99 = np.concatenate(
        [
            lift(f["depth"].astype(np.float64), f["k"], f["cam_to_world"])[
                f["depth"] > 0
            ]
            for f in frames
        ]
    )
    geometry = Geometry.load(folder / f"{BACKBONE}_{VIEWS}.npz")
    dv_scene = scene_from_geometry(geometry)
    points = points_from_rays(dv_scene.depth.double(), dv_scene.rays.double()).numpy()
    aligned, scale, _ = align_output(truth20, points)
    dv = aligned[np.isfinite(aligned).all(1)]
    lines.append(
        f"points: GT20 {len(truth20.points):,}, GT99 {len(gt99):,}, Déjà View {len(dv):,} "
        f"(aligned once onto GT20, scale {scale:.4f})"
    )

    lines.append(
        "\nDéjà View, Chamfer = (accuracy + completeness) / 2, cm, means, uncapped"
    )
    lines.append(f"  vs GT20 pixels (the protocol)   {fmt(scores(dv, truth20.points))}")
    lines.append(f"  vs GT99 pixels                  {fmt(scores(dv, gt99))}")
    for side in SIDES:
        v20, v99 = voxel_centroids(truth20.points, side), voxel_centroids(gt99, side)
        if side == SIDES[0]:
            np.savez_compressed(
                folder / f"gt99v{CM * side:.0f}cm_{VIEWS}.npz", points=v99
            )
        lines.append(
            f"\nvoxel side {CM * side:.0f} cm: GT20v {len(v20):,} points, GT99v {len(v99):,}"
        )
        lines.append(f"  vs GT20v                        {fmt(scores(dv, v20))}")
        for tol in TOLERANCES:
            seen = seen_by(v99, views, gt20, tol)
            lines.append(
                f"  tolerance {CM * tol[0]:.0f} cm / {100 * tol[1]:.0f} %: unseen {100 * (~seen).mean():.1f} % of GT99v; "
                f"vs GT99v {fmt(scores(dv, v99, seen))}"
            )

    inputs = {str(n) for n in views.names}
    others = [f for f in frames if f["name"] not in inputs]
    unseen, n_unseen, n_valid = [], 0, 0
    for f in others:
        valid = f["depth"] > 0
        mask = np.zeros(valid.shape, dtype=bool)
        mask[valid] = ~seen_by(
            lift(f["depth"].astype(np.float64), f["k"], f["cam_to_world"])[valid],
            views,
            gt20,
            TOLERANCES[1],
        )
        unseen.append(mask)
        n_unseen, n_valid = n_unseen + mask.sum(), n_valid + valid.sum()
    lines.append(
        f"\npixels of the {len(others)} non-input frames whose point no input view sees "
        f"(tolerance 2 cm / 2 %): {100 * n_unseen / n_valid:.1f} %"
    )
    lines.append(f"total {time.time() - t0:.0f} s")

    results = Path("experiments/results") / f"e69d_{scene.scene_id}.txt"
    results.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    figure(
        others,
        unseen,
        Path("experiments/figures") / f"e69d_{scene.scene_id}_unseen.jpg",
        f"{scene.scene_id}: red = surface that none of the 20 input views sees (tolerance 2 cm / 2 %)",
    )


def fmt(s: dict) -> str:
    """One line of scores."""
    text = f"acc {s['accuracy']:.3f}  comp {s['completeness']:.3f}  Chamfer {s['chamfer']:.3f}"
    if "comp_seen" in s:
        text += f"  (comp seen {s['comp_seen']:.3f}, unseen {s['comp_unseen']:.3f})"
    return text


if __name__ == "__main__":
    main()
