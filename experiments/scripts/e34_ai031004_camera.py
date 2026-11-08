"""E34 loader fix: check that ai_031_004's GT is multi-view consistent after unscaling.

Hypersim stores cam_00 keyframe 0 of ai_031_004 as 10 R. The loader now divides by
cbrt(det) (`unscaled_rotation`). This rebuilds the 20 prepared views with the fixed loader
and, for every ordered view pair (i, j), lifts view i's GT z-depth to world points,
projects them into view j and compares their depth with view j's GT depth there:
median |z_proj - z_j| / z_j over pixels that land inside j with GT depth and pass a
visibility test (z_proj within 5% of z_j; occluded points fail it and are left out).
The share of pixels that pass is the overlap. View 0 against the rest is printed apart
from the other pairs; the camera stored before the fix is checked the same way, read
from the prepared views on disk (prepared 05.11 with the old loader).

uv run python experiments/scripts/e34_ai031004_camera.py \
    data/processed/hypersim/ai_031_004_cam_00/views_n20_uniform_504.npz \
    > experiments/results/e34_ai031004_camera.txt
"""

from pathlib import Path

import matplotlib as mpl
import numpy as np

mpl.use("Agg")
import matplotlib.pyplot as plt

from src.loaders.hypersim.config import load_config
from src.loaders.hypersim.prepare import build
from src.loaders.hypersim.scene import HypersimScene, read_hdf5

VISIBLE = 0.05  # relative depth gap above which a point counts as occluded in view j
FIGURE = Path("experiments/figures/e34_ai031004_camera.jpg")
FIRST, REST, INK = "#eb6834", "#9a9894", "#0b0b0b"


def pair_error(
    depth_i: np.ndarray,
    k_i: np.ndarray,
    pose_i: np.ndarray,
    depth_j: np.ndarray,
    k_j: np.ndarray,
    pose_j: np.ndarray,
) -> tuple[float, float]:
    """Median relative depth error and overlap of view i's GT seen from view j."""
    h, w = depth_i.shape
    u, v = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
    valid = np.isfinite(depth_i) & (depth_i > 0)
    rays = (
        np.stack([u[valid], v[valid], np.ones(valid.sum())], -1) @ np.linalg.inv(k_i).T
    )
    world = (rays * depth_i[valid, None]) @ pose_i[:3, :3].T + pose_i[:3, 3]
    cam = (world - pose_j[:3, 3]) @ pose_j[:3, :3]
    front = cam[:, 2] > 0
    pix = cam[front] @ k_j.T
    pix = pix[:, :2] / pix[:, 2:]
    z = cam[front, 2]
    x, y = np.floor(pix[:, 0]).astype(int), np.floor(pix[:, 1]).astype(int)
    inside = (x >= 0) & (x < w) & (y >= 0) & (y < h)
    z_j = depth_j[y[inside], x[inside]]
    has_gt = np.isfinite(z_j) & (z_j > 0)
    rel = np.abs(z[inside][has_gt] - z_j[has_gt]) / z_j[has_gt]
    seen = rel < VISIBLE
    if seen.sum() < 100:
        return float("nan"), 0.0
    return float(np.median(rel[seen])), float(seen.sum() / valid.sum())


def report(name: str, depth: np.ndarray, k: np.ndarray, poses: np.ndarray) -> None:
    """Print view 0 against the others and all other pairs."""
    n = len(depth)
    errors = np.full((n, n), np.nan)
    overlap = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                errors[i, j], overlap[i, j] = pair_error(
                    depth[i], k[i], poses[i], depth[j], k[j], poses[j]
                )
    rest = ~np.eye(n, dtype=bool)
    rest[0, :] = rest[:, 0] = False
    first = np.concatenate([errors[0, 1:], errors[1:, 0]])
    first_overlap = np.concatenate([overlap[0, 1:], overlap[1:, 0]])
    print(f"{name}:")
    for label, err, ovl in (
        ("view 0 <-> others", first, first_overlap),
        ("other pairs      ", errors[rest], overlap[rest]),
    ):
        hit = np.isfinite(err)
        median = f"{100 * np.median(err[hit]):.3f}%" if hit.any() else "-"
        print(
            f"  {label}: {hit.sum()} / {len(err)} pairs overlap,"
            f" median rel. depth error {median}, mean overlap {100 * ovl.mean():.1f}%"
        )
    pairs = ", ".join(
        f"{j}: {100 * errors[0, j]:.2f}/{100 * overlap[0, j]:.0f}"
        for j in range(1, n)
        if overlap[0, j] > 0
    )
    print(f"  view 0 -> view j (error % / overlap %): {pairs or '-'}")


def world_points(depth: np.ndarray, k: np.ndarray, pose: np.ndarray) -> np.ndarray:
    """GT z-depth of one view lifted to world points, every 4th pixel."""
    h, w = depth.shape
    u, v = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
    keep = np.isfinite(depth) & (depth > 0) & (u % 4 < 1) & (v % 4 < 1)
    rays = np.stack([u[keep], v[keep], np.ones(keep.sum())], -1) @ np.linalg.inv(k).T
    return (rays * depth[keep, None]) @ pose[:3, :3].T + pose[:3, 3]


def figure(
    depth: np.ndarray, k: np.ndarray, before: np.ndarray, after: np.ndarray
) -> None:
    """Top view (world x, y) of view 0's GT points over the other views', both cameras."""
    rest = np.concatenate(
        [world_points(depth[i], k[i], after[i]) for i in range(1, len(depth))]
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.6))
    for ax, poses, title in (
        (axes[0], before, "before: stored rotation of view 0 (10 R)"),
        (axes[1], after, "after: rotation divided by cbrt(det) = 10"),
    ):
        first = world_points(depth[0], k[0], poses[0])
        ax.scatter(rest[:, 0], rest[:, 1], s=0.2, c=REST, lw=0, label="views 1-19")
        ax.scatter(first[:, 0], first[:, 1], s=0.4, c=FIRST, lw=0, label="view 0")
        ax.plot(*poses[0, :2, 3], "k^", ms=7, label="camera 0")
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=11, color=INK)
        ax.set_xlabel("world x (m)")
        ax.set_ylabel("world y (m)")
        legend = ax.legend(loc="upper right", fontsize=8)
        for handle in legend.legend_handles[:2]:
            handle.set_sizes([12])
    fig.suptitle(
        "ai_031_004 ground truth from above (GT depth, every 4th pixel). With the stored "
        "camera, view 0's points land\nten times too far out and miss the scene; "
        "after the fix they fall onto the other views' surfaces.",
        fontsize=10,
        color=INK,
    )
    fig.tight_layout()
    fig.savefig(FIGURE, dpi=150, pil_kwargs={"quality": 90})


def main() -> None:
    """Rebuild the views with the fixed loader and compare with the old camera."""
    cfg = load_config(Path("configs/hypersim.yaml"))
    scene = HypersimScene(cfg.data_root, cfg.metadata_root, "ai_031_004", "cam_00")
    views, gt = build(scene, cfg.prepare)
    raw = read_hdf5(scene.detail_dir / "cam_00" / "camera_keyframe_orientations.hdf5")
    scale = np.cbrt(np.linalg.det(raw.astype(np.float64)))
    print(
        f"view 0 = {views.names[0]}; cbrt(det) of the stored rotations > 1 + 1e-5: "
        f"keyframes {np.nonzero(scale > 1 + 1e-5)[0].tolist()}, value {scale.max():.6f}"
    )
    after = views.cam_to_world.astype(np.float64)
    before = after.copy()
    before[0, :3, :3] *= scale[int(views.names[0].rsplit(".", 1)[1])]
    k = views.intrinsics.astype(np.float64)
    report("before (stored rotation)", gt.depth, k, before)
    report("after (unscaled rotation)", gt.depth, k, after)
    figure(gt.depth, k, before, after)


if __name__ == "__main__":
    main()
