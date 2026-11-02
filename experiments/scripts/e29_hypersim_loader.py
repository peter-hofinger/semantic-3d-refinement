"""E29: checks of the Hypersim loader (views + ground truth) on the three val scenes.

uv run python experiments/scripts/e29_hypersim_loader.py

Needs `uv run hypersim-download` and `uv run hypersim-prepare` first. The position-map
check also needs one extra file (skipped if missing), fetched with ml-hypersim's
contrib/99991/download.py into data/hypersim/check:
  --scene ai_015_001 --contains scene_cam_00_geometry_hdf5/frame.0040.position.hdf5
"""

from pathlib import Path

import matplotlib as mpl
import numpy as np

mpl.use("Agg")
import matplotlib.pyplot as plt

from src.evaluation.instances import STRUCTURE, contact_sheet, score
from src.loaders import Views
from src.loaders.hypersim.prepare import STUFF_BASE, frame_ground_truth
from src.loaders.hypersim.scene import HypersimScene, read_hdf5
from src.loaders.scannetpp.gt import NO_MESH, GroundTruth
from src.segmentation import Segmentation
from src.segmentation.associate import Associator, gt_geometry
from src.segmentation.base import ID_STRIDE
from src.segmentation.linked import working_associator

RAW = Path("data/hypersim/raw")
META = Path("data/hypersim/metadata")
DATA = Path("data/processed/hypersim")
CHECK = Path("data/hypersim/check")
FIGURES = Path("experiments/figures")
SCENES = ("ai_015_001_cam_00", "ai_047_009_cam_00", "ai_052_001_cam_00")
STEM = "views_n20_uniform_504"
AGREE = 0.01  # relative depth agreement for the cross-view check


def fmt(s: object) -> str:
    """One line per InstanceScores."""
    return "  ".join(
        f"{k} {v:.3f}" if isinstance(v, float) else f"{k} {v}"
        for k, v in vars(s).items()
    )


def per_view_ids(gt: GroundTruth) -> Segmentation:
    """GT masks with fresh ids in every view (what image mode produces)."""
    ids = np.full(gt.instance_ids.shape, -1, np.int32)
    classes = {}
    for i, view in enumerate(gt.instance_ids):
        for k, g in enumerate(np.unique(view[view >= 0])):
            ids[i][view == g] = i * ID_STRIDE + k
            classes[i * ID_STRIDE + k] = gt.classes[int(g)]
    return Segmentation(ids, classes, scores=dict.fromkeys(classes, 1.0))


def position_map_check() -> None:
    """Lift one full-resolution frame and compare with Hypersim's world positions."""
    path = (
        CHECK / "ai_015_001/images/scene_cam_00_geometry_hdf5/frame.0040.position.hdf5"
    )
    if not path.is_file():
        print("position-map check skipped (file missing)")
        return
    s = HypersimScene(RAW, META, "ai_015_001", "cam_00")
    k, (w, h) = s.intrinsics()
    frame = next(f for f in s.frames() if f.frame_id == 40)
    _, depth, _ = frame_ground_truth(s.load_labels(40), k)
    u, v = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
    cam = np.stack([u, v, np.ones_like(u)], -1) @ np.linalg.inv(k).T * depth[..., None]
    world = cam @ frame.cam_to_world[:3, :3].T + frame.cam_to_world[:3, 3]
    ref = read_hdf5(path).astype(np.float64) * s.meters_per_asset_unit
    ok = (depth > 0) & np.isfinite(ref).all(-1)
    err = np.linalg.norm(world - ref, axis=-1)[ok]
    print(
        f"position map ai_015_001 frame 40 (tilt-shift camera): 3D error median "
        f"{np.median(err) * 1000:.2f} mm, p99 {np.percentile(err, 99) * 1000:.2f} mm, "
        f"relative median {np.median(err / depth[ok]):.1e} (float16 step ~5e-4)"
    )


def cross_view_agreement(views: Views, gt: GroundTruth) -> np.ndarray:
    """Share of view i's points that land in view j with z-depth within AGREE.

    Only pairs where >= 10% of the points land inside view j count; occlusion makes
    the share < 1 even with perfect poses, so it is read relative to the median.
    """
    geo = gt_geometry(views, gt)
    points = geo.world_points()
    n = len(views)
    share = np.full((n, n), np.nan)
    for i in range(n):
        p = points[i][gt.depth[i] > 0]
        for j in range(n):
            if i == j:
                continue
            w2c = np.linalg.inv(views.cam_to_world[j])
            c = p @ w2c[:3, :3].T + w2c[:3, 3]
            z = c[:, 2]
            front = z > 0
            uv = c[front] @ views.intrinsics[j].T
            x, y = uv[:, 0] / uv[:, 2], uv[:, 1] / uv[:, 2]
            h, w = gt.depth.shape[1:]
            inside = (x >= 0) & (x < w) & (y >= 0) & (y < h)
            if inside.sum() < 0.1 * len(p):
                continue
            d = gt.depth[j][y[inside].astype(int), x[inside].astype(int)]
            agree = np.abs(d - z[front][inside]) < AGREE * z[front][inside]
            share[i, j] = agree.mean()
    return share


def main() -> None:
    """Run every check and write the GT contact sheets."""
    FIGURES.mkdir(parents=True, exist_ok=True)
    position_map_check()
    for scene in SCENES:
        views = Views.load(DATA / scene / f"{STEM}.npz")
        gt = GroundTruth.load(DATA / scene / f"gt_{STEM}.npz")
        ids = gt.instance_ids
        print(f"== {scene}")
        print(
            f" labeled {(ids >= 0).mean():.3f} (stuff {(ids >= STUFF_BASE).mean():.3f}),"
            f" unlabeled {(ids == -1).mean():.3f}, no geometry {(ids == NO_MESH).mean():.3f},"
            f" {len(np.unique(ids[ids >= 0]))} instances in the 20 views"
        )
        clean = np.where(ids >= 0, ids, -1)
        print(" scorer: GT vs GT (expect all 1, leakage 0)")
        print("   ", fmt(score(gt, Segmentation(clean, gt.classes))))

        local = per_view_ids(gt)
        geo = gt_geometry(views, gt)
        for name, assoc in (
            ("default", Associator()),
            ("working", working_associator()),
        ):
            linked = assoc.associate(local, geo)
            print(
                f" association ({name}): GT masks per view + GT geometry: "
                f"{len(local.classes)} masks -> {len(linked.classes)} objects"
            )
            for subset, exclude in (("all", ()), ("objects", STRUCTURE)):
                print(f"   {subset:8s}", fmt(score(gt, linked, exclude)))

        share = cross_view_agreement(views, gt)
        valid = share[np.isfinite(share)]
        print(
            f" cross-view depth agreement (< {AGREE:.0%}): {len(valid)} overlapping pairs,"
            f" median {np.median(valid):.3f}, 10th pct {np.percentile(valid, 10):.3f},"
            f" min {valid.min():.3f}"
        )
        contact_sheet(
            views,
            Segmentation(clean, gt.classes),
            f"{scene} ground truth",
            FIGURES / f"e29_{scene}_gt.jpg",
        )
    plt.close("all")


if __name__ == "__main__":
    main()
