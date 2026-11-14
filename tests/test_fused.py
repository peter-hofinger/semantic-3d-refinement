"""The opt-in fused-surface score: TSDF fusion of per-view depth maps."""

import numpy as np

from src.analysis.scene_eval import Scoring, score
from src.evaluation.labeled import (
    LabeledCloud,
    camera_depth,
    fused_metrics,
    ground_truth_cloud,
    tsdf_points,
)
from src.segmentation.base import Segmentation

VOXEL, TRUNC = 0.02, 0.08
TSDF = {"voxel": VOXEL, "trunc": TRUNC}


def cameras(views: int = 3, h: int = 24, w: int = 32) -> tuple[np.ndarray, np.ndarray]:
    """Pinholes (pixel centers at integers) along x, looking down +z."""
    k = np.tile(
        [[30.0, 0, (w - 1) / 2], [0, 30.0, (h - 1) / 2], [0, 0, 1]], (views, 1, 1)
    )
    c2w = np.tile(np.eye(4), (views, 1, 1))
    c2w[:, 0, 3] = np.arange(views) * 0.1
    return k, c2w


def plane_points(
    k: np.ndarray, c2w: np.ndarray, z: float, h: int = 24, w: int = 32
) -> np.ndarray:
    """Per-pixel world points (V, H, W, 3) on the plane at depth z."""
    v, u = np.mgrid[0:h, 0:w].astype(np.float64)
    rays = np.stack(
        [(u - k[0, 0, 2]) / k[0, 0, 0], (v - k[0, 1, 2]) / k[0, 1, 1], np.ones_like(u)],
        -1,
    )
    return np.stack([z * rays @ c[:3, :3].T + c[:3, 3] for c in c2w])


def test_a_plane_fuses_onto_itself() -> None:
    k, c2w = cameras()
    depth = np.full((3, 24, 32), 2.0)
    fused = tsdf_points(depth, k, c2w, VOXEL, TRUNC)
    assert len(fused) > 100
    np.testing.assert_allclose(fused[:, 2], 2.0, atol=1e-3)


def test_camera_depth_inverts_the_lift_and_skips_missing_points() -> None:
    k, c2w = cameras()
    pts = plane_points(k, c2w, 2.0)
    pts[0, 0, 0] = np.nan
    z = camera_depth(pts, c2w)
    assert z[0, 0, 0] == 0
    np.testing.assert_allclose(z.reshape(-1)[1:], 2.0)


def test_fused_scores_are_zero_on_the_truth_and_follow_the_alignment() -> None:
    k, c2w = cameras()
    truth = tsdf_points(np.full((3, 24, 32), 2.0), k, c2w, VOXEL, TRUNC)
    same = fused_metrics(plane_points(k, c2w, 2.0), (k, c2w), np.eye(4), truth, TSDF)
    assert same["fused_chamfer_cm"] < 0.1
    # the output in a frame half the size, aligned back by s = 2: the same scores
    half = c2w.copy()
    half[:, :3, 3] /= 2
    align = np.diag([2.0, 2.0, 2.0, 1.0])
    scaled = fused_metrics(plane_points(k, half, 1.0), (k, half), align, truth, TSDF)
    np.testing.assert_allclose(
        scaled["fused_chamfer_cm"], same["fused_chamfer_cm"], atol=0.05
    )
    # 3 cm too far: accuracy and completeness both about 3 cm
    off = fused_metrics(plane_points(k, c2w, 2.03), (k, c2w), np.eye(4), truth, TSDF)
    np.testing.assert_allclose(off["fused_accuracy_median_cm"], 3.0, atol=0.3)


def test_without_fusion_the_scores_have_no_fused_keys() -> None:
    k, c2w = cameras(views=1)
    depth = np.full((1, 24, 32), 2.0, dtype=np.float32)
    ids = np.zeros((1, 24, 32), dtype=np.int32)
    k_colmap = k.copy()
    k_colmap[:, :2, 2] += 0.5
    truth: LabeledCloud = ground_truth_cloud(depth, ids, {0: "floor"}, k_colmap, c2w)
    seg = Segmentation(ids, {0: "floor"}, {})
    out = score(Scoring(truth, seg, ["floor"]), plane_points(k, c2w, 2.0))
    assert out["chamfer_cm"] < 1e-9
    assert not any(key.startswith("fused") for key in out)
