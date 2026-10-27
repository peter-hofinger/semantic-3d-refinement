"""Accuracy against a scan: z-buffer, lifting conventions, scale alignment, and the ETH3D readers."""

from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree

from semantic_3d_refinement.eth3d import load_scan, quat_to_rotation, read_calibration, read_ply_xyz
from semantic_3d_refinement.evaluate import accuracy, lift, scan_depth, score
from semantic_3d_refinement.views import write_ply

K = np.array([[40.0, 0, 16.5], [0, 41.0, 12.0], [0, 0, 1]])
H, W = 24, 32


def pose() -> np.ndarray:
    c2w = np.eye(4)
    c2w[:3, :3] = quat_to_rotation(np.array([0.9, 0.1, -0.3, 0.2]))
    c2w[:3, 3] = [0.5, -1.0, 2.0]
    return c2w


def wall(z: float, c2w: np.ndarray, step: float = 0.002) -> np.ndarray:
    """Dense points of the fronto-parallel plane at depth z in the camera, in world coordinates."""
    s = np.arange(-1.5, 1.5, step) * z
    x, y = np.meshgrid(s, s)
    cam = np.stack([x.ravel(), y.ravel(), np.full(x.size, z)], 1)
    return cam @ c2w[:3, :3].T + c2w[:3, 3]


def depth_of(points: np.ndarray, c2w: np.ndarray) -> np.ndarray:
    return scan_depth(torch.from_numpy(points), torch.from_numpy(K), torch.from_numpy(c2w), H, W).numpy()


def test_scan_depth_keeps_the_nearest_surface() -> None:
    c2w = pose()
    depth = depth_of(np.concatenate([wall(3.0, c2w), wall(2.0, c2w)]), c2w)
    assert (depth > 0).all()
    np.testing.assert_allclose(depth, 2.0, atol=1e-9)


def test_lift_puts_scan_depth_back_on_the_scan() -> None:
    c2w = pose()
    scan = wall(2.0, c2w)
    points = lift(depth_of(scan, c2w), K, c2w)
    assert cKDTree(scan).query(points.reshape(-1, 3))[0].max() < 0.003


def test_score_aligns_scale_and_ignores_unseen_pixels() -> None:
    c2w = pose()
    scan = wall(2.0, c2w)
    gt = depth_of(scan, c2w)
    gt[:, :5] = 0
    dist, scale = score(3 * gt + 1.0 * (gt == 0), gt, K, c2w, cKDTree(scan))
    assert np.isclose(scale, 1 / 3)
    assert np.isnan(dist[:, :5]).all() and np.isfinite(dist[:, 5:]).all()
    assert np.nanmax(dist) < 0.003


def test_accuracy_counts_points_within_each_tolerance() -> None:
    m = accuracy(np.array([0.005, 0.015, 0.03, 0.2]), [1, 2, 5])
    assert m["acc_1cm"] == 0.25 and m["acc_2cm"] == 0.5 and m["acc_5cm"] == 0.75
    assert np.isclose(m["mean_cm"], 6.25) and np.isclose(m["median_cm"], 2.25)


def test_quaternion_rotates_a_quarter_turn_about_z() -> None:
    r = quat_to_rotation(np.array([np.cos(np.pi / 4), 0, 0, np.sin(np.pi / 4)]))
    np.testing.assert_allclose(r @ [1, 0, 0], [0, 1, 0], atol=1e-12)


def test_readers_return_camera_centers_and_aligned_scans(tmp_path: Path) -> None:
    (tmp_path / "cameras.txt").write_text("# header\n0 PINHOLE 64 48 40 41 32 24\n")
    images = "# header\n1 1 0 0 0 1 2 3 0 dslr_images_undistorted/b.JPG\n\n2 1 0 0 0 0 0 0 0 a.JPG\n\n"
    (tmp_path / "images.txt").write_text(images)
    calib = read_calibration(tmp_path)
    assert [c[0] for c in calib] == ["a.JPG", "b.JPG"]
    np.testing.assert_allclose(calib[1][2][:3, 3], [-1, -2, -3])
    assert calib[1][3] == (64, 48) and calib[1][1][0, 0] == 40

    xyz = np.random.default_rng(0).normal(size=(10, 3)).astype(np.float32)
    write_ply(tmp_path / "scan1.ply", xyz)
    np.testing.assert_array_equal(read_ply_xyz(tmp_path / "scan1.ply"), xyz)
    matrix = "1 0 0 5\n0 1 0 0\n0 0 1 0\n0 0 0 1"
    mlp = f'<MLMesh label="scan1.ply" filename="scan1.ply">\n<MLMatrix44>\n{matrix}\n</MLMatrix44>\n</MLMesh>'
    (tmp_path / "scan_alignment.mlp").write_text(mlp)
    np.testing.assert_allclose(load_scan(tmp_path), xyz + np.array([5, 0, 0]), atol=1e-6)
