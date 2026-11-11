from pathlib import Path

import numpy as np
import pytest

from src.loaders.ground_truth import NO_MESH, STUFF_BASE
from src.loaders.kitti360.prepare import instance_ids
from src.loaders.kitti360.scene import (
    PrepareConfig,
    centred_crop,
    cropped_intrinsics,
    load_config,
    pick_frames,
    read_intrinsics,
    read_ply,
    read_poses,
    render_depth,
    sample_nearest,
    split_scene,
)
from src.segmentation.base import NO_INSTANCE

ROAD, CAR, UNLABELED = 7, 26, 0
CALIBRATION = (
    "S_rect_00: 1408.000000 376.000000\n"
    "P_rect_00: 552.554261 0.000000 682.049453 0.000000 0.000000 552.554261"
    " 238.769549 0.000000 0.000000 0.000000 1.000000 0.000000\n"
)
K = np.array([[100.0, 0, 8], [0, 100.0, 4], [0, 0, 1]])


def grid_points(depth: float) -> np.ndarray:
    """One point through the centre of each pixel of an 16 x 8 view of K."""
    cols, rows = np.meshgrid(np.arange(16) + 0.5, np.arange(8) + 0.5)
    x = (cols - K[0, 2]) / K[0, 0] * depth
    y = (rows - K[1, 2]) / K[1, 1] * depth
    return np.stack([x, y, np.full_like(x, depth)], axis=-1).reshape(-1, 3)


def test_intrinsics_move_the_principal_point_to_pixel_corners() -> None:
    k, size = read_intrinsics(CALIBRATION)
    assert size == (1408, 376)
    assert k[0, 0] == k[1, 1] == pytest.approx(552.554261)
    assert k[0, 2] == pytest.approx(682.549453)
    assert k[1, 2] == pytest.approx(239.269549)


def test_crop_is_centred_on_the_principal_point_and_patch_sized() -> None:
    k, size = read_intrinsics(CALIBRATION)
    box, new_size = centred_crop(k, size, PrepareConfig())
    assert new_size == (504, 182)
    assert box[2] - box[0] == pytest.approx(756)
    assert box[3] - box[1] == pytest.approx(273)
    assert min(box[0], box[1], 1408 - box[2], 376 - box[3]) >= 0
    k_new = cropped_intrinsics(k, box, new_size)
    assert k_new[0, 2] == pytest.approx(252)
    assert k_new[1, 2] == pytest.approx(91)
    assert k_new[0, 0] == pytest.approx(552.554261 * 2 / 3)
    assert k_new[1, 1] == pytest.approx(552.554261 * 2 / 3)


def test_crop_and_resize_keep_a_ray_on_its_pixel() -> None:
    k, size = read_intrinsics(CALIBRATION)
    box, new_size = centred_crop(k, size, PrepareConfig())
    k_new = cropped_intrinsics(k, box, new_size)
    ray = np.array([0.2, -0.05, 1.0])
    u, v = (k @ ray)[:2]
    u_new, v_new = (k_new @ ray)[:2]
    assert u_new == pytest.approx((u - box[0]) * 2 / 3)
    assert v_new == pytest.approx((v - box[1]) * 2 / 3)


def test_sample_nearest_reads_the_pixel_under_each_centre() -> None:
    array = np.arange(6 * 9).reshape(6, 9)
    out = sample_nearest(array, (1.5, 0.0, 7.5, 6.0), (2, 2))
    np.testing.assert_array_equal(out, array[[1, 4]][:, [3, 6]])


def test_poses_parse_and_reject_a_scaled_rotation() -> None:
    row = "5 " + " ".join(str(v) for v in np.eye(4).ravel())
    poses = read_poses(row)
    np.testing.assert_array_equal(poses[5], np.eye(4))
    scaled = "5 " + " ".join(str(v) for v in np.diag([2.0, 2, 2, 1]).ravel())
    with pytest.raises(AssertionError):
        read_poses(scaled)


def test_ply_round_trip(tmp_path: Path) -> None:
    dtype = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("semantic", "<i4")])
    vertices = np.array([(1, 2, 3, 7), (4, 5, 6, 26)], dtype=dtype)
    header = (
        "ply\nformat binary_little_endian 1.0\nelement vertex 2\nproperty float x\n"
        "property float y\nproperty float z\nproperty int semantic\nend_header\n"
    )
    path = tmp_path / "cloud.ply"
    path.write_bytes(header.encode("ascii") + vertices.tobytes())
    np.testing.assert_array_equal(read_ply(path), vertices)


def test_frames_are_centred_and_a_stride_apart() -> None:
    frames = list(range(100, 150))
    assert pick_frames(frames, 5, 2) == [120, 122, 124, 126, 128]
    assert pick_frames(list(range(9)), 5, 2) == [0, 2, 4, 6, 8]
    with pytest.raises(AssertionError):
        pick_frames(list(range(8)), 5, 2)


def test_scene_name_splits_into_drive_and_frames() -> None:
    drive, first, last = split_scene("0003_0000000274_0000000401")
    assert (drive, first, last) == ("2013_05_28_drive_0003_sync", 274, 401)


def test_a_plane_renders_to_its_depth() -> None:
    points = grid_points(5.0)
    labels = np.full((8, 16), ROAD)
    depth = render_depth(points, np.full(len(points), ROAD), np.eye(4), K, labels)
    np.testing.assert_allclose(depth, 5.0, rtol=1e-6)


def test_rendering_follows_the_camera_pose() -> None:
    points = grid_points(5.0)
    pose = np.eye(4)
    pose[:3, :3] = [[0, 0, 1], [0, 1, 0], [-1, 0, 0]]
    pose[:3, 3] = [3, -2, 7]
    world = points @ pose[:3, :3].T + pose[:3, 3]
    labels = np.full((8, 16), ROAD)
    depth = render_depth(world, np.full(len(world), ROAD), pose, K, labels)
    np.testing.assert_allclose(depth, 5.0, rtol=1e-6)


def test_a_pixel_takes_the_nearest_point_of_its_own_class() -> None:
    points = np.concatenate([grid_points(4.0), grid_points(9.0), grid_points(12.0)])
    classes = np.repeat([CAR, ROAD, ROAD], 16 * 8)
    labels = np.full((8, 16), ROAD)
    labels[:, :8] = CAR
    labels[0, 0] = UNLABELED
    depth = render_depth(points, classes, np.eye(4), K, labels)
    np.testing.assert_allclose(depth[:, 8:], 9.0, rtol=1e-6)
    np.testing.assert_allclose(depth[1:, :8], 4.0, rtol=1e-6)
    assert depth[0, 0] == 0


def test_points_behind_the_camera_and_outside_the_image_are_dropped() -> None:
    points = np.array([[0.0, 0, -5], [10.0, 0, 5], [0.0, 0, 0.1]])
    depth = render_depth(points, np.full(3, ROAD), np.eye(4), K, np.full((8, 16), ROAD))
    assert (depth == 0).all()


def test_instance_ids_name_stuff_void_and_missing_depth() -> None:
    instance = np.array([[26015, 26015, 7000], [44002, 21000, 7000]], dtype=np.uint16)
    depth = np.array([[3.0, 3.0, 5.0], [4.0, 0.0, 5.0]], dtype=np.float32)
    ids, classes = instance_ids(instance, depth)
    expected = [[26015, 26015, STUFF_BASE + 7], [NO_INSTANCE, NO_MESH, STUFF_BASE + 7]]
    np.testing.assert_array_equal(ids, expected)
    assert ids.dtype == np.int32
    assert classes == {26015: "car", STUFF_BASE + 7: "road"}


def test_config_lists_every_window_once() -> None:
    cfg = load_config(Path("configs/kitti360.yaml"))
    assert len(cfg.scenes) == 24
    assert set(cfg.scenes.values()) == {"country", "town"}
    for scene in cfg.scenes:
        split_scene(scene)
