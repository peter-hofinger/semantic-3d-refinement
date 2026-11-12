import numpy as np
import pytest

from src.loaders.tanksandtemples.scene import (
    SUPPORT,
    PrepareConfig,
    centred_view,
    read_log,
    render_depth,
    to_scan_frame,
    undistort,
)

K = np.array([[100.0, 0, 8], [0, 100.0, 4], [0, 0, 1]])
SIZE = (16, 8)
OFFSETS = ((0.0, 0.0), (0.3, -0.2), (-0.25, 0.35))  # px from the pixel centre


def rotation(axis: int, angle: float) -> np.ndarray:
    """Rotation by `angle` about a coordinate axis."""
    c, s = np.cos(angle), np.sin(angle)
    i, j = [a for a in range(3) if a != axis]
    out = np.eye(3)
    out[i, i], out[i, j], out[j, i], out[j, j] = c, -s, s, c
    return out


def plane_points(depth_at: callable, columns: slice = slice(None)) -> np.ndarray:
    """SUPPORT points per pixel of the 16 x 8 view of K on the surface z(col, row)."""
    out = []
    for dx, dy in OFFSETS[:SUPPORT]:
        cols, rows = np.meshgrid(
            np.arange(16)[columns] + 0.5 + dx, np.arange(8) + 0.5 + dy
        )
        z = depth_at(cols, rows)
        x, y = (cols - K[0, 2]) / K[0, 0] * z, (rows - K[1, 2]) / K[1, 1] * z
        out.append(np.stack([x, y, z], axis=-1).reshape(-1, 3))
    return np.concatenate(out)


def flat(depth: float) -> callable:
    """A fronto-parallel plane at `depth`."""
    return lambda cols, _rows: np.full(cols.shape, depth)


def test_read_log_returns_the_poses_in_file_order() -> None:
    poses = np.tile(np.eye(4), (2, 1, 1))
    poses[1, :3, 3] = [1.5, -2, 3]
    text = "".join(
        f"{i} {i} 0\n" + "".join(" ".join(f"{v:.12f}" for v in row) + "\n" for row in p)
        for i, p in enumerate(poses)
    )
    assert np.array_equal(read_log(text), poses)


def test_read_log_rejects_a_block_out_of_order() -> None:
    block = "1 1 0\n" + "1 0 0 0\n0 1 0 0\n0 0 1 0\n0 0 0 1\n"
    with pytest.raises(AssertionError):
        read_log(block)


def test_scan_frame_poses_are_rigid_and_map_camera_points_like_the_similarity() -> None:
    scale = 1.3
    alignment = np.eye(4)
    alignment[:3, :3] = scale * rotation(2, 0.7) @ rotation(0, -0.4)
    alignment[:3, 3] = [2.0, -1.0, 5.0]
    pose = np.eye(4)
    pose[:3, :3] = rotation(1, 0.3)
    pose[:3, 3] = [0.5, 0.2, -3.0]
    out = to_scan_frame(pose[None], alignment)[0]
    assert np.allclose(out[:3, :3] @ out[:3, :3].T, np.eye(3))
    point = np.array([0.4, -0.7, 2.5, 1.0])  # in the camera, COLMAP units
    metres = np.append(scale * point[:3], 1.0)
    assert np.allclose(alignment @ pose @ point, out @ metres)


def test_scan_frame_rejects_an_alignment_that_is_not_a_similarity() -> None:
    alignment = np.diag([1.0, 2.0, 1.0, 1.0])
    with pytest.raises(AssertionError, match="similarity"):
        to_scan_frame(np.eye(4)[None], alignment)


def test_centred_view_keeps_one_scale_and_a_centred_principal_point() -> None:
    box, size, k = centred_view((1920, 1080), PrepareConfig(focal=0.6))
    assert size == (504, 280)
    assert box == pytest.approx((0, 540 - 280 / 504 * 960, 1920, 540 + 280 / 504 * 960))
    assert (box[2] - box[0]) / size[0] == pytest.approx((box[3] - box[1]) / size[1])
    assert np.allclose(k, [[302.4, 0, 252], [0, 302.4, 140], [0, 0, 1]])


def test_undistort_without_distortion_returns_the_image() -> None:
    image = np.random.default_rng(0).integers(0, 255, (6, 8, 3), dtype=np.uint8)
    assert np.array_equal(undistort(image, 0.6, 0.0), image)


def test_undistort_reads_each_pixel_where_the_lens_put_it() -> None:
    width, height, focal, k1 = 200, 100, 0.6, -0.05
    ramp = np.broadcast_to(
        np.arange(width, dtype=np.uint8)[None, :, None], (height, width, 3)
    )
    out = undistort(np.ascontiguousarray(ramp), focal, k1)
    ys, xs = np.mgrid[0:height, 0:width] + 0.5
    x, y = (xs - width / 2) / (focal * width), (ys - height / 2) / (focal * width)
    seen = focal * width * x * (1 + k1 * (x * x + y * y)) + width / 2 - 0.5
    assert np.abs(out[..., 0] - seen).max() <= 0.5 + 1e-4  # rounding to uint8
    assert out[50, 100, 0] == 100  # the centre stays
    assert out[50, 199, 0] < 199  # the border reads from further in: barrel


def test_render_returns_a_plane_at_its_depth() -> None:
    depth = render_depth(plane_points(flat(2.0)), np.eye(4), K, SIZE)
    assert depth.shape == (8, 16)
    assert depth.dtype == np.float32
    assert np.allclose(depth, 2.0)


def test_render_takes_the_point_nearest_the_pixel_centre_on_a_slanted_plane() -> None:
    def slanted(cols: np.ndarray, _rows: np.ndarray) -> np.ndarray:
        return 1 / (0.5 - 0.001 * cols)  # inverse depth linear in the pixel: a plane

    depth = render_depth(plane_points(slanted), np.eye(4), K, SIZE)
    centres = np.broadcast_to(slanted(np.arange(16) + 0.5, None), (8, 16))
    assert np.allclose(depth, centres, rtol=1e-6)


def test_render_ignores_points_no_surface_supports() -> None:
    stray = np.array([[0.0, 0.0, 1.0], [0.001, 0.0, 1.2]])  # in pixel (4, 8), alone
    points = np.concatenate([plane_points(flat(2.0)), stray])
    assert np.allclose(render_depth(points, np.eye(4), K, SIZE), 2.0)


def test_render_shows_the_back_surface_only_where_the_front_has_none() -> None:
    points = np.concatenate(
        [plane_points(flat(4.0)), plane_points(flat(2.0), slice(0, 8))]
    )
    depth = render_depth(points, np.eye(4), K, SIZE)
    assert np.allclose(depth[:, :8], 2.0)
    assert np.allclose(depth[:, 8:], 4.0)


def test_render_leaves_pixels_with_too_few_points_empty() -> None:
    points = plane_points(flat(2.0))
    points = points[: len(points) // SUPPORT * (SUPPORT - 1)]  # SUPPORT - 1 per pixel
    assert (render_depth(points, np.eye(4), K, SIZE) == 0).all()


def test_render_measures_depth_from_the_camera_of_the_pose() -> None:
    forward = np.eye(4)
    forward[2, 3] = 0.5  # the camera stands 0.5 m nearer the plane
    depth = render_depth(plane_points(flat(2.0)), forward, K, SIZE)
    assert np.allclose(depth[depth > 0], 1.5)
    assert (depth > 0).any()  # the nearer plane spreads: not every pixel keeps 3 points
