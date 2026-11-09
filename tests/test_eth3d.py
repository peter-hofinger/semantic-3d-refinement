from pathlib import Path

import numpy as np
import pytest

from src.loaders.eth3d.scene import (
    FISHEYE,
    PINHOLE,
    Camera,
    centred_crop,
    distort,
    load_config,
    quaternion_to_rotation,
    read_cameras,
    read_images,
    sample_depth,
)

ZERO = np.zeros(8)


def test_zero_coefficients_are_the_equidistant_fisheye() -> None:
    u, v = np.array([0.3, -1.2, 0.0]), np.array([0.4, 0.5, 2.0])
    ud, vd = distort(u, v, ZERO)
    np.testing.assert_allclose(np.hypot(ud, vd), np.arctan(np.hypot(u, v)), rtol=1e-12)
    np.testing.assert_allclose(np.arctan2(vd, ud), np.arctan2(v, u), atol=1e-12)


def test_centre_stays_fixed_for_any_coefficients() -> None:
    ud, vd = distort(np.zeros(1), np.zeros(1), np.arange(1.0, 9.0))
    assert ud[0] == 0
    assert vd[0] == 0


def test_thin_prism_term_by_hand() -> None:
    coeffs = np.array([0, 0, 0, 0, 0, 0, 0.01, 0.02])
    ud, vd = distort(np.array([0.1]), np.array([0.0]), coeffs)
    theta = np.arctan(0.1)
    assert ud[0] == pytest.approx(theta + 0.01 * theta**2)
    assert vd[0] == pytest.approx(0.02 * theta**2)


def test_quarter_turn_about_z() -> None:
    q = np.array([np.cos(np.pi / 4), 0, 0, np.sin(np.pi / 4)])
    expected = np.array([[0.0, -1, 0], [1, 0, 0], [0, 0, 1]])
    np.testing.assert_allclose(quaternion_to_rotation(q), expected, atol=1e-12)


def test_images_txt_pose_is_inverted_to_camera_to_world() -> None:
    q = np.array([0.9, 0.1, -0.3, 0.2])
    q /= np.linalg.norm(q)
    t = np.array([0.5, -1.0, 2.0])
    text = (
        "# header\n"
        f"2 {' '.join(map(str, q))} {' '.join(map(str, t))} 7 dslr_images/B.JPG\n"
        "1.0 2.0 -1\n"
        "1 1 0 0 0 0 0 0 7 dslr_images/A.JPG\n"
        "\n"
    )
    frames = read_images(text)
    assert [f.name for f in frames] == ["A.JPG", "B.JPG"]
    assert frames[1].camera_id == 7
    world = np.array([0.3, 0.7, -0.2])
    camera = quaternion_to_rotation(q) @ world + t
    np.testing.assert_allclose(frames[1].cam_to_world[:3] @ np.append(camera, 1), world)


def test_cameras_txt() -> None:
    cams = read_cameras("# c\n0 PINHOLE 6220 4141 3430.27 3429.23 3119.2 2057.75\n")
    assert cams[0].width == 6220
    assert cams[0].k[1, 2] == pytest.approx(2057.75)


@pytest.mark.parametrize(
    ("cx", "cy"), [(3119.2, 2057.75), (3110.0, 2070.5), (10.3, 7.9)]
)
def test_crop_is_centred_on_the_principal_point(cx: float, cy: float) -> None:
    camera = Camera(PINHOLE, 6220, 4141, np.array([3000.0, 3000.0, cx, cy]))
    x0, y0, w, h = centred_crop(camera)
    assert x0 >= 0
    assert y0 >= 0
    assert x0 + w <= camera.width
    assert y0 + h <= camera.height
    assert abs(x0 + w / 2 - cx) <= 0.5
    assert abs(y0 + h / 2 - cy) <= 0.5


def plane_depth(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Z-depth of the plane 0.6 x + 0.3 y + z = 2 along the ray (u, v, 1)."""
    return 2.0 / (0.6 * u + 0.3 * v + 1.0)


def test_depth_of_a_tilted_plane_is_found_through_the_distortion() -> None:
    f, width, height = 60.0, 80, 60
    coeffs = np.array([0.2, 0.05, 0.001, -0.002, 0, 0, 0.003, 0.001])
    fisheye = Camera(FISHEYE, width, height, np.r_[f, f, 40.0, 30.0, coeffs])
    pinhole = Camera(PINHOLE, width, height, np.array([f, f, 40.0, 30.0]))
    # distorted depth map by forward splatting a fine grid of rays, nearest pixel
    u, v = np.meshgrid(np.linspace(-0.9, 0.9, 1200), np.linspace(-0.7, 0.7, 900))
    ud, vd = distort(u, v, coeffs)
    cols, rows = np.floor(ud * f + 40).astype(int), np.floor(vd * f + 30).astype(int)
    ok = (cols >= 0) & (cols < width) & (rows >= 0) & (rows < height)
    depth = np.full((height, width), np.inf, dtype=np.float32)
    depth[rows[ok], cols[ok]] = plane_depth(u[ok], v[ok])

    crop = (0, 0, width, height)
    xs, ys = np.meshgrid(np.arange(width) + 0.5, np.arange(height) + 0.5)
    truth = plane_depth((xs - 40) / f, (ys - 30) / f)
    found = sample_depth(depth, fisheye, pinhole, crop, (width, height))
    hit = found > 0
    err = np.abs(found[hit] / truth[hit] - 1)
    assert hit.mean() > 0.9
    # nearest lookup: one distorted pixel moves this steep plane by 1-3%
    assert np.percentile(err, 90) < 0.02
    ignored = Camera(FISHEYE, width, height, np.r_[f, f, 40.0, 30.0, ZERO])
    naive = sample_depth(depth, ignored, pinhole, crop, (width, height))
    both = hit & (naive > 0)
    naive_err = np.abs(naive[both] / truth[both] - 1)
    assert np.percentile(naive_err, 90) > 2 * np.percentile(err, 90)


def test_dotted_depth_takes_the_dot_nearest_the_pixel_centre() -> None:
    fisheye = Camera(FISHEYE, 96, 96, np.r_[1e6, 1e6, 48.0, 48.0, ZERO])
    pinhole = Camera(PINHOLE, 96, 96, np.array([1e6, 1e6, 48.0, 48.0]))
    depth = np.full((96, 96), np.inf, dtype=np.float32)
    # one dot in 16 pixels; a spacing of 4 leaves no ties around the 8 x 8 footprints
    dot_rows, dot_cols = np.arange(1, 96, 4), np.arange(2, 96, 4)
    depth[np.ix_(dot_rows, dot_cols)] = 1 + dot_rows[:, None] + 1000 * dot_cols[None]
    out = sample_depth(depth, fisheye, pinhole, (0, 0, 96, 96), (12, 12))
    centre = np.arange(12) * 8 + 4.0  # footprint centres in source pixels

    def nearest(dots: np.ndarray) -> np.ndarray:
        return dots[np.abs(dots[:, None] + 0.5 - centre[None]).argmin(0)]

    expected = 1 + nearest(dot_rows)[:, None] + 1000 * nearest(dot_cols)[None]
    np.testing.assert_array_equal(out, expected)


def test_pixels_without_laser_depth_are_zero() -> None:
    fisheye = Camera(FISHEYE, 4, 4, np.r_[4.0, 4.0, 2.0, 2.0, ZERO])
    pinhole = Camera(PINHOLE, 4, 4, np.array([4.0, 4.0, 2.0, 2.0]))
    depth = np.full((4, 4), np.inf, dtype=np.float32)
    out = sample_depth(depth, fisheye, pinhole, (0, 0, 4, 4), (4, 4))
    assert (out == 0).all()


def test_repo_config_lists_the_13_training_scenes() -> None:
    cfg = load_config(Path("configs/eth3d.yaml"))
    assert len(cfg.scenes) == 13
    assert sum(k == "indoor" for k in cfg.scenes.values()) == 7
    assert cfg.archive("pipes", "dslr_depth").name == "pipes_dslr_depth.7z"


def test_config_rejects_an_unknown_scene_kind(tmp_path: Path) -> None:
    path = tmp_path / "e.yaml"
    path.write_text(
        "base_url: x\narchive_root: a\noutput_root: o\nscenes: {pipes: cave}\n"
    )
    with pytest.raises(ValueError, match="indoor or outdoor"):
        load_config(path)
