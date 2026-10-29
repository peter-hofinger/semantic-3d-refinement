import numpy as np
import pytest

from src.loaders.frames import (
    resize_view,
    select_frames,
    target_size,
)
from src.loaders.scannetpp.assets import remote_path


def test_uniform_selection_spans_capture() -> None:
    idx = select_frames(np.zeros((100, 3)), 20, "uniform")
    assert len(idx) == 20
    assert idx[0] == 0
    assert idx[-1] == 99
    assert np.all(np.diff(idx) > 0)


def test_farthest_selection_takes_both_ends_of_a_line() -> None:
    centers = np.stack([np.linspace(0, 10, 50), np.zeros(50), np.zeros(50)], axis=1)
    idx = select_frames(centers, 5, "farthest")
    assert len(idx) == 5
    assert {0, 49} <= set(idx.tolist())


def test_selection_keeps_all_frames_when_too_few() -> None:
    assert len(select_frames(np.zeros((10, 3)), 20)) == 10


def test_unknown_method_raises() -> None:
    with pytest.raises(ValueError, match="unknown"):
        select_frames(np.zeros((10, 3)), 5, "random")


def test_scannetpp_size_maps_to_504x336() -> None:
    assert target_size(1752, 1168, 504, 14) == (504, 336)


def test_resize_scales_intrinsics() -> None:
    k = np.array([[626.0, 0, 876.0], [0, 626.0, 584.0], [0, 0, 1]])
    image = np.zeros((1168, 1752, 3), dtype=np.uint8)
    resized, k_new = resize_view(image, k, (504, 336))
    assert resized.shape == (336, 504, 3)
    np.testing.assert_allclose(k_new[0, 0], 626.0 * 504 / 1752)
    np.testing.assert_allclose(k_new[:2, 2], [252.0, 168.0])


def test_zipped_assets_use_zip_on_server() -> None:
    assert (
        remote_path("abc", "dslr_resized_undistorted_dir")
        == "data/abc/dslr/resized_undistorted_images.zip"
    )
    assert remote_path("abc", "iphone_video_path") == "data/abc/iphone/rgb.mkv"
