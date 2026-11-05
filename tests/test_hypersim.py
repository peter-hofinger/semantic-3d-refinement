from pathlib import Path

import numpy as np
import pytest

from src.loaders.hypersim.config import (
    OFFICIAL_PARTITION,
    HypersimConfig,
    check_splits,
    load_config,
)
from src.loaders.hypersim.prepare import (
    STUFF_BASE,
    UNKNOWN_CLASS,
    frame_ground_truth,
    instance_classes,
    sample_nearest,
)
from src.loaders.hypersim.scene import (
    FrameLabels,
    camera_from_cam_from_uv,
    distance_to_depth,
)
from src.loaders.scannetpp.gt import NO_MESH
from src.segmentation.base import NO_INSTANCE

# ai_015_001: 60 degree horizontal field of view, 1024 x 768
M_PINHOLE = np.diag([np.tan(np.pi / 6), np.tan(np.pi / 6) * 768 / 1024, -1.0])


def test_plain_camera_matches_field_of_view() -> None:
    k, q = camera_from_cam_from_uv(M_PINHOLE, 1024, 768)
    assert k[0, 0] == pytest.approx(512 / np.tan(np.pi / 6))
    assert k[1, 1] == pytest.approx(k[0, 0])
    assert k[0, 2] == pytest.approx(512)
    assert k[1, 2] == pytest.approx(384)
    np.testing.assert_allclose(q, np.eye(3), atol=1e-12)


@pytest.mark.parametrize(
    "m",
    [
        M_PINHOLE,
        # ai_015_001: tilt-shift, the ray's z depends on v
        np.array([[0.57776, 0, 0], [0, 0.43301, 0], [0, 0.016371, -1.00071]]),
    ],
)
def test_camera_reproduces_hypersim_rays(m: np.ndarray) -> None:
    # pixel centres through K and Q (OpenCV) and through M_cam_from_uv (OpenGL, v up)
    k, q = camera_from_cam_from_uv(m, 1024, 768)
    for x, y in [(100.5, 600.5), (512.0, 384.0), (1000.5, 10.5)]:
        ray_cv = q @ np.linalg.inv(k) @ [x, y, 1.0]
        u, v = (x - 512) / 512, -(y - 384) / 384
        ray_gl_as_cv = (m @ [u, v, 1.0]) * [1, -1, -1]
        np.testing.assert_allclose(
            ray_cv / np.linalg.norm(ray_cv),
            ray_gl_as_cv / np.linalg.norm(ray_gl_as_cv),
            atol=1e-12,
        )
    assert k[0, 1] == 0
    assert np.linalg.det(q) == pytest.approx(1)


def test_skewed_camera_is_rejected() -> None:
    m = M_PINHOLE.copy()
    m[2, 0] = 0.1
    m[2, 1] = 0.1
    with pytest.raises(ValueError, match="skew"):
        camera_from_cam_from_uv(m, 1024, 768)


def test_distance_becomes_z_depth() -> None:
    k, _ = camera_from_cam_from_uv(M_PINHOLE, 8, 6)
    z = 3.0
    u, v = np.meshgrid(np.arange(8) + 0.5, np.arange(6) + 0.5)
    rays = np.stack([u, v, np.ones_like(u)], axis=-1) @ np.linalg.inv(k).T
    distance = z * np.linalg.norm(rays, axis=-1)  # fronto-parallel plane at z = 3
    np.testing.assert_allclose(distance_to_depth(distance, k), z, rtol=1e-6)


def test_nearest_sampling_takes_pixel_under_centre() -> None:
    a = np.arange(16).reshape(4, 4)
    np.testing.assert_array_equal(sample_nearest(a, (2, 2)), [[5, 7], [13, 15]])


def test_stuff_classes_get_one_shared_id_and_no_geometry_is_masked() -> None:
    labels = FrameLabels(
        distance=np.array([[1.0, 1.0, np.nan, 1.0]], dtype=np.float32),
        semantic=np.array([[1, 5, 1, -1]]),  # wall, chair, wall (no geometry), none
        instance=np.array([[-1, 7, -1, -1]]),
        entity=np.array([[3, 4, -1, 5]]),
    )
    k = np.eye(3)
    ids, depth, _ = frame_ground_truth(labels, k)
    np.testing.assert_array_equal(ids, [[STUFF_BASE + 1, 7, NO_MESH, NO_INSTANCE]])
    assert depth[0, 2] == 0
    assert np.isfinite(depth).all()


def test_instance_class_is_the_majority_label() -> None:
    ids = np.array([[7, 7, 7, STUFF_BASE + 1]])
    semantic = np.array([[5, 5, 40, 1]])
    assert instance_classes(ids, semantic) == {7: "chair", STUFF_BASE + 1: "wall"}


def test_instance_without_label_is_unknown() -> None:
    ids = np.array([[3, 3, 9]])
    semantic = np.array([[-1, -1, 5]])
    assert instance_classes(ids, semantic) == {3: UNKNOWN_CLASS, 9: "chair"}


M_TILT_SHIFT = np.array([[0.57776, 0, 0], [0, 0.43301, 0], [0, 0.016371, -1.00071]])


def write_metadata(root: Path) -> None:
    """Fake metadata: val ai_v_plain, ai_v_tilt; test ai_t_plain; cam_01 not public."""
    with (root / "metadata_images_split_scene_v1.csv").open("w", encoding="utf-8") as f:
        f.write("scene_name,camera_name,frame_id,included_in_public_release,")
        f.write("exclude_reason,split_partition_name\n")
        for scene, part in (
            ("ai_v_plain", "val"),
            ("ai_v_tilt", "val"),
            ("ai_t_plain", "test"),
        ):
            f.write(f"{scene},cam_00,0,True,,{part}\n")
            f.write(f"{scene},cam_01,0,False,CONTENT FLAGGED FOR REMOVAL,\n")
    cols = [f"M_cam_from_uv_{i}{j}" for i in range(3) for j in range(3)]
    with (root / "metadata_camera_parameters.csv").open("w", encoding="utf-8") as f:
        f.write(",".join(["scene_name", "settings_output_img_width",
                          "settings_output_img_height", *cols]) + "\n")  # fmt: skip
        for scene, m in (("ai_v_plain", M_PINHOLE), ("ai_v_tilt", M_TILT_SHIFT),
                         ("ai_t_plain", M_PINHOLE)):  # fmt: skip
            f.write(",".join([scene, "1024", "768", *map(str, m.ravel())]) + "\n")


def make_config(root: Path, **splits: dict[str, str]) -> HypersimConfig:
    """A config over the fake metadata with the given splits, the rest empty."""
    return HypersimConfig(
        data_root=root,
        metadata_root=root,
        output_root=root,
        splits={name: splits.get(name, {}) for name in OFFICIAL_PARTITION},
    )


def test_split_check_accepts_official_partitions(tmp_path: Path) -> None:
    write_metadata(tmp_path)
    cfg = make_config(
        tmp_path,
        dev={"ai_v_plain": "cam_00"},
        test={"ai_t_plain": "cam_00"},
        stress={"ai_v_tilt": "cam_00"},
    )
    check_splits(cfg)
    assert cfg.scenes(["dev", "stress"]) == {
        "ai_v_plain": "cam_00",
        "ai_v_tilt": "cam_00",
    }


@pytest.mark.parametrize(
    ("splits", "error"),
    [
        ({"dev": {"ai_t_plain": "cam_00"}}, "official test partition, not val"),
        ({"test": {"ai_v_plain": "cam_00"}}, "official val partition, not test"),
        ({"train": {"ai_v_plain": "cam_00"}}, "official val partition, not train"),
        ({"val": {"ai_v_tilt": "cam_00"}}, "off-centre"),
        ({"dev": {"ai_v_plain": "cam_01"}}, "not in the public release"),
    ],
)
def test_split_check_rejects(
    tmp_path: Path, splits: dict[str, dict[str, str]], error: str
) -> None:
    write_metadata(tmp_path)
    with pytest.raises(ValueError, match=error):
        check_splits(make_config(tmp_path, **splits))


def test_config_rejects_scene_in_two_splits(tmp_path: Path) -> None:
    path = tmp_path / "hypersim.yaml"
    path.write_text(
        "data_root: d\nmetadata_root: m\noutput_root: o\nsplits:\n"
        "  dev: {ai_v_plain: cam_00}\n  val: {ai_v_plain: cam_01}\n"
        "  test: {}\n  stress: {}\n  train: {}\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="more than one split"):
        load_config(path)
