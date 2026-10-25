from pathlib import Path

import numpy as np

from src.backbone import ConstantDepthBackbone
from src.geometry import Geometry
from src.loaders import Views
from src.pipeline import Pipeline
from src.refinement import IdentityRefiner
from src.segmentation import EmptySegmenter


def make_views(num: int = 2, height: int = 4, width: int = 6) -> Views:
    k = np.array([[5.0, 0, width / 2], [0, 5.0, height / 2], [0, 0, 1]])
    return Views(
        scene_id="toy",
        names=[f"{i}.jpg" for i in range(num)],
        images=np.zeros((num, height, width, 3), dtype=np.uint8),
        intrinsics=np.repeat(k[None], num, axis=0).astype(np.float32),
        cam_to_world=np.repeat(np.eye(4)[None], num, axis=0).astype(np.float32),
    )


def test_center_pixel_lifts_straight_ahead() -> None:
    k = np.array([[[10.0, 0, 1.0], [0, 10.0, 1.0], [0, 0, 1]]])
    geometry = Geometry(
        depth=np.full((1, 2, 2), 3.0),
        confidence=np.ones((1, 2, 2)),
        intrinsics=k,
        cam_to_world=np.eye(4)[None],
    )
    points = geometry.world_points()
    # pixel (0, 0) has center (0.5, 0.5), half a pixel left of and above (1, 1)
    np.testing.assert_allclose(points[0, 0, 0], [-0.15, -0.15, 3.0])


def test_camera_translation_moves_points() -> None:
    pose = np.eye(4)
    pose[:3, 3] = [1.0, 2.0, 3.0]
    geometry = Geometry(
        depth=np.full((1, 1, 1), 2.0),
        confidence=np.ones((1, 1, 1)),
        intrinsics=np.array([[[1.0, 0, 0.5], [0, 1.0, 0.5], [0, 0, 1]]]),
        cam_to_world=pose[None],
    )
    np.testing.assert_allclose(geometry.world_points()[0, 0, 0], [1.0, 2.0, 5.0])


def test_placeholder_pipeline_runs_end_to_end(tmp_path: Path) -> None:
    views = make_views()
    pipeline = Pipeline(ConstantDepthBackbone(), EmptySegmenter(), IdentityRefiner())
    result = pipeline.run(views)
    assert len(result.cloud) == 2 * 4 * 6
    assert np.all(result.cloud.instance_ids == -1)
    path = tmp_path / "cloud.ply"
    result.cloud.save_ply(path)
    assert path.read_bytes().startswith(b"ply\n")


def test_geometry_save_load_keeps_rays(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    geometry = Geometry(
        depth=rng.random((2, 3, 4), dtype=np.float32),
        confidence=rng.random((2, 3, 4), dtype=np.float32),
        intrinsics=np.repeat(np.eye(3, dtype=np.float32)[None], 2, axis=0),
        cam_to_world=np.repeat(np.eye(4, dtype=np.float32)[None], 2, axis=0),
        ray_origins=rng.random((2, 3, 4, 3), dtype=np.float32),
        ray_directions=rng.random((2, 3, 4, 3), dtype=np.float32),
    )
    path = tmp_path / "geometry.npz"
    geometry.save(path)
    loaded = Geometry.load(path)
    np.testing.assert_array_equal(loaded.depth, geometry.depth)
    np.testing.assert_array_equal(loaded.ray_directions, geometry.ray_directions)
