import numpy as np
import torch

from src.geometry import Geometry
from src.loaders.ground_truth import GroundTruth
from src.loaders.views import Views
from src.refinement.volume import (
    CLASSES,
    ScenePoints,
    VolumeRefiner,
    grid_channels,
    pixel_classes,
    refine,
    scene_points,
    target_residual,
    voxelize,
)


def scene(views: int = 2, h: int = 12, w: int = 16) -> tuple[Geometry, np.ndarray]:
    """A pinhole scene looking at a slanted plane, and its images."""
    rng = np.random.default_rng(0)
    k = np.array([[20.0, 0, w / 2], [0, 20.0, h / 2], [0, 0, 1]])
    c2w = np.tile(np.eye(4), (views, 1, 1))
    c2w[:, 0, 3] = np.arange(views) * 0.1
    v, u = np.mgrid[0:h, 0:w] + 0.5
    depth = np.tile((2 + 0.02 * u + 0.01 * v)[None], (views, 1, 1)).astype(np.float32)
    rays = np.stack([u, v, np.ones_like(u)], -1) @ np.linalg.inv(k).T
    origins = np.broadcast_to(c2w[:, None, None, :3, 3], (views, h, w, 3))
    dirs = np.broadcast_to(rays, (views, h, w, 3))
    geometry = Geometry(
        depth,
        np.ones_like(depth),
        np.tile(k, (views, 1, 1)),
        c2w,
        origins.copy(),
        dirs.copy(),
    )
    images = rng.integers(0, 256, size=(views, h, w, 3), dtype=np.uint8)
    return geometry, images


def test_untrained_refiner_returns_its_input_bit_for_bit() -> None:
    geometry, images = scene()
    pts, _, _, _ = scene_points(
        geometry, images, np.full(geometry.depth.shape, -1), {}, 8
    )
    model = VolumeRefiner(
        voxelize(pts, semantics=True).shape[0], [8, 16], 16, semantics=True
    )
    out = refine(model, pts, geometry, torch.device("cpu"), semantics=True)
    np.testing.assert_array_equal(out.depth, geometry.depth)


def test_voxelize_keeps_every_point_and_normalizes_the_histogram() -> None:
    geometry, images = scene()
    ids = np.zeros(geometry.depth.shape, dtype=np.int32)
    pts, _, _, _ = scene_points(geometry, images, ids, {0: "wall"}, 8)
    grid = voxelize(pts, semantics=True)
    count = grid[0].expm1()
    assert torch.isclose(count.sum(), torch.tensor(float(len(pts.points))))
    hist = grid[-CLASSES:]
    occupied = count > 0
    torch.testing.assert_close(hist.sum(0)[occupied], torch.ones(int(occupied.sum())))
    assert torch.all(hist[1][occupied] == 1)  # every point is a wall


def test_target_is_zero_when_the_prediction_is_the_ground_truth() -> None:
    geometry, images = scene()
    views = Views(
        "synthetic", ["a", "b"], images, geometry.intrinsics, geometry.cam_to_world
    )
    gt = GroundTruth(
        np.zeros(geometry.depth.shape, dtype=np.int32), geometry.depth.copy(), {}
    )
    _, rays, depth, _ = scene_points(geometry, images, gt.instance_ids, {}, 8)
    r, valid = target_residual(rays, depth, views, gt)
    assert valid.all()
    np.testing.assert_allclose(r, 0, atol=1e-5)


def test_target_recovers_a_known_depth_error() -> None:
    geometry, images = scene()
    views = Views(
        "synthetic", ["a", "b"], images, geometry.intrinsics, geometry.cam_to_world
    )
    truth = geometry.depth.copy()
    truth[:, 4:7, 6:10] *= 1.05  # a small patch: the similarity cannot absorb it
    gt = GroundTruth(np.zeros(truth.shape, dtype=np.int32), truth, {})
    _, rays, depth, _ = scene_points(geometry, images, gt.instance_ids, {}, 8)
    r, _ = target_residual(rays, depth, views, gt)
    r = r.reshape(truth.shape)
    gap = np.log(1.05)
    patch = np.zeros(truth.shape, dtype=bool)
    patch[:, 4:7, 6:10] = True
    assert abs((r[patch].mean() - r[~patch].mean()) - gap) < 0.1 * gap


def test_pixel_classes_map_instances_to_nyu40() -> None:
    ids = np.array([[[0, 1, -1]]])
    out = pixel_classes(ids, {0: "wall", 1: "bed"})
    np.testing.assert_array_equal(out, [[[1, 4, 0]]])


def test_sampling_reads_the_cell_a_point_lies_in() -> None:
    grid = torch.zeros(1, 8, 8, 8)
    grid[0, 2, 3, 5] = 1.0  # z, y, x
    model = VolumeRefiner(1, [8], 8, semantics=False)
    pts = ScenePoints(
        torch.tensor([[5.5, 3.5, 2.5]]) / 8,
        torch.zeros(1, 3),
        torch.zeros(1, 3),
        torch.zeros(1),
        torch.zeros(1, dtype=torch.long),
        torch.zeros(3),
        1 / 8,
        (8, 8, 8),
    )
    coords = 2 * (pts.points - pts.lo) / (torch.tensor(pts.dims) * pts.cell) - 1
    value = torch.nn.functional.grid_sample(
        grid[None], coords.view(1, -1, 1, 1, 3), align_corners=False
    )
    assert torch.isclose(value.flatten()[0], torch.tensor(1.0))
    assert model.head[-1].weight.abs().sum() == 0


def test_grid_channels_match_voxelize() -> None:
    geometry, images = scene()
    pts, _, _, _ = scene_points(
        geometry, images, np.full(geometry.depth.shape, -1), {}, 8
    )
    for semantics in (False, True):
        grid = voxelize(pts, semantics=semantics)
        assert grid.shape[0] == grid_channels(semantics=semantics)
