import dataclasses
from pathlib import Path

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from src.geometry import Geometry
from src.loaders.ground_truth import GroundTruth
from src.loaders.views import Views
from src.refinement.volume import (
    CLASSES,
    KeepHead,
    ScenePoints,
    TrainScene,
    VolumeConfig,
    VolumeRefiner,
    compact,
    distance_target,
    grid_channels,
    keep_labels,
    keep_score,
    pixel_classes,
    predict_heads,
    refine,
    scene_points,
    spearman,
    target_residual,
    train,
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
    out = refine(model, pts, geometry, torch.device("cpu"))
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


def test_untrained_keep_head_keeps_the_depth_and_scores_every_pixel_alike() -> None:
    geometry, images = scene()
    pts, _, _, _ = scene_points(
        geometry, images, np.full(geometry.depth.shape, -1), {}, 8
    )
    model = VolumeRefiner(
        voxelize(pts, semantics=False).shape[0], [8, 16], 16, semantics=False, outputs=2
    )
    heads = predict_heads(model, pts, torch.device("cpu"))
    assert heads.shape == (len(pts.points), 2)
    assert (heads == 0).all()
    out = refine(model, pts, geometry, torch.device("cpu"))
    np.testing.assert_array_equal(out.depth, geometry.depth)


def test_keep_labels_mark_the_least_errors_and_count_invalid_as_errors() -> None:
    err = torch.tensor([0.3, torch.inf, 0.1, 0.2, 0.0, torch.inf])
    labels = keep_labels(err, 0.5)
    torch.testing.assert_close(labels, torch.tensor([0.0, 0, 1, 1, 1, 0]))
    assert keep_labels(torch.full((4,), torch.inf), 0.5).sum() == 2


def test_keep_head_is_off_in_the_e41b_config_and_on_in_e57a() -> None:
    assert VolumeConfig.load(Path("configs/e41b_volume.yaml")).keep is None
    keep = VolumeConfig.load(Path("configs/e57a_volume.yaml")).keep
    assert isinstance(keep, KeepHead)
    assert keep.residual


def test_distance_target_is_zero_on_the_ground_truth_and_bounded_by_the_shift() -> None:
    geometry, images = scene()
    views = Views(
        "synthetic", ["a", "b"], images, geometry.intrinsics, geometry.cam_to_world
    )
    truth = geometry.depth.copy()
    gt = GroundTruth(np.zeros(truth.shape, dtype=np.int32), truth, {})
    _, rays, depth, scale = scene_points(geometry, images, gt.instance_ids, {}, 8)
    np.testing.assert_allclose(
        distance_target(rays, depth, views, gt, scale), 0, atol=1e-6
    )
    moved = geometry.depth.copy()
    moved[:, 4:7, 6:10] *= 1.05  # a small patch: the similarity cannot absorb it
    _, rays, depth, scale = scene_points(
        dataclasses.replace(geometry, depth=moved), images, gt.instance_ids, {}, 8
    )
    d = distance_target(rays, depth, views, gt, scale).reshape(truth.shape)
    patch = np.zeros(truth.shape, dtype=bool)
    patch[:, 4:7, 6:10] = True
    bound = (0.05 * truth[patch] / scale).max()  # the nearest GT point is no farther
    assert d[patch].min() > 0 and d[patch].max() <= 1.2 * bound
    assert d[~patch].mean() < 0.1 * d[patch].mean()


def test_spearman_is_one_for_order_minus_one_for_reverse_and_zero_when_constant() -> (
    None
):
    a = torch.tensor([0.3, 0.1, 0.9, 0.5])
    assert torch.isclose(spearman(a, 2 * a + 1), torch.tensor(1.0))
    assert torch.isclose(spearman(a, -a), torch.tensor(-1.0))
    assert spearman(torch.zeros(4), a) == 0


def test_the_distance_score_drops_the_largest_predicted_distance() -> None:
    heads = torch.tensor([[0.0, 0.3], [0.0, 0.1], [0.0, 0.9]])
    distance = KeepHead(quantile=0.5, residual=False, weight=1.0, target="distance")
    label = KeepHead(quantile=0.5, residual=False, weight=1.0)
    torch.testing.assert_close(
        keep_score(distance, heads), torch.tensor([-0.3, -0.1, -0.9])
    )
    torch.testing.assert_close(keep_score(label, heads), heads[:, 1])


def test_distance_config_trains_the_distance_and_the_older_configs_do_not() -> None:
    keep = VolumeConfig.load(Path("configs/e57c_volume.yaml")).keep
    assert keep is not None
    assert keep.target == "distance"
    assert not keep.residual
    assert (
        VolumeConfig.load(Path("configs/e57a_volume_keep.yaml")).keep.target == "label"
    )
    assert VolumeConfig.load(Path("configs/e57c_volume.yaml")).val


def test_distance_head_overfits_a_smooth_distance(tmp_path: Path) -> None:
    geometry, images = scene()
    pts, _, _, _ = scene_points(
        geometry, images, np.full(geometry.depth.shape, -1), {}, 8
    )
    n = len(pts.points)
    distance = 0.2 * (pts.points[:, 0] - pts.points[:, 0].min())
    kept = torch.arange(0, n, 3)
    data = [
        TrainScene(
            compact(pts), torch.zeros(n).half(), torch.ones(n, dtype=torch.bool), 2,
            n // 2, kept, torch.ones(n, dtype=torch.bool), distance,
        )
    ]  # fmt: skip
    cfg = dataclasses.replace(
        VolumeConfig.load(Path("configs/e57c_overfit_bedroom.yaml")),
        widths=[8, 16], head=16, steps=200, batch_points=300, kept_points=100,
        view_subset=[1, 2], resolution=8, amp=False,
    )  # fmt: skip
    with SummaryWriter(tmp_path) as writer:
        model, log = train(
            cfg, data, [], torch.device("cpu"), semantics=False, seed=0, writer=writer
        )
    assert log[-1]["distance_l1"] < 0.25 * log[0]["distance_l1"]
    assert model.outputs == 2
    assert max(r["spearman"] for r in log) > 0.8
