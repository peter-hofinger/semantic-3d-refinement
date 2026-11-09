import numpy as np
import torch

from src.geometry import Geometry
from src.refinement.learned import (
    Refiner,
    boundaries,
    features,
    log_error,
    loss,
    refine,
)
from src.segmentation.base import Segmentation


def test_untrained_refiner_returns_its_input_bit_for_bit() -> None:
    rng = np.random.default_rng(0)
    depth = rng.uniform(1, 3, size=(2, 20, 30)).astype(np.float32)
    geometry = Geometry(
        depth,
        np.ones_like(depth),
        np.tile(np.eye(3), (2, 1, 1)),
        np.tile(np.eye(4), (2, 1, 1)),
    )
    images = rng.integers(0, 256, size=(2, 20, 30, 3), dtype=np.uint8)
    x = features(images, depth, geometry.confidence, None)
    out = refine(Refiner(x.shape[1], 8), geometry, x, torch.device("cpu"))
    np.testing.assert_array_equal(out.depth, depth)


def test_refiner_keeps_odd_sizes() -> None:
    assert Refiner(7, 8)(torch.randn(2, 7, 30, 45)).shape == (2, 1, 30, 45)


def test_boundaries_mark_both_sides_of_an_id_change() -> None:
    ids = np.array([[[1, 1, 2, 2]]])
    np.testing.assert_array_equal(boundaries(ids), [[[0, 1, 1, 0]]])


def test_features_add_boundary_and_structure_maps() -> None:
    ids = np.array([[[0, 0, 1, 1], [0, 0, 1, 1]]], dtype=np.int32)
    masks = Segmentation(ids, {0: "wall", 1: "chair"}, {0: 1.0, 1: 1.0})
    x = features(
        np.zeros((1, 2, 4, 3), np.uint8), np.ones((1, 2, 4)), np.ones((1, 2, 4)), masks
    )
    assert x.shape == (1, 7, 2, 4)
    np.testing.assert_array_equal(x[0, 6], [[1, 1, 0, 0], [1, 1, 0, 0]])


def test_log_error_ignores_global_scale() -> None:
    depth = np.random.default_rng(1).uniform(1, 3, size=(2, 5, 6))
    gt = 2.5 * depth
    gt[0, 0, 0] = 0  # no ground truth there
    target, valid = log_error(depth, gt)
    np.testing.assert_allclose(target, 0, atol=1e-12)
    assert valid[0, 0, 0] == 0 and valid.sum() == valid.size - 1


def test_loss_is_zero_at_the_target() -> None:
    t = torch.randn(2, 16, 16)
    assert float(loss(t, t, torch.ones_like(t), 0.5)) == 0
