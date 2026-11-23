"""E84: the optimum of Déjà View's confidence loss, checked against dvlt's own loss."""

import numpy as np
import pytest
import torch
from dvlt.common.constants import DataField
from dvlt.model_components.loss import DepthLoss

from experiments.scripts.e84_dejaview_optimum_filter import (
    ALPHA,
    GAMMA,
    iou,
    optimal_confidence,
    top_share,
)


def per_sample_loss(depth: torch.Tensor, gt: torch.Tensor, conf: torch.Tensor) -> float:
    """Return dvlt's DepthLoss (confidence on, no quantile mask, no gradient loss)."""
    loss = DepthLoss(
        grad_loss=None, valid_range=-1.0, alpha=ALPHA, gamma=GAMMA, disable_conf=False
    )
    batch = {
        DataField.DEPTHS: gt,
        DataField.POINT_MASKS: torch.ones_like(gt, dtype=torch.bool),
        DataField.IS_SYNTHETIC: torch.ones(gt.shape[:2], dtype=torch.bool),
    }
    value, _ = loss({"depth": depth, "depth_conf": conf}, batch)
    return value.item()


def pixels(low: float, high: float) -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    """Return depth, GT depth (1, 1, 20, 20) and the absolute errors in [low, high]."""
    rng = np.random.default_rng(0)
    error = rng.uniform(low, high, (20, 20))
    gt = rng.uniform(1.0, 3.0, (20, 20))
    shape = (1, 1, 20, 20)
    return (
        torch.from_numpy(gt + error).reshape(shape),
        torch.from_numpy(gt).reshape(shape),
        error,
    )


def test_the_optimum_minimizes_deja_views_confidence_loss_inside_the_floor() -> None:
    depth, gt, error = pixels(
        0.02, 0.15
    )  # alpha / e in 1.3 .. 10, above the floor c >= 1
    best = torch.from_numpy(optimal_confidence(error)).reshape(depth.shape)
    assert (best > 1.2).all()
    at_best = per_sample_loss(depth, gt, best)
    for factor in (0.9, 0.99, 1.01, 1.1):
        assert per_sample_loss(depth, gt, best * factor) > at_best


def test_below_the_floor_the_optimum_is_c_equal_one() -> None:
    depth, gt, error = pixels(0.3, 0.5)  # alpha / e < 1: the activation's floor binds
    best = optimal_confidence(error)
    assert (best == 1.0).all()
    ones = torch.ones_like(depth)
    assert per_sample_loss(depth, gt, ones * 1.05) > per_sample_loss(depth, gt, ones)


def test_a_confidence_threshold_is_an_error_threshold() -> None:
    error = np.random.default_rng(1).uniform(0.001, 0.4, 5000)
    for tau in (2.0, 5.0, 20.0):
        keep = optimal_confidence(error) >= tau
        assert (keep == (error <= ALPHA / (GAMMA * tau))).all()


def test_top_share_keeps_the_exact_share_and_iou_is_a_jaccard_index() -> None:
    score = np.random.default_rng(2).random(1000)
    score[:100] = 0.5  # ties
    for share in (0.1, 0.5, 0.9):
        kept = top_share(score, share)
        assert kept.sum() == round(share * 1000)
        assert score[kept].min() >= score[~kept].max()
    a = np.array([True, True, False, False])
    b = np.array([True, False, True, False])
    assert iou(a, b) == pytest.approx(1 / 3)
