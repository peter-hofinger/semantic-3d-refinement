"""Artifact measurements: occlusion edges, plane fits, RANSAC, texture and the sparsification curve."""

import numpy as np
import torch

from semantic_3d_refinement.artifacts import (
    SceneData,
    confidence,
    depth_edges,
    plane_residuals,
    ransac_planes,
    texture,
)


def test_depth_edges_mark_both_sides_of_a_step_and_skip_holes() -> None:
    gt = np.full((1, 6, 8), 2.0)
    gt[:, :, 4:] = 3.0
    gt[:, 0, 0] = 0
    edges = depth_edges(gt, 0.05)
    assert edges[0][:, 3:5].all() and edges[0].sum() == 12


def test_plane_residuals_vanish_on_a_plane_and_measure_a_bump() -> None:
    g = np.random.default_rng(0).uniform(-1, 1, size=(500, 2))
    plane = np.c_[g, 0.3 * g[:, 0] - 0.2 * g[:, 1] + 1]
    assert np.abs(plane_residuals(plane)).max() < 1e-12
    bumped = plane + np.array([0, 0, 0.01]) * (np.arange(500) % 2)[:, None]
    assert 0.004 < np.sqrt(np.mean(plane_residuals(bumped) ** 2)) < 0.006


def test_ransac_finds_the_planes_of_a_corner() -> None:
    g = np.random.default_rng(1).uniform(0, 1, size=(3000, 2))
    floor, wall = np.c_[g, np.zeros(3000)], np.c_[g[:, :1], np.zeros(3000), g[:, 1:]]
    planes = ransac_planes(np.r_[floor, wall[:1500]], 4, 0.01, 0.1, torch.device("cpu"))
    assert len(planes) == 2
    normals = sorted(np.abs(n).argmax() for n, _ in planes)
    assert normals == [1, 2]


def test_texture_is_zero_on_flat_images_and_positive_on_a_checkerboard() -> None:
    flat = np.full((1, 20, 20, 3), 120, dtype=np.uint8)
    board = (255 * ((np.indices((20, 20)).sum(0) % 2)[None, ..., None])).repeat(3, -1).astype(np.uint8)
    assert texture(flat, 7).max() < 1e-6 and texture(board, 7)[0, 10, 10] > 0.4


def test_sparsification_by_a_perfect_confidence_is_the_oracle() -> None:
    rng = np.random.default_rng(2)
    gt = rng.uniform(1, 3, size=(1, 30, 40))
    pred = gt * np.exp(rng.normal(scale=0.05, size=gt.shape))
    blank = np.zeros(0)
    data = SceneData(blank, gt, pred, -np.abs(np.log(pred / gt)), blank, blank, blank, blank, blank, blank, blank)
    summary, arrays = confidence(data)
    np.testing.assert_allclose(arrays["by_conf"], arrays["oracle"])
    assert np.isclose(arrays["by_conf"][0], np.abs(np.log(pred / gt)).mean())
    assert summary["spearman_conf_vs_error"] < -0.99
