import dataclasses
from pathlib import Path

import numpy as np
import torch

from src.geometry import Geometry
from src.refinement.learned import Refiner
from src.refinement.reproject import (
    ReprojectConfig,
    SceneData,
    channels,
    inputs,
    refine,
    reprojection,
    soft_histogram,
    world_points,
)

BINS, MAX = 9, 0.4


def cfg() -> ReprojectConfig:
    return ReprojectConfig(
        Path(), "", "", "", [], 0, [], {}, BINS, MAX, 8, 1, 1, [2, 2], 1e-3, 0.5, [0]
    )


def plane(views: int = 2, h: int = 12, w: int = 16, z: float = 2.0) -> SceneData:
    """Cameras along x looking at the fronto-parallel plane at depth z."""
    k = torch.tensor([[20.0, 0, w / 2], [0, 20.0, h / 2], [0, 0, 1]])
    center = torch.zeros(views, 3)
    center[:, 0] = torch.arange(views) * 0.1
    classes = torch.ones(views, h, w, dtype=torch.uint8)
    classes[:, :, : w // 2] = 2
    return SceneData(
        torch.randint(0, 256, (views, h, w, 3), dtype=torch.uint8),
        torch.full((views, h, w), z),
        torch.zeros(views, h, w, dtype=torch.float16),
        classes,
        k.expand(views, 3, 3).clone(),
        torch.eye(3).expand(views, 3, 3).clone(),
        center,
        torch.zeros(views, h, w, dtype=torch.float16),
        torch.ones(views, h, w, dtype=torch.bool),
    )


def votes(scene: SceneData, *, semantics: bool = False) -> torch.Tensor:
    points = world_points(scene.depth, scene.k, scene.rotation, scene.center)
    if not semantics:
        scene = dataclasses.replace(scene, classes=None)
    return reprojection(0, torch.tensor([1]), points, scene, cfg())


def test_soft_histogram_keeps_mass_and_mean() -> None:
    rho = torch.tensor([0.13, -0.27, 0.0, 0.5, -0.9, 0.7])
    hist = soft_histogram(torch.zeros(6, dtype=torch.long), rho, 1, BINS, MAX)[0]
    centres = torch.linspace(-MAX, MAX, BINS)
    inside = rho[rho.abs() <= MAX]
    assert torch.isclose(hist[:BINS].sum(), torch.tensor(float(len(inside))))
    assert torch.isclose((hist[:BINS] * centres).sum(), inside.sum(), atol=1e-6)
    assert hist[BINS] == 1  # in front: -0.9
    assert hist[BINS + 1] == 2  # behind: 0.5, 0.7


def test_consistent_views_vote_for_zero_and_a_farther_view_moves_the_votes() -> None:
    scene = plane()
    hit = votes(scene)[:BINS].sum(0) > 0
    assert hit.float().mean() > 0.5  # the views overlap
    mean = (votes(scene)[:BINS] * torch.linspace(-MAX, MAX, BINS)[:, None, None]).sum(0)
    assert torch.allclose(mean[hit], torch.zeros(()), atol=1e-5)
    depth = scene.depth.clone()
    depth[1] *= np.exp(0.2)
    far = votes(dataclasses.replace(scene, depth=depth))
    mean = (far[:BINS] * torch.linspace(-MAX, MAX, BINS)[:, None, None]).sum(0)
    shifted = far[:BINS].sum(0) > 0
    assert torch.allclose(mean[shifted], torch.tensor(0.2), atol=1e-5)


def test_votes_do_not_change_with_the_scene_scale() -> None:
    scene = plane()
    scaled = dataclasses.replace(scene, depth=scene.depth * 3, center=scene.center * 3)
    torch.testing.assert_close(
        votes(scene, semantics=True), votes(scaled, semantics=True)
    )


def test_same_class_votes_are_a_subset() -> None:
    v = votes(plane(), semantics=True)
    assert v.shape[0] == channels(BINS, semantics=True) - 5
    all_votes, same = v[: BINS + 2], v[BINS + 3 :]
    assert (same <= all_votes + 1e-6).all()
    assert same.sum() > 0
    assert same.sum() < all_votes.sum()  # pixels near the class border see the other


def test_untrained_refiner_returns_its_input_bit_for_bit() -> None:
    scene = plane(views=3)
    geometry = Geometry(
        scene.depth.numpy(),
        np.ones(scene.depth.shape, np.float32),
        scene.k.numpy(),
        np.tile(np.eye(4, dtype=np.float32), (3, 1, 1)),
    )
    model = Refiner(channels(BINS, semantics=True), 8)
    out = refine(model, scene, geometry, cfg(), torch.device("cpu"))
    np.testing.assert_array_equal(out.depth, geometry.depth)


def test_inputs_leave_the_target_out_of_its_own_votes() -> None:
    scene = dataclasses.replace(plane(views=1), classes=None)
    x = inputs(scene, torch.tensor([0]), torch.tensor([0]), cfg(), torch.device("cpu"))
    assert x.shape == (1, channels(BINS, semantics=False), 12, 16)
    assert (x[0, 5:] == 0).all()
