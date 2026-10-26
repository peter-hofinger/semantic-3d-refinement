"""Debugging ladder for monocular refinement: identity, synthetic overfit, contracts."""

import dataclasses

import pytest
import torch

from semantic_3d_refinement.refine import (
    RefineConfig,
    depth_metrics,
    points_from_rays,
    refine,
    second_differences,
)
from semantic_3d_refinement.synthetic import room_scene

CFG = RefineConfig(device="cpu")


def scene(
    **kwargs: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    data = {k: torch.from_numpy(v) for k, v in room_scene(**kwargs).items()}
    return data["depth"], data["rays"], data["conf"], data["gt_depth"]


def test_planes_are_in_the_null_space_of_the_smoothness_term() -> None:
    v, u = torch.meshgrid(torch.arange(8.0), torch.arange(9.0), indexing="ij")
    for diff in second_differences(0.3 * u - 0.2 * v + 1.0):
        assert diff.abs().max() < 1e-6


@pytest.mark.parametrize("cfg", [dataclasses.replace(CFG, iters=0), dataclasses.replace(CFG, lambda_bd=0.0)])
def test_identity(cfg: RefineConfig) -> None:
    depth, rays, conf, _ = scene()
    depth_hat, points, delta, _ = refine(depth, rays, conf, cfg)
    assert torch.equal(delta, torch.zeros_like(delta))
    assert torch.equal(depth_hat, depth)
    assert torch.equal(points, points_from_rays(depth, rays))


def test_single_plane_is_recovered() -> None:
    depth, rays, conf, gt = scene(noise=0.02, outlier_frac=0.0, planes=(((0.2, 0.0, 1.0), 3.0),))
    depth_hat, *_ = refine(depth, rays, conf, dataclasses.replace(CFG, lambda_bd=1000.0, iters=500))
    before, after = depth_metrics(depth, rays, gt), depth_metrics(depth_hat, rays, gt)
    assert after["abs_rel"] < 0.1 * before["abs_rel"], (before, after)


def test_room_error_drops() -> None:
    depth, rays, conf, gt = scene()
    depth_hat, _, _, log = refine(depth, rays, conf, CFG)
    before, after = depth_metrics(depth, rays, gt), depth_metrics(depth_hat, rays, gt)
    assert log[-1]["E"] < log[0]["E"]
    assert after["point_rmse"] < 0.5 * before["point_rmse"], (before, after)


def test_rejects_bad_inputs() -> None:
    depth, rays, conf, _ = scene()
    with pytest.raises(AssertionError):
        refine(-depth, rays, conf, CFG)
    with pytest.raises(AssertionError):
        refine(depth, rays[..., :3], conf, CFG)
    with pytest.raises(AssertionError):
        RefineConfig(iters=-1)
