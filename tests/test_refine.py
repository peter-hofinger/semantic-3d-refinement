"""Debugging ladder for monocular refinement: identity, synthetic overfit, contracts."""

import dataclasses

import pytest
import torch

from semantic_3d_refinement.normals import BiniConfig, bini, grid_score, normals_from_depth
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


TILTED_PLANE = (((0.2, 0.0, 1.0), 3.0),)


def test_bini_recovers_a_plane_from_its_normal() -> None:
    depth, rays, _, gt = scene(outlier_frac=0.0, planes=TILTED_PLANE)
    n = torch.tensor(TILTED_PLANE[0][0])
    normals = (-n / n.norm()).expand(*depth.shape, 3).contiguous()
    depth_hat, _, log = bini(normals, rays, depth, BiniConfig(device="cpu"))
    assert log[-1] < log[0]
    assert depth_metrics(depth_hat, rays, gt)["abs_rel"] < 1e-3 < depth_metrics(depth, rays, gt)["abs_rel"]


def test_bini_identity_under_a_strong_prior() -> None:
    depth, rays, _, _ = scene(outlier_frac=0.0)
    normals = normals_from_depth(depth, rays)
    depth_hat, _, _ = bini(normals, rays, depth, BiniConfig(lambda_depth=1e8, device="cpu"))
    assert ((depth_hat - depth).abs() / depth).max() < 1e-4


def test_blurred_prior_lets_normals_set_the_detail() -> None:
    _, rays, _, gt = scene(outlier_frac=0.0, planes=TILTED_PLANE)
    noisy = gt * torch.exp(0.02 * torch.randn(gt.shape, generator=torch.Generator().manual_seed(0)))
    n = torch.tensor(TILTED_PLANE[0][0])
    normals = (-n / n.norm()).expand(*gt.shape, 3).contiguous()
    errors = {}
    for sigma in (0.0, 4.0):
        cfg = BiniConfig(lambda_depth=1e5, prior_sigma=sigma, device="cpu")
        errors[sigma] = depth_metrics(bini(normals, rays, noisy, cfg)[0], rays, gt)["abs_rel"]
    assert errors[4.0] < 0.25 * errors[0.0], errors


def test_patch_steps_in_the_prior_do_not_survive() -> None:
    _, rays, _, gt = scene(outlier_frac=0.0, planes=TILTED_PLANE, h=98, w=126)
    steps = 0.02 * torch.randn(7, 9, generator=torch.Generator().manual_seed(0))
    prior = gt * torch.exp(steps.repeat_interleave(14, 0).repeat_interleave(14, 1))
    n = torch.tensor(TILTED_PLANE[0][0])
    normals = (-n / n.norm()).expand(*gt.shape, 3).contiguous()
    depth_hat = bini(normals, rays, prior, BiniConfig(lambda_depth=100.0, device="cpu"))[0]
    assert depth_metrics(depth_hat, rays, gt)["abs_rel"] < 0.5 * depth_metrics(prior, rays, gt)["abs_rel"]


def test_grid_score_finds_patch_seams() -> None:
    noisy, rays, _, _ = scene(noise=0.002, outlier_frac=0.0, h=98, w=126)
    steps = 0.01 * torch.randn(7, 9, generator=torch.Generator().manual_seed(0))
    seamed = noisy * torch.exp(steps.repeat_interleave(14, 0).repeat_interleave(14, 1))
    clean = grid_score(normals_from_depth(noisy, rays))
    assert 0.7 < clean < 1.3 and grid_score(normals_from_depth(seamed, rays)) > 1.5 * clean, clean
