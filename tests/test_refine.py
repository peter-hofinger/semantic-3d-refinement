"""Debugging ladder for monocular refinement: identity, synthetic overfit, contracts."""

import dataclasses

import numpy as np
import pytest
import torch

from semantic_3d_refinement.normals import BiniConfig, bini, normals_from_depth
from semantic_3d_refinement.refine import (
    RefineConfig,
    align_sim3,
    depth_metrics,
    fit_projection,
    multiview_error,
    points_from_rays,
    refine,
    second_differences,
)
from semantic_3d_refinement.synthetic import ROOM_PLANES, pinhole_rays, ray_depth, room_scene

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


def test_sim3_is_recovered() -> None:
    src = torch.randn(10, 3, generator=torch.Generator().manual_seed(0))
    rot = torch.linalg.matrix_exp(torch.tensor([[0.0, -0.3, 0.2], [0.3, 0.0, -0.1], [-0.2, 0.1, 0.0]]))
    shift = torch.tensor([1.0, -2.0, 0.5])
    s, r, t = align_sim3(src, 1.7 * src @ rot.T + shift)
    assert abs(s - 1.7) < 1e-6
    assert torch.allclose(r.float(), rot, atol=1e-6) and torch.allclose(t.float(), shift, atol=1e-5)


def test_true_depth_agrees_across_views() -> None:
    views = [pinhole_rays(48, 64, 50.0, yaw, (0.1 * yaw, -0.1, 0.2)) for yaw in (0.0, 0.3)]
    rays = torch.from_numpy(np.stack(views))
    gt = torch.from_numpy(ray_depth(rays.numpy(), ROOM_PLANES))
    rays, gt = rays.float(), gt.float()
    cams = [fit_projection(r) for r in rays]
    err = multiview_error(gt, rays, cams)
    assert err.isfinite().float().mean() > 0.5 and err.nanmedian() < 0.01
    assert multiview_error(gt * 1.1 ** torch.arange(2.0)[:, None, None], rays, cams).nanmedian() > 0.03


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
