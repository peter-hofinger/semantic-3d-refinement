"""Debugging ladder for monocular refinement: identity, synthetic overfit, contracts."""

import dataclasses

import numpy as np
import pytest
import torch

from semantic_3d_refinement.normals import BiniConfig, bini, close_cuts, crack_tips, grid_score, normals_from_depth
from semantic_3d_refinement.refine import (
    RefineConfig,
    depth_metrics,
    energy_maps,
    normal_term,
    normalize,
    points_from_rays,
    refine,
    second_differences,
)
from semantic_3d_refinement.synthetic import junction_scene, occlusion_scene, room_scene
from semantic_3d_refinement.views import fit_pinhole

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


IDENTITY_CFGS = [
    dataclasses.replace(CFG, iters=0, lambda_n=1.0),
    dataclasses.replace(CFG, lambda_bd=0.0),
    dataclasses.replace(CFG, lambda_bd=0.0, lambda_n=0.0),
]


@pytest.mark.parametrize("cfg", IDENTITY_CFGS)
def test_identity(cfg: RefineConfig) -> None:
    depth, rays, conf, _ = scene()
    normals = torch.from_numpy(room_scene(normal_noise=0.03)["normals"])
    depth_hat, points, delta, _ = refine(depth, rays, conf, cfg, normals)
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


def test_normal_term_vanishes_on_a_plane_given_its_normal() -> None:
    depth, rays, conf, gt = scene(outlier_frac=0.0, planes=TILTED_PLANE)
    normals = torch.from_numpy(room_scene(planes=TILTED_PLANE)["normals"])
    _, d0, w = normalize(depth, conf)
    scale, d_gt, _ = normalize(gt, conf)
    delta_gt = d_gt * depth.median() / scale - d0  # GT on the input's normalization
    maps = energy_maps(delta_gt, d0, w, CFG, normal_term(normals, rays, d0, CFG.edge_tol))
    assert maps["E_n"].max() < 1e-8
    assert energy_maps(torch.zeros_like(d0), d0, w, CFG, normal_term(normals, rays, d0, 0.02))["E_n"].mean() > 1e-3


def test_normal_term_recovers_a_plane_by_lbfgs() -> None:
    depth, rays, conf, gt = scene(outlier_frac=0.0, planes=TILTED_PLANE)
    normals = torch.from_numpy(room_scene(planes=TILTED_PLANE)["normals"])
    cfg = dataclasses.replace(CFG, lambda_bd=0.0, lambda_n=1.0, iters=500, edge_tol=1.0)  # 2% noise: no cuts
    depth_hat, *_ = refine(depth, rays, conf, cfg, normals)
    assert depth_metrics(depth_hat, rays, gt)["abs_rel"] < 2e-3 < depth_metrics(depth, rays, gt)["abs_rel"]


def test_bending_alone_converges_to_the_least_squares_plane_of_the_input() -> None:
    """L-BFGS steps are sums of gradients, which lie in the range of the Hessian: d0 keeps its null-space part."""
    depth, rays, conf, _ = scene(h=8, w=9, outlier_frac=0.0)
    cfg = dataclasses.replace(CFG, lambda_d=0.0, lambda_bd=1.0, iters=2000)
    delta = refine(depth.double(), rays.double(), conf.double(), cfg)[2]
    _, d0, _ = normalize(depth.double(), conf.double())
    v, u = torch.meshgrid(torch.arange(8.0), torch.arange(9.0), indexing="ij")
    basis = torch.stack([u.ravel(), v.ravel(), torch.ones(72)], 1).double()
    plane = (basis @ torch.linalg.lstsq(basis, d0.ravel()[:, None]).solution).reshape(8, 9)
    assert (d0 - plane).abs().max() > 0.1
    assert (d0 + delta - plane).abs().max() < 1e-3


def test_lbfgs_keeps_an_occlusion_edge_cut_on_the_contour() -> None:
    scene_ = {k: torch.from_numpy(v) for k, v in occlusion_scene().items()}
    gt, rays, depth = scene_["gt_depth"], scene_["rays"], scene_["depth"]
    cfg = dataclasses.replace(CFG, lambda_bd=0.0, lambda_n=0.05, iters=1000)
    d = refine(depth, rays, torch.ones_like(depth), cfg, scene_["normals"])[0]
    assert ((d - gt).abs() / gt).max() < 0.05 < ((depth - gt).abs() / gt).max()


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


def test_pinhole_fit_recovers_a_pinhole_and_removes_per_patch_ray_noise() -> None:
    _, rays, _, _ = scene(outlier_frac=0.0, h=98, w=126)
    rot = torch.linalg.matrix_exp(torch.tensor([[0.0, 0.0, 0.3], [0.0, 0.0, 0.0], [-0.3, 0.0, 0.0]])).double().numpy()
    exact = rays.double().numpy()
    assert np.abs(fit_pinhole(exact, rot) - exact).max() < 1e-5
    jitter = 2e-3 * torch.randn(7, 9, 2, generator=torch.Generator().manual_seed(0)).double().numpy()
    noisy = exact.copy()
    noisy[..., 3:5] += jitter.repeat(14, 0).repeat(14, 1) @ rot[:2, :2].T
    assert np.abs(fit_pinhole(noisy, rot)[..., 3:] - exact[..., 3:]).max() < 0.2 * np.abs(jitter).max()


def test_occlusion_edge_is_cut_on_one_clean_contour() -> None:
    scene_ = {k: torch.from_numpy(v) for k, v in occlusion_scene().items()}
    gt, rays, normals, front = scene_["gt_depth"], scene_["rays"], scene_["normals"], scene_["front"].bool()
    true_edge = front[:, 1:] != front[:, :-1]

    def off_edge_jumps(edges: str) -> int:
        cfg = BiniConfig(lambda_depth=100.0, edges=edges, device="cpu")
        d = bini(normals, rays, scene_["depth"], cfg)[0]
        jumps = (d.log()[:, 1:] - d.log()[:, :-1]).abs() > 0.05
        return int((jumps & ~true_edge).sum())

    assert off_edge_jumps("depth") == 0 < off_edge_jumps("bilateral")
    d = bini(normals, rays, scene_["depth"], BiniConfig(lambda_depth=100.0, edges="depth", device="cpu"))[0]
    assert ((d - gt).abs() / gt).max() < 0.05


def test_close_cuts_bridges_a_gap_along_the_edge_and_leaves_a_fading_edge_open() -> None:
    h, w = 12, 12
    cut_x, cut_y = np.zeros((h, w - 1), bool), np.zeros((h - 1, w), bool)
    step_x, step_y = np.full((h, w - 1), 0.1), np.full((h - 1, w), 0.1)
    cut_y[5, :4] = cut_y[5, 6:] = True  # a horizontal contour with a 2-edge gap ...
    step_y[5] = 2.0  # ... across which the prior still steps
    cut_x[1:3, 5] = True  # and a short contour ending in a smooth surface
    tips_before = crack_tips(cut_x, cut_y)
    assert tips_before.sum() == 4
    cut_x, cut_y = close_cuts([cut_x, cut_y], [step_x, step_y], budget=7.0)
    assert cut_y[5].all() and cut_x[1:3, 5].all() and cut_x.sum() == 2
    assert crack_tips(cut_x, cut_y).sum() == 2


def test_three_layers_meeting_in_a_t_do_not_leak() -> None:
    scene_ = {k: torch.from_numpy(v) for k, v in junction_scene().items()}
    gt = scene_["gt_depth"]
    cfg = BiniConfig(lambda_depth=100.0, edges="depth", device="cpu")
    d = bini(scene_["normals"], scene_["rays"], scene_["depth"], cfg)[0]
    off = ((d - gt).abs() / gt > 0.05).numpy()  # 138 pixels before the contours were closed
    edge = np.zeros_like(off)
    jx, jy = (gt[:, 1:] != gt[:, :-1]).numpy(), (gt[1:] != gt[:-1]).numpy()
    edge[:, 1:] |= jx
    edge[:, :-1] |= jx
    edge[1:] |= jy
    edge[:-1] |= jy
    assert off.sum() <= 5 and not (off & ~edge).any()
