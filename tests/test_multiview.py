"""Joint refinement: projection, the agreement residual, fused-cloud measures."""

import dataclasses

import numpy as np
import torch

from src.loaders.views import Views
from src.refinement.multiview import (
    MATCH_BORDER,
    JointRefiner,
    Problem,
    RefineConfig,
    Scene,
    bilinear,
    build_problem,
    consistency,
    energy_maps,
    find_matches,
    fit_pinhole,
    gauss_newton,
    geometry_from_scene,
    grid_operator,
    match_jacobian,
    normal_equations,
    occlusion_threshold,
    points_from_rays,
    project,
    refine_scene,
    residuals,
    robust,
    same_instance_edges,
    same_segment_matches,
    scene_energy,
    scene_from_geometry,
    scene_residuals,
)
from src.refinement.normals import NormalTerm, normal_term, normals_from_depth
from src.segmentation.base import Segmentation

H, W = 48, 64
K = torch.tensor([[50.0, 0, 31.5], [0, 50.0, 23.5], [0, 0, 1]], dtype=torch.float64)
NORMAL = torch.tensor([0.2, -0.1, 1.0], dtype=torch.float64)


def rotation(angle: float) -> torch.Tensor:
    c, s = np.cos(angle), np.sin(angle)
    return torch.tensor([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=torch.float64)


def two_views(bump: float = 0.0, noise: float = 0.0) -> tuple[Scene, torch.Tensor]:
    """Two pinholes 0.3 apart facing the plane n.x = 2; view 1 scaled by a bump.

    A plate at z = 1, x < -0.3 hides part of the plane from view 0. With noise (log,
    per pixel of view 1, as a backbone's), the gaps between the views fall into the two
    modes `occlusion_threshold` splits: same surface and occluded.
    """
    rotations = torch.stack([rotation(0.05), rotation(-0.1)])
    centers = torch.tensor([[0.0, 0, 0], [0.3, 0.05, 0]], dtype=torch.float64)
    v, u = torch.meshgrid(
        torch.arange(H, dtype=torch.float64),
        torch.arange(W, dtype=torch.float64),
        indexing="ij",
    )
    cam = torch.stack([u, v, torch.ones_like(u)], -1) @ torch.linalg.inv(K).T
    dirs = torch.einsum("vij,hwj->vhwi", rotations, cam)
    origins = centers[:, None, None].expand_as(dirs)
    truth = (2 - origins @ NORMAL) / (dirs @ NORMAL)
    plate = (1 - origins[..., 2]) / dirs[..., 2]
    hit = (origins[..., 0] + plate * dirs[..., 0] < -0.3) & (plate < truth)
    truth = torch.where(hit, plate, truth)
    depth = truth.clone()
    depth[1] *= 1 + bump * torch.exp(-((u - 32) ** 2 + (v - 24) ** 2) / 200)
    generator = torch.Generator().manual_seed(0)
    depth[1] *= torch.exp(noise * torch.randn(H, W, generator=generator))
    rays = torch.cat([origins, dirs], -1)
    scene = Scene(
        depth.float(),
        rays.float(),
        torch.ones_like(depth).float(),
        K.float().expand(2, 3, 3),
        rotations.float(),
        centers.float(),
    )
    return scene, truth


def test_bilinear_is_exact_on_affine_maps() -> None:
    v, u = torch.meshgrid(
        torch.arange(H, dtype=torch.float64),
        torch.arange(W, dtype=torch.float64),
        indexing="ij",
    )
    d = torch.stack([1 + 0.01 * u + 0.02 * v, 2 - 0.01 * v])
    pu, pv = torch.tensor([3.3, 10.75, 40.5]), torch.tensor([7.2, 0.5, 30.9])
    value, _ = bilinear(d, torch.tensor([0, 1, 0]), pu.double(), pv.double())
    torch.testing.assert_close(
        value,
        torch.stack(
            [1 + 0.01 * pu + 0.02 * pv, 2 - 0.01 * pv, 1 + 0.01 * pu + 0.02 * pv]
        )
        .double()
        .diagonal(),
    )


def test_consistent_views_have_zero_residual_and_a_bump_does_not() -> None:
    scene, _ = two_views()
    scale = scene.depth.median()
    m = find_matches(scene, scale / scene.depth, scale, 4, 0.05)
    assert len(m.src) > 0.5 * scene.depth.numel()
    assert residuals(scale / scene.depth, scale, scene, m).abs().max() < 1e-4
    bumped, _ = two_views(bump=0.03)
    assert residuals(scale / bumped.depth, scale, bumped, m).abs().max() > 0.02


def test_zero_iterations_and_zero_weight_return_the_input() -> None:
    scene, _ = two_views(bump=0.03)
    for cfg in [
        RefineConfig(lambda_n=0.0, gn_iters=0, device="cpu"),
        RefineConfig(lambda_n=0.0, lambda_mv=0.0, device="cpu"),
    ]:
        depth, _, _ = refine_scene(scene, cfg)
        assert (depth - scene.depth).abs().max() < 1e-6


def test_occlusion_threshold_splits_same_surface_from_occluded_gaps() -> None:
    rng = np.random.default_rng(0)
    same = 0.01 * rng.standard_normal(8000)  # one surface, 1 % apart
    hidden = rng.uniform(0.2, 1.5, 2000)  # behind another surface
    tol = occlusion_threshold(torch.from_numpy(np.concatenate([same, hidden])))
    assert np.abs(same).max() < tol < hidden.min()


def test_agreement_meets_halfway_between_two_views() -> None:
    scene, truth = two_views(bump=0.03, noise=0.01)
    cfg = RefineConfig(lambda_mv=100.0, lambda_n=0.0, lambda_delta=0.0, device="cpu")
    depth, _, _ = refine_scene(scene, cfg)
    scale = scene.depth.median()
    m = find_matches(scene, scale / scene.depth, scale, 4, 0.05)
    before = residuals(scale / scene.depth, scale, scene, m).abs().max()
    assert residuals(scale / depth, scale, scene, m).abs().max() < 0.1 * before
    center = (slice(20, 28), slice(28, 36))
    error = [
        (d[v][center] / truth[v][center].float()).log().mean().item()
        for d in (scene.depth, depth)
        for v in (0, 1)
    ]
    assert abs(error[0]) < 1e-6 and 0.025 < error[1] < 0.035
    assert (
        0.3 * error[1] < error[2] < 0.7 * error[1]
        and 0.3 * error[1] < error[3] < 0.7 * error[1]
    )


def test_geometry_lifts_every_pixel_where_the_rays_do() -> None:
    scene, _ = two_views(bump=0.03)
    geometry = geometry_from_scene(scene, scene.depth, scene.conf)
    lifted = points_from_rays(scene.depth.double(), scene.rays.double()).numpy()
    np.testing.assert_allclose(geometry.world_points(), lifted, atol=1e-5)
    back = scene_from_geometry(geometry)
    np.testing.assert_allclose(back.k.numpy(), scene.k.numpy(), atol=1e-4)
    np.testing.assert_allclose(back.rays.numpy(), scene.rays.numpy(), atol=1e-5)


def test_joint_refiner_is_refine_scene_on_the_pipeline_types() -> None:
    scene, _ = two_views(bump=0.03, noise=0.01)
    geometry = geometry_from_scene(scene, scene.depth, scene.conf)
    views = Views(
        "toy",
        ["0", "1"],
        np.zeros((2, H, W, 3), dtype=np.uint8),
        geometry.intrinsics,
        geometry.cam_to_world,
    )
    cfg = RefineConfig(lambda_mv=100.0, lambda_n=0.0, device="cpu")
    refiner = JointRefiner(cfg)
    refined = refiner.refine(views, geometry, Segmentation.empty(2, H, W))
    expected, _, _ = refine_scene(scene_from_geometry(geometry), cfg)
    np.testing.assert_allclose(refined.depth, expected.numpy(), rtol=1e-5)
    np.testing.assert_array_equal(refined.cam_to_world, geometry.cam_to_world)
    assert (np.abs(refined.depth / geometry.depth - 1) > 1e-3).any()


def test_pinhole_fit_recovers_a_pinhole_and_removes_per_patch_ray_noise() -> None:
    h, w = 98, 126  # 7 x 9 patches of 14 px, as Déjà View's
    v, u = np.mgrid[0:h, 0:w].astype(np.float64)
    k = np.array([[80.0, 0, 62.3], [0, 81.0, 48.7], [0, 0, 1]])
    rot = rotation(0.3).numpy()
    cam = np.stack([u, v, np.ones_like(u)], -1) @ np.linalg.inv(k).T
    exact = np.concatenate([np.full_like(cam, 0.5), cam @ rot.T], -1)
    fitted, k_fit = fit_pinhole(exact, rot)
    np.testing.assert_allclose(fitted, exact, atol=1e-9)
    np.testing.assert_allclose(k_fit, k, atol=1e-9)
    jitter = 2e-3 * np.random.default_rng(0).standard_normal((7, 9, 2))
    noisy = exact.copy()
    noisy[..., 3:5] += jitter.repeat(14, 0).repeat(14, 1) @ rot[:2, :2].T
    assert (
        np.abs(fit_pinhole(noisy, rot)[0][..., 3:] - exact[..., 3:]).max()
        < 0.2 * np.abs(jitter).max()
    )


def test_linear_operators_match_the_exact_jacobian() -> None:
    scene, _ = two_views(bump=0.03)
    scene = Scene(*(getattr(scene, f).double() for f in scene.__dataclass_fields__))
    scale = scene.depth.median()
    d0 = scale / scene.depth
    terms = [
        normal_term(
            normals_from_depth(scene.depth[i] * 1.01, scene.rays[i]),
            scene.rays[i],
            d0[i],
            0.02,
        )
        for i in range(2)
    ]
    stacked = NormalTerm(
        [
            tuple(torch.stack([t.terms[a][k] for t in terms]) for k in range(4))
            for a in range(2)
        ],
        [
            tuple(torch.stack([t.weights[a][k] for t in terms]) for k in range(2))
            for a in range(2)
        ],
    )
    w = (
        torch.rand(
            d0.shape, generator=torch.Generator().manual_seed(0), dtype=torch.float64
        )
        + 0.5
    )
    p = Problem(
        scene,
        scale,
        d0,
        w,
        stacked,
        find_matches(scene, d0, scale, 4, 0.05),
    )
    cfg = RefineConfig(
        lambda_mv=2.0,
        lambda_n=0.5,
        lambda_delta=0.1,
        huber=0.01,  # half the pixels of delta lie on Huber's linear branch
        device="cpu",
    )
    delta = 0.01 * torch.randn(
        d0.shape, generator=torch.Generator().manual_seed(1), dtype=torch.float64
    )
    v = torch.randn(
        d0.shape, generator=torch.Generator().manual_seed(2), dtype=torch.float64
    )

    def grid_part(x: torch.Tensor) -> torch.Tensor:
        parts = scene_residuals(x, cfg, p)
        parts.pop("E_mv")
        parts.pop("E_delta")
        return torch.cat(list(parts.values()))

    _, exact = torch.func.jvp(grid_part, (delta,), (v,))
    torch.testing.assert_close(grid_operator(d0 + delta, cfg, p)(v), exact)
    _, transpose = torch.func.vjp(
        grid_operator(d0 + delta, cfg, p, squared=True), torch.zeros_like(d0)
    )
    diag = transpose(torch.ones(len(exact), dtype=torch.float64))[0].flatten()
    for c in torch.randperm(d0.numel(), generator=torch.Generator().manual_seed(3))[
        :16
    ]:
        unit = torch.zeros(d0.numel(), dtype=torch.float64)
        unit[c] = 1
        _, column = torch.func.jvp(grid_part, (delta,), (unit.view_as(d0),))
        torch.testing.assert_close(diag[c], column.square().sum())

    d = d0 + delta
    cols, vals = match_jacobian(d, scale, scene, p.matches)
    _, exact_mv = torch.func.jvp(
        lambda x: residuals(x, scale, scene, p.matches), (d,), (v,)
    )
    torch.testing.assert_close((vals * v.flatten()[cols]).sum(0), exact_mv)


def halves() -> torch.Tensor:
    """Instance ids of both views: left half 3, right half 5."""
    ids = torch.full((2, H, W), 3, dtype=torch.long)
    ids[:, :, W // 2 :] = 5
    return ids


def test_edges_across_an_instance_boundary_are_cut() -> None:
    ids = halves()
    ids[:, :, :10] = -1
    x, y = same_instance_edges(ids)
    assert y.all()  # the boundary is vertical
    assert not x[..., W // 2 - 1].any() and not x[..., 9].any()
    assert x[..., :9].all()  # pixels without an instance count as one
    assert x.sum() == 2 * H * (W - 3)


def test_normals_do_not_join_two_instances() -> None:
    scene, _ = two_views()
    normals = torch.stack(
        [normals_from_depth(scene.depth[i], scene.rays[i]) for i in range(2)]
    )
    p = build_problem(scene, RefineConfig(device="cpu"), normals, halves())
    assert p.normals is not None
    (x_forward, x_backward), (y_forward, _) = p.normals.weights
    for weights in (x_forward, x_backward):
        assert (weights[..., W // 2 - 1] == 0).all()  # the wall and the table apart
        assert (weights[..., W // 2 :] > 0).float().mean() > 0.9
    assert (y_forward[..., W // 2] > 0).float().mean() > 0.9


def test_robust_residual_squares_to_huber_and_its_slope_is_exact() -> None:
    u = torch.linspace(-0.2, 0.2, 401, dtype=torch.float64)
    r, slope = robust(u, 0.05)
    huber = torch.where(u.abs() <= 0.05, u.square() / 2, 0.05 * u.abs() - 0.05**2 / 2)
    torch.testing.assert_close(r.square() / 2, huber)
    _, exact = torch.func.jvp(lambda x: robust(x, 0.05)[0], (u,), (torch.ones_like(u),))
    torch.testing.assert_close(slope, exact)


def test_consistency_is_the_share_of_views_that_agree_on_the_id_own_included() -> None:
    scene, _ = two_views()
    scale = scene.depth.median()
    d0 = scale / scene.depth
    same = torch.zeros(2, H, W, dtype=torch.long)
    u = consistency(scene, d0, scale, same)
    assert (u == 1).all()  # every view that sees the point agrees
    split = same.clone()
    split[1] = 7  # view 1 calls everything another object
    u = consistency(scene, d0, scale, split)
    seen = u[0] < 1
    # seen by both views, only its own agrees: 1 of 2; seen by view 0 alone: 1 of 1
    assert seen.sum() > 0.5 * H * W and (u[0][seen] == 0.5).all()
    assert (u[0][~seen] == 1).all() and (u[1] == 0.5).any()
    unmasked = torch.full((2, H, W), -1, dtype=torch.long)
    assert (consistency(scene, d0, scale, unmasked) == 1).all()


def test_anchor_weight_is_confidence_times_consistency() -> None:
    scene, _ = two_views()
    conf = torch.rand(2, H, W, generator=torch.Generator().manual_seed(0)) + 0.1
    scene = dataclasses.replace(scene, conf=conf)
    c = conf / conf.mean(dim=(1, 2), keepdim=True)
    scale = scene.depth.median()
    u = consistency(scene, scale / scene.depth, scale, halves())
    p = build_problem(scene, RefineConfig(lambda_n=0.0, device="cpu"), None, halves())
    torch.testing.assert_close(p.w, c * u)


def test_scaling_every_weight_leaves_the_minimizer_so_e_d_needs_none() -> None:
    scene, _ = two_views(bump=0.03, noise=0.01)
    scene = Scene(*(getattr(scene, f).double() for f in scene.__dataclass_fields__))
    normals = torch.stack(
        [normals_from_depth(scene.depth[i] * 1.01, scene.rays[i]) for i in range(2)]
    )
    # few steps: near the minimum E's decrease is at rounding level, and whether a
    # step is accepted there is not scale invariant in floating point
    cfg = RefineConfig(
        lambda_mv=2.0, lambda_n=0.01, lambda_delta=0.5, gn_iters=3, device="cpu"
    )
    p = build_problem(scene, cfg, normals, halves())
    k = 7.0
    scaled = dataclasses.replace(
        cfg,
        lambda_mv=k * cfg.lambda_mv,
        lambda_n=k * cfg.lambda_n,
        lambda_delta=k * cfg.lambda_delta,
    )
    x, log, _ = gauss_newton(p, cfg)
    x_k, log_k, _ = gauss_newton(dataclasses.replace(p, w=k * p.w), scaled)
    assert len(log) == len(log_k) == cfg.gn_iters + 1
    torch.testing.assert_close(x_k, x, rtol=1e-6, atol=1e-9)
    torch.testing.assert_close(log_k[-1]["E"], k * log[-1]["E"], rtol=1e-6, atol=0)


def test_matches_keep_the_border_margin() -> None:
    scene, _ = two_views()
    scale = scene.depth.median()
    d0 = scale / scene.depth
    m = find_matches(scene, d0, scale, 4, 0.05)
    depth = scale / d0.reshape(-1)[m.src]
    u, v, _ = project(m.offset + depth[:, None] * m.direction, scene.k[m.dst])
    b = MATCH_BORDER
    assert (u >= b).all() and (u <= W - 1 - b).all()
    assert (v >= b).all() and (v <= H - 1 - b).all()


def test_energy_maps_add_up_to_the_energy() -> None:
    scene, _ = two_views(bump=0.03)
    scene = Scene(*(getattr(scene, f).double() for f in scene.__dataclass_fields__))
    normals = torch.stack(
        [normals_from_depth(scene.depth[i] * 1.01, scene.rays[i]) for i in range(2)]
    )
    cfg = RefineConfig(lambda_delta=0.0, huber=0.01, device="cpu")
    p = build_problem(scene, cfg, normals, halves())
    x = 0.02 * torch.randn(
        p.d0.shape, generator=torch.Generator().manual_seed(0), dtype=torch.float64
    )
    terms = scene_energy(x, cfg, p)
    maps = energy_maps(p.d0 + x, cfg, p)
    for key, cost in maps.items():
        torch.testing.assert_close(cost.mean().item(), terms[key], rtol=1e-9, atol=0)


def test_semantic_solve_with_no_steps_returns_the_input() -> None:
    scene, _ = two_views(bump=0.03)
    depth, _, _ = refine_scene(
        scene,
        RefineConfig(lambda_n=0.0, gn_iters=0, device="cpu"),
        segments=halves(),
    )
    assert (depth - scene.depth).abs().max() < 1e-6


def test_normal_equations_with_instances_match_autograd() -> None:
    scene, _ = two_views(bump=0.03)
    scene = Scene(*(getattr(scene, f).double() for f in scene.__dataclass_fields__))
    cfg = RefineConfig(lambda_mv=2.0, lambda_n=0.0, lambda_delta=0.5, device="cpu")
    p = build_problem(scene, cfg, segments=halves())
    g = torch.Generator().manual_seed(0)
    x = 0.01 * torch.randn(p.d0.shape, generator=g, dtype=torch.float64)
    v = torch.randn(p.d0.shape, generator=g, dtype=torch.float64)

    def r(y: torch.Tensor) -> torch.Tensor:
        return torch.cat(list(scene_residuals(y, cfg, p).values()))

    gradient, hv, _ = normal_equations(x, cfg, p)
    exact = torch.func.grad(lambda y: 0.5 * r(y).square().sum())(x)
    torch.testing.assert_close(gradient, exact.flatten())
    _, jv = torch.func.jvp(r, (x,), (v,))
    _, vjp = torch.func.vjp(r, x)
    torch.testing.assert_close(
        hv(v.flatten()), vjp(jv)[0].flatten(), rtol=1e-6, atol=1e-8
    )


def test_normals_carry_a_correction_a_few_pixels_beyond_the_matches() -> None:
    scene, truth = two_views(noise=0.01)
    depth = scene.depth.clone()
    depth[1] *= 1.03  # view 1 too far everywhere: one wrong scale
    scene = dataclasses.replace(scene, depth=depth)
    scale = depth.median()
    m = find_matches(scene, scale / depth, scale, 4)  # as refine_scene matches
    seen = torch.zeros(2 * H * W, dtype=torch.bool)
    seen[m.src] = True
    seen = seen.view(2, H, W)[1]
    near = torch.nn.functional.max_pool2d(seen[None, None].float(), 5, 1, 2)[0, 0] > 0
    near &= ~seen  # unmatched, at most 2 px from a matched pixel
    assert near.sum() > 50
    normals = torch.stack(
        [normals_from_depth(truth[i].float(), scene.rays[i]) for i in range(2)]
    )
    errors = []
    # λ_n 1e-4 at Hypersim's focal of ~400 px is 64 times it at this toy's 50 px
    for lambda_n, given in ((0.0, None), (6.4e-3, normals)):
        cfg = RefineConfig(
            lambda_mv=100.0, lambda_n=lambda_n, gn_iters=20, cg_iters=100, device="cpu"
        )
        ids = torch.zeros(2, H, W, dtype=torch.long)
        out, _, _ = refine_scene(scene, cfg, given, ids)
        error = (out[1] / truth[1].float()).log()
        errors.append((error[seen].median().item(), error[near].median().item()))
    (seen_alone, near_alone), (seen_n, near_n) = errors
    assert seen_alone < 0.02 and seen_n < 0.02  # matched pixels meet the other view
    assert near_alone > 0.025  # δ alone: unmatched pixels keep most of the 3 %
    assert near_n < near_alone - 0.004  # E_n carries the correction a few pixels


def test_same_segment_keeps_only_matches_that_land_on_the_same_id() -> None:
    scene, _ = two_views()
    scale = scene.depth.median()
    d0 = scale / scene.depth
    m = find_matches(scene, d0, scale, 4, 0.05)
    same = torch.zeros(2, H, W, dtype=torch.long)
    assert len(same_segment_matches(m, scene, d0, scale, same).src) == len(m.src)
    split = same.clone()
    split[1] = 7  # view 1 is one other object: nothing may match across
    assert len(same_segment_matches(m, scene, d0, scale, split).src) == 0
    unmasked = torch.full((2, H, W), -1, dtype=torch.long)  # nothing to gate on
    assert len(same_segment_matches(m, scene, d0, scale, unmasked).src) == len(m.src)
    shifted = halves()
    shifted[1] = torch.roll(shifted[1], 20, dims=1)  # view 1's boundary elsewhere
    kept = same_segment_matches(m, scene, d0, scale, shifted)
    assert 0.2 * len(m.src) < len(kept.src) < 0.9 * len(m.src)
