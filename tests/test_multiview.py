"""Joint refinement: projection, the agreement residual, fused-cloud measures."""

import dataclasses

import numpy as np
import torch

from src.loaders.views import Views
from src.refinement.multiview import (
    JointRefiner,
    MultiviewConfig,
    Problem,
    RefineConfig,
    Scene,
    SemanticConfig,
    bilinear,
    build_problem,
    find_matches,
    fit_pinhole,
    geometry_from_scene,
    grid_operator,
    match_jacobian,
    normal_equations,
    points_from_rays,
    refine_scene,
    residuals,
    same_id_stencils,
    same_segment_matches,
    scene_from_geometry,
    scene_residuals,
    segment_index,
)
from src.refinement.normals import NormalTerm, normal_term, normals_from_depth
from src.segmentation.base import Segmentation

H, W = 48, 64
K = torch.tensor([[50.0, 0, 31.5], [0, 50.0, 23.5], [0, 0, 1]], dtype=torch.float64)
NORMAL = torch.tensor([0.2, -0.1, 1.0], dtype=torch.float64)


def rotation(angle: float) -> torch.Tensor:
    c, s = np.cos(angle), np.sin(angle)
    return torch.tensor([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=torch.float64)


def two_views(bump: float = 0.0) -> tuple[Scene, torch.Tensor]:
    """Two pinholes 0.3 apart facing the plane n.x = 2; view 1 scaled by a bump."""
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
    depth = truth.clone()
    depth[1] *= 1 + bump * torch.exp(-((u - 32) ** 2 + (v - 24) ** 2) / 200)
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
    m = find_matches(scene, scale / scene.depth, scale, 0.05, 4)
    assert len(m.src) > 0.5 * scene.depth.numel()
    assert residuals(scale / scene.depth, scale, scene, m).abs().max() < 1e-4
    bumped, _ = two_views(bump=0.03)
    assert residuals(scale / bumped.depth, scale, bumped, m).abs().max() > 0.02


def test_zero_iterations_and_zero_weight_return_the_input() -> None:
    scene, _ = two_views(bump=0.03)
    for cfg, mv in [
        (RefineConfig(device="cpu"), MultiviewConfig(gn_iters=0)),
        (RefineConfig(lambda_bd=0.0, device="cpu"), MultiviewConfig(lambda_mv=0.0)),
    ]:
        depth, _, _ = refine_scene(scene, cfg, mv)
        assert (depth - scene.depth).abs().max() < 1e-6


def test_agreement_meets_halfway_between_two_views() -> None:
    scene, truth = two_views(bump=0.03)
    cfg = RefineConfig(lambda_bd=0.0, device="cpu")
    depth, _, _ = refine_scene(scene, cfg, MultiviewConfig(lambda_mv=100.0))
    scale = scene.depth.median()
    m = find_matches(scene, scale / scene.depth, scale, 0.05, 4)
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
    scene, _ = two_views(bump=0.03)
    geometry = geometry_from_scene(scene, scene.depth, scene.conf)
    views = Views(
        "toy",
        ["0", "1"],
        np.zeros((2, H, W, 3), dtype=np.uint8),
        geometry.intrinsics,
        geometry.cam_to_world,
    )
    cfg, mv = (
        RefineConfig(lambda_bd=0.0, device="cpu"),
        MultiviewConfig(lambda_mv=100.0),
    )
    refiner = JointRefiner(cfg, mv)
    refined = refiner.refine(views, geometry, Segmentation.empty(2, H, W))
    expected, _, _ = refine_scene(scene_from_geometry(geometry), cfg, mv)
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
    p = Problem(scene, scale, d0, w, stacked, find_matches(scene, d0, scale, 0.05, 4))
    cfg, mv = (
        RefineConfig(lambda_bd=3.0, lambda_n=0.5, lambda_delta=0.1, device="cpu"),
        MultiviewConfig(lambda_mv=2.0),
    )
    delta = 0.01 * torch.randn(
        d0.shape, generator=torch.Generator().manual_seed(1), dtype=torch.float64
    )
    v = torch.randn(
        d0.shape, generator=torch.Generator().manual_seed(2), dtype=torch.float64
    )

    def grid_part(x: torch.Tensor) -> torch.Tensor:
        parts = scene_residuals(x, cfg, mv, p)
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


def test_stencils_across_a_segment_boundary_are_cut() -> None:
    xx, yy, xy = same_id_stencils(halves())
    assert yy.all()
    assert not xx[..., W // 2 - 2].any() and not xx[..., W // 2 - 1].any()
    assert xx.sum() == 2 * H * (W - 4) and not xy[..., W // 2 - 1].any()
    index, count = segment_index(halves(), 10)
    assert count == 4 and set(index.unique().tolist()) == {0, 1, 2, 3}
    _, count = segment_index(halves(), H * W)
    assert count == 0


def test_semantic_solve_with_no_steps_returns_the_input() -> None:
    scene, _ = two_views(bump=0.03)
    semantic = SemanticConfig(cut_boundaries=True, affine=True)
    depth, _, counts = refine_scene(
        scene,
        RefineConfig(lambda_bd=1.0, lambda_delta=1.0, device="cpu"),
        MultiviewConfig(gn_iters=0),
        segments=halves(),
        semantic=semantic,
    )
    assert counts["segments"] == 4
    assert (depth - scene.depth).abs().max() < 1e-6


def test_normal_equations_with_segments_match_autograd() -> None:
    scene, _ = two_views(bump=0.03)
    scene = Scene(*(getattr(scene, f).double() for f in scene.__dataclass_fields__))
    cfg = RefineConfig(lambda_bd=3.0, lambda_delta=0.5, device="cpu")
    mv = MultiviewConfig(lambda_mv=2.0)
    semantic = SemanticConfig(cut_boundaries=True, affine=True)
    p = build_problem(scene, cfg, mv, segments=halves(), semantic=semantic)
    g = torch.Generator().manual_seed(0)
    n = p.d0.numel() + 2 * p.num_segments
    x = 0.01 * torch.randn(n, generator=g, dtype=torch.float64)
    v = torch.randn(n, generator=g, dtype=torch.float64)

    def r(y: torch.Tensor) -> torch.Tensor:
        return torch.cat(list(scene_residuals(y, cfg, mv, p).values()))

    gradient, hv, _ = normal_equations(x, cfg, mv, p)
    exact = torch.func.grad(lambda y: 0.5 * r(y).square().sum())(x)
    torch.testing.assert_close(gradient, exact)
    _, jv = torch.func.jvp(r, (x,), (v,))
    _, vjp = torch.func.vjp(r, x)
    torch.testing.assert_close(hv(v), vjp(jv)[0], rtol=1e-6, atol=1e-8)


def test_segment_scale_carries_over_pixels_no_other_view_sees() -> None:
    scene, truth = two_views()
    depth = scene.depth.clone()
    depth[1] *= 1.03  # view 1 too far everywhere: one wrong scale
    scene = dataclasses.replace(scene, depth=depth)
    scale = depth.median()
    m = find_matches(scene, scale / depth, scale, 0.05, 4)
    seen = torch.zeros(2 * H * W, dtype=torch.bool)
    seen[m.src] = True
    unseen = ~seen.view(2, H, W)[1]
    assert unseen.sum() > 50
    cfg = RefineConfig(lambda_bd=0.0, lambda_delta=10.0, device="cpu")
    mv = MultiviewConfig(lambda_mv=100.0, gn_iters=20, cg_iters=100)
    whole = torch.zeros(2, H, W, dtype=torch.long)
    moved = []
    for semantic in (SemanticConfig(), SemanticConfig(affine=True)):
        out, _, _ = refine_scene(scene, cfg, mv, segments=whole, semantic=semantic)
        error = (out[1] / truth[1].float()).log()
        moved.append((error[unseen].median().item(), error[~unseen].median().item()))
    # agreement meets halfway: matched pixels end ~1.5 % off, as both views move
    (unseen_d, seen_d), (unseen_ab, seen_ab) = moved
    assert 0.01 < seen_d < 0.02 and 0.01 < seen_ab < 0.02
    assert unseen_d > 0.028  # δ alone: pixels without a match keep the 3 %
    assert abs(unseen_ab - seen_ab) < 0.002  # with a, b they move with the segment


def test_same_segment_keeps_only_matches_that_land_on_the_same_id() -> None:
    scene, _ = two_views()
    scale = scene.depth.median()
    d0 = scale / scene.depth
    m = find_matches(scene, d0, scale, 0.05, 4)
    same = torch.zeros(2, H, W, dtype=torch.long)
    assert len(same_segment_matches(m, scene, d0, scale, same).src) == len(m.src)
    split = same.clone()
    split[1] = 7  # view 1 is one other object: nothing may match across
    assert len(same_segment_matches(m, scene, d0, scale, split).src) == 0
    shifted = halves()
    shifted[1] = torch.roll(shifted[1], 20, dims=1)  # view 1's boundary elsewhere
    kept = same_segment_matches(m, scene, d0, scale, shifted)
    assert 0.2 * len(m.src) < len(kept.src) < 0.9 * len(m.src)
