"""Joint refinement: projection between views, the agreement residual, and the fused-cloud measures."""

import numpy as np
import torch
from scipy.spatial import cKDTree

from semantic_3d_refinement.multiview import (
    MultiviewConfig,
    Problem,
    Scene,
    bilinear,
    find_matches,
    grid_operator,
    local_planes,
    match_jacobian,
    refine_scene,
    residuals,
    scene_residuals,
    similarity,
    voxel_downsample,
)
from semantic_3d_refinement.normals import normals_from_depth
from semantic_3d_refinement.refine import NormalTerm, RefineConfig, normal_term

H, W = 48, 64
K = torch.tensor([[50.0, 0, 31.5], [0, 50.0, 23.5], [0, 0, 1]], dtype=torch.float64)
NORMAL = torch.tensor([0.2, -0.1, 1.0], dtype=torch.float64)


def rotation(angle: float) -> torch.Tensor:
    c, s = np.cos(angle), np.sin(angle)
    return torch.tensor([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=torch.float64)


def two_views(bump: float = 0.0) -> tuple[Scene, torch.Tensor]:
    """Two pinholes 0.3 apart facing the slanted plane n.x = 2; view 1's depth scaled by a smooth bump."""
    rotations = torch.stack([rotation(0.05), rotation(-0.1)])
    centers = torch.tensor([[0.0, 0, 0], [0.3, 0.05, 0]], dtype=torch.float64)
    v, u = torch.meshgrid(torch.arange(H, dtype=torch.float64), torch.arange(W, dtype=torch.float64), indexing="ij")
    cam = torch.stack([u, v, torch.ones_like(u)], -1) @ torch.linalg.inv(K).T
    dirs = torch.einsum("vij,hwj->vhwi", rotations, cam)
    origins = centers[:, None, None].expand_as(dirs)
    truth = (2 - origins @ NORMAL) / (dirs @ NORMAL)
    depth = truth.clone()
    depth[1] *= 1 + bump * torch.exp(-((u - 32) ** 2 + (v - 24) ** 2) / 200)
    rays = torch.cat([origins, dirs], -1)
    scene = Scene(depth.float(), rays.float(), torch.ones_like(depth).float(), K.float().expand(2, 3, 3),
                  rotations.float(), centers.float())  # fmt: skip
    return scene, truth


def test_bilinear_is_exact_on_affine_maps() -> None:
    v, u = torch.meshgrid(torch.arange(H, dtype=torch.float64), torch.arange(W, dtype=torch.float64), indexing="ij")
    d = torch.stack([1 + 0.01 * u + 0.02 * v, 2 - 0.01 * v])
    pu, pv = torch.tensor([3.3, 10.75, 40.5]), torch.tensor([7.2, 0.5, 30.9])
    value, _ = bilinear(d, torch.tensor([0, 1, 0]), pu.double(), pv.double())
    torch.testing.assert_close(
        value, torch.stack([1 + 0.01 * pu + 0.02 * pv, 2 - 0.01 * pv, 1 + 0.01 * pu + 0.02 * pv]).double().diagonal()
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
    for cfg, mv in [(RefineConfig(device="cpu"), MultiviewConfig(gn_iters=0)),
                    (RefineConfig(lambda_bd=0.0, device="cpu"), MultiviewConfig(lambda_mv=0.0))]:  # fmt: skip
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
        (d[v][center] / truth[v][center].float()).log().mean().item() for d in (scene.depth, depth) for v in (0, 1)
    ]
    assert abs(error[0]) < 1e-6 and 0.025 < error[1] < 0.035
    assert 0.3 * error[1] < error[2] < 0.7 * error[1] and 0.3 * error[1] < error[3] < 0.7 * error[1]


def test_similarity_recovers_a_known_transform() -> None:
    a = np.random.default_rng(0).normal(size=(20, 3))
    r = rotation(0.7).numpy() @ rotation(-0.2).numpy().T[[1, 2, 0]][:, [1, 2, 0]]
    s, rot, t = similarity(a, 2.5 * a @ r.T + [1, -2, 3])
    assert np.isclose(s, 2.5) and np.allclose(rot, r) and np.allclose(t, [1, -2, 3])


def test_planes_are_flat_and_layers_are_thick() -> None:
    g = np.random.default_rng(0).uniform(-0.1, 0.1, size=(20000, 2))
    plane = np.c_[g, 0.01 * g[:, 0]]
    grid = voxel_downsample(plane, 0.005)
    assert len(grid) < len(plane) and len(np.unique(np.floor(grid / 0.005), axis=0)) <= len(grid)
    normals, thickness = local_planes(cKDTree(grid), grid[:100], 24)
    assert thickness.max() < 1e-4 and (np.abs(normals @ [-0.01, 0, 1] / np.hypot(0.01, 1)) > 0.999).all()
    layered = voxel_downsample(np.r_[plane, plane + np.array([0, 0, 0.01])], 0.005)
    _, thick = local_planes(cKDTree(layered), layered[:100], 24)
    assert np.median(thick) > 0.003


def test_linear_operators_match_the_exact_jacobian() -> None:
    scene, _ = two_views(bump=0.03)
    scene = Scene(*(getattr(scene, f).double() for f in scene.__dataclass_fields__))
    scale = scene.depth.median()
    d0 = scale / scene.depth
    terms = [normal_term(normals_from_depth(scene.depth[i] * 1.01, scene.rays[i]), scene.rays[i], d0[i], 0.02)
             for i in range(2)]  # fmt: skip
    stacked = NormalTerm(
        [tuple(torch.stack([t.terms[a][k] for t in terms]) for k in range(4)) for a in range(2)],
        [tuple(torch.stack([t.weights[a][k] for t in terms]) for k in range(2)) for a in range(2)],
    )
    w = torch.rand(d0.shape, generator=torch.Generator().manual_seed(0), dtype=torch.float64) + 0.5
    p = Problem(scene, scale, d0, w, stacked, find_matches(scene, d0, scale, 0.05, 4))
    cfg, mv = RefineConfig(lambda_bd=3.0, lambda_n=0.5, lambda_delta=0.1, device="cpu"), MultiviewConfig(lambda_mv=2.0)
    delta = 0.01 * torch.randn(d0.shape, generator=torch.Generator().manual_seed(1), dtype=torch.float64)
    v = torch.randn(d0.shape, generator=torch.Generator().manual_seed(2), dtype=torch.float64)

    def grid_part(x: torch.Tensor) -> torch.Tensor:
        parts = scene_residuals(x, cfg, mv, p)
        parts.pop("E_mv")
        return torch.cat(list(parts.values()))

    _, exact = torch.func.jvp(grid_part, (delta,), (v,))
    torch.testing.assert_close(grid_operator(d0 + delta, cfg, p)(v), exact)
    _, transpose = torch.func.vjp(grid_operator(d0 + delta, cfg, p, squared=True), torch.zeros_like(d0))
    diag = transpose(torch.ones(len(exact), dtype=torch.float64))[0].flatten()
    for c in torch.randperm(d0.numel(), generator=torch.Generator().manual_seed(3))[:16]:
        unit = torch.zeros(d0.numel(), dtype=torch.float64)
        unit[c] = 1
        _, column = torch.func.jvp(grid_part, (delta,), (unit.view_as(d0),))
        torch.testing.assert_close(diag[c], column.square().sum())

    d = d0 + delta
    cols, vals = match_jacobian(d, scale, scene, p.matches)
    _, exact_mv = torch.func.jvp(lambda x: residuals(x, scale, scene, p.matches), (d,), (v,))
    torch.testing.assert_close((vals * v.flatten()[cols]).sum(0), exact_mv)
