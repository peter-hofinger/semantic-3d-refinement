"""Dense bundle adjustment (issue #82): Jacobians, recovery of known poses, identity."""

from pathlib import Path

import numpy as np
import torch

from src.geometry import Geometry
from src.refinement.bundle import (
    BundleConfig,
    Matches,
    State,
    adjust,
    free_weights,
    grid,
    refined_geometry,
    residuals,
    same_object,
    so3_exp,
    so3_log,
    start,
    window_pairs,
)
from src.refinement.multiview import Scene, pinhole_rays

H, W, STRIDE = 24, 32, 4


def surface(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """A height field z = f(x, y) the cameras look down at (+z)."""
    return 3.0 + 0.3 * torch.sin(2 * x) * torch.cos(1.5 * y)


def make_scene(rotations: torch.Tensor, centers: torch.Tensor) -> Scene:
    """Views of the height field: depth by ray marching each pixel's ray onto it."""
    v = len(centers)
    k = torch.tensor(
        [[30.0, 0, (W - 1) / 2], [0, 30.0, (H - 1) / 2], [0, 0, 1]]
    ).double()
    k = k.expand(v, 3, 3).clone()
    rays = pinhole_rays(k, rotations, centers, H, W)
    o, d = rays[..., :3], rays[..., 3:]
    t = torch.full(rays.shape[:3], 3.0, dtype=torch.float64)
    for _ in range(50):
        p = o + t[..., None] * d
        t = t - (p[..., 2] - surface(p[..., 0], p[..., 1])) / d[..., 2]
    depth = t  # directions have unit camera z, so t is the z-depth
    return Scene(depth, rays, torch.ones_like(depth), k, rotations, centers)


def cameras(v: int = 4) -> tuple[torch.Tensor, torch.Tensor]:
    """Cameras on a small circle, slightly turned, looking down +z."""
    angles = torch.arange(v, dtype=torch.float64) * 2 * np.pi / v
    centers = torch.stack(
        [0.3 * torch.cos(angles), 0.3 * torch.sin(angles), 0 * angles], -1
    )
    omega = torch.stack(
        [0.05 * torch.sin(angles), -0.05 * torch.cos(angles), 0.1 * angles], -1
    )
    return so3_exp(omega), centers


def config(**kw: float) -> BundleConfig:
    """A config for the synthetic scene."""
    base = {
        "root": Path(), "views": "", "backbone": "", "segmentation": "", "train": [],
        "held_out": 0, "predict": [], "stride": STRIDE, "window": 2,
        "iterations": 30, "robust": 1e3, "sigma": 1.0, "prior": 1.0, "damping": 1e-9,
        "pose_prior": 0.0,
        "models": {}, "hidden": 8, "steps": 0, "lr": 0.0, "seeds": [],
    }  # fmt: skip
    return BundleConfig(**(base | kw))


def true_matches(truth: Scene) -> Matches:
    """Every ordered pair of views, each grid pixel's exact match from the truth."""
    v = len(truth.center)
    pairs = torch.tensor([(i, j) for i in range(v) for j in range(v) if i != j])
    state = start(truth, STRIDE)
    p = grid(H, W, STRIDE, torch.device("cpu"))
    shape = (len(pairs), *p.shape[:2])
    empty = Matches(
        pairs,
        torch.zeros(*shape, 2),
        torch.zeros(shape),
        torch.zeros(shape, dtype=torch.bool),
    )
    r = residuals(truth, state, empty, p)[0]
    target = r.reshape(len(pairs), *p.shape[:2], 2)  # the projection itself
    return Matches(pairs, target, torch.zeros(target.shape[:3], dtype=torch.float64),
                   torch.zeros(target.shape[:3], dtype=torch.bool))  # fmt: skip


def test_jacobians_match_autograd() -> None:
    truth = make_scene(*cameras(3))
    matches = true_matches(truth)
    matches = Matches(
        matches.pairs[:2], matches.target[:2] + 0.3, matches.fb[:2], matches.same[:2]
    )
    state = start(truth, STRIDE)
    p = grid(H, W, STRIDE, torch.device("cpu"))
    _, j_wi, j_ti, j_wj, j_tj, j_rho = residuals(truth, state, matches, p)
    n = 5  # a match of pair 0 at grid pixel n
    i, j = matches.pairs[0].tolist()

    def f(x: torch.Tensor) -> torch.Tensor:
        views = range(len(state.center))
        turn = {i: so3_exp(x[:3]), j: so3_exp(x[6:9])}
        move = {i: x[3:6], j: x[9:12]}
        eye, zero = torch.eye(3, dtype=x.dtype), torch.zeros(3, dtype=x.dtype)
        rot = torch.stack([turn.get(v, eye) @ state.rotation[v] for v in views])
        cen = torch.stack([state.center[v] + move.get(v, zero) for v in views])
        bump = torch.zeros(state.rho.numel(), dtype=x.dtype)
        bump = bump + x[12] * (
            torch.arange(state.rho.numel()) == i * state.rho[0].numel() + n
        )
        moved = State(rot, cen, state.rho + bump.reshape(state.rho.shape))
        return residuals(truth, moved, matches, p)[0][n]

    jac = torch.autograd.functional.jacobian(
        f, torch.full((13,), 1e-12, dtype=torch.float64)
    )
    ours = torch.cat([j_wi[n], j_ti[n], j_wj[n], j_tj[n], j_rho[n][:, None]], -1)
    np.testing.assert_allclose(ours.numpy(), jac.numpy(), rtol=1e-6, atol=1e-6)


def test_bundle_adjustment_recovers_perturbed_poses() -> None:
    rot, cen = cameras(4)
    truth = make_scene(rot, cen)
    matches = true_matches(truth)
    bump = torch.tensor(
        [[0, 0, 0], [0.01, -0.02, 0.01], [-0.015, 0.01, 0.02], [0.02, 0.01, -0.01]]
    ).double()
    turned = so3_exp(bump) @ rot
    moved = Scene(
        truth.depth, truth.rays, truth.conf, truth.k, turned, cen + 0.02 * bump
    )
    p = grid(H, W, STRIDE, torch.device("cpu"))
    before = residuals(moved, start(moved, STRIDE), matches, p)[0].norm(dim=-1)
    state = adjust(
        moved, matches, torch.ones(matches.fb.shape, dtype=torch.float64), config()
    )
    after = residuals(moved, state, matches, p)[0].norm(dim=-1)
    assert before.nanmean() > 0.5
    assert after.nanmean() < 1e-3
    np.testing.assert_allclose(state.rotation.numpy(), rot.numpy(), atol=1e-4)
    np.testing.assert_allclose(state.center.numpy(), cen.numpy(), atol=1e-4)


def test_no_steps_return_the_backbone_bit_for_bit() -> None:
    truth = make_scene(*cameras(3))
    geometry = Geometry(
        truth.depth.float().numpy(),
        np.ones((3, H, W), np.float32),
        truth.k.float().numpy(),
        np.tile(np.eye(4, dtype=np.float32), (3, 1, 1)),
    )
    out = refined_geometry(truth, geometry, start(truth, STRIDE), STRIDE)
    np.testing.assert_array_equal(out.depth, truth.depth.float().numpy())
    np.testing.assert_allclose(
        out.cam_to_world[:, :3, :3], truth.rotation.float().numpy()
    )
    np.testing.assert_allclose(
        out.ray_directions, truth.rays[..., 3:].float().numpy(), atol=1e-6
    )


def test_weights_and_same_object_flags() -> None:
    fb = torch.tensor([0.0, 1.0, torch.inf]).double()
    m = Matches(
        torch.zeros(1, 2, dtype=torch.long),
        torch.zeros(1, 1, 3, 2),
        fb[None, None],
        fb[None, None] > 0,
    )
    np.testing.assert_allclose(
        free_weights(m, 1.0)[0, 0].numpy(), [1.0, np.exp(-1.0), 0.0]
    )
    ids = torch.zeros(2, H, W, dtype=torch.long)
    ids[1, :, W // 2 :] = 5
    pairs = torch.tensor([[0, 1]])
    p = grid(H, W, STRIDE, torch.device("cpu"))
    same = same_object(ids, pairs, p[None].clone(), STRIDE)
    assert same[0, :, : W // (2 * STRIDE)].all()
    assert not same[0, :, -1].any()


def test_rotation_log_inverts_the_exponential() -> None:
    omega = torch.tensor([[0.3, -0.2, 0.5], [1e-9, 0, 0], [0, 2.5, 0.1]]).double()
    np.testing.assert_allclose(
        so3_log(so3_exp(omega)).numpy(), omega.numpy(), atol=1e-8
    )


def test_window_pairs_link_each_view_to_its_path_neighbours() -> None:
    pairs = window_pairs(5, 2).tolist()
    assert len(pairs) == 14
    assert [0, 2] in pairs
    assert [4, 2] in pairs
    assert [0, 3] not in pairs
