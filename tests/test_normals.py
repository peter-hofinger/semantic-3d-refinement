"""E_n: BiNI's residuals vanish on the truth; the input's occlusion edges are cut."""

import numpy as np
import torch

from src.refinement.normals import (
    close_cuts,
    closer_neighbor_normals,
    crack_tips,
    cut_weights,
    neighbor_terms,
    normal_term,
)

H, W = 48, 64


def pinhole_rays(focal: float = 0.8 * W) -> torch.Tensor:
    """(H, W, 6) rays of a camera at the origin looking along +z, unit z directions."""
    v, u = torch.meshgrid(
        torch.arange(H, dtype=torch.float64),
        torch.arange(W, dtype=torch.float64),
        indexing="ij",
    )
    dirs = torch.stack(
        [(u + 0.5 - W / 2) / focal, (v + 0.5 - H / 2) / focal, torch.ones_like(u)], -1
    )
    return torch.cat([torch.zeros_like(dirs), dirs], -1)


def two_layers(ramp_passes: int = 2) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fronto-parallel planes at 2 and 3 split by a slanted edge, the step softened.

    Feed-forward depth draws occlusion edges as such ramps. Returns depth, the true
    front mask and the normals (all facing the camera, so they carry no edge).
    """
    v, u = torch.meshgrid(
        torch.arange(H, dtype=torch.float64),
        torch.arange(W, dtype=torch.float64),
        indexing="ij",
    )
    front = (u - W / 2) < 0.6 * (v - H / 2)
    z = torch.where(front, 2.0, 3.0).log()
    for _ in range(ramp_passes):
        z = torch.nn.functional.avg_pool2d(
            torch.nn.functional.pad(z[None, None], (1, 1, 1, 1), mode="replicate"),
            3,
            stride=1,
        )[0, 0]
    normals = torch.tensor([0.0, 0.0, -1.0], dtype=torch.float64).expand(H, W, 3)
    return z.exp(), front, normals


def test_normal_term_vanishes_on_a_plane_given_its_normal() -> None:
    rays = pinhole_rays()
    n = torch.tensor([0.2, -0.1, 1.0], dtype=torch.float64)
    n = n / n.norm()
    depth = 2 / (rays[..., 3:] @ n)  # the plane n.x = 2, z-depth (unit z rays)
    term = normal_term(-n.expand(H, W, 3).clone(), rays, depth.median() / depth, 0.02)
    z = depth.log()
    for (af, gf, ab, gb), (wf, wb), dz in zip(
        term.terms,
        term.weights,
        (z[:, 1:] - z[:, :-1], z[1:] - z[:-1]),
        strict=True,
    ):
        assert (wf > 0).all() and (wb > 0).all()  # a plane has no occlusion edge
        assert (af * dz + gf).abs().max() < 1e-9 and (ab * dz + gb).abs().max() < 1e-9
        bumped = dz + 0.01
        assert (af * bumped + gf).abs().min() > 1e-4


def test_the_input_s_occlusion_edge_is_cut_on_one_closed_contour() -> None:
    depth, front, normals = two_layers()
    rays = pinhole_rays()
    _, terms = neighbor_terms(normals, rays)
    (keep_x, _), (keep_y, _) = cut_weights(depth.log(), terms, 0.02)
    cut_x, cut_y = (keep_x == 0).numpy(), (keep_y == 0).numpy()
    edge_x = (front[:, 1:] != front[:, :-1]).numpy()
    near = np.zeros_like(edge_x)
    for shift in range(-2, 3):
        near |= np.roll(edge_x, shift, axis=1)
    assert cut_x.any() and not (cut_x & ~near).any()  # cuts only along the true edge
    assert (cut_x.sum(1) == 1).all()  # every row crosses it exactly once
    assert crack_tips(cut_x, cut_y).sum() == 0  # closed: no contour ends inside


def test_close_cuts_bridges_a_gap_along_the_edge_and_leaves_a_fading_edge_open() -> (
    None
):
    h, w = 12, 12
    cut_x, cut_y = np.zeros((h, w - 1), bool), np.zeros((h - 1, w), bool)
    step_x, step_y = np.full((h, w - 1), 0.1), np.full((h - 1, w), 0.1)
    cut_y[5, :4] = cut_y[5, 6:] = True  # a horizontal contour with a 2-edge gap ...
    step_y[5] = 2.0  # ... across which the prior still steps
    cut_x[1:3, 5] = True  # and a short contour ending in a smooth surface
    assert crack_tips(cut_x, cut_y).sum() == 4
    cut_x, cut_y = close_cuts([cut_x, cut_y], [step_x, step_y], budget=7.0)
    assert cut_y[5].all() and cut_x[1:3, 5].all() and cut_x.sum() == 2
    assert crack_tips(cut_x, cut_y).sum() == 2


def test_closer_neighbor_normals_are_exact_on_a_plane_and_ignore_a_jump() -> None:
    rays = pinhole_rays()
    n = torch.tensor([0.3, -0.2, -1.0], dtype=torch.float64)
    n = n / n.norm()
    depth = -2 / (rays[..., 3:] @ n)  # the plane n . x = -2, in front of the camera
    points = depth[..., None] * rays[..., 3:]
    torch.testing.assert_close(closer_neighbor_normals(points), n.expand(H, W, 3))
    stepped = depth.clone()
    stepped[:, W // 2 :] *= 1.5  # the parallel plane n . x = -3 behind the right half
    points = stepped[..., None] * rays[..., 3:]
    got = closer_neighbor_normals(points)
    # the columns next to the jump take their tangent from their own side
    torch.testing.assert_close(got, n.expand(H, W, 3), rtol=0, atol=1e-6)
