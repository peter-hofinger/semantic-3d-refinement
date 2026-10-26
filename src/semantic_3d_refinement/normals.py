"""Detail from predicted normals: bilateral normal integration (BiNI, Cao et al. 2022) with a depth prior.

Own matrix-free GPU implementation of the BiNI energy on log depth, generalized from a pinhole K to the ray map:
for neighbors p, q = p + 1 along x or y, x_q lies on the plane of (x_p, n_p) iff dz = log|n_p.d_p| - log|n_p.d_q|.

  E(z) = sum_p w_p r+_p^2 + (1 - w_p) r-_p^2 + lambda_depth * sum_p (z_p - z0_p)^2,   z = log depth

w_p = sigmoid(k ((a_p dz-)^2 - (a_p dz+)^2)) lets the surface break on the side with the larger jump.
"""

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class BiniConfig:
    """BiNI settings; k and lambda_depth mean what they mean in the reference implementation."""

    k: float = 2.0
    lambda_depth: float = 1e-4
    iters: int = 150
    tol: float = 1e-4
    cg_iters: int = 5000
    cg_tol: float = 1e-3
    device: str = "cuda"

    def __post_init__(self) -> None:
        """Check ranges."""
        assert self.k >= 0
        assert self.lambda_depth > 0
        assert self.iters >= 0 and self.cg_iters > 0
        assert self.tol > 0 and self.cg_tol > 0


def normals_from_depth(depth: Tensor, rays: Tensor) -> Tensor:
    """Return unit normals of the lifted point map by central differences, facing the camera; edges replicated."""
    pts = rays[..., :3] + depth[..., None] * rays[..., 3:]
    n = torch.linalg.cross(pts[1:-1, 2:] - pts[1:-1, :-2], pts[2:, 1:-1] - pts[:-2, 1:-1])
    n = torch.nn.functional.pad(n.permute(2, 0, 1)[None], (1, 1, 1, 1), mode="replicate")[0].permute(1, 2, 0)
    n = n / n.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return torch.where(((n * rays[..., 3:]).sum(-1) > 0)[..., None], -n, n)


def neighbor_terms(normals: Tensor, rays: Tensor) -> tuple[Tensor, list[tuple[Tensor, Tensor, Tensor, Tensor]]]:
    """Per pixel a = s |n.d|, and per axis (x, y) the a, g of p's forward and q's backward residual on edge (p, q).

    A plane through x_p with normal n holds x_q iff z_q - z_p = log|n.d_p| - log|n.d_q|, exactly; the residual is
    a * (z_q - z_p + log|n.d_q| - log|n.d_p|). BiNI uses its first-order form n.(d_q - d_p) / n.d_p instead.
    s = 1 / median |d_q - d_p| (the focal length for a pinhole) keeps k and lambda_depth on BiNI's scale.
    """
    d = rays[..., 3:]
    dx, dy = d[:, 1:] - d[:, :-1], d[1:] - d[:-1]
    s = 1 / torch.cat([dx.norm(dim=-1).ravel(), dy.norm(dim=-1).ravel()]).median()

    def log_cos(n: Tensor, rays_d: Tensor) -> Tensor:
        return (n * rays_d).sum(-1).abs().clamp_min(1e-3).log()

    a = s * (normals * d).sum(-1).abs()
    terms = [
        (
            a[:, :-1],
            a[:, :-1] * (log_cos(normals[:, :-1], d[:, 1:]) - log_cos(normals[:, :-1], d[:, :-1])),
            a[:, 1:],
            a[:, 1:] * (log_cos(normals[:, 1:], d[:, 1:]) - log_cos(normals[:, 1:], d[:, :-1])),
        ),
        (
            a[:-1],
            a[:-1] * (log_cos(normals[:-1], d[1:]) - log_cos(normals[:-1], d[:-1])),
            a[1:],
            a[1:] * (log_cos(normals[1:], d[1:]) - log_cos(normals[1:], d[:-1])),
        ),
    ]
    return a, terms


def diffs(z: Tensor) -> tuple[Tensor, Tensor]:
    """Forward differences along x (H, W-1) and y (H-1, W)."""
    return z[:, 1:] - z[:, :-1], z[1:] - z[:-1]


def side_weights(z: Tensor, a: Tensor, k: float) -> list[tuple[Tensor, Tensor]]:
    """Per axis, the weight of p's forward residual and of q's backward residual on edge (p, q).

    A pixel with one neighbor on that axis puts all weight on it.
    """
    dx, dy = diffs(z)
    pad = torch.nn.functional.pad
    wx = torch.sigmoid(k * ((a * pad(dx, (1, 0))).square() - (a * pad(dx, (0, 1))).square()))
    wy = torch.sigmoid(k * ((a * pad(dy, (0, 0, 1, 0))).square() - (a * pad(dy, (0, 0, 0, 1))).square()))
    wx[:, 0], wx[:, -1], wy[0], wy[-1] = 1.0, 0.0, 1.0, 0.0
    return [(wx[:, :-1], 1 - wx[:, 1:]), (wy[:-1], 1 - wy[1:])]


def energy(z: Tensor, z0: Tensor, terms: list, weights: list, lam: float) -> float:
    """BiNI energy at log depth z for fixed weights."""
    e = lam * (z - z0).square().sum()
    for (af, gf, ab, gb), (wf, wb), dz in zip(terms, weights, diffs(z), strict=True):
        e = e + (wf * (af * dz + gf).square() + wb * (ab * dz + gb).square()).sum()
    return e.item()


def solve(z: Tensor, z0: Tensor, terms: list, weights: list, cfg: BiniConfig) -> Tensor:
    """Minimize the quadratic energy for fixed weights with Jacobi-preconditioned CG, warm-started at z."""
    cs, bs = [], []
    for (af, gf, ab, gb), (wf, wb) in zip(terms, weights, strict=True):
        cs.append(wf * af.square() + wb * ab.square())
        bs.append(wf * af * gf + wb * ab * gb)

    def scatter(ex: Tensor, ey: Tensor, out: Tensor) -> Tensor:
        out[:, 1:] += ex
        out[:, :-1] -= ex
        out[1:] += ey
        out[:-1] -= ey
        return out

    def apply(x: Tensor) -> Tensor:
        dx, dy = diffs(x)
        return scatter(cs[0] * dx, cs[1] * dy, cfg.lambda_depth * x)

    rhs = scatter(-bs[0], -bs[1], cfg.lambda_depth * z0)
    diag = torch.full_like(z, cfg.lambda_depth)
    diag[:, 1:] += cs[0]
    diag[:, :-1] += cs[0]
    diag[1:] += cs[1]
    diag[:-1] += cs[1]
    x = z.clone()
    r = rhs - apply(x)
    y = r / diag
    p, ry = y.clone(), (r * y).sum()
    stop = cfg.cg_tol * rhs.norm()
    for _ in range(cfg.cg_iters):
        if r.norm() <= stop:
            break
        ap = apply(p)
        alpha = ry / (p * ap).sum()
        x += alpha * p
        r -= alpha * ap
        y = r / diag
        ry, ry_old = (r * y).sum(), ry
        p = y + (ry / ry_old) * p
    return x


def bini(
    normals: Tensor, rays: Tensor, depth: Tensor, cfg: BiniConfig
) -> tuple[Tensor, list[tuple[Tensor, Tensor]], list[float]]:
    """Integrate normals (frame of the rays) into depth, anchored to a depth prior; return depth, weights, energies."""
    h, w = depth.shape
    assert normals.shape == (h, w, 3) and rays.shape == (h, w, 6), (normals.shape, rays.shape)
    assert normals.isfinite().all() and ((normals.norm(dim=-1) - 1).abs() < 1e-3).all()  # noqa: PLR2004
    assert depth.isfinite().all() and (depth > 0).all()
    z0 = depth.double().log()
    a, terms = neighbor_terms(normals.double(), rays.double())
    z = z0.clone()
    weights = side_weights(z, a, cfg.k)
    log = [energy(z, z0, terms, weights, cfg.lambda_depth)]
    for _ in range(cfg.iters):
        z = solve(z, z0, terms, weights, cfg)
        weights = side_weights(z, a, cfg.k)
        log.append(energy(z, z0, terms, weights, cfg.lambda_depth))
        if abs(log[-1] - log[-2]) <= cfg.tol * log[-2]:
            break
    depth_hat = z.exp().to(depth.dtype)
    assert depth_hat.shape == depth.shape and depth_hat.isfinite().all() and (depth_hat > 0).all()
    return depth_hat, weights, log
