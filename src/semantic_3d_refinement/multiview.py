"""Joint refinement of every view of a scene: the monocular energy of each view plus cross-view agreement E_mv.

Cameras stay fixed: Déjà View's, each with the pinhole fitted to its rays (`views.fit_pinhole`). All views share one
normalization, d_i = s / D_i with s the scene's median depth, so a residual means the same in every view. Each pixel
p of view i is lifted, x = o_i + D_i(p) r_i(p), and projected into each neighboring view j (the views it overlaps
most, `neighbors`):

  E_mv = 1/N Σ_{i≠j} Σ_{p ∈ M_ij} ( log z_j(x) + log d_j(π_j(x)) - log s )²

z_j(x) is x's depth in camera j and d_j(π_j(x)) is j's inverse depth, bilinear at the projection, so the residual is
the log ratio of the two depths: zero where both views put the surface in the same place. N is the number of pixels
over all views, as for the per-view terms. The matches M_ij are fixed on the input, as BiNI's cut edges are, so E
stays smooth: x lands in j at least a pixel inside the border, the two depths agree within `mv_tol`, and so do j's
four interpolated pixels. This drops what j does not see (occluded or out of view) and interpolation across j's
occlusion edges.

  E = 1/V Σ_i E_i + λ_mv E_mv,   E_i the monocular energy of view i (`refine.energy_maps`), δ_i its residual.

E is a sum of squared residuals, so it is minimized by Gauss-Newton, as BiNI minimizes its normal term by CG
(Cao et al. 2022): each step solves the normal equations J^T J s = -J^T r by Jacobi-preconditioned CG. J is never
formed for the per-view terms, which are stencils on the pixel grid and linear in δ (E_d, E_bd) or in log depth
(E_n); E_mv's J has five entries per match, the source pixel and j's four interpolated pixels. On 20 views of
504 x 336 this takes 2-3 s where L-BFGS took a minute.

This is the E_mv of the method in `Claude.md` without segments: agreement is required wherever the input already
roughly agrees, not only within a segment.
"""

import argparse
import dataclasses
import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import yaml
from scipy.spatial import cKDTree
from torch import Tensor

from semantic_3d_refinement.evaluate import accuracy, scan_depth, score
from semantic_3d_refinement.normals import predicted_normals
from semantic_3d_refinement.refine import (
    NormalTerm,
    RefineConfig,
    normal_term,
    points_from_rays,
    variant_label,
)
from semantic_3d_refinement.views import check_inputs, fit_pinhole, git_hash

VOXEL = 0.005
PLANE_K = 24
SAMPLES = 200_000


@dataclass(frozen=True)
class Scene:
    """All views of one backbone export with pinhole rays: depth, conf (V, H, W), rays (V, H, W, 6), cameras."""

    depth: Tensor
    rays: Tensor
    conf: Tensor
    k: Tensor
    rotation: Tensor
    center: Tensor

    def to(self, device: torch.device) -> "Scene":
        """Move every tensor to the device."""
        return Scene(*(getattr(self, f).to(device) for f in self.__dataclass_fields__))


def load_scene(path: Path) -> Scene:
    """Every view of a Déjà View export, its rays replaced by the fitted pinhole (`rays: pinhole`)."""
    data = np.load(path)
    rays, ks = [], []
    for v in range(len(data["depth"])):
        ray_map = np.concatenate([data["ray_origins"][v], data["ray_directions"][v]], axis=-1).astype(np.float64)
        r, k = fit_pinhole(ray_map, data["cam_to_world"][v, :3, :3].astype(np.float64))
        rays.append(r)
        ks.append(k)
    rays = np.stack(rays)
    scene = Scene(
        *(torch.from_numpy(np.ascontiguousarray(a)).float() for a in (data["depth"], rays, data["confidence"])),
        torch.from_numpy(np.stack(ks)).float(),
        torch.from_numpy(data["cam_to_world"][:, :3, :3]).float(),
        torch.from_numpy(rays[:, 0, 0, :3]).float(),
    )
    for v in range(len(scene.depth)):
        check_inputs(scene.depth[v], scene.rays[v], scene.conf[v])
    return scene


@dataclass(frozen=True)
class Matches:
    """Fixed matches (view i, pixel p) -> view j: p's flat index in (V, H, W), j, and the ray in j's camera frame.

    x_j = offset + D_i(p) * direction is the lifted point in camera j's coordinates.
    """

    src: Tensor
    dst: Tensor
    offset: Tensor
    direction: Tensor


def project(x: Tensor, k: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Pixel coordinates u, v (pixel centers at integers) and depth z of camera-frame points under per-point K."""
    z = x[:, 2]
    return k[:, 0, 0] * x[:, 0] / z + k[:, 0, 2], k[:, 1, 1] * x[:, 1] / z + k[:, 1, 2], z


def corners(view: Tensor, u: Tensor, v: Tensor, h: int, w: int) -> tuple[Tensor, Tensor]:
    """Flat indices in (V, H, W) of the four pixels around each fractional pixel, and their bilinear weights.

    u, v are clamped to [0, W-1] x [0, H-1], so a step that leaves the image stays defined.
    """
    u, v = u.clamp(0, w - 1 - 1e-4), v.clamp(0, h - 1 - 1e-4)
    x0, y0 = u.detach().floor().long(), v.detach().floor().long()
    fu, fv = u - x0, v - y0
    base = (view * h + y0) * w + x0
    index = torch.stack([base, base + 1, base + w, base + w + 1])
    return index, torch.stack([(1 - fu) * (1 - fv), fu * (1 - fv), (1 - fu) * fv, fu * fv])


def bilinear(d: Tensor, view: Tensor, u: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
    """Sample d (V, H, W) at fractional pixels of the given views; also return the four corners' max/min ratio."""
    index, weights = corners(view, u, v, *d.shape[1:])
    c = d.reshape(-1)[index]
    return (weights * c).sum(0), c.amax(0) / c.amin(0)


def residuals(d: Tensor, scale: Tensor, scene: Scene, m: Matches) -> Tensor:
    """Return log z_j(x) + log d_j(π_j(x)) - log s per match, for normalized inverse depth d (V, H, W)."""
    depth = scale / d.reshape(-1)[m.src]
    u, v, z = project(m.offset + depth[:, None] * m.direction, scene.k[m.dst])
    dj, _ = bilinear(d, m.dst, u, v)
    return z.clamp_min(1e-6).log() + dj.clamp_min(1e-6).log() - scale.log()


def match_jacobian(d: Tensor, scale: Tensor, scene: Scene, m: Matches) -> tuple[Tensor, Tensor]:
    """Columns (5, M) and values (5, M) of the residuals' Jacobian in d: the source pixel and j's four corners.

    The corners stay those of the linearization point; their bilinear weights and the source's ray carry the rest.
    """
    flat = d.reshape(-1)
    di = flat[m.src].clone().requires_grad_()
    u, v, z = project(m.offset + (scale / di)[:, None] * m.direction, scene.k[m.dst])
    index, weights = corners(m.dst, u, v, *d.shape[1:])
    c = flat[index].clone().requires_grad_()
    r = z.clamp_min(1e-6).log() + (weights * c).sum(0).clamp_min(1e-6).log()
    g_src, g_corners = torch.autograd.grad(r.sum(), [di, c])
    return torch.cat([m.src[None], index]), torch.cat([g_src[None], g_corners])


@torch.no_grad()
def find_matches(scene: Scene, d0: Tensor, scale: Tensor, tol: float, neighbors: int) -> Matches:
    """Match each view with its neighbors wherever the input agrees within tol (log depth).

    A view's neighbors are the `neighbors` views it shares the most matches with; a pair counts if either side
    picks the other, and is matched both ways.
    """
    n, h, w = d0.shape
    pairs: dict[tuple[int, int], tuple[Tensor, Tensor, Tensor]] = {}
    flat = torch.arange(h * w, device=d0.device)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            world_to_j = scene.rotation[j].T
            offset = (scene.rays[i, ..., :3].reshape(-1, 3) - scene.center[j]) @ world_to_j.T
            direction = scene.rays[i, ..., 3:].reshape(-1, 3) @ world_to_j.T
            u, v, z = project(offset + (scale / d0[i].reshape(-1))[:, None] * direction, scene.k[j].expand(h * w, 3, 3))
            dj, spread = bilinear(d0, torch.full_like(flat, j), u, v)
            log_ratio = (z.clamp_min(1e-6) * dj / scale).log().abs()
            keep = (z > 0) & (u >= 1) & (u <= w - 2) & (v >= 1) & (v <= h - 2)
            keep &= (log_ratio < tol) & (spread.log() < tol)
            pairs[i, j] = (flat[keep], offset[keep], direction[keep])
    counts = torch.zeros(n, n)
    for (i, j), (pix, _, _) in pairs.items():
        counts[i, j] = len(pix)
    top = torch.zeros(n, n, dtype=torch.bool)
    top[torch.arange(n)[:, None], counts.topk(min(neighbors, n - 1), dim=1).indices] = True
    chosen = [(i, j) for (i, j) in pairs if top[i, j] or top[j, i]]
    m = Matches(
        torch.cat([i * h * w + pairs[i, j][0] for i, j in chosen]),
        torch.cat([torch.full_like(pairs[i, j][0], j) for i, j in chosen]),
        torch.cat([pairs[i, j][1] for i, j in chosen]),
        torch.cat([pairs[i, j][2] for i, j in chosen]),
    )
    assert (m.src // (h * w) != m.dst).all() and m.offset.isfinite().all() and m.direction.isfinite().all()
    return m


@dataclass(frozen=True)
class MultiviewConfig:
    """Weight of E_mv, the matches' tolerance (log depth) and neighbor count, and the Gauss-Newton / CG budget."""

    lambda_mv: float = 1.0
    mv_tol: float = 0.05
    neighbors: int = 4
    gn_iters: int = 3
    cg_iters: int = 200
    cg_tol: float = 1e-3

    def __post_init__(self) -> None:
        """Check ranges."""
        assert self.lambda_mv >= 0 and 0 < self.mv_tol < 1 and self.neighbors >= 1
        assert self.gn_iters >= 0 and self.cg_iters >= 1 and 0 < self.cg_tol < 1


MV_KEYS = {f.name for f in dataclasses.fields(MultiviewConfig)}


@dataclass(frozen=True)
class Problem:
    """What stays fixed during the solve: the scene, its scale s, d0 = s / D0, weights, normal terms and matches.

    Normal terms are stacked over views: per axis (x, y), (a_p, g_p, a_q, g_q) and (w+, w-) as in `refine.NormalTerm`.
    """

    scene: Scene
    scale: Tensor
    d0: Tensor
    w: Tensor
    normals: NormalTerm | None
    matches: Matches


def second_differences(d: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Return d_xx, d_yy, d_xy over the last two axes (`refine.second_differences`, batched over views)."""
    dxx = d[..., 2:] - 2 * d[..., 1:-1] + d[..., :-2]
    dyy = d[..., 2:, :] - 2 * d[..., 1:-1, :] + d[..., :-2, :]
    dxy = d[..., 1:, 1:] - d[..., 1:, :-1] - d[..., :-1, 1:] + d[..., :-1, :-1]
    return dxx, dyy, dxy


def scene_residuals(delta: Tensor, cfg: RefineConfig, mv: MultiviewConfig, p: Problem) -> dict[str, Tensor]:
    """Weighted residuals of each term, flat, such that E = Σ of all their squares / N (N pixels over all views).

    The same terms as `refine.energy_maps`, except that E_d is quadratic, w δ²/2: Huber's quadratic branch, which
    holds for |δ| < `huber` (0.05; the input's views disagree by about 1 %).
    """
    d = p.d0 + delta
    out = {"E_d": (0.5 * cfg.lambda_d * p.w).sqrt() * delta}
    if cfg.lambda_delta > 0:
        out["E_delta"] = math.sqrt(cfg.lambda_delta) * delta
    if cfg.lambda_bd > 0:
        dxx, dyy, dxy = second_differences(d)
        if p.normals is not None:
            (kx, _), (ky, _) = [(wf > 0, wb > 0) for wf, wb in p.normals.weights]
            dxx, dyy = dxx * (kx[..., :-1] & kx[..., 1:]), dyy * (ky[..., :-1, :] & ky[..., 1:, :])
            dxy = dxy * (kx[..., :-1, :] & kx[..., 1:, :] & ky[..., :-1] & ky[..., 1:])
        root = math.sqrt(cfg.lambda_bd)
        out["E_bd"] = torch.cat([root * dxx.flatten(), root * dyy.flatten(), math.sqrt(2) * root * dxy.flatten()])
    if cfg.lambda_n > 0:
        assert p.normals is not None, "E_n needs normals"
        z = -d.clamp_min(1e-6).log()
        dzs = (z[..., 1:] - z[..., :-1], z[..., 1:, :] - z[..., :-1, :])
        parts = []
        for (af, gf, ab, gb), (wf, wb), dz in zip(p.normals.terms, p.normals.weights, dzs, strict=True):
            parts += [(cfg.lambda_n * wf).sqrt() * (af * dz + gf), (cfg.lambda_n * wb).sqrt() * (ab * dz + gb)]
        out["E_n"] = torch.cat([x.flatten() for x in parts])
    if mv.lambda_mv > 0:
        out["E_mv"] = math.sqrt(mv.lambda_mv) * residuals(d, p.scale, p.scene, p.matches)
    return {k: v.flatten() for k, v in out.items()}


def scene_energy(delta: Tensor, cfg: RefineConfig, mv: MultiviewConfig, p: Problem) -> dict[str, float]:
    """E and each weighted term's share of it."""
    terms = {k: (v.square().sum() / p.d0.numel()).item() for k, v in scene_residuals(delta, cfg, mv, p).items()}
    e = sum(terms.values())
    assert math.isfinite(e), terms
    return {"E": e} | terms


def grid_operator(d: Tensor, cfg: RefineConfig, p: Problem, *, squared: bool = False) -> Callable[[Tensor], Tensor]:
    """J v of the per-view residuals (all but E_mv) at d, in `scene_residuals` order; squared: with J's entries squared.

    E_d and E_bd are linear in δ; E_n is linear in z = -log d, so dz moves by -(v_q/d_q - v_p/d_p). With squared
    entries, J∘J applied to v and transposed onto the residuals' ones gives the diagonal of J^T J.
    """
    power = 2 if squared else 1
    a_d = (0.5 * cfg.lambda_d * p.w).sqrt() ** power
    root_bd = math.sqrt(cfg.lambda_bd) ** power
    masks = None
    if p.normals is not None:
        (kx, _), (ky, _) = [(wf > 0, wb > 0) for wf, wb in p.normals.weights]
        masks = (kx[..., :-1] & kx[..., 1:], ky[..., :-1, :] & ky[..., 1:, :],
                 kx[..., :-1, :] & kx[..., 1:, :] & ky[..., :-1] & ky[..., 1:])  # fmt: skip
    inv_d = (1 / d) ** power
    sign = 1 if squared else -1

    def apply(v: Tensor) -> Tensor:
        out = [a_d * v]
        if cfg.lambda_delta > 0:
            out.append(cfg.lambda_delta ** (power / 2) * v)
        if cfg.lambda_bd > 0:
            c = 4 if squared else -2
            dxx = v[..., 2:] + c * v[..., 1:-1] + v[..., :-2]
            dyy = v[..., 2:, :] + c * v[..., 1:-1, :] + v[..., :-2, :]
            dxy = v[..., 1:, 1:] + sign * v[..., 1:, :-1] + sign * v[..., :-1, 1:] + v[..., :-1, :-1]
            if masks is not None:
                dxx, dyy, dxy = dxx * masks[0], dyy * masks[1], dxy * masks[2]
            out += [root_bd * dxx, root_bd * dyy, 2 ** (power / 2) * root_bd * dxy]
        if cfg.lambda_n > 0:
            assert p.normals is not None
            q = v * inv_d
            dqs = (q[..., 1:] + sign * q[..., :-1], q[..., 1:, :] + sign * q[..., :-1, :])
            for (af, _, ab, _), (wf, wb), dq in zip(p.normals.terms, p.normals.weights, dqs, strict=True):
                for a, wt in ((af, wf), (ab, wb)):
                    coef = cfg.lambda_n * wt * a.square() if squared else -(cfg.lambda_n * wt).sqrt() * a
                    out.append(coef * dq)
        return torch.cat([x.flatten() for x in out])

    return apply


def conjugate_gradient(
    hv: Callable[[Tensor], Tensor], b: Tensor, precondition: Tensor, iters: int, tol: float
) -> tuple[Tensor, int]:
    """Solve H x = b, H symmetric positive definite given as a product, by CG with a diagonal preconditioner.

    Stops once the residual norm falls below tol times b's (checked every 10 iterations); returns x and the count.
    """
    x, r = torch.zeros_like(b), b.clone()
    z = precondition * r
    p, rz = z.clone(), (r * z).sum()
    stop = (tol * b.norm()) ** 2
    if rz == 0:
        return x, 0
    for k in range(1, iters + 1):
        hp = hv(p)
        alpha = rz / (p * hp).sum()
        x, r = x + alpha * p, r - alpha * hp
        if k % 10 == 0 and r.square().sum() < stop:
            return x, k
        z = precondition * r
        rz_new = (r * z).sum()
        p, rz = z + (rz_new / rz) * p, rz_new
    return x, iters


def gauss_newton_step(delta: Tensor, cfg: RefineConfig, mv: MultiviewConfig, p: Problem) -> tuple[Tensor, int]:
    """Solve J^T J s = -J^T r at delta by Jacobi-preconditioned CG; return s and the CG iterations used."""
    d = p.d0 + delta
    parts = scene_residuals(delta, cfg, mv, p)
    r_mv = parts.pop("E_mv", None)
    r_grid = torch.cat(list(parts.values()))
    grid = grid_operator(d, cfg, p)
    _, grid_t = torch.func.vjp(grid, torch.zeros_like(d))
    _, grid_sq_t = torch.func.vjp(grid_operator(d, cfg, p, squared=True), torch.zeros_like(d))
    diag = grid_sq_t(torch.ones_like(r_grid))[0].flatten()
    gradient = grid_t(r_grid)[0].flatten()
    cols = vals = None
    if r_mv is not None:
        cols, vals = match_jacobian(d, p.scale, p.scene, p.matches)
        vals = math.sqrt(mv.lambda_mv) * vals
        diag = diag.index_add(0, cols.flatten(), vals.square().flatten())
        gradient = gradient.index_add(0, cols.flatten(), (vals * r_mv).flatten())

    def hv(v: Tensor) -> Tensor:
        out = grid_t(grid(v.view_as(d)))[0].flatten()
        if cols is not None and vals is not None:
            jv = (vals * v[cols]).sum(0)
            out = out.index_add(0, cols.flatten(), (vals * jv).flatten())
        return out

    step, used = conjugate_gradient(hv, -gradient, 1 / diag.clamp_min(1e-12), mv.cg_iters, mv.cg_tol)
    return step.view_as(d), used


def refine_scene(
    scene: Scene, cfg: RefineConfig, mv: MultiviewConfig, normals: Tensor | None = None
) -> tuple[Tensor, list[dict[str, float]], dict[str, int]]:
    """Gauss-Newton on all views' residuals at once; return depth (V, H, W), energy log and counts.

    Each step linearizes every residual at the current δ and solves the normal equations J^T J s = -J^T r by
    Jacobi-preconditioned CG (`gauss_newton_step`), halving s until E decreases and no depth grows tenfold
    (the logs clamp d at 1e-6, so E alone would not see a step that crosses d = 0). Normals (V, H, W, 3), unit, in the
    frame of the rays, enable E_n and cut E_bd on the input's occlusion contours, as in `refine.refine`.
    """
    assert normals is not None or cfg.lambda_n == 0, "E_n needs normals"
    scale = scene.depth.median()
    d0 = scale / scene.depth
    w = scene.conf / scene.conf.mean(dim=(1, 2), keepdim=True)
    terms = None
    if normals is not None:
        per_view = [normal_term(normals[i], scene.rays[i], d0[i], cfg.edge_tol) for i in range(len(d0))]
        terms = NormalTerm(
            [tuple(torch.stack([t.terms[a][k] for t in per_view]) for k in range(4)) for a in range(2)],
            [tuple(torch.stack([t.weights[a][k] for t in per_view]) for k in range(2)) for a in range(2)],
        )
    p = Problem(scene, scale, d0, w, terms, find_matches(scene, d0, scale, mv.mv_tol, mv.neighbors))

    delta = torch.zeros_like(d0)
    log, cg_total = [scene_energy(delta, cfg, mv, p)], 0
    for _ in range(mv.gn_iters):
        step, used = gauss_newton_step(delta, cfg, mv, p)
        cg_total += used
        for _ in range(8):
            positive = bool((d0 + delta + step > 0.1 * d0).all())
            trial = scene_energy(delta + step, cfg, mv, p) if positive else {"E": math.inf}
            if trial["E"] < log[-1]["E"]:
                delta = delta + step
                log.append(trial)
                break
            step = step / 2
        else:
            break
    depth_hat = scale / (d0 + delta)
    assert depth_hat.shape == scene.depth.shape and depth_hat.isfinite().all() and (depth_hat > 0).all()
    assert log[-1]["E"] <= log[0]["E"] + 1e-9, (log[0], log[-1])
    return depth_hat, log, {"matches": len(p.matches.src), "cg_iterations": cg_total}


def similarity(a: np.ndarray, b: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Scale, rotation and translation minimizing Σ ‖s R a + t - b‖² over corresponding points (Umeyama 1991)."""
    assert a.shape == b.shape and a.shape[1] == 3 and len(a) >= 3, (a.shape, b.shape)  # noqa: PLR2004
    ma, mb = a.mean(0), b.mean(0)
    u, sv, vt = np.linalg.svd((b - mb).T @ (a - ma) / len(a))
    sign = np.diag([1.0, 1.0, np.sign(np.linalg.det(u @ vt))])
    rotation = u @ sign @ vt
    scale = float((sv * np.diag(sign)).sum() / ((a - ma) ** 2).sum(1).mean())
    return scale, rotation, mb - scale * rotation @ ma


def voxel_downsample(points: np.ndarray, size: float) -> np.ndarray:
    """One point per occupied voxel of the given size, the mean of its points."""
    keys = np.floor(points / size).astype(np.int64)
    _, inverse, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    out = np.zeros((len(counts), 3))
    np.add.at(out, inverse.ravel(), points)
    return out / counts[:, None]


def local_planes(tree: cKDTree, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """PCA over each query's k nearest points: the unit normal and the RMS distance to the fitted plane."""
    _, idx = tree.query(queries, k=k, workers=-1)
    nb = tree.data[idx]
    nb = nb - nb.mean(1, keepdims=True)
    values, vectors = np.linalg.eigh(np.einsum("nki,nkj->nij", nb, nb) / k)
    return vectors[..., 0], np.sqrt(values[:, 0].clip(0))


@dataclass(frozen=True)
class Truth:
    """A scene's scan and ETH3D's cameras, the scan's depth in each view, and the similarity from Déjà View's frame.

    `sample` holds the flat indices of 200k pixels (all views) the scan saw, drawn once, scored for every variant.
    """

    tree: cKDTree
    grid: cKDTree
    k: np.ndarray
    cam_to_world: np.ndarray
    depth: np.ndarray
    align: tuple[float, np.ndarray, np.ndarray]
    sample: np.ndarray


def load_truth(paths: dict, scene: Scene) -> Truth:
    """Scan (and its 5 mm grid), cameras and per-view scan depth; Déjà View's centers aligned to ETH3D's."""
    scan, cams = np.load(paths["scan"]), np.load(paths["gt_views"])
    scan_gpu = torch.from_numpy(scan).to(scene.depth.device)
    h, w = scene.depth.shape[1:]
    depth = np.stack([
        scan_depth(scan_gpu, torch.from_numpy(k).to(scan_gpu), torch.from_numpy(c).to(scan_gpu), h, w).cpu().numpy()
        for k, c in zip(cams["intrinsics"], cams["cam_to_world"], strict=True)
    ])  # fmt: skip
    centers = cams["cam_to_world"][:, :3, 3].astype(np.float64)
    align = similarity(scene.center.cpu().numpy().astype(np.float64), centers)
    grid = cKDTree(voxel_downsample(scan.astype(np.float64), VOXEL))
    k, c2w = cams["intrinsics"].astype(np.float64), cams["cam_to_world"].astype(np.float64)
    seen = np.flatnonzero(depth > 0)
    sample = np.sort(np.random.default_rng(0).choice(seen, min(SAMPLES, len(seen)), replace=False))
    return Truth(cKDTree(scan), grid, k, c2w, depth.astype(np.float64), align, sample)


def fused_metrics(depth: np.ndarray, rays: np.ndarray, truth: Truth, tolerances_cm: list[int | float]) -> dict:
    """Accuracy of all views' points (where the scan saw the pixel), and the fidelity of the surface they form.

    The cloud is mapped by the camera similarity, the same for every variant. Accuracy on the fixed pixel sample
    (`Truth.sample`): exact distances, but 200k of about 3M, since a nearest-neighbor query for points far from the
    scan costs ~40 µs each (2 min per variant for all); the fractions' standard error is under 0.12 points. Fidelity
    on a 5 mm voxel grid of all points, so point
    density does not change it: at 200k sampled grid points, the RMS distance of the 24 nearest grid points to their
    plane (thickness: layering and noise, no ground truth needed), and the angle between that plane and the scan's.
    """
    s, rotation, t = truth.align
    points = s * points_from_rays(torch.from_numpy(depth), torch.from_numpy(rays)).numpy() @ rotation.T + t
    dist = truth.tree.query(points.reshape(-1, 3)[truth.sample], workers=-1)[0]
    out = {f"fused_{k}": v for k, v in accuracy(dist, tolerances_cm).items()}
    points = points[truth.depth > 0]
    grid = voxel_downsample(points, VOXEL)
    queries = grid[np.random.default_rng(0).choice(len(grid), min(SAMPLES, len(grid)), replace=False)]
    normals, thickness = local_planes(cKDTree(grid), queries, PLANE_K)
    scan_normals, _ = local_planes(truth.grid, truth.grid.data[truth.grid.query(queries, workers=-1)[1]], PLANE_K)
    angle = np.degrees(np.arccos(np.abs((normals * scan_normals).sum(-1)).clip(0, 1)))
    return out | {"thickness_mm": float(1000 * np.median(thickness)), "normal_angle_deg": float(np.median(angle))}


def scan_fidelity(truth: Truth) -> dict[str, float]:
    """Return the scan's own thickness on its grid, the floor of `thickness_mm`."""
    queries = truth.grid.data[np.random.default_rng(0).choice(truth.grid.n, min(SAMPLES, truth.grid.n), replace=False)]
    return {"thickness_mm": float(1000 * np.median(local_planes(truth.grid, queries, PLANE_K)[1]))}


def scene_label(cfg: RefineConfig, mv: MultiviewConfig) -> str:
    """Name the active terms, e.g. "E_d + E_mv"; E_d alone is the input."""
    label = variant_label(cfg) + (" + E_mv" if mv.lambda_mv > 0 else "")
    return label + (" (= input)" if label == "E_d" else "")


def run_variant(
    scene: Scene, normals: Tensor, cfg: RefineConfig, mv: MultiviewConfig, problem: tuple[Truth, Matches, dict]
) -> tuple[dict, np.ndarray]:
    """Refine all views under one configuration and score it: fused cloud, per-view protocol, disagreement."""
    truth, matches, raw = problem
    t0 = time.perf_counter()
    depth_hat, log, counts = refine_scene(scene, cfg, mv, normals)
    if depth_hat.is_cuda:
        torch.cuda.synchronize()
    seconds = time.perf_counter() - t0
    scale = scene.depth.median()
    disagreement = 100 * residuals(scale / depth_hat, scale, scene, matches).abs().median().item()
    depth = depth_hat.cpu().numpy().astype(np.float64)
    record = {
        "label": scene_label(cfg, mv),
        "seconds": seconds,
        "gn_steps": len(log) - 1,
        **counts,
        "energy_initial": log[0],
        "energy_final": log[-1],
        "disagreement_pct": disagreement,
        **fused_metrics(depth, scene.rays.cpu().numpy().astype(np.float64), truth, raw["tolerances_cm"]),
        "per_view": {},
    }
    for v in raw["views"]:
        dist, _ = score(depth[v], truth.depth[v], truth.k[v], truth.cam_to_world[v], truth.tree)
        record["per_view"][v] = accuracy(dist[np.isfinite(dist)], raw["tolerances_cm"])
    return record, depth_hat.cpu().numpy()


def main() -> None:
    """Refine every scene jointly per variant: `uv run python -m semantic_3d_refinement.multiview --cfg <cfg>`."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=Path, required=True)
    cfg_path: Path = parser.parse_args().cfg
    raw = yaml.safe_load(cfg_path.read_text())
    base = RefineConfig(**raw["refine"])
    device = torch.device(base.device if base.device != "cuda" or torch.cuda.is_available() else "cpu")
    base = dataclasses.replace(base, device=device.type)

    out = Path(raw["output_dir"]) / f"{cfg_path.stem}-{time.strftime('%Y%m%d-%H%M%S')}"
    out.mkdir(parents=True)
    scenes, arrays = {}, {}
    for name, paths in raw["scenes"].items():
        t0 = time.perf_counter()
        scene = load_scene(Path(paths["input"])).to(device)
        normals = torch.stack([
            predicted_normals({**raw, **paths, "view": v}, {"depth": scene.depth[v], "rays": scene.rays[v]})[0]
            for v in range(len(scene.depth))
        ])  # fmt: skip
        truth = load_truth(paths, scene)
        scale = scene.depth.median()
        base_mv = MultiviewConfig(**raw["multiview"])
        matches = find_matches(scene, scale / scene.depth, scale, base_mv.mv_tol, base_mv.neighbors)
        s, rotation, t = truth.align
        centers = s * scene.center.cpu().numpy() @ rotation.T + t
        center_residual = np.linalg.norm(centers - truth.cam_to_world[:, :3, 3], axis=1)
        scenes[name] = {
            "setup_seconds": time.perf_counter() - t0,
            "align": {"scale": s, "rotation": rotation.tolist(), "translation": t.tolist()},
            "center_residual_cm": float(100 * np.median(center_residual)),
            "scan": scan_fidelity(truth),
            "variants": [],
        }
        refine_scene(scene, base, dataclasses.replace(base_mv, gn_iters=1, cg_iters=1), normals)
        for i, o in enumerate(raw["variants"]):
            mv = MultiviewConfig(**{**raw["multiview"], **{k: v for k, v in o.items() if k in MV_KEYS}})
            cfg = dataclasses.replace(base, **{k: v for k, v in o.items() if k not in MV_KEYS})
            record, depth = run_variant(scene, normals, cfg, mv, (truth, matches, raw))
            scenes[name]["variants"].append(record)
            arrays[f"{name}_{i}_depth"] = depth
            keys = ("fused_median_cm", "thickness_mm", "normal_angle_deg", "disagreement_pct", "seconds")
            print(name, record["label"], {k: round(record[k], 3) for k in keys}, flush=True)  # noqa: T201

    summary = {"config": raw, "resolved": dataclasses.asdict(base), "git": git_hash(), "scenes": scenes}
    (out / "run.yaml").write_text(yaml.safe_dump(summary, sort_keys=False, allow_unicode=True))
    np.savez_compressed(out / "results.npz", **arrays)
    print(f"wrote {out}")  # noqa: T201


if __name__ == "__main__":
    main()
