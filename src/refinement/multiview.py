"""Joint refinement of all views by Gauss-Newton: the method in `Claude.md`.

Cameras stay fixed: Déjà View's, each with the pinhole fitted to its rays
(`fit_pinhole`). All views share one normalization, d_i = s / D_i with s the scene's
median depth, so a residual means the same in every view. Inverse depth is refined per
pixel, d_i(p) = d0_i(p) + δ_i(p), minimizing

  E = λ_d E_d + λ_mv E_mv + λ_n E_n + λ_δ ‖δ‖²,   per-pixel terms as means.

- E_d = w h(d - d0): Huber's cost h (`robust`) anchors to the prediction, weighted by
  w = confidence (mean one per view) times the consistency u: the share of the other
  views a pixel lands in whose pixel there carries the same instance id
  (`consistency`). Where views disagree on the object, at bleeding edges and flying
  pixels, the anchor lets go.
- E_mv: each pixel p of view i is lifted, x = o_i + D_i(p) r_i(p), and projected into
  each neighboring view j (the `neighbors` views it overlaps most); the residual is
  log z_j(x) + log d_j(π_j(x)) - log s, the log ratio of the two depths, d_j bilinear
  at the projection. The matches are fixed on the input, as BiNI's cut edges are, so E
  stays smooth: x lands in j at least a pixel inside the border, the two depths agree,
  so do j's four interpolated pixels, and j's nearest pixel lies in p's segment
  (`same_segment_matches`; pixels without one are not gated). Agree means a gap below
  Otsu's threshold on the scene's gaps (`occlusion_threshold`): they fall into a mode of
  same-surface noise and one of occlusions, and the threshold splits them, per scene.
- E_n: BiNI's normal term on DSINE's normals (`normals.py`), in log depth, on the
  edges between two pixels of one linked instance (`same_instance_edges`; pixels without
  one count as one), and cut on the input's occlusion contours: a wall's pixels
  do not shape the table in front of it. E_n is the only shape term; a smoothness
  term on top of it changed the depth by a median 0.006-0.016 % on Hypersim.
- λ_δ ‖δ‖² anchors every pixel to the prediction, also where E_d lets go: about 8-10 %
  of the pixels on Hypersim have E_d's weight 0 (no other view agrees on their
  instance), and 1,300-2,700 of them per scene have no normal edge or match either, so
  without it the normal equations are singular there and the solve stalls. It also
  limits how far E_n carries a correction from matched pixels into unmatched ones: a few
  pixels (`test_normals_carry_a_correction_a_few_pixels_beyond_the_matches`). The
  proposal's per-segment scale and shift of inverse depth (a, b) would carry it across
  the instance, but on Hypersim the F-score at 5 cm is equal or better without them.

E is a sum of squared residuals, so it is minimized by Gauss-Newton, as BiNI minimizes
its normal term by CG (Cao et al. 2022): each step solves the normal equations J^T J s =
-J^T r by Jacobi-preconditioned CG. J is never formed for the per-view terms, which are
stencils on the pixel grid, robust in δ (E_d) or linear in log depth (E_n); E_mv's J has
five entries per match, the source pixel and j's four interpolated pixels. On 20 views
of 504 x 378 a solve takes about 4 s on an RTX 3080.
"""

import logging
import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import Tensor

from src.geometry import Geometry
from src.loaders.views import Views
from src.refinement.normals import NormalTerm, dsine_normals, normal_term, to_world
from src.segmentation.base import Segmentation

log_info = logging.getLogger(__name__).info


@dataclass(frozen=True)
class RefineConfig:
    """Energy weights, the robust and gating constants, the Gauss-Newton / CG budget.

    Inverse depth is normalized by the scene's median depth, so huber and the tolerances
    are relative: huber 0.05 is a 5 % change of inverse depth.
    """

    lambda_d: float = 1.0
    lambda_mv: float = 1.0
    # BiNI's residuals are log-depth steps times the focal length
    lambda_n: float = 1e-4
    lambda_delta: float = 1.0
    huber: float = 0.05  # E_d's Huber threshold on d - d0
    edge_tol: float = 0.02  # E_n's occlusion cut (log depth, `normals.cut_weights`)
    neighbors: int = 4  # views each view is matched with, the ones it overlaps most
    gn_iters: int = 10
    cg_iters: int = 20
    cg_tol: float = 1e-3
    device: str = "cuda"

    def __post_init__(self) -> None:
        """Check ranges."""
        assert min(self.lambda_d, self.lambda_mv, self.lambda_n, self.lambda_delta) >= 0
        assert self.huber > 0 and self.edge_tol > 0
        assert self.neighbors >= 1
        assert self.gn_iters >= 0 and self.cg_iters >= 1 and 0 < self.cg_tol < 1


def points_from_rays(depth: Tensor, rays: Tensor) -> Tensor:
    """Lift depth along rays: x = origin + depth * direction."""
    return rays[..., :3] + depth[..., None] * rays[..., 3:]


def check_inputs(depth: Tensor, rays: Tensor, conf: Tensor) -> None:
    """Assert shapes, dtypes and ranges of a single view."""
    assert depth.ndim == 2, depth.shape  # noqa: PLR2004
    h, w = depth.shape
    assert h >= 3 and w >= 3, depth.shape  # noqa: PLR2004
    assert rays.shape == (h, w, 6), rays.shape
    assert conf.shape == (h, w), conf.shape
    assert depth.dtype == rays.dtype == conf.dtype and depth.dtype.is_floating_point
    assert depth.isfinite().all() and (depth > 0).all()
    assert rays.isfinite().all()
    assert conf.isfinite().all() and (conf >= 0).all() and (conf > 0).any()


def fit_pinhole(
    rays: np.ndarray, rotation: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Replace a ray map by its best-fitting pinhole, same camera center and rotation.

    Least squares for fx, cx and fy, cy on the camera-frame directions; the directions
    keep unit z in the camera. Returns the rays and K, which maps camera coordinates to
    pixel indices (pixel centers at integers). Déjà View's rays deviate from it by
    0.6-0.9 px (median) on ScanNet++, in patch-sized steps that show as a 14 px grid in
    every normal map lifted through them, and by up to 2.7 px on ETH3D office (1.8 px
    even with a free rotation, so the rays themselves are less pinhole-like there).
    """
    h, w = rays.shape[:2]
    d = rays[..., 3:] @ rotation
    x, y = d[..., 0] / d[..., 2], d[..., 1] / d[..., 2]
    v, u = np.mgrid[0:h, 0:w].astype(np.float64)
    fx, cx = np.polyfit(x.ravel(), u.ravel(), 1)
    fy, cy = np.polyfit(y.ravel(), v.ravel(), 1)
    residual = np.hypot(fx * x + cx - u, fy * y + cy - v)
    assert np.median(residual) < 3, np.median(residual)  # noqa: PLR2004
    dirs = (
        np.stack([(u - cx) / fx, (v - cy) / fy, np.ones_like(u)], axis=-1) @ rotation.T
    )
    origin = np.broadcast_to(rays[..., :3].reshape(-1, 3).mean(0), dirs.shape)
    k = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    return np.concatenate([origin, dirs], axis=-1), k


@dataclass(frozen=True)
class Scene:
    """All views, pinhole rays: depth, conf (V, H, W), rays (V, H, W, 6), cameras."""

    depth: Tensor
    rays: Tensor
    conf: Tensor
    k: Tensor
    rotation: Tensor
    center: Tensor

    def to(self, device: torch.device) -> "Scene":
        """Move every tensor to the device."""
        return Scene(*(getattr(self, f).to(device) for f in self.__dataclass_fields__))


def scene_from_geometry(geometry: Geometry) -> Scene:
    """All views of a backbone's geometry, each ray map replaced by its pinhole fit."""
    assert geometry.ray_origins is not None and geometry.ray_directions is not None
    rays, ks = [], []
    for v in range(len(geometry.depth)):
        ray_map = np.concatenate(
            [geometry.ray_origins[v], geometry.ray_directions[v]], axis=-1
        ).astype(np.float64)
        rotation = geometry.cam_to_world[v, :3, :3].astype(np.float64)
        r, k = fit_pinhole(ray_map, rotation)
        rays.append(r)
        ks.append(k)
    rays = np.stack(rays)
    scene = Scene(
        *(
            torch.from_numpy(np.ascontiguousarray(a)).float()
            for a in (geometry.depth, rays, geometry.confidence)
        ),
        torch.from_numpy(np.stack(ks)).float(),
        torch.from_numpy(geometry.cam_to_world[:, :3, :3]).float(),
        torch.from_numpy(rays[:, 0, 0, :3]).float(),
    )
    for v in range(len(scene.depth)):
        check_inputs(scene.depth[v], scene.rays[v], scene.conf[v])
    return scene


def geometry_from_scene(scene: Scene, depth: Tensor, confidence: Tensor) -> Geometry:
    """Return the pipeline's Geometry of the scene's pinholes with new depth.

    Geometry's intrinsics put pixel centers at +0.5 (COLMAP), the scene's at integers,
    so cx and cy move by half a pixel; cam_to_world keeps the backbone's rotation and
    the fitted center, and the rays are the pinhole's.
    """
    assert depth.shape == scene.depth.shape and (depth > 0).all(), depth.shape
    k = scene.k.double().cpu().numpy()
    k[:, :2, 2] += 0.5
    cam_to_world = np.tile(np.eye(4), (len(k), 1, 1))
    cam_to_world[:, :3, :3] = scene.rotation.double().cpu().numpy()
    cam_to_world[:, :3, 3] = scene.center.double().cpu().numpy()
    rays = scene.rays.cpu().numpy()
    return Geometry(
        depth=depth.cpu().numpy().astype(np.float32),
        confidence=confidence.cpu().numpy().astype(np.float32),
        intrinsics=k.astype(np.float32),
        cam_to_world=cam_to_world.astype(np.float32),
        ray_origins=np.ascontiguousarray(rays[..., :3]),
        ray_directions=np.ascontiguousarray(rays[..., 3:]),
    )


@dataclass(frozen=True)
class Matches:
    """Fixed matches (view i, pixel p) -> view j: p's flat index, j, and the ray in j.

    x_j = offset + D_i(p) * direction is the lifted point in camera j's coordinates.
    """

    src: Tensor
    dst: Tensor
    offset: Tensor
    direction: Tensor


def project(x: Tensor, k: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Pixel u, v (centers at integers) and depth z of camera points, per-point K."""
    z = x[:, 2]
    return (
        k[:, 0, 0] * x[:, 0] / z + k[:, 0, 2],
        k[:, 1, 1] * x[:, 1] / z + k[:, 1, 2],
        z,
    )


def corners(
    view: Tensor, u: Tensor, v: Tensor, h: int, w: int
) -> tuple[Tensor, Tensor]:
    """Flat indices of the four pixels around each fractional pixel, bilinear weights.

    u, v are clamped to [0, W-1] x [0, H-1], so a step that leaves the image stays
    defined.
    """
    u, v = u.clamp(0, w - 1 - 1e-4), v.clamp(0, h - 1 - 1e-4)
    x0, y0 = u.detach().floor().long(), v.detach().floor().long()
    fu, fv = u - x0, v - y0
    base = (view * h + y0) * w + x0
    index = torch.stack([base, base + 1, base + w, base + w + 1])
    return index, torch.stack(
        [(1 - fu) * (1 - fv), fu * (1 - fv), (1 - fu) * fv, fu * fv]
    )


def bilinear(d: Tensor, view: Tensor, u: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
    """Sample d (V, H, W) at fractional pixels; also the four corners' max/min ratio."""
    index, weights = corners(view, u, v, *d.shape[1:])
    c = d.reshape(-1)[index]
    return (weights * c).sum(0), c.amax(0) / c.amin(0)


def ray_in(scene: Scene, i: int, j: int, pixels: Tensor) -> tuple[Tensor, Tensor]:
    """Rays of view i's (flat) pixels in camera j's coordinates: offset, direction."""
    world_to_j = scene.rotation[j].T
    rays = scene.rays[i].reshape(-1, 6)[pixels]
    return (rays[:, :3] - scene.center[j]) @ world_to_j.T, rays[:, 3:] @ world_to_j.T


def residuals(d: Tensor, scale: Tensor, scene: Scene, m: Matches) -> Tensor:
    """Return log z_j(x) + log d_j(π_j(x)) - log s per match, d (V, H, W)."""
    depth = scale / d.reshape(-1)[m.src]
    u, v, z = project(m.offset + depth[:, None] * m.direction, scene.k[m.dst])
    dj, _ = bilinear(d, m.dst, u, v)
    return z.clamp_min(1e-6).log() + dj.clamp_min(1e-6).log() - scale.log()


def match_jacobian(
    d: Tensor, scale: Tensor, scene: Scene, m: Matches
) -> tuple[Tensor, Tensor]:
    """Columns and values (5, M) of the Jacobian in d: source pixel, j's four corners.

    The corners stay those of the linearization point; their bilinear weights and the
    source's ray carry the rest.
    """
    flat = d.reshape(-1)
    di = flat[m.src].clone().requires_grad_()
    u, v, z = project(m.offset + (scale / di)[:, None] * m.direction, scene.k[m.dst])
    index, weights = corners(m.dst, u, v, *d.shape[1:])
    c = flat[index].clone().requires_grad_()
    r = z.clamp_min(1e-6).log() + (weights * c).sum(0).clamp_min(1e-6).log()
    g_src, g_corners = torch.autograd.grad(r.sum(), [di, c])
    return torch.cat([m.src[None], index]), torch.cat([g_src[None], g_corners])


def occlusion_threshold(gaps: Tensor) -> float:
    """Otsu's threshold on log |gap|: the split of least within-class variance.

    The depth gaps between overlapping views fall into two modes: the same surface seen
    twice (about 1 % on Hypersim) and a surface hidden behind another (about 50 %).
    Otsu (1979) splits them where the two classes' log gaps vary least, exactly over the
    sorted gaps, with no bins. It assumes both modes are there, as they are in every
    scene with an occlusion.
    """
    x = gaps.abs().clamp_min(1e-6).log().double().sort().values
    n = len(x)
    assert n >= 2, n  # noqa: PLR2004
    below = torch.arange(1, n, device=x.device, dtype=x.dtype)
    total = x.cumsum(0)
    mean_below = total[:-1] / below
    mean_above = (total[-1] - total[:-1]) / (n - below)
    k = int((below * (n - below) * (mean_below - mean_above).square()).argmax())
    return math.exp(0.5 * (x[k] + x[k + 1]).item())


@torch.no_grad()
def find_matches(
    scene: Scene, d0: Tensor, scale: Tensor, neighbors: int, tol: float | None = None
) -> Matches:
    """Match each view with its neighbors where the input agrees within tol (log).

    tol defaults to `occlusion_threshold` over the gaps of all in-image candidates.
    Both the gap and the spread of j's four interpolated pixels must stay below it. A
    view's neighbors are the `neighbors` views it shares the most matches with; a pair
    counts if either side picks the other, and is matched both ways.
    """
    n, h, w = d0.shape
    flat = torch.arange(h * w, device=d0.device)
    candidates: dict[tuple[int, int], tuple[Tensor, Tensor, Tensor]] = {}
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            offset, direction = ray_in(scene, i, j, flat)
            u, v, z = project(
                offset + (scale / d0[i].reshape(-1))[:, None] * direction,
                scene.k[j].expand(h * w, 3, 3),
            )
            dj, spread = bilinear(d0, torch.full_like(flat, j), u, v)
            inside = (z > 0) & (u >= 1) & (u <= w - 2) & (v >= 1) & (v <= h - 2)
            gap = (z.clamp_min(1e-6) * dj / scale).log()  # > 0: behind j's surface
            candidates[i, j] = (flat[inside], gap[inside], spread[inside].log())
    if tol is None:
        tol = occlusion_threshold(torch.cat([g for _, g, _ in candidates.values()]))
        log_info("occlusion threshold: %.4f (log depth)", tol)
    pairs = {
        ij: pix[(gap.abs() < tol) & (spread < tol)]
        for ij, (pix, gap, spread) in candidates.items()
    }
    counts = torch.zeros(n, n)
    for (i, j), pix in pairs.items():
        counts[i, j] = len(pix)
    top = torch.zeros(n, n, dtype=torch.bool)
    picked = counts.topk(min(neighbors, n - 1), dim=1).indices
    top[torch.arange(n)[:, None], picked] = True
    chosen = [(i, j) for (i, j) in pairs if top[i, j] or top[j, i]]
    rays = [ray_in(scene, i, j, pairs[i, j]) for i, j in chosen]
    m = Matches(
        torch.cat([i * h * w + pairs[i, j] for i, j in chosen]),
        torch.cat([torch.full_like(pairs[i, j], j) for i, j in chosen]),
        torch.cat([offset for offset, _ in rays]),
        torch.cat([direction for _, direction in rays]),
    )
    assert (
        (m.src // (h * w) != m.dst).all()
        and m.offset.isfinite().all()
        and m.direction.isfinite().all()
    )
    return m


def disagreement_pct(depth: Tensor, scene: Scene, matches: Matches) -> float:
    """Median |log depth gap| between overlapping views, in percent."""
    scale = scene.depth.median()
    return 100 * residuals(scale / depth, scale, scene, matches).abs().median().item()


@dataclass(frozen=True)
class Problem:
    """What stays fixed in the solve: scene, scale s, d0, weights, normals, matches.

    w is E_d's weight per pixel, confidence times consistency. Normal terms are stacked
    over views: per axis (x, y), (a_p, g_p, a_q, g_q) and (w+, w-) as in
    `normals.NormalTerm`, the weights 0 between instances.
    """

    scene: Scene
    scale: Tensor
    d0: Tensor
    w: Tensor
    normals: NormalTerm | None
    matches: Matches


def same_instance_edges(ids: Tensor) -> tuple[Tensor, Tensor]:
    """Masks (V, H, W-1), (V, H-1, W) of the x and y edges within one instance.

    Pixels without an instance (-1) count as one, so E_n still shapes them.
    """
    return ids[..., 1:] == ids[..., :-1], ids[..., 1:, :] == ids[..., :-1, :]


def same_segment_matches(
    m: Matches, scene: Scene, d0: Tensor, scale: Tensor, ids: Tensor
) -> Matches:
    """Keep the matches that land on their source's segment (nearest pixel in j).

    A pixel without a segment has nothing to gate on and keeps its matches.
    """
    _, h, w = d0.shape
    depth = scale / d0.reshape(-1)[m.src]
    u, v, _ = project(m.offset + depth[:, None] * m.direction, scene.k[m.dst])
    col = u.round().long().clamp(0, w - 1)
    row = v.round().long().clamp(0, h - 1)
    flat = ids.reshape(-1)
    source = flat[m.src]
    keep = (source < 0) | (source == flat[(m.dst * h + row) * w + col])
    return Matches(m.src[keep], m.dst[keep], m.offset[keep], m.direction[keep])


def robust(u: Tensor, k: float) -> tuple[Tensor, Tensor]:
    """Huber's cost as one squared residual: r with r² = 2 h(u), and dr/du.

    h(u) = u²/2 for |u| ≤ k and k|u| - k²/2 beyond, so r = u inside and sign(u)
    √(2k|u| - k²) beyond, with dr/du = 1 and k / |r|. E stays a sum of squares, and
    Gauss-Newton on r minimizes the robust cost itself: a large change of a pixel costs
    linearly, and its weight in the normal equations falls as k / |r|.
    """
    outside = u.abs() > k
    r = torch.where(
        outside, u.sign() * (2 * k * u.abs() - k * k).clamp_min(0).sqrt(), u
    )
    return r, torch.where(outside, k / r.abs().clamp_min(k), 1.0)


def scene_residuals(x: Tensor, cfg: RefineConfig, p: Problem) -> dict[str, Tensor]:
    """Weighted residuals of each term, flat: E = Σ of their squares / N.

    N is the number of pixels over all views; x is δ = d - d0.

    E_d is w h(δ), h Huber's cost (`robust`).
    """
    d = p.d0 + x
    out = {"E_d": (0.5 * cfg.lambda_d * p.w).sqrt() * robust(x, cfg.huber)[0]}
    if cfg.lambda_delta > 0:
        out["E_delta"] = math.sqrt(cfg.lambda_delta) * x
    if cfg.lambda_n > 0:
        assert p.normals is not None, "E_n needs normals"
        z = -d.clamp_min(1e-6).log()
        dzs = (z[..., 1:] - z[..., :-1], z[..., 1:, :] - z[..., :-1, :])
        parts = []
        for (af, gf, ab, gb), (wf, wb), dz in zip(
            p.normals.terms, p.normals.weights, dzs, strict=True
        ):
            parts += [
                (cfg.lambda_n * wf).sqrt() * (af * dz + gf),
                (cfg.lambda_n * wb).sqrt() * (ab * dz + gb),
            ]
        out["E_n"] = torch.cat([r.flatten() for r in parts])
    if cfg.lambda_mv > 0:
        out["E_mv"] = math.sqrt(cfg.lambda_mv) * residuals(
            d, p.scale, p.scene, p.matches
        )
    return {k: v.flatten() for k, v in out.items()}


def scene_energy(x: Tensor, cfg: RefineConfig, p: Problem) -> dict[str, float]:
    """E and each weighted term's share of it."""
    terms = {
        k: (v.square().sum() / p.d0.numel()).item()
        for k, v in scene_residuals(x, cfg, p).items()
    }
    e = sum(terms.values())
    assert math.isfinite(e), terms
    return {"E": e} | terms


def energy_maps(d: Tensor, cfg: RefineConfig, p: Problem) -> dict[str, Tensor]:
    """Weighted cost of E_d, E_n and E_mv per pixel (V, H, W) at inverse depth d.

    Their means are the terms of E (λ_δ ‖δ‖² aside, which needs δ, not d). An edge's
    cost sits on its first pixel, a match's on its source pixel.
    """
    pad = torch.nn.functional.pad
    maps = {"E_d": 0.5 * cfg.lambda_d * p.w * robust(d - p.d0, cfg.huber)[0].square()}
    if p.normals is not None:
        z = -d.clamp_min(1e-6).log()
        dzs = (z[..., 1:] - z[..., :-1], z[..., 1:, :] - z[..., :-1, :])
        cost = torch.zeros_like(d)
        for (af, gf, ab, gb), (wf, wb), dz, side in zip(
            p.normals.terms, p.normals.weights, dzs, ((0, 1), (0, 0, 0, 1)), strict=True
        ):
            cost = cost + pad(
                wf * (af * dz + gf).square() + wb * (ab * dz + gb).square(), side
            )
        maps["E_n"] = cfg.lambda_n * cost
    r = residuals(d, p.scale, p.scene, p.matches)
    maps["E_mv"] = cfg.lambda_mv * torch.zeros(
        d.numel(), dtype=d.dtype, device=d.device
    ).index_add(0, p.matches.src, r.square()).view_as(d)
    return maps


def grid_operator(
    d: Tensor, cfg: RefineConfig, p: Problem, *, squared: bool = False
) -> Callable[[Tensor], Tensor]:
    """J v of the residuals on the pixel grid at d: E_d, then E_n.

    v is a change of d; E_mv and E_delta (a function of δ alone) are added by
    `normal_equations`. With squared, J's entries are squared.

    E_d's Jacobian is its weight times dr/du of `robust`; E_n is linear in z = -log d,
    so dz moves by -(v_q/d_q - v_p/d_p). With squared entries, J∘J applied to v and
    transposed onto the residuals' ones gives the diagonal of J^T J.
    """
    power = 2 if squared else 1
    a_d = ((0.5 * cfg.lambda_d * p.w).sqrt() * robust(d - p.d0, cfg.huber)[1]) ** power
    inv_d = (1 / d) ** power
    sign = 1 if squared else -1

    def apply(v: Tensor) -> Tensor:
        out = [a_d * v]
        if cfg.lambda_n > 0:
            assert p.normals is not None
            q = v * inv_d
            dqs = (
                q[..., 1:] + sign * q[..., :-1],
                q[..., 1:, :] + sign * q[..., :-1, :],
            )
            for (af, _, ab, _), (wf, wb), dq in zip(
                p.normals.terms, p.normals.weights, dqs, strict=True
            ):
                for a, wt in ((af, wf), (ab, wb)):
                    coef = (
                        cfg.lambda_n * wt * a.square()
                        if squared
                        else -(cfg.lambda_n * wt).sqrt() * a
                    )
                    out.append(coef * dq)
        return torch.cat([x.flatten() for x in out])

    return apply


def conjugate_gradient(
    hv: Callable[[Tensor], Tensor],
    b: Tensor,
    precondition: Tensor,
    iters: int,
    tol: float,
) -> tuple[Tensor, int]:
    """Solve H x = b for SPD H given as a product, by Jacobi-preconditioned CG.

    Stops once the residual norm falls below tol times b's (checked every 10
    iterations); returns x and the count.
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


def normal_equations(
    x: Tensor, cfg: RefineConfig, p: Problem
) -> tuple[Tensor, Callable[[Tensor], Tensor], Tensor]:
    """Gradient J^T r, the product v -> J^T J v and diag(J^T J) at x, all flat.

    J stacks the grid terms' (`grid_operator`), E_mv's and E_delta's √λ_δ.
    """
    d = p.d0 + x
    parts = scene_residuals(x, cfg, p)
    r_mv = parts.pop("E_mv", None)
    r_delta = parts.pop("E_delta", None)
    r_grid = torch.cat(list(parts.values()))
    grid = grid_operator(d, cfg, p)
    _, grid_t = torch.func.vjp(grid, torch.zeros_like(d))
    _, grid_sq_t = torch.func.vjp(
        grid_operator(d, cfg, p, squared=True), torch.zeros_like(d)
    )
    diag = grid_sq_t(torch.ones_like(r_grid))[0].flatten()
    gradient = grid_t(r_grid)[0].flatten()
    cols = vals = None
    if r_mv is not None:
        cols, vals = match_jacobian(d, p.scale, p.scene, p.matches)
        vals = math.sqrt(cfg.lambda_mv) * vals
        diag = diag.index_add(0, cols.flatten(), vals.square().flatten())
        gradient = gradient.index_add(0, cols.flatten(), (vals * r_mv).flatten())
    if r_delta is not None:
        gradient = gradient + math.sqrt(cfg.lambda_delta) * r_delta
        diag = diag + cfg.lambda_delta

    def hv(v: Tensor) -> Tensor:
        out = grid_t(grid(v.view_as(d)))[0].flatten()
        if cols is not None and vals is not None:
            jv = (vals * v[cols]).sum(0)
            out = out.index_add(0, cols.flatten(), (vals * jv).flatten())
        if r_delta is not None:
            out = out + cfg.lambda_delta * v
        return out

    return gradient, hv, diag


def gauss_newton_step(x: Tensor, cfg: RefineConfig, p: Problem) -> tuple[Tensor, int]:
    """Solve J^T J s = -J^T r at x by preconditioned CG; return s and CG iterations."""
    gradient, hv, diag = normal_equations(x, cfg, p)
    step, used = conjugate_gradient(
        hv, -gradient, 1 / diag.clamp_min(1e-12), cfg.cg_iters, cfg.cg_tol
    )
    return step.view_as(x), used


def refine_scene(
    scene: Scene,
    cfg: RefineConfig,
    normals: Tensor | None = None,
    segments: Tensor | None = None,
) -> tuple[Tensor, list[dict[str, float]], dict[str, int]]:
    """Gauss-Newton on all views at once; return depth (V, H, W), energy log, counts.

    Each step linearizes every residual at the current x and solves the normal equations
    J^T J s = -J^T r by Jacobi-preconditioned CG (`gauss_newton_step`), halving s until
    E decreases and no depth grows tenfold (the logs clamp d at 1e-6, so E alone would
    not see a step that crosses d = 0). Normals (V, H, W, 3), unit, in the frame of the
    rays, enable E_n. Segments (V, H, W) are linked instance ids, -1 for none (default:
    no pixel has one).
    """
    p = build_problem(scene, cfg, normals, segments)
    x, log, cg_total = gauss_newton(p, cfg)
    depth_hat = p.scale / (p.d0 + x)
    assert (
        depth_hat.shape == scene.depth.shape
        and depth_hat.isfinite().all()
        and (depth_hat > 0).all()
    )
    assert log[-1]["E"] <= log[0]["E"] + 1e-9, (log[0], log[-1])
    return (
        depth_hat,
        log,
        {"matches": len(p.matches.src), "cg_iterations": cg_total},
    )


@torch.no_grad()
def consistency(scene: Scene, d0: Tensor, scale: Tensor, ids: Tensor) -> Tensor:
    """Consistency u (V, H, W): of the other views a pixel lands in, the share agreeing.

    Each pixel is lifted on the input depth and projected into every other view; a view
    agrees if its nearest pixel there carries the pixel's linked instance id (the class
    does not matter). Pixels without an id, or that land in no other view, keep u = 1.
    """
    n, h, w = d0.shape
    seen, agree = torch.zeros_like(d0), torch.zeros_like(d0)
    for i in range(n):
        depth = (scale / d0[i]).reshape(-1)
        own = ids[i].reshape(-1)
        for j in range(n):
            if j == i:
                continue
            world_to_j = scene.rotation[j].T
            x = (
                scene.rays[i, ..., :3].reshape(-1, 3)
                - scene.center[j]
                + depth[:, None] * scene.rays[i, ..., 3:].reshape(-1, 3)
            ) @ world_to_j.T
            u, v, z = project(x, scene.k[j].expand(h * w, 3, 3))
            col, row = u.round().long(), v.round().long()
            inside = (z > 0) & (col >= 0) & (col < w) & (row >= 0) & (row < h)
            there = ids[j].reshape(-1)[row.clamp(0, h - 1) * w + col.clamp(0, w - 1)]
            seen[i] += inside.view(h, w)
            agree[i] += (inside & (there == own)).view(h, w)
    u = torch.where(seen > 0, agree / seen.clamp_min(1), 1.0)
    out = torch.where(ids >= 0, u, 1.0)
    assert ((out >= 0) & (out <= 1)).all()
    return out


def build_problem(
    scene: Scene,
    cfg: RefineConfig,
    normals: Tensor | None = None,
    segments: Tensor | None = None,
) -> Problem:
    """Fix what the solve keeps constant: scale, d0, weights, normal terms, matches.

    E_d's weight is the confidence (mean one per view) times the consistency u; E_n's
    edges stay within one instance; E_mv keeps the matches that land on their source's
    instance.
    """
    assert normals is not None or cfg.lambda_n == 0, "E_n needs normals"
    ids = (
        torch.full(scene.depth.shape, -1, dtype=torch.long)
        if segments is None
        else segments
    ).to(scene.depth.device)
    assert ids.shape == scene.depth.shape, ids.shape
    scale = scene.depth.median()
    d0 = scale / scene.depth
    w = scene.conf / scene.conf.mean(dim=(1, 2), keepdim=True)
    w = w * consistency(scene, d0, scale, ids)
    terms = None
    if normals is not None:
        per_view = [
            normal_term(normals[i], scene.rays[i], d0[i], cfg.edge_tol)
            for i in range(len(d0))
        ]
        terms = NormalTerm(
            [
                tuple(torch.stack([t.terms[a][k] for t in per_view]) for k in range(4))
                for a in range(2)
            ],
            [
                tuple(
                    same * torch.stack([t.weights[a][k] for t in per_view])
                    for k in range(2)
                )
                for a, same in enumerate(same_instance_edges(ids))
            ],
        )
    matches = find_matches(scene, d0, scale, cfg.neighbors)
    matches = same_segment_matches(matches, scene, d0, scale, ids)
    return Problem(scene, scale, d0, w, terms, matches)


def gauss_newton(
    p: Problem,
    cfg: RefineConfig,
    on_step: Callable[[Tensor, dict, int], None] | None = None,
) -> tuple[Tensor, list[dict[str, float]], int]:
    """Run `cfg.gn_iters` Gauss-Newton steps from x = 0; return x, energy log, CG count.

    x is δ = d - d0 (shaped as d0). Each step is halved until E decreases and no depth
    grows tenfold; on_step(x, energies, CG iterations so far) is called after every
    accepted step.
    """
    x = torch.zeros_like(p.d0)
    log, cg_total = [scene_energy(x, cfg, p)], 0
    for _ in range(cfg.gn_iters):
        step, used = gauss_newton_step(x, cfg, p)
        cg_total += used
        for _ in range(8):
            positive = bool((p.d0 + x + step > 0.1 * p.d0).all())
            trial = scene_energy(x + step, cfg, p) if positive else {"E": math.inf}
            if trial["E"] < log[-1]["E"]:
                x = x + step
                log.append(trial)
                break
            step = step / 2
        else:
            break
        if on_step is not None:
            on_step(x, log[-1], cg_total)
    return x, log, cg_total


def scene_normals(camera: np.ndarray, geometry: Geometry, scene: Scene) -> Tensor:
    """DSINE's camera-frame normals (V, H, W, 3) rotated into the frame of the rays."""
    return torch.stack(
        [
            to_world(c, c2w, depth, rays)[0]
            for c, c2w, depth, rays in zip(
                camera, geometry.cam_to_world, scene.depth, scene.rays, strict=True
            )
        ]
    )


@dataclass
class JointRefiner:
    """The pipeline's refiner: all views at once by Gauss-Newton, cameras fixed.

    E_n runs DSINE on the views (`dsine_repo`, needed when λ_n > 0); the linked instance
    ids are the segments. The output keeps the backbone's cameras with each ray map
    replaced by its fitted pinhole (`fit_pinhole`).
    """

    cfg: RefineConfig
    dsine_repo: Path | None = None

    def __post_init__(self) -> None:
        """Check that E_n has normals to work with."""
        assert self.cfg.lambda_n == 0 or self.dsine_repo is not None, "E_n: DSINE"

    @classmethod
    def from_config(cls, path: Path) -> "JointRefiner":
        """Read `refine` and `dsine_repo` (YAML)."""
        raw = yaml.safe_load(path.read_text())
        repo = raw.get("dsine_repo")
        return cls(RefineConfig(**raw["refine"]), None if repo is None else Path(repo))

    def refine(
        self,
        views: Views,
        geometry: Geometry,
        segmentation: Segmentation,
    ) -> Geometry:
        """Return the jointly refined depth of every view, same shapes as the input."""
        assert views.images.shape[:3] == geometry.depth.shape, geometry.depth.shape
        cuda = self.cfg.device == "cuda" and torch.cuda.is_available()
        device = torch.device("cuda" if cuda else "cpu")
        scene = scene_from_geometry(geometry).to(device)
        normals = None
        if self.cfg.lambda_n > 0:
            assert self.dsine_repo is not None
            camera = dsine_normals(views.images, geometry.intrinsics, self.dsine_repo)
            normals = scene_normals(camera, geometry, scene)
        t0 = time.perf_counter()
        segments = torch.from_numpy(segmentation.instance_ids.astype(np.int64))
        depth, log, counts = refine_scene(scene, self.cfg, normals, segments)
        log_info(
            "refined %d views in %.1f s: E %.4g -> %.4g, %d Gauss-Newton steps, %s",
            len(views),
            time.perf_counter() - t0,
            log[0]["E"],
            log[-1]["E"],
            len(log) - 1,
            counts,
        )
        return geometry_from_scene(scene, depth, scene.conf)
