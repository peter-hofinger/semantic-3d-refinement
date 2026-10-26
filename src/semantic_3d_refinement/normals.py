"""Detail from predicted normals: bilateral normal integration (BiNI, Cao et al. 2022) with a depth prior.

Own matrix-free GPU implementation of the BiNI energy on log depth, generalized from a pinhole K to the ray map:
for neighbors p, q = p + 1 along x or y, x_q lies on the plane of (x_p, n_p) iff dz = log|n_p.d_p| - log|n_p.d_q|.

  E(z) = sum_p w_p r+_p^2 + (1 - w_p) r-_p^2 + lambda_depth * sum_p (G (z - z0))_p^2,   z = log depth

w_p = sigmoid(k ((a_p dz-)^2 - (a_p dz+)^2)) lets the surface break on the side with the larger jump. As in
BiNI, the weights start at 0.5, so the first solve integrates smoothly and the jumps come from its result; taking
them from the prior instead locks in its steps (a backbone's patch seams, its noise) as discontinuities.
G is a Gaussian blur of width prior_sigma px (identity at 0, as in BiNI): the prior then only holds the low
frequencies of the depth, and the normals alone set the detail.
"""

import argparse
import dataclasses
import importlib.util
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import Tensor

from semantic_3d_refinement.refine import git_hash, load_view


@dataclass(frozen=True)
class BiniConfig:
    """BiNI settings; k and lambda_depth mean what they mean in the reference implementation."""

    k: float = 2.0
    lambda_depth: float = 1e-4
    prior_sigma: float = 0.0
    iters: int = 150
    tol: float = 1e-4
    cg_iters: int = 5000
    cg_tol: float = 1e-3
    device: str = "cuda"

    def __post_init__(self) -> None:
        """Check ranges."""
        assert self.k >= 0
        assert self.lambda_depth > 0
        assert self.prior_sigma >= 0
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


def grid_score(normals: Tensor, patch: int = 14) -> float:
    """Median normal change beside the backbone's patch seams over the median elsewhere; about 1 without a grid.

    A depth step between pixels s - 1 and s (s a multiple of patch) tilts the central-difference normals at both,
    so the change shows on the edges (s - 2, s - 1) and (s, s + 1), at positions patch - 2 and 0.
    """
    h, w, _ = normals.shape
    gx = (normals[:, 1:] - normals[:, :-1]).norm(dim=-1)
    gy = (normals[1:] - normals[:-1]).norm(dim=-1)

    def beside(n: int) -> Tensor:
        m = torch.arange(n - 1, device=normals.device) % patch
        return (m == 0) | (m == patch - 2)

    sx, sy = beside(w).expand_as(gx), beside(h)[:, None].expand_as(gy)
    g, seam = torch.cat([gx.ravel(), gy.ravel()]), torch.cat([sx.ravel(), sy.ravel()])
    return (g[seam].median() / g[~seam].median()).item()


def blur(x: Tensor, sigma: float, *, squared: bool = False) -> Tensor:
    """Separable Gaussian blur with zero padding (self-adjoint); with squared, blur by the squared kernel."""
    if sigma == 0:
        return x.clone()
    r = int(3 * sigma)
    g = torch.exp(-0.5 * (torch.arange(-r, r + 1, dtype=x.dtype, device=x.device) / sigma).square())
    g = g / g.sum()
    g = g.square() if squared else g
    y = torch.nn.functional.conv2d(x[None, None], g.view(1, 1, 1, -1), padding=(0, r))
    return torch.nn.functional.conv2d(y, g.view(1, 1, -1, 1), padding=(r, 0))[0, 0]


def energy(z: Tensor, z0: Tensor, terms: list, weights: list, cfg: BiniConfig) -> float:
    """BiNI energy at log depth z for fixed weights."""
    e = cfg.lambda_depth * blur(z - z0, cfg.prior_sigma).square().sum()
    for (af, gf, ab, gb), (wf, wb), dz in zip(terms, weights, diffs(z), strict=True):
        e = e + (wf * (af * dz + gf).square() + wb * (ab * dz + gb).square()).sum()
    return e.item()


def solve(z: Tensor, z0: Tensor, terms: list, weights: list, cfg: BiniConfig) -> Tensor:
    """Minimize the quadratic energy for fixed weights with Jacobi-preconditioned CG, warm-started at z.

    Stops when the residual fell by cg_tol; relative to |rhs| (as in BiNI) a strong prior stops it at once.
    """
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

    def prior(x: Tensor) -> Tensor:
        return cfg.lambda_depth * blur(blur(x, cfg.prior_sigma), cfg.prior_sigma)

    def apply(x: Tensor) -> Tensor:
        dx, dy = diffs(x)
        return scatter(cs[0] * dx, cs[1] * dy, prior(x))

    rhs = scatter(-bs[0], -bs[1], prior(z0))
    diag = cfg.lambda_depth * blur(torch.ones_like(z), cfg.prior_sigma, squared=True)
    diag[:, 1:] += cs[0]
    diag[:, :-1] += cs[0]
    diag[1:] += cs[1]
    diag[:-1] += cs[1]
    x = z.clone()
    r = rhs - apply(x)
    y = r / diag
    p, ry = y.clone(), (r * y).sum()
    stop = cfg.cg_tol * r.norm()
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
    weights = [(torch.full_like(af, 0.5), torch.full_like(ab, 0.5)) for af, _, ab, _ in terms]
    log = [energy(z, z0, terms, weights, cfg)]
    for _ in range(cfg.iters):
        z = solve(z, z0, terms, weights, cfg)
        weights = side_weights(z, a, cfg.k)
        log.append(energy(z, z0, terms, weights, cfg))
        if abs(log[-1] - log[-2]) <= cfg.tol * log[-2]:
            break
    depth_hat = z.exp().to(depth.dtype)
    assert depth_hat.shape == depth.shape and depth_hat.isfinite().all() and (depth_hat > 0).all()
    return depth_hat, weights, log


DSINE_SIGNS = {"as_is": (1, 1, 1), "negated": (-1, -1, -1), "opengl": (1, -1, -1), "flip_xy": (-1, -1, 1)}


def run_dsine(image: np.ndarray, intrinsics: np.ndarray, repo: Path) -> np.ndarray:
    """DSINE normals of one image, in the camera frame as DSINE returns them, (H, W, 3)."""
    sys.path.insert(0, str(repo))
    spec = importlib.util.spec_from_file_location("dsine_hubconf", repo / "hubconf.py")
    assert spec is not None and spec.loader is not None
    hub = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(hub)
    predictor = hub.DSINE()
    with torch.no_grad():
        x = torch.from_numpy(image).float().permute(2, 0, 1)[None] / 255
        n = predictor.infer_tensor(x, torch.from_numpy(intrinsics).float()[None].cuda())[0]
    return n.permute(1, 2, 0).cpu().numpy().astype(np.float32)


def to_world(normals_cam: np.ndarray, cam_to_world: np.ndarray, depth: Tensor, rays: Tensor) -> tuple[Tensor, str]:
    """Rotate camera-frame normals into the ray frame, with the axis signs that best match the depth's own normals."""
    rot = torch.from_numpy(cam_to_world[:3, :3]).to(rays)
    ref = normals_from_depth(depth, rays)
    cam = torch.from_numpy(normals_cam).to(rays)
    worlds = {name: (cam * torch.tensor(sign).to(cam)) @ rot.T for name, sign in DSINE_SIGNS.items()}
    agree = {name: (world * ref).sum(-1).median().item() for name, world in worlds.items()}
    best = max(agree, key=agree.__getitem__)
    assert agree[best] > 0.5, agree  # noqa: PLR2004
    world = worlds[best]
    return world / world.norm(dim=-1, keepdim=True), best


def predicted_normals(raw: dict, data: dict[str, Tensor]) -> tuple[Tensor, dict]:
    """Return the configured view's normals in the ray frame; DSINE outputs are cached next to the backbone's."""
    if raw["normals"] == "depth":
        return normals_from_depth(data["depth"], data["rays"]), {}
    assert raw["normals"] == "dsine", raw["normals"]
    view, src = raw["view"], Path(raw["input"])
    backbone = np.load(src)
    cache = src.with_name(f"dsine_view{view}_{src.name}")
    if not cache.exists():
        image = np.load(raw["gt_views"])["images"][view]
        np.savez(cache, normals=run_dsine(image, backbone["intrinsics"][view], Path(raw["dsine_repo"])))
    world, convention = to_world(np.load(cache)["normals"], backbone["cam_to_world"][view], data["depth"], data["rays"])
    return world, {"dsine_cache": str(cache), "dsine_convention": convention}


def angle_to(normals: Tensor, depth: Tensor, rays: Tensor) -> float:
    """Median angle in degrees between the depth's own normals and the given normals."""
    own = normals_from_depth(depth, rays)
    return torch.rad2deg(torch.acos((own * normals).sum(-1).clamp(-1, 1))).median().item()


def drift(depth_hat: Tensor, depth: Tensor, sigma: float = 16.0) -> float:
    """Median |log D_hat - log D| after a Gaussian blur, in percent: how far the coarse shape moved."""
    return 100 * blur((depth_hat.double() / depth.double()).log(), sigma).abs().median().item()


def main() -> None:
    """Integrate normals into one view: `uv run python -m semantic_3d_refinement.normals --cfg configs/<name>.yaml`."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=Path, required=True)
    cfg_path: Path = parser.parse_args().cfg
    raw = yaml.safe_load(cfg_path.read_text())
    cfg = BiniConfig(**raw["bini"])
    device = torch.device(cfg.device if cfg.device != "cuda" or torch.cuda.is_available() else "cpu")
    data = {k: v.to(device) for k, v in load_view(Path(raw["input"]), raw.get("view")).items()}
    depth, rays = data["depth"], data["rays"]

    t0 = time.perf_counter()
    normals, normals_info = predicted_normals(raw, data)
    normals_seconds = time.perf_counter() - t0
    t0 = time.perf_counter()
    depth_hat, _, log = bini(normals, rays, depth, cfg)
    if device.type == "cuda":
        torch.cuda.synchronize()
    seconds = time.perf_counter() - t0

    summary = {
        "config": raw,
        "resolved": dataclasses.asdict(cfg),
        "git": git_hash(),
        "normals_seconds": normals_seconds,
        "refine_seconds": seconds,
        "outer_iterations": len(log) - 1,
        **normals_info,
        "grid_score_normals": grid_score(normals),
        "metrics_input": {
            "angle_to_normals_deg": angle_to(normals, depth, rays),
            "grid_score": grid_score(normals_from_depth(depth, rays)),
            "drift_pct": 0.0,
        },
        "metrics_refined": {
            "angle_to_normals_deg": angle_to(normals, depth_hat, rays),
            "grid_score": grid_score(normals_from_depth(depth_hat, rays)),
            "drift_pct": drift(depth_hat, depth),
        },
    }
    out = Path(raw["output_dir"]) / f"{cfg_path.stem}-{time.strftime('%Y%m%d-%H%M%S')}"
    out.mkdir(parents=True)
    (out / "run.yaml").write_text(yaml.safe_dump(summary, sort_keys=False))
    np.savez(out / "refined.npz", depth=depth_hat.cpu().numpy(), normals=normals.cpu().numpy())
    print(yaml.safe_dump({k: v for k, v in summary.items() if k != "config"}, sort_keys=False))  # noqa: T201
    print(f"wrote {out}")  # noqa: T201


if __name__ == "__main__":
    main()
