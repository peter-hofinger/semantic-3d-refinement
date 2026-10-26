"""Detail from predicted normals: bilateral normal integration (BiNI, Cao et al. 2022) with a depth prior.

Own matrix-free GPU implementation of the BiNI energy on log depth, generalized from a pinhole K to the ray map:
for neighbors p, q = p + 1 along x or y, x_q lies on the plane of (x_p, n_p) iff dz = log|n_p.d_p| - log|n_p.d_q|.

  E(z) = sum_p w_p r+_p^2 + (1 - w_p) r-_p^2 + lambda_depth * sum_p (z_p - z0_p)^2,   z = log depth

w_p = sigmoid(k ((a_p dz-)^2 - (a_p dz+)^2)) lets the surface break on the side with the larger jump.
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

from semantic_3d_refinement.refine import (
    camera_alignment,
    fit_projection,
    git_hash,
    load_views,
    multiview_error,
    multiview_metrics,
    points_from_rays,
)


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


DSINE_SIGNS = {"as_is": (1, 1, 1), "negated": (-1, -1, -1), "opengl": (1, -1, -1), "flip_xy": (-1, -1, 1)}


def run_dsine(images: np.ndarray, intrinsics: np.ndarray, repo: Path) -> np.ndarray:
    """Per-view DSINE normals in the camera frame as DSINE returns them, (V, H, W, 3)."""
    sys.path.insert(0, str(repo))
    spec = importlib.util.spec_from_file_location("dsine_hubconf", repo / "hubconf.py")
    assert spec is not None and spec.loader is not None
    hub = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(hub)
    predictor = hub.DSINE()
    out = []
    with torch.no_grad():
        for img, k in zip(images, intrinsics, strict=True):
            x = torch.from_numpy(img).float().permute(2, 0, 1)[None] / 255
            n = predictor.infer_tensor(x, torch.from_numpy(k).float()[None].cuda())[0]
            out.append(n.permute(1, 2, 0).cpu().numpy())
    return np.stack(out).astype(np.float32)


def to_world(normals_cam: np.ndarray, cam_to_world: np.ndarray, depth: Tensor, rays: Tensor) -> tuple[Tensor, str]:
    """Rotate camera-frame normals into the ray frame, with the axis signs that best match the depth's own normals."""
    rot = torch.from_numpy(cam_to_world[:, :3, :3]).to(rays)
    ref = torch.stack([normals_from_depth(d, r) for d, r in zip(depth, rays, strict=True)])
    cam = torch.from_numpy(normals_cam).to(rays)
    agree = {}
    for name, sign in DSINE_SIGNS.items():
        world = torch.einsum("vij,vhwj->vhwi", rot, cam * torch.tensor(sign).to(cam))
        agree[name] = (world * ref).sum(-1).median().item()
    best = max(agree, key=agree.__getitem__)
    assert agree[best] > 0.5, agree  # noqa: PLR2004
    world = torch.einsum("vij,vhwj->vhwi", rot, cam * torch.tensor(DSINE_SIGNS[best]).to(cam))
    return world / world.norm(dim=-1, keepdim=True), best


def predicted_normals(raw: dict, data: dict[str, Tensor]) -> tuple[Tensor, dict]:
    """World-frame normals per view from the configured source, DSINE outputs cached next to the backbone's."""
    if raw["normals"] == "depth":
        return torch.stack([normals_from_depth(d, r) for d, r in zip(data["depth"], data["rays"], strict=True)]), {}
    assert raw["normals"] == "dsine", raw["normals"]
    backbone = np.load(raw["input"])
    views = raw.get("views")
    cache = Path(raw["input"]).with_name("dsine_" + Path(raw["input"]).name)
    if not cache.exists():
        images = np.load(raw["gt_views"])["images"]
        np.savez(cache, normals=run_dsine(images, backbone["intrinsics"], Path(raw["dsine_repo"])))
    cam = np.load(cache)["normals"]
    c2w = backbone["cam_to_world"]
    cam, c2w = (cam[views], c2w[views]) if views is not None else (cam, c2w)
    world, convention = to_world(cam, c2w, data["depth"], data["rays"])
    return world, {"dsine_cache": str(cache), "dsine_convention": convention}


def integrate_all(normals: Tensor, data: dict[str, Tensor], cfg: BiniConfig) -> tuple[Tensor, list[int], list[float]]:
    """BiNI per view; return depth, outer iterations and seconds per view."""
    out, iters, seconds = [], [], []
    for n, r, d in zip(normals, data["rays"], data["depth"], strict=True):
        t0 = time.perf_counter()
        depth_hat, _, log = bini(n, r, d, cfg)
        if d.is_cuda:
            torch.cuda.synchronize()
        out.append(depth_hat)
        iters.append(len(log) - 1)
        seconds.append(round(time.perf_counter() - t0, 3))
    return torch.stack(out), iters, seconds


def main() -> None:
    """Integrate normals per view: `uv run python -m semantic_3d_refinement.normals --cfg configs/<name>.yaml`."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=Path, required=True)
    cfg_path: Path = parser.parse_args().cfg
    raw = yaml.safe_load(cfg_path.read_text())
    cfg = BiniConfig(**raw["bini"])
    device = torch.device(cfg.device if cfg.device != "cuda" or torch.cuda.is_available() else "cpu")
    data = {k: v.to(device) for k, v in load_views(Path(raw["input"]), raw.get("views")).items()}
    depth, rays = data["depth"], data["rays"]

    t0 = time.perf_counter()
    normals, normals_info = predicted_normals(raw, data)
    normals_seconds = time.perf_counter() - t0
    depth_hat, iters, seconds = integrate_all(normals, data, cfg)

    cams = [fit_projection(r) for r in rays]
    mv_in, mv_out = multiview_error(depth, rays, cams), multiview_error(depth_hat, rays, cams)
    summary = {
        "config": raw,
        "resolved": dataclasses.asdict(cfg),
        "git": git_hash(),
        "views": len(depth),
        "normals_seconds": normals_seconds,
        "refine_seconds": sum(seconds),
        "refine_seconds_per_view": seconds,
        "outer_iterations": iters,
        **normals_info,
        "metrics_input": multiview_metrics(mv_in),
        "metrics_refined": multiview_metrics(mv_out),
    }
    if "gt_views" in raw:
        cams_gt = camera_alignment(rays, Path(raw["gt_views"]), raw.get("views"))
        shift = (points_from_rays(depth_hat, rays) - points_from_rays(depth, rays)).norm(dim=-1)
        summary |= {"cameras": cams_gt, "mean_point_shift_mm": 1000 * cams_gt["scale_to_m"] * shift.mean().item()}
    sweep = {}
    for key, values in raw.get("sweep", {}).items():
        sweep[key] = []
        for v in values:
            d = integrate_all(normals, data, dataclasses.replace(cfg, **{key: v}))[0]
            sweep[key].append(multiview_metrics(multiview_error(d, rays, cams))["mv_median"])
    summary["sweep_mv_median"] = sweep

    out = Path(raw["output_dir"]) / f"{cfg_path.stem}-{time.strftime('%Y%m%d-%H%M%S')}"
    out.mkdir(parents=True)
    (out / "run.yaml").write_text(yaml.safe_dump(summary, sort_keys=False))
    np.savez(out / "refined.npz", depth=depth_hat.cpu().numpy(), normals=normals.cpu().numpy())
    np.savez(out / "multiview_error.npz", input=mv_in.cpu().numpy(), refined=mv_out.cpu().numpy())
    print(yaml.safe_dump({k: v for k, v in summary.items() if k != "config"}, sort_keys=False))  # noqa: T201
    print(f"wrote {out}")  # noqa: T201


if __name__ == "__main__":
    main()
