"""Monocular variational refinement of depth maps, view by view: dense inverse-depth residual only."""

import argparse
import dataclasses
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
import yaml
from torch import Tensor


@dataclass(frozen=True)
class RefineConfig:
    """Energy weights and solver settings; inverse depth is normalized by the median depth."""

    iters: int = 200
    huber: float = 0.05
    lambda_bd: float = 10.0
    lambda_delta: float = 0.0
    history_size: int = 20
    device: str = "cuda"

    def __post_init__(self) -> None:
        """Check ranges."""
        assert self.iters >= 0
        assert self.huber > 0
        assert self.lambda_bd >= 0
        assert self.lambda_delta >= 0
        assert self.history_size > 0


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


def points_from_rays(depth: Tensor, rays: Tensor) -> Tensor:
    """Lift depth along rays: x = origin + depth * direction."""
    return rays[..., :3] + depth[..., None] * rays[..., 3:]


def second_differences(d: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Return d_xx, d_yy, d_xy; all vanish where d is affine in pixel coordinates (planes)."""
    dxx = d[:, 2:] - 2 * d[:, 1:-1] + d[:, :-2]
    dyy = d[2:] - 2 * d[1:-1] + d[:-2]
    dxy = d[1:, 1:] - d[1:, :-1] - d[:-1, 1:] + d[:-1, :-1]
    return dxx, dyy, dxy


LOG_KEYS = ("E", "E_d", "E_bd", "E_delta")


def normalize(depth: Tensor, conf: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Median depth scale, normalized inverse depth d0 = scale / depth, and mean-one confidence weights."""
    scale = depth.median()
    return scale, scale / depth, conf / conf.mean()


def term_weights(cfg: RefineConfig) -> dict[str, float]:
    """Weight of each energy term in E."""
    return {"E_d": 1.0, "E_bd": cfg.lambda_bd, "E_delta": cfg.lambda_delta}


def energy_maps(delta: Tensor, d0: Tensor, w: Tensor, cfg: RefineConfig) -> dict[str, Tensor]:
    """Unweighted per-pixel cost of each term; their means are the terms of E."""
    dxx, dyy, dxy = second_differences(d0 + delta)
    smooth = F.pad(dxx.square(), (1, 1)) + F.pad(dyy.square(), (0, 0, 1, 1)) + 2 * F.pad(dxy.square(), (0, 1, 0, 1))
    return {
        "E_d": w * F.huber_loss(delta, torch.zeros_like(delta), reduction="none", delta=cfg.huber),
        "E_bd": smooth,
        "E_delta": delta.square(),
    }


def energy(delta: Tensor, d0: Tensor, w: Tensor, cfg: RefineConfig) -> tuple[Tensor, dict[str, float]]:
    """E = E_d + lambda_bd E_bd + lambda_delta ||delta||^2, each a per-pixel mean."""
    terms = {k: v.mean() for k, v in energy_maps(delta, d0, w, cfg).items()}
    e = sum(term_weights(cfg)[k] * terms[k] for k in terms)
    assert isinstance(e, Tensor) and e.isfinite(), e
    return e, {"E": e.item()} | {k: v.item() for k, v in terms.items()}


def refine(
    depth: Tensor,
    rays: Tensor,
    conf: Tensor,
    cfg: RefineConfig,
) -> tuple[Tensor, Tensor, Tensor, list[dict[str, float]]]:
    """Optimize the dense residual delta on normalized inverse depth; return depth, points, delta, energy log."""
    check_inputs(depth, rays, conf)
    scale, d0, w = normalize(depth, conf)
    delta = torch.zeros_like(depth, requires_grad=True)

    log = [energy(delta, d0, w, cfg)[1]]
    if cfg.iters > 0:
        opt = torch.optim.LBFGS(
            [delta],
            lr=1.0,
            max_iter=cfg.iters,
            history_size=cfg.history_size,
            tolerance_grad=1e-9,
            tolerance_change=1e-12,
            line_search_fn="strong_wolfe",
        )

        def closure() -> Tensor:
            opt.zero_grad()
            e, terms = energy(delta, d0, w, cfg)
            e.backward()
            log.append(terms)
            return e

        opt.step(closure)
        with torch.no_grad():
            log.append(energy(delta, d0, w, cfg)[1])
    delta = delta.detach()

    depth_hat = depth / (1 + delta * depth / scale)
    points = points_from_rays(depth_hat, rays)
    assert depth_hat.shape == depth.shape and depth_hat.isfinite().all() and (depth_hat > 0).all()
    assert points.shape == (*depth.shape, 3) and points.isfinite().all()
    assert log[-1]["E"] <= log[0]["E"] + 1e-6, (log[0], log[-1])
    return depth_hat, points, delta, log


def depth_metrics(depth: Tensor, rays: Tensor, gt_depth: Tensor) -> dict[str, float]:
    """Sanity metrics against a known ground truth (synthetic scenes only)."""
    err = (points_from_rays(depth, rays) - points_from_rays(gt_depth, rays)).norm(dim=-1)
    return {
        "abs_rel": ((depth - gt_depth).abs() / gt_depth).mean().item(),
        "point_rmse": err.square().mean().sqrt().item(),
    }


def fit_projection(rays: Tensor) -> tuple[Tensor, Tensor]:
    """Pinhole (M, c) consistent with a ray map: pixel (col, row, 1) ~ M (x - c), fit by DLT on the directions.

    Predicted ray maps are only near-pinhole: median residual 0.3-1.4 px on ScanNet++ with Déjà View.
    """
    h, w = rays.shape[:2]
    v, u = torch.meshgrid(torch.arange(h, device=rays.device), torch.arange(w, device=rays.device), indexing="ij")
    un, vn = ((u + 0.5) / w - 0.5).reshape(-1, 1).double(), ((v + 0.5) / h - 0.5).reshape(-1, 1).double()
    d = rays[..., 3:].reshape(-1, 3).double()
    zero = torch.zeros_like(d)
    a = torch.cat([torch.cat([-d, zero, un * d], 1), torch.cat([zero, -d, vn * d], 1)])
    m = torch.linalg.eigh(a.T @ a).eigenvectors[:, 0].reshape(3, 3)
    norm = torch.tensor([[1 / w, 0, 0.5 / w - 0.5], [0, 1 / h, 0.5 / h - 0.5], [0, 0, 1]], dtype=m.dtype)
    m = torch.linalg.solve(norm.to(m.device), m)
    m = m * (d @ m[2]).mean().sign() / m[2].norm()
    c = rays[..., :3].reshape(-1, 3).double().mean(0)
    q = d @ m.T
    px = (q[:, :2] / q[:, 2:] - torch.stack([u, v], -1).reshape(-1, 2)).norm(dim=-1)
    assert (q[:, 2] > 0).all() and px.median() < 2, px.median()  # noqa: PLR2004
    return m.float(), c.float()


def multiview_error(depth: Tensor, rays: Tensor, cams: list[tuple[Tensor, Tensor]]) -> Tensor:
    """Per pixel, the median over the other views j of |x - x_j| / D_j, where x_j is the point j sees at x's pixel.

    Cameras are the frozen prediction's own; nan where no other view sees the point.
    """
    n, h, w = depth.shape
    assert rays.shape == (n, h, w, 6) and len(cams) == n
    pts = points_from_rays(depth, rays)
    out = torch.full_like(depth, float("nan"))
    for i in range(n):
        errs = []
        for j in (j for j in range(n) if j != i):
            m, c = cams[j]
            q = (pts[i] - c) @ m.T
            col, row = (q[..., 0] / q[..., 2]).round(), (q[..., 1] / q[..., 2]).round()
            seen = (q[..., 2] > 0) & (col >= 0) & (col < w) & (row >= 0) & (row < h)
            col, row = col.clamp(0, w - 1).long(), row.clamp(0, h - 1).long()
            err = (pts[i] - pts[j][row, col]).norm(dim=-1) / depth[j][row, col]
            errs.append(torch.where(seen, err, float("nan")))
        out[i] = torch.stack(errs).nanmedian(dim=0).values
    assert out.shape == depth.shape and (out[out.isfinite()] >= 0).all()
    return out


def multiview_metrics(err: Tensor) -> dict[str, float]:
    """Summaries of multiview_error over all pixels some other view sees."""
    e = err[err.isfinite()]
    return {"mv_median": e.median().item(), "mv_inlier_1pct": (e < 0.01).float().mean().item()}  # noqa: PLR2004


def align_sim3(src: Tensor, dst: Tensor) -> tuple[float, Tensor, Tensor]:
    """Umeyama: s, R, t minimizing sum |s R src + t - dst|^2 over corresponding (N, 3) points."""
    assert src.shape == dst.shape and src.shape[0] >= 3 and src.shape[1] == 3  # noqa: PLR2004
    src, dst = src.double(), dst.double()
    mu_s, mu_d = src.mean(0), dst.mean(0)
    a, b = src - mu_s, dst - mu_d
    u, sig, vt = torch.linalg.svd(b.T @ a / len(src))
    sign = torch.ones(3, dtype=src.dtype)
    sign[2] = torch.linalg.det(u @ vt).sign()
    rot = u @ torch.diag(sign) @ vt
    s = (sig * sign).sum() / a.square().sum(1).mean()
    return s.item(), rot, mu_d - s * rot @ mu_s


def load_views(path: Path, views: list[int] | None = None) -> dict[str, Tensor]:
    """Load cached views as (V, H, W[, C]) float tensors: depth, rays, conf, and gt_depth when known.

    Accepts the synthetic single-view format and the Déjà View export (ray_origins, ray_directions, confidence).
    """
    data = dict(np.load(path))
    if "ray_origins" in data:
        data = {
            "depth": data["depth"],
            "rays": np.concatenate([data["ray_origins"], data["ray_directions"]], axis=-1),
            "conf": data["confidence"],
        }
    if data["depth"].ndim == 2:  # noqa: PLR2004
        data = {k: v[None] for k, v in data.items()}
    data.setdefault("conf", np.ones_like(data["depth"]))
    if views is not None:
        data = {k: v[views] for k, v in data.items()}
    out = {k: torch.from_numpy(np.ascontiguousarray(v)).float() for k, v in data.items()}
    for i in range(len(out["depth"])):
        check_inputs(out["depth"][i], out["rays"][i], out["conf"][i])
    return out


def camera_alignment(rays: Tensor, gt_views: Path, views: list[int] | None) -> dict[str, float]:
    """Sim(3) from predicted camera centers to ground-truth ones (meters): scale and camera ATE."""
    gt = np.load(gt_views)["cam_to_world"][:, :3, 3]
    gt = torch.from_numpy(gt[views] if views is not None else gt)
    centers = rays[..., :3].mean((1, 2)).cpu()
    s, rot, t = align_sim3(centers, gt)
    ate = ((s * centers.double() @ rot.T + t) - gt.double()).norm(dim=-1)
    return {"scale_to_m": s, "camera_ate_m": ate.square().mean().sqrt().item()}


def write_ply(path: Path, points: np.ndarray) -> None:
    """Write an (N, 3) float32 point cloud as binary PLY."""
    assert points.ndim == 2 and points.shape[1] == 3, points.shape  # noqa: PLR2004
    header = f"ply\nformat binary_little_endian 1.0\nelement vertex {len(points)}\n"
    header += "property float x\nproperty float y\nproperty float z\nend_header\n"
    with path.open("wb") as f:
        f.write(header.encode())
        f.write(points.astype("<f4").tobytes())


def git_hash() -> str:
    """Return HEAD, marked dirty if the tree has uncommitted changes."""
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()  # noqa: S607
    dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, check=True).stdout.strip()  # noqa: S607
    return head + ("-dirty" if dirty else "")


def main() -> None:
    """Refine cached views one by one: `uv run python -m semantic_3d_refinement.refine --cfg configs/<name>.yaml`."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=Path, required=True)
    cfg_path: Path = parser.parse_args().cfg
    raw = yaml.safe_load(cfg_path.read_text())
    cfg = RefineConfig(**raw["refine"])
    device = torch.device(cfg.device if cfg.device != "cuda" or torch.cuda.is_available() else "cpu")

    data = {k: v.to(device) for k, v in load_views(Path(raw["input"]), raw.get("views")).items()}
    depth, rays, conf = data["depth"], data["rays"], data["conf"]

    results, seconds = [], []
    for i in range(len(depth)):
        t0 = time.perf_counter()
        results.append(refine(depth[i], rays[i], conf[i], cfg))
        if device.type == "cuda":
            torch.cuda.synchronize()
        seconds.append(time.perf_counter() - t0)
    depth_hat = torch.stack([r[0] for r in results])
    delta = torch.stack([r[2] for r in results])
    logs = [r[3] for r in results]

    out = Path(raw["output_dir"]) / f"{cfg_path.stem}-{time.strftime('%Y%m%d-%H%M%S')}"
    out.mkdir(parents=True)
    summary = {
        "config": raw,
        "resolved": dataclasses.asdict(cfg),
        "git": git_hash(),
        "views": len(depth),
        "refine_seconds": sum(seconds),
        "refine_seconds_per_view": [round(t, 3) for t in seconds],
        "evaluations": [len(log) - 1 for log in logs],
        "energy_initial": logs[0][0],
        "energy_final": logs[0][-1],
    }
    if "gt_depth" in data:
        summary |= {
            "metrics_input": depth_metrics(depth, rays, data["gt_depth"]),
            "metrics_refined": depth_metrics(depth_hat, rays, data["gt_depth"]),
        }
    if len(depth) > 1:
        t0 = time.perf_counter()
        cams = [fit_projection(r) for r in rays]
        mv_in, mv_out = multiview_error(depth, rays, cams), multiview_error(depth_hat, rays, cams)
        summary |= {
            "metrics_input": summary.get("metrics_input", {}) | multiview_metrics(mv_in),
            "metrics_refined": summary.get("metrics_refined", {}) | multiview_metrics(mv_out),
            "eval_seconds": time.perf_counter() - t0,
        }
        np.savez(out / "multiview_error.npz", input=mv_in.cpu().numpy(), refined=mv_out.cpu().numpy())
    if "gt_views" in raw:
        cams_gt = camera_alignment(rays, Path(raw["gt_views"]), raw.get("views"))
        shift = (points_from_rays(depth_hat, rays) - points_from_rays(depth, rays)).norm(dim=-1)
        summary |= {"cameras": cams_gt, "mean_point_shift_mm": 1000 * cams_gt["scale_to_m"] * shift.mean().item()}
    (out / "run.yaml").write_text(yaml.safe_dump(summary, sort_keys=False))
    length = max(len(log) for log in logs)
    energy_log = np.full((len(logs), length, len(LOG_KEYS)), np.nan)
    for i, log in enumerate(logs):
        energy_log[i, : len(log)] = [[e[k] for k in LOG_KEYS] for e in log]
    np.savez(out / "refined.npz", depth=depth_hat.cpu().numpy(), delta=delta.cpu().numpy(), energy=energy_log)
    write_ply(out / "input.ply", points_from_rays(depth, rays).reshape(-1, 3).cpu().numpy())
    write_ply(out / "refined.ply", points_from_rays(depth_hat, rays).reshape(-1, 3).cpu().numpy())
    print(yaml.safe_dump(summary, sort_keys=False))  # noqa: T201
    print(f"wrote {out}")  # noqa: T201


if __name__ == "__main__":
    main()
