"""Monocular variational refinement of one depth map: dense inverse-depth residual only."""

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


def load_view(path: Path, view: int | None = None) -> dict[str, Tensor]:
    """Load one cached view as (H, W[, C]) float tensors: depth, rays, conf, and gt_depth when known.

    Accepts the synthetic single-view format and the Déjà View export (ray_origins, ray_directions, confidence),
    from which `view` picks one frame.
    """
    data = dict(np.load(path))
    if "ray_origins" in data:
        assert view is not None, "a Déjà View export holds several frames; set `view` in the config"
        data = {
            "depth": data["depth"][view],
            "rays": np.concatenate([data["ray_origins"][view], data["ray_directions"][view]], axis=-1),
            "conf": data["confidence"][view],
        }
    data.setdefault("conf", np.ones_like(data["depth"]))
    out = {k: torch.from_numpy(np.ascontiguousarray(v)).float() for k, v in data.items()}
    check_inputs(out["depth"], out["rays"], out["conf"])
    return out


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
    """Refine one cached view: `uv run python -m semantic_3d_refinement.refine --cfg configs/<name>.yaml`."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=Path, required=True)
    cfg_path: Path = parser.parse_args().cfg
    raw = yaml.safe_load(cfg_path.read_text())
    cfg = RefineConfig(**raw["refine"])
    device = torch.device(cfg.device if cfg.device != "cuda" or torch.cuda.is_available() else "cpu")

    data = {k: v.to(device) for k, v in load_view(Path(raw["input"]), raw.get("view")).items()}
    depth, rays = data["depth"], data["rays"]

    t0 = time.perf_counter()
    depth_hat, points, delta, log = refine(depth, rays, data["conf"], cfg)
    if device.type == "cuda":
        torch.cuda.synchronize()
    seconds = time.perf_counter() - t0

    out = Path(raw["output_dir"]) / f"{cfg_path.stem}-{time.strftime('%Y%m%d-%H%M%S')}"
    out.mkdir(parents=True)
    summary = {
        "config": raw,
        "resolved": dataclasses.asdict(cfg),
        "git": git_hash(),
        "refine_seconds": seconds,
        "evaluations": len(log) - 1,
        "energy_initial": log[0],
        "energy_final": log[-1],
    }
    if "gt_depth" in data:
        summary |= {
            "metrics_input": depth_metrics(depth, rays, data["gt_depth"]),
            "metrics_refined": depth_metrics(depth_hat, rays, data["gt_depth"]),
        }
    (out / "run.yaml").write_text(yaml.safe_dump(summary, sort_keys=False))
    np.savez(
        out / "refined.npz",
        depth=depth_hat.cpu().numpy(),
        delta=delta.cpu().numpy(),
        energy=np.array([[e[k] for k in LOG_KEYS] for e in log]),
    )
    write_ply(out / "input.ply", points_from_rays(depth, rays).reshape(-1, 3).cpu().numpy())
    write_ply(out / "refined.ply", points.reshape(-1, 3).cpu().numpy())
    print(yaml.safe_dump(summary, sort_keys=False))  # noqa: T201
    print(f"wrote {out}")  # noqa: T201


if __name__ == "__main__":
    main()
