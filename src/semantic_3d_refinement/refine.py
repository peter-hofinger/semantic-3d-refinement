"""Monocular variational refinement of one depth map: a dense residual δ on normalized inverse depth, by L-BFGS.

  d = d0 + δ,   d0 = median(D0) / D0,   z = -log d (log depth up to a constant),   D = median(D0) / d

  E(δ) = λ_d E_d + λ_bd E_bd + λ_n E_n + λ_δ ‖δ‖²,   each a mean over pixels:

- E_d = conf-weighted Huber(δ): anchor to the prediction.
- E_bd = d_xx² + d_yy² + 2 d_xy²: thin-plate bending energy on inverse depth, zero on every plane, since a plane's
  inverse depth is affine in pixel coordinates.
- E_n: BiNI's normal term (Cao et al. 2022, as in `normals.py`) on log depth, where a normal fixes the slope
  independently of the plane's distance: on each edge (p, q), w+ (a_p dz + g_p)² + w- (a_q dz + g_q)².

With normals given, edges crossing the prior's occlusion contours (`normals.cut_weights`) are cut from both E_n
and E_bd. Unlike BiNI, which alternates its weights with a CG solve of a quadratic, the weights stay fixed, so
E is one smooth energy and the robust data term costs nothing extra.
"""

import argparse
import dataclasses
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
import yaml
from torch import Tensor

from semantic_3d_refinement.normals import angle_to, cut_weights, diffs, drift, neighbor_terms, predicted_normals
from semantic_3d_refinement.views import check_inputs, git_hash, view_from_config, write_ply


@dataclass(frozen=True)
class RefineConfig:
    """Energy weights and solver settings; inverse depth is normalized by the median depth."""

    iters: int = 200
    huber: float = 0.05
    lambda_d: float = 1.0
    lambda_bd: float = 10.0
    lambda_n: float = 0.0
    lambda_delta: float = 0.0
    edge_tol: float = 0.02
    history_size: int = 20
    device: str = "cuda"

    def __post_init__(self) -> None:
        """Check ranges."""
        assert self.iters >= 0
        assert self.huber > 0
        assert self.lambda_d >= 0 and self.lambda_bd >= 0 and self.lambda_n >= 0 and self.lambda_delta >= 0
        assert self.edge_tol > 0
        assert self.history_size > 0


def points_from_rays(depth: Tensor, rays: Tensor) -> Tensor:
    """Lift depth along rays: x = origin + depth * direction."""
    return rays[..., :3] + depth[..., None] * rays[..., 3:]


def second_differences(d: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Return d_xx, d_yy, d_xy; all vanish where d is affine in pixel coordinates (planes)."""
    dxx = d[:, 2:] - 2 * d[:, 1:-1] + d[:, :-2]
    dyy = d[2:] - 2 * d[1:-1] + d[:-2]
    dxy = d[1:, 1:] - d[1:, :-1] - d[:-1, 1:] + d[:-1, :-1]
    return dxx, dyy, dxy


@dataclass(frozen=True)
class NormalTerm:
    """Per axis (x, y), BiNI's edge coefficients (a_p, g_p, a_q, g_q) and the weights (w+, w-), 0 on cut edges."""

    terms: list[tuple[Tensor, Tensor, Tensor, Tensor]]
    weights: list[tuple[Tensor, Tensor]]


def normal_term(normals: Tensor, rays: Tensor, d0: Tensor, edge_tol: float) -> NormalTerm:
    """BiNI's residuals for the given normals, cut on the occlusion contours of the prior d0 (tol in log depth)."""
    h, w = d0.shape
    assert normals.shape == (h, w, 3) and normals.isfinite().all(), normals.shape
    assert ((normals.norm(dim=-1) - 1).abs() < 1e-3).all()  # noqa: PLR2004
    _, terms = neighbor_terms(normals.double(), rays.double())
    weights = cut_weights(-d0.double().log(), terms, edge_tol)
    return NormalTerm(
        [tuple(t.to(d0.dtype) for t in axis) for axis in terms],
        [tuple(t.to(d0.dtype) for t in axis) for axis in weights],
    )


LOG_KEYS = ("E", "E_d", "E_bd", "E_n", "E_delta")


def normalize(depth: Tensor, conf: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Median depth scale, normalized inverse depth d0 = scale / depth, and mean-one confidence weights."""
    scale = depth.median()
    return scale, scale / depth, conf / conf.mean()


def term_weights(cfg: RefineConfig) -> dict[str, float]:
    """Weight of each energy term in E."""
    return {"E_d": cfg.lambda_d, "E_bd": cfg.lambda_bd, "E_n": cfg.lambda_n, "E_delta": cfg.lambda_delta}


def energy_maps(
    delta: Tensor, d0: Tensor, w: Tensor, cfg: RefineConfig, normal: NormalTerm | None = None
) -> dict[str, Tensor]:
    """Unweighted per-pixel cost of each term (an edge's on its upper-left pixel); their means are the terms of E.

    A second difference whose stencil crosses a cut edge is dropped. Log depth clamps d at 1e-6, so a line-search
    trial step past d = 0 costs a lot instead of returning NaN.
    """
    d = d0 + delta
    dxx, dyy, dxy = second_differences(d)
    normal_cost = torch.zeros_like(d)
    if normal is not None:
        (kx, _), (ky, _) = [(wf > 0, wb > 0) for wf, wb in normal.weights]
        dxx, dyy = dxx * (kx[:, :-1] & kx[:, 1:]), dyy * (ky[:-1] & ky[1:])
        dxy = dxy * (kx[:-1] & kx[1:] & ky[:, :-1] & ky[:, 1:])
        z = -d.clamp_min(1e-6).log()
        for (af, gf, ab, gb), (wf, wb), dz, pad in zip(
            normal.terms, normal.weights, diffs(z), ((0, 1), (0, 0, 0, 1)), strict=True
        ):
            normal_cost = normal_cost + F.pad(wf * (af * dz + gf).square() + wb * (ab * dz + gb).square(), pad)
    smooth = F.pad(dxx.square(), (1, 1)) + F.pad(dyy.square(), (0, 0, 1, 1)) + 2 * F.pad(dxy.square(), (0, 1, 0, 1))
    return {
        "E_d": w * F.huber_loss(delta, torch.zeros_like(delta), reduction="none", delta=cfg.huber),
        "E_bd": smooth,
        "E_n": normal_cost,
        "E_delta": delta.square(),
    }


def energy(
    delta: Tensor, d0: Tensor, w: Tensor, cfg: RefineConfig, normal: NormalTerm | None = None
) -> tuple[Tensor, dict[str, float]]:
    """E = lambda_d E_d + lambda_bd E_bd + lambda_n E_n + lambda_delta ||delta||^2, each a per-pixel mean."""
    terms = {k: v.mean() for k, v in energy_maps(delta, d0, w, cfg, normal).items()}
    e = sum(term_weights(cfg)[k] * terms[k] for k in terms)
    assert isinstance(e, Tensor) and e.isfinite(), e
    return e, {"E": e.item()} | {k: v.item() for k, v in terms.items()}


def refine(
    depth: Tensor,
    rays: Tensor,
    conf: Tensor,
    cfg: RefineConfig,
    normals: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor, list[dict[str, float]]]:
    """Optimize the dense residual delta on normalized inverse depth; return depth, points, delta, energy log.

    Normals (unit, in the frame of the rays) enable E_n and cut E_bd on the prior's occlusion contours.
    """
    check_inputs(depth, rays, conf)
    assert normals is not None or cfg.lambda_n == 0, "E_n needs normals"
    scale, d0, w = normalize(depth, conf)
    normal = None if normals is None else normal_term(normals, rays, d0, cfg.edge_tol)
    delta = torch.zeros_like(depth, requires_grad=True)

    log = [energy(delta, d0, w, cfg, normal)[1]]
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
            e, terms = energy(delta, d0, w, cfg, normal)
            e.backward()
            log.append(terms)
            return e

        opt.step(closure)
        with torch.no_grad():
            log.append(energy(delta, d0, w, cfg, normal)[1])
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


def view_metrics(depth_hat: Tensor, data: dict[str, Tensor], normals: Tensor | None) -> dict[str, float]:
    """Drift from the input, angle to the given normals, and the error against ground truth where it is known."""
    depth, rays = data["depth"], data["rays"]
    out = {"drift_pct": drift(depth_hat, depth)}
    if normals is not None:
        out["angle_to_normals_deg"] = angle_to(normals, depth_hat, rays)
    if "gt_depth" in data:
        out |= depth_metrics(depth_hat, rays, data["gt_depth"])
    if "gt_normals" in data:
        out["angle_to_gt_normals_deg"] = angle_to(data["gt_normals"], depth_hat, rays)
    return out


def main() -> None:
    """Refine one cached view: `uv run python -m semantic_3d_refinement.refine --cfg configs/<name>.yaml`."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=Path, required=True)
    cfg_path: Path = parser.parse_args().cfg
    raw = yaml.safe_load(cfg_path.read_text())
    cfg = RefineConfig(**raw["refine"])
    device = torch.device(cfg.device if cfg.device != "cuda" or torch.cuda.is_available() else "cpu")

    data = {k: v.to(device) for k, v in view_from_config(raw).items()}
    depth, rays = data["depth"], data["rays"]
    normals, normals_info = predicted_normals(raw, data) if "normals" in raw else (None, {})

    t0 = time.perf_counter()
    depth_hat, points, delta, log = refine(depth, rays, data["conf"], cfg, normals)
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
        **normals_info,
        "metrics_input": view_metrics(depth, data, normals),
        "metrics_refined": view_metrics(depth_hat, data, normals),
    }
    (out / "run.yaml").write_text(yaml.safe_dump(summary, sort_keys=False))
    np.savez(
        out / "refined.npz",
        depth=depth_hat.cpu().numpy(),
        delta=delta.cpu().numpy(),
        energy=np.array([[e[k] for k in LOG_KEYS] for e in log]),
        **({} if normals is None else {"normals": normals.cpu().numpy()}),
    )
    write_ply(out / "input.ply", points_from_rays(depth, rays).reshape(-1, 3).cpu().numpy())
    write_ply(out / "refined.ply", points.reshape(-1, 3).cpu().numpy())
    print(yaml.safe_dump(summary, sort_keys=False))  # noqa: T201
    print(f"wrote {out}")  # noqa: T201


if __name__ == "__main__":
    main()
