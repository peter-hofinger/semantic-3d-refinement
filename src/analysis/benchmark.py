"""L-BFGS against Gauss-Newton + CG on the same energy, for every view alone and for all views together.

Two setups on each scene (ETH3D or ScanNet++), 20 views of 504 x 336:

- monocular: E_d + E_n, λ_mv = 0, so the views are independent problems stacked into one
  vector;
- multi-view: E_d + E_n + E_mv, coupled by the cross-view matches.

Both solvers minimize `multiview.scene_residuals` squared, on one fixed
`multiview.Problem` per setup (DSINE normals, the occlusion cuts, the matches), from δ =
0; only the optimizer differs. L-BFGS is torch's, as in `monocular.py` (strong Wolfe line
search, 20 pairs of history), on the autograd gradient; Gauss-Newton is
`multiview.gauss_newton`. Each is traced (energy, wall time, iterations) until it
converges: Gauss-Newton for `gn_iters` steps, L-BFGS until E changes by less than
`rel_change` over `chunk` iterations, or its iteration or time cap. A solver has
converged once E is within `converged_frac` of E* (relative), E* the lowest energy
either reached.

Accuracy is the fused cloud's (`geometry.fused_metrics`): all views' points mapped by
the camera similarity, the distance of each sampled pixel's point to its nearest scan
point. It is scored at each solver's end, at Gauss-Newton's operating point (10 steps,
as in `eval_multiview.yaml`), and for L-BFGS after the same wall time.

`uv run python -m src.analysis.benchmark --cfg configs/benchmark_solvers.yaml` writes
`runs/<cfg>-<time>/{run.yaml, traces.npz}` and the figure and README to `figures/<cfg>-<time>/`.
"""

import argparse
import dataclasses
import math
import time
from collections.abc import Callable
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from torch import Tensor

from src.analysis.visualize import BLUE, GRID, INK, INK_2, SURFACE
from src.evaluation.geometry import Truth, fused_metrics, load_truth
from src.refinement.inputs import git_hash
from src.refinement.monocular import RefineConfig
from src.refinement.multiview import (
    MultiviewConfig,
    Problem,
    Scene,
    build_problem,
    gauss_newton,
    load_scene,
    scene_residuals,
)
from src.refinement.normals import predicted_normals

SOLVER_COLORS = {"L-BFGS": INK_2, "Gauss-Newton + CG": BLUE}


def energy_fn(
    cfg: RefineConfig, mv: MultiviewConfig, p: Problem
) -> Callable[[Tensor], Tensor]:
    """E(δ) as a differentiable scalar, equal to `multiview.scene_energy`'s E."""

    def energy(delta: Tensor) -> Tensor:
        return (
            torch.cat(list(scene_residuals(delta, cfg, mv, p).values())).square().sum()
            / p.d0.numel()
        )

    return energy


def synced() -> float:
    """Wall time after the GPU has finished its queued work."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter()


def run_gauss_newton(
    p: Problem, cfg: RefineConfig, mv: MultiviewConfig, operating_point: int
) -> dict:
    """Trace Gauss-Newton: per step the time, energy and CG iterations; δ at the operating point and at the end."""
    trace = {"seconds": [0.0], "energy": [], "steps": [0], "cg": [0]}
    kept = {}
    t0 = synced()

    def on_step(delta: Tensor, energies: dict, cg_total: int) -> None:
        trace["seconds"].append(synced() - t0)
        trace["energy"].append(energies["E"])
        trace["steps"].append(trace["steps"][-1] + 1)
        trace["cg"].append(cg_total)
        if trace["steps"][-1] == operating_point:
            kept["operating_point"] = delta.clone()

    delta, log, _ = gauss_newton(p, cfg, mv, on_step)
    trace["energy"].insert(0, log[0]["E"])
    return {"trace": {k: np.array(v) for k, v in trace.items()}, "delta": delta, **kept}


def run_lbfgs(
    p: Problem, cfg: RefineConfig, mv: MultiviewConfig, raw: dict, t: float
) -> dict:
    """Trace L-BFGS every `chunk` iterations until E settles or hits a cap; keep δ at the first reading past t."""
    lb = raw["lbfgs"]
    energy = energy_fn(cfg, mv, p)
    delta = torch.zeros_like(p.d0, requires_grad=True)
    opt = torch.optim.LBFGS(
        [delta],
        lr=1.0,
        max_iter=lb["chunk"],
        history_size=lb["history_size"],
        tolerance_grad=1e-12,
        tolerance_change=1e-15,
        line_search_fn="strong_wolfe",
    )

    def closure() -> Tensor:
        opt.zero_grad()
        e = energy(delta)
        e.backward()
        return e

    with torch.no_grad():
        trace = {
            "seconds": [0.0],
            "energy": [energy(delta).item()],
            "iterations": [0],
            "evaluations": [0],
        }
    kept, stopped = {}, "max_iters"
    t0 = synced()
    while trace["iterations"][-1] < lb["max_iters"]:
        opt.step(closure)
        state = opt.state[delta]
        with torch.no_grad():
            e = energy(delta).item()
        trace["seconds"].append(synced() - t0)
        trace["energy"].append(e)
        trace["iterations"].append(state["n_iter"])
        trace["evaluations"].append(state["func_evals"])
        if "at_gn_time" not in kept and trace["seconds"][-1] >= t:
            kept["at_gn_time"] = delta.detach().clone()
            kept["at_gn_time_seconds"] = trace["seconds"][-1]
        previous = trace["energy"][-2]
        if not math.isfinite(e):
            stopped = "diverged"
            break
        if abs(previous - e) <= lb["rel_change"] * abs(previous):
            stopped = "converged"
            break
        if trace["seconds"][-1] > lb["max_seconds"]:
            stopped = "max_seconds"
            break
    return {
        "trace": {k: np.array(v) for k, v in trace.items()},
        "delta": delta.detach(),
        "stopped": stopped,
        **kept,
    }


def converged_at(trace: dict, threshold: float, counters: list[str]) -> dict:
    """Time and counters at the first reading with E at or below the threshold; None if it never gets there."""
    hit = np.flatnonzero(trace["energy"] <= threshold)
    if len(hit) == 0:
        return {"seconds": None} | dict.fromkeys(counters)
    i = int(hit[0])
    return {"seconds": float(trace["seconds"][i])} | {
        c: float(trace[c][i]) for c in counters
    }


def score(
    delta: Tensor, p: Problem, rays: np.ndarray, truth: object, raw: dict
) -> dict:
    """Fused-cloud accuracy and fidelity of a δ; flags depths that left the positive range."""
    d = p.d0 + delta
    if not bool((d > 0).all()):
        return {"valid": False}
    depth = (p.scale / d).cpu().numpy().astype(np.float64)
    m = fused_metrics(depth, rays, truth, raw["tolerances_cm"])
    keys = (
        "fused_mean_cm",
        "fused_median_cm",
        "fused_acc_1cm",
        "fused_acc_2cm",
        "thickness_mm",
        "normal_angle_deg",
    )
    return {"valid": True} | {k: float(m[k]) for k in keys}


def figure(out: Path, results: dict, frac: float) -> None:
    """Per scene and setup: energy gap to E* against wall time, both solvers, with the convergence threshold."""
    scenes, setups = list(results), list(next(iter(results.values())))
    fig, axes = plt.subplots(
        len(scenes),
        len(setups),
        figsize=(6.2 * len(setups), 3.9 * len(scenes)),
        constrained_layout=True,
        squeeze=False,
    )
    for r, scene in enumerate(scenes):
        for c, setup in enumerate(setups):
            ax, res = axes[r, c], results[scene][setup]
            best = res["E_star"]
            for solver, run in res["solvers"].items():
                t = run["trace"]
                gap = np.maximum((t["energy"] - best) / best, 1e-6)
                ax.plot(
                    t["seconds"],
                    gap,
                    color=SOLVER_COLORS[solver],
                    lw=2.2,
                    marker="o",
                    ms=2.5,
                    label=solver,
                )
                conv = run["converged"]["seconds"]
                if conv is not None:
                    ax.axvline(conv, color=SOLVER_COLORS[solver], lw=1, ls=":")
            op = res["solvers"]["Gauss-Newton + CG"]["operating_point"]
            ax.plot(
                op["seconds"],
                max((op["E"] - best) / best, 1e-6),
                marker="*",
                ms=14,
                color=BLUE,
                ls="",
                label="Gauss-Newton, 10 steps (used)",
            )
            ax.axhline(frac, color=GRID, lw=1.2, ls="--")
            ax.set(
                yscale="log",
                xscale="log",
                xlim=(0.1, None),
                ylim=(1e-6, 1.5),
                xlabel="seconds",
                ylabel="(E - E*) / E*",
            )
            ax.set_title(f"{scene}: {res['label']}", fontsize=11, color=INK)
            if r == 0 and c == 0:
                ax.legend(frameon=False, fontsize=9)
    fig.savefig(out / "convergence.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)


def table(results: dict) -> list[str]:
    """Markdown: per scene and setup, both solvers' effort to converge, their end, and the accuracy there."""

    def num(x: float | None, fmt: str) -> str:
        return "-" if x is None else format(x, fmt)

    header = (
        "| scene | setup | solver | iterations to converge | seconds to converge | seconds run | final E / E* "
        "| mean dist (cm) | median (cm) | ≤ 1 cm | thickness (mm) |"
    )
    lines = [header, "|---" * 11 + "|"]
    for scene, setups in results.items():
        for res in setups.values():
            for solver, run in res["solvers"].items():
                conv, m = run["converged"], run["final_metrics"]
                counts = (
                    ("steps", "CG", "cg")
                    if solver != "L-BFGS"
                    else ("it", "evals", "evaluations")
                )
                first = conv["steps"] if solver != "L-BFGS" else conv["iterations"]
                its = f"{num(first, '.0f')} {counts[0]} / {num(conv[counts[2]], '.0f')} {counts[1]}"
                acc = (
                    f"{m['fused_mean_cm']:.3f} | {m['fused_median_cm']:.3f} | {100 * m['fused_acc_1cm']:.1f}% | "
                    f"{m['thickness_mm']:.2f}"
                    if m["valid"]
                    else "invalid | | |"
                )
                ratio = run["E_final"] / res["E_star"]
                lines.append(
                    f"| {scene} | {res['label']} | {solver} | {its} | {num(conv['seconds'], '.2f')} | "
                    f"{run['seconds_total']:.2f} | {ratio:.4f} | {acc} |"
                )
            op = res["solvers"]["Gauss-Newton + CG"]["operating_point"]
            at = res["solvers"]["L-BFGS"]["at_gn_time"]
            for solver, point in (
                ("Gauss-Newton + CG, 10 steps (used)", op),
                ("L-BFGS, same wall time", at),
            ):
                m = point["metrics"]
                acc = (
                    f"{m['fused_mean_cm']:.3f} | {m['fused_median_cm']:.3f} | {100 * m['fused_acc_1cm']:.1f}% | "
                    f"{m['thickness_mm']:.2f}"
                    if m.get("valid")
                    else "- | | |"
                )
                ratio = point["E"] / res["E_star"] if point["E"] is not None else None
                lines.append(
                    f"| {scene} | {res['label']} | {solver} | | | {num(point['seconds'], '.2f')} | "
                    f"{num(ratio, '.4f')} | {acc} |"
                )
    return lines


def benchmark_setup(
    data: tuple[Scene, Tensor, Truth],
    cfg: RefineConfig,
    raw: dict,
    setup: tuple[str, float],
) -> dict:
    """Run both solvers on one setup of one scene; return their traces, convergence, end points and accuracy."""
    (scene, normals, truth), (name, lambda_mv) = data, setup
    rays = scene.rays.cpu().numpy().astype(np.float64)
    mv = MultiviewConfig(
        **raw["multiview"], lambda_mv=lambda_mv, gn_iters=raw["gn_iters"]
    )
    t0 = synced()
    p = build_problem(scene, cfg, mv, normals)
    setup_seconds = synced() - t0
    warm = energy_fn(cfg, mv, p)(torch.zeros_like(p.d0, requires_grad=True))
    warm.backward()
    gauss_newton(p, cfg, dataclasses.replace(mv, gn_iters=1, cg_iters=1))

    gn = run_gauss_newton(p, cfg, mv, raw["gn_operating_point"])
    op_step = raw["gn_operating_point"]
    op_seconds = (
        float(gn["trace"]["seconds"][op_step])
        if len(gn["trace"]["seconds"]) > op_step
        else None
    )
    lb = run_lbfgs(p, cfg, mv, raw, op_seconds if op_seconds is not None else math.inf)
    e0 = float(gn["trace"]["energy"][0])
    best = min(float(gn["trace"]["energy"].min()), float(lb["trace"]["energy"].min()))
    threshold = best * (1 + raw["converged_frac"])
    label = "E_d + E_n" + (" + E_mv" if lambda_mv > 0 else "") + f" ({name})"
    gn_op = {
        "seconds": op_seconds,
        "E": float(gn["trace"]["energy"][op_step]) if op_seconds else None,
        "metrics": score(gn["operating_point"], p, rays, truth, raw)
        if op_seconds
        else {},
    }
    at = lb.get("at_gn_time")
    lb_at = {
        "seconds": lb.get("at_gn_time_seconds"),
        "E": energy_fn(cfg, mv, p)(at).item() if at is not None else None,
        "metrics": score(at, p, rays, truth, raw) if at is not None else {},
    }
    return {
        "label": label,
        "lambda_mv": lambda_mv,
        "setup_seconds": setup_seconds,
        "matches": len(p.matches.src),
        "E0": e0,
        "E_star": best,
        "threshold": threshold,
        "solvers": {
            "Gauss-Newton + CG": {
                "trace": gn["trace"],
                "converged": converged_at(gn["trace"], threshold, ["steps", "cg"]),
                "seconds_total": float(gn["trace"]["seconds"][-1]),
                "E_final": float(gn["trace"]["energy"][-1]),
                "steps": int(gn["trace"]["steps"][-1]),
                "cg": int(gn["trace"]["cg"][-1]),
                "final_metrics": score(gn["delta"], p, rays, truth, raw),
                "operating_point": gn_op,
            },
            "L-BFGS": {
                "trace": lb["trace"],
                "converged": converged_at(
                    lb["trace"], threshold, ["iterations", "evaluations"]
                ),
                "seconds_total": float(lb["trace"]["seconds"][-1]),
                "E_final": float(lb["trace"]["energy"][-1]),
                "iterations": int(lb["trace"]["iterations"][-1]),
                "evaluations": int(lb["trace"]["evaluations"][-1]),
                "stopped": lb["stopped"],
                "final_metrics": score(lb["delta"], p, rays, truth, raw),
                "at_gn_time": lb_at,
            },
        },
    }


def main() -> None:
    """Benchmark both solvers: `uv run python -m src.analysis.benchmark --cfg <cfg>`."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=Path, required=True)
    cfg_path: Path = parser.parse_args().cfg
    raw = yaml.safe_load(cfg_path.read_text())
    cfg = RefineConfig(**raw["refine"])
    stamp, commit = f"{cfg_path.stem}-{time.strftime('%Y%m%d-%H%M%S')}", git_hash()
    run_dir, out = Path(raw["output_dir"]) / stamp, Path("figures") / stamp
    run_dir.mkdir(parents=True)
    out.mkdir(parents=True)
    device = torch.device(cfg.device)

    results, traces = {}, {}
    for name, paths in raw["scenes"].items():
        scene = load_scene(Path(paths["input"])).to(device)
        normals = torch.stack(
            [
                predicted_normals(
                    {**raw, **paths, "view": v},
                    {"depth": scene.depth[v], "rays": scene.rays[v]},
                )[0]
                for v in range(len(scene.depth))
            ]
        )
        truth = load_truth(paths, scene)
        results[name] = {}
        for setup, lambda_mv in raw["setups"].items():
            res = benchmark_setup((scene, normals, truth), cfg, raw, (setup, lambda_mv))
            results[name][setup] = res
            for solver, run in res["solvers"].items():
                for k, v in run["trace"].items():
                    traces[f"{name}/{setup}/{solver}/{k}"] = v
                summary = {k: run[k] for k in ("converged", "seconds_total", "E_final")}
                print(name, setup, solver, summary, run["final_metrics"], flush=True)  # noqa: T201
            torch.cuda.empty_cache()

    def untraced(r: dict) -> dict:
        solvers = {
            sv: {k: v for k, v in run.items() if k != "trace"}
            for sv, run in r["solvers"].items()
        }
        return {**{k: v for k, v in r.items() if k != "solvers"}, "solvers": solvers}

    plain = {n: {s: untraced(r) for s, r in sc.items()} for n, sc in results.items()}
    (run_dir / "run.yaml").write_text(
        yaml.safe_dump(
            {"config": raw, "git": commit, "scenes": plain},
            sort_keys=False,
            allow_unicode=True,
        )
    )
    np.savez_compressed(run_dir / "traces.npz", **traces)
    figure(out, results, raw["converged_frac"])
    lines = [
        f"# {stamp}",
        "",
        f"git `{commit}`",
        "",
        f"{raw['credit']} The data itself is not in this repository.",
        "",
        (
            "Both solvers minimize the same energy on the same problem from δ = 0 (see `benchmark.py`). Iterations: "
            "Gauss-Newton steps / CG iterations (one J^T J product each); L-BFGS iterations / energy-and-gradient "
            "evaluations. Converged: E within 1 % of E*, the lowest energy either solver reached. Accuracy: fused "
            "cloud of all 20 views, distance of each sampled pixel's point to its nearest scan point."
        ),
        "",
        *table(results),
        "",
        "## convergence",
        "",
        "![convergence](convergence.png)",
        "",
    ]
    (out / "README.md").write_text("\n".join(lines))

    print(f"wrote {run_dir} and {out}")  # noqa: T201


if __name__ == "__main__":
    main()
