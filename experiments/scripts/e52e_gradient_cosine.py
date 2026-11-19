"""E52e: does a cross-view term descend towards the GT from Déjà View's output.

E52c's solver check found c's minima away from the GT, by two routes: directly, and by
shrinking the Sim(3) alignment (c was divided by the fixed GT depth after the fit). The
cheap test of a replacement, before any training, is first order: at Déjà View's depth,
the cosine between the gradient of the term and the gradient of the GT error. A step
along -grad c lowers the GT error to first order iff the cosine is positive.

Free variables as in `e52c_solver.py`: a log-scale per 14 px patch, zero at Déjà View's
depth, d' = d exp(u). Terms:

- built: E52c's c, on the points after the exact Sim(3), over the GT depth.
- invariant: the same point-to-plane distance on the raw points before any alignment,
  over the predicted depth d': |n_q . (x_p - x_q)| / d'_p. A similarity x -> s R x + t
  scales numerator and depth alike and leaves nearest partners and normal directions,
  so the term is independent of the Sim(3) (checked at run time) and route 2 is gone.

Each with partners in the same GT instance (gate) or any (twin). The target gradient is
the GT error (mean relative 3D error capped 0.5 after the exact Sim(3)) over all GT-valid
pixels. A term's gradient is the mean over DRAWS draws of 6000 pixels; the GT error on
the same draws is the ceiling a 6000-pixel estimate can reach, and a random direction the
floor (cosine ~ 0 +- 1 / sqrt(dim)). Per-draw cosines (what one training step sees) are
recorded too.

uv run python experiments/scripts/e52e_gradient_cosine.py
writes `experiments/results/e52e_gradient_cosine.yaml` and
`experiments/figures/e52e_gradient_cosine.jpg`; `--plot` redraws from the yaml.
"""

import argparse
import time
from pathlib import Path

import matplotlib as mpl
import numpy as np
import torch
import yaml
from torch import Tensor

mpl.use("Agg")
import matplotlib.pyplot as plt
from e52a_figures import FIGURES, style
from e52c_solver import CAP, PIXELS, SCENE, STRIDE, VIEWS, Problem, depth_of

from src.analysis.scene_eval import git_hash
from src.backbone.adapters import PATCH_SIZE, cross_view_distance, map_normals
from src.refinement.multiview import points_from_rays

DRAWS = 20
SIZES = (1e-3, 1e-2, 3e-2)  # rms of a step in log-depth; Adam's first step is lr * sign
TERMS = {"built": "E52c's c (after Sim(3), / GT depth)",
         "invariant": "scale-invariant c (before Sim(3), / predicted depth)"}  # fmt: skip
RESULTS = Path("experiments/results/e52e_gradient_cosine.yaml")


def invariant(
    problem: Problem, x: Tensor, depth: Tensor, at: Tensor, *, gate: bool
) -> Tensor:
    """Point-to-plane distance over predicted depth on raw points, drawn pixels."""
    normal = map_normals(x.detach(), STRIDE)[problem.valid][at]
    return cross_view_distance(
        x[problem.valid][at], normal, problem.view[at], problem.inst[at],
        depth[problem.valid][at], gate=gate, cap=CAP,
    )  # fmt: skip


def loss(
    problem: Problem, theta: Tensor, name: str, at: Tensor | None, *, gate: bool
) -> Tensor:
    """One loss at the patch log-scales `theta`; `at` None means all GT-valid pixels."""
    depth = depth_of(problem, "patch", theta)
    x = points_from_rays(depth, problem.rays)
    if name == "invariant":
        assert at is not None
        return invariant(problem, x, depth, at, gate=gate)
    aligned, _ = problem.align(x)
    if name == "built":
        assert at is not None
        return problem.distance(aligned, at, gate=gate)
    assert name == "gt", name
    error = problem.error(aligned)
    return error.mean() if at is None else error[at].mean()


def patches(problem: Problem) -> tuple[int, int, int]:
    """Shape of the patch log-scales."""
    v, h, w = problem.depth.shape
    return (v, -(-h // PATCH_SIZE), -(-w // PATCH_SIZE))


def gradient(problem: Problem, name: str, at: Tensor | None, *, gate: bool) -> Tensor:
    """Flat gradient of one loss with respect to the patch log-scales at zero."""
    shape = patches(problem)
    theta = torch.zeros(shape, device=problem.depth.device, requires_grad=True)
    loss(problem, theta, name, at, gate=gate).backward()
    assert theta.grad is not None
    assert torch.isfinite(theta.grad).all()
    return theta.grad.flatten().double()


def gt_after(problem: Problem, step: Tensor) -> float:
    """GT error after moving the log-scales by the flat `step`, over the error at zero."""
    shape = patches(problem)
    with torch.no_grad():
        zero = torch.zeros(shape, device=problem.depth.device)
        moved = step.float().reshape(shape)
        start = loss(problem, zero, "gt", None, gate=True)
        return float(loss(problem, moved, "gt", None, gate=True) / start)


def descent(problem: Problem, grad: Tensor) -> dict:
    """GT ratio after steps along -grad (scaled to rms 1) and -sign(grad), per size."""
    unit = grad / grad.pow(2).mean().sqrt()
    return {
        f"{kind}_{size:g}": gt_after(problem, -size * d)
        for kind, d in (("gradient", unit), ("sign", grad.sign()))
        for size in SIZES
    }


def cosine(a: Tensor, b: Tensor) -> float:
    """Cosine of two flat vectors."""
    return float(a @ b / (a.norm() * b.norm()))


def check_invariance(problem: Problem) -> float:
    """Relative change of the invariant term under a random similarity of the points."""
    depth = problem.depth
    x = points_from_rays(depth, problem.rays)
    q, _ = torch.linalg.qr(torch.randn(3, 3, device=x.device, generator=None))
    moved = 2.7 * x @ q.T + torch.tensor([1.0, -2.0, 0.5], device=x.device)
    at = problem.eval
    a = invariant(problem, x, depth, at, gate=True)
    b = invariant(problem, moved, 2.7 * depth, at, gate=True)
    return abs(float(b / a) - 1.0)


def figure(record: dict) -> None:
    """Left: cosines against ceiling and floor; right: GT ratio after a step per size."""
    fig, (ax, right) = plt.subplots(
        1, 2, figsize=(13, 4.6), gridspec_kw={"width_ratios": [1.3, 1]}
    )
    curves = {"GT error (ceiling)": (record["ceiling"], "#2a9d55", "-"),
              "built c, same instance": (record["terms"]["built_gate"], "#eb6834", "-"),
              "invariant c, same instance": (record["terms"]["invariant_gate"],
                                             "#eb6834", "--"),
              "invariant c, any partner": (record["terms"]["invariant_any"],
                                           "#3a7bd5", ":"),
              "random direction": ({"gt_ratio_after_step":
                                    record["random_gt_ratio_after_step"]}, "0.5", "-")}  # fmt: skip
    for label, (r, colour, ls) in curves.items():
        steps = r["gt_ratio_after_step"]
        right.plot(SIZES, [steps[f"gradient_{s:g}"] for s in SIZES], color=colour,
                   ls=ls, marker="o", ms=3, label=label)  # fmt: skip
    right.axhline(1.0, color="0.4", lw=0.6)
    right.set_xscale("log")
    right.set_xlabel("step along minus the gradient, rms change of log-depth")
    right.set_ylabel("GT error after the step / before (all GT pixels)")
    right.legend(fontsize=7, frameon=False)
    style(right)
    labels, means, spreads, colours = [], [], [], []
    for name, text in TERMS.items():
        for gate, arm in ((True, "same instance"), (False, "any partner")):
            r = record["terms"][f"{name}_{'gate' if gate else 'any'}"]
            labels.append(f"{text.split(' (')[0]}\n{arm}")
            means.append(r["cosine_mean_gradient"])
            spreads.append(r["cosine_per_draw_std"])
            colours.append("#eb6834" if gate else "#3a7bd5")
    pos = np.arange(len(labels))
    ax.bar(pos, means, color=colours, width=0.6)
    for p, r in zip(pos, record["terms"].values(), strict=True):
        ax.plot([p, p], [r["cosine_per_draw_mean"] - r["cosine_per_draw_std"],
                         r["cosine_per_draw_mean"] + r["cosine_per_draw_std"]],
                color="0.2", lw=1)  # fmt: skip
        ax.plot(p, r["cosine_per_draw_mean"], "o", color="0.2", ms=3)
    ceiling = record["ceiling"]["cosine_mean_gradient"]
    floor = record["floor_std"]
    ax.axhline(ceiling, color="#2a9d55", ls="--", lw=1)
    ax.text(len(labels) - 0.5, ceiling, " GT error on the same draws",
            color="#2a9d55", fontsize=8, va="bottom", ha="right")  # fmt: skip
    ax.axhspan(-2 * floor, 2 * floor, color="0.85")
    ax.axhline(0, color="0.4", lw=0.6)
    ax.set_xticks(pos, labels, fontsize=8)
    ax.set_ylabel("cosine with the GT error's gradient")
    ax.set_ylim(-1, 1)
    style(ax)
    fig.suptitle(
        "Does a cross-view term point towards the GT at Déjà View's depth? Bedroom, 12 "
        "views, log-scale per 14 px patch.\nLeft: cosine with the GT error's gradient; "
        f"bars: gradient averaged over {DRAWS} draws of {PIXELS} px, dots: one draw "
        "(mean +- std), grey band: random direction (+- 2 std).\nRight: the term's step "
        "lowers the GT error only at 0.1% of depth (by 0.5%); at 1% it raises it almost "
        "as much as a random step.",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(FIGURES / "e52e_gradient_cosine.jpg", dpi=150)


def main() -> None:
    """Gradients of every term and of the GT error at Déjà View's depth; their cosines."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plot", action="store_true", help="redraw from the yaml")
    args = parser.parse_args()
    if args.plot:
        figure(yaml.safe_load(RESULTS.read_text(encoding="utf-8")))
        return
    cfg = yaml.safe_load(Path("configs/e52a_adapters.yaml").read_text(encoding="utf-8"))
    raw = yaml.safe_load(Path(cfg["eval"]).read_text(encoding="utf-8"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    start = time.perf_counter()
    problem = Problem(raw, device)
    drift = check_invariance(problem)
    assert drift < 1e-3, drift
    target = gradient(problem, "gt", None, gate=True)
    rng = np.random.default_rng(3)
    draws = [
        torch.from_numpy(rng.choice(problem.n, PIXELS, replace=False)).to(device)
        for _ in range(DRAWS)
    ]
    arms = [("gt", True)] + [(n, g) for n in TERMS for g in (True, False)]
    record: dict = {"terms": {}}
    for name, gate in arms:
        grads = [gradient(problem, name, at, gate=gate) for at in draws]
        per = [cosine(g, target) for g in grads]
        mean = torch.stack(grads).mean(0)
        r = {
            "cosine_mean_gradient": cosine(mean, target),
            "cosine_per_draw_mean": float(np.mean(per)),
            "cosine_per_draw_std": float(np.std(per)),
            "cosine_sign": cosine(mean.sign(), target),
            "gt_ratio_after_step": descent(problem, mean),
        }
        key = "ceiling" if name == "gt" else f"{name}_{'gate' if gate else 'any'}"
        (record if name == "gt" else record["terms"])[key] = r
        print(key, r, flush=True)
    dim = target.numel()
    record["random_gt_ratio_after_step"] = descent(
        problem, torch.randn(dim, dtype=torch.float64, device=device)
    )
    print("random", record["random_gt_ratio_after_step"], flush=True)
    random = [cosine(torch.randn(dim, dtype=torch.float64, device=device), target)
              for _ in range(200)]  # fmt: skip
    record["floor_std"] = float(np.std(random))
    record["dim"] = dim
    record["invariance_drift"] = drift
    record["seconds"] = time.perf_counter() - start
    settings = {"scene": SCENE, "views": VIEWS, "pixels": PIXELS, "draws": DRAWS,
                "cap": CAP, "stride": STRIDE, "device": str(device)}  # fmt: skip
    record = {"git": git_hash(), "settings": settings, **record}
    RESULTS.write_text(yaml.safe_dump(record, sort_keys=False), encoding="utf-8")
    print(f"floor std {record['floor_std']:.4f}, dim {dim}, {record['seconds']:.0f} s")
    figure(record)


if __name__ == "__main__":
    main()
