"""E52d, figure and compact raw output: why E52a's adapters diverge (issue #52).

Reads the six arms' `e52a.yaml` of `e52a_segment_adapters.py` (pool-grid, A as E52a's
third run, B fp32, C weight decay on B_k, D gradient clipping, E cosine lr decay, F the
exact gradient through a differentiable Sim(3), run on the 3080 while A-E ran on a 5090)
and the third run's grid models for the identity check.

1. `figures/e52d_training.jpg`: per arm over training steps, the held-out error (over Déjà
   View's, mean of 16 scenes), the training loss (over Déjà View's, mean of 250 steps;
   every arm draws the same scenes and views per step), the norm of B_k (all 16 steps) and
   the median gradient norm of B_k over 250 steps.
2. `results/e52d_train.yaml`: held-out curves, parameter norms, clip counts and the per-step
   series as means over 50-step windows (the per-step series stay in the runs).

3. With `--gradcheck <run>` (`e52d_gradient_check.py`): `figures/e52d_gradcheck.jpg`, per
   adapter state and draw the cosine of the detached and the exact gradient, and the
   finite-difference ratios (measured over promised slope) of the three checks; the run's
   records copied to `results/e52d_gradcheck.yaml`.

uv run python experiments/scripts/e52d_figures.py runs/e52d runs/e52a3_grid --gradcheck runs/e52d_gradcheck
"""

import argparse
from pathlib import Path

import matplotlib as mpl
import numpy as np
import yaml

mpl.use("Agg")
import matplotlib.pyplot as plt
from e52a_figures import FIGURES, style

ARMS = {  # arm -> label, colour
    "A": ("A: as the third run", "#444444"),
    "B": ("B: fp32 forward", "#3a7bd5"),
    "C": ("C: weight decay on B_k", "#2a9d55"),
    "D": ("D: gradient clipping", "#9b59b6"),
    "E": ("E: cosine lr decay", "#eb6834"),
    "F": ("F: exact Sim(3) gradient (3080)", "#d62f5f"),
}
SMOOTH = 250  # steps per training-loss mean and gradient median
WINDOW = 50  # steps per mean in the compact raw output
RESULTS = Path("experiments/results")


def windows(x: np.ndarray, n: int, fn: np.ufunc = np.mean) -> np.ndarray:
    """Reduce a per-step series to one value per n consecutive steps."""
    assert len(x) % n == 0, (len(x), n)
    return fn(x.reshape(-1, n), axis=1)


def gradcheck(run: Path) -> None:
    """Draw the gradient check per state; copy its records to the results."""
    text = (run / "e52d_gradcheck.yaml").read_text(encoding="utf-8")
    rec = yaml.safe_load(text)["records"]
    states = list(rec["mean_gradient"])
    scenes = sorted({r["scene"] for r in rec["rows"]})
    marks = dict(zip(scenes, "os^", strict=True))
    checks = (("autograd", "autograd: S frozen, along -g"),
              ("exact", "exact: S re-fit, along -g*"),
              ("detached", "detached: S re-fit, along -g (training)"))  # fmt: skip
    fig, (left, right) = plt.subplots(1, 2, figsize=(13, 4.4))
    for i, state in enumerate(states):
        rows = [r for r in rec["rows"] if r["state"] == state]
        for r in rows:
            left.scatter(i, r["cosine"], marker=marks[r["scene"]], color="#3a7bd5",
                         alpha=0.7, s=30)  # fmt: skip
            for j, (name, _) in enumerate(checks):
                right.scatter(i + (j - 1) * 0.22, r[f"{name}_ratio_0.001"],
                              marker=marks[r["scene"]], s=30, alpha=0.7,
                              color=("0.5", "#2a9d55", "#eb6834")[j])  # fmt: skip
        left.scatter(i, rec["mean_gradient"][state]["cosine"], marker="_", s=900,
                     color="#eb6834", lw=2.5)  # fmt: skip
    for ax in (left, right):
        style(ax)
        ax.set_xticks(
            range(len(states)), ["B_k = 0", "kept (step 1000)", "kept, B_k x 2"]
        )
    for name in scenes:
        left.scatter(
            [], [], marker=marks[name], color="0.4", label=name.split("_cam")[0]
        )
    left.scatter([], [], marker="_", s=300, color="#eb6834", lw=2.5,
                 label="mean gradients over the 9 draws")  # fmt: skip
    for j, (_, label) in enumerate(checks):
        right.scatter([], [], color=("0.5", "#2a9d55", "#eb6834")[j], label=label)
    left.set_ylim(0, 1.02)
    left.set_title("cosine of the detached gradient g and the exact g*\n(per draw: "
                   "4 views, 20k alignment pixels)", fontsize=10)  # fmt: skip
    right.axhline(1, color="0.3", lw=0.8)
    right.axhline(0, color="0.3", lw=0.8)
    right.set_title("finite-difference slope over the slope the gradient promises\n"
                    "(step 1e-3 in parameter norm; 1 = exact)", fontsize=10)  # fmt: skip
    left.legend(fontsize=8, frameon=False, loc="lower left")
    right.legend(fontsize=8, frameon=False, loc="center right")
    fig.suptitle("E52d gradient check: backprop and the exact gradient are right; the "
                 "detached Sim(3) gradient gives 3-100% of its promised descent and turns "
                 "away from g* as B_k grows", fontsize=11, y=1.03)  # fmt: skip
    fig.savefig(FIGURES / "e52d_gradcheck.jpg", dpi=110, bbox_inches="tight")
    (RESULTS / "e52d_gradcheck.yaml").write_text(
        "# E52d gradient check: laptop RTX 3050, fp32, e52d_gradient_check.py\n" + text,
        encoding="utf-8",
    )


def main() -> None:
    """Load the arms, draw the figure, write the compact raw output."""
    parser = argparse.ArgumentParser()
    parser.add_argument("prefix", type=Path, help="runs/e52d -> runs/e52d_A ... _E")
    parser.add_argument("third", type=Path, help="E52a's third run, grid models")
    parser.add_argument("--gradcheck", type=Path, help="e52d_gradient_check.py's run")
    args = parser.parse_args()
    if args.gradcheck:
        gradcheck(args.gradcheck)
    runs = {
        a: yaml.safe_load(
            Path(f"{args.prefix}_{a}/e52a.yaml").read_text(encoding="utf-8")
        )
        for a in ARMS
    }
    models = {a: r["records"]["models"]["pool-grid"] for a, r in runs.items()}
    third = yaml.safe_load((args.third / "e52a.yaml").read_text(encoding="utf-8"))
    third_curve = third["records"]["models"]["pool-grid"]["held_out"]

    fig, axes = plt.subplots(1, 4, figsize=(19, 4.2))
    for a, (label, colour) in ARMS.items():
        m, d = models[a], models[a]["diagnostics"]
        steps = sorted(m["held_out"])
        axes[0].plot(steps, [m["held_out"][s] for s in steps], color=colour,
                     marker="o", ms=3, label=label)  # fmt: skip
        loss = np.array(m["train_loss"])
        x = np.arange(1, len(loss) // SMOOTH + 1) * SMOOTH
        axes[1].plot(x, windows(loss, SMOOTH), color=colour, marker="o", ms=3)
        norms = d["norm"]["up"]
        axes[2].plot(sorted(norms), [norms[s] for s in sorted(norms)], color=colour)
        grad = np.array(d["grad"]["up"])
        axes[3].plot(x, windows(grad, SMOOTH, np.median), color=colour, marker="o",
                     ms=3)  # fmt: skip
    steps = sorted(third_curve)
    axes[0].plot(steps, [third_curve[s] for s in steps], color="0.6", ls="--", lw=1,
                 label="third run's pool-grid (identity)")  # fmt: skip
    for ax in axes[:2]:
        ax.axhline(1, color="0.3", lw=0.8)
    titles = (
        "held-out error over Déjà View's\n(mean of 16 scenes, 12 fixed views)",
        f"training loss over Déjà View's\n(mean of {SMOOTH} steps, same draws per arm)",
        "norm of B_k (all 16 loop steps)",
        f"gradient norm of B_k (median of {SMOOTH} steps)",
    )
    for ax, title in zip(axes, titles, strict=True):
        style(ax)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("training step")
    axes[0].legend(fontsize=8, frameon=False)
    fig.suptitle(
        "E52d (pool-grid, lr 3e-6): with the detached Sim(3) (A-E), B_k grows steadily "
        "while both losses rise, unless its motion is slowed (C) or stopped (E);\nwith "
        "the exact gradient (F), B_k grows slower and both losses keep falling to step "
        "3000 (held-out +0.0004 after its minimum at 2750)",
        fontsize=11,
        y=1.1,
    )
    fig.savefig(FIGURES / "e52d_training.jpg", dpi=110, bbox_inches="tight")

    compact = {}
    for a, m in models.items():
        d = m["diagnostics"]
        series = {"train_loss": m["train_loss"], "base_error": d["base"]}
        series |= {f"grad_{p}": g for p, g in d["grad"].items()}
        compact[a] = {
            "label": ARMS[a][0],
            "set": runs[a]["args"]["set"],
            "git": runs[a]["git"],
            "seconds": round(m["seconds"]),
            "best_step": m["best_step"],
            "held_out": {k: round(v, 5) for k, v in m["held_out"].items()},
            "clipped_steps": d["clipped"],
            "norm": {
                p: {k: round(v, 5) for k, v in n.items()} for p, n in d["norm"].items()
            },
            f"per_{WINDOW}_steps": {
                k: windows(np.array(v), WINDOW).round(5).tolist()
                for k, v in series.items()
            },
        }
    header = (
        "# E52d: cluster jobs 2889-2893 (arms A-E, RTX 5090) and the 3080 server (arm F),"
        f" pool-grid; per-step series as means over {WINDOW}-step windows\n"
    )
    (RESULTS / "e52d_train.yaml").write_text(
        header + yaml.safe_dump(compact, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
