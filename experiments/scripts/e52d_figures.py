"""E52d, figure and compact raw output: why E52a's adapters diverge (issue #52).

Reads the five arms' `e52a.yaml` of `e52a_segment_adapters.py` (pool-grid, A as E52a's
third run, B fp32, C weight decay on B_k, D gradient clipping, E cosine lr decay) and the
third run's grid models for the identity check.

1. `figures/e52d_training.jpg`: per arm over training steps, the held-out error (over Déjà
   View's, mean of 16 scenes), the training loss (over Déjà View's, mean of 250 steps;
   every arm draws the same scenes and views per step), the norm of B_k (all 16 steps) and
   the median gradient norm of B_k over 250 steps.
2. `results/e52d_train.yaml`: held-out curves, parameter norms, clip counts and the per-step
   series as means over 50-step windows (the per-step series stay in the runs).

uv run python experiments/scripts/e52d_figures.py runs/e52d runs/e52a3_grid
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
}
SMOOTH = 250  # steps per training-loss mean and gradient median
WINDOW = 50  # steps per mean in the compact raw output
RESULTS = Path("experiments/results")


def windows(x: np.ndarray, n: int, fn: np.ufunc = np.mean) -> np.ndarray:
    """Reduce a per-step series to one value per n consecutive steps."""
    assert len(x) % n == 0, (len(x), n)
    return fn(x.reshape(-1, n), axis=1)


def main() -> None:
    """Load the arms, draw the figure, write the compact raw output."""
    parser = argparse.ArgumentParser()
    parser.add_argument("prefix", type=Path, help="runs/e52d -> runs/e52d_A ... _E")
    parser.add_argument("third", type=Path, help="E52a's third run, grid models")
    args = parser.parse_args()
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
        "E52d (pool-grid, lr 3e-6): fp32 and clipping change nothing; B_k grows steadily "
        "while both losses rise; weight decay slows its growth and the rise, cosine lr "
        "decay stops both",
        fontsize=11,
        y=1.06,
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
        "# E52d: cluster jobs 2889-2893 (arms A-E), pool-grid; per-step series as means "
        f"over {WINDOW}-step windows\n"
    )
    (RESULTS / "e52d_train.yaml").write_text(
        header + yaml.safe_dump(compact, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
