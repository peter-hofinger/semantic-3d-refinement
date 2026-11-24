"""E87a: the lr_gate sweep of LoRA + gated pooling, per arm at its E52b lr_pool.

Reads `runs/e87a_gate_sweep_<arm>_<lr>` (`e52a_segment_adapters.py` on the 3080, 500
steps, seed 0, one arm per run): the held-out (16 train scenes) and val (11 scenes)
ratio to Déjà View at steps 250 and 500, and the gate parameters' norm at step 500.
The rule (plan entry): each arm takes the lr of least val ratio at step 500.

1. `e87a_sweep.jpg`: val ratio per eval step, one line per lr, one panel per arm; the
   val probe (one random val scene every 10 steps) as 50-step means behind it.
2. `results/e87a_sweep.yaml`: those numbers and each arm's chosen lr.

uv run python experiments/scripts/e87a_sweep_figures.py
"""

from pathlib import Path

import matplotlib as mpl
import numpy as np
import yaml

mpl.use("Agg")
import matplotlib.pyplot as plt
from e52a_figures import FIGURES, style

RUNS = Path("runs")
RESULTS = Path("experiments/results/e87a_sweep.yaml")
ARMS = {"gt": "GT instances, lr_pool 3e-6", "grid": "grid tiles, lr_pool 1e-5"}
LRS = {"1e-5": "#8ecae6", "1e-4": "#219ebc", "1e-3": "#023047"}
STEPS = (250, 500)
PROBE_MEAN = 5  # probes per mean: 50 steps at probe_every 10


def record(arm: str, lr: str) -> dict:
    """Return one sweep run's training record."""
    path = RUNS / f"e87a_gate_sweep_{arm}_{lr}" / "e52a.yaml"
    models = yaml.safe_load(path.read_text(encoding="utf-8"))["records"]["models"]
    return models[f"lora_pool_gate-{arm}"]


def sweep() -> dict:
    """Held-out and val ratio at 250 / 500, gate norm, and each arm's chosen lr."""
    out: dict = {}
    for arm in ARMS:
        rows = {}
        for lr in LRS:
            r = record(arm, lr)
            norms = r["diagnostics"]["norm"]["gate"]
            rows[lr] = {
                "held_out": [float(r["held_out"][s]) for s in STEPS],
                "val": [float(r["val"][s]) for s in STEPS],
                "gate_norm_500": float(norms[max(norms)]),
                "best_step": int(r["best_step"]),
            }
        chosen = min(rows, key=lambda lr: rows[lr]["val"][-1])
        out[arm] = {"by_lr": rows, "chosen_lr": chosen}
    return out


def figure(result: dict) -> None:
    """Val ratio per eval and probe means, per lr, one panel per arm."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    for ax, (arm, title) in zip(axes, ARMS.items(), strict=True):
        for lr, colour in LRS.items():
            r = record(arm, lr)
            probe = r["val_probe"]
            steps = sorted(probe)
            n = len(steps) // PROBE_MEAN * PROBE_MEAN
            means = np.array([probe[s] for s in steps[:n]]).reshape(-1, PROBE_MEAN)
            ends = np.array(steps[:n]).reshape(-1, PROBE_MEAN)[:, -1]
            ax.plot(ends, means.mean(1), color=colour, alpha=0.35, lw=1)
            evals = sorted(r["val"])
            ax.plot(evals, [r["val"][s] for s in evals], "o-", color=colour,
                    label=f"lr_gate {lr}")  # fmt: skip
        ax.axhline(1.0, color="0.4", lw=0.8, ls=":")
        chosen = result[arm]["chosen_lr"]
        ax.set_title(f"{title}; least val at 500: {chosen}", fontsize=10)
        ax.set_xlabel("step")
        ax.legend(fontsize=8, frameon=False)
        style(ax)
    axes[0].set_ylabel("val ratio to Déjà View (dots: 11 scenes; pale: probe means)")
    fig.suptitle(
        "E87a lr_gate sweep, 500 steps, seed 0: val ratio to Déjà View per eval "
        "(1 = Déjà View, lower is better).",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(FIGURES / "e87a_sweep.jpg", dpi=150)


def main() -> None:
    """Write the raw numbers and the figure."""
    result = sweep()
    RESULTS.write_text(yaml.safe_dump(result, sort_keys=False), encoding="utf-8")
    print(yaml.safe_dump(result, sort_keys=False))
    figure(result)


if __name__ == "__main__":
    main()
