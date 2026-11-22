"""E87a: overfit rungs 3 and 4 of LoRA + gated pooling, against E52b's ungated arms.

Reads `runs/e87a_rung{3,4}` (both gated arms, `e52a_segment_adapters.py` on the 3080),
E52b's ungated arms `runs/e52b_rung{3,4}` and LoRA alone `runs/e52b_rung{3,4}_1e-4`; all
seed 0, so every model sees the same scene and view draws.

1. `e87a_rungs.jpg`: rung 3 (bedroom) training ratio per step, 20-step means; rung 4
   (dev) last-100-step ratio per scene; the gate parameters' norm over rung 4's steps.
2. `results/e87a_rungs.yaml`: those numbers and the rungs' pass values.

uv run python experiments/scripts/e87a_rungs_figures.py
"""

from pathlib import Path

import matplotlib as mpl
import numpy as np
import yaml

mpl.use("Agg")
import matplotlib.pyplot as plt
from e52a_figures import FIGURES, style
from e52b_rungs_figures import RUNG3_STEPS, RUNG4_STEPS, SCENES, SMOOTH, smooth

RUNS = Path("runs")
RESULTS = Path("experiments/results/e87a_rungs.yaml")
MODELS = {  # label -> (run prefix, run suffix, model name, colour, line style)
    "LoRA alone (control)": ("e52b", "_1e-4", "lora-none", "#2a9d55", "-"),
    "LoRA + pool, GT (E52b)": ("e52b", "", "lora_pool-gt", "#eb6834", ":"),
    "LoRA + pool, grid (E52b)": ("e52b", "", "lora_pool-grid", "#3a7bd5", ":"),
    "LoRA + gated pool, GT": ("e87a", "", "lora_pool_gate-gt", "#eb6834", "-"),
    "LoRA + gated pool, grid": ("e87a", "", "lora_pool_gate-grid", "#3a7bd5", "-"),
}


def model(rung: int, prefix: str, suffix: str, name: str) -> dict:
    """Return one model's training record of a rung run."""
    path = RUNS / f"{prefix}_rung{rung}{suffix}" / "e52a.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))["records"]["models"][name]


def rungs() -> dict:
    """Rung 3 last-50 / previous-50 and rung 4 last-100 per-scene ratios per model."""
    out: dict = {}
    for label, (prefix, suffix, name, _, _) in MODELS.items():
        t3 = np.array(model(3, prefix, suffix, name)["train_loss"])
        m4 = model(4, prefix, suffix, name)
        t4, scene = np.array(m4["train_loss"]), np.array(m4["diagnostics"]["scene"])
        assert len(t3) == RUNG3_STEPS, len(t3)
        assert len(t4) == RUNG4_STEPS, len(t4)
        out[label] = {
            "rung3_last50": float(t3[-50:].mean()),
            "rung3_prev50": float(t3[-100:-50].mean()),
            "rung4_last100": {
                SCENES[s]: float(t4[-100:][scene[-100:] == s].mean()) for s in SCENES
            },
        }
        norms = m4["diagnostics"]["norm"].get("gate")
        if norms:
            out[label]["rung4_gate_norm_last"] = float(norms[max(norms)])
    return out


def figure(rung: dict) -> None:
    """Rung 3 curves, rung 4 per-scene bars and the gate's norm in rung 4."""
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.2))
    ax = axes[0]
    for label, (prefix, suffix, name, colour, ls) in MODELS.items():
        loss = smooth(model(3, prefix, suffix, name)["train_loss"])
        x = (np.arange(len(loss)) + 1) * SMOOTH
        ax.plot(x, loss, color=colour, ls=ls, label=label)
    ax.axhline(0.9, color="0.4", lw=0.8, ls=":")
    ax.set_xlabel("step")
    ax.set_ylabel(f"training ratio to Déjà View ({SMOOTH}-step mean)")
    ax.set_title("rung 3, bedroom, 8 views (pass bar 0.90 dotted)", fontsize=10)
    ax.legend(fontsize=7, frameon=False)
    ax = axes[1]
    width = 0.8 / len(MODELS)
    for b, (label, (_, _, _, colour, ls)) in enumerate(MODELS.items()):
        r = rung[label]["rung4_last100"]
        ax.bar(np.arange(3) + (b - 2) * width, [r[s] for s in SCENES.values()], width,
               color=colour, alpha=0.45 if ls == ":" else 1.0, label=label)  # fmt: skip
    ax.axhline(0.95, color="0.4", lw=0.8, ls=":")
    ax.set_xticks(np.arange(3), list(SCENES.values()))
    ax.set_ylim(0.6, 1.08)
    ax.set_ylabel("training ratio to Déjà View, last 100 steps")
    ax.set_title("rung 4, dev (pass bar 0.95 dotted; pale: E52b, ungated)", fontsize=10)
    ax = axes[2]
    for label, (prefix, suffix, name, colour, _) in MODELS.items():
        norms = model(4, prefix, suffix, name)["diagnostics"]["norm"].get("gate")
        if norms:
            steps = sorted(norms)
            ax.plot(steps, [norms[s] for s in steps], color=colour, label=label)
    ax.set_xlabel("step")
    ax.set_ylabel("‖(w_k, b_k)‖ over all loop steps")
    ax.set_title(
        "rung 4, gate parameters' norm (0 at the start: gate 1/2)", fontsize=10
    )
    ax.legend(fontsize=7, frameon=False)
    for ax in axes:
        style(ax)
    fig.suptitle(
        "Both gated arms pass rungs 3 and 4 and lie on E52b's ungated arms and LoRA "
        "alone (same seed and draws); the gate's parameters move steadily, faster "
        "with grid tiles than with GT instances.",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(FIGURES / "e87a_rungs.jpg", dpi=150)


def main() -> None:
    """Write the raw numbers and the figure."""
    rung = rungs()
    RESULTS.write_text(yaml.safe_dump(rung, sort_keys=False, allow_unicode=True),
                       encoding="utf-8")  # fmt: skip
    print(yaml.safe_dump(rung, sort_keys=False, allow_unicode=True))
    figure(rung)


if __name__ == "__main__":
    main()
