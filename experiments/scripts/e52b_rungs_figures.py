"""E52b, main arms: overfit rungs 3 and 4 of LoRA + pooling, GT instances vs grid tiles.

Reads `runs/e52b_rung{3,4}` (both arms, `e52a_segment_adapters.py` on the 3080) and the
control's rungs at lr_lora 1e-4 (`runs/e52b_rung{3,4}_1e-4`, LoRA alone, 18.11).

1. `e52b_rungs.jpg`: rung 3 (bedroom) training ratio per step, 20-step means, the three
   models on one scale; rung 4 (dev) last-100-step ratio per scene.
2. `results/e52b_rungs.yaml`: those numbers and the rungs' pass values.

uv run python experiments/scripts/e52b_rungs_figures.py
"""

from pathlib import Path

import matplotlib as mpl
import numpy as np
import yaml

mpl.use("Agg")
import matplotlib.pyplot as plt
from e52a_figures import FIGURES, style

RUNS = Path("runs")
RESULTS = Path("experiments/results/e52b_rungs.yaml")
SCENES = {"ai_010_007_cam_00": "living room", "ai_006_007_cam_00": "bedroom",
          "ai_044_001_cam_00": "lobby"}  # fmt: skip
MODELS = {  # label -> (run suffix, model name, colour)
    "LoRA alone (control)": ("_1e-4", "lora-none", "#2a9d55"),
    "LoRA + pool, GT instances": ("", "lora_pool-gt", "#eb6834"),
    "LoRA + pool, grid tiles": ("", "lora_pool-grid", "#3a7bd5"),
}
SMOOTH = 20
RUNG3_STEPS, RUNG4_STEPS = 200, 300


def model(rung: int, suffix: str, name: str) -> dict:
    """Return one model's training record of a rung run."""
    path = RUNS / f"e52b_rung{rung}{suffix}" / "e52a.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))["records"]["models"][name]


def smooth(values: list[float]) -> np.ndarray:
    """Means over consecutive SMOOTH steps."""
    v = np.array(values)
    return v[: len(v) // SMOOTH * SMOOTH].reshape(-1, SMOOTH).mean(1)


def rungs() -> dict:
    """Rung 3 last-50 / previous-50 and rung 4 last-100 per-scene ratios per model."""
    out: dict = {}
    for label, (suffix, name, _) in MODELS.items():
        t3 = np.array(model(3, suffix, name)["train_loss"])
        m4 = model(4, suffix, name)
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
    return out


def figure(rung: dict) -> None:
    """Rung 3 curves and rung 4 per-scene bars."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    ax = axes[0]
    for label, (suffix, name, colour) in MODELS.items():
        loss = smooth(model(3, suffix, name)["train_loss"])
        ax.plot((np.arange(len(loss)) + 1) * SMOOTH, loss, color=colour, label=label)
    ax.axhline(0.9, color="0.4", lw=0.8, ls=":")
    ax.set_xlabel("step")
    ax.set_ylabel(f"training ratio to Déjà View ({SMOOTH}-step mean)")
    ax.set_title("rung 3, bedroom, 8 views (pass bar 0.90 dotted)", fontsize=10)
    ax.legend(fontsize=8, frameon=False)
    ax = axes[1]
    for b, (label, (_, _, colour)) in enumerate(MODELS.items()):
        r = rung[label]["rung4_last100"]
        ax.bar(np.arange(3) + (b - 1) * 0.27, [r[s] for s in SCENES.values()], 0.27,
               color=colour, label=label)  # fmt: skip
    ax.axhline(0.95, color="0.4", lw=0.8, ls=":")
    ax.set_xticks(np.arange(3), list(SCENES.values()))
    ax.set_ylim(0.6, 1.08)
    ax.set_ylabel("training ratio to Déjà View, last 100 steps")
    ax.set_title("rung 4, dev (pass bar 0.95 dotted)", fontsize=10)
    ax.legend(fontsize=8, frameon=False, ncol=3, loc="upper center")
    for ax in axes:
        style(ax)
    fig.suptitle(
        "Both arms pass rungs 3 and 4, and neither differs from LoRA alone: "
        "the pooling adapter adds nothing while overfitting, with GT instances or grid tiles.",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(FIGURES / "e52b_rungs.jpg", dpi=150)


def main() -> None:
    """Write the raw numbers and the figure."""
    rung = rungs()
    RESULTS.write_text(yaml.safe_dump(rung, sort_keys=False, allow_unicode=True),
                       encoding="utf-8")  # fmt: skip
    print(yaml.safe_dump(rung, sort_keys=False, allow_unicode=True))
    figure(rung)


if __name__ == "__main__":
    main()
