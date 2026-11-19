"""E52c, before the sweep: the control's seed spread and the two arms' overfit rungs.

Reads the E52b control at seeds 0, 1, 2 (`runs/e52b_control`, `runs/e52b_control_s1`,
`runs/e52b_control_s2`) and E52c's rungs 3 and 4 for both arms
(`runs/e52c_rung{3,4}_{instance,none}`, λ = 1), rung 3 at λ = 10
(`runs/e52c_rung3_{instance,none}_l10`) and with the cross-view term alone
(`runs/e52c_rung3_{instance,none}_conly`, `gt_weight=0`), all `e52a.yaml` of `e52a_segment_adapters.py`
on the 3080.

1. `e52c_rungs.jpg`: dev Chamfer per scene of each control seed against Déjà View; rung 3
   (bedroom) GT ratio and cross-view ratio c / c_DV per step, 20-step means, both arms,
   at λ = 1, λ = 10 and with the term alone, on one scale; rung 4 (dev) last-100-step GT and cross-view ratios per scene, both arms.
2. `results/e52c_rungs.yaml`: the numbers above and the rungs' pass values.

uv run python experiments/scripts/e52c_rungs_figures.py
"""

from pathlib import Path

import matplotlib as mpl
import numpy as np
import yaml

mpl.use("Agg")
import matplotlib.pyplot as plt
from e52a_figures import FIGURES, style
from e52a_segment_adapters import BASE

RUNS = Path("runs")
RESULTS = Path("experiments/results/e52c_rungs.yaml")
SCENES = {"ai_010_007_cam_00": "living room", "ai_006_007_cam_00": "bedroom",
          "ai_044_001_cam_00": "lobby"}  # fmt: skip
SEEDS = {0: "e52b_control", 1: "e52b_control_s1", 2: "e52b_control_s2"}
ARMS = {"instance": ("gate: same instance", "#eb6834"),
        "none": ("any partner", "#3a7bd5")}  # fmt: skip
SMOOTH = 20
RUNG3 = {"": "λ = 1", "_l10": "λ = 10", "_conly": "cross-view term alone"}


def record(run: str) -> dict:
    """Load the records of one run."""
    return yaml.safe_load((RUNS / run / "e52a.yaml").read_text(encoding="utf-8"))


def model(run: str) -> dict:
    """Return the one model's training record of a run."""
    return next(iter(record(run)["records"]["models"].values()))


def seeds() -> dict:
    """Dev Chamfer (cm) per scene of Déjà View and every control seed, kept steps."""
    out: dict = {}
    for seed, run in SEEDS.items():
        rec = record(run)["records"]
        if seed == 0:
            out[BASE] = [rec["eval"][s][BASE]["chamfer_cm"] for s in SCENES]
        out[f"seed {seed}"] = [
            rec["eval"][s]["lora-none"]["chamfer_cm"] for s in SCENES
        ]
        out[f"seed {seed} kept step"] = rec["models"]["lora-none"]["best_step"]
    return out


def rungs() -> dict:
    """Rung 3 last-50 and rung 4 last-100 per-scene ratios of both arms."""
    out: dict = {}
    for arm in ARMS:
        for suffix in RUNG3:
            m3 = model(f"e52c_rung3_{arm}{suffix}")
            t, c = np.array(m3["train_loss"]), np.array(m3["train_cons_ratio"])
            out[f"rung3_{arm}{suffix}"] = {"gt_last50": float(t[-50:].mean()),
                                           "cons_first50": float(c[:50].mean()),
                                           "cons_last50": float(c[-50:].mean())}  # fmt: skip
        m4 = model(f"e52c_rung4_{arm}")
        t, c = np.array(m4["train_loss"]), np.array(m4["train_cons_ratio"])
        scene = np.array(m4["diagnostics"]["scene"])
        out[f"rung4_{arm}"] = {
            SCENES[s]: {
                "gt_last100": float(t[-100:][scene[-100:] == s].mean()),
                "cons_first100": float(c[:100][scene[:100] == s].mean()),
                "cons_last100": float(c[-100:][scene[-100:] == s].mean()),
            }
            for s in SCENES
        }
    return out


def smooth(values: list[float]) -> np.ndarray:
    """Means over consecutive SMOOTH steps."""
    v = np.array(values)
    return v[: len(v) // SMOOTH * SMOOTH].reshape(-1, SMOOTH).mean(1)


def figure(dev: dict, rung: dict) -> None:
    """Seed spread, rung 3 curves, rung 4 per-scene ratios."""
    fig, axes = plt.subplots(1, 5, figsize=(25, 4))
    ax = axes[0]
    labels = [BASE, "seed 0", "seed 1", "seed 2"]
    colours = ["0.6", "#2a9d55", "#5cbf80", "#98d9ae"]
    for b, (label, colour) in enumerate(zip(labels, colours, strict=True)):
        ax.bar(
            np.arange(3) + (b - 1.5) * 0.2, dev[label], 0.2, label=label, color=colour
        )
    ax.set_xticks(np.arange(3), list(SCENES.values()))
    ax.set_ylabel("dev Chamfer (cm)")
    ax.set_title("E52b control (LoRA, no segments): 3 seeds", fontsize=10)
    ax.legend(fontsize=8, frameon=False)
    for ax, (suffix, name) in zip(axes[1:4], RUNG3.items(), strict=True):
        for arm, (label, colour) in ARMS.items():
            m = model(f"e52c_rung3_{arm}{suffix}")
            steps = (np.arange(len(smooth(m["train_loss"]))) + 1) * SMOOTH
            ax.plot(steps, smooth(m["train_loss"]), color=colour,
                    label=f"GT ratio, {label}")  # fmt: skip
            ax.plot(steps, smooth(m["train_cons_ratio"]), color=colour, ls="--",
                    label=f"c / c_DV, {label}")  # fmt: skip
        ax.axhline(0.9, color="0.4", lw=0.8, ls=":")
        ax.set_ylim(0.78, 1.4)
        ax.set_xlabel("step")
        ax.set_ylabel(f"ratio to Déjà View ({SMOOTH}-step mean)")
        ax.set_title(f"rung 3, bedroom, {name} (pass bar 0.90 dotted)", fontsize=10)
        ax.legend(fontsize=7, frameon=False)
    ax = axes[4]
    for a, (arm, (label, colour)) in enumerate(ARMS.items()):
        r = rung[f"rung4_{arm}"]
        x = np.arange(3) + (a - 0.5) * 0.35
        ax.bar(x, [r[s]["gt_last100"] for s in SCENES.values()], 0.17, color=colour,
               label=f"GT ratio, {label}")  # fmt: skip
        ax.bar(x + 0.17, [r[s]["cons_last100"] for s in SCENES.values()], 0.17,
               color=colour, alpha=0.45, label=f"c / c_DV, {label}")  # fmt: skip
    ax.axhline(0.95, color="0.4", lw=0.8, ls=":")
    ax.set_xticks(np.arange(3), list(SCENES.values()))
    ax.set_ylim(0.6, 1.0)
    ax.set_ylabel("ratio to Déjà View, last 100 steps")
    ax.set_title("rung 4, dev, λ = 1 (GT pass bar 0.95 dotted)", fontsize=10)
    ax.legend(fontsize=7, frameon=False, ncol=2, loc="upper center",
              bbox_to_anchor=(0.5, -0.1))  # fmt: skip
    for ax in axes:
        style(ax)
    fig.suptitle(
        "Seeds of the control differ by up to 0.08 cm dev mean (0.25 cm on the lobby). "
        "Both E52c arms pass the GT-ratio bars at λ = 1;\nthe cross-view ratio falls "
        "but misses rung 3's 0.90 at λ = 1 (0.94) and at λ = 10 (0.91), where the GT "
        "ratio rises to 0.92-0.93; trained on the term alone it stops at 0.89-0.90 while "
        "the GT ratio rises above Déjà View (1.13-1.23). The two arms are indistinguishable.",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(FIGURES / "e52c_rungs.jpg", dpi=150)


def main() -> None:
    """Write the raw numbers and the figure."""
    dev, rung = seeds(), rungs()
    means = {k: float(np.mean(v)) for k, v in dev.items() if isinstance(v, list)}
    out = {"dev_chamfer_cm": dev, "dev_mean_cm": means, "rungs": rung}
    RESULTS.write_text(yaml.safe_dump(out, sort_keys=False, allow_unicode=True),
                       encoding="utf-8")  # fmt: skip
    print(yaml.safe_dump(out, sort_keys=False, allow_unicode=True))
    figure(dev, rung)


if __name__ == "__main__":
    main()
