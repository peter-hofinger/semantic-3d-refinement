"""E52f: GT instances vs grid tiles for LoRA + pooling inside Déjà View, trained on 3-120 scenes.

Tests the data half of the hypothesis of issue #81: a semantic advantage is largest when the
training data is small. Reads `runs/e52f_n<n>_<arm>` (`e52a_segment_adapters.py` on the 3080,
500 steps, val every 100 steps, state kept on val). The val ratio of a state is the mean over
Hypersim's 11 val scenes of its 3D error over Déjà View's on the same views (step 0); the gap
is grid's kept val ratio minus GT's, so a positive gap favours GT instances.

1. `e52f_scaling.jpg`: val ratio per eval step for every size and arm; kept val ratio per size;
   the gap per size with the rerun spread (+-0.007) and the per-scene spread of the gap.
2. `results/e52f_scaling.yaml`: those numbers.

uv run python experiments/scripts/e52f_figures.py
"""

from pathlib import Path

import matplotlib as mpl
import numpy as np
import yaml

mpl.use("Agg")
import matplotlib.pyplot as plt
from e52a_figures import FIGURES, style

RUNS = Path("runs")
RESULTS = Path("experiments/results/e52f_scaling.yaml")
SIZES = (3, 10, 30, 120)
ARMS = {"gt": ("GT instances", "#eb6834"), "grid": ("grid tiles", "#3a7bd5")}
SPREAD = 0.007  # val rerun spread at step 500 (E52b control)


def record(n: int, arm: str) -> dict:
    """Return one run's model record and check it is the planned run."""
    run = yaml.safe_load(
        (RUNS / f"e52f_n{n}_{arm}" / "e52a.yaml").read_text(encoding="utf-8")
    )
    rec = run["records"]["models"][f"lora_pool-{arm}"]
    assert rec["selected_on"] == "val", rec["selected_on"]
    assert len(rec["train_loss"]) == 500, len(rec["train_loss"])
    assert len(run["records"]["train_scenes"]) == (
        136 if n == 120 else n
    )  # +16 held out
    return rec


def scene_ratios(rec: dict, step: int) -> np.ndarray:
    """Per val scene error ratio to Déjà View (step 0) at one eval step."""
    ratio = np.array(rec["val_error"][step]) / np.array(rec["val_error"][0])
    assert np.isclose(ratio.mean(), rec["val"][step], atol=1e-6), (
        ratio.mean(),
        rec["val"][step],
    )
    return ratio


def numbers() -> dict:
    """Kept val ratio per arm and size, the gap and its per-scene spread."""
    out: dict = {}
    for n in SIZES:
        recs = {arm: record(n, arm) for arm in ARMS}
        kept = {arm: scene_ratios(r, r["best_step"]) for arm, r in recs.items()}
        gap = kept["grid"] - kept["gt"]
        out[n] = {
            **{
                arm: {
                    "best_step": r["best_step"],
                    "val": float(kept[arm].mean()),
                    "curve": {int(s): float(v) for s, v in r["val"].items()},
                }
                for arm, r in recs.items()
            },
            "gap": float(gap.mean()),
            "gap_scenes_q25_q75": [
                float(np.quantile(gap, 0.25)),
                float(np.quantile(gap, 0.75)),
            ],
            "gap_scenes_gt_better": int((gap > 0).sum()),
        }
    return out


def figure(res: dict) -> None:
    """Val curves, kept val ratio and the gap against the number of train scenes."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    ax = axes[0]
    for k, n in enumerate(SIZES):
        for arm, (label, colour) in ARMS.items():
            c = res[n][arm]["curve"]
            ax.plot(np.array(list(c)) + 8 * (k - 1.5), list(c.values()), color=colour,
                    alpha=0.35 + 0.65 * k / 3, marker="o", ms=3,
                    label=f"{label}, {n} scenes")  # fmt: skip
    ax.axhline(1.0, color="0.4", lw=0.8, ls=":")
    ax.set_xlabel("step (sizes offset by a few steps for legibility)")
    ax.set_ylabel("val ratio to Déjà View's 3D error")
    ax.set_title("val during training (darker = more train scenes)", fontsize=10)
    ax.legend(fontsize=6.5, frameon=False, ncol=2)
    ax = axes[1]
    for arm, (label, colour) in ARMS.items():
        ax.plot(
            SIZES,
            [res[n][arm]["val"] for n in SIZES],
            color=colour,
            marker="o",
            label=label,
        )
    ax.axhline(1.0, color="0.4", lw=0.8, ls=":")
    ax.set_xscale("log")
    ax.set_xticks(SIZES, [str(n) for n in SIZES])
    ax.set_xlabel("train scenes")
    ax.set_ylabel("kept val ratio to Déjà View (lower is better)")
    ax.set_title("kept state (least val ratio over 5 evals)", fontsize=10)
    ax.legend(fontsize=8, frameon=False)
    ax = axes[2]
    gap = [res[n]["gap"] for n in SIZES]
    q = np.array([res[n]["gap_scenes_q25_q75"] for n in SIZES])
    ax.axhspan(-SPREAD, SPREAD, color="0.9", label=f"rerun spread +-{SPREAD}")
    ax.errorbar(SIZES, gap, yerr=[np.array(gap) - q[:, 0], q[:, 1] - np.array(gap)],
                color="k", marker="o", capsize=3,
                label="mean over 11 val scenes (bars: scene quartiles)")  # fmt: skip
    ax.axhline(0.0, color="0.4", lw=0.8)
    ax.set_xscale("log")
    ax.set_xticks(SIZES, [str(n) for n in SIZES])
    ax.set_xlabel("train scenes")
    ax.set_ylabel("gap = grid's val ratio - GT's (> 0: GT better)")
    ax.set_ylim(-0.045, 0.035)
    ax.set_title("semantic gain against training data", fontsize=10)
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    for ax in axes:
        style(ax)
    fig.suptitle(
        "GT instances gain nothing over grid tiles at any number of train scenes: the gap stays "
        "inside the rerun spread from 3 to 120 scenes (one seed, one subset per size).",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(FIGURES / "e52f_scaling.jpg", dpi=150)


def main() -> None:
    """Write the raw numbers and the figure."""
    res = numbers()
    RESULTS.write_text(
        yaml.safe_dump(res, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    for n in SIZES:
        r = res[n]
        print(f"n={n:3d} gt {r['gt']['val']:.4f} (step {r['gt']['best_step']}) "
              f"grid {r['grid']['val']:.4f} (step {r['grid']['best_step']}) gap {r['gap']:+.4f} "
              f"scenes q25-q75 {r['gap_scenes_q25_q75'][0]:+.4f}..{r['gap_scenes_q25_q75'][1]:+.4f}, "
              f"GT better on {r['gap_scenes_gt_better']}/11")  # fmt: skip
    figure(res)


if __name__ == "__main__":
    main()
