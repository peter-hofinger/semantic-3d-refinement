"""E34 rescored with PR #19's protocol: what the joint refiner changes on dev and val.

Reads the `run.yaml` that `src.analysis.scene_eval` (main f645d08) wrote for
`configs/eval_hypersim.yaml` with `per_scene: [dev, val]` and a val list without
ai_031_004 (its keyframe-0 rotation is 10 x orthonormal, see E34), cluster job 994.
Prints both variants per scene, the split aggregates and how many scenes each surface
score improves on. Figure: Refined minus Déjà View per scene for accuracy, completeness
and Chamfer (cm; left of zero is better), split means on top.

uv run python experiments/scripts/e34_val_eval_v2.py \
    runs/eval_hypersim_v2_val10-20261105-123026/run.yaml \
    > experiments/results/e34_val_eval_v2.txt
"""

import sys
from pathlib import Path

import matplotlib as mpl
import numpy as np
import yaml

mpl.use("Agg")
import matplotlib.pyplot as plt

VARIANTS = ("Déjà View", "Refined")
SURFACE = ("accuracy_cm", "completeness_cm", "chamfer_cm")
LABELS = ("miou", "pq", "ap")
SPLITS = ("dev", "val")
NAMES = {
    "ai_010_007": "living room",
    "ai_006_007": "bedroom",
    "ai_044_001": "lobby",
    "ai_004_003": "bathroom",
    "ai_009_009": "28 m hall",
}
BETTER, WORSE, INK, MUTED = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e"
FIGURE = Path("experiments/figures/e34_val_eval_v2.jpg")


def short(scene: str) -> str:
    """Scene id without the camera, with a room name where one helps."""
    name = scene.rsplit("_cam_", 1)[0]
    return f"{name} ({NAMES[name]})" if name in NAMES else name


def main() -> None:
    """Print the tables and write the figure."""
    run = yaml.safe_load(Path(sys.argv[1]).read_text(encoding="utf-8"))
    print(f"run {sys.argv[1]}, git {run['git'][:7]}")
    rows: list[tuple[str, dict]] = []
    for split in SPLITS:
        report = run["splits"][split]
        print(f"\n{split}: scene | variant | " + " | ".join(SURFACE + LABELS))
        for scene, record in report["scenes"].items():
            for v in VARIANTS:
                h = record["variants"][v]["headline"]
                print(
                    f"{scene} | {v} | "
                    + " | ".join(f"{h[m]:.3f}" for m in SURFACE + LABELS)
                )
            rows.append((short(scene), record["variants"]))
        for v in VARIANTS:
            h = report["aggregate"][v]["headline"]
            print(
                f"{split} mean | {v} | "
                + " | ".join(f"{h[m]:.3f}" for m in SURFACE + LABELS)
            )
        delta = np.array(
            [
                [
                    r["variants"]["Refined"]["headline"][m]
                    - r["variants"]["Déjà View"]["headline"][m]
                    for m in SURFACE
                ]
                for r in report["scenes"].values()
            ]
        )
        for j, m in enumerate(SURFACE):
            print(
                f"{split} {m}: better on {(delta[:, j] < 0).sum()} / {len(delta)}, "
                f"worse on {(delta[:, j] > 0).sum()}"
            )
    means = [
        (
            f"{split} mean ({len(run['splits'][split]['scenes'])} scenes)",
            run["splits"][split]["aggregate"],
        )
        for split in SPLITS
    ]

    entries = means + rows
    y = np.arange(len(entries))[::-1].astype(float)
    y[: len(means)] += 0.6  # a gap between the means and the scenes
    fig, axes = plt.subplots(1, 3, figsize=(13, 6.2), sharex=True, sharey=True)
    titles = ("accuracy", "completeness", "Chamfer")
    for ax, metric, title in zip(axes, SURFACE, titles, strict=True):
        d = np.array(
            [
                e["Refined"]["headline"][metric] - e["Déjà View"]["headline"][metric]
                for _, e in entries
            ]
        )
        colours = [BETTER if x < 0 else WORSE for x in d]
        ax.barh(y, d, 0.7, color=colours)
        ax.axvline(0, color=INK, lw=0.8)
        for k in range(len(means)):
            ax.text(
                d[k],
                y[k],
                f" {d[k]:+.2f} ",
                va="center",
                fontsize=9,
                color=INK,
                ha="left" if d[k] >= 0 else "right",
                fontweight="bold",
            )
        ax.set_title(f"{title}: Refined - Déjà View (cm)", fontsize=11, color=INK)
        ax.set_xlabel("← better  |  worse →", color=MUTED)
        ax.grid(axis="x", color="0.9", lw=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    low, high = axes[0].get_xlim()  # one cm scale for all three scores
    axes[0].set_xlim(low - 0.08 * (high - low), high + 0.06 * (high - low))
    axes[0].set_yticks(y, [n for n, _ in entries], fontsize=9)
    for tick in axes[0].get_yticklabels()[: len(means)]:
        tick.set_fontweight("bold")
    axes[0].axhline(y[len(means) - 1] - 0.8, color="0.6", lw=0.6)
    dev_end = y[len(means) + 2] - 0.5
    for ax in axes:
        ax.axhline(dev_end, color="0.8", lw=0.6, ls="--")
    fig.suptitle(
        "Joint refiner vs Déjà View on Hypersim, PR #19 protocol (Umeyama on pixel pairs), "
        "every weight 1, 20 views at 504 px.\nAccuracy improves on most scenes, "
        "completeness gets worse on every scene, so Chamfer gets worse. "
        "Top 3 scenes: dev; the rest: val (ai_031_004 excluded, bad GT camera).",
        fontsize=10,
        color=INK,
    )
    fig.tight_layout()
    fig.savefig(FIGURE, dpi=150, pil_kwargs={"quality": 90})


if __name__ == "__main__":
    main()
