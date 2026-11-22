"""E86a: the instance voxel density cut by none, SAM 3, GT and permuted ids; table, figure.

Reads one `scene_eval --cfg experiments/configs/e86a.yaml` run.
- stdout: per split, each variant's accuracy / completeness / Chamfer as mean / median
  over the scenes, then each arm minus `none` (plain voxel density) at the same q and
  refinement, which keeps the same number of pixels: mean, median and the scenes on
  which the arm is better.
- `e86a_density.jpg`: per split, top: median accuracy against median completeness over
  q = 0.3, 0.5, 0.7, 0.9 and 1 (Déjà View, every pixel) for each arm; bottom: each
  scene's Chamfer of the arm minus `none` at q 0.5, without and with E27a's refinement
  (symlog: linear within ±0.5 cm). Medians, as one scene's cliff (the cut dropping a
  whole region) sets a mean curve.

uv run python experiments/scripts/e86a_figures.py runs/<run> \
    > experiments/results/e86a_figures.txt
"""

import sys
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

OUT = Path("experiments/figures")
METRICS = ("accuracy_cm", "completeness_cm", "chamfer_cm")
ARMS = ("none", "SAM 3", "GT", "permuted")
COLORS = {"none": "0.45", "SAM 3": "C0", "GT": "C2", "permuted": "C3"}
QS = ("0.3", "0.5", "0.7", "0.9")
PREFIXES = ("Déjà View, kept 0.5", "tangent 4e-2, kept 0.5")


def values(scenes: dict, variant: str) -> np.ndarray:
    """(scenes, 3) accuracy, completeness and Chamfer of one variant, in cm."""
    rows = [record["variants"][variant]["headline"] for record in scenes.values()]
    return np.array([[r[m] for m in METRICS] for r in rows])


def table(report: dict) -> None:
    """Print each split's scores and every arm against `none` at the same cut."""
    for split, record in report.items():
        scenes = record["scenes"]
        n = len(scenes)
        print(f"\n{split} ({n} scenes), cm, accuracy | completeness | Chamfer")
        print("variant | mean / median over scenes")
        for v in next(iter(scenes.values()))["variants"]:
            a = values(scenes, v)
            cells = " | ".join(
                f"{a[:, i].mean():.3f} / {np.median(a[:, i]):.3f}" for i in range(3)
            )
            print(f"{v:34s} | {cells}")
        print("minus none at the same cut: mean | median | scenes better")
        prefixes = (*PREFIXES, *(f"Déjà View, kept {q}" for q in QS if q != "0.5"))
        for prefix in prefixes:
            base = values(scenes, f"{prefix}, none")
            for arm in ARMS[1:]:
                d = values(scenes, f"{prefix}, {arm}") - base
                mean = " / ".join(f"{x:+.3f}" for x in d.mean(0))
                median = " / ".join(f"{x:+.3f}" for x in np.median(d, 0))
                better = " / ".join(str(int(x)) for x in (d < 0).sum(0))
                print(
                    f"{prefix + ', ' + arm:34s} | {mean} | {median} | {better} of {n}"
                )


def figure(report: dict) -> None:
    """Draw the curves and the per-scene differences, one column per split."""
    splits = list(report)
    fig, axes = plt.subplots(
        2, len(splits), figsize=(6.4 * len(splits), 9.0), layout="constrained"
    )
    for col, split in enumerate(splits):
        scenes = report[split]["scenes"]
        ax = axes[0, col]
        dv = np.median(values(scenes, "Déjà View"), 0)
        for arm in ARMS:
            pts = [
                np.median(values(scenes, f"Déjà View, kept {q}, {arm}"), 0) for q in QS
            ]
            pts = np.array([*pts, dv])
            ax.plot(pts[:, 1], pts[:, 0], "o-", color=COLORS[arm], label=arm, ms=4)
        for q, p in zip(
            (*QS, "1"),
            [
                *(
                    np.median(values(scenes, f"Déjà View, kept {q}, none"), 0)
                    for q in QS
                ),
                dv,
            ],
            strict=True,
        ):
            ax.annotate(
                f"q {q}",
                (p[1], p[0]),
                fontsize=8,
                xytext=(4, 4),
                textcoords="offset points",
            )
        ax.set_xlabel("completeness, median over scenes (cm)")
        ax.set_ylabel("accuracy, median over scenes (cm)")
        ax.set_title(f"{split} ({len(scenes)} scenes): Déjà View cut at q, by ids")
        ax.legend(title="cut's instance ids")
        ax.grid(alpha=0.3)
        ax = axes[1, col]
        for k, prefix in enumerate(PREFIXES):
            base = values(scenes, f"{prefix}, none")[:, 2]
            for j, arm in enumerate(ARMS[1:]):
                d = values(scenes, f"{prefix}, {arm}")[:, 2] - base
                x = k * 4 + j + np.linspace(-0.15, 0.15, len(d))
                ax.scatter(x, d, color=COLORS[arm], s=14, label=arm if k == 0 else None)
                ax.hlines(d.mean(), x[0] - 0.2, x[-1] + 0.2, color=COLORS[arm])
        ax.axhspan(-0.036, 0.036, color="0.85", label="floor ±0.036 cm")
        ax.axhline(0, color="k", lw=0.6)
        ax.set_yscale("symlog", linthresh=0.5)
        ax.set_xticks([1, 5], ["Déjà View, q 0.5", "E_t 4e-2 refined, q 0.5"])
        ax.set_ylabel("Chamfer, arm minus none (cm); dots scenes, bar mean")
        ax.set_title(f"{split}: below 0 the arm's ids beat plain voxel density")
        ax.legend()
        ax.grid(alpha=0.3, axis="y")
    fig.suptitle(
        "E86a: instance voxel density cut by none (plain voxel density), SAM 3, GT "
        "or per-view permuted SAM 3 ids",
        x=0.01,
        ha="left",
        fontsize=12,
    )
    fig.savefig(OUT / "e86a_density.jpg", dpi=80)
    plt.close(fig)


def main() -> None:
    """Print the table, then draw the figure."""
    run = Path(sys.argv[1])
    summary = yaml.safe_load((run / "run.yaml").read_text(encoding="utf-8"))
    print(f"run {run.name} (git {summary['git']})")
    table(summary["splits"])
    figure(summary["splits"])


if __name__ == "__main__":
    main()
