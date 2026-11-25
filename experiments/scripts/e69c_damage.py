"""E69c rung 0: draw the damage map and write the region in which a model beats Deja View.

Reads the damaged oracles that `e69c_oracle.py damage` and `fine` scored (`runs/e69c/damage_s<k>`,
`_fine`), Deja View's and `dv-ovoxel`'s scores of the newest rung-0 run, and `refs.json`.
Every variant has measured child precision, recall and vertex error (mean over the scenes of a
split), the coordinates on which a trained model is placed. The figure: per vertex noise sigma,
the mean Chamfer of the variants over (recall, precision), the contour where it equals Deja View's
on that split, and `dv-ovoxel` placed by its own measured recall and precision. The text lists, per
sigma and dropped share, the most added children that still beat Deja View by the plan's rule
(on the 11 val scenes `beats`: the mean gain exceeds the larger of the band and twice the
standard error; on the 3 dev scenes, which have no usable standard error, rung 4's rule: the
mean gain exceeds the band and at least 2 scenes gain more than the band).

uv run python -m experiments.scripts.e69c_damage 4 experiments/figures/e69c_damage.jpg \
    experiments/results/e69c_damage.txt
"""

import json
import sys
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.colors import LogNorm

from experiments.scripts.e69a_gate import BAND, beats
from experiments.scripts.e69c_gate import RUNS, SPLITS

SPLIT_NAMES = ("dev", "val")
DEV_SCENES = (
    3  # three scenes have no usable standard error: the dev rule of rung 4 applies
)


def load(split: int) -> tuple[dict, dict, dict]:
    """Return the damage rows per scene, Deja View's and dv-ovoxel's scores per split."""
    rows = {}
    for folder in (f"damage_s{split}", f"damage_s{split}_fine"):
        for path in sorted((RUNS / folder).glob("*.json")):
            rows.setdefault(path.stem, []).extend(json.loads(path.read_text()))
    report = yaml.safe_load(
        max(RUNS.glob("rung0_eval-*")).joinpath("run.yaml").read_text()
    )["splits"]
    return rows, report, json.loads((RUNS / "refs.json").read_text())


def settings(rows: dict) -> list[tuple[float, float, float]]:
    """Return the (r, f, sigma) of the variants, sorted."""
    first = next(iter(rows.values()))
    return sorted({(r["r"], r["f"], r["sigma"]) for r in first})


def stack(rows: dict, report: dict, split: str, key: str, grid: list) -> np.ndarray:
    """Return `key` of every variant on every scene of the split, shape (variants, scenes)."""
    out = []
    for r, f, sigma in grid:
        row = []
        for scene in sorted(report[split]["scenes"]):
            hit = next(
                v for v in rows[scene] if (v["r"], v["f"], v["sigma"]) == (r, f, sigma)
            )
            row.append(hit[key])
        out.append(row)
    return np.array(out)


def wins(dv: np.ndarray, arm: np.ndarray) -> bool:
    """Return whether a variant beats Deja View: rung 4's rule on 3 scenes, `beats` on more."""
    gain = dv - arm
    if len(dv) == DEV_SCENES:
        return bool(gain.mean() > BAND and (gain > BAND).sum() >= 2)
    return beats(dv, arm)


def main() -> None:
    """Draw the map and write the table of the region that beats Deja View."""
    split, figure, table = int(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    assert split in SPLITS, split
    rows, report, refs = load(split)
    grid = settings(rows)
    sigmas = sorted({s for _, _, s in grid})
    fig, axes = plt.subplots(
        2, len(sigmas), figsize=(4.4 * len(sigmas), 8.6), constrained_layout=True
    )
    lines = [
        (
            f"E69c rung 0, damage map at split {split}: variants of oracle-ovoxel that beat "
            "Deja View (val: mean gain above the larger of 0.1 cm and twice the standard error "
            "of the per-scene differences; dev, 3 scenes: mean gain above 0.1 cm and at least 2 "
            "scenes above 0.1 cm). Precision, recall, vertex error: measured, mean over the "
            "split's scenes."
        )
    ]
    for row, name in zip(axes, SPLIT_NAMES, strict=True):
        scenes = sorted(report[name]["scenes"])
        dv = np.array(
            [report[name]["scenes"][s]["Déjà View"]["chamfer_cm"] for s in scenes]
        )
        chamfer = stack(rows, report, name, "chamfer_cm", grid)
        precision = stack(rows, report, name, "precision", grid).mean(1)
        recall = stack(rows, report, name, "recall", grid).mean(1)
        dv_ovoxel = [refs[f"{s}"][str(split)] for s in scenes]
        mark = (
            np.mean([d["recall"] for d in dv_ovoxel]),
            np.mean([d["precision"] for d in dv_ovoxel]),
            np.mean([d["vertex_err"] for d in dv_ovoxel]) * np.sqrt(np.pi / 2),
        )
        lines.append(
            f"\n{name}: Deja View {dv.mean():.3f} cm; dv-ovoxel (s{split}) recall {mark[0]:.3f}, "
            f"precision {mark[1]:.3f}, vertex error equal to Gaussian sigma {mark[2]:.2f} child sizes"
        )
        for ax, sigma in zip(row, sigmas, strict=True):
            pick = [i for i, (_, _, s) in enumerate(grid) if s == sigma]
            shown = chamfer[pick].mean(1)
            points = ax.scatter(
                recall[pick],
                precision[pick],
                c=shown,
                norm=LogNorm(0.5, 20),
                cmap="viridis",
                s=34,
            )
            ax.tricontour(
                recall[pick], precision[pick], shown, levels=[dv.mean()], colors="red"
            )
            ax.scatter(
                mark[0], mark[1], marker="*", s=220, c="red", edgecolors="k", zorder=5
            )
            ax.set_title(
                f"{name}, vertex noise sigma {sigma} (child sizes)", fontsize=10
            )
            ax.set_xlabel("child recall")
            ax.set_ylabel("child precision")
            ax.set_xlim(0.4, 1.02)
            ax.set_ylim(0, 1.02)
            for r in sorted({g[0] for g in grid}):
                ok = [
                    f
                    for i in pick
                    for (rr, f, _) in [grid[i]]
                    if rr == r and wins(dv, chamfer[i])
                ]
                lines.append(
                    f"  sigma {sigma}, r {r}: "
                    + (f"beats with up to f {max(ok):g}" if ok else "no f beats")
                )
        fig.colorbar(points, ax=row, label="mean Chamfer (cm), log scale", shrink=0.8)
    fig.suptitle(
        f"E69c rung 0, split {split}: oracle-ovoxel damaged by dropped children (recall), added empty children "
        "(precision) and vertex noise. Red line: Deja View's Chamfer on that split; star: dv-ovoxel.",
        fontsize=11,
    )
    fig.savefig(figure, dpi=90)
    table.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
