"""E45: accuracy-completeness curves of each filter, the refiner as a point, the bedroom.

Reads `e45_filters.py`'s `runs/<run>/e45_<split>.json` and the run itself.
- `e45_curves_<split>.jpg`: one panel per filter (rows) and scene (columns, then the
  split's mean at each q): the filter's curve on Déjà View's points, random removal
  (the floor), the refiner's own confidence curve and the refiner. Lower is better on
  both axes, so a curve below and left of the refiner's point beats it.
- `e45_equal_completeness.jpg`: each filter's accuracy and Chamfer minus the refiner's at
  the refiner's completeness, mean of each split's scenes (dev, stress), against E40's floor.
- `e45_bedroom.jpg`: the bedroom's views 18, 0, 12, 6, the best filter on dev (lowest
  mean accuracy at equal completeness) cut at the refiner's completeness there: the
  pixels it removes, each pixel's distance to the GT for it and for the refiner.
- `e45_ai_006_007_render.png`: both clouds rendered from the same cameras
  (`scene_figures.render_figure`).

uv run python experiments/scripts/e45_figures.py runs/<run> \
    > experiments/results/e45_figures.txt
"""

import json
import sys
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from matplotlib.colors import Normalize
from scipy.spatial import cKDTree

from src.analysis.filters import keep_top, ranks
from src.analysis.scene_eval import load_inputs
from src.analysis.scene_figures import (
    PRESENTATION_SCENE,
    PRESENTATION_VIEWS,
    SCENE_COLORS,
    SEQUENTIAL,
    render_figure,
)
from src.evaluation.labeled import align_output
from src.refinement.multiview import points_from_rays

sys.path.insert(0, str(Path(__file__).parent))
from e40_figures import distances, variant_points
from e45_filters import FLOOR, REFINED_FILTER, filter_scores

OUT = Path("experiments/figures")
FILTER, REFINER, REFERENCE = SCENE_COLORS[0], SCENE_COLORS[1], "#8c8b86"
SPLITS = ("dev", "stress")  # val was not run (E45)
DISTANCE_CM = 10.0
X_REACH = 1.6  # panels show completeness up to this multiple of the refiner's


def curve(result: dict, key: str) -> np.ndarray:
    """One metric along a sweep's grid, q descending."""
    return np.array([p[key] for p in result["curve"]])


def mean_curve(results: list[dict], key: str) -> np.ndarray:
    """Return the metric at each q, mean over scenes."""
    return np.mean([curve(r, key) for r in results], axis=0)


def curves_figure(split: str, data: dict) -> None:
    """Small multiples: filters x (scenes, mean)."""
    scenes = list(data["scenes"])
    filters = [
        f for f in data["scenes"][scenes[0]] if f not in ("random", REFINED_FILTER)
    ]
    columns = [*scenes, "mean"]
    fig, axes = plt.subplots(
        len(filters),
        len(columns),
        figsize=(2.7 * len(columns), 1.9 * len(filters) + 0.8),
        sharex="col",
        sharey="col",
        layout="constrained",
    )
    for c, col in enumerate(columns):
        names = scenes if col == "mean" else [col]
        records = [data["scenes"][n] for n in names]
        ref = {
            k: np.mean([data["refiner"][n][k] for n in names])
            for k in ("accuracy_cm", "completeness_cm")
        }

        def xy(f: str, recs: list[dict] = records) -> tuple[np.ndarray, np.ndarray]:
            rs = [r[f] for r in recs]
            return mean_curve(rs, "completeness_cm"), mean_curve(rs, "accuracy_cm")

        for r, f in enumerate(filters):
            ax = axes[r, c]
            ax.plot(*xy("random"), color=REFERENCE, lw=1, ls="--", label="random")
            ax.plot(*xy(REFINED_FILTER), color=REFINER, lw=1, label=REFINED_FILTER)
            ax.plot(*xy(f), color=FILTER, lw=2, label="filter")
            x0, y0 = xy(f)
            ax.plot(x0[0], y0[0], "o", color="black", ms=4, label="Déjà View")
            ax.plot(
                ref["completeness_cm"],
                ref["accuracy_cm"],
                "*",
                color=REFINER,
                ms=11,
                mec="white",
                label="refiner",
            )
            ax.set_xlim(0.9 * x0[0], X_REACH * ref["completeness_cm"])
            ax.grid(alpha=0.3)
            ax.tick_params(labelsize=7)
            if r == 0:
                ax.set_title(col.removesuffix("_cam_00"), fontsize=9)
            if c == 0:
                ax.set_ylabel(f"{f}\naccuracy (cm)", fontsize=8)
            if r == len(filters) - 1:
                ax.set_xlabel("completeness (cm)", fontsize=8)
        for ax in axes[:, c]:
            visible = [
                y
                for f in filters
                for x, y in zip(*xy(f), strict=True)
                if x <= X_REACH * ref["completeness_cm"]
            ] + [ref["accuracy_cm"], *xy(REFINED_FILTER)[1]]
            low, high = min(visible), max(visible)
            ax.set_ylim(low - 0.05 * (high - low), high + 0.05 * (high - low))
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncols=5, fontsize=8)
    fig.suptitle(
        f"E45, Hypersim {split}: accuracy against completeness as each filter drops "
        "Déjà View's points (q = 1 at the black dot, then 0.95, 0.9, ...); lower is "
        "better on both axes",
        x=0.01,
        ha="left",
        fontsize=9,
    )
    fig.savefig(OUT / f"e45_curves_{split}.jpg", dpi=100)
    plt.close(fig)


def equal_completeness(data: dict) -> dict[str, dict[str, float]]:
    """Per filter, mean over scenes of (filter - refiner) at equal completeness."""
    scenes = list(data["scenes"])
    out = {}
    for f in data["scenes"][scenes[0]]:
        matches = [data["scenes"][n][f]["match"] for n in scenes]
        if f == REFINED_FILTER or any(m is None for m in matches):
            continue
        out[f] = {
            k: float(
                np.mean(
                    [
                        m[k] - data["refiner"][n][k]
                        for n, m in zip(scenes, matches, strict=True)
                    ]
                )
            )
            for k in ("accuracy_cm", "chamfer_cm")
        }
    return out


def summary_figure(deltas: dict[str, dict]) -> None:
    """Dot plot of each filter's difference to the refiner, per split, with the floor."""
    filters = sorted(deltas["dev"], key=lambda f: deltas["dev"][f]["accuracy_cm"])
    fig, axes = plt.subplots(
        1, 2, figsize=(9, 0.4 * len(filters) + 1.6), layout="constrained"
    )
    for ax, key, title in zip(
        axes, ("accuracy_cm", "chamfer_cm"), ("accuracy", "Chamfer"), strict=True
    ):
        ax.axvspan(-FLOOR[key], FLOOR[key], color=REFERENCE, alpha=0.25, lw=0)
        ax.axvline(0, color=REFERENCE, lw=1)
        for s, (split, color) in enumerate(zip(SPLITS, SCENE_COLORS, strict=False)):
            ys = np.arange(len(filters)) + (s - 0.5) * 0.25
            xs = [deltas[split].get(f, {}).get(key, np.nan) for f in filters]
            ax.plot(xs, ys, "o", color=color, ms=6, label=split, mec="white")
        ax.set_yticks(np.arange(len(filters)), filters, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel(
            f"filter minus refiner, {title} (cm); > 0: refiner better", fontsize=8
        )
        ax.grid(axis="x", alpha=0.3)
    axes[1].set_yticklabels([])
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncols=2, fontsize=8)
    fig.suptitle(
        "E45: each filter of Déjà View's points cut to the refiner's completeness on\n"
        "every scene, mean over the split's scenes; gray band: E40's noise floor",
        x=0.01,
        ha="left",
        fontsize=9,
    )
    fig.savefig(OUT / "e45_equal_completeness.jpg", dpi=110)
    plt.close(fig)


def bedroom_figures(run_dir: Path, raw: dict, best: str, q: float) -> None:
    """Draw the bedroom: what the best filter removes, distance to GT, renders."""
    inputs = load_inputs(raw, PRESENTATION_SCENE, torch.device("cpu"))
    scene = inputs.scene
    keep = keep_top(ranks(filter_scores(inputs)[best]), q)
    points = points_from_rays(scene.depth.double(), scene.rays.double()).numpy()
    _, _, align = align_output(inputs.truth, np.where(keep[..., None], points, np.nan))
    registered = points @ align[:3, :3].T + align[:3, 3]
    filtered = np.where(keep[..., None], registered, np.nan)
    refined = variant_points(run_dir, "Refined", inputs)
    tree = cKDTree(inputs.truth.points)
    d_filter = np.where(keep, distances(tree, np.nan_to_num(filtered)), np.nan)
    d_refined = distances(tree, refined)
    print(
        f"bedroom: {best} at q = {q:.3f} keeps {keep.mean():.3f} of the pixels; mean "
        f"distance to GT over kept pixels {np.nanmean(d_filter):.3f} cm, refiner over "
        f"all {d_refined.mean():.3f} cm"
    )
    views = PRESENTATION_VIEWS
    titles = [
        f"image, red: removed by\n{best}",
        f"{best}, kept points\ndistance to GT (cm)",
        "refiner\ndistance to GT (cm)",
    ]
    fig, axes = plt.subplots(
        len(views), 3, figsize=(8.0, 2.0 * len(views) + 1.2), layout="constrained"
    )
    norm = Normalize(0, DISTANCE_CM)
    cmap = SEQUENTIAL.with_extremes(bad="#d9d8d4")
    for row, v in zip(axes, views, strict=True):
        image = inputs.views.images[v].astype(float) / 255
        image[~keep[v]] = 0.4 * image[~keep[v]] + 0.6 * np.array([0.89, 0.29, 0.28])
        row[0].imshow(image)
        im = row[1].imshow(d_filter[v], cmap=cmap, norm=norm)
        row[2].imshow(d_refined[v], cmap=cmap, norm=norm)
        for ax in row:
            ax.set_xticks([])
            ax.set_yticks([])
        row[0].set_ylabel(f"view {v} (kept {keep[v].mean():.2f})", fontsize=8)
    for ax, title in zip(axes[0], titles, strict=True):
        ax.set_title(title, fontsize=8)
    fig.colorbar(
        im, ax=axes[-1, 1:], location="bottom", label="cm, capped at 10; gray: removed"
    )
    fig.suptitle(
        f"Bedroom ai_006_007: {best} cut to the refiner's completeness (q = {q:.2f}) "
        "next to the refiner",
        x=0.01,
        ha="left",
        fontsize=9,
    )
    fig.savefig(OUT / "e45_bedroom.jpg", dpi=110)
    plt.close(fig)
    render_figure(
        inputs,
        {"refiner": refined, f"{best}, q = {q:.2f}": (registered, keep)},
        views,
        "e45_ai_006_007",
        OUT,
    )


def main() -> None:
    """Draw every figure and print what they are drawn from."""
    run_dir = Path(sys.argv[1])
    raw = yaml.safe_load((run_dir / "run.yaml").read_text(encoding="utf-8"))["config"]
    data = {
        s: json.loads((run_dir / f"e45_{s}.json").read_text(encoding="utf-8"))
        for s in SPLITS
    }
    OUT.mkdir(parents=True, exist_ok=True)
    for split in ("dev", "stress"):
        curves_figure(split, data[split])
    deltas = {s: equal_completeness(data[s]) for s in SPLITS}
    for s in SPLITS:
        print(
            f"{s}: filter minus refiner at equal completeness (accuracy / Chamfer, cm)"
        )
        for f, d in deltas[s].items():
            print(f"  {f}: {d['accuracy_cm']:+.3f} / {d['chamfer_cm']:+.3f}")
    summary_figure(deltas)
    best = min(deltas["dev"], key=lambda f: deltas["dev"][f]["accuracy_cm"])
    q = data["dev"]["scenes"][PRESENTATION_SCENE][best]["match"]["q"]
    print(f"best filter on dev (accuracy at equal completeness): {best}")
    bedroom_figures(run_dir, raw, best, q)


if __name__ == "__main__":
    main()
