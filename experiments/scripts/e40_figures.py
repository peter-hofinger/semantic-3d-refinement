"""E40: the bedroom's views 18, 0, 12, 6 under each change that passed dev, and after.

Per pixel, the distance of its refined point to the nearest ground-truth point (after
the variant's own Umeyama Sim(3), as the protocol's accuracy, capped at 30 cm); first
the refiner before E40 (the sweep's "Refined"), then, as that distance minus the
refiner's before, on one ±2 cm scale: each change that passed dev alone (sweep run),
the two combinations checked on val (runs at 954bfc1 and a97064a, variant "Refined")
and the final energy (the refiner before E40 up to λ_d and conf_power, so this column
is the GPU's run-to-run spread). All runs keep dev's per-scene depth.

uv run python experiments/scripts/e40_figures.py runs/<sweep> runs/<combination 1> \
    runs/<combination 2> runs/<final> > experiments/results/e40_figures.txt
"""

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

from src.analysis.scene_eval import load_inputs
from src.analysis.scene_figures import (
    DIVERGING,
    ERROR_MAX_CM,
    PRESENTATION_SCENE,
    PRESENTATION_VIEWS,
    SEQUENTIAL,
    pixel_points,
)

REFERENCE = "Refined"  # the refiner of a run's code: before E40 in the sweep
ADOPTED = ("huber quadratic", "border 0 px", "edge_tol Otsu")
TITLES = {
    "huber quadratic": "no Huber",
    "border 0 px": "no border margin",
    "edge_tol Otsu": "E_n cut at Otsu",
}
RUNS = (
    "combination 1: all three,\nno confidence",
    "combination 2: all three,\nconfidence kept",
    "final (= before)",
)
CHANGE_CM = 2.0
OUT = Path("experiments/figures/e40_bedroom.jpg")


def distances(tree: cKDTree, points: np.ndarray) -> np.ndarray:
    """Distance (cm) of each pixel's point to the nearest GT point, capped."""
    cap = ERROR_MAX_CM / 100
    d = tree.query(points.reshape(-1, 3), workers=-1, distance_upper_bound=cap)[0]
    return 100 * np.minimum(d, cap).reshape(points.shape[:-1])


def variant_points(run_dir: Path, variant: str, inputs: object) -> np.ndarray:
    """Per-pixel points of one variant of a run, registered onto the ground truth."""
    run = yaml.safe_load((run_dir / "run.yaml").read_text(encoding="utf-8"))
    names = [v["name"] for v in run["config"]["variants"]]
    arrays = np.load(run_dir / "results.npz")
    i = names.index(variant)
    key = f"{PRESENTATION_SCENE}/{i}"
    return pixel_points(inputs, arrays[key], arrays[f"{key}/align"])


def main() -> None:
    """Draw the figure and print the mean change per column."""
    sweep, *others = map(Path, sys.argv[1:])
    assert len(others) == len(RUNS), sys.argv
    raw = yaml.safe_load((sweep / "run.yaml").read_text(encoding="utf-8"))["config"]
    inputs = load_inputs(raw, PRESENTATION_SCENE, torch.device("cpu"))
    tree = cKDTree(inputs.truth.points)
    reference = distances(tree, variant_points(sweep, REFERENCE, inputs))
    columns = {TITLES[v]: variant_points(sweep, v, inputs) for v in ADOPTED}
    for title, run in zip(RUNS, others, strict=True):
        columns[title] = variant_points(run, REFERENCE, inputs)
    changes = {k: distances(tree, p) - reference for k, p in columns.items()}
    print(f"bedroom {PRESENTATION_SCENE}, all 20 views: mean distance change (cm)")
    for k, c in changes.items():
        print(
            f"{k.replace(chr(10), ' ')}: {c.mean():+.4f}, pixels changed by > 0.5 cm: "
            f"{(np.abs(c) > 0.5).mean():.4f}"
        )
    titles = ["image", f"{REFERENCE} before E40\ndistance (cm)"]
    titles += [f"{k}\nminus before" for k in changes]
    views = PRESENTATION_VIEWS
    fig, axes = plt.subplots(
        len(views),
        len(titles),
        figsize=(2.3 * len(titles), 1.9 * len(views) + 1.2),
        layout="constrained",
    )
    for row, v in zip(axes, views, strict=True):
        row[0].imshow(inputs.views.images[v])
        im = row[1].imshow(reference[v], cmap=SEQUENTIAL, norm=Normalize(0, 10))
        for ax, c in zip(row[2:], changes.values(), strict=True):
            ch = ax.imshow(c[v], cmap=DIVERGING, norm=Normalize(-CHANGE_CM, CHANGE_CM))
        for ax in row:
            ax.set_xticks([])
            ax.set_yticks([])
        row[0].set_ylabel(f"view {v}")
    for ax, title in zip(axes[0], titles, strict=True):
        ax.set_title(title, fontsize=8)
    fig.colorbar(im, ax=axes[-1, 1], location="bottom", label="cm, capped at 10")
    fig.colorbar(
        ch, ax=axes[-1, 2:], location="bottom", shrink=0.6, label="cm, blue: closer"
    )
    fig.suptitle(
        "Bedroom ai_006_007: each pixel's distance to the nearest ground-truth point, "
        "before E40, and its change under each setting that passed dev",
        x=0.01,
        ha="left",
        fontsize=9,
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=110)
    plt.close(fig)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
