"""E8a: the bedroom (views 18, 0, 12, 6) refined with each kind of segments.

Columns: image; the segments SAM 3 gives the refiner (one colour per linked id); each
pixel's distance to the nearest GT point refined with SAM 3's segments; then without
segments, with GT instances, and with SAM 3's ids permuted per view, each minus SAM 3
on ±2 cm (blue: closer to the GT than with SAM 3). One run of
`experiments/configs/e8a_segments.yaml`.

uv run python experiments/scripts/e8a_figures.py runs/<run> \
    > experiments/results/e8a_figures.txt
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
    PRESENTATION_SCENE,
    PRESENTATION_VIEWS,
    SEQUENTIAL,
)

sys.path.insert(0, str(Path(__file__).parent))
from e40_figures import distances, variant_points

CHANGE_CM = 2.0
COMPARED = ("Off", "GT", "permuted", "shifted")
OUT = Path("experiments/figures/e8a_bedroom.jpg")


def id_colours(ids: np.ndarray) -> np.ndarray:
    """RGB per pixel, one fixed random colour per id, grey where there is none."""
    rng = np.random.default_rng(0)
    unique, index = np.unique(ids, return_inverse=True)
    palette = rng.uniform(0.15, 0.95, (len(unique), 3))
    palette[unique < 0] = 0.5
    return palette[index.reshape(ids.shape)]


def main() -> None:
    """Draw the figure and print each variant's mean change on the bedroom."""
    run_dir = Path(sys.argv[1])
    raw = yaml.safe_load((run_dir / "run.yaml").read_text(encoding="utf-8"))["config"]
    inputs = load_inputs(raw, PRESENTATION_SCENE, torch.device("cpu"))
    tree = cKDTree(inputs.truth.points)
    ref = distances(tree, variant_points(run_dir, "SAM 3", inputs))
    changes = {}
    for name in COMPARED:
        changes[name] = distances(tree, variant_points(run_dir, name, inputs)) - ref
        c = changes[name]
        print(
            f"{PRESENTATION_SCENE}, all 20 views, {name} minus SAM 3: mean"
            f" {c.mean():+.4f} cm, pixels changed by > 0.5 cm {(np.abs(c) > 0.5).mean():.4f}"
        )
    shown = COMPARED[:3]
    titles = [
        "image",
        "SAM 3 segments\n(linked ids)",
        "refined with SAM 3\ndistance to GT (cm)",
        *(f"{n}\nminus SAM 3" for n in shown),
    ]
    views = PRESENTATION_VIEWS
    fig, axes = plt.subplots(
        len(views),
        len(titles),
        figsize=(2.0 * len(titles), 1.6 * len(views) + 1.2),
        layout="constrained",
    )
    colours = id_colours(inputs.labels.instance_ids)
    for row, v in zip(axes, views, strict=True):
        row[0].imshow(inputs.views.images[v])
        row[1].imshow(colours[v])
        im = row[2].imshow(ref[v], cmap=SEQUENTIAL, norm=Normalize(0, 10))
        for ax, name in zip(row[3:], shown, strict=True):
            ch = ax.imshow(
                changes[name][v], cmap=DIVERGING, norm=Normalize(-CHANGE_CM, CHANGE_CM)
            )
        for ax in row:
            ax.set_xticks([])
            ax.set_yticks([])
        row[0].set_ylabel(f"view {v}")
    for ax, title in zip(axes[0], titles, strict=True):
        ax.set_title(title, fontsize=8)
    fig.colorbar(im, ax=axes[-1, 2], location="bottom", label="cm, capped at 10")
    fig.colorbar(
        ch,
        ax=axes[-1, 3:].tolist(),
        location="bottom",
        label="cm, blue: closer than SAM 3",
    )
    fig.suptitle(
        "Bedroom ai_006_007: the joint refiner given SAM 3's, no, GT and permuted segments",
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
