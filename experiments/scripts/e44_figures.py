"""E44: the bedroom (views 18, 0, 12, 6) with E_mv at the nearest pixel vs bilinear.

Columns: image; each pixel's distance to the nearest GT point with E_mv reading the
other view bilinearly (before, the run at 150db44), and the change with E_mv reading
its nearest pixel (after, the run at d7cbef4), after minus before on ±2 cm. Both runs
are `configs/eval_hypersim.yaml --splits dev stress` on the same GPU.

uv run python experiments/scripts/e44_figures.py runs/<before> runs/<after> \
    > experiments/results/e44_figures.txt
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
OUT = Path("experiments/figures/e44_bedroom.jpg")


def main() -> None:
    """Draw the figure and print the mean change."""
    before, after = Path(sys.argv[1]), Path(sys.argv[2])
    raw = yaml.safe_load((after / "run.yaml").read_text(encoding="utf-8"))["config"]
    inputs = load_inputs(raw, PRESENTATION_SCENE, torch.device("cpu"))
    tree = cKDTree(inputs.truth.points)
    ref = distances(tree, variant_points(before, "Refined", inputs))
    change = distances(tree, variant_points(after, "Refined", inputs)) - ref
    print(
        f"{PRESENTATION_SCENE}, all 20 views: distance to GT, nearest minus bilinear: "
        f"mean {change.mean():+.4f} cm, pixels changed by > 0.5 cm: "
        f"{(np.abs(change) > 0.5).mean():.4f}"
    )
    titles = [
        "image",
        "bilinear E_mv (before)\ndistance (cm)",
        "nearest pixel\nminus bilinear",
    ]
    views = PRESENTATION_VIEWS
    fig, axes = plt.subplots(
        len(views), 3, figsize=(7.2, 1.9 * len(views) + 1.2), layout="constrained"
    )
    for row, v in zip(axes, views, strict=True):
        row[0].imshow(inputs.views.images[v])
        im = row[1].imshow(ref[v], cmap=SEQUENTIAL, norm=Normalize(0, 10))
        ch = row[2].imshow(
            change[v], cmap=DIVERGING, norm=Normalize(-CHANGE_CM, CHANGE_CM)
        )
        for ax in row:
            ax.set_xticks([])
            ax.set_yticks([])
        row[0].set_ylabel(f"view {v}")
    for ax, title in zip(axes[0], titles, strict=True):
        ax.set_title(title, fontsize=8)
    fig.colorbar(im, ax=axes[-1, 1], location="bottom", label="cm, capped at 10")
    fig.colorbar(
        ch, ax=axes[-1, 2], location="bottom", label="cm, blue: nearest closer"
    )
    fig.suptitle(
        "Bedroom ai_006_007: E_mv reading the other view at its nearest pixel",
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
