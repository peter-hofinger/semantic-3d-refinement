"""E57c: the distance target and what cutting by it keeps, on the bedroom.

Rows: views 18, 0, 12, 6 of `ai_006_007`. Columns: the image; the true distance of each
pixel's point to the ground-truth surface; the distance a head predicts (`--head`, a
model that trained on this scene when it comes from an overfit rung: a diagnostic, not a
result); the pixels dropped at keep 0.5 by the true distance (the oracle) and by instance
voxel density (E45, E27a). All distances in percent of the scene's median depth, one
shared colour scale.

uv run python experiments/scripts/e57c_figures.py --head volume-dist-none \
    > experiments/results/e57c_figures.txt
"""

import argparse
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from src.analysis.filters import instance_voxel_density, keep_top, ranks
from src.analysis.scene_eval import load_inputs
from src.analysis.scene_figures import PRESENTATION_SCENE, PRESENTATION_VIEWS
from src.refinement.multiview import points_from_rays
from src.refinement.volume import VolumeConfig, load_scene

OUT = Path("experiments/figures")
LIMIT = 3.0  # percent of the median depth, the colour scale's top


def main() -> None:
    """Draw the figure and print the dropped shares per view."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--head", help="model whose `<name>_keep_<views>.npz` to draw")
    parser.add_argument("--out", default="e57c_bedroom.jpg")
    args = parser.parse_args()
    cfg = VolumeConfig.load(Path("configs/e57c_volume.yaml"))
    raw = yaml.safe_load(Path("experiments/configs/e57c_oracle.yaml").read_text())
    _, geometry, _, _, distance = load_scene(cfg, PRESENTATION_SCENE, "none")
    assert distance is not None
    true = 100 * distance.reshape(geometry.depth.shape)
    inputs = load_inputs(raw, PRESENTATION_SCENE, torch.device("cpu"))
    start = points_from_rays(inputs.scene.depth.double(), inputs.scene.rays.double())
    density = instance_voxel_density(
        start.numpy(),
        inputs.scene.depth.numpy(),
        inputs.scene.k.numpy(),
        inputs.labels.instance_ids,
    )
    kept_density = keep_top(ranks(density), 0.5)
    kept_true = keep_top(ranks(-true), 0.5)
    folder = cfg.root / PRESENTATION_SCENE
    predicted = None
    if args.head:
        keep = np.load(folder / f"{args.head}_keep_{cfg.views}.npz")["keep"]
        predicted = -100 * keep
    columns = ["image", "true distance (%)"]
    columns += ["predicted distance (%)"] if predicted is not None else []
    columns += ["dropped by true distance", "dropped by density"]
    fig, axes = plt.subplots(
        len(PRESENTATION_VIEWS), len(columns), figsize=(3.3 * len(columns), 2.6 * 4)
    )
    for row, v in enumerate(PRESENTATION_VIEWS):
        panels = [("image", inputs.views.images[v], None)]
        panels += [("true", true[v], LIMIT)]
        if predicted is not None:
            panels += [("pred", predicted[v], LIMIT)]
        panels += [("o", ~kept_true[v], 1), ("d", ~kept_density[v], 1)]
        for col, (_, image, top) in enumerate(panels):
            ax = axes[row, col]
            drawn = ax.imshow(
                image, cmap=None if top is None else "magma", vmin=0, vmax=top
            )
            if col == 1:
                shown = drawn
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(columns[col], fontsize=10)
            if col == 0:
                ax.set_ylabel(f"view {v}")
        print(
            f"view {v}: dropped by true distance {100 * (~kept_true[v]).mean():.1f}%, "
            f"by density {100 * (~kept_density[v]).mean():.1f}%, "
            f"mean true distance {true[v].mean():.2f}%"
        )
    fig.colorbar(shown, ax=axes, shrink=0.5, label="percent of the median depth")
    fig.suptitle("E57c, bedroom ai_006_007: kept 0.5 of the pixels per scene")
    fig.savefig(
        OUT / args.out, dpi=110, bbox_inches="tight", pil_kwargs={"quality": 88}
    )


if __name__ == "__main__":
    main()
