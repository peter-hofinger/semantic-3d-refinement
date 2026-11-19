"""E57c: the squiggle of the straight-run fusion in category colors, from the pulled-back camera and in its source views.

Categories (`e57c_two_fills`): white kept by the confidence cut, green oracle-kept beyond 8 cm from it, gold blue pixels
brought back by the straight-run rule, blue oracle-kept but not brought back, red dropped by both. Figure 1: the camera of the
pulled-back view 18, each pixel of all 20 views colored by its category, nearest point wins, with the red points and without
(the points of the fusion plus the blue ones), and a zoom on `e57c_squiggles.ZONE`. Figure 2: the zone's three main source views in
image space, the squiggle's source pixels (the blue pixels that would cover the gap, as `e57c_squiggles`) circled. Printed: the 3D
distance (cm) of those pixels to the nearest white and to the nearest green pixel.

uv run python -m experiments.scripts.e57c_squiggle_categories > experiments/results/e57c_squiggle_categories.txt
"""

from collections import Counter
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from matplotlib.colors import ListedColormap
from scipy.spatial import cKDTree

from experiments.scripts.e57c_hybrid import far_from
from experiments.scripts.e57c_oracle_curve import greedy, rank_scores
from experiments.scripts.e57c_seams import DELTA, LEVELS, drop_order
from experiments.scripts.e57c_squiggles import ZONE, render_ids
from experiments.scripts.e57c_two_fills import (
    BLUE,
    COLORS,
    GOLD,
    GREEN,
    RED,
    WHITE,
    runs_on_lines,
)
from src.analysis.scene_eval import load_inputs
from src.analysis.scene_figures import (
    PRESENTATION_SCENE,
    PRESENTATION_VIEWS,
    render_cameras,
)

BACKGROUND = 0.55  # gray: nothing drawn
NAMES = [
    "kept by confidence",
    "dropped by confidence and oracle",
    "oracle > 8 cm from kept",
    "oracle-kept, not brought back (blue)",
    "blue brought back by the straight-run rule",
]


def main() -> None:
    """Draw the two figures and print the distances."""
    raw = yaml.safe_load(Path("experiments/configs/e57c_oracle.yaml").read_text())
    inputs = load_inputs(raw, PRESENTATION_SCENE, torch.device("cpu"))
    truth, pixels = inputs.truth, inputs.truth.pixels
    assert pixels is not None
    points, align, pred, a, _, ranking = rank_scores(inputs)
    v_conf, v_oracle = drop_order(
        ranking["confidence"].astype(np.float64), greedy(truth, pred, a, LEVELS)
    )
    base, oracle = (1 - v_conf) >= 0.5, v_oracle <= 0.5
    gate = np.where(
        base,
        WHITE,
        np.where(oracle, np.where(far_from(pred, base, DELTA), GREEN, BLUE), RED),
    )
    category = np.full(pixels.shape, WHITE)
    category[pixels] = gate
    ids = inputs.labels.instance_ids
    category = np.where(
        np.stack([runs_on_lines(category[v], ids[v]) for v in range(len(ids))]),
        GOLD,
        category,
    )
    aligned = points @ align[:3, :3].T + align[:3, 3]
    camera, _ = render_cameras(inputs, PRESENTATION_VIEWS)["view 18, pulled back"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, h, w = category.shape
    palette = np.array([mpl.colors.to_rgb(c) for c in COLORS])  # indexed by category

    def paint(keep: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        image = render_ids(inputs, aligned, keep, camera, device)
        shown = np.where(
            (image >= 0)[..., None],
            palette[category.reshape(-1)[np.maximum(image, 0)]],
            BACKGROUND,
        )
        return shown, image

    everything, _ = paint(np.ones(category.shape, dtype=bool))
    without_red, _ = paint(category != RED)
    # the squiggle's source pixels: blue pixels that would cover a gap of the fusion
    kept = np.isin(category, [WHITE, GREEN, GOLD])
    ids_kept = render_ids(inputs, aligned, kept, camera, device)
    ids_blue = render_ids(inputs, aligned, category == BLUE, camera, device)
    zone = np.zeros(ids_kept.shape, dtype=bool)
    zone[ZONE] = True
    src = np.unique(ids_blue[(ids_kept < 0) & (ids_blue >= 0) & zone])
    flat = aligned.reshape(-1, 3)
    for name, kind in (("white", WHITE), ("green", GREEN)):
        tree = cKDTree(
            flat[np.flatnonzero((category == kind).ravel())],
            balanced_tree=False,
            compact_nodes=False,
        )
        dist = 100 * tree.query(flat[src], workers=-1)[0]
        print(
            f"3D distance of the strip's {len(src)} blue source pixels to the nearest {name} pixel (cm): median {np.median(dist):.2f}, 10th {np.percentile(dist, 10):.2f}, 90th {np.percentile(dist, 90):.2f}"
        )
        if name == "white":
            to_white = dist
        else:
            to_green = dist
    total = to_white + to_green
    print(
        f"sum of the two (cm): median {np.median(total):.2f}, 10th {np.percentile(total, 10):.2f}, 90th {np.percentile(total, 90):.2f}; gate {DELTA:g} cm"
    )

    fig, axes = plt.subplots(2, 2, figsize=(15, 11.5), layout="constrained")
    for ax, image, title in (
        (axes[0, 0], everything, "all pixels, nearest point wins (red included)"),
        (
            axes[0, 1],
            without_red,
            "without the red pixels: the fusion's kept pixels and the blue ones",
        ),
        (axes[1, 0], everything[ZONE], "zoom on the zone, red included"),
        (axes[1, 1], without_red[ZONE], "zoom on the zone, without red"),
    ):
        ax.imshow(image, interpolation="nearest")
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=COLORS[k], edgecolor="0.5")
        for k in (WHITE, GREEN, GOLD, BLUE, RED)
    ]
    fig.legend(
        handles,
        [NAMES[k] for k in (WHITE, GREEN, GOLD, BLUE, RED)],
        loc="outside lower center",
        ncols=5,
        fontsize=8,
    )
    fig.suptitle(
        f"{PRESENTATION_SCENE}, pulled-back view 18, pixels colored by category (gray: nothing)",
        fontsize=10,
    )
    fig.savefig("experiments/figures/e57c_squiggle_categories_render.jpg", dpi=80)
    plt.close(fig)

    top = [v for v, _ in Counter((src // (h * w)).tolist()).most_common(3)]
    fig, axes = plt.subplots(
        1, len(top), figsize=(6 * len(top), 5.6), layout="constrained"
    )
    for ax, v in zip(np.atleast_1d(axes), top, strict=True):
        mine = src[(src // (h * w)) == v]
        ax.imshow(
            category[v],
            cmap=ListedColormap(COLORS),
            vmin=-0.5,
            vmax=4.5,
            interpolation="nearest",
        )
        ax.scatter(
            (mine % w),
            (mine % (h * w)) // w,
            s=14,
            facecolors="none",
            edgecolors="magenta",
            linewidths=0.6,
        )
        ax.set_title(
            f"view {v}: the strip's {len(mine)} source pixels circled", fontsize=10
        )
        ax.set_xticks([])
        ax.set_yticks([])
    fig.legend(
        handles,
        [NAMES[k] for k in (WHITE, GREEN, GOLD, BLUE, RED)],
        loc="outside lower center",
        ncols=5,
        fontsize=8,
    )
    fig.savefig("experiments/figures/e57c_squiggle_categories_sources.jpg", dpi=80)
    plt.close(fig)


if __name__ == "__main__":
    main()
