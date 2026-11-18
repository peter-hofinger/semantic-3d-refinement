"""E57c: the confidence cut, the oracle's order and the SAM 3 masks side by side, and where the seams come from.

Per pixel of the bedroom's presentation views, the drop order v in [0, 1]: a pixel is kept at
the kept fraction q iff v <= q, for both rankings, so the two maps read the same way (red
drops first, green stays longest, the isoline is the q 0.5 cut):
- confidence: v = 1 - (the pixel's percentile of Déjà View's confidence over the cloud);
- the greedy oracle (`e57c_oracle_curve.greedy`, 10% of the pixels per round down to q 0.1): v
  = the largest kept fraction at which the pixel is still kept (1.0 dropped in the first round;
  0.05 never dropped by q 0.1). It is the oracle's own value, not a transform of a_i or m_i.
Third column the SAM 3 masks on the image; fourth the combination of `e57c_hybrid` B at
q_c 0.5, q_g 0.5, delta 8 cm: kept by confidence (green); added from the oracle, farther than
delta from every kept pixel (blue); kept by the oracle but within delta of a confidence-kept pixel,
so not added, the seam (orange); dropped (gray). The same pixels, cropped to the view's
largest SAM 3 mask and white outside it, in the second figure.

uv run python -m experiments.scripts.e57c_seams
"""

from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from matplotlib.colors import ListedColormap

from experiments.scripts.e57c_hybrid import far_from
from experiments.scripts.e57c_oracle_curve import greedy, rank_scores
from src.analysis.scene_eval import load_inputs
from src.analysis.scene_figures import (
    PRESENTATION_SCENE,
    PRESENTATION_VIEWS,
    instance_colors,
)

DELTA = 8.0  # cm
LEVELS = [round(1 - 0.1 * r, 1) for r in range(1, 10)]  # 0.9 ... 0.1
CATEGORIES = [
    "0.8",
    "tab:green",
    "tab:blue",
    "tab:orange",
]  # dropped, conf, added, seam
NAMES = [
    "dropped",
    "kept by confidence",
    "oracle, > 8 cm from kept",
    "oracle, within 8 cm (seam)",
]


def drop_order(
    conf: np.ndarray, path: dict[float, np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    """Return v for the confidence and for the oracle path, (N,) each."""
    percentile = np.argsort(np.argsort(conf, kind="stable"), kind="stable") / (
        len(conf) - 1
    )
    v_oracle = np.full(len(conf), 0.05)
    for level in LEVELS:  # descending: the last level a pixel is kept at wins
        v_oracle[path[level]] = level
    v_oracle[~path[LEVELS[0]]] = 1.0
    return 1 - percentile, v_oracle


def main() -> None:
    """Draw the two figures."""
    raw = yaml.safe_load(Path("experiments/configs/e57c_oracle.yaml").read_text())
    inputs = load_inputs(raw, PRESENTATION_SCENE, torch.device("cpu"))
    truth, pixels = inputs.truth, inputs.truth.pixels
    assert pixels is not None
    _, _, pred, a, _, ranking = rank_scores(inputs)
    conf = ranking["confidence"].astype(np.float64)
    path = greedy(truth, pred, a, LEVELS)
    v_conf, v_oracle = drop_order(conf, path)
    base = v_conf <= 0.5
    oracle = v_oracle <= 0.5
    far = far_from(pred, base, DELTA)
    category = np.where(base, 1, np.where(oracle & far, 2, np.where(oracle, 3, 0)))
    print(
        "share of pixels (GT pixels):",
        {n: round(float((category == i).mean()), 3) for i, n in enumerate(NAMES)},
    )

    block = (
        np.arange(pixels.shape[0])[:, None, None] * 10**6 + inputs.labels.instance_ids
    )[pixels]
    for name, kept in (("confidence", base), ("greedy oracle", oracle)):
        keys, inverse = np.unique(block, return_inverse=True)
        share = np.bincount(inverse, weights=kept) / np.bincount(inverse)
        pure = ((share < 0.1) | (share > 0.9))[inverse].mean()
        print(
            f"{name} q 0.5: {pure:.3f} of the pixels lie in a (view, instance) block that it keeps < 10% or > 90% of; {len(keys)} blocks"
        )

    def full(values: np.ndarray) -> np.ndarray:
        out = np.full(pixels.shape, np.nan)
        out[pixels] = values
        return out

    maps = {
        "confidence": full(v_conf),
        "oracle": full(v_oracle),
        "combination": full(category),
    }
    ids = inputs.labels.instance_ids
    paint = instance_colors(ids)
    image = inputs.views.images / 255
    drop = plt.get_cmap("RdYlGn_r").copy()
    drop.set_bad("white")
    seam = ListedColormap(CATEGORIES)
    seam.set_bad("white")
    for crop in (False, True):
        views = PRESENTATION_VIEWS
        fig, axes = plt.subplots(
            len(views), 4, figsize=(14, 3.3 * len(views)), layout="constrained"
        )
        for r, v in enumerate(views):
            window = (slice(None), slice(None))
            outside = np.zeros(ids[v].shape, dtype=bool)
            if crop:
                labels, counts = np.unique(ids[v][ids[v] >= 0], return_counts=True)
                biggest = labels[counts.argmax()]
                inside = ids[v] == biggest
                rows, cols = (
                    np.flatnonzero(inside.any(1)),
                    np.flatnonzero(inside.any(0)),
                )
                window = (slice(rows[0], rows[-1] + 1), slice(cols[0], cols[-1] + 1))
                outside = ~inside
            for c, title in enumerate(
                [
                    "confidence: drop order v",
                    "greedy oracle: drop order v",
                    "SAM 3 masks",
                    "confidence + oracle, delta 8 cm",
                ]
            ):
                ax = axes[r, c]
                if c == 2:
                    shown = 0.5 * image[v] + 0.5 * paint[v]
                    shown[outside] = 1
                    ax.imshow(shown[window])
                else:
                    key = ["confidence", "oracle", None, "combination"][c]
                    data = np.where(outside, np.nan, maps[key][v])
                    if key == "combination":
                        ax.imshow(
                            data[window],
                            cmap=seam,
                            vmin=-0.5,
                            vmax=3.5,
                            interpolation="nearest",
                        )
                    else:
                        ax.imshow(
                            data[window],
                            cmap=drop,
                            vmin=0,
                            vmax=1,
                            interpolation="nearest",
                        )
                for key in {
                    0: ["confidence"],
                    1: ["oracle"],
                    3: ["confidence", "oracle"],
                }.get(c, []):
                    field = np.where(np.isnan(maps[key][v]), 1.0, maps[key][v])[window]
                    colour = "k" if key == "confidence" else "magenta"
                    ax.contour(field, levels=[0.5], colors=colour, linewidths=0.8)
                ax.set_xticks([])
                ax.set_yticks([])
                if r == 0:
                    ax.set_title(title, fontsize=10)
            axes[r, 0].set_ylabel(f"view {v}")
        handles = [plt.Rectangle((0, 0), 1, 1, color=col) for col in CATEGORIES]
        fig.legend(handles, NAMES, loc="lower center", ncols=4, fontsize=9)
        what = (
            "largest SAM 3 mask of each view, white outside" if crop else "full views"
        )
        fig.suptitle(
            f"{PRESENTATION_SCENE}, {what}: drop order v (kept at q iff v <= q; red first, green last); "
            "isolines q 0.5: confidence black, oracle magenta",
            x=0.01,
            ha="left",
            fontsize=10,
        )
        name = "mask" if crop else "views"
        fig.savefig(f"experiments/figures/e57c_seams_{name}.jpg", dpi=75)
        plt.close(fig)


if __name__ == "__main__":
    main()
