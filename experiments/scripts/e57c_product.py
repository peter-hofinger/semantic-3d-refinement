"""E57c: the confidence cut, then the remaining confidence times the greedy oracle's keep order.

1. Pixels with confidence percentile c > 0.5 stay (the cut at q_c 0.5).
2. For the rest, u = c / 0.5 in [0, 1] and k in [0, 1], the position of the pixel in the
   greedy oracle's removal sequence (`greedy_order`: rounds of `STEP` of the pixels, within a round
   by the round's score, lowest first) over N; k = 1 for the pixel removed last. A pixel comes back iff
   u k >= tau. No distance, no gate: only the two rankings.
Thresholds are given as the kept share. For each: Chamfer (cm) of the cut, and of the oracle
alone at the same share (its top k, `k >= 1 - share`). The bedroom's view 18 shows u k (white =
kept by confidence), the joint distribution of (u, k) with the tau curves, and the kept maps.

uv run python -m experiments.scripts.e57c_product > experiments/results/e57c_product.txt
"""

from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from matplotlib.colors import ListedColormap

from experiments.scripts.e57c_oracle_curve import evaluate, rank_scores, removal_cost
from src.analysis.scene_eval import load_inputs
from src.analysis.scene_figures import PRESENTATION_SCENE
from src.evaluation.labeled import LabeledCloud

STEP = 0.05
FLOOR = 0.05
VIEW = 18
SHARES = (0.52, 0.55, 0.57, 0.60, 0.64, 0.68, 0.695)
FIGURE_SHARES = (0.57, 0.64, 0.68)


def greedy_order(truth: LabeledCloud, pred: np.ndarray, a: np.ndarray) -> np.ndarray:
    """Return k (N,) in [0, 1]: the pixel's position in the greedy removal sequence over N, 1 = last."""
    n, n_gt = len(pred), len(truth.points)
    kept = np.ones(n, dtype=bool)
    position = np.zeros(n)
    count = 0
    levels = list(np.arange(1 - STEP, FLOOR - 1e-9, -STEP))
    for level in levels:
        m = removal_cost(truth, pred, kept)
        score = m / n_gt - a[kept] / kept.sum()  # higher keeps
        order = np.argsort(score, kind="stable")
        drop = order[: kept.sum() - round(level * n)]
        index = np.flatnonzero(kept)[drop]
        position[index] = count + np.arange(len(index))
        count += len(index)
        kept[index] = False
    score = removal_cost(truth, pred, kept) / n_gt - a[kept] / kept.sum()
    survivors = np.flatnonzero(kept)[np.argsort(score, kind="stable")]
    position[survivors] = count + np.arange(len(survivors))
    return position / (n - 1)


def main() -> None:
    """Print the sweep and draw view 18."""
    raw = yaml.safe_load(Path("experiments/configs/e57c_oracle.yaml").read_text())
    inputs = load_inputs(raw, PRESENTATION_SCENE, torch.device("cpu"))
    truth, pixels = inputs.truth, inputs.truth.pixels
    assert pixels is not None
    points, _, pred, a, _, ranking = rank_scores(inputs)
    conf = ranking["confidence"].astype(np.float64)
    c = np.argsort(np.argsort(conf, kind="stable"), kind="stable") / (len(conf) - 1)
    k = greedy_order(truth, pred, a)
    base = c > 0.5
    value = np.where(base, np.nan, (c / 0.5) * k)

    def cut(kept: np.ndarray) -> float:
        full = np.ones(points.shape[:3], dtype=bool)  # a pixel without GT is not scored
        full[pixels] = kept
        return evaluate(truth, points, full)[2]

    print(f"confidence cut alone: kept {base.mean():.3f}, Chamfer {cut(base):.3f} cm")
    print(
        f"oracle alone at q 0.5: kept {(k >= 0.5).mean():.3f}, Chamfer {cut(k >= 0.5):.3f} cm"
    )
    print(
        "top fraction of the confidence-dropped pixels by u k (the quantile), kept share, tau, Chamfer of the product cut and of the oracle alone at the same share"
    )
    taus = {}
    drop_count = (~base).sum()
    for share in SHARES:
        dropped = np.sort(value[~base])[::-1]
        take = min(round((share - base.mean()) * len(c)), len(dropped))
        tau = dropped[take - 1]
        taus[share] = tau
        kept = base | (~base & (value >= tau))
        alone = k >= 1 - kept.mean()
        print(
            f"top {(kept.sum() - base.sum()) / drop_count:.3f}  kept {kept.mean():.3f}  tau {tau:.4f}  product {cut(kept):.3f}  oracle alone {cut(alone):.3f}"
        )

    # figure
    def full(values: np.ndarray) -> np.ndarray:
        out = np.full(pixels.shape, np.nan)
        out[pixels] = values
        return out

    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("white")
    fig, axes = plt.subplots(2, 3, figsize=(16, 10.5), layout="constrained")
    shown = axes[0, 0].imshow(
        full(value)[VIEW], cmap=cmap, vmin=0, vmax=1, interpolation="nearest"
    )
    axes[0, 0].set_title(
        "u k = (c / 0.5) x oracle keep order k; white: kept by confidence (c > 0.5)",
        fontsize=9,
    )
    fig.colorbar(shown, ax=axes[0, 0], shrink=0.7)
    axes[0, 2].imshow(
        full(np.where(base, np.nan, k))[VIEW],
        cmap=cmap,
        vmin=0,
        vmax=1,
        interpolation="nearest",
    )
    axes[0, 2].set_title(
        "k: the oracle's keep order of the confidence-dropped pixels (1 = removed last)",
        fontsize=9,
    )
    drop = ~base
    ax = axes[0, 1]
    ax.hist2d(
        c[drop] / 0.5, k[drop], bins=50, cmap="magma_r", norm=mpl.colors.LogNorm()
    )
    x = np.linspace(0.01, 1, 200)
    for share, tau in taus.items():
        if share in FIGURE_SHARES:
            ax.plot(x, np.minimum(tau / x, 1), label=f"u k = {tau:.3f} (kept {share})")
    ax.set_xlabel("u = c / 0.5 (remaining confidence)")
    ax.set_ylabel("k (oracle keep order)")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)
    ax.set_title(
        "confidence-dropped pixels, bedroom, all views (counts, log) and the tau curves",
        fontsize=9,
    )
    palette = ListedColormap(["white", "tab:green", "tab:red"])
    for ax, share in zip(axes[1], FIGURE_SHARES, strict=True):
        kept = base | (~base & (value >= taus[share]))
        cat = np.where(base, 0, np.where(kept, 1, 2))
        ax.imshow(
            full(cat)[VIEW].astype(float),
            cmap=palette,
            vmin=-0.5,
            vmax=2.5,
            interpolation="nearest",
        )
        ax.set_title(
            f"kept {share}: white confidence, green reintroduced (u k >= {taus[share]:.3f}), red dropped",
            fontsize=9,
        )
    for ax in axes.ravel():
        if ax is not axes[0, 1]:
            ax.set_xticks([])
            ax.set_yticks([])
    fig.suptitle(
        f"{PRESENTATION_SCENE}, view {VIEW}: the confidence cut, then confidence x oracle order",
        fontsize=10,
    )
    fig.savefig("experiments/figures/e57c_product_view18.jpg", dpi=80)
    plt.close(fig)


if __name__ == "__main__":
    main()
