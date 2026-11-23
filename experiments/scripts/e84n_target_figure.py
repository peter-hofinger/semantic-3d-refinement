"""E84n (issue #84): look at the oracle's keep value v and at what trained heads make of it.

Rows, views 18, 0, 12, 6 of the bedroom: the image; the target v (0 = the first 1 % the greedy drops,
0.99 = the last 1 %); then for each `--run` the head's logit as a rank in [0, 1] (the same scale as v)
and the rank difference, head minus target. Grey = no GT. Prints each Spearman(x, v).

`uv run python experiments/scripts/e84n_target_figure.py --run runs/a --run runs/b --out experiments/figures/e84n_mimic_target.jpg`
"""

import argparse
import copy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import rankdata, spearmanr

from experiments.scripts.e84_dejaview_optimum_filter import VIEWS
from experiments.scripts.e84m_oracle_order import CFG, SCENE
from experiments.scripts.e84m_sweep import load_net, logits
from src.loaders.views import Views
from src.refinement.keep_chamfer import KeepScene, standardize


def to_images(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Return (V, H, W) with `values` (N,) at the valid pixels and NaN elsewhere."""
    full = np.full(valid.shape, np.nan)
    full[valid] = values
    return full


def main() -> None:
    """Print the rank agreements and save the figure."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    folder = Path(CFG["root"]) / SCENE
    raw_scene = KeepScene.load(SCENE, CFG, "cuda")
    views = Views.load(folder / f"{CFG['views']}.npz")
    target = np.load(folder / "e84m_oracle_v.npy")
    valid = raw_scene.valid.cpu().numpy()
    rows = {
        "oracle value v\n(0 = dropped first)": (
            to_images(target, valid),
            "viridis",
            0,
            1,
        )
    }
    for run in args.run:
        net, config = load_net(run)
        scene = copy.copy(raw_scene)
        if config.get("standardize"):
            standardize([scene])
        x = logits(net, scene)
        rank = (rankdata(x) - 1) / (len(x) - 1)
        print(f"{run.name}: Spearman(x, v) {spearmanr(x[::13], target[::13])[0]:.3f}")
        label = run.name.rsplit("-", 2)[0]
        rows[f"{label}\nlogit as a rank"] = (to_images(rank, valid), "viridis", 0, 1)
        rows[f"{label}\nrank - v"] = (
            to_images(rank - target, valid),
            "coolwarm",
            -1,
            1,
        )
    fig, ax = plt.subplots(
        1 + len(rows), len(VIEWS), figsize=(3.6 * len(VIEWS), 3.0 * (1 + len(rows)))
    )
    for col, v in enumerate(VIEWS):
        ax[0, col].imshow(views.images[v])
        ax[0, col].set_title(f"view {v}")
    ax[0, 0].set_ylabel("image")
    for r, (label, (img, cmap, lo, hi)) in enumerate(rows.items(), start=1):
        for col, v in enumerate(VIEWS):
            shown = ax[r, col].imshow(img[v], cmap=cmap, vmin=lo, vmax=hi)
        ax[r, 0].set_ylabel(label, fontsize=7)
        fig.colorbar(shown, ax=ax[r, :], fraction=0.015, pad=0.01)
    for a in ax.ravel():
        a.set_xticks([])
        a.set_yticks([])
    fig.suptitle(
        f"bedroom {SCENE}: the greedy oracle's keep value and trained heads' outputs; "
        "look at where the head's rank departs from v",
        fontsize=9,
    )
    fig.savefig(args.out, dpi=100, bbox_inches="tight")


if __name__ == "__main__":
    main()
