"""E37: how much of the per-pixel anchor Déjà View's confidence sets.

Inside Huber's threshold the anchor's stiffness on δ (the second derivative of the
per-pixel anchor terms of `multiview.residuals`) is λ_d w / 2 from E_d plus λ_δ from
‖δ‖², w = c^p u, c the confidence divided by its view's mean. With every weight 1 and
u <= 1 it is at most 1 + c / 2 for p = 1 and 1.5 for p = 0: λ_δ, which no confidence
scales, sets two thirds or more of it. Prints, per scene with a cached Déjà View output,
the mean c in each confidence quartile (per scene, as E35's split) and the stiffness
bound there for p = 1 and p = 0 (u = 1).

uv run python experiments/scripts/e37_anchor.py > experiments/results/e37_anchor.txt
"""

from pathlib import Path

import numpy as np

from src.geometry import Geometry

ROOTS = (Path("data/processed/hypersim"), Path("data/processed/eth3d"))
CACHE = "dejaview_k16_views_n20_uniform_504.npz"


def main() -> None:
    """Print the table of the module docstring."""
    print(
        "scene | mean c per confidence quartile Q1..Q4 | stiffness bound Q1, p=1 / p=0"
    )
    rows = []
    for root in ROOTS:
        for path in sorted(root.glob(f"*/{CACHE}")):
            conf = Geometry.load(path).confidence.astype(np.float64)
            assert np.isfinite(conf).all(), path
            assert (conf >= 0).all(), path
            c = conf / conf.mean(axis=(1, 2), keepdims=True)
            edges = np.quantile(c, [0.25, 0.5, 0.75])
            q = np.digitize(c, edges)
            means = [float(c[q == k].mean()) for k in range(4)]
            rows.append(means)
            cells = " ".join(f"{m:.2f}" for m in means)
            print(f"{path.parent.name} | {cells} | {1 + means[0] / 2:.2f} / 1.50")
    mean = np.mean(rows, axis=0)
    print(
        f"mean of {len(rows)} scenes | "
        + " ".join(f"{m:.2f}" for m in mean)
        + f" | {1 + mean[0] / 2:.2f} / 1.50"
    )


if __name__ == "__main__":
    main()
