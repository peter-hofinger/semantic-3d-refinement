"""E84 diagnostic: what Déjà View's confidence ranks, against what the Chamfer's accuracy needs.

Déjà View's confidence loss `c * e - alpha * log c` has the optimum c = alpha / e for the
same-pixel error e (here the 3D distance |p_i - t_i| to the GT point of the same pixel, cm,
aligned as in the loss); the Chamfer's accuracy term needs a_i, the distance to the nearest
GT point of any pixel. Reports how often the two are the same GT point, how far apart they
are, and the rank correlation of Déjà View's confidence with each. Bedroom.

`uv run python experiments/scripts/e84_confidence_target.py --out experiments/results/e84_confidence_target.txt`
"""

import argparse
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import spearmanr

from src.geometry import Geometry
from src.refinement.keep_chamfer import CM, KeepScene

SCENE = "ai_006_007_cam_00"
CFG = {
    "root": "data/processed/hypersim",
    "views": "views_n20_uniform_504",
    "backbone": "dejaview_k16",
    "r": 8,
    "miss": "bound",
}


def main() -> None:
    """Print and save the numbers."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    scene = KeepScene.load(SCENE, CFG, "cpu")
    folder = Path(CFG["root"]) / SCENE
    geometry = Geometry.load(folder / f"{CFG['backbone']}_{CFG['views']}.npz")
    conf = geometry.confidence[scene.valid.numpy()]
    same = CM * np.linalg.norm(scene.pred - scene.truth, axis=1)  # same-pixel error e_i
    dist, nearest = cKDTree(scene.truth).query(scene.pred, workers=-1)
    a = CM * dist  # a_i, nearest GT point of any pixel
    rng = np.random.default_rng(0)
    pick = rng.choice(len(a), 300_000, replace=False)

    def rho(x: np.ndarray, y: np.ndarray) -> float:
        return float(spearmanr(x[pick], y[pick])[0])

    own = (nearest == np.arange(len(a))).mean()
    ratio = np.median(a / np.maximum(same, 1e-9))
    half = (a < same / 2).mean()
    r_e, r_a, r_ae = rho(conf, -same), rho(conf, -a), rho(a, same)
    lines = [
        f"bedroom {SCENE}, {len(a)} pixels with GT, cm after the loss's alignment",
        f"nearest GT point of prediction i is its own pixel's: {own:.3f}",
        f"a_i / e_i median {ratio:.3f}; share with a_i < e_i / 2: {half:.3f}",
        f"mean a_i {a.mean():.3f}, mean e_i {same.mean():.3f}",
        f"Spearman: (confidence, -e_i) {r_e:.3f}, (confidence, -a_i) {r_a:.3f}",
        f"Spearman: (a_i, e_i) {r_ae:.3f}",
    ]
    args.out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
