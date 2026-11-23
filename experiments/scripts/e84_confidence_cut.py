"""E84 baseline: Déjà View's confidence and the keep head's s as rankings, same shares, same alignment.

For each kept share q the top-q pixels by Déjà View's confidence (higher = kept) and by the
head's s are scored by `exact_chamfer` in the loss's fixed alignment, so the two rankings and
the uncut cloud (q = 1) are comparable. Bedroom, the training scene: a diagnostic, not a
result; no cut is chosen from it.

`uv run python experiments/scripts/e84_confidence_cut.py --run runs/<e84 run> --out experiments/results/e84_confidence_cut.txt`
"""

import argparse
from pathlib import Path

import numpy as np

from src.geometry import Geometry
from src.refinement.keep_chamfer import KeepScene, exact_chamfer

SCENE = "ai_006_007_cam_00"
CFG = {
    "root": "data/processed/hypersim",
    "views": "views_n20_uniform_504",
    "backbone": "dejaview_k16",
    "r": 8,
    "miss": "bound",
}
SHARES = (1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3)


def main() -> None:
    """Print and save the table."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    scene = KeepScene.load(SCENE, CFG, "cuda")
    valid = scene.valid.cpu().numpy()
    folder = Path(CFG["root"]) / SCENE
    conf = Geometry.load(folder / f"{CFG['backbone']}_{CFG['views']}.npz").confidence
    s = np.load(args.run / f"probabilities_{SCENE}.npy")
    assert conf.shape == s.shape == valid.shape, (conf.shape, s.shape, valid.shape)
    scores = {
        "Déjà View confidence": conf[valid],
        f"head s ({args.run.name})": s[valid],
    }
    lines = [
        f"bedroom {SCENE}, exact Chamfer cm (accuracy / completeness), fixed alignment"
    ]
    for name, score in scores.items():
        lines.append(name)
        for q in SHARES:
            kept = (
                score >= np.quantile(score, 1 - q)
                if q < 1
                else np.ones(len(score), bool)
            )
            e = exact_chamfer(kept, scene.probes, scene.pred, scene.truth)
            lines.append(
                f"  top {q:.1f}: {e['chamfer_cm']:.3f} ({e['accuracy_cm']:.3f} / "
                f"{e['completeness_cm']:.3f}), probe_miss {e['probe_miss']:.3f}"
            )
    args.out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
