"""E84d diagnostic: the exact Chamfer of the top-q share of s, at saved steps of a run.

The head is trained on Bernoulli(s) clouds; a deterministic cut keeps the q highest s.
On the training scene (bedroom), so this only shows what the head's ordering is worth,
no cut is chosen from it.

`uv run python experiments/scripts/e84d_cut_sweep.py --run runs/<e84d run> --out experiments/results/e84d_cut_sweep.txt`
"""

import argparse
from pathlib import Path

import numpy as np

from src.refinement.keep_chamfer import KeepScene, exact_chamfer

SCENE = "ai_006_007_cam_00"
CFG = {
    "root": "data/processed/hypersim",
    "views": "views_n20_uniform_504",
    "backbone": "dejaview_k16",
    "r": 8,
    "miss": "bound",
}


def main() -> None:
    """Print and save the sweep."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    scene = KeepScene.load(SCENE, CFG, "cuda")
    valid = scene.valid.cpu().numpy()
    lines = [
        f"bedroom {SCENE}, exact Chamfer cm (accuracy / completeness), uncut 1.846"
    ]
    for step in (0, 100, 500, 1000):
        s = np.load(args.run / f"probabilities_{SCENE}_step{step}.npy")[valid]
        s = s.astype(np.float32)
        q = np.quantile(s, [0.01, 0.1, 0.5, 0.9, 0.99])
        lines.append(f"step {step}: s quantiles 1/10/50/90/99 % {np.round(q, 3)}")
        for share in (0.9, 0.7, 0.5, 0.3, 0.2):
            kept = s >= np.quantile(s, 1 - share)
            e = exact_chamfer(kept, scene.probes, scene.pred, scene.truth)
            lines.append(
                f"  top {share:.1f}: {e['chamfer_cm']:.3f} ({e['accuracy_cm']:.3f} / "
                f"{e['completeness_cm']:.3f}), probe_miss {e['probe_miss']:.3f}"
            )
    args.out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
