"""E84d diagnostic: E84d's loss at the greedy oracle's keep masks, hard and softened.

At s = eps + (1 - 2 eps) mask the loss (Poisson-tail charge, bound probes) is compared with
E84d's trained value (1.485) and the exact Chamfer of the mask itself, which is also the
cloud a cut at s >= 0.5 gives. Shows where the loss's own minimum lies and what a binary s
costs it (kappa = 0 charges the cap). Bedroom, the training scene.

`uv run python experiments/scripts/e84d_oracle_soft.py --out experiments/results/e84d_oracle_soft.txt`
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from experiments.scripts.e57c_fast import greedy_fast
from src.evaluation.labeled import LabeledCloud
from src.refinement.keep_chamfer import KeepScene, exact_chamfer, expected_chamfer

LEVELS = [0.7, 0.5, 0.2]
CFG = {
    "root": "data/processed/hypersim",
    "views": "views_n20_uniform_504",
    "backbone": "dejaview_k16",
    "r": 8,
    "miss": "bound",
}


def main() -> None:
    """Print and save the table."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    scene = KeepScene.load("ai_006_007_cam_00", CFG, "cuda")
    truth = LabeledCloud(scene.truth, np.zeros(len(scene.truth), dtype=np.int64))
    kept = greedy_fast(truth, scene.pred, scene.probes.a.astype(np.float64), LEVELS)
    lines = [
        "bedroom ai_006_007, R = 8, cm; E84d trained s: loss 1.485, exact (Bernoulli) 1.467"
    ]
    for q in LEVELS:
        e = exact_chamfer(kept[q], scene.probes, scene.pred, scene.truth)
        lines.append(
            f"oracle q = {q}: exact Chamfer {e['chamfer_cm']:.3f}, probe_miss {e['probe_miss']:.3f}"
        )
        mask = torch.from_numpy(kept[q]).float().cuda()
        for eps in (0.0, 0.02, 0.1, 0.3):
            s = eps + (1 - 2 * eps) * mask
            tail = expected_chamfer(s, *scene.tensors, 1e-6, tail=True)
            cap = expected_chamfer(s, *scene.tensors, 1e-6)[0]
            lines.append(
                f"  s = {eps} + {1 - 2 * eps:.2f} mask: tail loss {tail[0].item():7.3f} "
                f"(A {tail[1].item():.3f}, C {tail[2].item():.3f}), cap-only loss {cap.item():6.1f}"
            )
    args.out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
