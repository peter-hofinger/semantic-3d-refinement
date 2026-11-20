"""E84a check: the loss against the exact Chamfer along the greedy oracle's path.

The greedy oracle's keep masks (E57c, `e57c_fast.greedy_fast`) at decreasing kept share q on
the bedroom, scored in the loss's fixed alignment: the exact Chamfer (KD-tree, no probe
limit) and the loss L at binary s (R probes, d_(R+1) charged on a miss). Shows where the
loss follows the Chamfer's U over q and where it undershoots.

`uv run python experiments/scripts/e84a_oracle_path.py --out experiments/results/e84a_oracle_path.txt`
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from experiments.scripts.e57c_fast import greedy_fast
from src.evaluation.labeled import LabeledCloud
from src.refinement.keep_chamfer import KeepScene, exact_chamfer, expected_chamfer

LEVELS = [0.9, 0.7, 0.5, 0.3, 0.2, 0.1, 0.05, 0.02, 0.01]


def main() -> None:
    """Print and save the table."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    cfg = {
        "root": "data/processed/hypersim",
        "views": "views_n20_uniform_504",
        "backbone": "dejaview_k16",
        "r": 8,
        "miss": "next",
    }
    scene = KeepScene.load("ai_006_007_cam_00", cfg, "cuda")
    truth = LabeledCloud(scene.truth, np.zeros(len(scene.truth), dtype=np.int64))
    kept = greedy_fast(truth, scene.pred, scene.probes.a.astype(np.float64), LEVELS)
    lines = [
        "greedy-oracle path, bedroom ai_006_007, R = 8, miss: next; cm, fixed alignment",
        " q    | exact Chamfer (A, C)        | loss L (A, C)             | GT points with no kept probe",
    ]
    for q in LEVELS:
        e = exact_chamfer(kept[q], scene.probes, scene.pred, scene.truth)
        loss, acc, comp = expected_chamfer(
            torch.from_numpy(kept[q]).float().cuda(), *scene.tensors, 1e-6
        )
        lines.append(
            f"{q:5.2f} | {e['chamfer_cm']:7.3f} ({e['accuracy_cm']:5.3f}, {e['completeness_cm']:6.3f})"
            f" | {loss.item():7.3f} ({acc.item():5.3f}, {comp.item():6.3f}) | {e['probe_miss']:.3f}"
        )
    args.out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
