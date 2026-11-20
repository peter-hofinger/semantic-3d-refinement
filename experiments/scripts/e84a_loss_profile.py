"""E84a diagnosis: where the expected-Chamfer loss leaves the Chamfer it stands for.

For a uniform keep probability s = c on the bedroom: the loss L(c) and its A, C; the exact
Chamfer of Bernoulli(c) draws (mean of `--draws`, same alignment); the same for the
thresholded cloud of a random 50 % at c = 0.5 is not defined, so only draws are given.
Then the empty cloud: L(0) = mean(d_{R+1}) / 2 for R = 8 ... 64, from a KD-tree query on a
random subset of GT points (the head's own R is 8).

`uv run python experiments/scripts/e84a_loss_profile.py --out experiments/results/e84a_loss_profile.txt`
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree

from src.refinement.keep_chamfer import CM, KeepScene, exact_chamfer, expected_chamfer

SCENE = "ai_006_007_cam_00"
LEVELS = (1.0, 0.9, 0.7, 0.5, 0.3, 0.2, 0.1, 0.05, 0.02, 0.01)


def main() -> None:
    """Print and save the profile."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--r", type=int, default=8)
    parser.add_argument("--draws", type=int, default=3)
    args = parser.parse_args()
    cfg = {
        "root": "data/processed/hypersim",
        "views": "views_n20_uniform_504",
        "backbone": "dejaview_k16",
        "r": args.r,
    }
    scene = KeepScene.load(SCENE, cfg, "cuda")
    n = len(scene.pred)
    lines = [
        f"bedroom {SCENE}, N = {n}, R = {args.r}; cm; uniform keep probability c",
        "",
    ]
    lines.append(
        "   c | L (A, C) of the loss          | exact Bernoulli(c): Chamfer (A, C), mean of draws"
    )
    for c in LEVELS:
        loss, acc, comp = expected_chamfer(
            torch.full((n,), c, device="cuda"), *scene.tensors, 1e-6
        )
        rng = np.random.default_rng(0)
        draws = [
            exact_chamfer(rng.random(n) < c, scene.probes, scene.pred, scene.truth)
            for _ in range(args.draws)
        ]
        mean = {k: np.mean([d[k] for d in draws]) for k in draws[0]}
        lines.append(
            f"{c:4.2f} | {loss.item():6.3f} ({acc.item():5.3f}, {comp.item():6.3f}) | "
            f"{mean['chamfer_cm']:7.3f} ({mean['accuracy_cm']:5.3f}, {mean['completeness_cm']:7.3f})"
            f"  probe_miss {mean['probe_miss']:.3f}"
        )
    lines += [
        "",
        "empty cloud: L(0) = (0 + mean d_{j,R+1}) / 2, A -> 0 as sum s -> 0, 20000 random GT points",
    ]
    sample = np.random.default_rng(0).choice(n, 20000, replace=False)
    tree = cKDTree(scene.pred, balanced_tree=False, compact_nodes=False)
    dist = CM * tree.query(scene.truth[sample], 65, workers=-1)[0]
    for r in (8, 16, 32, 64):
        lines.append(
            f"R = {r:2d}: mean d_(R+1) = {dist[:, r].mean():6.3f} cm, L(0) = {dist[:, r].mean() / 2:6.3f} cm"
        )
    lines.append(
        f"uncut Chamfer {exact_chamfer(np.ones(n, bool), scene.probes, scene.pred, scene.truth)['chamfer_cm']:.3f} cm"
    )
    args.out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
