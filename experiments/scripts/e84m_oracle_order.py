"""E84m (issue #84): the greedy Chamfer oracle's drop order as a keep value in [0, 1] per pixel.

The reverse greedy of E57c (`greedy_fast`: each round drops the pixels whose removal helps the Chamfer
most, recomputed every round) with the finest step, 1 % of the pixels per round, from the whole cloud
down to 1 % kept. A pixel dropped in the round that takes the kept share from q to q - 0.01 gets
v = 1 - q: 0 for the first round's, 0.99 for the last 1 %, which no round drops. The cloud of the top
share q is then {v >= 1 - q}, exactly the oracle's kept mask at every multiple of 1 %. Prints the
exact Chamfer of those masks (fixed alignment, the loss's) and saves v for `loss: mimic`.

`uv run python experiments/scripts/e84m_oracle_order.py --out data/processed/hypersim/ai_006_007_cam_00/e84m_oracle_v.npy`
"""

import argparse
import time
from pathlib import Path

import numpy as np

from experiments.scripts.e57c_fast import greedy_fast
from src.evaluation.labeled import LabeledCloud
from src.refinement.keep_chamfer import KeepScene, exact_chamfer

CFG = {
    "root": "data/processed/hypersim",
    "views": "views_n20_uniform_504",
    "backbone": "dejaview_k16",
    "r": 8,
    "miss": "bound",
}
SCENE = "ai_006_007_cam_00"
STEP = 0.01


def drop_order_values(kept: dict[float, np.ndarray], n: int) -> np.ndarray:
    """Return v (N,) float32: 1 - (the kept share before the round that drops the pixel)."""
    levels = sorted(kept, reverse=True)
    value = np.full(n, 1 - levels[-1], dtype=np.float32)  # never dropped: the last 1 %
    before, previous = 1.0, np.ones(n, dtype=bool)
    for level in levels:
        value[previous & ~kept[level]] = 1 - before
        before, previous = level, kept[level]
    assert value.min() == 0
    assert value.max() <= 1 - levels[-1] + 1e-6
    return value


def main() -> None:
    """Run the greedy, print its path's Chamfer and save v."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    scene = KeepScene.load(SCENE, CFG, "cpu")
    truth = LabeledCloud(scene.truth, np.zeros(len(scene.truth), dtype=np.int64))
    levels = [round(1 - STEP * k, 2) for k in range(1, round(1 / STEP))]
    t0 = time.perf_counter()
    kept = greedy_fast(truth, scene.pred, scene.probes.a.astype(np.float64), levels)
    print(
        f"greedy, {len(levels)} rounds of {STEP:.0%}: {time.perf_counter() - t0:.0f} s"
    )
    value = drop_order_values(kept, len(scene.pred))
    scores = {
        q: exact_chamfer(m, scene.probes, scene.pred, scene.truth)
        for q, m in kept.items()
    }
    for q in (0.9, 0.7, 0.5, 0.3, 0.2, 0.1):
        e = scores[q]
        print(
            f"q {q}: Chamfer {e['chamfer_cm']:.3f} (A {e['accuracy_cm']:.3f} C {e['completeness_cm']:.3f})"
        )
        assert np.array_equal(value >= 1 - q - 1e-6, kept[q])
    best = min(scores, key=lambda q: scores[q]["chamfer_cm"])
    print(f"best q {best}: Chamfer {scores[best]['chamfer_cm']:.3f}")
    np.save(args.out, value)


if __name__ == "__main__":
    main()
