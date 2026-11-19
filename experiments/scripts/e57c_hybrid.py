"""E57c: Déjà View's confidence in the greedy oracle, two ways, on dev.

A. `greedy` with the accuracy estimated from the confidence: the reverse greedy of
`e57c_oracle_curve` (10% of the pixels per round, true m_i from the GT) with a_i replaced by
kappa / conf_i, the loss optimum conf* = alpha / (gamma E[error]) read as an error estimate.
kappa is one scalar per scene, set so that the mean of kappa / conf equals the scene's true
mean a_i (a GT statistic, one number per scene); `x0.5` and `x2` scale it. Only the accuracy
term is estimated; m_i still uses the GT.

B. Confidence first, then the oracle's points back: cut the pixels below Déjà View's
confidence at the kept fraction q_c, then add each pixel the greedy oracle keeps at q_g that
lies farther than delta (cm, in the GT's frame) from every pixel kept so far; delta = 0 is
the union, delta = inf the confidence cut alone. Control `fill`: add every pixel farther
than delta, oracle or not.

All cuts are realigned and scored as `surface_metrics` does (cm), over the pixels with a GT
point, as in `e57c_oracle_curve`.

uv run python -m experiments.scripts.e57c_hybrid --splits dev
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from scipy.spatial import cKDTree

from experiments.scripts.e57c_oracle_curve import (
    GRID,
    evaluate,
    greedy,
    rank_scores,
)
from src.analysis.filters import keep_top, ranks
from src.analysis.scene_eval import load_inputs
from src.loaders.hypersim.config import load_config

QC = (0.5, 0.7, 0.9)  # confidence's kept fraction
QG = 0.5  # the greedy oracle's kept fraction (its best on dev)
DELTAS = (0.0, 1.0, 2.0, 4.0, 8.0, np.inf)  # cm
SCALES = (1.0, 0.5, 2.0)  # kappa multipliers


def far_from(pred: np.ndarray, kept: np.ndarray, delta: float) -> np.ndarray:
    """Mask (N,) of the pixels not in `kept` that lie farther than delta (cm) from every kept one."""
    tree = cKDTree(pred[kept], balanced_tree=False, compact_nodes=False)
    far = np.zeros(len(pred), dtype=bool)
    rest = np.flatnonzero(~kept)
    far[rest] = 100 * tree.query(pred[rest], workers=-1)[0] > delta
    return far


def scene_results(
    cfg: dict, scene: str
) -> tuple[dict[str, np.ndarray], dict[tuple[float, float, str], np.ndarray]]:
    """Return A's curves per variant and B's (share, accuracy, completeness, Chamfer) per cut."""
    inputs = load_inputs(cfg, scene, torch.device("cpu"))
    truth, pixels = inputs.truth, inputs.truth.pixels
    assert pixels is not None
    points, _, pred, a, _, ranking = rank_scores(inputs)

    def cut(kept: np.ndarray) -> list[float]:
        full = np.ones(points.shape[:3], dtype=bool)  # a pixel without GT is not scored
        full[pixels] = kept
        return evaluate(truth, points, full)

    levels = [q for q in GRID if q < 1]
    conf = ranking["confidence"].astype(np.float64)
    assert (conf >= 1).all()
    kappa = a.mean() / (1 / conf).mean()
    curves = {}
    for name, estimate in [("true a", a)] + [
        (f"alpha/conf x{s:g}", s * kappa / conf) for s in SCALES
    ]:
        path = greedy(truth, pred, estimate, levels)
        curves[name] = np.array([cut(path[q]) for q in levels])
    oracle = greedy(truth, pred, a, [0.9, 0.8, 0.7, 0.6, QG])[QG]
    hybrid = {}
    for qc in QC:
        base = keep_top(ranks(conf), qc)
        for delta in DELTAS:
            for kind, source in (("oracle", oracle), ("fill", np.ones_like(oracle))):
                if kind == "fill" and delta in (0.0, np.inf):
                    continue  # the uncut cloud and the confidence cut alone
                added = far_from(pred, base, delta) & source
                kept = base | added
                hybrid[(qc, delta, kind)] = np.array([kept.mean(), *cut(kept)])
    return curves, hybrid


def main() -> None:
    """Print A's and B's means over the split's scenes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", nargs="*", default=["dev"])
    args = parser.parse_args()
    cfg = yaml.safe_load(Path("experiments/configs/e57c_oracle.yaml").read_text())
    split = load_config(Path(cfg["dataset"]))
    for name in args.splits:
        scenes = [f"{s}_{c}" for s, c in split.scenes([name]).items()]
        runs = []
        for scene in scenes:
            t0 = time.perf_counter()
            runs.append(scene_results(cfg, scene))
            print(f"{scene}: {time.perf_counter() - t0:.0f} s", flush=True)
        print(f"{name}: mean over {len(runs)} scenes")
        print(
            "A. greedy with the accuracy from confidence: Chamfer (accuracy / completeness) cm"
        )
        levels = [q for q in GRID if q < 1]
        for variant in runs[0][0]:
            mean = np.mean([r[0][variant] for r in runs], axis=0)
            row = "  ".join(
                f"{q:.1f}: {c:.3f} ({a:.2f}/{m:.2f})"
                for q, (a, m, c) in zip(levels, mean, strict=True)
            )
            print(f"  {variant:>16}  {row}")
        print(
            "B. confidence q_c, then oracle (greedy q_g 0.5) or any pixel farther than delta cm"
        )
        print("  kept share, accuracy / completeness / Chamfer (cm)")
        for key in runs[0][1]:
            share, acc, comp, ch = np.mean([r[1][key] for r in runs], axis=0)
            qc, delta, kind = key
            print(
                f"  q_c {qc:.1f}  delta {delta:>3g}  {kind:>6}: {share:.3f}  {acc:.3f} / {comp:.3f} / {ch:.3f}"
            )
        for key in runs[0][1]:
            if key[1] == 0 and key[2] == "oracle":
                per = [f"{r[1][key][3]:.2f}" for r in runs]
                print(f"  per scene Chamfer, q_c {key[0]:.1f} union: {' / '.join(per)}")


if __name__ == "__main__":
    main()
