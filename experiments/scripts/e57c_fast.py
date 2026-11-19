"""E57c: faster, exactly equivalent versions of the slow steps of the oracle fusion.

`greedy_fast`: the reverse greedy of `e57c_oracle_curve.greedy` with an incremental nearest / second nearest per GT point. Dropping pixels changes the top two of
a GT point only when one of them is dropped, so only those GT points are queried again (about 1 - kept^2 of them, 19% for the first 10% round instead of all of them);
the cost of a pixel, m_i = sum (d2 - d1) over the GT points whose nearest it is, is a bincount over all of them. The kept masks equal the reference's (checked by the callers
on the GT of every scene they run).
"""

import numpy as np
from scipy.spatial import cKDTree

from src.evaluation.labeled import LabeledCloud


def greedy_fast(
    truth: LabeledCloud, pred: np.ndarray, a: np.ndarray, levels: list[float]
) -> dict[float, np.ndarray]:
    """Reverse greedy path (kept mask (N,) per level, descending), identical to `e57c_oracle_curve.greedy`."""
    n, n_gt = len(pred), len(truth.points)
    kept = np.ones(n, dtype=bool)
    near = np.zeros((n_gt, 2), dtype=np.int64)
    dist = np.zeros((n_gt, 2))
    affected = np.arange(n_gt)
    out = {}
    for level in levels:
        index = np.flatnonzero(kept)
        tree = cKDTree(pred[index], balanced_tree=False, compact_nodes=False)
        d, i = tree.query(truth.points[affected], k=2, workers=-1)
        dist[affected], near[affected] = d, index[i]
        m = 100 * np.bincount(near[:, 0], weights=dist[:, 1] - dist[:, 0], minlength=n)
        score = m[kept] / n_gt - a[kept] / kept.sum()  # higher keeps
        drop = np.argsort(score, kind="stable")[: kept.sum() - round(level * n)]
        kept[index[drop]] = False
        out[level] = kept.copy()
        affected = np.flatnonzero(~kept[near[:, 0]] | ~kept[near[:, 1]])
    return out
