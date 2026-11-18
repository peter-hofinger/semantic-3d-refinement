"""E57c: Déjà View cut by oracles, and by the baselines it is compared with, at every kept fraction q.

Per pixel i of Déjà View's full cloud, aligned by one Umeyama onto the GT as the evaluation does:
- a_i, accuracy: the distance of its predicted point to the nearest GT point;
- c_i, completeness: the distance of its GT point to the nearest predicted point;
- Chamfer: (a_i + c_i) / 2;
- m_i, the completeness cost of dropping i alone: the GT points whose nearest predicted point
  is i move to their second nearest, m_i = sum (d2 - d1) over them. Dropping i lowers the accuracy
  mean by (a_i - mean a) / (K - 1) and raises the completeness mean by m_i / N, so with K ~ N it
  helps Chamfer iff a_i - mean a > m_i (first order: two neighbours covering each other both have
  m_i ~ 0, and dropping both uncovers more). Oracle `marginal`: score m_i - a_i, and its own cut.
Oracles added in the second round (rank over the same pixels, higher keeps):
- `pixel`: -e_i, the distance of the predicted point to the GT point of the same pixel, the
  error Déjà View's confidence loss is a function of (conf* = alpha / (gamma E[e_i]));
- `smooth`: -(a_i averaged over its 7 x 7 pixel window), what a head sees if it cannot resolve
  single pixels but knows the neighbourhood's errors;
- `greedy`: reverse greedy on the exact first-order Chamfer gain, in steps of 10% of the pixels:
  each round recomputes m_i and mean a on the kept set S and drops the pixels with the lowest
  m_i / N_GT - a_i / |S| (the one-shot `marginal` is its first step); `greedy-own` drops every
  pixel with positive gain each round, until none is left (<= 16 rounds), its q is its own;
- baselines, not oracles: `density`, E45's instance voxel density, and `confidence`, Déjà View's.
Each pixel has one prediction and one GT point, so all map 1:1 to pixels. Each oracle
drops the pixels of largest value (scores -a_i, -c_i, -(a_i + c_i) / 2, higher keeps), cut at
q and ranked as `scene_eval` does (`ranks`, `keep_top`), over the pixels with a GT point (all
but 0.01% on dev). c_i is read off the unfiltered cloud, not recomputed for the kept set. Each
kept set is realigned and scored as `surface_metrics` scores a scene (cm).

uv run python experiments/scripts/e57c_oracle_curve.py --splits dev --per-scene
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from scipy.ndimage import uniform_filter
from scipy.spatial import cKDTree

from src.analysis.filters import instance_voxel_density, keep_top, ranks
from src.analysis.scene_eval import SceneInputs, load_inputs
from src.evaluation.labeled import (
    LabeledCloud,
    align_output,
    nearest_distances,
    surface_scores,
)
from src.loaders.hypersim.config import load_config
from src.refinement.multiview import points_from_rays

GRID = (1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1)


def per_pixel(truth: LabeledCloud, pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return a_i and c_i (N,) of the aligned full cloud, in cm."""
    assert np.isfinite(pred).all()
    return 100 * nearest_distances(truth.points, pred), 100 * nearest_distances(
        pred, truth.points
    )


def removal_cost(
    truth: LabeledCloud, pred: np.ndarray, kept: np.ndarray | None = None
) -> np.ndarray:
    """Return m_i (N,) in cm: the sum of (d2 - d1) over the GT points whose nearest is i.

    With `kept`, the cloud is that subset and the result has its length.
    """
    pred = pred if kept is None else pred[kept]
    # sliding-midpoint splits as `nearest_distances`: far-off queries are slow otherwise
    tree = cKDTree(pred, balanced_tree=False, compact_nodes=False)
    dist, index = tree.query(truth.points, k=2, workers=-1)
    m = np.bincount(index[:, 0], weights=dist[:, 1] - dist[:, 0], minlength=len(pred))
    assert m.shape == (len(pred),)
    assert (m >= 0).all()
    return 100 * m


def greedy(
    truth: LabeledCloud, pred: np.ndarray, a: np.ndarray, levels: list[float]
) -> dict[float, np.ndarray]:
    """Reverse greedy path: the kept mask (N,) at each level (descending), dropping the lowest gain."""
    n_gt, kept = len(truth.points), np.ones(len(pred), dtype=bool)
    out = {}
    for level in levels:
        m = removal_cost(truth, pred, kept)
        score = (
            m / n_gt - a[kept] / kept.sum()
        )  # higher keeps; mean a is the same for all
        drop = np.argsort(score, kind="stable")[: kept.sum() - round(level * len(pred))]
        kept[np.flatnonzero(kept)[drop]] = False
        out[level] = kept.copy()
    return out


def greedy_own(
    truth: LabeledCloud, pred: np.ndarray, a: np.ndarray, rounds: int
) -> np.ndarray:
    """Drop every pixel whose Chamfer gain is positive, recomputed each round, until none."""
    n_gt, kept = len(truth.points), np.ones(len(pred), dtype=bool)
    for _ in range(rounds):
        m = removal_cost(truth, pred, kept)
        gain = (a[kept] - a[kept].mean()) / kept.sum() - m / n_gt
        if not (gain > 0).any():
            break
        kept[np.flatnonzero(kept)[gain > 0]] = False
    return kept


def window_mean(values: np.ndarray, pixels: np.ndarray, side: int) -> np.ndarray:
    """Mean of the GT-pixel values over each pixel's side x side window per view (N,)."""
    full = np.zeros(pixels.shape)
    full[pixels] = values
    n = np.zeros(pixels.shape)
    n[pixels] = 1
    box = lambda x: uniform_filter(x, size=(1, side, side), mode="constant")  # noqa: E731
    return (box(full) / np.maximum(box(n), 1e-9))[pixels]


def scores(
    a: np.ndarray, c: np.ndarray, m: np.ndarray, e: np.ndarray, smooth: np.ndarray
) -> dict[str, np.ndarray]:
    """Return the oracles' per-pixel keep scores, higher keeps."""
    return {
        "accuracy": -a,
        "completeness": -c,
        "chamfer": -(a + c) / 2,
        "marginal": m - a,
        "pixel": -e,
        "smooth": -smooth,
    }


def evaluate(truth: LabeledCloud, points: np.ndarray, kept: np.ndarray) -> list[float]:
    """Accuracy, completeness, Chamfer (cm) of the points that are kept (V, H, W)."""
    cut = np.where(kept[..., None], points, np.nan)
    pred, _, _ = align_output(truth, cut)
    found = np.isfinite(pred).all(1)
    s = surface_scores(
        nearest_distances(truth.points, pred[found]),
        nearest_distances(pred[found], truth.points),
    )
    return [s["accuracy_cm"], s["completeness_cm"], s["chamfer_cm"]]


ORACLES = ["accuracy", "marginal", "greedy", "greedy-own", "pixel", "smooth"]
BASELINES = ["density", "confidence"]


def rank_scores(
    inputs: SceneInputs,
) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]
]:
    """Return the points (V, H, W, 3), their alignment (4, 4), the aligned GT pixels' points, a_i, m_i and keep scores."""
    truth, pixels = inputs.truth, inputs.truth.pixels
    assert pixels is not None
    points = points_from_rays(
        inputs.scene.depth.double(), inputs.scene.rays.double()
    ).numpy()
    pred, _, align = align_output(truth, points)
    a, c = per_pixel(truth, pred)
    e = 100 * np.linalg.norm(pred - truth.points, axis=1)
    assert (e >= a - 1e-6).all(), (
        "the same-pixel GT point is no nearer than the nearest"
    )
    m = removal_cost(truth, pred)
    ranking = scores(a, c, m, e, window_mean(a, pixels, 7))
    ranking["density"] = instance_voxel_density(
        points,
        inputs.scene.depth.numpy(),
        inputs.scene.k.numpy(),
        inputs.labels.instance_ids,
    )[pixels]
    ranking["confidence"] = inputs.geometry.confidence[pixels]
    return points, align, pred, a, m, ranking


def curves(
    cfg: dict, scene: str, grid: tuple[float, ...], names: list[str], step: float
) -> dict[str, np.ndarray]:
    """Per ranking (len(grid), 3): accuracy, completeness, Chamfer (cm) of each cut of a scene."""
    inputs = load_inputs(cfg, scene, torch.device("cpu"))
    truth, pixels = inputs.truth, inputs.truth.pixels
    assert pixels is not None
    points, _, pred, a, m, ranking = rank_scores(inputs)

    def cut(kept: np.ndarray) -> list[float]:
        full = np.ones(points.shape[:3], dtype=bool)  # a pixel without GT is not scored
        full[pixels] = kept
        return evaluate(truth, points, full)

    out = {}
    for name in names:
        if name == "greedy-own":
            kept = greedy_own(truth, pred, a, 16)
            out[name] = np.array([[kept.mean(), *cut(kept)]])
        elif name == "greedy":
            levels = [round(1 - step * r, 6) for r in range(1, round(1 / step))]
            path = greedy(
                truth, pred, a, [lv for lv in levels if lv >= min(grid) - 1e-9]
            )
            out[name] = np.array(
                [
                    cut(path[round(q, 6)])
                    if q < 1
                    else cut(np.ones_like(a, dtype=bool))
                    for q in grid
                ]
            )
        else:
            out[name] = np.array([cut(keep_top(ranks(ranking[name]), q)) for q in grid])
    if "marginal" in names:
        own = m >= a - a.mean()  # dropping the others helps Chamfer to first order
        out["marginal, own cut"] = np.array([[own.mean(), *cut(own)]])
    return out


def show(grid: list[float], result: dict[str, np.ndarray]) -> None:
    """Print one block per ranking."""
    for name, rows in result.items():
        print(f"  {name}: accuracy / completeness / Chamfer (cm)")
        if rows.shape[1] == 4:  # its own cut: the kept share first
            print(
                f"    q {rows[0, 0]:.3f}: {rows[0, 1]:.3f} / {rows[0, 2]:.3f} / {rows[0, 3]:.3f}"
            )
            continue
        for q, (acc, comp, chamfer) in zip(grid, rows, strict=True):
            print(f"    q {q:.1f}: {acc:.3f} / {comp:.3f} / {chamfer:.3f}")


def table(grid: list[float], result: dict[str, np.ndarray]) -> None:
    """Print Chamfer (cm) by q, one column per ranking cut at q, then its mean over the grid."""
    cols = {k: v[:, 2] for k, v in result.items() if v.shape[1] == 3}
    print("  Chamfer (cm)      " + "".join(f"{k[:10]:>11}" for k in cols))
    for i, q in enumerate(grid):
        print(
            f"    q {q:.1f}          " + "".join(f"{v[i]:11.3f}" for v in cols.values())
        )
    print("    mean over q   " + "".join(f"{v.mean():11.3f}" for v in cols.values()))


def main() -> None:
    """Print each scene's and each split's mean curves."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", nargs="*", default=[])
    parser.add_argument("--scenes", nargs="*", default=[], help="scene folders")
    parser.add_argument("--grid", nargs="*", type=float, default=list(GRID))
    parser.add_argument(
        "--step", type=float, default=0.1, help="greedy's share dropped per round"
    )
    parser.add_argument("--per-scene", action="store_true")
    parser.add_argument("--rankings", nargs="*", default=ORACLES + BASELINES)
    args = parser.parse_args()
    cfg = yaml.safe_load(Path("experiments/configs/e57c_oracle.yaml").read_text())
    split = load_config(Path(cfg["dataset"]))
    groups = {
        n: [f"{s}_{c}" for s, c in split.scenes([n]).items()] for n in args.splits
    }
    if args.scenes:
        groups["scenes"] = args.scenes
    assert groups, "name --splits or --scenes"
    for group, scenes in groups.items():
        results = {}
        for scene in scenes:
            t0 = time.perf_counter()
            results[scene] = curves(
                cfg, scene, tuple(args.grid), args.rankings, args.step
            )
            print(f"{scene}: {time.perf_counter() - t0:.0f} s", flush=True)
            if args.per_scene:
                show(args.grid, results[scene])
        mean = {
            k: np.mean([r[k] for r in results.values()], axis=0) for k in results[scene]
        }
        print(f"{group}: mean over {len(results)} scenes")
        show(args.grid, mean)
        table(args.grid, mean)


if __name__ == "__main__":
    main()
