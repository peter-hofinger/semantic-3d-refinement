"""E57c: two gradients and a quantile cut of the undecided pixels, lean.

- g_c = min(c / 0.5, 1), c the confidence percentile: 0 at c = 0, 1 at c >= 0.5;
- g_o: for a pixel the greedy oracle keeps (q_g 0.5, 10% rounds), clip(d / 10 cm, 0, 1), d its distance to
  the nearest confidence-kept pixel; 0 for the pixels the oracle drops.
Decided and kept, not counted to the quantile: c >= 0.5 and oracle-kept pixels with d > 10 cm. The undecided rest
is ranked by p = g_c g_o (ties by `filters.ranks`' seeded random order) and its top fraction q is kept,
q = 0, 0.1, ..., 1.0. Chamfer (cm) of the bedroom as `evaluate` scores it (one Umeyama per cut), with the oracle
alone at the same kept share (the 5% rounds' order k of `e57c_product.greedy_order`) as the floor.

Lean: the cached inputs only (no SAM 3, no density, no normals); the per-scene arrays (points, a_i, v, k, c, d) are
cached in `runs/e57c/gradients_<scene>.npz` (delete it to recompute); stage times are printed. Figure: view 18's p
with the thresholds of the quantiles as isolines, the heatmap of the quantile with its isolines, Chamfer against q.

uv run python -m experiments.scripts.e57c_gradients [--combine mean] [--floor] > experiments/results/e57c_gradients_v2.txt
"""

import argparse
import time
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from scipy.spatial import cKDTree
from scipy.stats import rankdata

from experiments.scripts.e57c_oracle_curve import evaluate, greedy
from experiments.scripts.e57c_product import greedy_order
from experiments.scripts.e57c_seams import LEVELS
from src.analysis.filters import keep_top, ranks
from src.analysis.scene_figures import PRESENTATION_SCENE
from src.evaluation.labeled import align_output, ground_truth_cloud, nearest_distances
from src.geometry import Geometry
from src.loaders.ground_truth import GroundTruth
from src.loaders.views import Views
from src.refinement.multiview import points_from_rays, scene_from_geometry

VIEW = 18
FAR = 10.0  # cm, g_o = 1 from here
QS = tuple(round(0.1 * i, 1) for i in range(11))
CACHE = Path("runs/e57c")


def distance_to_kept(pred: np.ndarray, kept: np.ndarray) -> np.ndarray:
    """Distance (cm) of each pixel to the nearest kept one; 0 for the kept."""
    tree = cKDTree(pred[kept], balanced_tree=False, compact_nodes=False)
    out = np.zeros(len(pred))
    out[~kept] = 100 * tree.query(pred[~kept], workers=-1)[0]
    return out


def main() -> None:
    """Print the q sweep with stage times and draw the figure."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--floor",
        action="store_true",
        help="also score the oracle alone at each kept share (doubles the time)",
    )
    parser.add_argument("--combine", choices=["product", "mean"], default="product")
    args = parser.parse_args()
    t0 = time.perf_counter()
    raw = yaml.safe_load(Path("experiments/configs/e57c_oracle.yaml").read_text())
    folder = Path(raw["root"]) / PRESENTATION_SCENE
    v = Views.load(folder / f"{raw['views']}.npz")
    gt = GroundTruth.load(folder / f"gt_{raw['views']}.npz")
    truth = ground_truth_cloud(
        gt.depth, gt.instance_ids, gt.classes, v.intrinsics, v.cam_to_world
    )
    pixels = truth.pixels
    assert pixels is not None
    geometry = Geometry.load(folder / f"dejaview_k16_{raw['views']}.npz")
    scene = scene_from_geometry(geometry)
    points = points_from_rays(scene.depth.double(), scene.rays.double()).numpy()
    print(f"load: {time.perf_counter() - t0:.1f} s", flush=True)
    t0 = time.perf_counter()
    cache = CACHE / f"gradients_{PRESENTATION_SCENE}.npz"
    cached = cache.exists()
    if cached:
        stored = np.load(cache)
        a, v_oracle, k, c = (stored[n] for n in ("a", "v_oracle", "k", "c"))
        assert len(a) == len(truth.points)
    else:
        pred, _, _ = align_output(truth, points)
        a = 100 * nearest_distances(truth.points, pred)
        path = greedy(truth, pred, a, LEVELS)
        v_oracle = np.full(len(a), 0.05)
        for level in LEVELS:  # descending: the last level a pixel is kept at wins
            v_oracle[path[level]] = level
        v_oracle[~path[LEVELS[0]]] = 1.0
        k = greedy_order(truth, pred, a)
        conf = geometry.confidence[pixels].astype(np.float64)
        c = np.argsort(np.argsort(conf, kind="stable"), kind="stable") / (len(conf) - 1)
        np.savez(cache, a=a, v_oracle=v_oracle, k=k, c=c)
    pred, _, _ = align_output(truth, points)
    print(
        f"arrays ({'cached' if cached else 'computed'}): {time.perf_counter() - t0:.1f} s",
        flush=True,
    )
    t0 = time.perf_counter()
    oracle = v_oracle <= 0.5
    base = c >= 0.5
    d = distance_to_kept(pred, base)
    g_c = np.minimum(c / 0.5, 1.0)
    g_o = np.where(oracle, np.clip(d / FAR, 0, 1), 0.0)
    decided = base | (oracle & (d > FAR))
    pool = ~decided
    p = g_c * g_o if args.combine == "product" else (g_c + g_o) / 2
    formula = "g_c g_o" if args.combine == "product" else "(g_c + g_o) / 2"
    print(f"gradients: {time.perf_counter() - t0:.1f} s", flush=True)
    print(
        f"decided and kept {decided.mean():.3f} (confidence {base.mean():.3f}, oracle farther than {FAR:g} cm {(decided & ~base).mean():.3f}); undecided {pool.mean():.3f}, "
        f"of which p = 0: {(p[pool] == 0).mean():.3f} (oracle-dropped {(~oracle[pool]).mean():.3f})"
    )
    t0 = time.perf_counter()

    def score(kept: np.ndarray) -> list[float]:
        full = np.ones(points.shape[:3], dtype=bool)  # a pixel without GT is not scored
        full[pixels] = kept
        return evaluate(truth, points, full)

    rank = ranks(p[pool])
    results = []
    print(
        "q     kept   accuracy / completeness / Chamfer   oracle alone at the same share"
    )
    for q in QS:
        kept = decided.copy()
        kept[np.flatnonzero(pool)[keep_top(rank, q)]] = True
        acc, comp, chamfer = score(kept)
        alone = score(k >= 1 - kept.mean())[2] if args.floor else float("nan")
        results.append((q, kept.mean(), chamfer, alone))
        print(
            f"{q:.1f}  {kept.mean():.3f}  {acc:.3f} / {comp:.3f} / {chamfer:.3f}   {alone:.3f}",
            flush=True,
        )
    print(
        f"sweep ({len(QS) * (1 + args.floor)} Chamfers): {time.perf_counter() - t0:.1f} s"
    )

    # figure: thresholds of the quantiles as isolines
    tau = {
        q: np.sort(p[pool])[::-1][max(round(q * pool.sum()) - 1, 0)] for q in QS[1:-1]
    }
    top = (
        1 - (rankdata(p[pool], method="average") - 0.5) / pool.sum()
    )  # the quantile of a pixel's p; ties at their mean
    quantile = np.full(len(p), np.nan)
    quantile[pool] = top

    def view_map(values: np.ndarray) -> np.ndarray:
        out = np.full(pixels.shape, np.nan)
        out[pixels] = values
        return out[VIEW]

    gradient = LinearSegmentedColormap.from_list(
        "drop_keep", ["tab:red", "gold", "tab:green"]
    )
    colors = np.ones((len(p), 3))  # confidence-kept: white
    colors[decided & ~base] = mpl.colors.to_rgb(
        "tab:green"
    )  # oracle-kept farther than FAR
    colors[pool] = gradient(p[pool])[:, :3]
    rgb = np.ones((*pixels.shape, 3))
    rgb[pixels] = colors
    fig, axes = plt.subplots(1, 3, figsize=(20, 6), layout="constrained")
    p_map = view_map(np.where(pool, p, np.nan))
    axes[0].imshow(rgb[VIEW], interpolation="nearest")
    fig.colorbar(
        mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(0, 1), cmap=gradient),
        ax=axes[0],
        shrink=0.7,
        label="p of the undecided pixels",
    )
    positive = {q: t for q, t in tau.items() if t > 0}
    if positive:
        lines = axes[0].contour(
            np.ma.masked_invalid(p_map),
            levels=sorted(positive.values()),
            colors="red",
            linewidths=0.8,
        )
        axes[0].clabel(
            lines, fmt={t: f"q {q}" for q, t in positive.items()}, fontsize=7
        )
    axes[0].set_title(
        f"white: kept by confidence; green: oracle-kept > {FAR:g} cm; between: p = {formula} (red 0 to green 1)\n"
        "black isolines: the threshold of each quantile q (none where p is 0: ties)",
        fontsize=8,
    )
    q_map = view_map(quantile)
    shown = axes[1].imshow(
        q_map,
        cmap=plt.get_cmap("magma_r").copy(),
        vmin=0,
        vmax=1,
        interpolation="nearest",
    )
    axes[1].imshow(
        view_map(np.where(decided & ~base, 1.0, np.nan)),
        cmap=ListedColormap(["tab:green"]),
        interpolation="nearest",
    )
    fig.colorbar(shown, ax=axes[1], shrink=0.7)
    lines = axes[1].contour(
        np.ma.masked_invalid(q_map), levels=list(QS[1:-1]), colors="k", linewidths=0.6
    )
    axes[1].clabel(lines, fmt="%.1f", fontsize=7)
    axes[1].set_title(
        "quantile of p among the undecided pixels (ties at their mean rank)\nwhite: kept by confidence, green: oracle-kept > 10 cm; isolines q = 0.1 ... 0.9",
        fontsize=8,
    )
    for ax in axes[:2]:
        ax.set_xticks([])
        ax.set_yticks([])
    qs, _, chamfer, alone = zip(*results, strict=True)
    axes[2].plot(qs, chamfer, "o-", label="two gradients, quantile q of the undecided")
    if args.floor:
        axes[2].plot(qs, alone, "s--", label="oracle alone at the same kept share")
    axes[2].axhline(1.846, color="gray", ls=":", label="uncut Déjà View 1.846")
    axes[2].axhline(
        1.400, color="tab:green", ls=":", label="union, no gate 1.400 (kept 0.695)"
    )
    axes[2].axhline(
        1.488,
        color="tab:orange",
        ls=":",
        label="band 0.3 < c <= 0.5, 1.488 (kept 0.637)",
    )
    axes[2].set_xlabel("q")
    axes[2].set_ylabel("Chamfer (cm), bedroom")
    axes[2].legend(fontsize=8)
    axes[2].set_title("Chamfer against q", fontsize=9)
    fig.suptitle(
        f"{PRESENTATION_SCENE}, view {VIEW}: two gradients (g_c = min(c / 0.5, 1), g_o = clip(d / {FAR:g} cm)), quantile cut of the undecided pixels",
        fontsize=10,
    )
    suffix = "" if args.combine == "product" else "_mean"
    fig.savefig(
        f"experiments/figures/e57c_gradients{suffix}_isolines_view18.jpg", dpi=80
    )
    plt.close(fig)


if __name__ == "__main__":
    main()
