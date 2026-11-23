"""E84: a filter from the optimum of Déjà View's confidence loss, against its actual confidence.

Déjà View's confidence loss (`dvlt.model_components.loss.common.ConfLoss`, defaults gamma = 1,
alpha = 0.2) is, per pixel, gamma c e - alpha log c with c = 1 + exp(x) >= 1 and e the absolute
error of the pixel's depth, both depths divided by the mean distance of their world points to
the origin (here the first camera's centre, which is Déjà View's frame). It is convex in c with
the optimum c* = max(1, alpha / (gamma e)): a ranking by -e, the clip at 1 only merging ties.
`c >= tau` for tau > 1 keeps e <= alpha / (gamma tau). This script builds that filter from the
GT depth of each view, thresholds the actual confidence the same way, and compares the masks:
at equal kept share q, at equal numeric threshold tau (calibration of c against c*), and by
the exact Chamfer (the loss's fixed alignment) of each cut. The released config trains no
confidence loss, so c need not match c*. Bedroom, a diagnostic: the filter uses the GT.

`uv run python experiments/scripts/e84_dejaview_optimum_filter.py --out experiments/results/e84_dejaview_optimum_filter.txt --figures experiments/figures`
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.patches import Patch
from scipy.stats import spearmanr

from src.evaluation.labeled import ground_truth_cloud
from src.geometry import Geometry
from src.loaders.ground_truth import GroundTruth
from src.loaders.views import Views
from src.refinement.keep_chamfer import KeepScene, exact_chamfer

SCENE = "ai_006_007_cam_00"
VIEWS = (18, 0, 12, 6)
ALPHA, GAMMA = 0.2, 1.0  # ConfLoss defaults
SHARES = (0.9, 0.7, 0.5, 0.3, 0.1)
TAUS = (2.0, 5.0, 10.0, 20.0)
CFG = {
    "root": "data/processed/hypersim",
    "views": "views_n20_uniform_504",
    "backbone": "dejaview_k16",
    "r": 8,
    "miss": "bound",
}


def optimal_confidence(error: np.ndarray) -> np.ndarray:
    """Return c* = argmin over c >= 1 of gamma c e - alpha log c, for errors e >= 0."""
    assert (error >= 0).all()
    assert np.isfinite(error).all()
    return np.maximum(1.0, ALPHA / (GAMMA * np.maximum(error, 1e-9)))


def normalized_error(
    geometry: Geometry, gt: GroundTruth, views: Views
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return e (V, H, W), the GT-pixel mask and the GT scale: ConfLoss's depth error.

    e = |d / s_pred - d_gt / s_gt| with s the mean distance of the valid world points to the
    first camera's centre (Déjà View's origin), for the prediction and for the GT.
    """
    truth = ground_truth_cloud(
        gt.depth, gt.instance_ids, gt.classes, views.intrinsics, views.cam_to_world
    )
    valid = truth.pixels
    assert valid is not None
    points = geometry.ray_origins + geometry.depth[..., None] * geometry.ray_directions
    s_pred = np.linalg.norm(
        points[valid] - geometry.cam_to_world[0][:3, 3], axis=-1
    ).mean()
    s_gt = np.linalg.norm(truth.points - views.cam_to_world[0][:3, 3], axis=-1).mean()
    error = np.where(valid, np.abs(geometry.depth / s_pred - gt.depth / s_gt), 0.0)
    return error.astype(np.float64), valid, float(s_gt)


def top_share(score: np.ndarray, share: float) -> np.ndarray:
    """Return the mask of the `share` highest scores of a 1-D array (exact share, ties by order)."""
    kept = np.zeros(len(score), dtype=bool)
    kept[np.argsort(-score, kind="stable")[: round(share * len(score))]] = True
    return kept


def iou(a: np.ndarray, b: np.ndarray) -> float:
    """Return |a & b| / |a | b|."""
    return float((a & b).sum() / (a | b).sum())


def figures(out: Path, data: dict) -> None:
    """Write the mask figure (views 18, 0, 12, 6) and the calibration figure."""
    views, conf, c_opt, valid, err = (
        data[k] for k in ("views", "conf", "c_opt", "valid", "err")
    )
    n = len(VIEWS)
    fig, ax = plt.subplots(5, n, figsize=(3.6 * n, 14))
    both = {}
    for col, v in enumerate(VIEWS):
        keep_dv = conf[v] >= np.quantile(conf[valid], 0.5)
        keep_opt = err[v] <= np.quantile(err[valid], 0.5)
        agree = np.zeros((*keep_dv.shape, 3))
        agree[keep_dv & keep_opt] = (1.0, 1.0, 1.0)
        agree[keep_opt & ~keep_dv] = (0.9, 0.5, 0.1)
        agree[keep_dv & ~keep_opt] = (0.2, 0.5, 0.9)
        agree[~keep_dv & ~keep_opt] = (0.15, 0.15, 0.15)
        agree[~valid[v]] = (0.5, 0.5, 0.5)
        both[v] = agree
        ax[0, col].imshow(views.images[v])
        ax[0, col].set_title(f"view {v}")
        ax[1, col].imshow(conf[v], norm=LogNorm(1, 60), cmap="viridis")
        ax[2, col].imshow(c_opt[v], norm=LogNorm(1, 60), cmap="viridis")
        ax[3, col].imshow(err[v] * data["s_gt"] * 100, vmin=0, vmax=20, cmap="magma")
        ax[4, col].imshow(agree)
    for r, label in enumerate(
        (
            "image",
            "Déjà View confidence c (log, 1-60)",
            "optimum c* = max(1, 0.2 / e) (same scale)",
            "e: depth error (cm, clipped at 20)",
            "masks, top 50 %",
        )
    ):
        ax[r, 0].set_ylabel(label, fontsize=8)
    for a in ax.ravel():
        a.set_xticks([])
        a.set_yticks([])
    ax[4, 0].legend(
        handles=[
            Patch(color=(1, 1, 1), label="both keep"),
            Patch(color=(0.9, 0.5, 0.1), label="only the optimum keeps"),
            Patch(color=(0.2, 0.5, 0.9), label="only confidence keeps"),
            Patch(color=(0.15, 0.15, 0.15), label="both drop"),
        ],
        loc="lower left",
        fontsize=6,
    )
    fig.suptitle(
        f"bedroom {SCENE}: Déjà View's confidence against the optimum of its own loss "
        "(GT depth); masks at the same kept share 50 %",
        fontsize=9,
    )
    fig.savefig(out / "e84_dejaview_optimum_masks.jpg", dpi=110, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    pick = np.random.default_rng(0).choice(int(valid.sum()), 400_000, replace=False)
    ax[0].hist2d(
        np.log10(conf[valid][pick]),
        np.log10(c_opt[valid][pick]),
        bins=120,
        norm=LogNorm(),
    )
    ax[0].plot([0, 2], [0, 2], "w--", lw=1)
    ax[0].set(
        xlabel="log10 Déjà View confidence c",
        ylabel="log10 optimum c*",
        title="c against c* (dashed: c = c*)",
    )
    q = data["q"]
    ax[1].plot(q, data["iou"], "o-")
    ax[1].set(
        xlabel="kept share q", ylabel="IoU of the two masks", title="mask overlap"
    )
    ax[2].plot(q, data["chamfer_dv"], "o-", label="Déjà View confidence")
    ax[2].plot(q, data["chamfer_opt"], "o-", label="optimum of its loss (GT)")
    ax[2].axhline(data["uncut"], color="gray", ls="--", label="uncut")
    ax[2].set(
        xlabel="kept share q",
        ylabel="exact Chamfer (cm)",
        yscale="log",
        title="Chamfer of each cut",
    )
    ax[2].legend(fontsize=8)
    fig.suptitle(f"bedroom {SCENE}, all 20 views, fixed alignment (E84)", fontsize=9)
    fig.savefig(
        out / "e84_dejaview_optimum_calibration.jpg", dpi=120, bbox_inches="tight"
    )
    plt.close(fig)


def main() -> None:
    """Compare the masks, write the table and the figures."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--figures", type=Path, required=True)
    args = parser.parse_args()
    folder = Path(CFG["root"]) / SCENE
    views = Views.load(folder / f"{CFG['views']}.npz")
    gt = GroundTruth.load(folder / f"gt_{CFG['views']}.npz")
    geometry = Geometry.load(folder / f"{CFG['backbone']}_{CFG['views']}.npz")
    err, valid, s_gt = normalized_error(geometry, gt, views)
    conf = geometry.confidence.astype(np.float64)
    c_opt = optimal_confidence(err)
    e_v, c_v, o_v = err[valid], conf[valid], c_opt[valid]
    pick = np.random.default_rng(0).choice(len(e_v), 300_000, replace=False)
    scene = KeepScene.load(SCENE, CFG, "cpu")
    assert len(scene.pred) == len(e_v)

    def score(kept: np.ndarray) -> dict[str, float]:
        return exact_chamfer(kept, scene.probes, scene.pred, scene.truth)

    uncut = score(np.ones(len(e_v), dtype=bool))["chamfer_cm"]
    rho = float(spearmanr(c_v[pick], o_v[pick])[0])
    head = f"bedroom {SCENE}, {len(e_v)} pixels with GT; alpha {ALPHA}, gamma {GAMMA}"
    scale = f"GT scale {s_gt:.3f} m (e is the depth error over this scale)"
    stats_c = (
        f"c: median {np.median(c_v):.2f}, min {c_v.min():.2f}, max {c_v.max():.1f}"
    )
    stats_o = (
        f"c*: median {np.median(o_v):.2f}, at the floor c* = 1: {(o_v <= 1).mean():.3f}"
    )
    stats_r = f"Spearman(c, c*) {rho:.3f}; median c / c* {np.median(c_v / o_v):.3f}"
    lines = [
        head,
        scale,
        stats_c,
        stats_o,
        stats_r,
        "same numeric threshold tau: kept share of c >= tau | of c* >= tau | IoU",
    ]
    for tau in TAUS:
        a, b = c_v >= tau, o_v >= tau
        lines.append(
            f"  tau {tau:4.0f}: {a.mean():.3f} | {b.mean():.3f} | {iou(a, b):.3f}"
        )
    lines.append(
        f"same kept share q (uncut Chamfer {uncut:.3f}); per cut: IoU, then exact "
        "Chamfer cm (accuracy / completeness) of confidence | of the optimum"
    )
    stats = {"iou": [], "chamfer_dv": [], "chamfer_opt": []}
    for q in SHARES:
        dv, opt = top_share(c_v, q), top_share(-e_v, q)
        s_dv, s_opt = score(dv), score(opt)
        stats["iou"].append(iou(dv, opt))
        stats["chamfer_dv"].append(s_dv["chamfer_cm"])
        stats["chamfer_opt"].append(s_opt["chamfer_cm"])
        lines.append(
            f"  q {q:.1f}: IoU {iou(dv, opt):.3f} | {s_dv['chamfer_cm']:.3f} "
            f"({s_dv['accuracy_cm']:.3f} / {s_dv['completeness_cm']:.3f}) | "
            f"{s_opt['chamfer_cm']:.3f} ({s_opt['accuracy_cm']:.3f} / "
            f"{s_opt['completeness_cm']:.3f})"
        )
    lines.append("per view at q = 0.5 (each view's own pixels, pooled threshold): IoU")
    t_dv, t_opt = np.quantile(c_v, 0.5), np.quantile(e_v, 0.5)
    for v in VIEWS:
        m = valid[v]
        lines.append(f"  view {v}: {iou(conf[v][m] >= t_dv, err[v][m] <= t_opt):.3f}")
    args.out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    figures(
        args.figures,
        {
            "views": views,
            "conf": conf,
            "c_opt": c_opt,
            "valid": valid,
            "err": err,
            "s_gt": s_gt,
            "q": SHARES,
            "uncut": uncut,
            **stats,
        },
    )


if __name__ == "__main__":
    main()
