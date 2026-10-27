"""Figures for one refinement run: results, what each energy term does, and how each weight trades off.

The ablation re-refines with one energy alone, with pairs and with all of them, next to BiNI's CG solution of the
normal term (`normals.py`) as a reference.
"""

import argparse
import dataclasses
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap, LogNorm, Normalize
from matplotlib.figure import Figure
from matplotlib.image import AxesImage
from matplotlib.ticker import NullFormatter
from scipy.spatial import cKDTree

from semantic_3d_refinement.multiview import PLANE_K, VOXEL, load_scene, local_planes, voxel_downsample
from semantic_3d_refinement.normals import BiniConfig, bini, grid_score, normals_from_depth
from semantic_3d_refinement.refine import (
    LOG_KEYS,
    NormalTerm,
    RefineConfig,
    Variant,
    energy_maps,
    normal_term,
    normalize,
    points_from_rays,
    refine,
    run_ablation,
    term_weights,
    timed_refine,
    view_metrics,
)
from semantic_3d_refinement.views import view_from_config

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#8a5cc2"
TERM_COLORS = {"E_d": ORANGE, "E_bd": BLUE, "E_n": VIOLET, "E_delta": AQUA}
TERM_NAMES = {
    "E_d": "data  λ_d·E_d",
    "E_bd": "bending  λ_bd·E_bd",
    "E_n": "normals  λ_n·E_n",
    "E_delta": "residual prior  λ_δ·‖δ‖²",
}
RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SEQUENTIAL = LinearSegmentedColormap.from_list("seq_blue", RAMP)
DIVERGING = LinearSegmentedColormap.from_list("div", ["#104281", "#6da7ec", "#f0efec", "#ec8a89", "#b42d2d"])

CAPTIONS = {
    "depth": "Depth maps on one color scale: input, refined, and ground truth when known.",
    "error": "Relative depth error per pixel before and after refinement. What remains sits on the creases.",
    "delta": "The dense residual δ that refinement adds to the normalized inverse depth.",
    "profile": "Depth along one image row through a crease.",
    "points_topdown": "Point clouds seen from above (x-z).",
    "normal_sources": "Normals of the view: Déjà View's depth lifted through its predicted rays and through a pinhole "
    "fit of them, DSINE's prediction (rotated into the same frame), and the depth after BiNI integrates DSINE's "
    "normals against Déjà View's depth. Grid: patch-grid score, about 1 without a 14 px grid.",
    "detail_crops": "The same panels, zoomed where DSINE's normals carry the most detail.",
    "compare": "BiNI's own bilateral edges, then cuts on the prior's occlusion contours under several depth-prior "
    "weights λ (lower lets DSINE's normals set more of the shape); "
    "bottom row zoomed where DSINE has the most detail. In brackets: median angle to DSINE's normals.",
    "ablation": "Each energy alone, in pairs, and all together (columns), with BiNI's CG solution of E_d + E_n as "
    "reference. Top: normals of the result (central differences). Bottom: error against ground truth, or a zoom "
    "where the normals have the most detail. Iterations: up to 1000, 5000 where no data term anchors the result.",
    "ablation_profile": "Depth along one image row for each variant, against the input and ground truth when "
    "known. For E_bd alone without contour cuts, dashed: the least-squares plane of the input's inverse depth, "
    "where it converges in the limit (L-BFGS steps never change the input's null-space part); with cuts, the "
    "limit is one plane per region.",
    "ablation_topdown": "Point clouds of each variant seen from above (x-z), on shared axes.",
    "convergence": "E_d + E_n by L-BFGS under growing iteration budgets, against BiNI's CG solution of nearly the "
    "same energy (its data term is quadratic in log depth instead of Huber in inverse depth).",
    "energy_terms": "Weighted contribution of each energy term during L-BFGS: smoothing cost is bought down by "
    "spending data cost.",
    "cost_maps": "Where each term pays, per pixel (log scale).",
    "forces": "Tug of war at convergence: the force -∂E/∂δ each term exerts per pixel. Data pulls back toward "
    "the input, bending pushes toward planes, normals toward the predicted slopes; they cancel (net ≈ 0).",
    "sweep_lambda_bd": "Effect of the bending weight λ_bd: too low keeps the noise, too high rounds off the creases.",
    "data_term_ablation": "Effect of the data term's form on the 1% outliers: an unweighted quadratic lets them "
    "drag their neighbors into blobs. Confidence weighting alone removes them here; Huber only matters without it.",
    "accuracy_curves": "Fraction of each variant's points within τ of the laser scan, pooled over a scene's three "
    "views (top), and its change against Déjà View's input (bottom). Each depth map is aligned by one median scale.",
    "error_maps": "Distance of each pixel's point to the nearest scan point, after median scaling (gray: the scan "
    "saw nothing there). Last column: E_d + E_n minus the input, blue where refinement moved points closer.",
    "eval_normals": "Normals of each depth map lifted through the calibrated camera; last column the scan's own, "
    "z-buffered into the view (gray: no scan point).",
    "mv_renders": "All 20 views fused, from a viewpoint 0.8 m behind and 0.5 m above view 10, on a 5 mm voxel grid. "
    "Top: headlight shading of each grid point's local plane (24 neighbors). Bottom: that plane's RMS thickness. "
    "Layers of views that disagree show as mottled shading and a thick surface; the scan is the reference. "
    "Medians here are over every point in view, the table's over the pixels the scan saw.",
    "mv_slices": "A 2 cm horizontal slab at camera height, from above, zoomed on the wall where the input is most "
    "layered (chosen on the input). Gray: scan; blue: all 20 views fused. One wall should be one line.",
    "per_view": "Change of each variant's median distance against the input, one dot per view; left of zero is better.",
    "sensitivity": "Error versus each weight, all else fixed at the run's config (circle). Dashed: input error.",
}

plt.rcParams |= {
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID,
    "axes.labelcolor": INK_2,
    "axes.titlecolor": INK,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.color": INK_2,
    "ytick.color": INK_2,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "lines.linewidth": 2,
    "legend.frameon": False,
    "legend.fontsize": 8,
    "font.size": 9,
}


def show_image(ax: Axes, img: np.ndarray, title: str, cmap: LinearSegmentedColormap, norm: Normalize) -> AxesImage:
    """One image panel without axes."""
    im = ax.imshow(img, cmap=cmap, norm=norm)
    ax.set_title(title)
    ax.set_axis_off()
    return im


def image_row(
    images: list[np.ndarray], titles: list[str], cmap: LinearSegmentedColormap, label: str, norm: Normalize
) -> Figure:
    """Images side by side on one shared color scale."""
    fig, axes = plt.subplots(1, len(images), figsize=(3.6 * len(images), 3.2), constrained_layout=True, squeeze=False)
    for ax, img, title in zip(axes[0], images, titles, strict=True):
        im = show_image(ax, img, title, cmap, norm)
    fig.colorbar(im, ax=axes[0], shrink=0.85, label=label)
    return fig


def rel_error(depth: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Per-pixel |D - D_gt| / D_gt."""
    return np.abs(depth - gt) / gt


def crease_window(gt: np.ndarray, row: int, half: int = 20) -> slice:
    """Columns around the strongest slope change of the ground-truth inverse depth along a row."""
    kink = int(np.abs(np.diff(1 / gt[row], 2)).argmax()) + 1
    return slice(max(kink - half, 0), min(kink + half, gt.shape[1]))


def overview(out: Path, depth: np.ndarray, rays: np.ndarray, gt: np.ndarray | None, refined: dict) -> None:
    """Results: depth, error, residual, row profile, top-down point clouds."""
    depth_hat, delta = refined["depth"], refined["delta"]
    images, titles = [depth, depth_hat], ["input", "refined"]
    if gt is not None:
        images, titles = [*images, gt], [*titles, "ground truth"]
    lo, hi = np.percentile(np.concatenate([i.ravel() for i in images]), [1, 99])
    image_row(images, titles, SEQUENTIAL, "depth", Normalize(lo, hi)).savefig(out / "depth.png", dpi=150)

    if gt is not None:
        err_in, err_out = rel_error(depth, gt), rel_error(depth_hat, gt)
        titles = [f"input  (mean {err_in.mean():.2%})", f"refined  (mean {err_out.mean():.2%})"]
        norm = Normalize(0, np.percentile(err_in, 99))
        image_row([err_in, err_out], titles, SEQUENTIAL, "|D - D_gt| / D_gt", norm).savefig(out / "error.png", dpi=150)

    m = float(np.percentile(np.abs(delta), 99))
    fig = image_row([delta], ["residual δ, clipped at p99 of |δ|"], DIVERGING, "δ", Normalize(-m, m))
    fig.savefig(out / "delta.png", dpi=150)

    row = depth.shape[0] * 3 // 4
    fig, ax = plt.subplots(figsize=(7, 3.2), constrained_layout=True)
    cols = np.arange(depth.shape[1])
    ax.plot(cols, depth[row], color=ORANGE, linewidth=1, label="input")
    if gt is not None:
        ax.plot(cols, gt[row], color=AQUA, label="ground truth")
    ax.plot(cols, depth_hat[row], color=BLUE, label="refined")
    ax.set(title=f"depth profile, image row {row}", xlabel="column (px)", ylabel="depth")
    ax.legend(loc="upper left")
    fig.savefig(out / "profile.png", dpi=150)

    clouds = {"input": depth, "refined": depth_hat} | ({"ground truth": gt} if gt is not None else {})
    rays_t = torch.from_numpy(rays)
    pts = {k: points_from_rays(torch.from_numpy(d), rays_t).reshape(-1, 3).numpy() for k, d in clouds.items()}
    (x0, z0), (x1, z1) = np.percentile(pts["refined"][:, [0, 2]], [0.5, 99.5], axis=0)
    pad = 0.1 * max(x1 - x0, z1 - z0)
    fig, axes = plt.subplots(1, len(clouds), figsize=(3 * len(clouds), 3.8), sharex=True, sharey=True)
    fig.set_constrained_layout(True)
    for ax, (name, p) in zip(axes, pts.items(), strict=True):
        ax.scatter(p[:, 0], p[:, 2], s=0.3, color=BLUE, alpha=0.4, linewidths=0, rasterized=True)
        ax.set(title=name, xlabel="x", aspect="equal", xlim=(x0 - pad, x1 + pad), ylim=(z0 - pad, z1 + pad))
    axes[0].set_ylabel("z")
    fig.suptitle("point cloud, top-down (x-z)", color=INK)
    fig.savefig(out / "points_topdown.png", dpi=150)
    plt.close("all")


def energy_terms(out: Path, log: np.ndarray, cfg: RefineConfig) -> None:
    """Weighted contribution of each term over the optimization."""
    weights = term_weights(cfg)
    fig, ax = plt.subplots(figsize=(7, 3.4), constrained_layout=True)
    steps = np.arange(len(log))
    ax.semilogy(steps, log[:, 0], color=INK, linewidth=1, linestyle="--", label="total E")
    for i, k in enumerate(LOG_KEYS[1:], start=1):
        if weights[k] == 0:
            continue
        y = np.where(log[:, i] > 0, weights[k] * log[:, i], np.nan)
        ax.semilogy(steps, y, color=TERM_COLORS[k], label=TERM_NAMES[k])
        ax.annotate(
            f"{y[-1]:.1e}", (steps[-1], y[-1]), xytext=(4, 0), textcoords="offset points", fontsize=8, color=INK_2
        )
    ax.set(title="energy terms during L-BFGS (weighted)", xlabel="function evaluation", ylabel="contribution to E")
    ax.legend(loc="upper right")
    fig.savefig(out / "energy_terms.png", dpi=150)
    plt.close(fig)


def term_figures(  # noqa: PLR0913
    out: Path, depth: np.ndarray, conf: np.ndarray, delta: np.ndarray, cfg: RefineConfig, *, normal: NormalTerm | None
) -> None:
    """Per-pixel cost of each term and the force each exerts at the solution."""
    _, d0, w = normalize(torch.from_numpy(depth), torch.from_numpy(conf))
    weights = term_weights(cfg)
    delta_t = torch.from_numpy(delta).requires_grad_()
    start = {k: weights[k] * v for k, v in energy_maps(torch.zeros_like(d0), d0, w, cfg, normal).items()}
    end = {k: weights[k] * v for k, v in energy_maps(delta_t, d0, w, cfg, normal).items()}
    active = [k for k in TERM_NAMES if weights[k] > 0]
    smoothing = [k for k in ("E_bd", "E_n") if k in active]

    maps = [m for k in smoothing for m in (start[k], end[k])] + [end[k] for k in active if k not in smoothing]
    titles = [f"{TERM_NAMES[k].split('  ')[0]}, {when}" for k in smoothing for when in ("before", "after")]
    titles += [f"{TERM_NAMES[k].split('  ')[0]}, after" for k in active if k not in smoothing]
    imgs = [m.detach().numpy() for m in maps]
    hi = max(float(np.percentile(i, 99.5)) for i in imgs)
    fig = image_row(imgs, titles, SEQUENTIAL, "weighted cost per pixel", LogNorm(hi * 1e-6, hi, clip=True))
    fig.suptitle("where each energy term pays", color=INK)
    fig.savefig(out / "cost_maps.png", dpi=150)

    forces = {k: -torch.autograd.grad(end[k].sum(), delta_t, retain_graph=True)[0].numpy() for k in active}
    net = sum(forces.values())
    imgs = [*forces.values(), net]
    titles = [f"{TERM_NAMES[k].split('  ')[0]} force" for k in active] + [
        f"net force (max |·| {np.abs(net).max():.0e})"
    ]
    m = float(np.percentile(np.abs(imgs[0]), 99))
    fig = image_row(imgs, titles, DIVERGING, "-∂E/∂δ  (red: nearer, blue: farther)", Normalize(-m, m))
    fig.suptitle("tug of war at convergence: each term's force on δ", color=INK)
    fig.savefig(out / "forces.png", dpi=150)
    plt.close("all")


def run_variant(data: dict[str, torch.Tensor], cfg: RefineConfig, *, use_conf: bool = True) -> np.ndarray:
    """Refine the run's input (with its normals, if any) under a modified config; return the refined depth."""
    conf = data["conf"] if use_conf else torch.ones_like(data["conf"])
    return refine(data["depth"], data["rays"], conf, cfg, data.get("normals"))[0].cpu().numpy()


def sweep_lambda_bd(out: Path, depth: np.ndarray, gt: np.ndarray, lams: list[float], refined: list[np.ndarray]) -> None:
    """Error maps per bending weight above depth profiles across a crease."""
    norm = Normalize(0, np.percentile(rel_error(depth, gt), 99))
    fig = plt.figure(figsize=(2.6 * len(lams), 6.4), constrained_layout=True)
    grid = fig.add_gridspec(2, len(lams), height_ratios=[1, 1.15])
    axes = [fig.add_subplot(grid[0, i]) for i in range(len(lams))]
    for ax, lam, d in zip(axes, lams, refined, strict=True):
        im = show_image(ax, rel_error(d, gt), f"λ_bd = {lam:g}\nmean {rel_error(d, gt).mean():.2%}", SEQUENTIAL, norm)
    fig.colorbar(im, ax=axes, shrink=0.8, label="|D - D_gt| / D_gt")
    row = depth.shape[0] * 3 // 4
    window = crease_window(gt, row)
    cols = np.arange(depth.shape[1])[window]
    ax = fig.add_subplot(grid[1, :])
    steps = np.linspace(2, len(RAMP) - 1, len(lams)).round().astype(int)
    for lam, d, s in zip(lams, refined, steps, strict=True):
        ax.plot(
            cols,
            d[row, window],
            color=RAMP[s],
            linewidth=1.6,
            label=f"λ_bd = {lam:g}" + (" (input)" if lam == 0 else ""),
        )
    ax.plot(cols, gt[row, window], color=INK, linewidth=1.2, linestyle="--", label="ground truth")
    ax.set(title=f"depth across a crease, image row {row}", xlabel="column (px)", ylabel="depth")
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
    fig.suptitle("bending weight: noise ↔ rounded creases", color=INK)
    fig.savefig(out / "sweep_lambda_bd.png", dpi=150)
    plt.close(fig)


def data_term_ablation(out: Path, data: dict[str, torch.Tensor], cfg: RefineConfig) -> None:
    """Quadratic vs Huber penalty, uniform vs confidence weights."""
    gt, depth = data["gt_depth"].cpu().numpy(), data["depth"].cpu().numpy()
    norm = Normalize(0, np.percentile(rel_error(depth, gt), 99))
    quadratic = dataclasses.replace(cfg, huber=1e3)
    variants = {
        "uniform weights, quadratic": (quadratic, False),
        f"uniform weights, Huber κ = {cfg.huber:g}": (cfg, False),
        "confidence weights, quadratic": (quadratic, True),
        f"confidence weights, Huber κ = {cfg.huber:g}": (cfg, True),
    }
    fig, axes = plt.subplots(2, 2, figsize=(7.6, 6.4), constrained_layout=True)
    for ax, (title, (c, use_conf)) in zip(axes.ravel(), variants.items(), strict=True):
        err = rel_error(run_variant(data, c, use_conf=use_conf), gt)
        im = show_image(ax, err, f"{title}\nmean {err.mean():.2%}", SEQUENTIAL, norm)
    fig.colorbar(im, ax=axes, shrink=0.8, label="|D - D_gt| / D_gt")
    fig.suptitle("data term: what keeps outliers from dragging their neighbors", color=INK)
    fig.savefig(out / "data_term_ablation.png", dpi=150)
    plt.close(fig)


def sensitivity(
    out: Path, err_in: float, cfg: RefineConfig, sweep: dict[str, list[float]], errs: dict[str, list[float]]
) -> None:
    """Mean error versus each weight, one at a time, on a shared y-axis."""
    labels = {
        "lambda_bd": "bending weight λ_bd",
        "lambda_n": "normal weight λ_n",
        "huber": "Huber threshold κ",
        "lambda_delta": "residual prior λ_δ",
    }
    fig, axes = plt.subplots(1, len(sweep), figsize=(3.4 * len(sweep), 3.2), sharey=True, constrained_layout=True)
    axes = np.atleast_1d(axes)
    for ax, (k, vs) in zip(axes, sweep.items(), strict=True):
        ax.plot(vs, errs[k], color=BLUE, marker="o", markersize=4)
        ax.axhline(err_in, color=INK_2, linewidth=1, linestyle="--")
        here = getattr(cfg, k)
        if here in vs:
            y = errs[k][vs.index(here)]
            ax.plot([here], [y], marker="o", markersize=10, markerfacecolor="none", markeredgecolor=INK)
        ax.set_xscale("symlog", linthresh=min(v for v in vs if v > 0))
        ax.set(title=labels.get(k, k), xlabel="value (symlog)", yscale="log")
    axes[0].set_ylabel("mean |D - D_gt| / D_gt")
    axes[0].annotate("input", (0.02, err_in), xycoords=("axes fraction", "data"), xytext=(0, 3),
                     textcoords="offset points", fontsize=8, color=INK_2)  # fmt: skip
    fig.suptitle("sensitivity: one weight at a time", color=INK)
    fig.savefig(out / "sensitivity.png", dpi=150)
    plt.close(fig)


def sweep_figures(out: Path, data: dict[str, torch.Tensor], cfg: RefineConfig, sweep: dict[str, list[float]]) -> None:
    """Re-refine with one weight changed at a time."""
    gt, depth = data["gt_depth"].cpu().numpy(), data["depth"].cpu().numpy()
    refined = {k: [run_variant(data, dataclasses.replace(cfg, **{k: v})) for v in vs] for k, vs in sweep.items()}
    if "lambda_bd" in sweep:
        sweep_lambda_bd(out, depth, gt, sweep["lambda_bd"], refined["lambda_bd"])
    data_term_ablation(out, data, cfg)
    errs = {k: [float(rel_error(d, gt).mean()) for d in ds] for k, ds in refined.items()}
    sensitivity(out, float(rel_error(depth, gt).mean()), cfg, sweep, errs)


def shade(depth: np.ndarray, rays: np.ndarray) -> np.ndarray:
    """Shade the normals of the lifted depth as RGB in [0, 1]."""
    return normals_from_depth(torch.from_numpy(depth), torch.from_numpy(rays)).numpy() * 0.5 + 0.5


def plane_limit(depth: np.ndarray) -> np.ndarray:
    """Depth of the least-squares plane of the normalized inverse depth: where E_bd alone converges from d0."""
    h, w = depth.shape
    scale, d0, _ = normalize(torch.from_numpy(depth).double(), torch.ones(h, w, dtype=torch.float64))
    v, u = np.mgrid[0:h, 0:w]
    basis = np.stack([u.ravel(), v.ravel(), np.ones(h * w)], 1)
    plane = (basis @ np.linalg.lstsq(basis, d0.numpy().ravel(), rcond=None)[0]).reshape(h, w)
    return np.where(plane > 0, scale.item() / np.where(plane > 0, plane, 1), np.nan)


def metric_note(m: dict[str, float]) -> str:
    """Short metric line for a panel title."""
    if "abs_rel" in m:
        return f"abs rel {m['abs_rel']:.2%}, {m['angle_to_gt_normals_deg']:.1f}° to GT"
    return f"{m['angle_to_normals_deg']:.1f}° to DSINE, drift {m['drift_pct']:.2f}%"


def ablation_figures(out: Path, data: dict[str, np.ndarray], variants: list[Variant], *, cut: bool) -> None:
    """Plot normals and error (or detail) per variant, then row profiles and top-down point clouds."""
    depth, rays, gt = data["depth"], data["rays"], data.get("gt_depth")
    shaded = [shade(v.depth, rays) for v in variants]
    win = detail_window(data["normals"] * 0.5 + 0.5, 112) if gt is None else None
    n = len(variants)
    fig, axes = plt.subplots(2, n, figsize=(2.9 * n, 5.6 if gt is None else 4.6), constrained_layout=True)
    norm = Normalize(0, np.percentile(rel_error(depth, gt), 99)) if gt is not None else None
    for c, (v, img) in enumerate(zip(variants, shaded, strict=True)):
        axes[0, c].imshow(img, interpolation="nearest")
        axes[0, c].set_title(f"{v.label}\n{metric_note(v.metrics)}", fontsize=7)
        if gt is None:
            axes[1, c].imshow(img[win], interpolation="nearest")
        else:
            im = axes[1, c].imshow(rel_error(v.depth, gt), cmap=SEQUENTIAL, norm=norm)
        for r in (0, 1):
            axes[r, c].set_axis_off()
    if gt is not None:
        fig.colorbar(im, ax=axes[1], shrink=0.8, label="|D - D_gt| / D_gt")
    fig.suptitle("ablation: one energy, pairs, all, and BiNI", color=INK)
    fig.savefig(out / "ablation.png", dpi=150)
    plt.close(fig)
    ablation_profile(out, data, variants, cut=cut)
    ablation_topdown(out, data, variants)


def ablation_profile(out: Path, data: dict[str, np.ndarray], variants: list[Variant], *, cut: bool) -> None:
    """Depth along one row per variant, small multiples on shared axes; the plane limit of E_bd only without cuts."""
    depth, gt, n = data["depth"], data.get("gt_depth"), len(variants)
    row, cols = depth.shape[0] * 3 // 4, np.arange(depth.shape[1])
    lo, hi = np.percentile(np.concatenate([depth[row]] + ([gt[row]] if gt is not None else [])), [0, 100])
    pad = 0.3 * (hi - lo)
    rows = (n + 3) // 4
    fig, axes = plt.subplots(rows, 4, figsize=(13, 2.6 * rows), sharex=True, sharey=True, constrained_layout=True)
    for ax, v in zip(axes.ravel(), variants, strict=False):
        ax.plot(cols, depth[row], color=GRID, linewidth=1.2, label="input")
        if gt is not None:
            ax.plot(cols, gt[row], color=INK, linewidth=1, linestyle="--", label="ground truth")
        ax.plot(cols, v.depth[row], color=BLUE, linewidth=1.4, label="refined")
        if v.label == "E_bd" and not cut:
            ax.plot(cols, plane_limit(depth)[row], color=ORANGE, linewidth=1, linestyle="--", label="LS plane")
        ax.set(title=v.label, ylim=(lo - pad, hi + pad))
        ax.legend(loc="upper left", fontsize=6)
    for ax in axes.ravel()[n:]:
        ax.set_axis_off()
    fig.supxlabel("column (px)", fontsize=9, color=INK_2)
    fig.supylabel("depth", fontsize=9, color=INK_2)
    fig.suptitle(f"depth along image row {row}", color=INK)
    fig.savefig(out / "ablation_profile.png", dpi=150)
    plt.close(fig)


def ablation_topdown(out: Path, data: dict[str, np.ndarray], variants: list[Variant]) -> None:
    """Point clouds per variant from above, small multiples on shared axes."""
    depth, rays, n = data["depth"], data["rays"], len(variants)
    rows = (n + 3) // 4
    rays_t = torch.from_numpy(rays)
    every = max(1, depth.size // 40000)
    clouds = [(v.label, points_from_rays(torch.from_numpy(v.depth), rays_t).reshape(-1, 3).numpy()[::every])
              for v in variants]  # fmt: skip
    ref = points_from_rays(torch.from_numpy(depth), rays_t).reshape(-1, 3).numpy()
    (x0, z0), (x1, z1) = np.percentile(ref[:, [0, 2]], [0.5, 99.5], axis=0)
    pad = 0.15 * max(x1 - x0, z1 - z0)
    fig, axes = plt.subplots(rows, 4, figsize=(12, 3.2 * rows), sharex=True, sharey=True, constrained_layout=True)
    for ax, (label, pts) in zip(axes.ravel(), clouds, strict=False):
        ax.scatter(pts[:, 0], pts[:, 2], s=0.3, color=BLUE, alpha=0.4, linewidths=0, rasterized=True)
        ax.set(title=label, aspect="equal", xlim=(x0 - pad, x1 + pad), ylim=(z0 - pad, z1 + pad))
    for ax in axes.ravel()[n:]:
        ax.set_axis_off()
    fig.suptitle("point clouds, top-down (x-z)", color=INK)
    fig.savefig(out / "ablation_topdown.png", dpi=150)
    plt.close(fig)


def convergence(
    out: Path, data: dict[str, torch.Tensor], cfg: RefineConfig, budgets: list[int], reference: Variant
) -> None:
    """E_d + E_n by L-BFGS under growing iteration budgets against BiNI's solution."""
    ref = torch.from_numpy(reference.depth).to(data["depth"])
    rows = []
    for budget in budgets:
        depth_hat, evals, seconds = timed_refine(data, dataclasses.replace(cfg, lambda_bd=0.0, iters=budget))
        angle = view_metrics(depth_hat, data, data["normals"])["angle_to_normals_deg"]
        rows.append((evals, seconds, angle, 100 * ((depth_hat - ref).abs() / ref).median().item()))
    evals, seconds, angles, gaps = map(np.array, zip(*rows, strict=True))
    angle_in = view_metrics(data["depth"], data, data["normals"])["angle_to_normals_deg"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2), constrained_layout=True)
    axes[0].plot(evals, angles, color=BLUE, marker="o", markersize=4, label="L-BFGS, E_d + E_n")
    axes[0].axhline(reference.metrics["angle_to_normals_deg"], color=VIOLET, linestyle="--", linewidth=1,
                    label=f"BiNI CG ({reference.seconds:.2f} s)")  # fmt: skip
    axes[0].axhline(angle_in, color=INK_2, linestyle=":", linewidth=1, label="input")
    axes[0].set(xscale="log", xlabel="function evaluations", ylabel="median angle to DSINE (°)")
    axes[0].legend()
    axes[1].plot(evals, gaps, color=BLUE, marker="o", markersize=4)
    for e, g, t in zip(evals, gaps, seconds, strict=True):
        axes[1].annotate(f"{t:.1f} s", (e, g), xytext=(4, 4), textcoords="offset points", fontsize=7, color=INK_2)
    axes[1].set(xscale="log", xlabel="function evaluations", ylabel="median |D - D_BiNI| / D_BiNI (%)")
    for ax in axes:
        ax.xaxis.set_minor_formatter(NullFormatter())
    fig.suptitle("L-BFGS against BiNI's CG on the normal term", color=INK)
    fig.savefig(out / "convergence.png", dpi=150)
    plt.close(fig)


def ablation_table(variants: list[Variant]) -> list[str]:
    """Markdown table of every variant's metrics."""
    keys = list(variants[0].metrics)
    lines = ["| variant | evaluations | seconds | " + " | ".join(keys) + " |", "|---" * (len(keys) + 3) + "|"]
    lines += [
        f"| {v.label} | {v.evaluations} | {v.seconds:.2f} | " + " | ".join(f"{v.metrics[k]:.4g}" for k in keys) + " |"
        for v in variants
    ]
    return lines


def figures(run: Path, out: Path) -> dict[str, float]:
    """Write all figures for a refinement run into `out`; return the metric table."""
    summary = yaml.safe_load((run / "run.yaml").read_text())
    cfg = dataclasses.replace(RefineConfig(**summary["resolved"]), device="cpu")
    data = view_from_config(summary["config"])
    refined = dict(np.load(run / "refined.npz"))
    raw = {k: v.numpy() for k, v in data.items()}
    depth, rays, gt, log = raw["depth"], raw["rays"], raw.get("gt_depth"), refined["energy"]
    assert depth.shape == refined["depth"].shape == refined["delta"].shape
    assert log.ndim == 2 and log.shape[1] == len(LOG_KEYS), log.shape  # noqa: PLR2004
    out.mkdir(parents=True, exist_ok=True)

    normal = None
    if "normals" in refined:
        data["normals"] = torch.from_numpy(refined["normals"])
        raw["normals"] = refined["normals"]
        _, d0, _ = normalize(data["depth"], data["conf"])
        normal = normal_term(data["normals"], data["rays"], d0, cfg.edge_tol)

    overview(out, depth, rays, gt, refined)
    energy_terms(out, log, cfg)
    term_figures(out, depth, raw["conf"], refined["delta"], cfg, normal=normal)
    config = summary["config"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu = {k: v.to(device) for k, v in data.items()}
    gpu_cfg = dataclasses.replace(cfg, device=device.type)
    if gt is not None and config.get("sweep"):
        sweep_figures(out, gpu, gpu_cfg, config["sweep"])
    extra = []
    if config.get("ablation"):
        bini_cfg = BiniConfig(**{**config["bini"], "device": device.type}) if "bini" in config else None
        variants = run_ablation(gpu, gpu_cfg, config["ablation"], bini_cfg)
        cut = normal is not None and any(bool((wf == 0).any()) for wf, _ in normal.weights)
        ablation_figures(out, raw, variants, cut=cut)
        extra = ["", "## ablation table", "", *ablation_table(variants)]
        if config.get("convergence") and bini_cfg is not None:
            convergence(out, gpu, gpu_cfg, config["convergence"], variants[-1])

    table = {"refine_seconds": summary["refine_seconds"], "E_initial": log[0, 0], "E_final": log[-1, 0]}
    return write_readme(run, out, summary, table, list(CAPTIONS), extra=extra)


def write_readme(  # noqa: PLR0913
    run: Path, out: Path, summary: dict, table: dict[str, float], names: list[str], *, extra: list[str] | None = None
) -> dict[str, float]:
    """Write the metric table, any extra lines, and every existing figure with its caption; return the table."""
    if "metrics_input" in summary:
        for k in summary["metrics_input"]:
            table |= {f"{k}_input": summary["metrics_input"][k], f"{k}_refined": summary["metrics_refined"][k]}
    lines = [f"# {run.name}", "", f"git `{summary['git']}`", "", "| metric | value |", "|---|---|"]
    lines += [f"| {k} | {v:.4g} |" for k, v in table.items()]
    lines += extra or []
    for name in names:
        if (out / f"{name}.png").exists():
            lines += ["", f"## {name}", "", CAPTIONS[name], "", f"![{name}]({name}.png)"]
    (out / "README.md").write_text("\n".join([*lines, ""]))
    return table


def detail_window(n: np.ndarray, size: int) -> tuple[slice, slice]:
    """Square window with the largest mean normal gradient."""
    g = np.linalg.norm(np.diff(n, axis=0)[:, :-1], axis=-1) + np.linalg.norm(np.diff(n, axis=1)[:-1], axis=-1)
    c = np.cumsum(np.cumsum(np.pad(g, ((1, 0), (1, 0))), 0), 1)
    sums = c[size:, size:] - c[:-size, size:] - c[size:, :-size] + c[:-size, :-size]
    r, col = np.unravel_index(sums.argmax(), sums.shape)
    return slice(r, r + size), slice(col, col + size)


def bini_compare(out: Path, raw: dict, data: dict[str, np.ndarray], refined: dict, image: np.ndarray) -> None:
    """Plot the view's normals under each configured BiNI setting, full and zoomed where DSINE has detail."""
    settings = raw.get("compare", [])
    if not settings:
        return
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rays = torch.from_numpy(data["rays"]).to(device)
    depth = torch.from_numpy(data["depth"]).to(device)
    dsine = torch.from_numpy(refined["normals"]).to(device)

    def shade(d: torch.Tensor) -> tuple[np.ndarray, float]:
        n = normals_from_depth(d, rays)
        angle = torch.rad2deg(torch.acos((n * dsine).sum(-1).clamp(-1, 1))).median().item()
        return n.cpu().numpy() * 0.5 + 0.5, angle

    panels = [("image", image), (f"Déjà View  ({shade(depth)[1]:.1f}°)", shade(depth)[0])]
    panels.append(("DSINE", dsine.cpu().numpy() * 0.5 + 0.5))
    for s in settings:
        cfg = BiniConfig(**{**raw["bini"], **s, "device": device})
        img, angle = shade(bini(dsine, rays, depth, cfg)[0])
        blur_note = f", sigma = {cfg.prior_sigma:g}" if cfg.prior_sigma else ""
        panels.append((f"BiNI, {cfg.edges} edges, λ = {cfg.lambda_depth:g}{blur_note}  ({angle:.1f}°)", img))
    win = detail_window(panels[2][1], 112)
    fig, axes = plt.subplots(2, len(panels), figsize=(3.3 * len(panels), 5.2), constrained_layout=True)
    for c, (title, img) in enumerate(panels):
        for r, w in enumerate(((slice(None), slice(None)), win)):
            axes[r, c].imshow(img[w], interpolation="nearest")
            axes[r, c].set_axis_off()
        axes[0, c].set_title(title, fontsize=8)
    fig.suptitle("BiNI edges and depth-prior weight (° = median angle to DSINE normals)", color=INK)
    fig.savefig(out / "compare.png", dpi=150)
    plt.close(fig)


def bini_figures(run: Path, out: Path) -> dict[str, float]:
    """Write the figures for a normal-integration run (semantic_3d_refinement.normals); return the metric table."""
    summary = yaml.safe_load((run / "run.yaml").read_text())
    raw = summary["config"]
    data = {k: v.numpy() for k, v in view_from_config(raw).items()}
    refined = dict(np.load(run / "refined.npz"))
    image = np.load(raw["gt_views"])["images"][raw["view"]]
    out.mkdir(parents=True, exist_ok=True)
    bini_compare(out, raw, data, refined, image)

    def shade(d: np.ndarray, rays: np.ndarray) -> tuple[np.ndarray, float]:
        n = normals_from_depth(torch.from_numpy(d), torch.from_numpy(rays))
        return n.numpy() * 0.5 + 0.5, grid_score(n)

    panels = [(f"view {raw['view']}", image)]
    if raw.get("rays") == "pinhole":
        predicted = view_from_config(raw | {"rays": "predicted"})["rays"].numpy()
        img, grid = shade(data["depth"], predicted)
        panels.append((f"Déjà View, predicted rays (grid {grid:.2f})", img))
    img, grid = shade(data["depth"], data["rays"])
    panels.append((f"Déjà View{', pinhole rays' if raw.get('rays') == 'pinhole' else ''} (grid {grid:.2f})", img))
    dsine = refined["normals"] * 0.5 + 0.5
    panels.append((f"DSINE (grid {summary['grid_score_normals']:.2f})", dsine))
    img, grid = shade(refined["depth"], data["rays"])
    panels.append((f"BiNI (grid {grid:.2f})", img))
    for name, crop in (("normal_sources", None), ("detail_crops", 112)):
        win = detail_window(dsine, crop) if crop else (slice(None), slice(None))
        fig, axes = plt.subplots(1, len(panels), figsize=(3.3 * len(panels), 2.8), constrained_layout=True)
        for ax, (title, img) in zip(axes, panels, strict=True):
            ax.imshow(img[win], interpolation="nearest")
            ax.set_title(title, fontsize=8)
            ax.set_axis_off()
        fig.savefig(out / f"{name}.png", dpi=150)
        plt.close(fig)

    table = {"normals_seconds": summary["normals_seconds"], "refine_seconds": summary["refine_seconds"]}
    table["grid_score_dsine"] = summary["grid_score_normals"]
    return write_readme(run, out, summary, table, ["compare", "normal_sources", "detail_crops"])


EVAL_STYLES = {
    "E_d (= input)": (INK_2, ":"),
    "E_n": (VIOLET, "--"),
    "E_d + E_bd": (BLUE, "-"),
    "E_d + E_n": (VIOLET, "-"),
    "E_d + E_bd + E_n": (AQUA, "--"),
}
EVAL_MAPS = ["E_d (= input)", "E_d + E_bd", "E_d + E_n", "E_d + E_bd + E_n"]


def eval_style(label: str) -> tuple[str, str] | None:
    """Color and line style of the variants drawn in curves; BiNI in orange, the rest only in tables."""
    return (ORANGE, "-") if label.startswith("BiNI") else EVAL_STYLES.get(label)


def camera_rays(k: np.ndarray, h: int, w: int) -> torch.Tensor:
    """Camera-frame rays of a pinhole with unit z, pixel centers at +0.5, as an (H, W, 6) map."""
    v, u = np.mgrid[0:h, 0:w] + 0.5
    dirs = np.stack([u, v, np.ones_like(u)], axis=-1) @ np.linalg.inv(k).T
    return torch.from_numpy(np.concatenate([np.zeros_like(dirs), dirs], axis=-1))


def accuracy_curves(out: Path, views: list[dict], labels: list[str], res: dict[str, np.ndarray]) -> None:
    """Fraction within τ against τ per scene, pooled over its views, and the change against the input."""
    scenes = list(dict.fromkeys(r["scene"] for r in views))
    tau = np.geomspace(0.2, 50, 120)
    fig, axes = plt.subplots(2, len(scenes), figsize=(5.2 * len(scenes), 6.4), sharex=True, constrained_layout=True)
    for c, scene in enumerate(scenes):
        keys = [f"{r['scene']}_{r['view']}" for r in views if r["scene"] == scene]
        curves = []
        for i in range(len(labels)):
            cm = 100 * np.concatenate([res[f"{k}_{i}_dist"][np.isfinite(res[f"{k}_{i}_dist"])] for k in keys])
            curves.append(np.searchsorted(np.sort(cm), tau, side="right") / cm.size)
        for i, label in enumerate(labels):
            if (style := eval_style(label)) is None:
                continue
            color, ls = style
            axes[0, c].plot(tau, curves[i], color=color, linestyle=ls, linewidth=1.5, label=label)
            axes[1, c].plot(tau, 100 * (curves[i] - curves[0]), color=color, linestyle=ls, linewidth=1.5)
        axes[0, c].set(title=f"{scene} ({len(keys)} views)", ylabel="fraction of points within τ", ylim=(0, 1))
        axes[1, c].set(xscale="log", xlabel="τ (cm)", ylabel="change against input (points, %)")
        axes[1, c].axhline(0, color=INK_2, linewidth=0.8)
        axes[1, c].xaxis.set_minor_formatter(NullFormatter())
    axes[0, 0].legend(loc="lower right")
    fig.suptitle("accuracy against the laser scan", color=INK)
    fig.savefig(out / "accuracy_curves.png", dpi=150)
    plt.close(fig)


def error_maps(out: Path, views: list[dict], labels: list[str], res: dict[str, np.ndarray]) -> None:
    """Per view: image, distance maps of the main variants and BiNI, and the change E_d + E_n brings."""
    cols = [labels.index(lab) for lab in EVAL_MAPS] + [len(labels) - 1]
    gray = SEQUENTIAL.with_extremes(bad=GRID)
    diverging = DIVERGING.with_extremes(bad=GRID)
    fig, axes = plt.subplots(len(views), len(cols) + 2, figsize=(2.6 * (len(cols) + 2), 1.9 * len(views)),
                             constrained_layout=True)  # fmt: skip
    for r, rec in enumerate(views):
        key = f"{rec['scene']}_{rec['view']}"
        axes[r, 0].imshow(res[f"{key}_image"])
        axes[r, 0].set_title(f"{rec['scene']} view {rec['view']}", fontsize=8)
        for c, i in enumerate(cols, start=1):
            im = axes[r, c].imshow(100 * res[f"{key}_{i}_dist"], cmap=gray, norm=Normalize(0, 10))
            axes[r, c].set_title(f"{labels[i]}: {rec['variants'][i]['median_cm']:.2f} cm", fontsize=7)
        gain = 100 * (res[f"{key}_{labels.index('E_d + E_n')}_dist"] - res[f"{key}_0_dist"])
        dm = axes[r, -1].imshow(gain, cmap=diverging, norm=Normalize(-2, 2))
        axes[r, -1].set_title("E_d + E_n minus input", fontsize=7)
        for ax in axes[r]:
            ax.set_axis_off()
    fig.colorbar(im, ax=axes[-1, 1:-1], location="bottom", shrink=0.5, label="distance to scan (cm)")
    fig.colorbar(dm, ax=axes[-1, -1], location="bottom", shrink=0.9, label="change (cm)")
    fig.savefig(out / "error_maps.png", dpi=150)
    plt.close(fig)


def eval_normals(
    out: Path, views: list[dict], labels: list[str], res: dict[str, np.ndarray], intrinsics: list[np.ndarray]
) -> None:
    """Per view: normals of the main variants and BiNI through the calibrated camera, and of the scan."""
    cols = [labels.index(lab) for lab in EVAL_MAPS] + [len(labels) - 1]
    fig, axes = plt.subplots(len(views), len(cols) + 1, figsize=(2.6 * (len(cols) + 1), 1.9 * len(views)),
                             constrained_layout=True)  # fmt: skip
    for r, (rec, k) in enumerate(zip(views, intrinsics, strict=True)):
        key = f"{rec['scene']}_{rec['view']}"
        gt = res[f"{key}_gt_depth"]
        rays = camera_rays(k.astype(np.float64), *gt.shape)
        panels = [(labels[i], res[f"{key}_{i}_depth"].astype(np.float64)) for i in cols]
        for c, (label, depth) in enumerate([*panels, ("scan", np.where(gt > 0, gt, 1.0))]):
            img = normals_from_depth(torch.from_numpy(depth), rays).numpy() * 0.5 + 0.5
            if label == "scan":
                hole = ~np.isfinite(img).all(-1) | (np.lib.stride_tricks.sliding_window_view(
                    np.pad(gt == 0, 1, constant_values=True), (3, 3)).any((-1, -2)))  # fmt: skip
                img[hole] = 0.9
            axes[r, c].imshow(np.clip(img, 0, 1), interpolation="nearest")
            axes[r, c].set_title(label if r == 0 else "", fontsize=8)
            axes[r, c].set_axis_off()
        axes[r, 0].set_title(f"{rec['scene']} view {rec['view']}" + (f"\n{panels[0][0]}" if r == 0 else ""), fontsize=8)
    fig.savefig(out / "eval_normals.png", dpi=150)
    plt.close(fig)


def per_view(out: Path, views: list[dict], labels: list[str]) -> None:
    """Dot plot: change in median distance against the input per variant, one dot per view."""
    fig, ax = plt.subplots(figsize=(7, 0.45 * len(labels) + 1), constrained_layout=True)
    markers = dict(zip(dict.fromkeys(r["scene"] for r in views), "os", strict=False))
    first = {s: next(r for r in views if r["scene"] == s) for s in markers}
    for i in range(1, len(labels)):
        for rec in views:
            change = rec["variants"][i]["median_cm"] - rec["variants"][0]["median_cm"]
            label = rec["scene"] if i == 1 and rec is first[rec["scene"]] else None
            ax.scatter(change, i, color=(eval_style(labels[i]) or (INK_2, ""))[0], marker=markers[rec["scene"]],
                       s=22, alpha=0.8, label=label)  # fmt: skip
    ax.axvline(0, color=INK_2, linewidth=0.8)
    ax.set(yticks=range(1, len(labels)), yticklabels=labels[1:], xlabel="median distance minus input's (cm)")
    ax.invert_yaxis()
    ax.legend(loc="lower right")
    ax.set_title("per view: which variants help, and how consistently", color=INK)
    fig.savefig(out / "per_view.png", dpi=150)
    plt.close(fig)


def eval_tables(views: list[dict], labels: list[str], tolerances: list[int | float]) -> list[str]:
    """Markdown tables: per-variant means over all views, median distance per view, and the views themselves."""

    def mean(i: int, key: str) -> float:
        return float(np.mean([r["variants"][i][key] for r in views]))

    acc = [f"acc_{t:g}cm" for t in tolerances]
    lines = ["## accuracy, mean over all views", "",
             "| variant | median (cm) | mean (cm) | " + " | ".join(f"≤ {t:g} cm" for t in tolerances)
             + " | median vs input | angle to DSINE (°) | drift (%) | seconds |",
             "|---" * (len(tolerances) + 7) + "|"]  # fmt: skip
    for i, label in enumerate(labels):
        change = 100 * (mean(i, "median_cm") / mean(0, "median_cm") - 1)
        lines.append(
            f"| {label} | {mean(i, 'median_cm'):.2f} | {mean(i, 'mean_cm'):.2f} | "
            + " | ".join(f"{100 * mean(i, a):.1f}%" for a in acc)
            + f" | {change:+.1f}% | {mean(i, 'angle_to_normals_deg'):.2f} | {mean(i, 'drift_pct'):.2f} | "
            f"{mean(i, 'seconds'):.2f} |"
        )
    heads = [f"{r['scene']} {r['view']}" for r in views]
    lines += ["", "## median distance per view (cm)", "", "| variant | " + " | ".join(heads) + " |",
              "|---" * (len(heads) + 1) + "|"]  # fmt: skip
    lines += [f"| {lab} | " + " | ".join(f"{r['variants'][i]['median_cm']:.2f}" for r in views) + " |"
              for i, lab in enumerate(labels)]  # fmt: skip
    lines += ["", "## views", "", "| view | image | scan coverage | Déjà View focal / ETH3D | scale |",
              "|---|---|---|---|---|"]  # fmt: skip
    lines += [f"| {r['scene']} {r['view']} | {r['image']} | {r['coverage']:.0%} | {r['focal_ratio']:.3f} | "
              f"{r['variants'][0]['scale']:.3f} |" for r in views]  # fmt: skip
    return lines


def eval_figures(run: Path, out: Path) -> None:
    """Write the accuracy figures and tables of an evaluation run (`evaluate.py`)."""
    summary = yaml.safe_load((run / "run.yaml").read_text())
    res = dict(np.load(run / "results.npz"))
    views = summary["views"]
    labels = [v["label"] for v in views[0]["variants"]]
    assert all([v["label"] for v in r["variants"]] == labels for r in views)
    assert labels[0] == "E_d (= input)", labels
    out.mkdir(parents=True, exist_ok=True)
    accuracy_curves(out, views, labels, res)
    error_maps(out, views, labels, res)
    scenes = summary["config"]["scenes"]
    intrinsics = [np.load(scenes[r["scene"]]["gt_views"])["intrinsics"][r["view"]] for r in views]
    eval_normals(out, views, labels, res, intrinsics)
    per_view(out, views, labels)
    lines = [f"# {run.name}", "", f"git `{summary['git']}`", "", *eval_tables(views, labels,
             summary["config"]["tolerances_cm"])]  # fmt: skip
    for name in ["accuracy_curves", "per_view", "error_maps", "eval_normals"]:
        lines += ["", f"## {name}", "", CAPTIONS[name], "", f"![{name}]({name}.png)"]
    (out / "README.md").write_text("\n".join([*lines, ""]))


MV_COLUMNS = ["E_d (= input)", "E_d + E_bd + E_n", "E_d + E_mv", "E_d + E_bd + E_n + E_mv"]
# Figure framing, chosen on the raw input's slab before looking at any refined cloud: the wall where the input is
# most layered (meters, in `up_frame` coordinates), and the camera the novel viewpoint starts from.
MV_VIEWS = {
    "office": {"render_view": 10, "render_distance": 2.0, "slice_window": [[-1.3, 0.3], [-3.5, -2.5]]},
    "kicker": {"render_view": 10, "render_distance": 2.0, "slice_window": [[2.9, 4.3], [-1.3, 0.1]]},
}


def fused_clouds(paths: dict, align: dict, res: dict[str, np.ndarray], scene: str, n: int) -> list[np.ndarray]:
    """Each variant's points of all views, (V*H*W, 3), mapped into the scan's frame by the run's camera similarity."""
    rays = load_scene(Path(paths["input"])).rays.double().numpy()
    s, rotation, t = align["scale"], np.array(align["rotation"]), np.array(align["translation"])
    clouds = []
    for i in range(n):
        depth = res[f"{scene}_{i}_depth"].astype(np.float64)
        clouds.append((s * (rays[..., :3] + depth[..., None] * rays[..., 3:]) @ rotation.T + t).reshape(-1, 3))
    return clouds


def up_frame(cam_to_world: np.ndarray) -> np.ndarray:
    """Rows: two horizontal axes and up (the cameras' mean -y, as all photos are taken upright)."""
    up = -cam_to_world[:, :3, 1].mean(0)
    up /= np.linalg.norm(up)
    ahead = cam_to_world[:, :3, 2].mean(0)
    e1 = np.cross(up, ahead)
    e1 /= np.linalg.norm(e1)
    return np.stack([e1, np.cross(up, e1), up])


def grid_planes(points: np.ndarray, chunk: int = 400_000) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the 5 mm voxel grid of a cloud and each grid point's local normal and thickness, as the metrics do."""
    grid = voxel_downsample(points, VOXEL)
    tree = cKDTree(grid)
    planes = [local_planes(tree, grid[i : i + chunk], PLANE_K) for i in range(0, len(grid), chunk)]
    return grid, np.concatenate([p[0] for p in planes]), np.concatenate([p[1] for p in planes])


def splat(points: np.ndarray, values: np.ndarray, k: np.ndarray, world_to_cam: np.ndarray, size: tuple[int, int]
          ) -> np.ndarray:  # fmt: skip
    """Render per-point values by a z-buffer, each point as a 2x2 pixel square; NaN where nothing lands."""
    w, h = size
    cam = points @ world_to_cam[:3, :3].T + world_to_cam[:3, 3]
    front = cam[:, 2] > 0.1  # noqa: PLR2004
    cam, values = cam[front], values[front]
    uv = (cam[:, :2] / cam[:, 2:]) @ k[:2, :2].T + k[:2, 2]
    image, depth = np.full(h * w, np.nan), np.full(h * w, np.inf)
    for du, dv in [(0, 0), (1, 0), (0, 1), (1, 1)]:
        col, row = np.floor(uv[:, 0]).astype(int) + du, np.floor(uv[:, 1]).astype(int) + dv
        inside = (col >= 0) & (col < w) & (row >= 0) & (row < h)
        idx, z, val = row[inside] * w + col[inside], cam[inside, 2], values[inside]
        order = np.argsort(-z)
        idx, z, val = idx[order], z[order], val[order]
        nearer = z < depth[idx]
        image[idx[nearer]], depth[idx[nearer]] = val[nearer], z[nearer]
    return image.reshape(h, w)


def novel_camera(cam_to_world: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    """World-to-camera of a viewpoint 0.8 m behind and 0.5 m above the given camera, looking at the target."""
    eye = cam_to_world[:3, 3] - 0.8 * cam_to_world[:3, 2] + 0.5 * up
    z = target - eye
    z /= np.linalg.norm(z)
    x = np.cross(z, up)
    x /= np.linalg.norm(x)
    rotation = np.stack([x, np.cross(z, x), z])
    return np.r_[np.c_[rotation, -rotation @ eye], [[0, 0, 0, 1]]]


def mv_renders(out: Path, scenes: dict, clouds: dict, cams: dict, labels: list[str]) -> None:
    """Per scene, the fused clouds and the scan from one viewpoint no camera had: headlight shading and thickness."""
    cols = [labels.index(lab) for lab in MV_COLUMNS]
    fig, axes = plt.subplots(2 * len(scenes), len(cols) + 1, figsize=(3.4 * (len(cols) + 1), 2.5 * 2 * len(scenes)),
                             constrained_layout=True)  # fmt: skip
    size = (480, 320)
    k = np.array([[380.0, 0, size[0] / 2], [0, 380.0, size[1] / 2], [0, 0, 1]])
    for r, name in enumerate(scenes):
        c2w = cams[name]["cam_to_world"]
        view = scenes[name]["render_view"]
        target = c2w[view, :3, 3] + scenes[name]["render_distance"] * c2w[view, :3, 2]
        world_to_cam = novel_camera(c2w[view], target, up_frame(c2w)[2])
        panels = [("scan", clouds[name]["scan"]), *((labels[i], clouds[name][i]) for i in cols)]
        for c, (label, (grid, normals, thickness)) in enumerate(panels):
            eye = np.linalg.inv(world_to_cam)[:3, 3]
            view_dir = grid - eye
            view_dir /= np.linalg.norm(view_dir, axis=1, keepdims=True)
            shading = np.abs((normals * view_dir).sum(1))
            axes[2 * r, c].imshow(splat(grid, shading, k, world_to_cam, size), cmap="gray", vmin=0, vmax=1)
            im = axes[2 * r + 1, c].imshow(splat(grid, 1000 * thickness, k, world_to_cam, size), cmap=SEQUENTIAL,
                                           norm=Normalize(0, 6))  # fmt: skip
            median = f"{1000 * np.median(thickness):.1f} mm"
            axes[2 * r, c].set_title(f"{name}: {label}" if c == 0 else label, fontsize=8)
            axes[2 * r + 1, c].set_title(f"thickness, median {median}", fontsize=8)
            for ax in axes[2 * r : 2 * r + 2, c]:
                ax.set_facecolor(GRID)
                ax.set_xticks([])
                ax.set_yticks([])
    fig.colorbar(im, ax=axes[-1, :], location="bottom", shrink=0.4, label="RMS distance to the local plane (mm)")
    fig.savefig(out / "mv_renders.png", dpi=150)
    plt.close(fig)


def mv_slices(out: Path, scenes: dict, points: dict, cams: dict, labels: list[str]) -> None:
    """Per scene, a 2 cm horizontal slab at camera height seen from above: scan in gray, fused cloud in blue."""
    cols = [labels.index(lab) for lab in MV_COLUMNS]
    fig, axes = plt.subplots(len(scenes), len(cols), figsize=(4.2 * len(cols), 4.2 * len(scenes)),
                             constrained_layout=True)  # fmt: skip
    for r, name in enumerate(scenes):
        frame = up_frame(cams[name]["cam_to_world"])
        height = (cams[name]["cam_to_world"][:, :3, 3] @ frame[2]).mean()
        window = scenes[name]["slice_window"]

        def slab(p: np.ndarray, frame: np.ndarray = frame, height: float = height) -> np.ndarray:
            q = p @ frame.T
            return q[np.abs(q[:, 2] - height) < 0.01, :2]  # noqa: PLR2004

        scan = slab(points[name]["scan"])
        for c, i in enumerate(cols):
            ax = axes[r, c]
            ax.scatter(*scan.T, s=0.3, color="#9a9994", linewidths=0, rasterized=True)
            ax.scatter(*slab(points[name][i]).T, s=0.3, color=BLUE, linewidths=0, alpha=0.6, rasterized=True)
            ax.set(xlim=window[0], ylim=window[1], aspect="equal")
            ax.set_title(f"{name}: {labels[i]}" if c == 0 else labels[i], fontsize=9)
            ax.tick_params(labelsize=7)
    fig.supxlabel("meters (horizontal, scan frame)", color=INK_2)
    fig.savefig(out / "mv_slices.png", dpi=150)
    plt.close(fig)


def mv_tables(summary: dict, labels: list[str]) -> list[str]:
    """Markdown tables: the fused cloud per scene, and the per-view protocol over the three views of each scene."""
    tol = summary["config"]["tolerances_cm"]
    lines = ["## fused cloud of all 20 views", "",
             "| scene | variant | median (cm) | mean (cm) | " + " | ".join(f"≤ {t:g} cm" for t in tol[:3])
             + " | thickness (mm) | angle to scan (°) | disagreement (%) | seconds |",
             "|---" * (len(tol[:3]) + 8) + "|"]  # fmt: skip
    for name, sc in summary["scenes"].items():
        for i, v in enumerate(sc["variants"]):
            assert v["label"] == labels[i], (v["label"], labels[i])
            lines.append(f"| {name} | {v['label']} | {v['fused_median_cm']:.2f} | {v['fused_mean_cm']:.2f} | "
                         + " | ".join(f"{100 * v[f'fused_acc_{t:g}cm']:.1f}%" for t in tol[:3])
                         + f" | {v['thickness_mm']:.2f} | {v['normal_angle_deg']:.1f} | {v['disagreement_pct']:.3f}"
                         f" | {v['seconds']:.0f} |")  # fmt: skip
        lines.append(f"| {name} | scan | | | | | | {sc['scan']['thickness_mm']:.2f} | | | |")
    views = summary["config"]["views"]
    lines += ["", f"## per-view protocol: views {views} of each scene, mean over all", "",
              "| variant | median (cm) | mean (cm) | ≤ 1 cm | ≤ 2 cm | median per view (cm) |",
              "|---" * 6 + "|"]  # fmt: skip
    for i, label in enumerate(labels):
        rows = [sc["variants"][i]["per_view"][v] for sc in summary["scenes"].values() for v in views]
        mean = {k: np.mean([r[k] for r in rows]) for k in ("median_cm", "mean_cm", "acc_1cm", "acc_2cm")}
        per = " ".join(f"{r['median_cm']:.2f}" for r in rows)
        lines.append(f"| {label} | {mean['median_cm']:.2f} | {mean['mean_cm']:.2f} | {100 * mean['acc_1cm']:.1f}% | "
                     f"{100 * mean['acc_2cm']:.1f}% | {per} |")  # fmt: skip
    a = summary["scenes"]
    lines += ["", "Camera similarity (Déjà View to ETH3D centers), median residual: "
              + ", ".join(f"{n} {s['center_residual_cm']:.2f} cm" for n, s in a.items()) + "."]  # fmt: skip
    return lines


def mv_figures(run: Path, out: Path) -> None:
    """Write the fused-cloud figures and tables of a joint refinement run (`multiview.py`)."""
    summary = yaml.safe_load((run / "run.yaml").read_text())
    raw, res = summary["config"], dict(np.load(run / "results.npz"))
    labels = [v["label"] for v in next(iter(summary["scenes"].values()))["variants"]]
    out.mkdir(parents=True, exist_ok=True)
    scenes = {name: raw["scenes"][name] | MV_VIEWS[name] for name in summary["scenes"]}
    cams, points, clouds = {}, {}, {}
    for name, paths in scenes.items():
        cams[name] = dict(np.load(paths["gt_views"]))
        cams[name]["cam_to_world"] = cams[name]["cam_to_world"].astype(np.float64)
        fused = fused_clouds(paths, summary["scenes"][name]["align"], res, name, len(labels))
        scan = np.load(paths["scan"]).astype(np.float64)
        points[name] = {"scan": scan, **dict(enumerate(fused))}
        clouds[name] = {"scan": grid_planes(scan)} | {i: grid_planes(fused[i]) for i in
                                                       [labels.index(lab) for lab in MV_COLUMNS]}  # fmt: skip
    mv_renders(out, scenes, clouds, cams, labels)
    mv_slices(out, scenes, points, cams, labels)
    lines = [f"# {run.name}", "", f"git `{summary['git']}`", "", *mv_tables(summary, labels)]
    for name in ["mv_renders", "mv_slices"]:
        lines += ["", f"## {name}", "", CAPTIONS[name], "", f"![{name}]({name}.png)"]
    (out / "README.md").write_text("\n".join([*lines, ""]))


def main() -> None:
    """Render a run: `uv run python -m semantic_3d_refinement.visualize --run runs/<run>`."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    out = args.out or Path("figures") / args.run.name
    config = yaml.safe_load((args.run / "run.yaml").read_text())["config"]
    if "multiview" in config:
        mv_figures(args.run, out)
    elif "scenes" in config:
        eval_figures(args.run, out)
    elif "refine" in config:
        figures(args.run, out)
    else:
        bini_figures(args.run, out)
    print(f"wrote {out}")  # noqa: T201


if __name__ == "__main__":
    main()
