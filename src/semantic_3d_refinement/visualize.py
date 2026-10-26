"""Figures for one refinement run: results, what each energy term does, and how each weight trades off."""

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

from semantic_3d_refinement.normals import BiniConfig, bini, normals_from_depth
from semantic_3d_refinement.refine import (
    LOG_KEYS,
    RefineConfig,
    energy_maps,
    normalize,
    points_from_rays,
    refine,
    term_weights,
    view_from_config,
)

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
TERM_COLORS = {"E_d": ORANGE, "E_bd": BLUE, "E_delta": AQUA}
TERM_NAMES = {"E_d": "data  E_d", "E_bd": "smoothness  λ_bd·E_bd", "E_delta": "residual prior  λ_δ·‖δ‖²"}
RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SEQUENTIAL = LinearSegmentedColormap.from_list("seq_blue", RAMP)
DIVERGING = LinearSegmentedColormap.from_list("div", ["#104281", "#6da7ec", "#f0efec", "#ec8a89", "#b42d2d"])

CAPTIONS = {
    "depth": "Depth maps on one color scale: input, refined, and ground truth when known.",
    "error": "Relative depth error per pixel before and after refinement. What remains sits on the creases.",
    "delta": "The dense residual δ that refinement adds to the normalized inverse depth.",
    "profile": "Depth along one image row through a crease.",
    "points_topdown": "Point clouds seen from above (x-z).",
    "normal_sources": "Normals of the view: Déjà View's depth, DSINE's prediction (rotated into the same frame), "
    "and the depth after BiNI integrates DSINE's normals against Déjà View's depth.",
    "detail_crops": "The same three, zoomed where DSINE's normals carry the most detail.",
    "compare": "One view under several BiNI prior settings (sigma = blur of the depth prior in px, λ = its weight); "
    "bottom row zoomed where DSINE has the most detail. In brackets: median angle to DSINE's normals.",
    "energy_terms": "Weighted contribution of each energy term during L-BFGS: smoothness cost is bought down by "
    "spending data cost.",
    "cost_maps": "Where each term pays, per pixel (log scale).",
    "forces": "Tug of war at convergence: the force -∂E/∂δ each term exerts per pixel. Data pulls back toward "
    "the input, smoothness pushes toward planes, and at convergence they cancel (net ≈ 0).",
    "sweep_lambda_bd": "Effect of the smoothness weight λ_bd: too low keeps the noise, too high rounds off the "
    "creases.",
    "data_term_ablation": "Effect of the data term's form on the 1% outliers: an unweighted quadratic lets them "
    "drag their neighbors into blobs. Confidence weighting alone removes them here; Huber only matters without it.",
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


def term_figures(out: Path, depth: np.ndarray, conf: np.ndarray, delta: np.ndarray, cfg: RefineConfig) -> None:
    """Per-pixel cost of each term and the force each exerts at the solution."""
    _, d0, w = normalize(torch.from_numpy(depth), torch.from_numpy(conf))
    weights = term_weights(cfg)
    delta_t = torch.from_numpy(delta).requires_grad_()
    start = {k: weights[k] * v for k, v in energy_maps(torch.zeros_like(d0), d0, w, cfg).items()}
    end = {k: weights[k] * v for k, v in energy_maps(delta_t, d0, w, cfg).items()}
    active = [k for k in ("E_d", "E_bd", "E_delta") if weights[k] > 0]

    maps = [start["E_bd"], end["E_bd"], *(end[k] for k in active if k != "E_bd")]
    titles = [
        "smoothness, before",
        "smoothness, after",
        *(f"{TERM_NAMES[k].split('  ')[0]}, after" for k in active if k != "E_bd"),
    ]
    imgs = [m.detach().numpy() for m in maps]
    hi = float(np.percentile(imgs[0], 99.5))
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
    """Refine the run's input with a modified config; return the refined depth."""
    conf = data["conf"] if use_conf else torch.ones_like(data["conf"])
    return refine(data["depth"], data["rays"], conf, cfg)[0].cpu().numpy()


def sweep_lambda_bd(out: Path, depth: np.ndarray, gt: np.ndarray, lams: list[float], refined: list[np.ndarray]) -> None:
    """Error maps per smoothness weight above depth profiles across a crease."""
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
    fig.suptitle("smoothness weight: noise ↔ rounded creases", color=INK)
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
    labels = {"lambda_bd": "smoothness weight λ_bd", "huber": "Huber threshold κ", "lambda_delta": "residual prior λ_δ"}
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

    overview(out, depth, rays, gt, refined)
    energy_terms(out, log, cfg)
    term_figures(out, depth, raw["conf"], refined["delta"], cfg)
    sweep = summary["config"].get("sweep")
    if gt is not None and sweep:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        gpu = {k: v.to(device) for k, v in data.items()}
        sweep_figures(out, gpu, dataclasses.replace(cfg, device=device.type), sweep)

    table = {"refine_seconds": summary["refine_seconds"], "E_initial": log[0, 0], "E_final": log[-1, 0]}
    return write_readme(run, out, summary, table, list(CAPTIONS))


def write_readme(run: Path, out: Path, summary: dict, table: dict[str, float], names: list[str]) -> dict[str, float]:
    """Write the metric table and every existing figure with its caption; return the table."""
    if "metrics_input" in summary:
        for k in summary["metrics_input"]:
            table |= {f"{k}_input": summary["metrics_input"][k], f"{k}_refined": summary["metrics_refined"][k]}
    lines = [f"# {run.name}", "", f"git `{summary['git']}`", "", "| metric | value |", "|---|---|"]
    lines += [f"| {k} | {v:.4g} |" for k, v in table.items()]
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
        panels.append((f"sigma = {cfg.prior_sigma:g}, λ = {cfg.lambda_depth:g}  ({angle:.1f}°)", img))
    win = detail_window(panels[2][1], 112)
    fig, axes = plt.subplots(2, len(panels), figsize=(3.3 * len(panels), 5.2), constrained_layout=True)
    for c, (title, img) in enumerate(panels):
        for r, w in enumerate(((slice(None), slice(None)), win)):
            axes[r, c].imshow(img[w], interpolation="nearest")
            axes[r, c].set_axis_off()
        axes[0, c].set_title(title, fontsize=8)
    fig.suptitle("BiNI prior settings, one view (° = median angle to DSINE normals)", color=INK)
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

    def rgb(d: np.ndarray) -> np.ndarray:
        return normals_from_depth(torch.from_numpy(d), torch.from_numpy(data["rays"])).numpy() * 0.5 + 0.5

    panels = [
        (f"view {raw['view']}", image),
        ("Déjà View", rgb(data["depth"])),
        ("DSINE", refined["normals"] * 0.5 + 0.5),
        ("BiNI", rgb(refined["depth"])),
    ]
    for name, crop in (("normal_sources", None), ("detail_crops", 112)):
        win = detail_window(panels[2][1], crop) if crop else (slice(None), slice(None))
        fig, axes = plt.subplots(1, 4, figsize=(13, 2.7), constrained_layout=True)
        for ax, (title, img) in zip(axes, panels, strict=True):
            ax.imshow(img[win], interpolation="nearest")
            ax.set_title(title)
            ax.set_axis_off()
        fig.savefig(out / f"{name}.png", dpi=150)
        plt.close(fig)

    table = {"normals_seconds": summary["normals_seconds"], "refine_seconds": summary["refine_seconds"]}
    table["grid_score_dsine"] = summary["grid_score_normals"]
    return write_readme(run, out, summary, table, ["compare", "normal_sources", "detail_crops"])


def main() -> None:
    """Render a run: `uv run python -m semantic_3d_refinement.visualize --run runs/<run>`."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    out = args.out or Path("figures") / args.run.name
    if "bini" in yaml.safe_load((args.run / "run.yaml").read_text())["config"]:
        bini_figures(args.run, out)
    else:
        figures(args.run, out)
    print(f"wrote {out}")  # noqa: T201


if __name__ == "__main__":
    main()
