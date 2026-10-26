"""Figures for one refinement run: depth, error, residual, profile, point cloud, energy, metrics table."""

import argparse
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
SEQUENTIAL = LinearSegmentedColormap.from_list(
    "seq_blue", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
)
DIVERGING = LinearSegmentedColormap.from_list("div", ["#104281", "#6da7ec", "#f0efec", "#ec8a89", "#b42d2d"])

plt.rcParams |= {
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID,
    "axes.labelcolor": INK_2,
    "axes.titlecolor": INK,
    "axes.titlesize": 11,
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


def image_row(  # noqa: PLR0913
    images: list[np.ndarray], titles: list[str], cmap: LinearSegmentedColormap, label: str, *, vmin: float, vmax: float
) -> Figure:
    """Images side by side on one shared color scale."""
    fig, axes = plt.subplots(1, len(images), figsize=(3.6 * len(images), 3.2), constrained_layout=True, squeeze=False)
    axes = axes[0]
    for ax, img, title in zip(axes, images, titles, strict=True):
        im = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.set_axis_off()
    fig.colorbar(im, ax=axes, shrink=0.85, label=label)
    return fig


def figures(run: Path, out: Path) -> dict[str, float]:  # noqa: PLR0915
    """Write all figures for a run directory into `out`; return the metric table."""
    summary = yaml.safe_load((run / "run.yaml").read_text())
    data = np.load(summary["config"]["input"])
    refined = np.load(run / "refined.npz")
    depth, rays, gt = data["depth"], data["rays"], data.get("gt_depth")
    depth_hat, delta, energy = refined["depth"], refined["delta"], refined["energy"]
    assert depth.shape == depth_hat.shape == delta.shape
    out.mkdir(parents=True, exist_ok=True)

    images, titles = [depth, depth_hat], ["input", "refined"]
    if gt is not None:
        images, titles = [*images, gt], [*titles, "ground truth"]
    lo, hi = np.percentile(np.concatenate([i.ravel() for i in images]), [1, 99])
    image_row(images, titles, SEQUENTIAL, "depth", vmin=float(lo), vmax=float(hi)).savefig(out / "depth.png", dpi=150)

    if gt is not None:
        err_in, err_out = np.abs(depth - gt) / gt, np.abs(depth_hat - gt) / gt
        titles = [f"input  (mean {err_in.mean():.2%})", f"refined  (mean {err_out.mean():.2%})"]
        vmax = np.percentile(err_in, 99)
        fig = image_row([err_in, err_out], titles, SEQUENTIAL, "|D - D_gt| / D_gt", vmin=0.0, vmax=float(vmax))
        fig.savefig(out / "error.png", dpi=150)

    m = float(np.percentile(np.abs(delta), 99))
    fig = image_row(
        [delta], ["residual δ, clipped at p99 of |δ|"], DIVERGING, "δ (normalized inverse depth)", vmin=-m, vmax=m
    )
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
    pts = {k: (rays[..., :3] + d[..., None] * rays[..., 3:]).reshape(-1, 3) for k, d in clouds.items()}
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

    fig, ax = plt.subplots(figsize=(6, 3.2), constrained_layout=True)
    ax.semilogy(np.arange(len(energy)), energy, color=BLUE)
    ax.set(title="energy E during L-BFGS", xlabel="function evaluation", ylabel="E (log)")
    fig.savefig(out / "energy.png", dpi=150)
    plt.close("all")

    table = {"refine_seconds": summary["refine_seconds"], "E_initial": energy[0], "E_final": energy[-1]}
    if "metrics_input" in summary:
        for k in summary["metrics_input"]:
            table |= {f"{k}_input": summary["metrics_input"][k], f"{k}_refined": summary["metrics_refined"][k]}
    lines = [f"# {run.name}", "", f"git `{summary['git']}`", "", "| metric | value |", "|---|---|"]
    lines += [f"| {k} | {v:.4g} |" for k, v in table.items()]
    lines += ["", *[f"![{p.stem}]({p.name})" for p in sorted(out.glob("*.png"))], ""]
    (out / "README.md").write_text("\n".join(lines))
    return table


def main() -> None:
    """Render a run: `uv run python -m semantic_3d_refinement.visualize --run runs/<run>`."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    out = args.out or Path("figures") / args.run.name
    figures(args.run, out)
    print(f"wrote {out}")  # noqa: T201


if __name__ == "__main__":
    main()
