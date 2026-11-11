"""E25a figures: Déjà View's error per class, over what each pixel's geometry predicts.

Reads what `e25a_class_split.py` wrote (`e25a.yaml`, `e25a_maps.npz`) and the bedroom's
cached views.

1. `e25a_class_excess.jpg`: per class (>= 3 scenes), the excess Σr / Σr̂ of each scene as a
   dot and their geometric mean as a bar, on a log axis; Hypersim dev + val with GT classes,
   Hypersim with SAM 3's eroded classes, ETH3D with SAM 3's. Below: the permutation null of
   S with the measured S, and SAM 3's per-class mean log excess against GT's on the same
   pixels.
2. `e25a_bedroom.jpg`: bedroom ai_006_007 views 18, 0, 12, 6: image, GT class, relative
   error r on one % scale, and r / r̂ on one log scale (red: worse than its geometry).

uv run python experiments/scripts/e25a_figures.py runs/e25a_hypersim runs/e25a_eth3d
"""

import argparse
from pathlib import Path

import matplotlib as mpl
import numpy as np
import yaml

mpl.use("Agg")
import matplotlib.pyplot as plt
from e25a_class_split import FAILS, MIN_SCENES, log_excess

from src.analysis.scene_eval import input_paths

FIGURES = Path("experiments/figures")
BEDROOM, VIEWS = "ai_006_007_cam_00", (18, 0, 12, 6)
R_MAX = 0.08  # relative error, shared colour scale of the r maps
RATIO = 4.0  # r / r̂ shown from 1 / RATIO to RATIO
DV, LIGHT, REF, INK = "#52514e", "#9b9a96", "#eb6834", "#0b0b0b"


def style(ax: plt.Axes) -> None:
    """Light grid, no top / right spines."""
    ax.grid(axis="x", color="0.9", lw=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def class_panel(ax: plt.Axes, records: list[dict], key: str, title: str) -> None:
    """Per-scene excess dots and their geometric mean per class, log x axis."""
    per = {c: v for c, v in log_excess(records, key).items() if len(v) >= MIN_SCENES}
    order = sorted(per, key=lambda c: np.mean(per[c]))
    y = np.arange(len(order))
    ax.axvspan(1 / FAILS, FAILS, color="0.94", zorder=0)
    ax.axvline(1, color=LIGHT, lw=0.8)
    ax.barh(
        y,
        [np.exp(np.mean(per[c])) - 1 for c in order],
        left=1,
        color=REF,
        alpha=0.5,
        height=0.6,
    )
    for k, c in enumerate(order):
        ax.plot(np.exp(per[c]), np.full(len(per[c]), k), "o", ms=3, color=DV)
    ax.set_xscale("log")
    ticks = [0.5, 1 / FAILS, 1, FAILS, 2, 3]
    ax.set_xticks(ticks, [f"{t:.2f}" for t in ticks])
    ax.minorticks_off()
    ax.set_xlim(0.33, 3.5)
    ax.set_yticks(y, [f"{c} ({len(per[c])})" for c in order], fontsize=8)
    ax.set_xlabel("excess = Σr / Σr̂ (log; 1 = as hard as its geometry)")
    ax.set_title(title, fontsize=10)
    style(ax)


def excess_figure(hypersim: dict, eth3d: dict) -> None:
    """Write figure 1 (see module)."""
    h = [hypersim["scenes"][n] for n in hypersim["groups"]["all"]]
    e = [eth3d["scenes"][n] for n in eth3d["groups"]["all"]]
    fig = plt.figure(figsize=(15, 13))
    grid = fig.add_gridspec(2, 3, height_ratios=(3, 1.1), hspace=0.3, wspace=0.55)
    class_panel(
        fig.add_subplot(grid[0, 0]),
        h,
        "gt",
        f"Hypersim dev + val ({len(h)} scenes), GT classes",
    )
    class_panel(
        fig.add_subplot(grid[0, 1]), h, "sam3", "Hypersim, SAM 3 classes (eroded 3 px)"
    )
    class_panel(
        fig.add_subplot(grid[0, 2]),
        e,
        "sam3",
        f"ETH3D ({len(e)} scenes), SAM 3 classes (eroded 3 px)",
    )
    summary_panels(
        fig.add_subplot(grid[1, 0]),
        fig.add_subplot(grid[1, 1]),
        h,
        hypersim["summary"],
        "Hypersim",
    )
    fig.suptitle(
        "E25a: Déjà View's relative error per class over the error its pixels' geometry predicts "
        "(edge distance x depth quartile x incidence angle); dots = scenes, bars = geometric mean, grey band = within 1.5x",
        fontsize=11,
    )
    fig.savefig(
        FIGURES / "e25a_class_excess.jpg",
        dpi=100,
        bbox_inches="tight",
        pil_kwargs={"quality": 88},
    )


def summary_panels(
    ax: plt.Axes, right: plt.Axes, records: list[dict], s: dict, name: str
) -> None:
    """Draw the permutation null of S with the measured S, and SAM 3's class excess against GT's."""
    ax.hist(s["null"], bins=40, color=LIGHT)
    ax.axvline(s["S"], color=REF, lw=2, label=f"measured S = {s['S']:.3f}")
    ax.axvline(
        np.quantile(s["null"], 0.95),
        color=DV,
        ls="--",
        lw=1,
        label="null's 95th percentile",
    )
    ax.set_xlabel("S = std over classes of mean log excess")
    ax.set_ylabel("permutations")
    ax.set_title(
        f"{name} GT: class labels permuted among instances, p = {s['p']:.3f}",
        fontsize=10,
    )
    ax.legend(fontsize=8, frameon=False)
    style(ax)

    ax = right
    gt, sam3 = log_excess(records, "gt_on_sam3_px"), log_excess(records, "sam3")
    shared = [
        c for c in gt if c in sam3 and min(len(gt[c]), len(sam3[c])) >= MIN_SCENES
    ]
    x = np.exp([np.mean(gt[c]) for c in shared])
    y = np.exp([np.mean(sam3[c]) for c in shared])
    ax.plot([0.33, 3], [0.33, 3], color=LIGHT, lw=0.8)
    ax.plot(x, y, "o", color=DV, ms=4)
    for c, xi, yi in zip(shared, x, y, strict=True):
        ax.annotate(c, (xi, yi), fontsize=7, xytext=(3, 2), textcoords="offset points")
    ax.set_xscale("log")
    ax.set_yscale("log")
    lim = (min(x.min(), y.min()) * 0.9, max(x.max(), y.max()) * 1.1)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.minorticks_off()
    ax.set_xlabel("excess with GT classes")
    ax.set_ylabel("excess with SAM 3 classes")
    ax.set_title(
        f"same pixels: slope {s['slope']:.2f}, correlation {s['correlation']:.2f}",
        fontsize=10,
    )
    style(ax)


def scene_figure(
    run: Path,
    raw: dict,
    scene: str,
    views: tuple[int, ...],
    row_height: float,
    title: str,
    out: Path,
) -> None:
    """Per view: image, GT class, relative error r and r / r̂, on shared scales."""
    with np.load(run / "e25a_maps.npz") as f:
        r, r_hat, cls = (f[f"{scene}/{k}"] for k in ("r", "r_hat", "gt_class"))
    images = np.load(input_paths(raw, scene)["views"])["images"]
    names = sorted(set(cls[list(views)].ravel()) - {""})
    palette = plt.get_cmap("tab20")(np.arange(len(names)) % 20)
    fig, axes = plt.subplots(len(views), 4, figsize=(16, row_height * len(views)))
    for row, i in enumerate(views):
        colour = np.ones((*cls[i].shape, 3))
        for n, c in enumerate(names):
            colour[cls[i] == c] = palette[n, :3]
        with np.errstate(
            divide="ignore", invalid="ignore"
        ):  # pixels without GT stay NaN
            ratio = r[i] / r_hat[i]
        axes[row, 0].imshow(images[i])
        axes[row, 1].imshow(colour)
        a = axes[row, 2].imshow(100 * r[i], cmap="magma", vmin=0, vmax=100 * R_MAX)
        b = axes[row, 3].imshow(
            ratio, cmap="RdBu_r", norm=mpl.colors.LogNorm(1 / RATIO, RATIO)
        )
        axes[row, 0].set_ylabel(f"view {i}")
        for ax in axes[row]:
            ax.set_xticks([])
            ax.set_yticks([])
    for ax, t in zip(
        axes[0],
        (
            "image",
            "GT class",
            "relative error r = e / depth",
            "r / r̂ (geometry's prediction)",
        ),
        strict=True,
    ):
        ax.set_title(t, fontsize=10)
    fig.colorbar(a, ax=axes[:, 2], fraction=0.03, label="%")
    bar = fig.colorbar(b, ax=axes[:, 3], fraction=0.03, label="ratio, log")
    ticks = [1 / RATIO, 1 / 2, 1, 2, RATIO]
    bar.set_ticks(ticks, labels=[f"{t:g}" for t in ticks])
    bar.minorticks_off()
    handles = [
        mpl.patches.Patch(color=palette[n], label=c) for n, c in enumerate(names)
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=min(len(names), 9),
        fontsize=8,
        frameon=False,
    )
    fig.suptitle(title, fontsize=11)
    fig.savefig(out, dpi=90, bbox_inches="tight", pil_kwargs={"quality": 88})


def main() -> None:
    """Write both figures; see the module docstring."""
    parser = argparse.ArgumentParser()
    parser.add_argument("hypersim", type=Path)
    parser.add_argument("eth3d", type=Path)
    args = parser.parse_args()
    hypersim, eth3d = (
        yaml.safe_load((p / "e25a.yaml").read_text(encoding="utf-8"))
        for p in (args.hypersim, args.eth3d)
    )
    excess_figure(hypersim, eth3d)
    raw = yaml.safe_load(Path(hypersim["config"]).read_text(encoding="utf-8"))
    scene_figure(
        args.hypersim,
        raw,
        BEDROOM,
        VIEWS,
        3.1,
        "E25a, bedroom ai_006_007: Déjà View's relative error, and the same over what each pixel's "
        "edge distance, depth and incidence angle predict (red = worse than its geometry)",
        FIGURES / "e25a_bedroom.jpg",
    )


if __name__ == "__main__":
    main()
