"""E35 figures: the error split across datasets, and where the refiner loses coverage.

Reads what `e35_error_split.py` wrote into the Hypersim and the ETH3D run
(`e35_split.yaml`, `e35_maps.npz`) and the scenes' cached views and confidence.

1. `e35_error_split.jpg`: per scene group (Hypersim dev / val / stress, ETH3D indoor /
   outdoor; scenes weighted equally): Déjà View's per-pixel error with one Sim(3) per
   scene vs one per view; the share of its error vs of its pixels within 3 px of a GT
   depth edge; the refiner's completeness change by Déjà View confidence quartile and
   by edge-distance bin (contributions in cm, they sum to the change).
2. `e35_coverage_<scene>.jpg`: per view, the image, Déjà View's confidence, each GT
   pixel's completeness distance (to the nearest predicted point) for Déjà View and
   the refiner on one cm scale, and their difference. Bedroom ai_006_007 views 18, 0,
   12, 6 (the presentation views); ETH3D office views 0, 5, 10, 15 (fixed before
   looking). ETH3D images: CC BY-NC-SA 4.0, Schöps et al., CVPR 2017.

uv run python experiments/scripts/e35_figures.py runs/eval_hypersim-20261109-145132 runs/eval_eth3d-20261109-152420
"""

import argparse
from pathlib import Path

import matplotlib as mpl
import numpy as np
import yaml

mpl.use("Agg")
import matplotlib.pyplot as plt

from src.analysis.scene_eval import input_paths
from src.geometry import Geometry
from src.loaders.views import Views

FIGURES = Path("experiments/figures")
GROUPS = (
    ("hypersim", "dev", "Hypersim dev"),
    ("hypersim", "val", "Hypersim val"),
    ("hypersim", "stress", "Hypersim stress"),
    ("eth3d", "indoor", "ETH3D indoor"),
    ("eth3d", "outdoor", "ETH3D outdoor"),
)
MAPS = {
    "ai_006_007_cam_00": ("hypersim", (18, 0, 12, 6), "Hypersim bedroom ai_006_007"),
    "office": (
        "eth3d",
        (0, 5, 10, 15),
        "ETH3D office (data: ETH3D, CC BY-NC-SA 4.0, Schöps et al. 2017)",
    ),
}
COMP_MAX = 0.10  # m, shared colour scale of the completeness maps
CHANGE_MAX = 0.05  # m, of their difference
DV, LIGHT, REF, INK = "#52514e", "#9b9a96", "#eb6834", "#0b0b0b"


def group_mean(records: list[dict], path: tuple) -> float:
    """Mean over scenes of one nested value."""
    values = []
    for record in records:
        value = record
        for key in path:
            value = value[key]
        values.append(value)
    return float(np.nanmean(values))


def style(ax: plt.Axes) -> None:
    """Light grid, no top / right spines."""
    ax.grid(axis="y", color="0.9", lw=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def near_share(records: list[dict], path: tuple, total: tuple) -> float:
    """Share of a group's value that sits within 3 px of a GT depth edge."""
    near = sum(group_mean(records, (*path, b, 1)) for b in ("<=1 px", "1-3 px"))
    return near / group_mean(records, total)


def labelled_bars(
    ax: plt.Axes, x: np.ndarray, series: list[tuple[str, list[float], str]]
) -> None:
    """Draw grouped bars, one per series, centred on each x."""
    width = 0.8 / len(series)
    for b, (label, values, colour) in enumerate(series):
        ax.bar(x - 0.4 + width * (b + 0.5), values, width, color=colour, label=label)


def split_figure(splits: dict[str, dict]) -> None:
    """Write figure 1 (see module) and print the numbers it shows."""
    groups = [
        (label, [splits[data]["scenes"][n] for n in splits[data]["groups"][g]])
        for data, g, label in GROUPS
    ]
    x = np.arange(len(groups))
    names = [f"{label}\n({len(r)} scenes)" for label, r in groups]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9.5))

    ax = axes[0, 0]
    scene = [group_mean(r, ("variants", 0, "error_cm")) for _, r in groups]
    view = [group_mean(r, ("variants", 0, "error_per_view_cm")) for _, r in groups]
    labelled_bars(
        ax,
        x,
        [
            ("one Sim(3) per scene (protocol)", scene, DV),
            ("one Sim(3) per view", view, LIGHT),
        ],
    )
    for k in range(len(groups)):
        ax.text(
            x[k],
            max(scene[k], view[k]),
            f"{view[k] / scene[k] - 1:+.0%}",
            ha="center",
            va="bottom",
            fontsize=9,
            color=INK,
        )
    ax.set_title(
        "Camera / global: Déjà View's per-pixel error (cm)\nlabel: change when every view is aligned on its own",
        fontsize=10,
    )

    ax = axes[0, 1]
    near_px = [
        sum(group_mean(r, ("variants", 0, "edges", b, 0)) for b in ("<=1 px", "1-3 px"))
        for _, r in groups
    ]
    near_err = [
        near_share(r, ("variants", 0, "edges"), ("variants", 0, "error_cm"))
        for _, r in groups
    ]
    near_loss = [
        near_share(r, ("completeness", "by_edge"), ("completeness", "change_cm"))
        for _, r in groups
    ]
    labelled_bars(
        ax,
        x,
        [
            ("share of pixels", near_px, LIGHT),
            ("share of Déjà View's error", near_err, DV),
            ("share of the completeness loss", near_loss, REF),
        ],
    )
    ax.set_title(
        "Edges: what lies within 3 px of a GT depth edge\n(ETH3D's laser GT is dotted, so its edges are under-counted)",
        fontsize=10,
    )
    ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0))

    ax = axes[1, 0]
    quartiles = list(groups[0][1][0]["completeness"]["by_confidence"])
    shades = plt.get_cmap("Oranges")(np.linspace(0.85, 0.3, len(quartiles)))
    labelled_bars(
        ax,
        x,
        [
            (
                q,
                [
                    group_mean(r, ("completeness", "by_confidence", q, 1))
                    for _, r in groups
                ],
                shades[b],
            )
            for b, q in enumerate(quartiles)
        ],
    )
    totals = [group_mean(r, ("completeness", "change_cm")) for _, r in groups]
    ax.set_xticks(
        x, [f"{n}\ntotal {t:+.2f} cm" for n, t in zip(names, totals, strict=True)]
    )
    ax.set_title(
        "Completeness change, refined - Déjà View, by Déjà View confidence quartile (cm)\nquartiles per scene, equal pixel counts; bars sum to the total",
        fontsize=10,
    )

    ax = axes[1, 1]
    objects = [
        (
            f"{src} instances, {when}",
            [group_mean(r, ("variants", 0, f"{key}_{col}")) for _, r in groups],
            colour,
        )
        for src, key, cols in (
            ("GT", "gt", ("object_cm", "fit_cm")),
            ("SAM 3", "sam3", ("object_cm", "fit_cm")),
        )
        for when, col, colour in zip(
            ("scene Sim(3)", "+ own scale and shift"),
            cols,
            (DV, LIGHT) if key == "gt" else (REF, "#f4b393"),
            strict=True,
        )
    ]
    labelled_bars(ax, x, objects)
    ax.set_title(
        "Objects: Déjà View's error on objects >= 200 px (cm)\nbefore / after fitting each its own scale and shift (ETH3D: no GT instances)",
        fontsize=10,
    )

    for ax in axes.flat:
        if ax is not axes[1, 0]:
            ax.set_xticks(x, names)
        ax.tick_params(axis="x", labelsize=9)
        ax.set_ylim(0, ax.get_ylim()[1] * 1.25)  # room for the legend
        ax.legend(fontsize=8, frameon=False, ncol=2, loc="upper left")
        style(ax)
    axes[0, 1].yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0))
    fig.suptitle(
        "E35: where Déjà View's error and the joint refiner's completeness loss come from (20 views at 504 px, protocol alignment, "
        "every refiner weight 1; scenes weighted equally).\nAligning each view on its own removes 4-7% of the error on Hypersim, "
        "20-25% on ETH3D. Within 3 px of a depth edge lie 42-48% of Déjà View's error on Hypersim but only 20-28% of the\n"
        "completeness loss, which follows low confidence instead. The object fit depends on how the segments are cut (SAM 3 vs GT "
        "instances on Hypersim), so it is no clean placement measure.",
        fontsize=10,
        color=INK,
    )
    fig.tight_layout()
    fig.savefig(FIGURES / "e35_error_split.jpg", dpi=110, pil_kwargs={"quality": 90})
    plt.close(fig)
    for (label, _), s, v, p, e, c in zip(
        groups, scene, view, near_px, near_err, near_loss, strict=True
    ):
        print(
            f"{label}: error {s:.2f} -> per-view {v:.2f} cm; <=3 px: {p:.3f} of px, {e:.3f} of error, {c:.3f} of comp loss"
        )


def coverage_figure(
    run: Path, scene: str, views_shown: tuple[int, ...], title: str
) -> None:
    """Write figure 2 (see module) for one scene."""
    summary = yaml.safe_load((run / "run.yaml").read_text(encoding="utf-8"))
    paths = input_paths(summary["config"], scene)
    views = Views.load(paths["views"])
    confidence = Geometry.load(paths["geometry"]).confidence
    with np.load(run / "e35_maps.npz") as f:
        dv, ref = f[f"{scene}/comp/0"], f[f"{scene}/comp/1"]
    cols = (
        "image",
        "Déjà View confidence",
        "completeness distance, Déjà View",
        "completeness distance, refined",
        "refined - Déjà View",
    )
    finite = np.isfinite(dv) & np.isfinite(ref)
    low = finite & (confidence <= np.quantile(confidence[finite], 0.25))
    change_all = 100 * np.mean(ref[finite] - dv[finite])
    change_low = 100 * np.mean(ref[low] - dv[low])
    change_rest = 100 * np.mean((ref - dv)[finite & ~low])
    fig, axes = plt.subplots(
        len(views_shown),
        5,
        figsize=(16, 2.4 * len(views_shown) + 0.9),
        layout="constrained",
    )
    for row, i in enumerate(views_shown):
        conf = axes[row, 1].imshow(
            confidence[i],
            cmap="viridis",
            vmin=np.quantile(confidence, 0.02),
            vmax=np.quantile(confidence, 0.98),
        )
        axes[row, 0].imshow(views.images[i])
        comp = axes[row, 2].imshow(
            100 * dv[i], cmap="magma", vmin=0, vmax=100 * COMP_MAX
        )
        axes[row, 3].imshow(100 * ref[i], cmap="magma", vmin=0, vmax=100 * COMP_MAX)
        change = axes[row, 4].imshow(
            100 * (ref[i] - dv[i]),
            cmap="RdBu_r",
            vmin=-100 * CHANGE_MAX,
            vmax=100 * CHANGE_MAX,
        )
        axes[row, 0].set_ylabel(f"view {i}", fontsize=10)
        for c, ax in enumerate(axes[row]):
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(cols[c], fontsize=10)
    fig.colorbar(
        conf, ax=axes[:, 1], fraction=0.05, label="confidence (2-98% of the scene)"
    )
    fig.colorbar(comp, ax=axes[:, 2:4], fraction=0.03, label="cm (clipped at 10)")
    fig.colorbar(
        change, ax=axes[:, 4], fraction=0.05, label="cm, red = refined covers worse"
    )
    fig.suptitle(
        f"E35, {title}: each GT pixel's distance to the nearest predicted point (the completeness score, before averaging), "
        f"after the protocol's alignment; white in the distance maps = no GT.\nRefined - Déjà View over all 20 views: {change_all:+.2f} cm; on the "
        f"lowest-confidence quarter of pixels {change_low:+.2f} cm, on the rest {change_rest:+.2f} cm. Red spreads over whole "
        "walls, doors and ceilings,\nnot in thin bands along depth edges: the refined surfaces move off the GT as a whole.",
        fontsize=10,
        color=INK,
    )
    out = FIGURES / f"e35_coverage_{scene.removesuffix('_cam_00')}.jpg"
    fig.savefig(out, dpi=100, pil_kwargs={"quality": 88})
    plt.close(fig)
    print(
        f"{scene}: change {change_all:+.2f} cm; low-confidence quarter {change_low:+.2f} cm, rest {change_rest:+.2f} cm"
    )


def main() -> None:
    """Write both figures; see the module docstring."""
    parser = argparse.ArgumentParser()
    parser.add_argument("hypersim", type=Path)
    parser.add_argument("eth3d", type=Path)
    args = parser.parse_args()
    runs = {"hypersim": args.hypersim, "eth3d": args.eth3d}
    splits = {
        d: yaml.safe_load((r / "e35_split.yaml").read_text(encoding="utf-8"))
        for d, r in runs.items()
    }
    split_figure(splits)
    for scene, (data, shown, title) in MAPS.items():
        coverage_figure(runs[data], scene, shown, title)


if __name__ == "__main__":
    main()
