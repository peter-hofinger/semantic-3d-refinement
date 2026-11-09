"""E37 figures: does dropping Déjà View's confidence from E_d take the completeness loss away.

Reads the two E37 runs and what `e35_error_split.py --variants 0 1` / `0 2` wrote into
them (`e35_split_0_{1,2}.yaml`, `e35_maps_0_{1,2}.npz`; variant 1 is the refiner as it
is, p = 1, variant 2 the refiner with conf_power p = 0).

1. `e37_confidence.jpg`: per scene group, the completeness change against Déjà View of
   both refiners, stacked by Déjà View confidence quartile (cm; quartiles per scene,
   bars sum to the change; scenes weighted equally).
2. `e37_coverage_<scene>.jpg`: per view, Déjà View's confidence, each GT pixel's
   completeness change against Déjà View for p = 1 and p = 0 on one cm scale, and p = 0
   minus p = 1 on a finer one. Bedroom ai_006_007 views 18, 0, 12, 6 and ETH3D office
   views 0, 5, 10, 15, as E35. ETH3D data: CC BY-NC-SA 4.0, Schöps et al., CVPR 2017.

uv run python experiments/scripts/e37_figures.py runs/e37_hypersim-20261109-162959 runs/e37_eth3d-20261109-163228
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
        "ETH3D office (CC BY-NC-SA 4.0, Schöps et al.)",
    ),
}
CHANGE_MAX = 5.0  # cm, refined - Déjà View
DIFF_MAX = 1.0  # cm, p = 0 - p = 1
REFINERS = (("0_1", "p = 1 (as is)"), ("0_2", "p = 0 (no confidence)"))
INK = "#0b0b0b"


def group_mean(records: list[dict], path: tuple) -> float:
    """Mean over scenes of one nested value."""
    values = []
    for record in records:
        value = record
        for key in path:
            value = value[key]
        values.append(value)
    return float(np.nanmean(values))


def confidence_figure(splits: dict[tuple[str, str], dict]) -> None:
    """Write figure 1 (see module) and print the numbers it shows."""
    x = np.arange(len(GROUPS))
    quartiles = ["Q1 (low)", "Q2", "Q3", "Q4 (high)"]
    shades = plt.get_cmap("Oranges")(np.linspace(0.85, 0.3, len(quartiles)))
    fig, ax = plt.subplots(figsize=(12, 5.2))
    width = 0.36
    for b, (pair, label) in enumerate(REFINERS):
        bottom = np.zeros(len(GROUPS))
        offset = (b - 0.5) * (width + 0.04)
        totals = []
        for data, g, _ in GROUPS:
            split = splits[data, pair]
            records = [split["scenes"][n] for n in split["groups"][g]]
            totals.append(group_mean(records, ("completeness", "change_cm")))
        for k, q in enumerate(quartiles):
            values = []
            for data, g, _ in GROUPS:
                split = splits[data, pair]
                records = [split["scenes"][n] for n in split["groups"][g]]
                values.append(
                    group_mean(records, ("completeness", "by_confidence", q, 1))
                )
            ax.bar(
                x + offset,
                values,
                width,
                bottom=bottom,
                color=shades[k],
                edgecolor="white",
                hatch="" if b == 0 else "//",
                label=f"{q}" if b == 0 else None,
            )
            bottom += np.array(values)
        for k, t in enumerate(totals):
            ax.text(x[k] + offset, t, f"{t:+.2f}", ha="center", va="bottom", fontsize=9)
            ax.text(x[k] + offset, -0.12, label.split(" (")[0], ha="center", fontsize=8)
        print(
            f"{label}: "
            + ", ".join(f"{g[2]} {t:+.2f}" for g, t in zip(GROUPS, totals, strict=True))
        )
    ax.set_xticks(x, [g[2] for g in GROUPS])
    ax.tick_params(axis="x", pad=14)
    ax.set_ylabel("completeness change against Déjà View (cm)")
    ax.legend(
        title="Déjà View confidence quartile",
        fontsize=8,
        frameon=False,
        loc="upper left",
    )
    ax.grid(axis="y", color="0.9", lw=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_title(
        "E37: the joint refiner's completeness loss with E_d weighted by confidence (p = 1, plain) and without (p = 0, hatched).\n"
        "20 views at 504 px, protocol alignment, every weight 1, scenes weighted equally. Dropping the confidence removes 1-3% of the loss;\n"
        "the lowest-confidence quarter keeps its share. E_d is at most a third of the per-pixel anchor next to the unweighted λ_δ‖δ‖².",
        fontsize=10,
        color=INK,
    )
    fig.tight_layout()
    fig.savefig(FIGURES / "e37_confidence.jpg", dpi=110, pil_kwargs={"quality": 90})
    plt.close(fig)


def coverage_figure(runs: dict[str, Path], scene: str) -> None:
    """Write figure 2 (see module) for one scene."""
    data, shown, title = MAPS[scene]
    run = runs[data]
    summary = yaml.safe_load((run / "run.yaml").read_text(encoding="utf-8"))
    confidence = Geometry.load(
        input_paths(summary["config"], scene)["geometry"]
    ).confidence
    comps = {}
    for pair, _ in REFINERS:
        with np.load(run / f"e35_maps_{pair}.npz") as f:
            comps["dv"] = f[f"{scene}/comp/0"]
            comps[pair] = f[f"{scene}/comp/1"]
    one, zero = 100 * (comps["0_1"] - comps["dv"]), 100 * (comps["0_2"] - comps["dv"])
    finite = np.isfinite(one) & np.isfinite(zero)
    low = finite & (confidence <= np.quantile(confidence[finite], 0.25))
    cols = (
        "Déjà View confidence",
        "p = 1 - Déjà View",
        "p = 0 - Déjà View",
        "p = 0 - p = 1",
    )
    fig, axes = plt.subplots(
        len(shown), 4, figsize=(13, 2.4 * len(shown) + 1.0), layout="constrained"
    )
    for row, i in enumerate(shown):
        conf = axes[row, 0].imshow(
            confidence[i],
            cmap="viridis",
            vmin=np.quantile(confidence, 0.02),
            vmax=np.quantile(confidence, 0.98),
        )
        change = axes[row, 1].imshow(
            one[i], cmap="RdBu_r", vmin=-CHANGE_MAX, vmax=CHANGE_MAX
        )
        axes[row, 2].imshow(zero[i], cmap="RdBu_r", vmin=-CHANGE_MAX, vmax=CHANGE_MAX)
        diff = axes[row, 3].imshow(
            zero[i] - one[i], cmap="PiYG_r", vmin=-DIFF_MAX, vmax=DIFF_MAX
        )
        axes[row, 0].set_ylabel(f"view {i}", fontsize=10)
        for c, ax in enumerate(axes[row]):
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(cols[c], fontsize=10)
    fig.colorbar(
        conf, ax=axes[:, 0], fraction=0.05, label="confidence (2-98% of the scene)"
    )
    fig.colorbar(
        change,
        ax=axes[:, 1:3],
        fraction=0.03,
        label="cm, red = covers worse than Déjà View",
    )
    fig.colorbar(
        diff, ax=axes[:, 3], fraction=0.05, label="cm, green = p = 0 covers better"
    )
    lines = [
        f"{label}: {np.mean(m[finite]):+.2f} cm overall, {np.mean(m[low]):+.2f} on the lowest-confidence quarter"
        for m, (_, label) in zip((one, zero), REFINERS, strict=True)
    ]
    fig.suptitle(
        f"E37, {title}: change of each GT pixel's completeness distance against Déjà View (20 views, protocol alignment); white = no GT.\n"
        + "; ".join(lines)
        + ".\nThe same walls and ceilings turn red with and without the confidence weight; p = 0 - p = 1 is near zero everywhere.",
        fontsize=10,
        color=INK,
    )
    out = FIGURES / f"e37_coverage_{scene.removesuffix('_cam_00')}.jpg"
    fig.savefig(out, dpi=100, pil_kwargs={"quality": 88})
    plt.close(fig)
    print(f"{scene}: " + "; ".join(lines))


def main() -> None:
    """Write both figures; see the module docstring."""
    parser = argparse.ArgumentParser()
    parser.add_argument("hypersim", type=Path)
    parser.add_argument("eth3d", type=Path)
    args = parser.parse_args()
    runs = {"hypersim": args.hypersim, "eth3d": args.eth3d}
    splits = {
        (d, pair): yaml.safe_load(
            (r / f"e35_split_{pair}.yaml").read_text(encoding="utf-8")
        )
        for d, r in runs.items()
        for pair, _ in REFINERS
    }
    confidence_figure(splits)
    for scene in MAPS:
        coverage_figure(runs, scene)


if __name__ == "__main__":
    main()
