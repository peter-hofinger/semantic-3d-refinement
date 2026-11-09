"""E38 figures: does completeness reward the spread of Déjà View's views.

Reads what `e38_spread.py` wrote into the two E37 runs (`e38_spread_0_1.yaml`,
`e38_maps_0_1.npz`; variant 0 is Déjà View, 1 the joint refiner as it is).

1. `e38_split.jpg`: per scene group, the completeness change ΔC_f split into ΔC_o (each
   view's own cloud covers its own GT worse) and -ΔB (the other views cover for it less);
   next to it, the distribution of the number of views with a point within 2 cm of a GT
   point, Déjà View against Refined. Scenes weighted equally.
2. `e38_coverage_<scene>.jpg`: per view, ΔC_o and -ΔB of each GT pixel on one cm scale.
   Bedroom ai_006_007 views 18, 0, 12, 6 and ETH3D office views 0, 5, 10, 15, as E35.
   ETH3D data: CC BY-NC-SA 4.0, Schöps et al., CVPR 2017.

uv run python experiments/scripts/e38_figures.py runs/e37_hypersim-20261109-162959 runs/e37_eth3d-20261109-163228
"""

import argparse
from pathlib import Path

import matplotlib as mpl
import numpy as np
import yaml

mpl.use("Agg")
import matplotlib.pyplot as plt

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
CHANGE_MAX = 5.0  # cm
INK = "#0b0b0b"
OWN, BONUS = "#c0504d", "#4f81bd"
CAPTION = (
    "Most of the loss (59-99%) is -ΔB: each view covers its own GT about\n"
    "as well as before, but the views stop covering each other's gaps. Right: more GT points with no view within 2 cm, and more with many."
)


def mean_of(records: list[dict], path: tuple) -> float:
    """Mean over scenes of one nested value."""
    values = []
    for record in records:
        value = record
        for key in path:
            value = value[key]
        values.append(value)
    return float(np.nanmean(values))


def split_figure(spreads: dict[str, dict]) -> None:
    """Write figure 1 (see module) and print the numbers it shows."""
    fig, (ax, hx) = plt.subplots(
        1, 2, figsize=(14, 5.2), gridspec_kw={"width_ratios": [1.3, 1]}
    )
    x = np.arange(len(GROUPS))
    width = 0.36
    hists = {}
    for k, (data, g, label) in enumerate(GROUPS):
        spread = spreads[data]
        records = [spread["scenes"][n] for n in spread["groups"][g]]
        own = mean_of(records, ("change", "own_cm"))
        bonus = mean_of(records, ("change", "bonus_cm"))
        fused = own + bonus
        ax.bar(
            x[k] - width / 2 - 0.02,
            own,
            width,
            color=OWN,
            label="ΔC_o" if k == 0 else None,
        )
        ax.bar(
            x[k] + width / 2 + 0.02,
            bonus,
            width,
            color=BONUS,
            label="-ΔB" if k == 0 else None,
        )
        ax.text(
            x[k],
            max(own, bonus, 0) + 0.05,
            f"ΔC_f {fused:+.2f}",
            ha="center",
            fontsize=9,
        )
        print(f"{label}: ΔC_f {fused:+.2f} = ΔC_o {own:+.2f} + (-ΔB) {bonus:+.2f} cm")
        for v in range(2):
            hists[label, v] = np.mean(
                [r["variants"][v]["views_within_2cm_hist"] for r in records], axis=0
            )
    ax.axhline(0, color=INK, lw=0.6)
    ax.set_xticks(x, [g[2] for g in GROUPS])
    ax.set_ylabel("completeness change, Refined - Déjà View (cm)")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax.grid(axis="y", color="0.9", lw=0.6)
    ax.set_axisbelow(True)
    bins = np.arange(21)
    for label, style in (("Hypersim val", "-"), ("ETH3D indoor", "--")):
        for v, (name, color) in enumerate((("Déjà View", "0.45"), ("Refined", OWN))):
            hx.plot(bins, hists[label, v], style, color=color, label=f"{label}, {name}")
    hx.set_xlabel("views with a predicted point within 2 cm of a GT point")
    hx.set_ylabel("share of GT points")
    hx.set_xticks(bins[::2])
    hx.legend(frameon=False, fontsize=8)
    hx.grid(color="0.9", lw=0.6)
    for a in (ax, hx):
        for side in ("top", "right"):
            a.spines[side].set_visible(False)
    fig.suptitle(
        "E38: the joint refiner's completeness change ΔC_f = ΔC_o (each view's own cloud against its own GT) + (-ΔB) (the other views\n"
        "cover for it less). 20 views at 504 px, protocol alignment, every weight 1, scenes weighted equally. "
        + CAPTION,
        fontsize=10,
        color=INK,
    )
    fig.tight_layout()
    fig.savefig(FIGURES / "e38_split.jpg", dpi=110, pil_kwargs={"quality": 90})
    plt.close(fig)


def coverage_figure(runs: dict[str, Path], scene: str) -> None:
    """Write figure 2 (see module) for one scene."""
    data, shown, title = MAPS[scene]
    with np.load(runs[data] / "e38_maps_0_1.npz") as f:
        own = {i: 100 * f[f"{scene}/own/{i}"] for i in shown}
        bonus = {i: 100 * f[f"{scene}/bonus/{i}"] for i in shown}
    fig, axes = plt.subplots(
        2, len(shown), figsize=(3.3 * len(shown), 5.8), layout="constrained"
    )
    for c, i in enumerate(shown):
        for row, (m, name) in enumerate(((own, "ΔC_o"), (bonus, "-ΔB"))):
            im = axes[row, c].imshow(
                m[i], cmap="RdBu_r", vmin=-CHANGE_MAX, vmax=CHANGE_MAX
            )
            axes[row, c].set_xticks([])
            axes[row, c].set_yticks([])
            axes[row, c].set_title(
                f"view {i}, {name}: {np.nanmean(m[i]):+.2f} cm", fontsize=9
            )
    fig.colorbar(
        im, ax=axes, fraction=0.03, label="cm, red = covers worse after refinement"
    )
    fig.suptitle(
        f"E38, {title}: each GT pixel's change in own-view completeness (ΔC_o, top) and in what the other views add (-ΔB, bottom);\n"
        "they sum to the change of the protocol's completeness. White = no GT.",
        fontsize=10,
        color=INK,
    )
    out = FIGURES / f"e38_coverage_{scene.removesuffix('_cam_00')}.jpg"
    fig.savefig(out, dpi=100, pil_kwargs={"quality": 88})
    plt.close(fig)
    print(
        f"{scene}: "
        + "; ".join(
            f"view {i} ΔC_o {np.nanmean(own[i]):+.2f}, -ΔB {np.nanmean(bonus[i]):+.2f}"
            for i in shown
        )
    )


def main() -> None:
    """Write both figures; see the module docstring."""
    parser = argparse.ArgumentParser()
    parser.add_argument("hypersim", type=Path)
    parser.add_argument("eth3d", type=Path)
    args = parser.parse_args()
    runs = {"hypersim": args.hypersim, "eth3d": args.eth3d}
    spreads = {
        d: yaml.safe_load((r / "e38_spread_0_1.yaml").read_text(encoding="utf-8"))
        for d, r in runs.items()
    }
    split_figure(spreads)
    for scene in MAPS:
        coverage_figure(runs, scene)


if __name__ == "__main__":
    main()
