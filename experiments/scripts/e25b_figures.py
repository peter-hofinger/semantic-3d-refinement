"""E25b figures: Déjà View's error at depth edges, and after snapping edge pixels onto their segment.

Reads what `e25b_boundary_oracle.py` wrote (`e25b.yaml`, `e25b_maps.npz`) and the bedroom's
cached views.

1. `e25b_bedroom.jpg`: bedroom ai_006_007 views 18, 0, 12, 6: image, Déjà View's t in the
   GT edge band (where its depth lies between its own surface and the one across the edge),
   and the per-pixel error of Déjà View and of each snap on one shared cm scale.
2. `e25b_scores.jpg`: per scene, each snap's change of the band error and of Chamfer
   against Déjà View on one cm axis (x: the snap moved the alignment scale by > SCALE), and
   Déjà View's band pixels and band error by t.

uv run python experiments/scripts/e25b_figures.py runs/e25b
"""

import argparse
from pathlib import Path

import matplotlib as mpl
import numpy as np
import yaml

mpl.use("Agg")
import matplotlib.pyplot as plt
from e25b_boundary_oracle import SOURCES, T_LABELS

from src.analysis.scene_eval import input_paths
from src.loaders.views import Views

FIGURES = Path("experiments/figures")
BEDROOM, VIEWS = "ai_006_007_cam_00", (18, 0, 12, 6)
E_MAX = 30.0  # cm, shared colour scale of the error maps
Y_MAX = 20.0  # cm, shared y range of the per-scene changes
SCALE = 0.01  # alignment scale change that marks a scene whose Sim(3) a snap dragged
VARIANTS = ("Déjà View", *SOURCES)
DV, REF = "#52514e", "#eb6834"


def style(ax: plt.Axes) -> None:
    """Light grid, no top / right spines."""
    ax.grid(axis="y", color="0.9", lw=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def bedroom(out: Path, raw: dict) -> None:
    """Write figure 1 (see module)."""
    with np.load(out / "e25b_maps.npz") as f:
        maps = {k.split("/", 1)[1]: f[k] for k in f.files if k.startswith(BEDROOM)}
    images = Views.load(input_paths(raw, BEDROOM)["views"]).images
    rows = ["image", "t", *VARIANTS]
    fig, axes = plt.subplots(
        len(rows), len(VIEWS), figsize=(3.2 * len(VIEWS), 2.5 * len(rows))
    )
    for c, v in enumerate(VIEWS):
        axes[0, c].imshow(images[v])
        axes[0, c].set_title(f"view {v}", fontsize=10)
        t = axes[1, c].imshow(
            np.clip(maps["t"][v], -0.5, 1.5),
            cmap="coolwarm",
            vmin=-0.5,
            vmax=1.5,
            interpolation="nearest",
        )
        for r, variant in enumerate(VARIANTS, start=2):
            e = axes[r, c].imshow(
                100 * maps[f"error/{variant}"][v],
                cmap="magma",
                vmin=0,
                vmax=E_MAX,
                interpolation="nearest",
            )
    for r, label in enumerate(rows):
        axes[r, 0].set_ylabel(
            label if label != "t" else "Déjà View's t\n(edge band)", fontsize=10
        )
        for ax in axes[r]:
            ax.set_xticks([])
            ax.set_yticks([])
    fig.colorbar(
        t, ax=axes[1], fraction=0.02, label="t: 0 own surface, 1 across the edge"
    )
    fig.colorbar(
        e,
        ax=axes[2:].ravel().tolist(),
        fraction=0.02,
        label="error to own GT point (cm)",
    )
    fig.suptitle(
        "Bedroom ai_006_007: where Déjà View's edge pixels sit (t, GT edge band only) and the error before / after snapping\n"
        "each segment's boundary band onto a plane fitted inside it; look at the object outlines and at new errors inside segments",
        fontsize=10,
    )
    fig.savefig(
        FIGURES / "e25b_bedroom.jpg",
        dpi=110,
        bbox_inches="tight",
        pil_kwargs={"quality": 88},
    )
    plt.close(fig)


def scores(out: Path) -> None:
    """Write figure 2 (see module)."""
    data = yaml.safe_load((out / "e25b.yaml").read_text(encoding="utf-8"))
    scenes = data["scenes"]
    dev = set(data["groups"]["dev"])
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), width_ratios=(1, 1, 0.8))
    for ax, key, label in (
        (axes[0], "band_error_cm", "change of the error in the GT edge band (cm)"),
        (axes[1], "chamfer_cm", "change of Chamfer (cm)"),
    ):
        ax.axhline(0, color="0.6", lw=0.8)
        for k, variant in enumerate(SOURCES):
            for j, (n, rec) in enumerate(scenes.items()):
                base = rec["variants"]["Déjà View"]
                d = rec["variants"][variant][key] - base[key]
                moved = (
                    abs(rec["variants"][variant]["scale"] / base["scale"] - 1) > SCALE
                )
                x = k + 0.3 * (j / (len(scenes) - 1) - 0.5)
                ax.plot(
                    x,
                    np.clip(d, -Y_MAX, Y_MAX),
                    "x" if moved else "o",
                    ms=6 if moved else 4,
                    color=REF if n in dev else DV,
                )
        ax.set_ylim(-Y_MAX * 1.05, Y_MAX * 1.05)
        ax.set_xticks(range(len(SOURCES)), SOURCES, fontsize=9)
        ax.set_ylabel(label)
        style(ax)
    axes[0].set_title(
        "per scene, snap - Déjà View (orange: dev, grey: val);\n"
        f"x: the snap moved the alignment scale by > {SCALE:.0%} (clipped at ±{Y_MAX:.0f})",
        fontsize=9,
    )
    axes[1].set_title(
        "same axis: the band error drops, Chamfer barely moves", fontsize=9
    )
    ax = axes[2]
    labels = ("away (t ≤ 0)", "own (0-0.25)", "between (0.25-0.75)", "other (> 0.75)")
    recs = [r["variants"]["Déjà View"] for r in scenes.values()]
    for row, key in enumerate(("t_px", "t_error")):
        left = 0.0
        for lab, name, colour in zip(
            T_LABELS, labels, ("#c9c8c4", "#9b9a96", "#f2a37f", REF), strict=True
        ):
            share = float(np.mean([r[key][lab] for r in recs]))
            ax.barh(row, share, left=left, color=colour, label=name if row else None)
            ax.text(
                left + share / 2,
                row,
                f"{share:.2f}",
                ha="center",
                va="center",
                fontsize=8,
            )
            left += share
    ax.set_yticks((0, 1), ("pixels", "error"))
    ax.set_xlim(0, 1)
    ax.set_xlabel("share of the edge-band pixels with a surface across the edge")
    ax.set_title("Déjà View in the GT edge band, by t (14 scenes)", fontsize=9)
    ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=2)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGURES / "e25b_scores.jpg", dpi=110, pil_kwargs={"quality": 88})
    plt.close(fig)


def main() -> None:
    """Write both figures; see the module docstring."""
    parser = argparse.ArgumentParser()
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    raw = yaml.safe_load(Path("configs/eval_hypersim.yaml").read_text(encoding="utf-8"))
    bedroom(args.out, raw)
    scores(args.out)


if __name__ == "__main__":
    main()
