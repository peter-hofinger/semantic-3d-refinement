"""E53a figures: attention bias within GT instances inside Déjà View's loop.

Reads what `e53a_attention_bias.py` wrote (`e53a.yaml`, `e53a_maps.npz`) and the
bedroom's cached views and GT.

1. `e53a_scores.jpg`: Chamfer minus that of the same hook at beta = 0, over beta, one
   panel per step set; frame and global attention as two lines (the mean over the 3 dev
   scenes), each scene as a dot; and the bedroom's median ray turn.
2. `e53a_bedroom.jpg`: bedroom ai_006_007 views 18, 0, 12, 6: image, the GT instances
   as the bias sees them (one colour per object, per 14 px patch; black: no segment),
   and the per-pixel 3D error of Déjà View, of the best variant (global, first 4 steps,
   beta = 0.25) and of frame attention on the last 4 steps at beta = 1, on one shared
   cm scale.

uv run python experiments/scripts/e53a_figures.py runs/e53a_gt
"""

import argparse
from pathlib import Path

import matplotlib as mpl
import numpy as np
import yaml

mpl.use("Agg")
import matplotlib.pyplot as plt
from e53a_attention_bias import BASE, MAP_SCENE, MAP_VIEWS, STEP_SETS

from src.analysis.scene_eval import input_paths
from src.backbone.dejaview import PATCH_SIZE, WHERE, patch_segments
from src.loaders.ground_truth import GroundTruth
from src.loaders.views import Views

FIGURES = Path("experiments/figures")
E_MAX = 30.0  # cm, shared colour scale of the error maps
SHOWN = (BASE, "global first4 b=0.25", "frame last4 b=1")
COLOURS = {"frame": "#eb6834", "global": "#3a7bd5"}


def style(ax: plt.Axes) -> None:
    """Light grid, no top / right spines."""
    ax.grid(color="0.9", lw=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def instance_colours(segments: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Patch ids (N,) -> an RGB image, one fixed random colour per GT object."""
    h, w = shape
    ids = segments.reshape(h // PATCH_SIZE, w // PATCH_SIZE)
    palette = np.random.default_rng(0).uniform(0.15, 1.0, size=(int(ids.max()) + 2, 3))
    rgb = np.where((ids >= 0)[..., None], palette[np.maximum(ids, 0)], 0.0)
    return np.kron(rgb, np.ones((PATCH_SIZE, PATCH_SIZE, 1)))


def bedroom(out: Path, raw: dict) -> None:
    """Figure 2 of the module docstring."""
    with np.load(out / "e53a_maps.npz") as f:
        maps = {k: f[k].astype(np.float32) for k in SHOWN}
    paths = input_paths(raw, MAP_SCENE)
    images = Views.load(paths["views"]).images
    segments = patch_segments(GroundTruth.load(paths["gt"]).instance_ids)
    rows = ["image", "GT instances per patch", *SHOWN]
    fig, axes = plt.subplots(
        len(rows), len(MAP_VIEWS), figsize=(3.2 * len(MAP_VIEWS), 2.5 * len(rows))
    )
    for c, v in enumerate(MAP_VIEWS):
        axes[0, c].imshow(images[v])
        axes[0, c].set_title(f"view {v}", fontsize=10)
        axes[1, c].imshow(instance_colours(segments[v], images.shape[1:3]))
        for r, variant in enumerate(SHOWN, start=2):
            e = axes[r, c].imshow(
                100 * maps[variant][c],
                cmap="magma",
                vmin=0,
                vmax=E_MAX,
                interpolation="nearest",
            )
    for r, label in enumerate(rows):
        axes[r, 0].set_ylabel(label, fontsize=10)
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
    bar = fig.colorbar(e, ax=axes.ravel().tolist(), shrink=0.35, extend="max")
    bar.set_label("per-pixel 3D error after the protocol's Sim(3) (cm)")
    fig.suptitle(
        "E53a, bedroom: the best variant (global, first 4 steps, beta 0.25) looks like"
        " Déjà View. Frame attention biased on the last 4\nsteps at beta 1 lowers the"
        " ceiling's error (view 6) but raises it on the floor and furniture (bed,"
        " dresser, views 18, 0, 12).\nBiased depth on Déjà View's cameras; black"
        " patches: no instance on > half the patch",
        fontsize=11,
    )
    fig.savefig(FIGURES / "e53a_bedroom.jpg", dpi=110, bbox_inches="tight")
    plt.close(fig)


def scores(records: dict) -> None:
    """Figure 1 of the module docstring."""
    names, steps = list(records), list(STEP_SETS)
    fig, axes = plt.subplots(1, len(steps) + 1, figsize=(4.2 * (len(steps) + 1), 3.6))
    for ax, step in zip(axes, steps, strict=False):
        for where in WHERE:
            keys = [
                k
                for k in records[names[0]]["variants"]
                if k.startswith(f"{where} {step} ") and not k.endswith(" b=0")
            ]
            betas = [float(k.rsplit("=", 1)[1]) for k in keys]
            d = np.array(
                [
                    [
                        records[n]["variants"][k]["chamfer_cm"]
                        - records[n]["variants"][f"{where} {step} b=0"]["chamfer_cm"]
                        for k in keys
                    ]
                    for n in names
                ]
            )
            colour = COLOURS[where]
            ax.plot(betas, d.mean(0), "-", color=colour, label=f"{where} (mean)")
            for row in d:
                ax.plot(betas, row, "o", color=colour, ms=3, alpha=0.6)
        ax.axhline(0, color="0.3", lw=0.8)
        ax.set_xscale("log", base=2)
        ax.set_yscale("symlog", linthresh=0.1)
        ax.set_title(f"biased at steps: {step}", fontsize=10)
        ax.set_xlabel("beta")
        style(ax)
    axes[0].set_ylabel("Chamfer minus beta = 0's (cm)")
    axes[0].legend(fontsize=8, frameon=False)
    ax = axes[-1]
    v = records[MAP_SCENE]["variants"]
    for where in WHERE:
        for step, marker in zip(steps, "osd", strict=True):
            keys = [k for k in v if k.startswith(f"{where} {step} ")]
            ax.plot(
                [float(k.rsplit("=", 1)[1]) + 0.125 for k in keys],
                [v[k]["ray_deg"] for k in keys],
                marker + "-",
                color=COLOURS[where],
                ms=4,
                label=f"{where} {step}",
            )
    ax.set_xscale("log", base=2)
    ax.set_xlabel("beta + 0.125 (so that 0 shows)")
    ax.set_ylabel("median ray turn vs Déjà View (deg)")
    ax.set_title("bedroom: the bias bends the rays", fontsize=10)
    ax.legend(fontsize=7, frameon=False, ncol=2)
    style(ax)
    fig.suptitle(
        "E53a, Hypersim dev: no beta, step set or attention gains 0.05 cm Chamfer over"
        " its beta = 0 run (above 0 = worse; y symlog, linear within ±0.1 cm)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(FIGURES / "e53a_scores.jpg", dpi=110, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Draw both figures from one E53a run folder."""
    parser = argparse.ArgumentParser()
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    summary = yaml.safe_load((args.out / "e53a.yaml").read_text(encoding="utf-8"))
    bedroom(args.out, summary["config"])
    scores(summary["scenes"])


if __name__ == "__main__":
    main()
