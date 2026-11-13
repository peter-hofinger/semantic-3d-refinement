"""E51a figures: token averaging within GT instances inside Déjà View's loop.

Reads what `e51a_token_mix.py` wrote (`e51a.yaml`, `e51a_maps.npz`) and the bedroom's
cached views and GT.

1. `e51a_bedroom.jpg`: bedroom ai_006_007 views 18, 0, 12, 6: image, the GT instances as
   the hook sees them (one colour per object, per 14 px patch; black: no segment), and
   the per-pixel 3D error of Déjà View, of the mildest variant (a = 0.01 on the first 4
   steps) and of a = 0.1 on the last 4 steps, on one shared cm scale.
2. `e51a_scores.jpg`: Chamfer's change against Déjà View per scene over a, one panel per
   step set (log x; 0 = no change), and the median angle the rays turned.

uv run python experiments/scripts/e51a_figures.py runs/e51a_gt
"""

import argparse
from pathlib import Path

import matplotlib as mpl
import numpy as np
import yaml

mpl.use("Agg")
import matplotlib.pyplot as plt
from e51a_token_mix import BASE, MAP_SCENE, MAP_VIEWS, STEP_SETS

from src.analysis.scene_eval import input_paths
from src.backbone.dejaview import PATCH_SIZE, patch_segments
from src.loaders.ground_truth import GroundTruth
from src.loaders.views import Views

FIGURES = Path("experiments/figures")
E_MAX = 30.0  # cm, shared colour scale of the error maps
SHOWN = (BASE, "a=0.01 first4", "a=0.1 last4")
SCENES = {
    "ai_010_007_cam_00": "living room",
    "ai_006_007_cam_00": "bedroom",
    "ai_044_001_cam_00": "lobby",
}
COLOURS = ("#52514e", "#eb6834", "#3a7bd5")


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
    """Figure 1 of the module docstring."""
    with np.load(out / "e51a_maps.npz") as f:
        maps = {k: f[k].astype(np.float32) for k in f.files}
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
        "E51a, bedroom: a = 0.01 on the first 4 steps looks like Déjà View; at a = 0.1"
        " on the last 4 the walls and ceiling\n(one GT instance each) take a large"
        " error. Mixed depth on Déjà View's cameras; black patches: no instance on"
        " > half the patch",
        fontsize=11,
    )
    fig.savefig(FIGURES / "e51a_bedroom.jpg", dpi=110, bbox_inches="tight")
    plt.close(fig)


def scores(records: dict) -> None:
    """Figure 2 of the module docstring."""
    steps = list(STEP_SETS)
    fig, axes = plt.subplots(1, len(steps) + 1, figsize=(4.2 * (len(steps) + 1), 3.6))
    for ax, step in zip(axes, steps, strict=False):
        for (scene, label), colour in zip(SCENES.items(), COLOURS, strict=True):
            variants = records[scene]["variants"]
            base = variants[BASE]["chamfer_cm"]
            keys = [k for k in variants if k.endswith(f" {step}")]
            alphas = [float(k.split()[0][2:]) for k in keys]
            d = [variants[k]["chamfer_cm"] - base for k in keys]
            ax.plot(alphas, d, "o-", color=colour, label=f"{label} (DV {base:.2f})")
        ax.axhline(0, color="0.3", lw=0.8)
        ax.set_xscale("log")
        ax.set_yscale("symlog", linthresh=0.1)
        ax.set_title(f"mixed after steps: {step}", fontsize=10)
        ax.set_xlabel("alpha")
        style(ax)
    axes[0].set_ylabel("Chamfer minus Déjà View's (cm)")
    axes[0].legend(fontsize=8, frameon=False)
    ax = axes[-1]
    for step, marker in zip(steps, "osd", strict=True):
        v = records[MAP_SCENE]["variants"]
        keys = [k for k in v if k.endswith(f" {step}")]
        ax.plot(
            [float(k.split()[0][2:]) for k in keys],
            [v[k]["ray_deg"] for k in keys],
            marker + "-",
            color="0.35",
            label=step,
        )
    ax.set_xscale("log")
    ax.set_xlabel("alpha")
    ax.set_ylabel("median ray turn vs Déjà View (deg)")
    ax.set_title("bedroom: the rays move too", fontsize=10)
    ax.legend(fontsize=8, frameon=False)
    style(ax)
    fig.suptitle(
        "E51a, Hypersim dev: no alpha and no step set beats Déjà View on any scene"
        " (above 0 = worse; y symlog, linear within ±0.1 cm)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(FIGURES / "e51a_scores.jpg", dpi=110, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Draw both figures from one E51a run folder."""
    parser = argparse.ArgumentParser()
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    summary = yaml.safe_load((args.out / "e51a.yaml").read_text(encoding="utf-8"))
    bedroom(args.out, summary["config"])
    scores(summary["scenes"])


if __name__ == "__main__":
    main()
