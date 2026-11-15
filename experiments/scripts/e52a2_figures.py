"""E52a, second run, figures: segment adapters trained on the loss over Déjà View's.

Reads the two training runs (GT models, grid models: `e52a.yaml` each) and the check run
of the GT models (`e52a_maps.npz`) of `e52a_segment_adapters.py`, and the bedroom's views
and GT.

1. `e52a2_training.jpg`: held-out error (over Déjà View's, mean of 16 scenes) over training
   steps for all 6 models; dev Chamfer of each kept model minus Déjà View's, per scene,
   GT instances next to the grid control.
2. `e52a2_bedroom.jpg`: bedroom ai_006_007 views 18, 0, 12, 6: image, GT instances per
   patch, Déjà View's per-pixel 3D error, and each trained GT model's error minus it.

uv run python experiments/scripts/e52a2_figures.py runs/e52a2_gt runs/e52a2_grid runs/e52a2_check
"""

import argparse
from pathlib import Path

import matplotlib as mpl
import numpy as np
import yaml

mpl.use("Agg")
import matplotlib.pyplot as plt
from e52a_figures import COLOURS, E_MAX, FIGURES, instance_colours, style
from e52a_segment_adapters import BASE, MAP_SCENE, MAP_VIEWS

from src.analysis.scene_eval import input_paths
from src.backbone.adapters import patch_segments
from src.loaders.ground_truth import GroundTruth
from src.loaders.views import Views

Y_MAX = 1.1  # held-out axis cut: the pooling curves go on to 2.6-5.3
D_MAX = 5.0  # cm, shared scale of the error change maps
SHORT = {"ai_010_007_cam_00": "living room", "ai_006_007_cam_00": "bedroom",
         "ai_044_001_cam_00": "lobby"}  # fmt: skip


def training(models: dict, evals: dict) -> None:
    """Figure 1 of the module docstring."""
    fig, (left, right) = plt.subplots(1, 2, figsize=(13, 4.4))
    for label, rec in models.items():
        model, kind = label.split("-")
        steps = sorted(rec["held_out"])
        left.plot(
            steps,
            [rec["held_out"][s] for s in steps],
            "-" if kind == "gt" else "--",
            color=COLOURS[model],
            label=f"{label} (kept: step {rec['best_step']})",
        )
    left.axhline(1, color="0.3", lw=0.8)
    left.set_ylim(0.98, Y_MAX)
    left.set_xlabel("training step")
    left.set_ylabel("held-out error / Déjà View's (mean of 16 scenes)")
    left.set_title(f"held-out error, cut at {Y_MAX}; grey: Déjà View", fontsize=10)
    left.legend(fontsize=7, frameon=False)
    style(left)
    scenes = list(SHORT)
    labels = [f"{m}-{k}" for m in COLOURS for k in ("gt", "grid")]
    width = 0.8 / len(labels)
    for i, label in enumerate(labels):
        model, kind = label.split("-")
        delta = [
            evals[kind][s][label]["chamfer_cm"] - evals[kind][s][BASE]["chamfer_cm"]
            for s in scenes
        ]
        right.bar(
            np.arange(len(scenes)) + (i - len(labels) / 2 + 0.5) * width,
            delta,
            width,
            color=COLOURS[model],
            hatch="" if kind == "gt" else "//",
            edgecolor="white" if kind == "gt" else COLOURS[model],
            fill=kind == "gt",
            label=label,
        )
    right.axhline(0, color="0.3", lw=0.8)
    right.axhline(-0.05, color="0.6", lw=0.6, ls=":")
    right.set_xticks(range(len(scenes)), [SHORT[s] for s in scenes])
    right.set_ylabel("Chamfer minus Déjà View's (cm)")
    right.set_title("dev, kept models (dotted: the reading rule's -0.05 cm)",
                    fontsize=10)  # fmt: skip
    right.legend(fontsize=7, frameon=False, ncol=3)
    style(right)
    fig.suptitle(
        "E52a, second run: every model's training turns unstable after ~500 steps"
        " (left), so all kept states are early or zero.\nPooling gains on 2-3 dev"
        " scenes with GT instances and with the grid alike (right): adaptation, not"
        " segments",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(FIGURES / "e52a2_training.jpg", dpi=110, bbox_inches="tight")
    plt.close(fig)


def bedroom(check: Path, raw: dict) -> None:
    """Figure 2 of the module docstring."""
    with np.load(check / "e52a_maps.npz") as f:
        maps = {k: f[k].astype(np.float32) for k in f.files}
    trained = sorted(k for k in maps if k != BASE)
    paths = input_paths(raw, MAP_SCENE)
    images = Views.load(paths["views"]).images
    segments = patch_segments(GroundTruth.load(paths["gt"]).instance_ids)
    rows = ["image", "GT instances per patch", f"{BASE} error",
            *[f"{k} minus {BASE}" for k in trained]]  # fmt: skip
    fig, axes = plt.subplots(
        len(rows), len(MAP_VIEWS), figsize=(3.2 * len(MAP_VIEWS), 2.5 * len(rows))
    )
    for c, v in enumerate(MAP_VIEWS):
        axes[0, c].imshow(images[v])
        axes[0, c].set_title(f"view {v}", fontsize=10)
        axes[1, c].imshow(instance_colours(segments[v], images.shape[1:3]))
        e = axes[2, c].imshow(100 * maps[BASE][c], cmap="magma", vmin=0, vmax=E_MAX,
                              interpolation="nearest")  # fmt: skip
        for r, label in enumerate(trained, start=3):
            d = axes[r, c].imshow(
                100 * (maps[label][c] - maps[BASE][c]),
                cmap="RdBu_r",
                vmin=-D_MAX,
                vmax=D_MAX,
                interpolation="nearest",
            )
    for r, label in enumerate(rows):
        axes[r, 0].set_ylabel(label, fontsize=9)
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
    bar = fig.colorbar(e, ax=axes[2].tolist(), shrink=0.8, extend="max")
    bar.set_label("3D error (cm)")
    bar = fig.colorbar(d, ax=axes[3:].ravel().tolist(), shrink=0.8, extend="both")
    bar.set_label("error change (cm), blue = better")
    means = {k: 100 * float(np.nanmean(m)) for k, m in maps.items()}
    fig.suptitle(
        "E52a, second run, bedroom: mean per-pixel error "
        + ", ".join(f"{k} {x:.1f} cm" for k, x in means.items())
        + ".\nEach model on its own cameras, after the protocol's Sim(3)."
        " Black patches: no instance on > half the patch",
        fontsize=11,
    )
    fig.savefig(FIGURES / "e52a2_bedroom.jpg", dpi=110, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Draw both figures from the GT and grid training runs and the GT check run."""
    parser = argparse.ArgumentParser()
    parser.add_argument("gt", type=Path)
    parser.add_argument("grid", type=Path)
    parser.add_argument("check", type=Path)
    args = parser.parse_args()
    runs = {
        k: yaml.safe_load((p / "e52a.yaml").read_text(encoding="utf-8"))
        for k, p in (("gt", args.gt), ("grid", args.grid))
    }
    raw = yaml.safe_load(Path(runs["gt"]["config"]["eval"]).read_text(encoding="utf-8"))
    models = runs["gt"]["records"]["models"] | runs["grid"]["records"]["models"]
    training(models, {k: r["records"]["eval"] for k, r in runs.items()})
    bedroom(args.check, raw)


if __name__ == "__main__":
    main()
