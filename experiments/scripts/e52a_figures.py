"""E52a figures: learned segment adapters inside frozen Déjà View's loop.

Reads the training run (`e52a.yaml`, `bias-gt.pt`) and the check run (`e52a.yaml`,
`e52a_maps.npz`) of `e52a_segment_adapters.py`, and the bedroom's cached views and GT.

1. `e52a_training.jpg`: held-out error over training steps for all 6 models (GT solid,
   grid dashed); and per scene (held-out and dev, 20 fixed views) the trained GT models'
   training loss over Déjà View's, against Déjà View's own loss.
2. `e52a_beta.jpg`: bias-gt's learned β per loop step and head, frame and global.
3. `e52a_bedroom.jpg`: bedroom ai_006_007 views 18, 0, 12, 6: image, GT instances per
   patch, the per-pixel 3D error of Déjà View and of each trained GT model on one cm
   scale (every grid control kept its zero state, so it is Déjà View).

uv run python experiments/scripts/e52a_figures.py runs/e52a runs/e52a_check
"""

import argparse
from pathlib import Path

import matplotlib as mpl
import numpy as np
import torch
import yaml

mpl.use("Agg")
import matplotlib.pyplot as plt
from e52a_segment_adapters import BASE, MAP_SCENE, MAP_VIEWS

from src.analysis.scene_eval import input_paths
from src.backbone.adapters import PATCH_SIZE, patch_segments
from src.loaders.ground_truth import GroundTruth
from src.loaders.views import Views

FIGURES = Path("experiments/figures")
E_MAX = 30.0  # cm, shared colour scale of the error maps
COLOURS = {"bias": "#3a7bd5", "pool": "#eb6834", "both": "#2a9d55"}


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


def training(models: dict, loss: dict) -> None:
    """Figure 1 of the module docstring."""
    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 4.2))
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
    left.axhline(models["bias-gt"]["held_out"][0], color="0.3", lw=0.8)
    left.set_xlabel("training step")
    left.set_ylabel("held-out error (mean of 8 scenes, 12 views)")
    left.set_title("held-out error; grey: Déjà View (step 0)", fontsize=10)
    left.legend(fontsize=7, frameon=False)
    style(left)
    trained = [k for k in next(iter(loss["dev"].values()))[20] if k != BASE]
    for split, marker in (("held_out", "o"), ("dev", "s")):
        for label in trained:
            model = label.split("-")[0]
            rows = [scene[20] for scene in loss[split].values()]
            right.plot(
                [r[BASE] for r in rows],
                [r[label] / r[BASE] for r in rows],
                marker,
                color=COLOURS[model],
                mfc="none" if split == "dev" else COLOURS[model],
                ms=6,
                label=f"{label}, {split.replace('_', '-')} scenes",
            )
    right.axhline(1, color="0.3", lw=0.8)
    right.set_xscale("log")
    right.set_xlabel("Déjà View's training loss on the scene (20 views)")
    right.set_ylabel("model's loss / Déjà View's")
    right.set_title("per scene: below 1 = better than Déjà View", fontsize=10)
    right.legend(fontsize=7, frameon=False)
    style(right)
    fig.suptitle(
        "E52a: the held-out gain comes from the 3 scenes where Déjà View fails"
        " (right, loss > 0.2); on the other 5 held-out scenes\nthe trained models are"
        " worse but for one (bias-gt, -0.7%), and on all 3 dev scenes worse. Pooling"
        " diverges (left), so its kept state is the zero start",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(FIGURES / "e52a_training.jpg", dpi=110, bbox_inches="tight")
    plt.close(fig)


def betas(beta: np.ndarray) -> None:
    """Figure 2 of the module docstring."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), sharey=True)
    fig.subplots_adjust(top=0.78)
    lim = float(np.abs(beta).max())
    for a, (ax, name) in enumerate(zip(axes, ("frame", "global"), strict=True)):
        im = ax.imshow(beta[:, a].T, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
        ax.set_title(f"{name} attention", fontsize=10)
        ax.set_xlabel("loop step k")
    axes[0].set_ylabel("head")
    bar = fig.colorbar(im, ax=axes.tolist(), shrink=0.9)
    bar.set_label("β (logit added to same-segment pairs)")
    fig.suptitle(
        "E52a, bias-gt (kept at step 2250): β differs by head more than by step;"
        " global heads 3, 6, 7 pull tokens towards\ntheir own segment in the later"
        " steps (red), heads 9 and 11 push them away (blue). Its dev Chamfer is worse"
        " than Déjà View's",
        fontsize=11,
    )
    fig.savefig(FIGURES / "e52a_beta.jpg", dpi=110, bbox_inches="tight")
    plt.close(fig)


def bedroom(check: Path, raw: dict) -> None:
    """Figure 3 of the module docstring."""
    with np.load(check / "e52a_maps.npz") as f:
        maps = {k: f[k].astype(np.float32) for k in f.files}
    shown = [BASE, *sorted(k for k in maps if k != BASE)]
    paths = input_paths(raw, MAP_SCENE)
    images = Views.load(paths["views"]).images
    segments = patch_segments(GroundTruth.load(paths["gt"]).instance_ids)
    rows = ["image", "GT instances per patch", *shown]
    fig, axes = plt.subplots(
        len(rows), len(MAP_VIEWS), figsize=(3.2 * len(MAP_VIEWS), 2.5 * len(rows))
    )
    for c, v in enumerate(MAP_VIEWS):
        axes[0, c].imshow(images[v])
        axes[0, c].set_title(f"view {v}", fontsize=10)
        axes[1, c].imshow(instance_colours(segments[v], images.shape[1:3]))
        for r, label in enumerate(shown, start=2):
            e = axes[r, c].imshow(
                100 * maps[label][c],
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
        "E52a, bedroom: bias-gt looks like Déjà View (mean error 9.0 against 8.8 cm);"
        " both-gt raises it over whole walls and\nthe ceiling (11.2 cm). Each model on"
        " its own cameras; every grid control kept its zero state and equals Déjà"
        " View.\nBlack patches: no instance on > half the patch",
        fontsize=11,
    )
    fig.savefig(FIGURES / "e52a_bedroom.jpg", dpi=110, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Draw the three figures from one training run and its check run."""
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("check", type=Path)
    args = parser.parse_args()
    run = yaml.safe_load((args.run / "e52a.yaml").read_text(encoding="utf-8"))
    check = yaml.safe_load((args.check / "e52a.yaml").read_text(encoding="utf-8"))
    raw = yaml.safe_load(Path(run["config"]["eval"]).read_text(encoding="utf-8"))
    training(run["records"]["models"], check["records"]["loss"])
    state = torch.load(args.run / "bias-gt.pt", map_location="cpu")["state"]
    betas(state["beta"].numpy())
    bedroom(args.check, raw)


if __name__ == "__main__":
    main()
