"""E46's figures: filter + refine against the filter alone, curves and the bedroom.

Curves (`e46_curves.png`, `e46_curves_scenes.png`): accuracy against completeness,
one panel per filter; the filter alone (E45's curve), filter + refine (E46's grid),
refine-then-cut by confidence (E45's "refined + confidence"), Déjà View and the refiner
as points. The means are over the split's scenes at each q, as the protocol aggregates.

Bedroom (`e46_bedroom.jpg`), views 18, 0, 12, 6: the pixels
voxel density keeps at `BEDROOM_Q`, without and with refinement; each kept pixel's
distance to the nearest GT point (removed pixels gray) and the change. No render: its
shading takes normals from neighboring pixels, NaN next to most removed ones.

uv run python experiments/scripts/e46_figures.py runs/<run> \
    > experiments/results/e46_figures.txt
"""

import dataclasses
import json
import sys
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from matplotlib.colors import Normalize
from scipy.spatial import cKDTree

from src.analysis.scene_eval import load_inputs
from src.analysis.scene_figures import (
    DIVERGING,
    ERROR_MAX_CM,
    PRESENTATION_SCENE,
    PRESENTATION_VIEWS,
    SEQUENTIAL,
    pixel_points,
)
from src.evaluation.labeled import align_output
from src.refinement.multiview import RefineConfig, points_from_rays, refine_scene

sys.path.insert(0, str(Path(__file__).parent))
from e40_figures import distances
from e45_filters import REFINED_FILTER
from e46_filter_refine import FILTERS, filter_scores

from src.analysis.filters import keep_top, ranks

OUT = Path("experiments/figures")
SPLITS = ("dev",)  # stress and val were not run (Peter: dev only for now)
BEDROOM_FILTER = "voxel density"  # E45's best filter on dev
# the grid point where voxel density + refine is most complete on dev (3.058 cm)
BEDROOM_Q = 0.7
CHANGE_CM = 2.0
SHOWN_MAX_CM = 10.0  # the distance maps' color scale, as E44's
CURVE_MAX_CM = 20.0  # completeness shown; beyond it no filter is of use


def mean_curve(curves: list[list[dict]]) -> tuple[np.ndarray, np.ndarray]:
    """Mean completeness and accuracy over scenes at each q of one grid."""
    comp = np.mean([[p["completeness_cm"] for p in c] for c in curves], 0)
    acc = np.mean([[p["accuracy_cm"] for p in c] for c in curves], 0)
    return comp, acc


def panel(ax: plt.Axes, e45: list[dict], e46: list[list[dict]], f: str) -> None:
    """Draw one filter's curves on one axis from per-scene records."""
    alone = mean_curve([r[f]["curve"] for r in e45])
    refined = mean_curve([[p["refined"] for p in c] for c in e46])
    cut = mean_curve([r[REFINED_FILTER]["curve"] for r in e45])
    ax.plot(*alone, "o-", ms=3, color="tab:blue", label=f"{f} alone (E45)")
    ax.plot(*refined, "s-", ms=4, color="tab:red", label=f"{f}, then refine (E46)")
    ax.plot(*cut, ".--", ms=3, color="0.5", label="refine, then cut by confidence")
    ax.plot(alone[0][0], alone[1][0], "k^", ms=8, label="Déjà View")
    ax.plot(refined[0][0], refined[1][0], "kD", ms=7, label="refiner")
    ax.set_xlabel("completeness (cm)")
    ax.set_ylabel("accuracy (cm)")
    ax.set_xlim(0, CURVE_MAX_CM)
    ax.grid(visible=True, alpha=0.3)


def curves(run: Path) -> None:
    """Draw the split means and the per-scene curves."""
    data = {
        s: (
            json.loads((run / f"e45_{s}.json").read_text())["scenes"],
            json.loads((run / f"e46_{s}.json").read_text())["scenes"],
        )
        for s in SPLITS
    }
    fig, axes = plt.subplots(
        len(SPLITS),
        len(FILTERS),
        figsize=(14, 4.2 * len(SPLITS)),
        layout="constrained",
        squeeze=False,
    )
    for row, split in zip(axes, SPLITS, strict=True):
        e45, e46 = data[split]
        for ax, f in zip(row, FILTERS, strict=True):
            panel(ax, list(e45.values()), [e46[n][f] for n in e45], f)
            ax.set_title(f"{split}, mean of {len(e45)} scenes: {f}", fontsize=9)
    axes[0, 0].legend(fontsize=7)
    fig.suptitle(
        "E46: filter first, then refine (red), against the filter alone (blue); "
        "lower left is better; each red point is read at its own completeness",
        x=0.01,
        ha="left",
    )
    fig.savefig(OUT / "e46_curves.png", dpi=110)
    plt.close(fig)
    names = [(s, n) for s in SPLITS for n in data[s][0]]
    fig, axes = plt.subplots(
        len(names),
        len(FILTERS),
        figsize=(14, 3.6 * len(names)),
        layout="constrained",
        squeeze=False,
    )
    for row, (split, name) in zip(axes, names, strict=True):
        e45, e46 = data[split]
        for ax, f in zip(row, FILTERS, strict=True):
            panel(ax, [e45[name]], [e46[name][f]], f)
            ax.set_title(f"{split} {name}: {f}", fontsize=9)
    axes[0, 0].legend(fontsize=7)
    fig.savefig(OUT / "e46_curves_scenes.png", dpi=90)
    plt.close(fig)


def bedroom(run: Path) -> None:
    """Draw the bedroom's distance maps and renders, and print their means."""
    record = yaml.safe_load((run / "run.yaml").read_text(encoding="utf-8"))
    cuda = torch.cuda.is_available()
    cfg = dataclasses.replace(
        RefineConfig(**record["refine"]["refine"]), device="cuda" if cuda else "cpu"
    )
    inputs = load_inputs(record["config"], PRESENTATION_SCENE, torch.device(cfg.device))
    scene = inputs.scene
    keep = keep_top(ranks(filter_scores(inputs)[BEDROOM_FILTER]), BEDROOM_Q)
    segments = torch.from_numpy(inputs.labels.instance_ids.astype(np.int64))
    refined, _, _ = refine_scene(
        scene, cfg, inputs.normals, segments, torch.from_numpy(keep)
    )
    depths = {
        "alone": np.where(keep, scene.depth.cpu().numpy(), np.nan),
        "then refined": refined.cpu().numpy(),
    }
    inputs = dataclasses.replace(inputs, scene=scene.to(torch.device("cpu")))
    tree = cKDTree(inputs.truth.points)
    clouds, maps = {}, {}
    for title, depth in depths.items():
        lifted = points_from_rays(
            torch.from_numpy(depth).double(), inputs.scene.rays.double()
        ).numpy()
        _, _, align = align_output(inputs.truth, lifted)
        name = f"{BEDROOM_FILTER} {title}, q = {BEDROOM_Q}"
        clouds[name] = pixel_points(inputs, depth, align)
        dist = np.full(depth.shape, np.nan)
        dist[keep] = distances(tree, clouds[name][keep])
        maps[name] = dist
        print(
            f"{PRESENTATION_SCENE}, {name}: mean distance of the kept pixels to the "
            f"nearest GT point {np.nanmean(dist):.3f} cm (capped at {ERROR_MAX_CM:.0f})"
        )
    change = maps[next(iter(reversed(maps)))] - maps[next(iter(maps))]
    print(
        f"refined minus alone: mean {np.nanmean(change):+.3f} cm, kept pixels closer "
        f"by > 0.5 cm {(change[keep] < -0.5).mean():.3f}, farther {(change[keep] > 0.5).mean():.3f}"
    )
    sequential = SEQUENTIAL.copy()
    sequential.set_bad("0.55")
    diverging = DIVERGING.copy()
    diverging.set_bad("0.55")
    views = PRESENTATION_VIEWS
    fig, axes = plt.subplots(
        len(views), 4, figsize=(13, 2.5 * len(views) + 1.2), layout="constrained"
    )
    for row, v in zip(axes, views, strict=True):
        row[0].imshow(inputs.views.images[v])
        for ax, dist in zip(row[1:3], maps.values(), strict=True):
            im = ax.imshow(dist[v], cmap=sequential, norm=Normalize(0, SHOWN_MAX_CM))
        ch = row[3].imshow(
            change[v], cmap=diverging, norm=Normalize(-CHANGE_CM, CHANGE_CM)
        )
        for ax in row:
            ax.set_xticks([])
            ax.set_yticks([])
        row[0].set_ylabel(f"view {v}")
    titles = ["image", *maps, "refined minus alone"]
    for ax, title in zip(axes[0], titles, strict=True):
        ax.set_title(title, fontsize=8)
    fig.colorbar(
        im,
        ax=axes[-1, 1:3],
        location="bottom",
        label="distance to GT (cm), 10 and beyond",
    )
    fig.colorbar(
        ch, ax=axes[-1, 3], location="bottom", label="cm, blue: refined closer"
    )
    fig.suptitle(
        f"Bedroom {PRESENTATION_SCENE}: the same {BEDROOM_Q:.0%} of the pixels kept by "
        "voxel density, without and with refinement; each kept pixel's distance to "
        "the GT, gray: removed",
        x=0.01,
        ha="left",
        fontsize=9,
    )
    fig.savefig(OUT / "e46_bedroom.jpg", dpi=110)
    plt.close(fig)


def main() -> None:
    """Draw every figure of E46."""
    run = Path(sys.argv[1])
    curves(run)
    bedroom(run)


if __name__ == "__main__":
    main()
