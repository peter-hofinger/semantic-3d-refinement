"""E45, third arm: the chosen variant's curve and the bedroom, that variant only.

Reads `e45_voxel_fusion.py`'s `runs/<run>/e45_voxel_fusion_dev.json` and E45's curves.
The variant with the lowest mean dev accuracy at the refiner's completeness is the
method (the plan's amendment), drawn under `NAMES`; the others are not drawn. It turned
out not to average its points: own-instance count, "instance voxel density".
- `e45_voxel_fusion_curves.jpg`: accuracy against completeness per dev scene and their
  mean at each q: the method, voxel density alone (E45), random removal, the refiner.
- `e45_voxel_fusion_bedroom.jpg`: bedroom views 18, 0, 12, 6, both at the refiner's
  completeness there: each kept pixel's distance to the GT for voxel density alone and
  for the method (removed pixels gray); for a method that averages, also how far each
  kept point moved.

uv run python experiments/scripts/e45_voxel_fusion_figures.py runs/<run> \
    > experiments/results/e45_voxel_fusion_figures.txt
"""

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

sys.path.insert(0, str(Path(__file__).parent))
from e40_figures import distances
from e45_figures import mean_curve
from e45_voxel_fusion import ALONE, variants

from src.analysis.filters import keep_top, ranks
from src.analysis.scene_eval import load_inputs
from src.analysis.scene_figures import (
    PRESENTATION_SCENE,
    PRESENTATION_VIEWS,
    SCENE_COLORS,
    SEQUENTIAL,
)
from src.evaluation.labeled import align_output

OUT = Path("experiments/figures")
FUSION, ALONE_COLOR, REFINER, REFERENCE = (*SCENE_COLORS[:3], "#8c8b86")
DISTANCE_CM = 10.0
MOVE_MM = 5.0
X_REACH = 1.6  # panels show completeness up to this multiple of the refiner's
NAMES = {"instance count": "instance voxel density"}  # the variant as a method


def chosen(fusion: dict) -> str:
    """Return the variant with the lowest mean accuracy at the refiner's completeness.

    A variant that does not reach it on every scene (its completeness is worse than the
    refiner's even with every point kept) cannot be chosen.
    """
    scenes = list(fusion["scenes"].values())
    reached = [v for v in scenes[0] if all(s[v]["match"] is not None for s in scenes)]
    return min(
        reached, key=lambda v: np.mean([s[v]["match"]["accuracy_cm"] for s in scenes])
    )


def curves_figure(fusion: dict, e45: dict, best: str) -> None:
    """Accuracy against completeness per dev scene and the mean over them."""
    names = list(fusion["scenes"])
    columns = [*names, "mean"]
    fig, axes = plt.subplots(
        1, len(columns), figsize=(3.1 * len(columns), 3.3), layout="constrained"
    )
    for ax, col in zip(axes, columns, strict=True):
        picked = names if col == "mean" else [col]

        def xy(rs: list[dict]) -> tuple[np.ndarray, np.ndarray]:
            return mean_curve(rs, "completeness_cm"), mean_curve(rs, "accuracy_cm")

        ref = [
            np.mean([e45["refiner"][n][k] for n in picked])
            for k in ("completeness_cm", "accuracy_cm")
        ]
        ax.plot(
            *xy([e45["scenes"][n]["random"] for n in picked]),
            color=REFERENCE,
            lw=1,
            ls="--",
            label="random",
        )
        ax.plot(
            *xy([e45["scenes"][n][ALONE] for n in picked]),
            color=ALONE_COLOR,
            lw=2,
            label=f"{ALONE} (E45)",
        )
        ax.plot(
            *xy([fusion["scenes"][n][best] for n in picked]),
            color=FUSION,
            lw=2,
            label=NAMES.get(best, best),
        )
        ax.plot(*ref, "*", color=REFINER, ms=12, mec="white", label="refiner")
        x0 = xy([e45["scenes"][n][ALONE] for n in picked])[0][0]
        ax.set_xlim(0.9 * x0, X_REACH * ref[0])
        ax.set_title(col.removesuffix("_cam_00"), fontsize=9)
        ax.set_xlabel("completeness (cm)", fontsize=8)
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=7)
        visible = [ref[1]] + [
            y
            for rs in (
                [e45["scenes"][n][ALONE] for n in picked],
                [fusion["scenes"][n][best] for n in picked],
            )
            for x, y in zip(*xy(rs), strict=True)
            if x <= X_REACH * ref[0]
        ]
        low, high = min(visible), max(visible)
        ax.set_ylim(low - 0.08 * (high - low), high + 0.08 * (high - low))
    axes[0].set_ylabel("accuracy (cm)", fontsize=8)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncols=4, fontsize=8)
    fig.suptitle(
        f"E45, Hypersim dev: {NAMES.get(best, best)} against voxel density alone as q "
        "falls from 1; lower is better on both axes",
        x=0.01,
        ha="left",
        fontsize=9,
    )
    fig.savefig(OUT / "e45_voxel_fusion_curves.jpg", dpi=110)
    plt.close(fig)


def bedroom_figure(run_dir: Path, fusion: dict, e45: dict, best: str) -> None:
    """Both methods at the refiner's completeness on the bedroom's four views."""
    raw = yaml.safe_load((run_dir / "run.yaml").read_text(encoding="utf-8"))["config"]
    inputs = load_inputs(raw, PRESENTATION_SCENE, torch.device("cpu"))
    points, todo = variants(inputs)
    method = NAMES.get(best, best)
    tree = cKDTree(inputs.truth.points)
    rows = {
        f"{ALONE} alone": (todo["count + mean"][0], None, e45["scenes"]),
        method: (*todo[best], fusion["scenes"]),
    }
    distance, moved = {}, None
    for title, (score, fuse, scenes) in rows.items():
        key = ALONE if title.endswith(" alone") else best
        q = scenes[PRESENTATION_SCENE][key]["match"]["q"]
        keep = keep_top(ranks(score), q)
        cloud = points if fuse is None else fuse(points, keep)
        masked = np.where(keep[..., None], cloud, np.nan)
        _, _, align = align_output(inputs.truth, masked)
        registered = cloud @ align[:3, :3].T + align[:3, 3]
        distance[title] = np.where(keep, distances(tree, registered), np.nan)
        if fuse is not None and title == method:
            moved = np.where(
                keep, 1000 * np.linalg.norm(cloud - points, axis=-1), np.nan
            )
        print(
            f"bedroom, {title}: q = {q:.3f}, kept {keep.mean():.3f}, mean distance of "
            f"the kept points to the GT {np.nanmean(distance[title]):.3f} cm"
        )
    titles = ["image", *(f"{t}\ndistance to GT (cm)" for t in distance)]
    if moved is not None:
        titles.append(f"{method}\npoint moved (mm, Déjà View units)")
    views = PRESENTATION_VIEWS
    fig, axes = plt.subplots(
        len(views),
        len(titles),
        figsize=(2.65 * len(titles), 2.0 * len(views) + 1.2),
        layout="constrained",
    )
    cmap = SEQUENTIAL.with_extremes(bad="#d9d8d4")
    for row, v in zip(axes, views, strict=True):
        row[0].imshow(inputs.views.images[v])
        for ax, d in zip(row[1:], distance.values(), strict=False):
            im = ax.imshow(d[v], cmap=cmap, norm=Normalize(0, DISTANCE_CM))
        if moved is not None:
            mv = row[-1].imshow(moved[v], cmap=cmap, norm=Normalize(0, MOVE_MM))
        for ax in row:
            ax.set_xticks([])
            ax.set_yticks([])
        row[0].set_ylabel(f"view {v}", fontsize=8)
    for ax, title in zip(axes[0], titles, strict=True):
        ax.set_title(title, fontsize=8)
    fig.colorbar(
        im, ax=axes[-1, 1:3], location="bottom", label="cm, capped at 10; gray: removed"
    )
    if moved is not None:
        fig.colorbar(mv, ax=axes[-1, 3], location="bottom", label="mm")
    fig.suptitle(
        f"Bedroom ai_006_007: {ALONE} alone and {method}, each cut to the refiner's "
        "completeness",
        x=0.01,
        ha="left",
        fontsize=9,
    )
    fig.savefig(OUT / "e45_voxel_fusion_bedroom.jpg", dpi=110)
    plt.close(fig)


def main() -> None:
    """Draw both figures and print what they are drawn from."""
    run_dir = Path(sys.argv[1])
    fusion = json.loads((run_dir / "e45_voxel_fusion_dev.json").read_text())
    e45 = json.loads((run_dir / "e45_dev.json").read_text())
    best = chosen(fusion)
    print(
        f"chosen: {best} ({NAMES.get(best, best)}), the lowest mean dev accuracy at "
        "the refiner's completeness of the seven variants"
    )
    curves_figure(fusion, e45, best)
    bedroom_figure(run_dir, fusion, e45, best)


if __name__ == "__main__":
    main()
