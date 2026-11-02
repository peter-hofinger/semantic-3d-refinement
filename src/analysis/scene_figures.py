"""Figures and a README for a `scene_eval` run: scores, views, clouds from above.

`uv run python -m src.analysis.scene_figures --run runs/<run>` writes
`figures/<run>/`: the headline scores of every variant on every scene, and per scene
the views with ground-truth and SAM 3 objects next to each pixel's distance to the
ground truth, the fused clouds from above, and the IoU of every class.
"""

import argparse
import colorsys
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from matplotlib.colors import Normalize
from scipy.spatial import cKDTree

from src.analysis.scene_eval import HEADLINE, SceneInputs, load_inputs, variant_scene
from src.analysis.visualize import DIVERGING, SEQUENTIAL
from src.evaluation.depth import lift
from src.evaluation.labeled import NONE, predicted_cloud
from src.refinement.monocular import points_from_rays

LABELS = {
    "abs_rel": "AbsRel (per pixel) ↓",
    "accuracy_cm": "accuracy (cm) ↓",
    "completeness_cm": "completeness (cm) ↓",
    "fscore_2cm": "F-score at 2 cm ↑",
    "miou": "mIoU ↑",
    "ap50": "AP50 ↑",
    "precision_2cm": "precision at 2 cm ↑",
    "recall_2cm": "recall at 2 cm ↑",
    "disagreement_pct": "disagreement between views (%)",
}
TUNING = (
    "abs_rel",
    "precision_2cm",
    "recall_2cm",
    "fscore_2cm",
    "disagreement_pct",
    "miou",
)
SCENE_COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#8a5cc2")
ERROR_MAX_CM = 30.0
SHOWN_VIEWS = (0, 5, 10, 15)
TOP_POINTS = 150_000  # per panel of the clouds from above
SLAB = (0.05, 1.5)  # m above the floor shown from above: furniture, no floor or ceiling


def instance_colors(ids: np.ndarray) -> np.ndarray:
    """RGB in [0, 1] per id, spread by the golden ratio; gray for none."""
    unique, inverse = np.unique(ids, return_inverse=True)
    hues = (np.abs(unique) * 0.618034) % 1
    lut = np.array([colorsys.hsv_to_rgb(h, 0.65, 0.92) for h in hues])
    lut[unique < 0] = 0.85
    return lut[inverse.reshape(ids.shape)]


def pixel_points(inputs: SceneInputs, depth: np.ndarray, variant: dict) -> np.ndarray:
    """Per-pixel points (V, H, W, 3) of a variant's depth, in the ground truth's frame."""
    if variant.get("geometry") == "gt":
        return ground_truth_points(inputs)
    scene, (s, rotation, t) = variant_scene(inputs, variant)
    points = points_from_rays(
        torch.from_numpy(depth).double(), scene.rays.cpu().double()
    ).numpy()
    return s * points @ rotation.T + t


def ground_truth_points(inputs: SceneInputs) -> np.ndarray:
    """Per-pixel ground-truth points (V, H, W, 3)."""
    return np.stack(
        [
            lift(d.astype(np.float64), k.astype(np.float64), c.astype(np.float64))
            for d, k, c in zip(
                inputs.gt.depth,
                inputs.views.intrinsics,
                inputs.views.cam_to_world,
                strict=True,
            )
        ]
    )


def summary_figure(
    run: dict, out: Path, keys: tuple[str, ...] = HEADLINE, stem: str = "summary"
) -> None:
    """Scores: one panel per score, variants as rows, scenes as bars."""
    scenes = list(run["scenes"])
    variants = [v["name"] for v in run["config"]["variants"]]
    fig, axes = plt.subplots(
        2, 3, figsize=(15, 0.42 * len(variants) * 2 + 2), layout="constrained"
    )
    height = 0.8 / len(scenes)
    for ax, key in zip(axes.flat, keys, strict=True):
        for j, scene in enumerate(scenes):
            values = [run["scenes"][scene]["variants"][v][key] for v in variants]
            y = np.arange(len(variants)) + (j - (len(scenes) - 1) / 2) * height
            ax.barh(y, values, height, color=SCENE_COLORS[j], label=scene)
            ax.axvline(values[0], color=SCENE_COLORS[j], lw=0.8, ls="--")
        ax.set_yticks(np.arange(len(variants)), variants if ax in axes[:, 0] else [])
        ax.invert_yaxis()
        ax.set_title(LABELS[key])
        ax.grid(axis="y", visible=False)
    axes[0, -1].legend(loc="lower right")
    fig.suptitle(
        "Every variant on every scene; dashed: Déjà View's value per scene",
        x=0.01,
        ha="left",
    )
    fig.savefig(out / f"{stem}.png", dpi=130)
    plt.close(fig)


def views_figure(
    inputs: SceneInputs,
    depths: dict[str, np.ndarray],
    variants: dict[str, dict],
    name: str,
    out: Path,
) -> dict[str, float]:
    """Draw views: image, GT and SAM 3 objects, per-pixel distance to ground truth."""
    truth = ground_truth_points(inputs)
    seen = inputs.gt.depth > 0
    errors = {
        k: np.where(
            seen,
            100 * np.linalg.norm(pixel_points(inputs, d, variants[k]) - truth, axis=-1),
            np.nan,
        )
        for k, d in depths.items()
    }
    first, *_, last = errors
    titles = ["image", "ground-truth objects", "SAM 3 objects (linked)"]
    titles += [f"{k}\ndistance (cm)" for k in errors] + [f"{last}\nminus {first}"]
    fig, axes = plt.subplots(
        len(SHOWN_VIEWS),
        len(titles),
        figsize=(2.3 * len(titles), 1.9 * len(SHOWN_VIEWS) + 1.2),
        layout="constrained",
    )
    error_norm = Normalize(0, ERROR_MAX_CM)
    change_norm = Normalize(-10, 10)
    for row, v in zip(axes, SHOWN_VIEWS, strict=True):
        gt_ids = np.where(
            inputs.gt.instance_ids[v] >= 0, inputs.gt.instance_ids[v], NONE
        )
        panels = [
            inputs.views.images[v],
            instance_colors(gt_ids),
            instance_colors(inputs.labels.instance_ids[v]),
        ]
        for ax, image in zip(row[:3], panels, strict=True):
            ax.imshow(image)
        for ax, e in zip(row[3:-1], errors.values(), strict=True):
            im = ax.imshow(e[v], cmap=SEQUENTIAL, norm=error_norm)
        ch = row[-1].imshow(
            errors[last][v] - errors[first][v], cmap=DIVERGING, norm=change_norm
        )
        for ax in row:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(visible=False)
        row[0].set_ylabel(f"view {v}")
    for ax, title in zip(axes[0], titles, strict=True):
        ax.set_title(title, fontsize=8)
    fig.colorbar(
        im,
        ax=axes[-1, 3:-1],
        location="bottom",
        shrink=0.7,
        label=f"distance (cm), capped at {ERROR_MAX_CM:g}",
    )
    fig.colorbar(ch, ax=axes[-1, -1], location="bottom", label="cm, blue: closer")
    fig.suptitle(
        f"{name}: distance of each pixel's point to the same pixel's ground-truth point",
        x=0.01,
        ha="left",
    )
    fig.savefig(out / f"{name}_views.png", dpi=110)
    plt.close(fig)
    return {k: float(np.nanmedian(e)) for k, e in errors.items()}


def topdown_figure(
    inputs: SceneInputs,
    depths: dict[str, np.ndarray],
    variants: dict[str, dict],
    name: str,
    out: Path,
) -> None:
    """Fused clouds from above (x-y, z up), colored by object and by distance."""
    truth = inputs.truth
    rng = np.random.default_rng(0)
    lo, hi = np.percentile(truth.points[:, :2], [1, 99], axis=0)
    margin = 0.05 * (hi - lo)
    floor = np.percentile(truth.points[:, 2], 2)

    def slab(points: np.ndarray) -> np.ndarray:
        height = points[:, 2] - floor
        return (height > SLAB[0]) & (height < SLAB[1])

    keep = slab(truth.points)
    columns = [("ground truth", truth.points[keep], truth.instance[keep], None)]
    tree = cKDTree(truth.points)
    for k, d in depths.items():
        labels = inputs.labels
        cloud = predicted_cloud(
            pixel_points(inputs, d, variants[k]), labels.instance_ids, labels.classes, {},
            (1.0, np.eye(3), np.zeros(3)),
        )  # fmt: skip
        keep = slab(cloud.points)
        dist = tree.query(cloud.points[keep], workers=-1, distance_upper_bound=1.0)[0]
        columns.append(
            (k, cloud.points[keep], cloud.instance[keep], np.minimum(dist, 1.0))
        )
    fig, axes = plt.subplots(
        2, len(columns), figsize=(4.2 * len(columns), 8.6), layout="constrained"
    )
    for col, (title, points, ids, dist) in zip(axes.T, columns, strict=True):
        pick = rng.choice(len(points), min(TOP_POINTS, len(points)), replace=False)
        p = points[pick]
        col[0].scatter(p[:, 0], p[:, 1], s=1.0, c=instance_colors(ids[pick]), lw=0)
        col[0].set_title(f"{title}: objects", fontsize=9)
        if dist is not None:
            col[1].scatter(
                p[:, 0], p[:, 1], s=1.0, c=100 * dist[pick], cmap=SEQUENTIAL,
                norm=Normalize(0, ERROR_MAX_CM), lw=0,
            )  # fmt: skip
            col[1].set_title(f"{title}: distance to ground truth", fontsize=9)
        else:
            col[1].scatter(p[:, 0], p[:, 1], s=1.0, c=p[:, 2], cmap="Greys", lw=0)
            col[1].set_title("ground truth: height", fontsize=9)
        for ax in col:
            ax.set_xlim(lo[0] - margin[0], hi[0] + margin[0])
            ax.set_ylim(lo[1] - margin[1], hi[1] + margin[1])
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(visible=False)
    fig.suptitle(
        f"{name}: all 20 views fused on the 1 cm grid, seen from above, points "
        f"{100 * SLAB[0]:g} to {100 * SLAB[1]:g} cm above the floor; distance in cm, "
        f"0 (light) to {ERROR_MAX_CM:g} (dark)",
        x=0.01,
        ha="left",
    )
    fig.savefig(out / f"{name}_topdown.png", dpi=110)
    plt.close(fig)


def precision_recall_figure(run: dict, names: list[str], out: Path) -> None:
    """Precision and recall against tolerance, per scene, for the shown variants."""
    scenes = list(run["scenes"])
    fig, axes = plt.subplots(
        1, len(scenes), figsize=(4.4 * len(scenes), 3.6), squeeze=False
    )
    tolerances = (1, 2, 5)
    for ax, scene in zip(axes[0], scenes, strict=True):
        for j, n in enumerate(names):
            r = run["scenes"][scene]["variants"][n]
            for kind, style in (("precision", "-"), ("recall", "--")):
                ax.plot(
                    tolerances,
                    [r[f"{kind}_{t}cm"] for t in tolerances],
                    style,
                    marker="o",
                    color=SCENE_COLORS[j],
                    label=f"{n}, {kind}",
                )
        ax.set_xticks(tolerances, [f"{t} cm" for t in tolerances])
        ax.set_ylim(0, 1)
        ax.set_title(scene)
    axes[0, 0].set_ylabel("fraction within the tolerance")
    axes[0, -1].legend(loc="upper left", fontsize=7)
    fig.suptitle(
        "Precision (solid): predicted points near the ground truth. "
        "Recall (dashed): ground truth near a prediction.",
        x=0.01,
        ha="left",
    )
    fig.tight_layout()
    fig.savefig(out / "precision_recall.png", dpi=130)
    plt.close(fig)


def class_figure(record: dict, names: list[str], name: str, out: Path) -> None:
    """IoU of every evaluated class present in the scene, per shown variant."""
    classes = sorted(
        {c for n in names for c in record[n]["iou_per_class"]},
        key=lambda c: -record[names[0]]["iou_per_class"].get(c, 0),
    )
    fig, ax = plt.subplots(figsize=(max(6.0, 1 + 0.45 * len(classes)), 3.6))
    width = 0.8 / len(names)
    for j, n in enumerate(names):
        values = [record[n]["iou_per_class"].get(c, 0) for c in classes]
        x = np.arange(len(classes)) + (j - (len(names) - 1) / 2) * width
        ax.bar(x, values, width, color=SCENE_COLORS[j], label=n)
    ax.set_xticks(np.arange(len(classes)), classes, rotation=60, ha="right")
    ax.set_ylabel("IoU on ground-truth points")
    ax.set_ylim(0, 1)
    ax.legend(loc="upper right")
    ax.set_title(f"{name}: IoU per class (≥ 100 ground-truth voxels)", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / f"{name}_classes.png", dpi=120)
    plt.close(fig)


def main() -> None:
    """Draw the figures of one run; see the module docstring."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument(
        "--show",
        nargs="+",
        default=["Déjà View", "E_d + E_mv", "all three"],
        help="variants in the per-scene figures (first is the reference)",
    )
    parser.add_argument(
        "--tuning", action="store_true", help="a weight grid: only the score panels"
    )
    args = parser.parse_args()
    run = yaml.safe_load((args.run / "run.yaml").read_text())
    out = Path("figures") / args.run.name
    out.mkdir(parents=True, exist_ok=True)
    if args.tuning:
        summary_figure(run, out, TUNING, "grid")
        print(f"wrote {out}")  # noqa: T201
        return
    summary_figure(run, out)
    precision_recall_figure(run, args.show, out)
    arrays = np.load(args.run / "results.npz")
    names = [v["name"] for v in run["config"]["variants"]]
    device = torch.device("cpu")
    medians = {}
    for scene in run["scenes"]:
        inputs = load_inputs(run["config"], scene, device)
        assert inputs is not None, scene
        assert np.array_equal(arrays[f"{scene}/labels"], inputs.labels.instance_ids)
        depths = {n: arrays[f"{scene}/{names.index(n)}"] for n in args.show}
        variants = {v["name"]: v for v in run["config"]["variants"]}
        medians[scene] = views_figure(inputs, depths, variants, scene, out)
        topdown_figure(inputs, depths, variants, scene, out)
        class_figure(
            run["scenes"][scene]["variants"], [*args.show, "GT geometry"], scene, out
        )
        print(scene, {k: round(v, 3) for k, v in medians[scene].items()}, flush=True)  # noqa: T201
    (out / "pixel_medians.yaml").write_text(yaml.safe_dump(medians, allow_unicode=True))
    print(f"wrote {out}")  # noqa: T201


if __name__ == "__main__":
    main()
