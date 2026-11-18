"""Score point clouds that have no pixel correspondence, in 3D (issue #69).

`scene_eval` scores a point per input pixel, so an output the size and order of the
pixels. A generated cloud (`refinement/generator.py`) has neither. Here every output is
a bare cloud in Deja View's frame and is scored the way `labeled.surface_metrics` scores
pixels, which it reduces to when the cloud is the pixels: accuracy is the mean distance
of each output point to its nearest ground-truth point, completeness the mean distance
of each ground-truth point to its nearest output point, Chamfer their mean.

- Alignment: Deja View's own, one Umeyama similarity of its per-pixel points onto the
  ground truth (`labeled.align_output`), applied to every output. An output that
  carries its own scale or shift is penalised for it; a generator is trained for this
  frame (`volume.gt_in_frame`). `scene_eval` aligns each output on its own pixels.
- Labels: each output point takes the instance id of its nearest Deja View pixel's
  linked SAM 3 mask; each ground-truth point takes the id of its nearest output point,
  and `labeled.label_metrics` scores those. Deja View's pixels go through the same path
  (`pixels`), so the label scores of all variants are comparable, but they are not the
  per-pixel scores of `scene_eval` (nearest point, not same pixel).

Variants: `pixels` (Deja View's points, optionally cut to the top fraction `keep` by
`filters.instance_voxel_density`, E27a), `per_pixel` (the `scene_eval` scores of Deja
View, as the check that this scorer reproduces its surface numbers), `voxel_mean` (one
point, the mean of the pixels, per occupied cell of the generator's grid, optionally
the densest fraction `keep` of the cells) and `cloud`
(a file `<file>_<views>_cloud.npz` of the scene folder).

`uv run python -m src.analysis.cloud_eval --cfg runs/e69a/<rung>_eval.yaml --splits dev`
writes `runs/<cfg>-<time>/run.yaml`.
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from scipy.spatial import cKDTree

from src.analysis.filters import (
    instance_voxel_density,
    keep_top,
    ranks,
    voxel_index,
    voxel_means,
)
from src.analysis.scene_eval import (
    VOCABULARIES,
    SceneInputs,
    git_hash,
    load_inputs,
    split_scenes,
)
from src.evaluation.labeled import (
    LabeledCloud,
    align_output,
    label_metrics,
    nearest_distances,
    score_output,
    split_scores,
    surface_scores,
)
from src.refinement.multiview import points_from_rays
from src.refinement.volume import grid_box
from src.segmentation.base import Segmentation

SHOWN = ("accuracy_cm", "completeness_cm", "chamfer_cm", "points", "miou", "pq", "ap")


def apply_alignment(points: np.ndarray, align: np.ndarray) -> np.ndarray:
    """Return the points (N, 3) under the (4, 4) similarity s R | t."""
    assert points.ndim == 2 and points.shape[1] == 3 and align.shape == (4, 4)  # noqa: PLR2004
    return points @ align[:3, :3].T + align[:3, 3]


def transfer_ids(
    points: np.ndarray, source: np.ndarray, source_ids: np.ndarray
) -> np.ndarray:
    """Instance id (N,) of each point's nearest source point (M, 3)."""
    tree = cKDTree(source, balanced_tree=False, compact_nodes=False)
    return source_ids[tree.query(points, workers=-1)[1]]


def cloud_scores(
    cloud: np.ndarray,
    ids: np.ndarray,
    truth: LabeledCloud,
    labels: Segmentation,
    evaluated: list[str],
) -> dict:
    """Surface and label scores of an aligned cloud (M, 3) with an id (M,) per point."""
    assert cloud.ndim == 2 and cloud.shape[1] == 3 and ids.shape == (len(cloud),)  # noqa: PLR2004
    assert np.isfinite(cloud).all(), "a cloud has no missing points"
    tree = cKDTree(cloud, balanced_tree=False, compact_nodes=False)
    comp, nearest = tree.query(truth.points, workers=-1)
    acc = nearest_distances(truth.points, cloud)
    return (
        surface_scores(acc, comp)
        | {"points": float(len(cloud))}
        | label_metrics(truth, ids[nearest], labels.classes, labels.scores, evaluated)
    )


def variant_cloud(
    variant: dict, inputs: SceneInputs, pixels: np.ndarray, root: Path, views: str
) -> tuple[np.ndarray, np.ndarray]:
    """Return a variant's cloud in Deja View's frame and an id per point."""
    ids = inputs.labels.instance_ids.reshape(-1)
    flat = pixels.reshape(-1, 3)
    kind = variant["kind"]
    if kind == "pixels":
        keep = np.ones(len(flat), dtype=bool)
        if "keep" in variant:
            depth = inputs.scene.depth.cpu().numpy()
            k = inputs.scene.k.cpu().numpy()
            density = instance_voxel_density(
                pixels, depth, k, inputs.labels.instance_ids
            )
            keep = keep_top(ranks(density), variant["keep"]).reshape(-1)
        return flat[keep], ids[keep]
    if kind == "voxel_mean":
        scale = float(inputs.scene.depth.median())
        side = grid_box(flat / scale, variant["resolution"])[1] * scale
        index = voxel_index(flat, side)
        first = np.unique(index, return_index=True)[1]
        cloud, count = voxel_means(flat, side)[first], np.bincount(index)[index[first]]
        if (
            "keep" in variant
        ):  # the densest voxels, as `pixels` keeps the densest pixels
            kept = keep_top(ranks(count.astype(np.float64)), variant["keep"])
            cloud = cloud[kept]
        return cloud, transfer_ids(cloud, flat, ids)
    assert kind == "cloud", variant
    cloud = np.load(root / f"{variant['file']}_{views}_cloud.npz")["points"]
    return cloud.astype(np.float64), transfer_ids(cloud, flat, ids)


def evaluate_scene(raw: dict, name: str, evaluated: list[str]) -> dict[str, dict]:
    """Every variant's scores on one scene."""
    inputs = load_inputs(raw, name, torch.device("cpu"))
    scene = inputs.scene
    pixels = points_from_rays(scene.depth.double(), scene.rays.double()).numpy()
    align = align_output(inputs.truth, pixels)[2]
    out = {}
    for variant in raw["variants"]:
        t0 = time.perf_counter()
        if variant["kind"] == "per_pixel":
            result = score_output(inputs.truth, pixels, inputs.labels, evaluated)
            result.pop("align")
            result.pop("scale")
        else:
            cloud, ids = variant_cloud(
                variant, inputs, pixels, Path(raw["root"]) / name, raw["views"]
            )
            result = cloud_scores(
                apply_alignment(cloud, align),
                ids,
                inputs.truth,
                inputs.labels,
                evaluated,
            )
        out[variant["name"]] = result
        flat = " | ".join(f"{k} {result[k]:.3f}" for k in SHOWN if k in result)
        print(f"{name} | {variant['name']} | {flat} | {time.perf_counter() - t0:.0f} s")  # noqa: T201
    return out


def main() -> None:
    """Score every variant on the scenes of the chosen splits; see the module docstring."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", default=["dev"])
    args = parser.parse_args()
    raw = yaml.safe_load(args.cfg.read_text())
    evaluated = list(VOCABULARIES[raw["vocabulary"]])
    splits = split_scenes(raw, args.splits)
    scenes = {n: evaluate_scene(raw, n, evaluated) for v in splits.values() for n in v}
    report = {}
    for split, names in splits.items():
        report[split] = {"aggregate": {}, "scenes": {n: scenes[n] for n in names}}
        for variant in raw["variants"]:
            agg = split_scores([scenes[n][variant["name"]] for n in names])
            report[split]["aggregate"][variant["name"]] = agg
            flat = " | ".join(f"{k} {agg[k]:.3f}" for k in SHOWN if k in agg)
            print(f"{split} | {variant['name']} | {flat}")  # noqa: T201
    out = Path(raw["output_dir"]) / f"{args.cfg.stem}-{time.strftime('%Y%m%d-%H%M%S')}"
    out.mkdir(parents=True)
    summary = {"config": raw, "git": git_hash(), "splits": report}
    (out / "run.yaml").write_text(yaml.safe_dump(summary, sort_keys=False))
    print(f"wrote {out}")  # noqa: T201


if __name__ == "__main__":
    main()
