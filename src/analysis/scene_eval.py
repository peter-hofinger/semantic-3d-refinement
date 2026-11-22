"""Score pipeline variants on the scenes of Hypersim's splits, all in the same way.

Per scene the inputs are cached: views, ground truth, backbone geometry, normals
(DSINE's from `dsine-predict`, or another source's `<normals>_<views>.npz` that the
config names, e.g. MoGe-3's in E21b) and per-view SAM 3 masks, linked across views on the backbone
geometry exactly as `run-pipeline` does. Every variant then refines the same backbone
geometry (`refiner: joint`, with the variant's overrides of `configs/refine.yaml`) or
keeps it (`refiner: none`), and its output, a labeled point per input pixel in the
first view's frame, is scored (`evaluation.labeled`): aligned onto the ground truth
by Umeyama on the pixels, then surface and label scores, per scene and per class. The
labels are the linked SAM 3 masks and are scored per pixel, so every variant has the
same label scores; refinement shows in the surface scores. A segmenter that links its
masks across views itself (SegVGGT) sets `linked: true`; its masks skip the association.
A joint variant's `segments` picks what the refiner groups by, for the segment ablation
(E8a): the linked masks (default), the ground truth's instances, none, or the linked
masks with their identities or boundaries corrupted (`variant_segments`); the labels
that are scored stay the linked masks.
A variant's `keep: q` cuts its output to the top fraction q of the pixels by instance
voxel density (`filters.instance_voxel_density`, E45's best filter), flagged on the
backbone's cloud: flag, refine every pixel, then drop the flagged (E27a). The dropped
pixels have no point, so they count for completeness only; the label scores are per
pixel and stay as they are. `keep_segments` picks the cut's instances as `segments`
picks the refiner's (`none`: plain voxel density, E86a). With `keep_by: <name>` the pixels are ranked instead by the
score `<name>_<views>.npz` holds in the scene's folder (key `keep`, higher keeps), such
as the 3D refiner's keep logit (issue #57), at the same fraction; `keep_with: density`
ranks by both scores (`filters.both`), and `keep_per: instance` cuts each instance to q.
With `fill` the pixels the cut drops are kept and get a new depth from their view's
kept pixels instead (`filters.fill_harmonic`, E58a): `plain`, `rgb` (edges weighted by
colour), `sam3` or `gt` (edges cut between the linked masks' or the ground truth's
instances).
The config names the splits that get an aggregate (`split_scores`) and those that keep
per-scene records. `surface: tsdf` with `tsdf: {voxel, trunc}` (metres) adds the fused
surface scores (`fused_*`, `evaluation.labeled`) next to the per-pixel ones; without
it (`surface: pixels`, the default) nothing changes.
Every variant's `log` holds the refinement's own diagnostics, such as the disagreement
between overlapping views; it is not a score of the output.

`uv run python -m src.analysis.scene_eval --cfg configs/eval_hypersim.yaml --splits dev`
writes `runs/<cfg>-<time>/{run.yaml, results.npz}`; `scene_figures --run` draws it.
"""

import argparse
import dataclasses
import multiprocessing
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
import yaml

from src.analysis.filters import (
    both,
    fill_harmonic,
    instance_voxel_density,
    keep_top,
    keep_top_within,
    ranks,
)
from src.evaluation.labeled import (
    LabeledCloud,
    fused_metrics,
    ground_truth_cloud,
    score_output,
    split_scores,
    tsdf_points,
)
from src.geometry import Geometry
from src.loaders.ground_truth import GroundTruth
from src.loaders.hypersim.config import load_config
from src.loaders.hypersim.scene import NYU40
from src.loaders.kitti360.scene import CLASSES as KITTI360
from src.loaders.kitti360.scene import SKY
from src.loaders.views import Views
from src.refinement.multiview import (
    RefineConfig,
    Scene,
    disagreement_pct,
    find_matches,
    points_from_rays,
    refine_scene,
    scene_from_geometry,
    scene_normals,
)
from src.segmentation.base import NO_INSTANCE, Segmentation
from src.segmentation.linked import working_associator

# A variant's scores have three tiers: `headline`, the paper's table; `supplementary`,
# every other scene-level score and the refinement's runtime; `debug`, the per-class
# scores and the alignment scale, for the team only.
HEADLINE = (
    "accuracy_cm",
    "completeness_cm",
    "chamfer_cm",
    "miou",
    "pq",
    "ap",
)
# every class a dataset labels with geometry, not a scene's
VOCABULARIES = {
    "hypersim": NYU40[1:],
    "kitti360": tuple(c for i, c in KITTI360.items() if i != SKY),
}
WORKERS = 6  # scoring processes; each also uses several threads


def git_hash() -> str:
    """Return HEAD, marked dirty if the tree has uncommitted changes."""
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],  # noqa: S607
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain"],  # noqa: S607
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    return head + ("-dirty" if dirty else "")


@dataclasses.dataclass(frozen=True)
class SceneInputs:
    """Everything a variant needs, loaded once per scene."""

    views: Views
    gt: GroundTruth
    geometry: Geometry
    scene: Scene
    normals: torch.Tensor  # (V, H, W, 3) DSINE's, in the frame of the scene's rays
    labels: Segmentation
    truth: LabeledCloud


def input_paths(raw: dict, name: str) -> dict[str, Path]:
    """Return the cached input files of one scene."""
    folder = Path(raw["root"]) / name
    return {
        "views": folder / f"{raw['views']}.npz",
        "gt": folder / f"gt_{raw['views']}.npz",
        "geometry": folder / f"{raw['backbone']}_{raw['views']}.npz",
        "masks": folder / f"{raw['segmentation']}_{raw['views']}.npz",
        "normals": folder / f"{raw.get('normals', 'dsine')}_{raw['views']}.npz",
    }


def split_scenes(raw: dict, splits: list[str]) -> dict[str, list[str]]:
    """Scene folders per split; every input must exist, as a gap would skew an aggregate."""
    config = load_config(Path(raw["dataset"]))
    scenes = {
        split: [f"{s}_{c}" for s, c in config.scenes([split]).items()]
        for split in splits
    }
    missing = [
        str(path)
        for names in scenes.values()
        for name in names
        for path in input_paths(raw, name).values()
        if not path.is_file()
    ]
    if missing:
        msg = f"missing inputs: {missing}"
        raise FileNotFoundError(msg)
    return scenes


def load_inputs(raw: dict, name: str, device: torch.device) -> SceneInputs:
    """Load one scene's cached inputs and link the SAM 3 masks."""
    paths = input_paths(raw, name)
    views, gt = Views.load(paths["views"]), GroundTruth.load(paths["gt"])
    geometry = Geometry.load(paths["geometry"])
    masks = Segmentation.load(paths["masks"])
    linked = raw.get("linked", False)
    labels = masks if linked else working_associator().associate(masks, geometry)
    scene = scene_from_geometry(geometry).to(device)
    normals = scene_normals(np.load(paths["normals"])["normals"], geometry, scene)
    truth = ground_truth_cloud(
        gt.depth, gt.instance_ids, gt.classes, views.intrinsics, views.cam_to_world
    )
    return SceneInputs(views, gt, geometry, scene, normals, labels, truth)


@dataclasses.dataclass(frozen=True)
class Fusion:
    """The fused-surface score's inputs: GT surface, output pinholes, TSDF settings."""

    truth: np.ndarray  # (M, 3) the ground truth's fused surface
    k: np.ndarray  # (V, 3, 3) the output's pinholes, pixel centers at integers
    cam_to_world: np.ndarray  # (V, 4, 4)
    tsdf: dict[str, float]  # voxel, trunc in metres


@dataclasses.dataclass(frozen=True)
class Scoring:
    """What scoring one scene's outputs needs; sent to the worker processes."""

    truth: LabeledCloud
    labels: Segmentation
    evaluated: list[str]
    fusion: Fusion | None = None  # `surface: tsdf` only


def score(scoring: Scoring, points: np.ndarray) -> dict:
    """Score per-pixel points (V, H, W, 3) with the SAM 3 labels; `align` is s R | t.

    The fused score fuses the same pixels as the per-pixel one, those with GT depth.
    """
    out = score_output(scoring.truth, points, scoring.labels, scoring.evaluated)
    f = scoring.fusion
    if f is not None:
        cameras = (f.k, f.cam_to_world)
        scored = np.where(scoring.truth.pixels[..., None], points, np.nan)
        out |= fused_metrics(
            scored.astype(np.float64), cameras, out["align"], f.truth, f.tsdf
        )
    return out


def fusion(raw: dict, inputs: SceneInputs) -> Fusion | None:
    """Return the scene's Fusion if the config asks for `surface: tsdf`, else None."""
    surface = raw.get("surface", "pixels")
    assert surface in {"pixels", "tsdf"}, surface
    if surface == "pixels":
        return None
    tsdf = {"voxel": float(raw["tsdf"]["voxel"]), "trunc": float(raw["tsdf"]["trunc"])}
    gt_k = inputs.views.intrinsics.astype(np.float64).copy()
    gt_k[:, :2, 2] -= 0.5  # COLMAP's pixel centers at +0.5 to Open3D's at integers
    truth = tsdf_points(
        inputs.gt.depth.astype(np.float64),
        gt_k,
        inputs.views.cam_to_world.astype(np.float64),
        tsdf["voxel"],
        tsdf["trunc"],
    )
    scene = inputs.scene
    cam_to_world = np.tile(np.eye(4), (len(scene.k), 1, 1))
    cam_to_world[:, :3, :3] = scene.rotation.double().cpu().numpy()
    cam_to_world[:, :3, 3] = scene.center.double().cpu().numpy()
    return Fusion(truth, scene.k.double().cpu().numpy(), cam_to_world, tsdf)


def tiers(scores: dict) -> dict[str, dict]:
    """Sort a variant's flat scores into the three tiers (see HEADLINE)."""
    debug = {k: v for k, v in scores.items() if isinstance(v, dict) or k == "scale"}
    return {
        "headline": {k: scores[k] for k in HEADLINE},
        "supplementary": {
            k: v for k, v in scores.items() if k not in {*HEADLINE, *debug}
        },
        "debug": debug,
    }


def summary_line(label: str, variant: str, scores: dict) -> str:
    """One variant's headline scores and runtime, for the console."""
    values = " | ".join(f"{k} {scores[k]:.3f}" for k in HEADLINE)
    return f"{label} | {variant} | {values} | {scores['refinement_s']:.1f} s"


def permute_per_view(ids: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Relabel each view's instances by its own random permutation of its ids.

    Grouping and boundaries within a view stay; which instance an id names across views
    is scrambled (one permutation for all views would be a no-op). NO_INSTANCE stays.
    """
    assert ids.ndim == 3 and ids.dtype == np.int32, (ids.shape, ids.dtype)  # noqa: PLR2004
    out = ids.copy()
    for view, view_ids in zip(out, ids, strict=True):
        labeled = view_ids != NO_INSTANCE
        own, index = np.unique(view_ids[labeled], return_inverse=True)
        view[labeled] = rng.permutation(own)[index]
    assert ((out == NO_INSTANCE) == (ids == NO_INSTANCE)).all()
    return out


def shift_per_view(
    ids: np.ndarray, pixels: int, rng: np.random.Generator
) -> np.ndarray:
    """Translate each view's id map by `pixels` along a random axis direction.

    The masks keep their ids and shapes but sit off the image's edges by `pixels`; the
    border they uncover gets NO_INSTANCE.
    """
    assert ids.ndim == 3 and ids.dtype == np.int32 and pixels > 0, ids.shape  # noqa: PLR2004
    out = np.full_like(ids, NO_INSTANCE)
    h, w = ids.shape[1:]
    for view in range(len(ids)):
        dy, dx = ((pixels, 0), (-pixels, 0), (0, pixels), (0, -pixels))[rng.integers(4)]
        out[view, max(dy, 0) : h + min(dy, 0), max(dx, 0) : w + min(dx, 0)] = ids[
            view, max(-dy, 0) : h + min(-dy, 0), max(-dx, 0) : w + min(-dx, 0)
        ]
    return out


def segment_ids(
    kind: str, sam3: np.ndarray, gt: np.ndarray, variant: dict
) -> np.ndarray | None:
    """Return (V, H, W) int32 instance ids of the given kind, NO_INSTANCE for none.

    `kind` is `sam3` (the linked masks), `gt` (Hypersim's instances, the unlabeled and
    no-mesh pixels as none), `none` (None), or the linked masks corrupted: `permuted`
    (`permute_per_view`) or `shifted` by the variant's `shift_px` (`shift_per_view`),
    drawn with the variant's `seed`.
    """
    assert sam3.shape == gt.shape and sam3.dtype == np.int32, (sam3.shape, gt.shape)
    rng = np.random.default_rng(variant.get("seed", 0))
    match kind:
        case "none":
            return None
        case "sam3":
            return sam3
        case "gt":
            return np.where(gt < 0, NO_INSTANCE, gt).astype(np.int32)
        case "permuted":
            return permute_per_view(sam3, rng)
        case "shifted":
            return shift_per_view(sam3, variant["shift_px"], rng)
        case _:
            msg = f"unknown segments: {kind}"
            raise ValueError(msg)


def variant_segments(inputs: SceneInputs, variant: dict) -> torch.Tensor | None:
    """Return the refiner's segments (`segment_ids` of `segments`, default sam3) as int64."""
    ids = segment_ids(
        variant.get("segments", "sam3"),
        inputs.labels.instance_ids,
        inputs.gt.instance_ids,
        variant,
    )
    if ids is None:
        return None
    assert ids.shape == inputs.scene.depth.shape, (ids.shape, inputs.scene.depth.shape)
    return torch.from_numpy(ids.astype(np.int64))


def keep_ids(inputs: SceneInputs, variant: dict) -> np.ndarray:
    """Return the cut's instance ids: `segment_ids` of `keep_segments` (default sam3).

    `none` puts every pixel in one instance, which makes instance voxel density plain
    voxel density (the semantic ablation of the filter, E86a).
    """
    sam3 = inputs.labels.instance_ids
    kind = variant.get("keep_segments", "sam3")
    ids = segment_ids(kind, sam3, inputs.gt.instance_ids, variant)
    return np.full_like(sam3, NO_INSTANCE) if ids is None else ids


def refine_variant(
    inputs: SceneInputs,
    variant: dict,
    base: RefineConfig,
    keep_score: np.ndarray | None = None,
) -> tuple[float, dict, np.ndarray, np.ndarray]:
    """Refine (or not) as the variant says: runtime, log, depth and per-pixel points.

    `keep_score` (V, H, W) replaces instance voxel density as the cut's ranking.
    """
    scene = inputs.scene
    seconds, log_record = 0.0, {}
    depth = scene.depth
    if variant["refiner"] == "joint":
        cfg = dataclasses.replace(base, **variant.get("refine", {}))
        segments = variant_segments(inputs, variant)
        t0 = time.perf_counter()
        depth, log, counts = refine_scene(scene, cfg, inputs.normals, segments)
        if depth.is_cuda:
            torch.cuda.synchronize()
        seconds = time.perf_counter() - t0
        log_record = {
            "gn_steps": len(log) - 1,
            **counts,
            "energy_initial": log[0],
            "energy_final": log[-1],
        }
    else:
        assert variant["refiner"] == "none", variant
    scale = scene.depth.median()
    matches = find_matches(scene, scale / scene.depth, scale, base.neighbors)
    log_record["disagreement_pct"] = disagreement_pct(depth, scene, matches)
    points = points_from_rays(depth.double(), scene.rays.double()).cpu().numpy()
    if "keep" in variant:
        ids = keep_ids(inputs, variant)
        if keep_score is None or variant.get("keep_with") == "density":
            start = points_from_rays(scene.depth.double(), scene.rays.double())
            density = instance_voxel_density(
                start.cpu().numpy(),
                scene.depth.cpu().numpy(),
                scene.k.cpu().numpy(),
                ids,
            )
            keep_score = density if keep_score is None else both(density, keep_score)
        assert keep_score.shape == points.shape[:3], keep_score.shape
        if variant.get("keep_per") == "instance":
            kept = keep_top_within(keep_score, ids, variant["keep"])
        else:
            kept = keep_top(ranks(keep_score), variant["keep"])
        if "fill" in variant:
            filled = fill_variant(inputs, variant["fill"], depth.cpu().numpy(), kept)
            depth = torch.from_numpy(filled).to(scene.rays.device)
            points = points_from_rays(depth.double(), scene.rays.double()).cpu().numpy()
        else:
            points[~kept] = np.nan
    return seconds, log_record, depth.cpu().numpy(), points


def fill_variant(
    inputs: SceneInputs, kind: str, depth: np.ndarray, kept: np.ndarray
) -> np.ndarray:
    """Depth with the flagged pixels filled from the kept ones, as `fill` says."""
    match kind:
        case "plain":
            return fill_harmonic(depth, kept)
        case "rgb":
            return fill_harmonic(depth, kept, images=inputs.views.images)
        case "sam3":
            return fill_harmonic(depth, kept, ids=inputs.labels.instance_ids)
        case "gt":
            return fill_harmonic(depth, kept, ids=inputs.gt.instance_ids)
        case _:
            msg = f"unknown fill: {kind}"
            raise ValueError(msg)


def evaluate_scenes(
    raw: dict,
    splits: dict[str, list[str]],
    base: RefineConfig,
    evaluated: list[str],
) -> tuple[dict, dict, dict]:
    """Refine and score every scene: flat scores and logs per (scene, variant), arrays.

    Refinement runs here, on the GPU, one variant after another; the outputs are
    scored meanwhile by `WORKERS` processes on the CPU. Only `per_scene` splits keep
    arrays, for the figures.
    """
    device = torch.device(base.device)
    variants = [v["name"] for v in raw["variants"]]
    scores, logs, arrays, jobs = {}, {}, {}, {}
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(WORKERS, mp_context=context) as pool:
        for split, names in splits.items():
            for name in names:
                inputs = load_inputs(raw, name, device)
                if not jobs:  # so the first variant's time is not the CUDA start
                    refine_scene(
                        inputs.scene,
                        dataclasses.replace(base, gn_iters=1, cg_iters=1),
                        inputs.normals,
                    )
                scoring = Scoring(
                    inputs.truth, inputs.labels, evaluated, fusion(raw, inputs)
                )
                keep = split in raw["per_scene"]
                if keep:
                    arrays[f"{name}/labels"] = inputs.labels.instance_ids
                for i, variant in enumerate(raw["variants"]):
                    score_file = (
                        Path(raw["root"])
                        / name
                        / (f"{variant.get('keep_by')}_{raw['views']}.npz")
                    )
                    keep_score = (
                        np.load(score_file)["keep"] if "keep_by" in variant else None
                    )
                    seconds, log, depth, points = refine_variant(
                        inputs, variant, base, keep_score
                    )
                    if keep:
                        arrays[f"{name}/{i}"] = depth.astype(np.float32)
                        arrays[f"{name}/{i}/kept"] = np.isfinite(points).all(-1)
                    scores[name, variants[i]] = {"refinement_s": seconds}
                    logs[name, variants[i]] = log
                    jobs[name, i] = pool.submit(score, scoring, points)
        for (name, i), job in jobs.items():
            result = job.result()
            align = result.pop("align")
            if f"{name}/{i}" in arrays:
                arrays[f"{name}/{i}/align"] = align
            flat = scores[name, variants[i]] | result
            scores[name, variants[i]] = flat
            print(summary_line(name, variants[i], flat), flush=True)  # noqa: T201
    return scores, logs, arrays


def split_report(
    raw: dict, splits: dict[str, list[str]], scores: dict, logs: dict
) -> dict:
    """Return the tiers per split: its aggregate and scene records, as configured."""
    variants = [v["name"] for v in raw["variants"]]
    report = {}
    for split, names in splits.items():
        report[split] = {}
        if split in raw["aggregated"]:
            aggregate = {
                v: split_scores([scores[n, v] for n in names]) for v in variants
            }
            report[split]["aggregate"] = {v: tiers(a) for v, a in aggregate.items()}
            for v, a in aggregate.items():
                print(summary_line(split, v, a), flush=True)  # noqa: T201
        if split in raw["per_scene"]:
            report[split]["scenes"] = {
                n: {
                    "variants": {
                        v: tiers(scores[n, v]) | {"log": logs[n, v]} for v in variants
                    }
                }
                for n in names
            }
    return report


def main() -> None:
    """Score every variant on the scenes of the chosen splits; see the module docstring."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=Path, required=True)
    parser.add_argument(
        "--splits", nargs="+", default=["dev"], help="dev val test stress"
    )
    args = parser.parse_args()
    cfg_path: Path = args.cfg
    raw = yaml.safe_load(cfg_path.read_text())
    refine_raw = yaml.safe_load(Path(raw["refine"]).read_text())
    base = RefineConfig(**refine_raw["refine"])
    cuda = base.device == "cuda" and torch.cuda.is_available()
    base = dataclasses.replace(base, device="cuda" if cuda else "cpu")
    evaluated = list(VOCABULARIES[raw["vocabulary"]])
    splits = split_scenes(raw, args.splits)

    out = Path(raw["output_dir"]) / f"{cfg_path.stem}-{time.strftime('%Y%m%d-%H%M%S')}"
    out.mkdir(parents=True)
    scores, logs, arrays = evaluate_scenes(raw, splits, base, evaluated)
    summary = {
        "config": raw,
        "refine": refine_raw,
        "evaluated_classes": evaluated,
        "git": git_hash(),
        "splits": split_report(raw, splits, scores, logs),
    }
    (out / "run.yaml").write_text(
        yaml.safe_dump(summary, sort_keys=False, allow_unicode=True)
    )
    np.savez_compressed(out / "results.npz", **arrays)
    print(f"wrote {out}")  # noqa: T201


if __name__ == "__main__":
    main()
