"""Score pipeline variants on the scenes of Hypersim's splits, all in the same way.

Per scene the inputs are cached: views, ground truth, backbone geometry, DSINE normals
(`dsine-predict`) and per-view SAM 3 masks, linked across views on the backbone
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
that are scored stay the linked masks. Its `gate` picks, from the same choices, the ids
that alone gate E_mv's matches, and `keep_like` drops E_mv's matches down to the number
another choice's gate keeps on that scene, at random (E8b's controls) or, with
`keep_by: residual`, those of largest initial residual first (E8c).
The config names the splits that get an aggregate (`split_scores`) and those that keep
per-scene records.
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

from src.evaluation.labeled import (
    LabeledCloud,
    ground_truth_cloud,
    score_output,
    split_scores,
)
from src.geometry import Geometry
from src.loaders.ground_truth import GroundTruth
from src.loaders.hypersim.config import load_config
from src.loaders.hypersim.scene import NYU40
from src.loaders.views import Views
from src.refinement.multiview import (
    MatchSelection,
    RefineConfig,
    Scene,
    disagreement_pct,
    find_matches,
    points_from_rays,
    refine_scene,
    same_segment_matches,
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
VOCABULARIES = {"hypersim": NYU40[1:]}  # every class a dataset labels, not a scene's
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
        "normals": folder / f"dsine_{raw['views']}.npz",
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
class Scoring:
    """What scoring one scene's outputs needs; sent to the worker processes."""

    truth: LabeledCloud
    labels: Segmentation
    evaluated: list[str]


def score(scoring: Scoring, points: np.ndarray) -> dict:
    """Score per-pixel points (V, H, W, 3) with the SAM 3 labels; `align` is s R | t."""
    return score_output(scoring.truth, points, scoring.labels, scoring.evaluated)


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


def variant_segments(
    inputs: SceneInputs, variant: dict, key: str = "segments"
) -> torch.Tensor | None:
    """Return the ids variant[key] names: (V, H, W) int64, -1 for none; None for off.

    The choice is `sam3` (the linked masks), `gt` (Hypersim's instances, the
    unlabeled and no-mesh pixels as none), `none`, or the linked masks corrupted:
    `permuted` (`permute_per_view`) or `shifted` by `shift_px` (`shift_per_view`), drawn
    with the variant's `seed`. A missing key falls back to `segments`, its to `sam3`.
    """
    kind = variant.get(key, variant.get("segments", "sam3"))
    rng = np.random.default_rng(variant.get("seed", 0))
    sam3 = inputs.labels.instance_ids
    match kind:
        case "none":
            return None
        case "sam3":
            ids = sam3
        case "gt":
            ids = np.where(
                inputs.gt.instance_ids < 0, NO_INSTANCE, inputs.gt.instance_ids
            )
        case "permuted":
            ids = permute_per_view(sam3, rng)
        case "shifted":
            ids = shift_per_view(sam3, variant["shift_px"], rng)
        case _:
            msg = f"unknown segments: {kind}"
            raise ValueError(msg)
    assert ids.shape == inputs.scene.depth.shape, (ids.shape, inputs.scene.depth.shape)
    return torch.from_numpy(ids.astype(np.int64))


def gated_count(inputs: SceneInputs, ids: torch.Tensor | None, neighbors: int) -> int:
    """Return how many of E_mv's matches the gate on ids keeps (all of them for None)."""
    scene = inputs.scene
    scale = scene.depth.median()
    d0 = scale / scene.depth
    matches = find_matches(scene, d0, scale, neighbors)
    if ids is None:
        return len(matches.src)
    return len(same_segment_matches(matches, scene, d0, scale, ids.to(d0.device)).src)


def refine_variant(
    inputs: SceneInputs, variant: dict, base: RefineConfig
) -> tuple[float, dict, np.ndarray, np.ndarray]:
    """Refine (or not) as the variant says: runtime, log, depth and per-pixel points."""
    scene = inputs.scene
    seconds, log_record = 0.0, {}
    depth = scene.depth
    if variant["refiner"] == "joint":
        cfg = dataclasses.replace(base, **variant.get("refine", {}))
        segments = variant_segments(inputs, variant)
        gate = variant_segments(inputs, variant, "gate")
        keep = (
            gated_count(
                inputs, variant_segments(inputs, variant, "keep_like"), cfg.neighbors
            )
            if "keep_like" in variant
            else None
        )
        t0 = time.perf_counter()
        select = MatchSelection(
            gate, keep, variant.get("seed", 0), variant.get("keep_by", "random")
        )
        depth, log, counts = refine_scene(scene, cfg, inputs.normals, segments, select)
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
    return seconds, log_record, depth.cpu().numpy(), points


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
                scoring = Scoring(inputs.truth, inputs.labels, evaluated)
                keep = split in raw["per_scene"]
                if keep:
                    arrays[f"{name}/labels"] = inputs.labels.instance_ids
                for i, variant in enumerate(raw["variants"]):
                    seconds, log, depth, points = refine_variant(inputs, variant, base)
                    if keep:
                        arrays[f"{name}/{i}"] = depth.astype(np.float32)
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
