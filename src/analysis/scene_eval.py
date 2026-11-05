"""Score pipeline variants on the scenes of Hypersim's splits, all in the same way.

Per scene the inputs are cached: views, ground truth, backbone geometry, DSINE normals
(`dsine-predict`) and per-view SAM 3 masks, linked across views on the backbone
geometry exactly as `run-pipeline` does. Every variant then refines the same backbone
geometry (`refiner: joint`, with the variant's overrides of `configs/refine.yaml`) or
keeps it (`refiner: none`), and its output, a labeled point per input pixel in the
first view's frame, is scored (`evaluation.labeled`): aligned onto the ground truth
by Umeyama on the pixels, then surface and label scores, per scene and per class. The
labels are the linked SAM 3 masks and are scored per pixel, so every variant has the
same label scores; refinement shows in the surface scores. The config names the splits
that get an aggregate (`split_scores`) and those that keep per-scene records.
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
    RefineConfig,
    Scene,
    disagreement_pct,
    find_matches,
    points_from_rays,
    refine_scene,
    scene_from_geometry,
    scene_normals,
)
from src.segmentation.base import Segmentation
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
    labels = working_associator().associate(Segmentation.load(paths["masks"]), geometry)
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


def refine_variant(
    inputs: SceneInputs, variant: dict, base: RefineConfig
) -> tuple[float, dict, np.ndarray, np.ndarray]:
    """Refine (or not) as the variant says: runtime, log, depth and per-pixel points."""
    scene = inputs.scene
    seconds, log_record = 0.0, {}
    depth = scene.depth
    if variant["refiner"] == "joint":
        cfg = dataclasses.replace(base, **variant.get("refine", {}))
        segments = torch.from_numpy(inputs.labels.instance_ids.astype(np.int64))
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
