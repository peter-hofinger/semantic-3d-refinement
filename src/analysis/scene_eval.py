"""Score pipeline variants on a fixed set of scenes, all in the same way.

Per scene the inputs are cached: views, ground truth, backbone geometry, DSINE normals
(`dsine-predict`) and per-view SAM 3 masks, linked across views on the backbone
geometry exactly as `run-pipeline` does. Every variant then refines the same backbone
geometry (`refiner: joint`, with the variant's overrides of `configs/refine.yaml`) or
keeps it (`refiner: none`), and its output, a labeled point per input pixel in the
first view's frame, is scored (`evaluation.labeled`): aligned onto the ground truth
by Umeyama on the pixels, then surface and label scores, per scene and per class. The
labels are the linked SAM 3 masks and are scored per pixel, so every variant has the
same label scores; refinement shows in the surface scores.
Disagreement between overlapping views is logged as a diagnostic of the refinement; it
needs no ground truth and is not a score of the output.

`uv run python -m src.analysis.scene_eval --cfg configs/eval_hypersim.yaml` writes
`runs/<cfg>-<time>/{run.yaml, results.npz}`; `scene_figures --run` draws it.
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

from src.evaluation.labeled import LabeledCloud, ground_truth_cloud, score_output
from src.geometry import Geometry
from src.loaders.ground_truth import GroundTruth
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
from src.segmentation.sam3 import load_prompts

HEADLINE = (
    "accuracy_cm",
    "completeness_cm",
    "miou",
    "miou_th",
    "pq",
    "pq_th",
    "pq_st",
    "ap",
)
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


def load_inputs(raw: dict, name: str, device: torch.device) -> SceneInputs | None:
    """Load one scene's cached inputs (None if any is missing), link the SAM 3 masks."""
    folder = Path(raw["root"]) / name
    paths = {
        "views": folder / f"{raw['views']}.npz",
        "gt": folder / f"gt_{raw['views']}.npz",
        "geometry": folder / f"{raw['backbone']}_{raw['views']}.npz",
        "masks": folder / f"{raw['segmentation']}_{raw['views']}.npz",
        "normals": folder / f"dsine_{raw['views']}.npz",
    }
    missing = [str(p) for p in paths.values() if not p.is_file()]
    if missing:
        print(f"{name}: skipped, missing {missing}")  # noqa: T201
        return None
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


def refine_variant(
    inputs: SceneInputs, variant: dict, base: RefineConfig
) -> tuple[dict, np.ndarray, np.ndarray]:
    """Refine (or not) as the variant says: its record, depth and per-pixel points."""
    scene = inputs.scene
    record: dict = {"seconds": 0.0}
    depth = scene.depth
    if variant["refiner"] == "joint":
        cfg = dataclasses.replace(base, **variant.get("refine", {}))
        segments = torch.from_numpy(inputs.labels.instance_ids.astype(np.int64))
        t0 = time.perf_counter()
        depth, log, counts = refine_scene(scene, cfg, inputs.normals, segments)
        if depth.is_cuda:
            torch.cuda.synchronize()
        record = {
            "seconds": time.perf_counter() - t0,
            "gn_steps": len(log) - 1,
            **counts,
            "energy_initial": log[0],
            "energy_final": log[-1],
        }
    else:
        assert variant["refiner"] == "none", variant
    scale = scene.depth.median()
    matches = find_matches(scene, scale / scene.depth, scale, base.neighbors)
    record["disagreement_pct"] = disagreement_pct(depth, scene, matches)
    points = points_from_rays(depth.double(), scene.rays.double()).cpu().numpy()
    return record, depth.cpu().numpy(), points


def main() -> None:
    """Score every variant on every scene of a config; see the module docstring.

    Refinement runs here, on the GPU, one variant after another; the outputs are
    scored meanwhile by `WORKERS` processes on the CPU.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=Path, required=True)
    cfg_path: Path = parser.parse_args().cfg
    raw = yaml.safe_load(cfg_path.read_text())
    refine_raw = yaml.safe_load(Path(raw["refine"]).read_text())
    base = RefineConfig(**refine_raw["refine"])
    cuda = base.device == "cuda" and torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    base = dataclasses.replace(base, device=device.type)
    evaluated = sorted(set(load_prompts([Path(raw["classes"])]).values()))

    out = Path(raw["output_dir"]) / f"{cfg_path.stem}-{time.strftime('%Y%m%d-%H%M%S')}"
    out.mkdir(parents=True)
    scenes, arrays, jobs = {}, {}, {}
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(WORKERS, mp_context=context) as pool:
        for name in raw["scenes"]:
            inputs = load_inputs(raw, name, device)
            if inputs is None:
                continue
            refine_scene(  # warm-up, so the first variant's time is not the CUDA start
                inputs.scene,
                dataclasses.replace(base, gn_iters=1, cg_iters=1),
                inputs.normals,
            )
            scoring = Scoring(inputs.truth, inputs.labels, evaluated)
            scenes[name] = {
                "truth_points": len(inputs.truth.points),
                "objects": len(inputs.labels.classes),
                "variants": {},
            }
            arrays[f"{name}/labels"] = inputs.labels.instance_ids
            for i, variant in enumerate(raw["variants"]):
                record, depth, points = refine_variant(inputs, variant, base)
                scenes[name]["variants"][variant["name"]] = record
                arrays[f"{name}/{i}"] = depth.astype(np.float32)
                jobs[name, i] = pool.submit(score, scoring, points)
        for (name, i), job in jobs.items():
            scores = job.result()
            arrays[f"{name}/{i}/align"] = scores.pop("align")
            variant = raw["variants"][i]["name"]
            record = scenes[name]["variants"][variant]
            record |= scores
            print(  # noqa: T201
                f"{name} | {variant} | "
                + " | ".join(f"{k} {record[k]:.3f}" for k in HEADLINE)
                + f" | {record['seconds']:.1f} s",
                flush=True,
            )

    summary = {
        "config": raw,
        "refine": refine_raw,
        "evaluated_classes": evaluated,
        "git": git_hash(),
        "scenes": scenes,
    }
    (out / "run.yaml").write_text(
        yaml.safe_dump(summary, sort_keys=False, allow_unicode=True)
    )
    np.savez_compressed(out / "results.npz", **arrays)
    print(f"wrote {out}")  # noqa: T201


if __name__ == "__main__":
    main()
