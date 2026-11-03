"""Score pipeline variants on a fixed set of scenes, all in the same way.

Per scene the inputs are cached: views, ground truth, backbone geometry and per-view
SAM 3 masks, linked across views on the backbone geometry exactly as `run-pipeline`
does. Every variant then refines the same backbone geometry (or keeps it), and its
output, one labeled cloud in the first view's frame, is scored as a black box
(`evaluation.labeled`): registered onto the ground truth from the cloud alone, then
accuracy, completeness, F-score at 2 cm, mIoU and AP50. Each variant is registered on
its own.
Disagreement between overlapping views is logged as a diagnostic of the refinement; it
needs no ground truth and is not a score of the output. The labels on the cloud are
always the linked SAM 3 masks, so label scores change only through geometry; `masks`
picks what the refiner sees:

- sam3: the linked SAM 3 masks (the pipeline),
- gt: the ground-truth instances (the oracle: refinement without segmentation errors),
- shuffled: each view gets the next view's SAM 3 masks (same statistics, unrelated
  positions; if this scores like sam3, the masks are not what helps).

`geometry: gt` scores the ground-truth depth with the SAM 3 labels, put through the
same registration: the ceiling of the label scores under perfect geometry, and a check
that registration recovers the identity.

`uv run python -m src.analysis.scene_eval --cfg configs/eval_hypersim.yaml` writes
`runs/<cfg>-<time>/{run.yaml, results.npz}`; `scene_figures --run` draws it.
"""

import argparse
import dataclasses
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
import yaml

from src.evaluation.depth import lift
from src.evaluation.geometry import disagreement_pct
from src.evaluation.labeled import (
    LabeledCloud,
    ground_truth_cloud,
    label_metrics,
    predicted_cloud,
    register,
    surface_metrics,
)
from src.geometry import Geometry
from src.loaders.scannetpp.gt import GroundTruth
from src.loaders.views import Views
from src.refinement.inputs import git_hash
from src.refinement.monocular import RefineConfig, points_from_rays
from src.refinement.multiview import (
    MultiviewConfig,
    Scene,
    SemanticConfig,
    find_matches,
    refine_scene,
    scene_from_geometry,
)
from src.segmentation.base import Segmentation
from src.segmentation.linked import working_associator
from src.segmentation.sam3 import load_prompts

HEADLINE = ("accuracy_cm", "completeness_cm", "fscore", "miou", "ap50")
WORKERS = 6  # scoring processes; each also uses several threads


@dataclasses.dataclass(frozen=True)
class SceneInputs:
    """Everything a variant needs, loaded once per scene."""

    views: Views
    gt: GroundTruth
    geometry: Geometry
    scene: Scene
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
    }
    missing = [str(p) for p in paths.values() if not p.is_file()]
    if missing:
        print(f"{name}: skipped, missing {missing}")  # noqa: T201
        return None
    views, gt = Views.load(paths["views"]), GroundTruth.load(paths["gt"])
    geometry = Geometry.load(paths["geometry"])
    labels = working_associator().associate(Segmentation.load(paths["masks"]), geometry)
    scene = scene_from_geometry(geometry).to(device)
    truth = ground_truth_cloud(
        gt.depth, gt.instance_ids, gt.classes, views.intrinsics, views.cam_to_world
    )
    return SceneInputs(views, gt, geometry, scene, labels, truth)


def ground_truth_points(inputs: SceneInputs) -> np.ndarray:
    """Per-pixel ground-truth points (V, H, W, 3) in the first view's camera frame."""
    views, first = inputs.views, inputs.views.cam_to_world[0].astype(np.float64)
    points = np.stack(
        [
            lift(d.astype(np.float64), k.astype(np.float64), c.astype(np.float64))
            for d, k, c in zip(
                inputs.gt.depth, views.intrinsics, views.cam_to_world, strict=True
            )
        ]
    )
    points = (points - first[:3, 3]) @ first[:3, :3]
    return np.where((inputs.gt.depth > 0)[..., None], points, np.nan)


def refiner_masks(inputs: SceneInputs, kind: str) -> np.ndarray:
    """Instance ids (V, H, W) the refiner sees: sam3, gt or shuffled (see module)."""
    if kind == "gt":
        return np.where(inputs.gt.instance_ids >= 0, inputs.gt.instance_ids, -1)
    if kind == "shuffled":
        return np.roll(inputs.labels.instance_ids, 1, axis=0)
    assert kind == "sam3", kind
    return inputs.labels.instance_ids


@dataclasses.dataclass(frozen=True)
class Scoring:
    """What scoring one scene's outputs needs; sent to the worker processes."""

    truth: LabeledCloud
    labels: Segmentation
    first_camera: np.ndarray  # (4, 4) the ground truth's first camera-to-world
    evaluated: list[str]


def score(scoring: Scoring, points: np.ndarray) -> dict:
    """Register per-pixel points (V, H, W, 3) with SAM 3 labels, score them as a cloud.

    The registration is returned under `align` as a (4, 4) matrix, s R | t.
    """
    labels, truth = scoring.labels, scoring.truth
    finite = np.isfinite(points).all(-1)
    s, rotation, t = register(points[finite], truth.points, scoring.first_camera)
    cloud = predicted_cloud(
        points[finite], labels.instance_ids[finite], labels.classes, labels.scores,
        (s, rotation, t),
    )  # fmt: skip
    align = np.eye(4)
    align[:3, :3], align[:3, 3] = s * rotation, t
    return {
        "scale": s,
        "align": align,
        **surface_metrics(cloud.points, truth.points),
        **label_metrics(truth, cloud, scoring.evaluated),
    }


def refine_variant(
    inputs: SceneInputs, variant: dict, base: tuple[RefineConfig, MultiviewConfig]
) -> tuple[dict, np.ndarray, np.ndarray]:
    """Refine (or not) as the variant says: its record, depth and per-pixel points."""
    if variant.get("geometry") == "gt":
        return {"seconds": 0.0}, inputs.gt.depth, ground_truth_points(inputs)
    scene = inputs.scene
    record: dict = {"seconds": 0.0}
    depth = scene.depth
    if variant.get("refiner", "joint") == "joint":
        cfg = dataclasses.replace(base[0], **variant.get("refine", {}))
        mv = dataclasses.replace(base[1], **variant.get("multiview", {}))
        semantic = SemanticConfig(**variant.get("semantic", {}))
        masks = torch.from_numpy(
            refiner_masks(inputs, variant.get("masks", "sam3")).astype(np.int64)
        )
        t0 = time.perf_counter()
        depth, log, counts = refine_scene(
            scene, cfg, mv, segments=masks, semantic=semantic
        )
        if depth.is_cuda:
            torch.cuda.synchronize()
        record = {
            "seconds": time.perf_counter() - t0,
            "gn_steps": len(log) - 1,
            **counts,
            "energy_initial": log[0],
            "energy_final": log[-1],
        }
    scale = scene.depth.median()
    matches = find_matches(scene, scale / scene.depth, scale, 0.05, 4)
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
    base_cfg = RefineConfig(**refine_raw["refine"])
    cuda = base_cfg.device == "cuda" and torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    base = (
        dataclasses.replace(base_cfg, device=device.type),
        MultiviewConfig(**refine_raw["multiview"]),
    )
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
                base[0],
                dataclasses.replace(base[1], gn_iters=1, cg_iters=1),
            )
            scoring = Scoring(
                inputs.truth, inputs.labels, inputs.views.cam_to_world[0], evaluated
            )
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
