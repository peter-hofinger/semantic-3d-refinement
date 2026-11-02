"""Score pipeline variants on a fixed set of scenes, all in the same way.

Per scene the inputs are cached: views, ground truth, backbone geometry and per-view
SAM 3 masks, linked across views on the backbone geometry exactly as `run-pipeline`
does. Every variant then refines the same backbone geometry (or keeps it) and is scored
as a labeled cloud (`evaluation.labeled`): surface F-score, mIoU and instance AP
against the ground truth, plus the disagreement between views, which needs none, and
each pixel's depth error (AbsRel: mean |s D - D_gt| / D_gt, s the alignment's scale).
The cloud scores and AbsRel can disagree: a cloud of views that disagree is thick, and
its layers can straddle the true surface, which nearest-neighbor completeness rewards. The
labels on the cloud are always the linked SAM 3 masks, so label scores change only
through geometry; `masks` picks what the refiner sees:

- sam3: the linked SAM 3 masks (the pipeline),
- gt: the ground-truth instances (the oracle: refinement without segmentation errors),
- shuffled: each view gets the next view's SAM 3 masks (same statistics, unrelated
  positions; if this scores like sam3, the masks are not what helps).

`geometry: gt` scores ground-truth depth and cameras with the SAM 3 labels, the
ceiling of the label scores under perfect geometry. `cameras: gt` puts the backbone's
depth, scaled by the alignment, on the ground-truth cameras before refining: it
separates errors of the depth from errors of the predicted cameras, which the
refinement keeps fixed.

`uv run python -m src.analysis.scene_eval --cfg configs/eval_hypersim.yaml` writes
`runs/<cfg>-<time>/{run.yaml, results.npz}`; `scene_figures --run` draws it.
"""

import argparse
import dataclasses
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from src.evaluation.depth import lift
from src.evaluation.geometry import disagreement_pct
from src.evaluation.labeled import (
    LabeledCloud,
    align_points,
    ground_truth_cloud,
    label_metrics,
    predicted_cloud,
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

HEADLINE = ("abs_rel", "accuracy_cm", "completeness_cm", "fscore_2cm", "miou", "ap50")


@dataclasses.dataclass(frozen=True)
class SceneInputs:
    """Everything a variant needs, loaded once per scene."""

    views: Views
    gt: GroundTruth
    geometry: Geometry
    scene: Scene
    labels: Segmentation
    truth: LabeledCloud
    align: tuple[float, np.ndarray, np.ndarray]


def load_inputs(raw: dict, name: str, device: torch.device) -> SceneInputs | None:
    """Load one scene's cached inputs (None if any is missing), link the SAM 3 masks.

    The similarity into the ground truth's frame is fitted once, on the backbone's
    points, and shared by every variant: refinement keeps the cameras, so its points
    stay in the backbone's frame.
    """
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
    gt_points = np.stack(
        [
            lift(d.astype(np.float64), k.astype(np.float64), c.astype(np.float64))
            for d, k, c in zip(
                gt.depth, views.intrinsics, views.cam_to_world, strict=True
            )
        ]
    )
    points = points_from_rays(scene.depth.double(), scene.rays.double()).cpu().numpy()
    align = align_points(points, gt_points, gt.depth > 0)
    return SceneInputs(views, gt, geometry, scene, labels, truth, align)


def gt_camera_scene(inputs: SceneInputs) -> Scene:
    """Backbone depth in meters (alignment scale) on the ground-truth pinholes."""
    views, scene = inputs.views, inputs.scene
    k = views.intrinsics.astype(np.float64).copy()
    k[:, :2, 2] -= 0.5  # Scene's pixel centers are at integers
    h, w = scene.depth.shape[1:]
    v, u = np.mgrid[0:h, 0:w].astype(np.float64)
    pixels = np.stack([u, v, np.ones_like(u)], -1)
    c2w = views.cam_to_world.astype(np.float64)
    directions = np.einsum("nij,hwj->nhwi", c2w[:, :3, :3] @ np.linalg.inv(k), pixels)
    origins = np.broadcast_to(c2w[:, None, None, :3, 3], directions.shape)
    device = scene.depth.device
    return Scene(
        inputs.align[0] * scene.depth,
        torch.from_numpy(np.concatenate([origins, directions], -1)).float().to(device),
        scene.conf,
        torch.from_numpy(k).float().to(device),
        torch.from_numpy(c2w[:, :3, :3]).float().to(device),
        torch.from_numpy(c2w[:, :3, 3]).float().to(device),
    )


def variant_scene(inputs: SceneInputs, variant: dict) -> tuple[Scene, tuple]:
    """Return the scene a variant refines and its similarity into the GT frame."""
    if variant.get("cameras") == "gt":
        return gt_camera_scene(inputs), (1.0, np.eye(3), np.zeros(3))
    return inputs.scene, inputs.align


def refiner_masks(inputs: SceneInputs, kind: str) -> np.ndarray:
    """Instance ids (V, H, W) the refiner sees: sam3, gt or shuffled (see module)."""
    if kind == "gt":
        return np.where(inputs.gt.instance_ids >= 0, inputs.gt.instance_ids, -1)
    if kind == "shuffled":
        return np.roll(inputs.labels.instance_ids, 1, axis=0)
    assert kind == "sam3", kind
    return inputs.labels.instance_ids


def score(
    inputs: SceneInputs, points: np.ndarray, align: tuple, evaluated: list[str]
) -> dict:
    """Surface and label scores of per-pixel points (V, H, W, 3) with SAM 3 labels."""
    labels = inputs.labels
    cloud = predicted_cloud(
        points, labels.instance_ids, labels.classes, labels.scores, align
    )
    return {
        **surface_metrics(cloud.points, inputs.truth.points),
        **label_metrics(inputs.truth, cloud, evaluated),
    }


def run_variant(
    inputs: SceneInputs,
    variant: dict,
    base: tuple[RefineConfig, MultiviewConfig],
    evaluated: list[str],
) -> tuple[dict, np.ndarray]:
    """Refine (or not) as the variant says and score; return the record and depth."""
    scene, align = variant_scene(inputs, variant)
    if variant.get("geometry") == "gt":
        gt = Geometry(
            inputs.gt.depth,
            np.ones_like(inputs.gt.depth),
            inputs.views.intrinsics,
            inputs.views.cam_to_world,
        )
        identity = (1.0, np.eye(3), np.zeros(3))
        return {
            "seconds": 0.0,
            "abs_rel": 0.0,
            **score(inputs, gt.world_points(), identity, evaluated),
        }, inputs.gt.depth
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
    gt_depth = inputs.gt.depth
    seen = gt_depth > 0
    scaled = align[0] * depth.cpu().numpy()[seen]
    record["abs_rel"] = float(np.mean(np.abs(scaled - gt_depth[seen]) / gt_depth[seen]))
    points = points_from_rays(depth.double(), scene.rays.double()).cpu().numpy()
    record |= score(inputs, points, align, evaluated)
    return record, depth.cpu().numpy()


def main() -> None:
    """Score every variant on every scene of a config; see the module docstring."""
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
    scenes, arrays = {}, {}
    for name in raw["scenes"]:
        inputs = load_inputs(raw, name, device)
        if inputs is None:
            continue
        refine_scene(  # warm-up, so the first variant's time is not the CUDA start
            inputs.scene, base[0], dataclasses.replace(base[1], gn_iters=1, cg_iters=1)
        )
        s, rotation, t = inputs.align
        centers = s * inputs.scene.center.cpu().numpy() @ rotation.T + t
        residual = np.linalg.norm(centers - inputs.views.cam_to_world[:, :3, 3], axis=1)
        scenes[name] = {
            "align_scale": s,
            "center_residual_cm": float(100 * np.median(residual)),
            "truth_points": len(inputs.truth.points),
            "objects": len(inputs.labels.classes),
            "variants": {},
        }
        for i, variant in enumerate(raw["variants"]):
            record, depth = run_variant(inputs, variant, base, evaluated)
            scenes[name]["variants"][variant["name"]] = record
            arrays[f"{name}/{i}"] = depth.astype(np.float32)
            print(  # noqa: T201
                f"{name} | {variant['name']} | "
                + " | ".join(f"{k} {record[k]:.3f}" for k in HEADLINE)
                + f" | {record['seconds']:.1f} s",
                flush=True,
            )
        arrays[f"{name}/labels"] = inputs.labels.instance_ids

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
