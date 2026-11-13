"""E51a: token averaging within segments inside Déjà View's loop (issue #51).

Per scene Déjà View runs once plainly, once with the hook at a = 0 on every step (must
equal the plain run bit for bit: depth, confidence, rays), then once per grid point
(a, steps) with `TokenMix` (`src/backbone/dejaview.py`): after each chosen loop step a
view's patch tokens of a segment move a towards the segment's mean in that view.
Segments (--segments): `gt` Hypersim's GT instances (the oracle); `wrong` the GT
instances of the next dev scene at the same view index (the control: realistic shapes
in the wrong place); `sam3` the protocol's SAM 3 masks, linked on the plain run's
geometry.

A mixed run's depth and confidence are scored on the plain run's cameras and rays
(`fixed_cameras`): each patch token also carries its ray, so mixing bends the rays
(bedroom, last 4 steps, a = 0.1: 1.6 deg median, the ray map 6.9 px off a pinhole
against 1.8 px plain) and the pinhole fit of the protocol fails; the question is the
depth, and the project keeps Déjà View's cameras fixed. How far the rays moved is
recorded per variant (`ray_deg`, the median angle to the plain run's ray; `depth_rel`,
the median relative change of depth). Every output is then scored as the protocol:
pinhole rays (`scene_from_geometry`), one Umeyama Sim(3) on its pixels (`align_output`),
`surface_metrics`; plus the mean per-pixel 3D error to the pixel's own GT point. Times
are the forward pass with post-processing, synchronised by the copy to the CPU.

uv run python experiments/scripts/e51a_token_mix.py --cfg configs/eval_hypersim.yaml --segments gt --out runs/e51a_gt
uv run python experiments/scripts/e51a_token_mix.py --cfg configs/eval_hypersim.yaml --segments wrong --alphas A --steps S --out runs/e51a_wrong
writes `<out>/e51a.yaml` (every record, config, git hash) and `<out>/e51a_maps.npz` (the
bedroom's per-pixel error at views 18, 0, 12, 6 per variant), prints the report.
"""

import argparse
import dataclasses
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from src.analysis.scene_eval import git_hash, input_paths
from src.backbone.dejaview import (
    DejaViewBackbone,
    TokenMix,
    patch_segments,
    segment_groups,
)
from src.evaluation.labeled import (
    LabeledCloud,
    align_output,
    ground_truth_cloud,
    surface_metrics,
)
from src.geometry import Geometry
from src.loaders.ground_truth import GroundTruth
from src.loaders.hypersim.config import load_config as hypersim_config
from src.loaders.views import Views
from src.refinement.multiview import points_from_rays, scene_from_geometry
from src.segmentation.base import Segmentation
from src.segmentation.linked import working_associator

K = 16
STEP_SETS = {
    "all": frozenset(range(K)),
    "first4": frozenset(range(4)),
    "last4": frozenset(range(K - 4, K)),
}
ALPHAS = (0.01, 0.03, 0.1, 0.3, 1.0)
MAP_SCENE, MAP_VIEWS = "ai_006_007_cam_00", (18, 0, 12, 6)
BASE = "Déjà View"


def run(
    backbone: DejaViewBackbone, views: Views, mix: TokenMix | None
) -> tuple[Geometry, float]:
    """One forward pass and its time in seconds."""
    t0 = time.perf_counter()
    geometry = backbone.predict(views, mix)
    return geometry, time.perf_counter() - t0


def score(geometry: Geometry, truth: LabeledCloud) -> tuple[dict, np.ndarray]:
    """Score one output as the protocol (cm); return its per-pixel error map (m)."""
    scene = scene_from_geometry(geometry)
    points = points_from_rays(scene.depth, scene.rays).numpy()
    pred, scale, _ = align_output(truth, points)
    e = np.linalg.norm(pred - truth.points, axis=1)
    assert np.isfinite(e).all()
    surface = surface_metrics(pred, truth, [])
    rec = {
        k: float(surface[k]) for k in ("accuracy_cm", "completeness_cm", "chamfer_cm")
    }
    rec |= {"error_cm": 100 * float(e.mean()), "scale": float(scale)}
    err = np.full(truth.pixels.shape, np.nan, dtype=np.float32)
    err[truth.pixels] = e
    return rec, err


def fixed_cameras(mixed: Geometry, plain: Geometry) -> tuple[Geometry, dict]:
    """Mixed depth and confidence on the plain cameras and rays; how far the rays moved."""
    assert mixed.ray_directions is not None
    assert plain.ray_directions is not None
    a, b = mixed.ray_directions, plain.ray_directions
    cos = (a * b).sum(-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1))
    moved = {
        "ray_deg": float(np.degrees(np.median(np.arccos(np.clip(cos, -1, 1))))),
        "depth_rel": float(np.median(np.abs(mixed.depth / plain.depth - 1))),
    }
    geometry = dataclasses.replace(
        plain, depth=mixed.depth, confidence=mixed.confidence
    )
    return geometry, moved


def same(a: Geometry, b: Geometry) -> bool:
    """Bit-for-bit equal depth, confidence and rays."""
    return all(
        np.array_equal(getattr(a, f), getattr(b, f))
        for f in ("depth", "confidence", "ray_origins", "ray_directions")
    )


def main() -> None:
    """Run the grid on the chosen splits; see the module docstring."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", default=["dev"])
    parser.add_argument("--segments", choices=["gt", "wrong", "sam3"], required=True)
    parser.add_argument("--alphas", type=float, nargs="+", default=list(ALPHAS))
    parser.add_argument(
        "--steps", nargs="+", default=list(STEP_SETS), choices=STEP_SETS
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    raw = yaml.safe_load(args.cfg.read_text())
    assert raw["backbone"] == f"dejaview_k{K}", raw["backbone"]
    config = hypersim_config(Path(raw["dataset"]))
    scenes = [f"{n}_{c}" for n, c in config.scenes(args.splits).items()]
    args.out.mkdir(parents=True, exist_ok=False)
    backbone = DejaViewBackbone(steps=K)
    records: dict = {}
    maps: dict[str, np.ndarray] = {}

    for i, name in enumerate(scenes):
        paths = input_paths(raw, name)
        views, gt = Views.load(paths["views"]), GroundTruth.load(paths["gt"])
        truth = ground_truth_cloud(
            gt.depth, gt.instance_ids, gt.classes, views.intrinsics, views.cam_to_world
        )
        run(backbone, views, None)  # warm-up: weights, kernels
        plain, t_plain = run(backbone, views, None)
        if args.segments == "gt":
            ids = gt.instance_ids
        elif args.segments == "wrong":
            other = scenes[(i + 1) % len(scenes)]
            assert other != name
            ids = GroundTruth.load(input_paths(raw, other)["gt"]).instance_ids
        else:
            masks = Segmentation.load(paths["masks"])
            ids = working_associator().associate(masks, plain).instance_ids
        assert ids.shape == plain.depth.shape, (ids.shape, plain.depth.shape)
        segments = patch_segments(ids)
        groups = torch.from_numpy(segment_groups(segments))
        zero, _ = run(backbone, views, TokenMix(groups, 0.0, STEP_SETS["all"]))
        assert same(zero, plain), f"{name}: a = 0 is not Déjà View bit for bit"

        rec, err = score(plain, truth)
        scene_rec = {
            "segments_from": other if args.segments == "wrong" else args.segments,
            "patch_share": float((segments >= 0).mean()),
            "groups": int(groups.max()) + 1,
            "identity_bit_for_bit": True,
            "variants": {
                BASE: rec | {"ray_deg": 0.0, "depth_rel": 0.0, "seconds": t_plain}
            },
        }
        if name == MAP_SCENE:
            maps[BASE] = err[list(MAP_VIEWS)].astype(np.float16)
        for steps in args.steps:
            for alpha in args.alphas:
                label = f"a={alpha:g} {steps}"
                mix = TokenMix(groups, alpha, STEP_SETS[steps])
                mixed, t = run(backbone, views, mix)
                geometry, moved = fixed_cameras(mixed, plain)
                rec, err = score(geometry, truth)
                scene_rec["variants"][label] = rec | moved | {"seconds": t}
                if name == MAP_SCENE:
                    maps[label] = err[list(MAP_VIEWS)].astype(np.float16)
                print(f"{name} {label}: Chamfer {rec['chamfer_cm']:.3f} cm", flush=True)
        records[name] = scene_rec

    summary = {"args": {k: str(v) for k, v in vars(args).items()}, "git": git_hash()}
    summary |= {"config": raw, "scenes": records}
    (args.out / "e51a.yaml").write_text(
        yaml.safe_dump(summary, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    np.savez_compressed(args.out / "e51a_maps.npz", **maps)
    report(records)


def report(records: dict) -> None:
    """Per variant: mean scores over the scenes, change against Déjà View, wins."""
    names = list(records)
    variants = list(records[names[0]]["variants"])
    keys = (
        "accuracy_cm",
        "completeness_cm",
        "chamfer_cm",
        "error_cm",
        "ray_deg",
        "depth_rel",
        "seconds",
    )
    print(
        f"{'variant':<16}"
        + "".join(f"{k.removesuffix('_cm'):>14}" for k in keys)
        + "   dChamfer  wins"
    )
    for v in variants:
        mean = {k: np.mean([records[n]["variants"][v][k] for n in names]) for k in keys}
        d = [
            records[n]["variants"][v]["chamfer_cm"]
            - records[n]["variants"][BASE]["chamfer_cm"]
            for n in names
        ]
        wins = sum(x < 0 for x in d)
        print(
            f"{v:<16}"
            + "".join(f"{mean[k]:>14.3f}" for k in keys)
            + f"   {np.mean(d):+8.3f}  {wins}/{len(names)}"
        )


if __name__ == "__main__":
    main()
