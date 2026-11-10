"""E45, third arm: voxel fusion, a voxel's score keeps it and its points are averaged.

Reads the `scene_eval` run E45 read and E45's matches of voxel density alone
(`e45_<split>.json`). Per scene and variant, each pixel is scored by its voxel
(`filters.voxel_sums`: points, summed confidence, or the points of its own instance),
the top fraction q is kept, and for the fused variants every kept pixel's point is
replaced by the mean of the kept points in its voxel (`filters.voxel_means`), the rest
NaN. The output is scored by the protocol as is (`e45_filters.sweep` with `fuse`:
Umeyama on the kept pixels) on the grid of q, then solved to the refiner's completeness
and compared there with the refiner and with voxel density alone, by E45's rule.

uv run python experiments/scripts/e45_voxel_fusion.py runs/<run> --splits dev \
    > experiments/results/e45_voxel_fusion_dev.txt
writes runs/<run>/e45_voxel_fusion_<split>.json, read by `e45_voxel_fusion_figures.py`.
"""

import argparse
import json
import multiprocessing
import sys
from concurrent.futures import Future, ProcessPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent))
from e45_filters import FLOOR, METRICS, WORKERS, sweep

from src.analysis.filters import pixel_footprint, ranks, voxel_means, voxel_sums
from src.analysis.scene_eval import SceneInputs, git_hash, load_inputs
from src.refinement.multiview import points_from_rays

ALONE = "voxel density"  # E45's filter, the curve to beat
COARSER = (2, 4)  # voxel sides in footprints, read for the slope only


def fused(
    points: np.ndarray, keep: np.ndarray, side: float, groups: np.ndarray | None
) -> np.ndarray:
    """Return the kept points (V, H, W, 3) replaced by their voxel's mean, the rest NaN."""
    out = np.full_like(points, np.nan)
    g = None if groups is None else groups[keep]
    out[keep] = voxel_means(points[keep], side, g)
    return out


def variants(inputs: SceneInputs) -> tuple[np.ndarray, dict[str, tuple]]:
    """Déjà View's points and each variant's (score per pixel, fuse or None)."""
    scene = inputs.scene
    depth = scene.depth.cpu().numpy()
    points = points_from_rays(scene.depth.double(), scene.rays.double()).cpu().numpy()
    flat = points.reshape(-1, 3)
    ones = np.ones(len(flat))
    conf = scene.conf.cpu().numpy().astype(np.float64).ravel()
    ids = inputs.labels.instance_ids.astype(np.int64)
    side = pixel_footprint(depth, scene.k.cpu().numpy())

    def shaped(score: np.ndarray) -> np.ndarray:
        return score.reshape(depth.shape)

    count = shaped(voxel_sums(flat, side, ones))
    summed = shaped(voxel_sums(flat, side, conf))
    own = shaped(voxel_sums(flat, side, ones, ids.ravel()))
    out = {
        "count + mean": (count, partial(fused, side=side, groups=None)),
        "Σ confidence": (summed, None),
        "Σ confidence + mean": (summed, partial(fused, side=side, groups=None)),
        "instance count": (own, None),
        "instance count + instance mean": (own, partial(fused, side=side, groups=ids)),
    }
    for m in COARSER:
        coarse = shaped(voxel_sums(flat, m * side, ones))
        out[f"count + mean, {m} x footprint"] = (
            coarse,
            partial(fused, side=m * side, groups=None),
        )
    return points, out


def verdict(delta: dict) -> str:
    """Where a variant lies against a reference at equal completeness, E45's rule."""
    acc, cham = delta["accuracy_cm"], delta["chamfer_cm"]
    if acc < -FLOOR["accuracy_cm"] and cham < -FLOOR["chamfer_cm"]:
        return "above"
    if acc > FLOOR["accuracy_cm"] and cham > FLOOR["chamfer_cm"]:
        return "below"
    return "tie"


def report(scenes: dict, refiner: dict, alone: dict) -> None:
    """Print each variant at the refiner's completeness, per scene and as the mean."""
    names = list(scenes)
    print(
        "\nvariant | scene | q | accuracy | Chamfer | gap | Δ acc vs alone | vs refiner"
    )
    for v in scenes[names[0]]:
        for n in names:
            m = scenes[n][v]["match"]
            a = alone[n]
            if m is None:
                print(f"{v} | {n} | not reached")
                continue
            print(
                f"{v} | {n} | {m['q']:.3f} | {m['accuracy_cm']:.3f} | "
                f"{m['chamfer_cm']:.3f} | {m['gap_cm']:+.4f} | "
                f"{m['accuracy_cm'] - a['accuracy_cm']:+.3f} | "
                f"{m['accuracy_cm'] - refiner[n]['accuracy_cm']:+.3f}"
            )
    mean_alone = {k: np.mean([alone[n][k] for n in names]) for k in METRICS}
    mean_ref = {k: np.mean([refiner[n][k] for n in names]) for k in METRICS}
    print(
        f"\nmean of {len(names)} scenes at the refiner's completeness: refiner "
        f"{mean_ref['accuracy_cm']:.3f} / {mean_ref['chamfer_cm']:.3f}, voxel density "
        f"alone {mean_alone['accuracy_cm']:.3f} / {mean_alone['chamfer_cm']:.3f}"
    )
    print(
        "variant | q (mean) | accuracy | Chamfer | Δ vs alone (acc / Ch) | verdict "
        "| Δ vs refiner (acc / Ch) | verdict | scenes better than alone (acc)"
    )
    for v in scenes[names[0]]:
        matches = [scenes[n][v]["match"] for n in names]
        if any(m is None for m in matches):
            print(f"{v} | not reached on every scene")
            continue
        mean = {k: np.mean([m[k] for m in matches]) for k in METRICS}
        d_alone = {k: mean[k] - mean_alone[k] for k in METRICS}
        d_ref = {k: mean[k] - mean_ref[k] for k in METRICS}
        better = sum(
            m["accuracy_cm"] < alone[n]["accuracy_cm"]
            for n, m in zip(names, matches, strict=True)
        )
        print(
            f"{v} | {np.mean([m['q'] for m in matches]):.3f} | "
            f"{mean['accuracy_cm']:.3f} | {mean['chamfer_cm']:.3f} | "
            f"{d_alone['accuracy_cm']:+.3f} / {d_alone['chamfer_cm']:+.3f} | "
            f"{verdict(d_alone)} | {d_ref['accuracy_cm']:+.3f} / "
            f"{d_ref['chamfer_cm']:+.3f} | {verdict(d_ref)} | {better} of {len(names)}"
        )


def main() -> None:
    """Sweep every variant on every scene of the splits; see the module docstring."""
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--splits", nargs="+", default=["dev"])
    parser.add_argument("--step", type=float, default=0.05, help="grid of q")
    args = parser.parse_args()
    run = yaml.safe_load((args.run / "run.yaml").read_text(encoding="utf-8"))
    raw = run["config"]
    grid = [1 - i * args.step for i in range(round(1 / args.step))]
    context = multiprocessing.get_context("spawn")
    print(
        f"run {args.run} (git {run['git']}), this script at {git_hash()}, "
        f"grid q = 1 .. {grid[-1]:.2f}"
    )
    with ProcessPoolExecutor(WORKERS, mp_context=context) as pool:
        for split in args.splits:
            e45 = json.loads((args.run / f"e45_{split}.json").read_text())
            refiner = e45["refiner"]
            alone = {n: rec[ALONE]["match"] for n, rec in e45["scenes"].items()}
            scenes: dict[str, dict] = {}
            pending: dict[str, dict[str, Future]] = {}
            for name in e45["scenes"]:
                inputs = load_inputs(raw, name, torch.device("cpu"))
                points, todo = variants(inputs)
                target = refiner[name]["completeness_cm"]
                jobs = {
                    v: pool.submit(
                        sweep, inputs.truth, points, ranks(s), grid, target, None, f
                    )
                    for v, (s, f) in todo.items()
                }
                del inputs
                for k, done in pending.items():  # one scene in flight
                    scenes[k] = {v: j.result() for v, j in done.items()}
                pending = {name: jobs}
                print(f"submitted {name}", flush=True)
            for k, done in pending.items():
                scenes[k] = {v: j.result() for v, j in done.items()}
            for name, record in scenes.items():
                own = record["Σ confidence"]["curve"][0]
                ref = run["splits"][split]["scenes"][name]["variants"]["Déjà View"]
                gap = max(abs(own[m] - ref["headline"][m]) for m in METRICS)
                print(f"check {name}: unfused q = 1 against Déjà View, {gap:.6f} cm")
            report(scenes, refiner, alone)
            out = args.run / f"e45_voxel_fusion_{split}.json"
            out.write_text(json.dumps({"refiner": refiner, "scenes": scenes}))
            print(f"wrote {out}")


if __name__ == "__main__":
    main()
