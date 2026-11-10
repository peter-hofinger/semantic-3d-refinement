"""E45, second arm: filter Déjà View's points first, then refine what remains.

Reads the `scene_eval` run E45 read (its config, its refiner settings) and E45's curves
of each filter without refinement (`e45_<split>.json`, same run, same code). Per scene,
filter and q of the grid: the filter's top fraction q is kept (`filters.keep_top`, E45's
scores), the refiner runs on those pixels only (`refine_scene(keep=...)`, the rest NaN),
and the output is scored by the protocol as is (`e45_filters.surface`: Umeyama on the
kept pixels). Then the same filter without refinement is solved to the completeness
filter + refine reached (`e45_filters.sweep`'s bisection, from E45's curve), and the
two are compared there by E45's rule against E40's floor.

uv run python experiments/scripts/e45_filter_refine.py runs/<run> --splits dev \
    --grid 1 0.9 0.8 0.7 0.6 0.5 0.4 0.3 > experiments/results/e45_filter_refine_dev.txt
writes runs/<run>/e45_filter_refine_<split>.json, read by `e45_filter_refine_figures.py`.
"""

import argparse
import dataclasses
import json
import multiprocessing
import sys
import time
from collections import deque
from concurrent.futures import Future, ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent))
from e45_filters import FLOOR, METRICS, WORKERS, surface, sweep

from src.analysis.filters import (
    keep_top,
    neighbor_distance,
    pixel_footprint,
    ranks,
    voxel_counts,
)
from src.analysis.scene_eval import SceneInputs, git_hash, load_inputs
from src.evaluation.labeled import LabeledCloud
from src.refinement.multiview import RefineConfig, points_from_rays, refine_scene

FILTERS = ("voxel density", "confidence", "statistical outlier (PCL)")  # E45's best
IN_FLIGHT = 2 * WORKERS  # jobs queued at once; each carries ~0.4 GB of clouds


def filter_scores(inputs: SceneInputs) -> dict[str, np.ndarray]:
    """Score Déjà View's pixels (V, H, W) by the three filters, as E45 does."""
    scene = inputs.scene
    depth = scene.depth.cpu().numpy()
    points = points_from_rays(scene.depth.double(), scene.rays.double()).cpu().numpy()
    flat = points.reshape(-1, 3)
    side = pixel_footprint(depth, scene.k.cpu().numpy())
    return {
        "voxel density": voxel_counts(flat, side).reshape(depth.shape).astype(float),
        "confidence": scene.conf.cpu().numpy().astype(np.float64),
        "statistical outlier (PCL)": -neighbor_distance(flat).reshape(depth.shape),
    }


def compare(
    truth: LabeledCloud,
    refined: np.ndarray,
    keep: np.ndarray,
    points: np.ndarray,
    rank: np.ndarray,
    curve: list[dict],
) -> dict:
    """Score filter + refine, then the filter alone at the completeness it reached."""
    ours = surface(truth, refined, keep)
    alone = sweep(truth, points, rank, [], ours["completeness_cm"], curve)["match"]
    return {"refined": ours, "alone": alone}


def verdict(delta: dict) -> str:
    """Where filter + refine lies against the filter's curve, by the plan's rule."""
    acc, cham = delta["accuracy_cm"], delta["chamfer_cm"]
    if acc < -FLOOR["accuracy_cm"] and cham < -FLOOR["chamfer_cm"]:
        return "above"
    if acc > FLOOR["accuracy_cm"] and cham > FLOOR["chamfer_cm"]:
        return "below"
    return "tie"


def report(split: str, scenes: dict[str, dict], grid: list[float]) -> None:
    """Print each point of filter + refine against the filter alone, then the means."""
    names = list(scenes)
    for f in FILTERS:
        print(f"\n== {split}, {f}: filter + refine against the filter alone (cm) ==")
        print(
            "q | scene | refined acc / comp / Chamfer | alone q, acc, Chamfer | "
            "gap | Δ acc | Δ Chamfer (refined - alone)"
        )
        for name in names:
            for point in scenes[name][f]:
                r, a = point["refined"], point["alone"]
                head = f"{point['q']:.2f} | {name} | " + " / ".join(
                    f"{r[m]:.3f}" for m in METRICS
                )
                if a is None:
                    print(head + " | not reached")
                    continue
                print(
                    f"{head} | {a['q']:.3f}, {a['accuracy_cm']:.3f}, "
                    f"{a['chamfer_cm']:.3f} | {a['gap_cm']:+.4f} | "
                    f"{r['accuracy_cm'] - a['accuracy_cm']:+.3f} | "
                    f"{r['chamfer_cm'] - a['chamfer_cm']:+.3f}"
                )
        print(
            f"\n{split} mean of {len(names)} scenes, {f}: q | refined acc / comp / "
            "Chamfer | alone acc / Chamfer | Δ acc | Δ Chamfer | verdict | "
            "scenes better (acc)"
        )
        for i, q in enumerate(grid):
            points = [scenes[n][f][i] for n in names]
            if any(p["alone"] is None for p in points):
                print(f"{q:.2f} | not reached on every scene")
                continue
            r = {m: np.mean([p["refined"][m] for p in points]) for m in METRICS}
            a = {m: np.mean([p["alone"][m] for p in points]) for m in METRICS}
            delta = {m: r[m] - a[m] for m in METRICS}
            better = sum(
                p["refined"]["accuracy_cm"] < p["alone"]["accuracy_cm"] for p in points
            )
            print(
                f"{q:.2f} | "
                + " / ".join(f"{r[m]:.3f}" for m in METRICS)
                + f" | {a['accuracy_cm']:.3f} / {a['chamfer_cm']:.3f} | "
                f"{delta['accuracy_cm']:+.3f} | {delta['chamfer_cm']:+.3f} | "
                f"{verdict(delta)} | {better} of {len(names)}"
            )


def main() -> None:
    """Sweep filter + refine on every scene of the splits; see the module docstring."""
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--splits", nargs="+", default=["dev"])
    parser.add_argument("--grid", nargs="+", type=float, required=True)
    args = parser.parse_args()
    run = yaml.safe_load((args.run / "run.yaml").read_text(encoding="utf-8"))
    raw = run["config"]
    cuda = torch.cuda.is_available()
    cfg = dataclasses.replace(
        RefineConfig(**run["refine"]["refine"]), device="cuda" if cuda else "cpu"
    )
    device = torch.device(cfg.device)
    grid: list[float] = args.grid
    context = multiprocessing.get_context("spawn")
    print(
        f"run {args.run} (git {run['git']}), this script at {git_hash()}, "
        f"grid q = {grid}, {cfg}"
    )
    with ProcessPoolExecutor(WORKERS, mp_context=context) as pool:
        for split in args.splits:
            records = run["splits"][split]["scenes"]
            e45 = json.loads((args.run / f"e45_{split}.json").read_text())["scenes"]
            jobs: dict[tuple[str, str], list[Future]] = {}
            inflight: deque[Future] = deque()
            seconds = []
            for name in records:
                inputs = load_inputs(raw, name, device)
                scene = inputs.scene
                segments = torch.from_numpy(inputs.labels.instance_ids.astype(np.int64))
                points = (
                    points_from_rays(scene.depth.double(), scene.rays.double())
                    .cpu()
                    .numpy()
                )
                for f, score in filter_scores(inputs).items():
                    rank = ranks(score)
                    jobs[name, f] = []
                    for q in grid:
                        keep = keep_top(rank, q)
                        t0 = time.perf_counter()
                        depth, _, _ = refine_scene(
                            scene, cfg, inputs.normals, segments, torch.from_numpy(keep)
                        )
                        refined = (
                            points_from_rays(depth.double(), scene.rays.double())
                            .cpu()
                            .numpy()
                        )
                        seconds.append(time.perf_counter() - t0)
                        while len(inflight) >= IN_FLIGHT:
                            inflight.popleft().result()
                        job = pool.submit(
                            compare,
                            inputs.truth,
                            refined,
                            keep,
                            points,
                            rank,
                            e45[name][f]["curve"],
                        )
                        jobs[name, f].append(job)
                        inflight.append(job)
                del inputs
                torch.cuda.empty_cache()
                print(f"refined {name}, {np.mean(seconds):.1f} s each", flush=True)
            scenes: dict[str, dict] = {n: {} for n in records}
            for (name, f), futures in jobs.items():
                scenes[name][f] = [
                    {"q": q} | j.result() for q, j in zip(grid, futures, strict=True)
                ]
            if 1.0 in grid:
                for name in records:
                    ours = scenes[name][FILTERS[0]][grid.index(1.0)]["refined"]
                    ref = records[name]["variants"]["Refined"]["headline"]
                    gap = max(abs(ours[m] - ref[m]) for m in METRICS)
                    print(
                        f"check {name}: q = 1 against the run's refiner, {gap:.6f} cm"
                    )
            report(split, scenes, grid)
            out = args.run / f"e45_filter_refine_{split}.json"
            out.write_text(json.dumps({"grid": grid, "scenes": scenes}))
            print(f"wrote {out}")


if __name__ == "__main__":
    main()
