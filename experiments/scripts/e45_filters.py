"""E45: filters of Déjà View's points against the joint refiner at equal completeness.

Reads a `scene_eval` run with the variants Déjà View and Refined (per-scene depth kept)
and the scenes' cached inputs. Per scene and filter (`src/analysis/filters.py`), every
pixel gets a score; the top fraction q is kept, the rest set to NaN, and the output is
scored by the protocol as is (`labeled.align_output`, `surface_metrics`: Umeyama on the
pixels kept, accuracy over them, completeness over every GT point). The curve takes q
on a grid; then q is solved so that the filter's completeness equals the refiner's on
that scene (bisection of the bracketing grid step, each step scored exactly, to a gap
below `GAP_CM`; `--reuse` keeps the curves of an earlier sweep and solves again).
Scores are computed on the GPU where they project between views, the protocol's k-d
trees run on the CPU in `WORKERS` processes. Each finished scene is kept in
`e45_<split>.partial.json`, so a rerun after a crash goes on from there.

uv run python experiments/scripts/e45_filters.py runs/<run> --splits dev stress \
    --step 0.05 > experiments/results/e45_dev_stress.txt
writes runs/<run>/e45_<split>.json, read by `e45_figures.py`.
"""

import argparse
import itertools
import json
import multiprocessing
from collections.abc import Callable
from concurrent.futures import Future, ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
import yaml

from src.analysis.filters import (
    both,
    consistent_views,
    depth_edge,
    facing,
    keep_top,
    log_gradient,
    neighbor_distance,
    pixel_footprint,
    ranks,
    voxel_counts,
)
from src.analysis.scene_eval import SceneInputs, git_hash, load_inputs
from src.evaluation.labeled import LabeledCloud, align_output, surface_metrics
from src.refinement.multiview import consistency, points_from_rays

WORKERS = 6  # scoring processes, as scene_eval's
REFINED_FILTER = "refined + confidence"  # the refiner's own curve, not a baseline
FLOOR = {"accuracy_cm": 0.033, "completeness_cm": 0.037, "chamfer_cm": 0.036}  # E40
METRICS = ("accuracy_cm", "completeness_cm", "chamfer_cm")
# cm: a match within half the last printed digit of the scores is equal completeness
GAP_CM = 0.0005


def filter_scores(inputs: SceneInputs) -> dict[str, np.ndarray]:
    """Every filter's score (V, H, W) of Déjà View's pixels, higher meaning keep."""
    scene = inputs.scene
    scale = scene.depth.median()
    d0 = scale / scene.depth
    ids = torch.from_numpy(inputs.labels.instance_ids.astype(np.int64)).to(d0.device)
    depth = scene.depth.cpu().numpy()
    points = points_from_rays(scene.depth.double(), scene.rays.double()).cpu().numpy()
    flat = points.reshape(-1, 3)
    conf = scene.conf.cpu().numpy().astype(np.float64)
    gradient = -log_gradient(depth)
    side = pixel_footprint(depth, scene.k.cpu().numpy())
    rng = np.random.default_rng(0)
    return {
        "confidence": conf,
        "depth gradient": gradient,
        "confidence + gradient": both(conf, gradient),
        "u": consistency(scene, d0, scale, ids).cpu().numpy().astype(np.float64),
        "depth edge (MoGe)": -depth_edge(depth),
        "statistical outlier (PCL)": -neighbor_distance(flat).reshape(depth.shape),
        "multi-view consistency (MVS)": consistent_views(scene, d0, scale)
        .cpu()
        .numpy()
        .astype(np.float64),
        "voxel density": voxel_counts(flat, side).reshape(depth.shape).astype(float),
        "grazing angle": facing(
            inputs.normals.cpu().numpy(), scene.rays[..., 3:].cpu().numpy()
        ),
        "random": rng.random(depth.shape),
    }


def surface(truth: LabeledCloud, points: np.ndarray, keep: np.ndarray) -> dict:
    """Score the pixels kept by the protocol's surface scores, the rest left out."""
    pred, _, _ = align_output(truth, np.where(keep[..., None], points, np.nan))
    scores = surface_metrics(pred, truth, [])
    return {m: scores[m] for m in METRICS} | {"kept": float(keep.mean())}


def sweep(
    truth: LabeledCloud,
    points: np.ndarray,
    rank: np.ndarray,
    grid: list[float],
    target: float | None,
    curve: list[dict] | None = None,
    fuse: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None,
) -> dict:
    """Score the grid of q (or reuse its curve) and, given a target, the match.

    fuse(points, keep), if given, moves the kept points before scoring (voxel fusion).

    Completeness falls as q grows, though not strictly (Umeyama refits on every cut).
    The solve takes the highest grid step whose ends bracket the target and bisects it
    until an end lies within `GAP_CM` of it or the bracket is one pixel wide, so it
    converges whatever the curve's shape inside; the end nearer the target is the match.
    """

    def at(q: float) -> dict:
        keep = keep_top(rank, q)
        cloud = points if fuse is None else fuse(points, keep)
        return {"q": q} | surface(truth, cloud, keep)

    if curve is None:
        curve = [at(q) for q in grid]
    out: dict = {"curve": curve, "match": None}
    if target is None:
        return out
    by_q = sorted(curve, key=lambda p: p["q"])
    bracket = [
        (lo, hi)
        for lo, hi in itertools.pairwise(by_q)
        if hi["completeness_cm"] <= target <= lo["completeness_cm"]
    ]
    if not bracket:
        return out
    lo, hi = bracket[-1]
    steps = 0

    def gap(p: dict) -> float:
        return abs(p["completeness_cm"] - target)

    while (hi["q"] - lo["q"]) * rank.size > 1 and min(gap(lo), gap(hi)) >= GAP_CM:
        q = 0.5 * (lo["q"] + hi["q"])
        point = at(q)
        steps += 1
        if point["completeness_cm"] > target:
            lo = point
        else:
            hi = point
    best = min((lo, hi), key=gap)
    out["match"] = best | {
        "gap_cm": best["completeness_cm"] - target,
        "bisection_steps": steps,
    }
    return out


def scene_jobs(
    pool: ProcessPoolExecutor,
    inputs: SceneInputs,
    refined: np.ndarray,
    target: float,
    grid: list[float],
    earlier: dict[str, dict] | None,
) -> dict[str, Future]:
    """Submit one sweep per filter of one scene, reusing earlier curves if given."""
    scene = inputs.scene
    points = points_from_rays(scene.depth.double(), scene.rays.double()).cpu().numpy()
    refined_points = points_from_rays(
        torch.from_numpy(refined).double(), scene.rays.cpu().double()
    ).numpy()

    def curve(name: str) -> list[dict] | None:
        return None if earlier is None else earlier[name]["curve"]

    jobs = {
        name: pool.submit(
            sweep, inputs.truth, points, ranks(s), grid, target, curve(name)
        )
        for name, s in filter_scores(inputs).items()
    }
    conf = ranks(scene.conf.cpu().numpy())
    jobs[REFINED_FILTER] = pool.submit(
        sweep, inputs.truth, refined_points, conf, grid, None, curve(REFINED_FILTER)
    )
    return jobs


def collect(
    pending: dict[str, dict[str, Future]], scenes: dict[str, dict], partial: Path
) -> None:
    """Wait for the pending scenes' sweeps, add them, and keep all on disk."""
    for name, jobs in pending.items():
        scenes[name] = {f: j.result() for f, j in jobs.items()}
    partial.write_text(json.dumps(scenes))


def verdict(delta: dict) -> str:
    """Where the refiner lies against a filter's curve, by the plan's rule."""
    acc, cham = delta["accuracy_cm"], delta["chamfer_cm"]
    if acc < -FLOOR["accuracy_cm"] and cham < -FLOOR["chamfer_cm"]:
        return "refiner above"
    if acc > FLOOR["accuracy_cm"] and cham > FLOOR["chamfer_cm"]:
        return "refiner below"
    return "tie"


def report(split: str, scenes: dict[str, dict], refiner: dict[str, dict]) -> None:
    """Print each filter at the refiner's completeness, per scene and the split's mean."""
    print(f"\n== {split}: each filter at the refiner's completeness (cm) ==")
    for name, record in scenes.items():
        r = refiner[name]
        print(
            f"\n{name}: refiner {r['accuracy_cm']:.3f} / {r['completeness_cm']:.3f} / "
            f"{r['chamfer_cm']:.3f}, Déjà View (q = 1) "
            + " / ".join(f"{record['confidence']['curve'][0][m]:.3f}" for m in METRICS)
        )
        print("filter | q | accuracy | Chamfer | completeness gap | Δ acc | Δ Chamfer")
        for f, result in record.items():
            m = result["match"]
            if m is None:
                print(
                    f"{f} | "
                    + ("its own curve" if f == REFINED_FILTER else "not reached")
                )
                continue
            print(
                f"{f} | {m['q']:.3f} | {m['accuracy_cm']:.3f} | {m['chamfer_cm']:.3f} | "
                f"{m['gap_cm']:+.4f} | {r['accuracy_cm'] - m['accuracy_cm']:+.3f} | "
                f"{r['chamfer_cm'] - m['chamfer_cm']:+.3f}"
            )
    names = list(scenes)
    mean_r = {m: np.mean([refiner[n][m] for n in names]) for m in METRICS}
    print(
        f"\n{split} mean of {len(names)} scenes: refiner "
        + " / ".join(f"{mean_r[m]:.3f}" for m in METRICS)
    )
    print(
        "filter | q (mean) | accuracy | Chamfer | Δ acc (refiner - filter) | Δ Chamfer | verdict | scenes refiner better (acc)"
    )
    for f in scenes[names[0]]:
        matches = [scenes[n][f]["match"] for n in names]
        if any(m is None for m in matches):
            print(f"{f} | not reached on every scene")
            continue
        mean_f = {m: np.mean([x[m] for x in matches]) for m in METRICS}
        delta = {m: mean_r[m] - mean_f[m] for m in METRICS}
        better = sum(
            refiner[n]["accuracy_cm"] < x["accuracy_cm"]
            for n, x in zip(names, matches, strict=True)
        )
        print(
            f"{f} | {np.mean([x['q'] for x in matches]):.3f} | "
            f"{mean_f['accuracy_cm']:.3f} | {mean_f['chamfer_cm']:.3f} | "
            f"{delta['accuracy_cm']:+.3f} | {delta['chamfer_cm']:+.3f} | "
            f"{verdict(delta)} | {better} of {len(names)}"
        )


def main() -> None:
    """Sweep every filter on every scene of the splits; see the module docstring."""
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--splits", nargs="+", default=["dev"])
    parser.add_argument("--step", type=float, default=0.05, help="grid of q")
    parser.add_argument(
        "--reuse", action="store_true", help="keep the curves of e45_<split>.json"
    )
    args = parser.parse_args()
    run = yaml.safe_load((args.run / "run.yaml").read_text(encoding="utf-8"))
    raw, arrays = run["config"], np.load(args.run / "results.npz")
    variants = [v["name"] for v in raw["variants"]]
    refined_index = variants.index("Refined")
    n = round(1 / args.step)
    grid = [1 - i * args.step for i in range(n)]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    context = multiprocessing.get_context("spawn")
    print(
        f"run {args.run} (git {run['git']}), this script at {git_hash()}, "
        f"grid q = 1 .. {grid[-1]:.2f}"
    )
    with ProcessPoolExecutor(WORKERS, mp_context=context) as pool:
        for split in args.splits:
            records = run["splits"][split]["scenes"]
            refiner = {
                name: rec["variants"]["Refined"]["headline"]
                for name, rec in records.items()
            }
            out = args.run / f"e45_{split}.json"
            earlier = json.loads(out.read_text())["scenes"] if args.reuse else None
            partial = out.with_suffix(".partial.json")
            scenes: dict[str, dict] = (
                json.loads(partial.read_text()) if partial.is_file() else {}
            )
            pending: dict[str, dict[str, Future]] = {}

            for name in records:
                if name in scenes:
                    continue
                inputs = load_inputs(raw, name, device)
                refined = arrays[f"{name}/{refined_index}"]
                target = refiner[name]["completeness_cm"]
                jobs = scene_jobs(
                    pool, inputs, refined, target, grid, earlier and earlier[name]
                )
                del inputs
                torch.cuda.empty_cache()
                collect(pending, scenes, partial)  # one in flight while the next loads
                pending = {name: jobs}
            collect(pending, scenes, partial)
            scenes = {name: scenes[name] for name in records}
            for name, record in scenes.items():
                dv = record["confidence"]["curve"][0]
                ref = records[name]["variants"]["Déjà View"]["headline"]
                gap = max(abs(dv[m] - ref[m]) for m in METRICS)
                print(f"check {name}: q = 1 against the run's Déjà View, {gap:.6f} cm")
            report(split, scenes, refiner)
            out.write_text(json.dumps({"refiner": refiner, "scenes": scenes}))
            partial.unlink()
            print(f"wrote {out}")


if __name__ == "__main__":
    main()
