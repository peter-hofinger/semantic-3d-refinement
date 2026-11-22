"""E86b: E86a's four instance-id arms of the density cut, compared at equal completeness.

Reads the E86a `scene_eval` run (`experiments/configs/e86a.yaml`) and the scenes' cached
inputs. Per scene, Déjà View's pixels are ranked by instance voxel density with each
arm's ids (`segment_ids`: none, SAM 3, GT, permuted per view with seed 0, as E86a), the
top fraction q is kept and scored by the protocol (`e45_filters.sweep`: Umeyama on the
kept pixels, the q grid, then bisection to a target completeness). Two targets per scene,
read from the run: SAM 3's completeness at q 0.5 (the filter as E27a uses it) and none's
at q 0.5. At equal completeness the arms differ by accuracy alone, as in E45.

uv run python experiments/scripts/e86b_equal_completeness.py runs/<e86a run> \
    --splits dev val > experiments/results/e86b_equal_completeness.txt
writes runs/<run>/e86b_<split>.json and experiments/figures/e86b_equal_completeness.jpg.
"""

import argparse
import json
import multiprocessing
import sys
from concurrent.futures import Future, ProcessPoolExecutor
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent))
from e45_filters import FLOOR, METRICS, WORKERS, sweep

from src.analysis.filters import instance_voxel_density, ranks
from src.analysis.scene_eval import git_hash, load_inputs, segment_ids
from src.evaluation.labeled import LabeledCloud
from src.refinement.multiview import points_from_rays
from src.segmentation.base import NO_INSTANCE

ARMS = {"none": "none", "SAM 3": "sam3", "GT": "gt", "permuted": "permuted"}
COLORS = {"SAM 3": "C0", "GT": "C2", "permuted": "C3"}
TARGETS = ("SAM 3", "none")  # whose completeness at q 0.5 the others are solved to
FIGURE = Path("experiments/figures/e86b_equal_completeness.jpg")


def two_targets(
    truth: LabeledCloud,
    points: np.ndarray,
    rank: np.ndarray,
    grid: list[float],
    targets: dict[str, float],
) -> dict:
    """One arm's curve on the grid and its match to each target completeness."""
    out: dict = {"curve": None, "match": {}}
    for name, target in targets.items():
        result = sweep(truth, points, rank, grid, target, out["curve"])
        out["curve"] = result["curve"]
        out["match"][name] = result["match"]
    return out


def scene_jobs(
    pool: ProcessPoolExecutor, raw: dict, name: str, targets: dict, grid: list[float]
) -> dict[str, Future]:
    """Submit one sweep per arm of one scene."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inputs = load_inputs(raw, name, device)
    scene = inputs.scene
    depth, k = scene.depth.cpu().numpy(), scene.k.cpu().numpy()
    points = points_from_rays(scene.depth.double(), scene.rays.double()).cpu().numpy()
    sam3, gt = inputs.labels.instance_ids, inputs.gt.instance_ids
    jobs = {}
    for arm, kind in ARMS.items():
        ids = segment_ids(kind, sam3, gt, {"seed": 0})
        ids = np.full_like(sam3, NO_INSTANCE) if ids is None else ids
        rank = ranks(instance_voxel_density(points, depth, k, ids))
        jobs[arm] = pool.submit(two_targets, inputs.truth, points, rank, grid, targets)
    return jobs


def differences(scenes: dict, target: str, arm: str) -> np.ndarray:
    """(scenes, 3) arm minus none at the target's completeness, NaN where unmatched."""
    rows = []
    for record in scenes.values():
        a, b = record[arm]["match"][target], record["none"]["match"][target]
        if a is None or b is None:
            rows.append([np.nan] * 3)
        else:
            rows.append([a[m] - b[m] for m in METRICS])
    return np.array(rows)


def report(split: str, scenes: dict) -> None:
    """Print the matches per scene and each arm minus none at both targets."""
    names = list(scenes)
    print(f"\n{split} ({len(names)} scenes): accuracy cm at the target completeness")
    for target in TARGETS:
        print(f"target: {target}'s completeness at q 0.5")
        for name in names:
            cells = []
            for arm in ARMS:
                m = scenes[name][arm]["match"][target]
                cells.append(
                    "unmatched"
                    if m is None
                    else f"{m['accuracy_cm']:.3f} (q {m['q']:.3f}, gap {m['gap_cm']:+.4f})"
                )
            print(f"  {name:20s} | " + " | ".join(cells))
        print("  minus none: accuracy mean | median | scenes better | matched")
        for arm in list(ARMS)[1:]:
            d = differences(scenes, target, arm)[:, 0]
            ok = np.isfinite(d)
            print(
                f"  {arm:9s} | {np.mean(d[ok]):+.3f} | {np.median(d[ok]):+.3f} | "
                f"{int((d[ok] < 0).sum())} of {int(ok.sum())} | {int(ok.sum())} of {len(d)}"
            )


def figure(results: dict) -> None:
    """Each scene's accuracy of the arm minus none at equal completeness."""
    splits = list(results)
    fig, axes = plt.subplots(
        1, len(splits), figsize=(6.4 * len(splits), 4.6), layout="constrained"
    )
    for ax, split in zip(np.atleast_1d(axes), splits, strict=True):
        for k, target in enumerate(TARGETS):
            for j, arm in enumerate(list(ARMS)[1:]):
                d = differences(results[split], target, arm)[:, 0]
                d = d[np.isfinite(d)]
                x = k * 4 + j + np.linspace(-0.15, 0.15, len(d))
                ax.scatter(x, d, color=COLORS[arm], s=14, label=arm if k == 0 else None)
                ax.hlines(np.mean(d), x[0] - 0.2, x[-1] + 0.2, color=COLORS[arm])
        ax.axhspan(
            -FLOOR["accuracy_cm"],
            FLOOR["accuracy_cm"],
            color="0.85",
            zorder=0,
            label=f"floor ±{FLOOR['accuracy_cm']} cm",
        )
        ax.axhline(0, color="k", lw=0.6)
        ax.set_yscale("symlog", linthresh=0.5)
        ax.set_xticks(
            [1, 5],
            [f"at {t}'s completeness (q 0.5)" for t in TARGETS],
        )
        ax.set_ylabel("accuracy, arm minus none (cm); dots scenes, bar mean")
        ax.set_title(f"{split} ({len(results[split])} scenes): below 0 the ids help")
        ax.legend()
        ax.grid(alpha=0.3, axis="y")
    fig.suptitle(
        "E86b: Déjà View cut by instance voxel density, each arm's cut solved to the "
        "same completeness per scene (symlog: linear within ±0.5 cm)",
        x=0.01,
        ha="left",
        fontsize=11,
    )
    fig.savefig(FIGURE, dpi=80)
    plt.close(fig)


def main() -> None:
    """Sweep every arm on every scene of the splits; see the module docstring."""
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--splits", nargs="+", default=["dev"])
    parser.add_argument("--step", type=float, default=0.05, help="grid of q")
    parser.add_argument("--scenes", type=int, default=None, help="first n, smoke")
    args = parser.parse_args()
    run = yaml.safe_load((args.run / "run.yaml").read_text(encoding="utf-8"))
    raw = run["config"]
    grid = [1 - i * args.step for i in range(round(0.8 / args.step) + 1)]
    assert any(abs(q - 0.5) < 1e-9 for q in grid), "the grid holds q 0.5"
    context = multiprocessing.get_context("spawn")
    print(f"run {args.run} (git {run['git']}), this script at {git_hash()}")
    print(f"grid q = 1 .. {grid[-1]:.2f}, step {args.step}")
    results = {}
    with ProcessPoolExecutor(WORKERS, mp_context=context) as pool:
        for split in args.splits:
            records = dict(list(run["splits"][split]["scenes"].items())[: args.scenes])
            scenes: dict[str, dict] = {}
            pending: dict[str, dict[str, Future]] = {}
            for name, record in records.items():
                v = record["variants"]
                targets = {
                    t: v[f"Déjà View, kept 0.5, {t}"]["headline"]["completeness_cm"]
                    for t in TARGETS
                }
                jobs = scene_jobs(pool, raw, name, targets, grid)
                torch.cuda.empty_cache()
                for n, j in pending.items():  # one scene in flight while the next loads
                    scenes[n] = {a: f.result() for a, f in j.items()}
                pending = {name: jobs}
            for n, j in pending.items():
                scenes[n] = {a: f.result() for a, f in j.items()}
            for name, record in scenes.items():
                v = records[name]["variants"]
                for arm in ARMS:
                    curve = {round(p["q"], 6): p for p in record[arm]["curve"]}
                    for q, ref in (
                        (1.0, "Déjà View"),
                        (0.5, f"Déjà View, kept 0.5, {arm}"),
                    ):
                        head = v[ref]["headline"]
                        gap = max(abs(curve[q][m] - head[m]) for m in METRICS)
                        assert gap < 1e-3, (name, arm, q, gap)
            print(f"check {split}: q 1 and q 0.5 reproduce the run, every arm")
            report(split, scenes)
            (args.run / f"e86b_{split}.json").write_text(json.dumps(scenes))
            results[split] = scenes
    figure(results)


if __name__ == "__main__":
    main()
