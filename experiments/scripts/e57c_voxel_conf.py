"""E57c: confidence ranked within each voxel instead of over the whole cloud.

A global confidence cut removes whole surfaces (dev completeness 48.5 cm at q 0.5): the low
confidences sit together, in a view and in space, and every view sees the same surface. Ranking
within a voxel of the 20-view cloud keeps the most confident round(q n) of the n pixels in
each voxel, so a surface is thinned where several views cover it, and kept where one does
(n = 1 is dropped for q <= 0.5, the rule of `keep_top_within`). No hole, no band, and where one
view is much more confident than the others the others go first, as the oracle does with
redundant views. Voxel side = `pixel_footprint` x `--sides` (the side of E45's density filter
at x1), also within each SAM 3 instance (`instance_voxel_density`'s grouping).

uv run python -m experiments.scripts.e57c_voxel_conf --splits dev
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from experiments.scripts.e57c_oracle_curve import evaluate
from src.analysis.filters import keep_top_within, pixel_footprint, voxel_index
from src.analysis.scene_eval import load_inputs
from src.loaders.hypersim.config import load_config
from src.refinement.multiview import points_from_rays

GRID = (0.9, 0.7, 0.5, 0.3)


def scene_rows(cfg: dict, scene: str, sides: list[float]) -> dict:
    """Return (accuracy, completeness, Chamfer) per (side, grouped by instance, q)."""
    inputs = load_inputs(cfg, scene, torch.device("cpu"))
    depth, scale = inputs.scene.depth.numpy(), None
    points = points_from_rays(
        inputs.scene.depth.double(), inputs.scene.rays.double()
    ).numpy()
    scale = pixel_footprint(depth, inputs.scene.k.numpy())
    conf = inputs.geometry.confidence
    ids = inputs.labels.instance_ids
    rows = {}
    for side in sides:
        for by_instance in (False, True):
            group = voxel_index(
                points.reshape(-1, 3),
                side * scale,
                ids.ravel() if by_instance else None,
            )
            for q in GRID:
                kept = keep_top_within(conf, group.reshape(conf.shape), q)
                rows[(side, by_instance, q)] = np.array(
                    [kept.mean(), *evaluate(inputs.truth, points, kept)]
                )
    return rows


def main() -> None:
    """Print the split means."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", nargs="*", default=["dev"])
    parser.add_argument("--sides", nargs="*", type=float, default=[1.0, 2.0, 4.0])
    args = parser.parse_args()
    cfg = yaml.safe_load(Path("experiments/configs/e57c_oracle.yaml").read_text())
    split = load_config(Path(cfg["dataset"]))
    for name in args.splits:
        scenes = [f"{s}_{c}" for s, c in split.scenes([name]).items()]
        runs = []
        for scene in scenes:
            t0 = time.perf_counter()
            runs.append(scene_rows(cfg, scene, args.sides))
            print(f"{scene}: {time.perf_counter() - t0:.0f} s", flush=True)
        print(
            f"{name}: mean over {len(runs)} scenes; kept share, accuracy / completeness / Chamfer (cm)"
        )
        for key in runs[0]:
            share, acc, comp, ch = np.mean([r[key] for r in runs], axis=0)
            side, by_instance, q = key
            label = "per instance" if by_instance else "all pixels"
            print(
                f"  side x{side:g} {label:>12} q {q:.1f}: {share:.3f}  {acc:.3f} / {comp:.3f} / {ch:.3f}"
            )


if __name__ == "__main__":
    main()
