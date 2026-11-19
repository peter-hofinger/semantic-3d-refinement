"""E57c: the greedy oracle on Déjà View's depth lifted through the ground-truth cameras.

Two clouds per scene from the same depth maps d (Déjà View's z-depth):
- `usual`: Déjà View's ray map replaced by its pinhole fit (`scene_from_geometry`), as everywhere else;
- `GT rays`: lift(s d, K_GT, c2w_GT) per view (`labeled.lift`, pixel centers +0.5), s the scale of the Umeyama alignment of the usual cloud (the depth is in Déjà View's
  own scale; this is the only fit of the depth).
Each is scored uncut and cut by the greedy oracle (`e57c_oracle_curve.greedy`, 10% rounds) at q = 0.9 ... 0.1: accuracy, completeness, Chamfer (cm), one Umeyama per cut
as `evaluate` does. No SAM 3, no density. The GT cameras are privileged information: a diagnostic of the camera error, not a method.

uv run python -m experiments.scripts.e57c_gt_rays --splits dev > experiments/results/e57c_gt_rays_dev.txt
"""

import argparse
import time
from pathlib import Path

import numpy as np
import yaml

from experiments.scripts.e57c_oracle_curve import GRID, evaluate, greedy
from src.evaluation.labeled import (
    align_output,
    ground_truth_cloud,
    lift,
    nearest_distances,
)
from src.geometry import Geometry
from src.loaders.ground_truth import GroundTruth
from src.loaders.hypersim.config import load_config
from src.loaders.views import Views
from src.refinement.multiview import points_from_rays, scene_from_geometry

LEVELS = [q for q in GRID if q < 1]


def scene_rows(cfg: dict, scene: str) -> dict[str, np.ndarray]:
    """Return (kept share, accuracy, completeness, Chamfer) per variant and cut of one scene."""
    folder = Path(cfg["root"]) / scene
    views = Views.load(folder / f"{cfg['views']}.npz")
    gt = GroundTruth.load(folder / f"gt_{cfg['views']}.npz")
    truth = ground_truth_cloud(
        gt.depth, gt.instance_ids, gt.classes, views.intrinsics, views.cam_to_world
    )
    pixels = truth.pixels
    assert pixels is not None
    geometry = Geometry.load(folder / f"dejaview_k16_{cfg['views']}.npz")
    scene_ = scene_from_geometry(geometry)
    usual = points_from_rays(scene_.depth.double(), scene_.rays.double()).numpy()
    _, scale, _ = align_output(truth, usual)
    depth = scale * geometry.depth.astype(np.float64)
    gt_rays = np.stack(
        [
            lift(d, k.astype(np.float64), c.astype(np.float64))
            for d, k, c in zip(depth, views.intrinsics, views.cam_to_world, strict=True)
        ]
    )
    rows: dict[str, np.ndarray] = {}
    for name, points in (("usual", usual), ("GT rays", gt_rays)):
        pred, _, _ = align_output(truth, points)
        a = 100 * nearest_distances(truth.points, pred)
        path = greedy(truth, pred, a, LEVELS)

        def score(
            label: str, kept: np.ndarray, points: np.ndarray = points, name: str = name
        ) -> None:
            full = np.ones(pixels.shape, dtype=bool)  # a pixel without GT is not scored
            full[pixels] = kept
            rows[f"{name}, {label}"] = np.array(
                [full.mean(), *evaluate(truth, points, full)]
            )

        score("uncut", np.ones(len(a), dtype=bool))
        for q in LEVELS:
            score(f"oracle q {q}", path[q])
    return rows


def main() -> None:
    """Print each scene's and the split's mean rows."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", nargs="*", default=["dev"])
    args = parser.parse_args()
    cfg = yaml.safe_load(Path("experiments/configs/e57c_oracle.yaml").read_text())
    split = load_config(Path(cfg["dataset"]))
    for name in args.splits:
        scenes = [f"{s}_{c}" for s, c in split.scenes([name]).items()]
        runs = []
        for scene in scenes:
            t0 = time.perf_counter()
            runs.append(scene_rows(cfg, scene))
            print(f"{scene}: {time.perf_counter() - t0:.0f} s", flush=True)
        mean = {k: np.mean([r[k] for r in runs], axis=0) for k in runs[0]}
        print(
            f"{name}: mean over {len(runs)} scenes; kept share, accuracy / completeness / Chamfer (cm); per scene Chamfer"
        )
        for k, (share, acc, comp, chamfer) in mean.items():
            per = " / ".join(f"{r[k][3]:.2f}" for r in runs)
            print(
                f"  {k:<22} {share:.3f}  {acc:.3f} / {comp:.3f} / {chamfer:.3f}   {per}"
            )
        for variant in ("usual", "GT rays"):
            best = min(
                (k for k in mean if k.startswith(f"{variant}, oracle")),
                key=lambda k: mean[k][3],
            )
            per = " / ".join(
                f"{min(r[k][3] for k in r if k.startswith(f'{variant}, oracle')):.2f}"
                for r in runs
            )
            print(
                f"  best {variant}: {best}, Chamfer {mean[best][3]:.3f}; per scene best over q: {per}"
            )


if __name__ == "__main__":
    main()
