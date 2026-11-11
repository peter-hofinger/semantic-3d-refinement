"""E26a: score a competitor's per-pixel points on ETH3D without our inputs.

For a machine that has the ETH3D views and ground truth but not our Déjà View, DSINE or
SAM 3 caches (the cluster that runs PE3R): each `<dir>/<scene>.npz` with `points`
(V, H, W, 3) is scored exactly as `e26a_eth3d.py` scores it, through E35's `score` (one
Umeyama Sim(3) on the pixel pairs, `labeled.surface_metrics`). Each scene's views are
fingerprinted (SHA-1 of the image array) so the scores can be matched to the views the
other methods saw.

Prints per scene and the mean / median over the scenes (indoor, outdoor, all); writes
`<dir>/scores.yaml`.

uv run python experiments/scripts/e26a_score_points.py runs/e26a/pe3r
"""

import hashlib
import sys
from pathlib import Path

import numpy as np
import yaml
from e35_eth3d_eval import SURFACE, score

from src.evaluation.labeled import ground_truth_cloud
from src.loaders.eth3d.scene import load_config
from src.loaders.ground_truth import GroundTruth
from src.loaders.views import Views

ROOT = Path("data/processed/eth3d")
VIEWS = "views_n20_uniform_504"
HEADLINE = SURFACE[:3]


def main() -> None:
    """Score every scene's points in the folder; see the module docstring."""
    folder = Path(sys.argv[1])
    kinds = load_config(Path("configs/eth3d.yaml")).scenes
    scenes = {}
    for name, kind in kinds.items():
        path = folder / f"{name}.npz"
        if not path.is_file():
            print(f"{name} | missing")
            continue
        views = Views.load(ROOT / name / f"{VIEWS}.npz")
        gt = GroundTruth.load(ROOT / name / f"gt_{VIEWS}.npz")
        truth = ground_truth_cloud(
            gt.depth, gt.instance_ids, gt.classes, views.intrinsics, views.cam_to_world
        )
        with np.load(path) as f:
            points, seconds = f["points"].astype(np.float64), float(f["seconds"])
        scores = score(truth, points)
        scores.pop("align")
        sha = hashlib.sha1(  # noqa: S324 - a fingerprint, not security
            np.ascontiguousarray(views.images).tobytes()
        ).hexdigest()
        scenes[name] = {"kind": kind, "views_sha1": sha, "seconds": seconds} | scores
        values = " | ".join(f"{k} {scores[k]:.3f}" for k in HEADLINE)
        print(f"{name} | {values} | {seconds:.1f} s | views {sha[:12]}", flush=True)
    for group in ("indoor", "outdoor", "all"):
        rows = [s for s in scenes.values() if group in ("all", s["kind"])]
        if not rows:
            continue
        a = np.array([[r[k] for k in HEADLINE] for r in rows])
        cells = " | ".join(
            f"{m:.2f} / {d:.2f}"
            for m, d in zip(a.mean(0), np.median(a, 0), strict=True)
        )
        print(f"{group} ({len(rows)}) mean / median, acc | comp | Chamfer cm: {cells}")
    (folder / "scores.yaml").write_text(
        yaml.safe_dump(scenes, sort_keys=False), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
