"""E34 after the loader fix: the joint refiner on all 11 val scenes, PR #19 protocol.

Reads the `run.yaml` that `src.analysis.scene_eval` (main f645d08) wrote for the val
split with ai_031_004's views re-prepared by the fixed loader (keyframe-0 rotation
divided by cbrt(det), see `e34_ai031004_camera.py`), cluster job 1373. Prints both
variants per scene, the 11-scene aggregate, the mean of the other 10 scenes for
comparison with the addendum's run (job 994) and how many scenes each score improves on.

uv run python experiments/scripts/e34_val11.py runs/<run>/run.yaml \
    > experiments/results/e34_val11.txt
"""

import sys
from pathlib import Path

import numpy as np
import yaml

VARIANTS = ("Déjà View", "Refined")
SURFACE = ("accuracy_cm", "completeness_cm", "chamfer_cm")
LABELS = ("miou", "pq", "ap")
FIXED = "ai_031_004_cam_00"


def main() -> None:
    """Print the per-scene table, the aggregates and the better / worse counts."""
    run = yaml.safe_load(Path(sys.argv[1]).read_text(encoding="utf-8"))
    report = run["splits"]["val"]
    scenes = report["scenes"]
    assert len(scenes) == 11, list(scenes)
    assert FIXED in scenes, list(scenes)
    print(f"run {sys.argv[1]}, git {run['git'][:7]}")
    print("\nval: scene | variant | " + " | ".join(SURFACE + LABELS))
    for scene, record in scenes.items():
        for v in VARIANTS:
            h = record["variants"][v]["headline"]
            print(
                f"{scene} | {v} | "
                + " | ".join(f"{h[m]:.3f}" for m in SURFACE + LABELS)
            )
    for v in VARIANTS:
        h = report["aggregate"][v]["headline"]
        print(
            f"val mean (11, aggregate) | {v} | "
            + " | ".join(f"{h[m]:.3f}" for m in SURFACE + LABELS)
        )
    others = [s for s in scenes if s != FIXED]
    for v in VARIANTS:
        means = [
            np.mean([scenes[s]["variants"][v]["headline"][m] for s in others])
            for m in SURFACE
        ]
        print(
            f"mean of the other 10 scenes | {v} | "
            + " | ".join(f"{x:.3f}" for x in means)
        )
    for m in SURFACE:
        delta = np.array(
            [
                r["variants"]["Refined"]["headline"][m]
                - r["variants"]["Déjà View"]["headline"][m]
                for r in scenes.values()
            ]
        )
        print(
            f"val {m}: better on {(delta < 0).sum()} / {len(delta)}, worse on {(delta > 0).sum()}"
        )
    scale = {
        s: round(r["variants"]["Déjà View"]["debug"]["scale"], 3)
        for s, r in scenes.items()
    }
    print(f"\nUmeyama scale, Déjà View: {scale}")


if __name__ == "__main__":
    main()
