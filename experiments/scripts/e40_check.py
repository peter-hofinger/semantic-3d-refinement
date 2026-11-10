"""E40: the "Refined" variant of several `scene_eval` runs against a reference run's.

Each run is `configs/eval_hypersim.yaml --splits dev val` at one commit on the same
GPU, so only the refiner's code differs. Prints, per split, every scene's accuracy,
completeness and Chamfer change against the reference, the split aggregate's, the
median over scenes, the scenes worse and the mean without ai_009_009 (Déjà View's
failure hall, which carries the val loss), and the label scores once.

uv run python experiments/scripts/e40_check.py runs/<reference> runs/<run> ... \
    > experiments/results/e40_check.txt
"""

import sys
from pathlib import Path

import numpy as np
import yaml

SURFACE = ("accuracy_cm", "completeness_cm", "chamfer_cm")
LABELS = ("miou", "pq", "ap")
VARIANT = "Refined"
FAILURE = "ai_009_009_cam_00"


def load(path: Path) -> dict:
    """Return a run's run.yaml."""
    return yaml.safe_load((path / "run.yaml").read_text(encoding="utf-8"))


def main() -> None:
    """Print the comparison of every run with the reference."""
    reference, *runs = map(Path, sys.argv[1:])
    ref = load(reference)
    print(f"reference {reference} (git {ref['git'][:7]}); Δ = run - reference, cm")
    for split in ("dev", "val"):
        agg = ref["splits"][split]["aggregate"]
        for v in ("Déjà View", VARIANT):
            h = agg[v]["headline"]
            values = " ".join(f"{m} {h[m]:.3f}" for m in SURFACE + LABELS)
            print(f"{split} reference {v}: {values}")
    for path in runs:
        run = load(path)
        print(f"\n{path} (git {run['git'][:7]})")
        for split in ("dev", "val"):
            scenes = run["splits"][split]["scenes"]
            names = list(scenes)
            deltas = np.array(
                [
                    [
                        scenes[n]["variants"][VARIANT]["headline"][m]
                        - ref["splits"][split]["scenes"][n]["variants"][VARIANT][
                            "headline"
                        ][m]
                        for m in SURFACE
                    ]
                    for n in names
                ]
            )
            for n, d in zip(names, deltas, strict=True):
                print(f"{split} {n}: " + " ".join(f"{x:+.4f}" for x in d))
            a = run["splits"][split]["aggregate"][VARIANT]["headline"]
            r = ref["splits"][split]["aggregate"][VARIANT]["headline"]
            keep = [i for i, n in enumerate(names) if n != FAILURE]
            rows = {
                "aggregate": [a[m] - r[m] for m in SURFACE],
                "median over scenes": np.median(deltas, 0),
                "scenes worse": (deltas > 0).sum(0),
                f"mean without {FAILURE}": deltas[keep].mean(0),
            }
            for k, d in rows.items():
                print(f"{split} {k}: " + " ".join(f"{x:+.4f}" for x in d))
            print(f"{split} labels: " + " ".join(f"{m} {a[m]:.3f}" for m in LABELS))


if __name__ == "__main__":
    main()
