"""E40: every variant of a `scene_eval` run against the unchanged refiner, per scene.

Reads the `run.yaml` of `experiments/configs/e40_sweep.yaml` (dev) or
`e40_final.yaml` (dev or val). Prints accuracy, completeness and Chamfer per scene and
for the split aggregate, each variant's difference from the reference variant
("Refined", the refiner of `configs/refine.yaml` before E40), and whether the variant
passes: every aggregate difference at most the noise floor (cm, `--floor`, one value or
one per metric), or better. With `--labels` the label scores are printed too; they do
not depend on the variant.

uv run python experiments/scripts/e40_table.py runs/<run>/run.yaml --split dev \
    --floor 0.03 > experiments/results/e40_<name>.txt
"""

import argparse
from pathlib import Path

import yaml

SURFACE = ("accuracy_cm", "completeness_cm", "chamfer_cm")
LABELS = ("miou", "pq", "ap")
REFERENCE = "Refined"


def main() -> None:
    """Print the tables and the pass / fail per variant."""
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--split", default="dev")
    parser.add_argument("--floor", type=float, nargs="+", default=[0.0])
    parser.add_argument("--labels", action="store_true")
    args = parser.parse_args()
    values = args.floor * len(SURFACE) if len(args.floor) == 1 else args.floor
    assert len(values) == len(SURFACE), args.floor
    floor = dict(zip(SURFACE, values, strict=True))
    run = yaml.safe_load(args.run.read_text(encoding="utf-8"))
    report = run["splits"][args.split]
    variants = list(report["aggregate"])
    assert REFERENCE in variants, variants
    print(f"run {args.run}, git {run['git']}, split {args.split}")
    print(f"noise floor (cm): {floor}")
    for name, record in report["scenes"].items():
        print(f"\n{name}: variant | " + " | ".join(SURFACE) + " | Δ vs Refined")
        ref = record["variants"][REFERENCE]["headline"]
        for v in variants:
            h = record["variants"][v]["headline"]
            values = " | ".join(f"{h[m]:.3f}" for m in SURFACE)
            delta = " ".join(f"{h[m] - ref[m]:+.3f}" for m in SURFACE)
            print(f"{v} | {values} | {delta}")
    print(f"\n{args.split} aggregate: variant | " + " | ".join(SURFACE), end="")
    print(" | Δ vs Refined | passes")
    ref = report["aggregate"][REFERENCE]["headline"]
    for v in variants:
        h = report["aggregate"][v]["headline"]
        values = " | ".join(f"{h[m]:.3f}" for m in SURFACE)
        delta = {m: h[m] - ref[m] for m in SURFACE}
        passes = all(delta[m] <= floor[m] for m in SURFACE)
        text = " ".join(f"{d:+.3f}" for d in delta.values())
        print(f"{v} | {values} | {text} | {'yes' if passes else 'no'}")
    if args.labels:
        print(f"\n{args.split} labels: variant | " + " | ".join(LABELS))
        for v in variants:
            h = report["aggregate"][v]["headline"]
            print(f"{v} | " + " | ".join(f"{h[m]:.3f}" for m in LABELS))
    print("\nrefinement time per scene (s), mean over scenes")
    for v in variants:
        seconds = [
            r["variants"][v]["supplementary"]["refinement_s"]
            for r in report["scenes"].values()
        ]
        print(f"{v} | {sum(seconds) / len(seconds):.2f}")


if __name__ == "__main__":
    main()
