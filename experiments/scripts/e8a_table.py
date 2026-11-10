"""E8a: the segment ablation's surface scores, each variant against refining without segments.

Reads the `run.yaml` of `experiments/configs/e8a_segments.yaml`. Per split: every
scene's accuracy, completeness and Chamfer per variant; then the aggregate, each
variant's difference from "Off" (the refiner without segments), the median scene's
difference and the number of scenes on which the variant is worse than Off. Lower is
better for all three; label scores are the same for every variant and not printed.

uv run python experiments/scripts/e8a_table.py runs/<run>/run.yaml \
    > experiments/results/e8a_table.txt
"""

import sys
from pathlib import Path

import numpy as np
import yaml

SURFACE = ("accuracy_cm", "completeness_cm", "chamfer_cm")
REFERENCE = "Off"


def fmt(values: list[float], sign: str = "") -> str:
    """Join three numbers as accuracy / completeness / Chamfer."""
    return " / ".join(f"{v:{sign}.3f}" for v in values)


def kept_shares(path: Path) -> dict[tuple[str, str], float]:
    """Share of E_mv's matches kept per (scene, variant), from `e8a_inputs.py`'s lines."""
    kept = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        scene, variant, *_, share = (p.strip() for p in line.split("|"))
        kept[scene, variant] = float(share.split()[-1])
    return kept


def ranks(values: list[float]) -> np.ndarray:
    """Ranks 0..n-1 of distinct values."""
    return np.argsort(np.argsort(values))


def main() -> None:
    """Print the per-scene and aggregate tables of every split in the run."""
    path, inputs = Path(sys.argv[1]), Path(sys.argv[2])
    kept = kept_shares(inputs)
    run = yaml.safe_load(path.read_text(encoding="utf-8"))
    variants = [v["name"] for v in run["config"]["variants"]]
    print(f"run {path}, git {run['git']}; cm, accuracy / completeness / Chamfer")
    for split, report in run["splits"].items():
        scenes = report.get("scenes", {})
        print(f"\n## {split}: per scene")
        for name, record in scenes.items():
            print(f"{name}")
            for v in variants:
                h = record["variants"][v]["headline"]
                print(f"  {v:>9} | {fmt([h[m] for m in SURFACE])}")
        if "aggregate" not in report:
            continue
        print(f"\n## {split}: aggregate ({len(scenes)} scenes), Δ vs {REFERENCE}")
        print("variant | aggregate | Δ aggregate | Δ median scene | scenes worse")
        ref = report["aggregate"][REFERENCE]["headline"]
        for v in variants:
            h = report["aggregate"][v]["headline"]
            per_scene = np.array(
                [
                    [
                        r["variants"][v]["headline"][m]
                        - r["variants"][REFERENCE]["headline"][m]
                        for m in SURFACE
                    ]
                    for r in scenes.values()
                ]
            )
            worse = " / ".join(str(int(n)) for n in (per_scene > 0).sum(axis=0))
            print(
                f"{v} | {fmt([h[m] for m in SURFACE])}"
                f" | {fmt([h[m] - ref[m] for m in SURFACE], '+')}"
                f" | {fmt(list(np.median(per_scene, axis=0)), '+')} | {worse}"
            )
        seconds = [
            np.mean(
                [
                    r["variants"][v]["supplementary"]["refinement_s"]
                    for r in scenes.values()
                ]
            )
            for v in variants
        ]
        print(
            "refinement s per scene: "
            + ", ".join(f"{v} {s:.2f}" for v, s in zip(variants, seconds, strict=True))
        )
        refined = [v for v in variants if any((n, v) in kept for n in scenes)]
        print(
            "E_mv kept, mean over scenes: "
            + ", ".join(
                f"{v} {np.mean([kept[n, v] for n in scenes]):.3f}" for v in refined
            )
        )
        rho = [
            np.corrcoef(
                ranks([kept[n, v] for v in refined]),
                ranks(
                    [r["variants"][v]["headline"]["completeness_cm"] for v in refined]
                ),
            )[0, 1]
            for n, r in scenes.items()
        ]
        print(
            f"Spearman(kept, completeness) per scene over {len(refined)} variants:"
            f" median {np.median(rho):.2f}, min {np.min(rho):.2f}"
        )


if __name__ == "__main__":
    main()
