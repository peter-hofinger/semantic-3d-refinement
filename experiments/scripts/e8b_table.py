"""E8b: the controls' surface scores, each variant against a reference variant.

Reads the `run.yaml` of `experiments/configs/e8b_dropout.yaml`. Per split and reference
(Off, and the random cut at SAM 3's size, seed 0): each variant's aggregate accuracy,
completeness and Chamfer, its difference from the reference, the median scene's
difference and the number of scenes on which it is worse; and E_mv's matches per
variant as a share of Off's, mean over scenes. Lower is better for all three scores.

uv run python experiments/scripts/e8b_table.py runs/<run>/run.yaml \
    > experiments/results/e8b_table.txt
"""

import sys
from pathlib import Path

import numpy as np
import yaml

SURFACE = ("accuracy_cm", "completeness_cm", "chamfer_cm")
REFERENCES = ("Off", "random @ SAM 3, seed 0")


def fmt(values: list[float], sign: str = "") -> str:
    """Join three numbers as accuracy / completeness / Chamfer."""
    return " / ".join(f"{v:{sign}.3f}" for v in values)


def headline(record: dict, variant: str) -> list[float]:
    """Return one variant's three surface scores in a scene or aggregate record."""
    return [record["variants"][variant]["headline"][m] for m in SURFACE]


def main() -> None:
    """Print per-scene scores, then the aggregate tables of every split."""
    path = Path(sys.argv[1])
    run = yaml.safe_load(path.read_text(encoding="utf-8"))
    variants = [v["name"] for v in run["config"]["variants"]]
    refined = [v["name"] for v in run["config"]["variants"] if v["refiner"] == "joint"]
    print(f"run {path}, git {run['git']}; cm, accuracy / completeness / Chamfer")
    for split, report in run["splits"].items():
        scenes = report.get("scenes", {})
        print(f"\n## {split}: per scene")
        for name, record in scenes.items():
            print(name)
            for v in variants:
                print(f"  {v:>22} | {fmt(headline(record, v))}")
        shares = {
            v: np.mean(
                [
                    r["variants"][v]["log"]["matches"]
                    / r["variants"]["Off"]["log"]["matches"]
                    for r in scenes.values()
                ]
            )
            for v in refined
        }
        print(
            f"\n{split}: E_mv's matches as a share of Off's, mean over scenes: "
            + ", ".join(f"{v} {s:.3f}" for v, s in shares.items())
        )
        if "aggregate" not in report:
            continue
        aggregate = {"variants": report["aggregate"]}
        for ref in REFERENCES:
            print(f"\n## {split}: aggregate ({len(scenes)} scenes), Δ vs {ref}")
            print("variant | aggregate | Δ aggregate | Δ median scene | scenes worse")
            for v in variants:
                per_scene = np.array(
                    [
                        np.subtract(headline(r, v), headline(r, ref))
                        for r in scenes.values()
                    ]
                )
                worse = " / ".join(str(int(n)) for n in (per_scene > 0).sum(axis=0))
                delta = np.subtract(headline(aggregate, v), headline(aggregate, ref))
                print(
                    f"{v} | {fmt(headline(aggregate, v))} | {fmt(list(delta), '+')}"
                    f" | {fmt(list(np.median(per_scene, axis=0)), '+')} | {worse}"
                )


if __name__ == "__main__":
    main()
