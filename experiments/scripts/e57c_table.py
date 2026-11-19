"""E57c: per scene and per split, each cut variant's accuracy / completeness / Chamfer (cm).

Reads one `scene_eval --cfg experiments/configs/e57c_oracle.yaml` run (or e57c.yaml's, with
`--versus` naming the variant that is compared against the first cut one). Per split: the
mean over scenes of each variant, and on how many scenes the last variant's Chamfer is
below the density cut's.

uv run python experiments/scripts/e57c_table.py runs/<run> > experiments/results/e57c_oracle.txt
"""

import sys
from pathlib import Path

import numpy as np
import yaml

METRICS = ("accuracy_cm", "completeness_cm", "chamfer_cm")


def main() -> None:
    """Print the per-scene table and the split means for the run folder in argv."""
    report = yaml.safe_load((Path(sys.argv[1]) / "run.yaml").read_text())["splits"]
    for split, record in report.items():
        scenes = record["scenes"]
        variants = list(next(iter(scenes.values()))["variants"])
        print(
            f"{split}: {' | '.join(variants)}  (accuracy / completeness / Chamfer, cm)"
        )
        table = np.array(
            [
                [[s["variants"][v]["headline"][m] for m in METRICS] for v in variants]
                for s in scenes.values()
            ]
        )
        for name, row in zip(scenes, table, strict=True):
            print(
                f"  {name:20s}",
                " | ".join("/".join(f"{x:.2f}" for x in r) for r in row),
            )
        print(
            f"  {'mean':20s}",
            " | ".join("/".join(f"{x:.3f}" for x in r) for r in table.mean(0)),
        )
        density, last = table[:, 1, 2], table[:, -1, 2]
        print(
            f"  {variants[-1]} below {variants[1]} in Chamfer on {int((last < density).sum())} of {len(scenes)} scenes\n"
        )


if __name__ == "__main__":
    main()
