"""E69c: where the ladder stopped, against its gates.

Left: each dev scene's oracle-ovoxel Chamfer as a share of Deja View's at split 2, 3, 4 (the
gate: at most 0.5 on every dev scene; the smallest split that holds is the one used). Right:
`dv-ovoxel` minus Deja View (cm, mean over the split's scenes) at each split, against the
plan's tie band of +-0.1 cm (the other half of rung 0's gate). Both from the newest rung-0
run of `cloud_eval`. Later rungs have no bars: none was run.

uv run python -m experiments.scripts.e69c_story experiments/figures/e69c_story.jpg
"""

import sys

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

from experiments.scripts.e69a_gate import BAND
from experiments.scripts.e69c_gate import RUNS, SPLITS, chamfers


def main() -> None:
    """Draw the figure from the newest rung-0 scores."""
    report = yaml.safe_load(
        max(RUNS.glob("rung0_eval-*")).joinpath("run.yaml").read_text()
    )["splits"]
    fig, (left, right) = plt.subplots(
        1, 2, figsize=(12.5, 4.8), constrained_layout=True
    )
    dv = chamfers(report, "dev", ["Déjà View"])
    scenes = [s.rsplit("_cam", 1)[0] for s in sorted(report["dev"]["scenes"])]
    width = 0.25
    for i, k in enumerate(SPLITS):
        ratio = chamfers(report, "dev", [f"oracle-ovoxel-s{k}"]) / dv
        left.bar(np.arange(3) + (i - 1) * width, ratio, width, label=f"split {k}")
    left.axhline(0.5, color="red")
    left.text(2.4, 0.51, "gate: at most 0.5", color="red", ha="right", va="bottom")
    left.set_xticks(range(3), scenes)
    left.set_ylabel("oracle-ovoxel Chamfer / Déjà View's (dev scene)")
    left.set_title("oracle: below half of Déjà View on every dev scene?")
    left.legend()
    for name, colour in (("dev", "tab:blue"), ("val", "tab:orange")):
        base = chamfers(report, name, ["Déjà View"])
        gap = [
            (chamfers(report, name, [f"dv-ovoxel-s{k}"]) - base).mean() for k in SPLITS
        ]
        right.plot(SPLITS, gap, "o-", c=colour, label=name)
    right.axhspan(-BAND, BAND, color="red", alpha=0.25, label=f"tie band +-{BAND} cm")
    right.set_xticks(SPLITS, [f"split {k}" for k in SPLITS])
    right.set_ylabel("dv-ovoxel minus Déjà View (cm, mean over scenes)")
    right.set_title("Déjà View in the O-Voxel representation: free of cost?")
    right.legend()
    fig.suptitle(
        "E69c rung 0: the oracle gate holds from split 4; the dv-ovoxel gate fails at every split, "
        "so no later rung ran",
        fontsize=11,
    )
    fig.savefig(sys.argv[1], dpi=100)


if __name__ == "__main__":
    main()
