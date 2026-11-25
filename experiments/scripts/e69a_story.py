"""E69a in one figure: how far each arm got to its oracle on the bedroom, and what the best arm does on scenes it did not train on.

uv run python experiments/scripts/e69a_story.py runs/e69a experiments/figures/e69a_story.jpg
"""

import sys
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

DEJAVIEW = "Déjà View"
BEDROOM = "ai_006_007_cam_00"
UNSEEN = ["ai_010_007_cam_00", "ai_044_001_cam_00"]
ARMS = ["seeded", "split", "gated", "lattice", "implicit", "seeded-render"]
GATE = 0.5


def main() -> None:
    """Draw the gap each arm closes against the gate, and the best arm per dev scene."""
    runs, out = Path(sys.argv[1]), Path(sys.argv[2])
    scenes = yaml.safe_load(
        (max(runs.glob("overfit1_eval-*")) / "run.yaml").read_text()
    )["splits"]["dev"]["scenes"]
    chamfer = {s: {v: d["chamfer_cm"] for v, d in scenes[s].items()} for s in scenes}
    bed = chamfer[BEDROOM]
    gaps = {}
    for arm in ARMS:
        oracle = "oracle-seeded" if arm.startswith("seeded") else "oracle-lattice"
        gaps[arm] = (bed[DEJAVIEW] - bed[f"{arm}-s0"]) / (bed[DEJAVIEW] - bed[oracle])
    fig, (left, right) = plt.subplots(
        1, 2, figsize=(13, 4.6), gridspec_kw={"width_ratios": [1.1, 1]}
    )
    colors = ["tab:blue" if g > 0 else "tab:red" for g in gaps.values()]
    left.barh(list(gaps), list(gaps.values()), color=colors)
    left.axvline(GATE, color="k", ls="--")
    left.axvline(0, color="0.5", lw=0.8)
    left.text(GATE, -0.75, " gate: half the gap", fontsize=9, va="bottom")
    left.set_xlim(-0.3, 0.6)
    left.invert_yaxis()
    for i, g in enumerate(gaps.values()):
        left.text(
            g + (0.01 if g > 0 else -0.01),
            i,
            f"{g:+.2f}",
            va="center",
            ha="left" if g > 0 else "right",
            fontsize=9,
        )
    left.set_xlabel("share of the gap from Déjà View to the oracle that the arm closes")
    left.set_title("Overfit the bedroom: no arm gets halfway")
    names = ["trained on\nbedroom", "unseen\nai_010_007", "unseen\nai_044_001"]
    x = np.arange(3)
    for k, (label, color) in enumerate(
        [("seeded-s0", "tab:blue"), ("oracle-seeded", "tab:green")]
    ):
        ratio = [chamfer[s][label] / chamfer[s][DEJAVIEW] for s in [BEDROOM, *UNSEEN]]
        bars = right.bar(
            x + (k - 0.5) * 0.36,
            ratio,
            0.36,
            color=color,
            label="best arm (seeded)" if k == 0 else "oracle (ceiling)",
        )
        for b, r in zip(bars, ratio, strict=True):
            right.text(
                b.get_x() + b.get_width() / 2,
                r,
                f"{r:.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
    right.axhline(1, color="k", ls="--")
    right.text(2.45, 1.02, "Déjà View", ha="right", fontsize=9)
    right.set_xticks(x, names)
    right.set_ylabel("Chamfer relative to Déjà View (below 1 is better)")
    right.set_title("It wins the scene it trained on, loses the other two")
    right.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=110)


if __name__ == "__main__":
    main()
