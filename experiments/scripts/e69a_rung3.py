"""E69a rung 3 figure: the bedroom's Chamfer of every arm against Deja View and the oracles, and the fixed error of each arm's training.

uv run python experiments/scripts/e69a_rung3.py runs/e69a experiments/figures/e69a_rung3.jpg
"""

import sys
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import yaml

BEDROOM = "ai_006_007_cam_00"
DEJAVIEW = "Déjà View"
ARMS = ["lattice", "gated", "implicit", "split", "seeded", "seeded-render"]
BASELINES = [
    DEJAVIEW,
    "Déjà View, kept 0.75",
    "Déjà View, kept 0.5",
    "voxel mean, densest half",
]


def main() -> None:
    """Draw the bedroom's Chamfer per variant and the fixed train error per step."""
    runs, out = Path(sys.argv[1]), Path(sys.argv[2])
    scores = max(runs.glob("overfit1_eval-*")) / "run.yaml"
    scene = yaml.safe_load(scores.read_text())["splits"]["dev"]["scenes"][BEDROOM]
    names = [*BASELINES, "oracle-lattice", "oracle-seeded", *(f"{a}-s0" for a in ARMS)]
    fig, (left, right) = plt.subplots(1, 2, figsize=(13, 4.8))
    values = [scene[n]["chamfer_cm"] for n in names]
    colors = ["0.6"] * len(BASELINES) + ["tab:green"] * 2 + ["tab:blue"] * len(ARMS)
    left.barh(names, values, color=colors)
    left.axvline(scene[DEJAVIEW]["chamfer_cm"], color="k", lw=1, ls="--")
    left.invert_yaxis()
    left.set_xlabel("Chamfer on the bedroom ai_006_007 (cm), lower is better")
    left.set_title("Overfit the bedroom: no arm reaches its oracle")
    for i, v in enumerate(values):
        left.text(v, i, f" {v:.2f}", va="center", fontsize=8)
    for arm in ARMS:
        record = yaml.safe_load(
            (runs / "overfit1" / f"e69a-overfit1-{arm}-s0.yaml").read_text()
        )
        steps = [e["step"] for e in record["curve"]]
        right.plot(
            steps, [e["train"]["chamfer_cells"] for e in record["curve"]], label=arm
        )
    right.set_xlabel("training step (constant learning rate, 1 200 steps)")
    right.set_ylabel("fixed train error, Chamfer (cells of 192)")
    right.set_title("The error falls, then stalls at about 0.3 cells")
    right.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=110)


if __name__ == "__main__":
    main()
