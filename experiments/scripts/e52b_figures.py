"""E52b, control, figure and raw results: LoRA on Déjà View's loop block, no segments.

Reads the control (`runs/e52b_control/e52a.yaml`: lora-none, kept state chosen on val), the
lr sweep before it (`runs/e52b_sweep_<lr>/e52a.yaml`, 500 steps each) and E52a's fourth run
(`runs/e52a4/e52a.yaml`: pool-gt, pool-grid; same 3080, same draws), all from
`e52a_segment_adapters.py`.

1. `e52b_control.jpg`: training ratio (model's error over Déjà View's on the step's draw,
   50-step means); held-out and val ratios every 250 steps; dev Chamfer of each kept model
   minus Déjà View's, per scene.
2. `results/e52b_control_train.yaml`: the sweep's ratios, the curves above, kept steps, the
   probe's stability rule per model and the dev scores.

uv run python experiments/scripts/e52b_figures.py runs/e52b_control runs/e52a4 runs/e52b_sweep
"""

import argparse
from pathlib import Path

import matplotlib as mpl
import numpy as np
import yaml

mpl.use("Agg")
import matplotlib.pyplot as plt
from e52a2_figures import SHORT
from e52a4_figures import BLOCK, blocks, summary
from e52a_figures import FIGURES, style
from e52a_segment_adapters import BASE

RESULTS = Path("experiments/results")
LRS = ("1e-5", "3e-5", "1e-4")
COLOURS = {"pool-gt": "#eb6834", "pool-grid": "#3a7bd5", "lora-none": "#2a9d55"}


def load(run: Path) -> dict:
    """Load the records of one run of `e52a_segment_adapters.py`."""
    return yaml.safe_load((run / "e52a.yaml").read_text(encoding="utf-8"))


def figure(models: dict, evals: dict) -> None:
    """Figure 1 of the module docstring."""
    fig, (train, held, dev) = plt.subplots(1, 3, figsize=(17, 4.6))
    for label, rec in models.items():
        b = blocks(rec["train_loss"])
        train.plot(BLOCK * (np.arange(len(b)) + 0.5), b, color=COLOURS[label], lw=1.0,
                   label=label)  # fmt: skip
        steps = sorted(rec["held_out"])
        held.plot(steps, [rec["held_out"][s] for s in steps], "-", color=COLOURS[label],
                  label=f"{label} held-out")  # fmt: skip
        held.plot(steps, [rec["val"][s] for s in steps], "--", color=COLOURS[label],
                  label=f"{label} val (kept: step {rec['best_step']})")  # fmt: skip
    for ax in (train, held):
        ax.axhline(1, color="0.3", lw=0.8)
        ax.set_xlabel("training step")
        style(ax)
    train.set_ylabel("error / Déjà View's on the step's draw")
    train.set_title(f"training ratio, {BLOCK}-step means; grey line: Déjà View",
                    fontsize=10)  # fmt: skip
    train.legend(fontsize=7, frameon=False)
    held.set_ylabel("error / Déjà View's (mean over scenes, 12 fixed views)")
    held.set_title("held-out (16 train scenes) solid, val (11 scenes) dashed",
                   fontsize=10)  # fmt: skip
    held.legend(fontsize=7, frameon=False)
    scenes = list(SHORT)
    width = 0.8 / len(models)
    for i, label in enumerate(models):
        delta = [evals[s][label]["chamfer_cm"] - evals[s][BASE]["chamfer_cm"]
                 for s in scenes]  # fmt: skip
        dev.bar(np.arange(len(scenes)) + (i - (len(models) - 1) / 2) * width, delta,
                width, color=COLOURS[label], label=label)  # fmt: skip
    dev.axhline(0, color="0.3", lw=0.8)
    dev.set_xticks(range(len(scenes)), [SHORT[s] for s in scenes])
    dev.set_ylabel("Chamfer minus Déjà View's (cm, 20 views)")
    dev.set_title("dev, kept models (pool: E52a's fourth run)", fontsize=10)
    dev.legend(fontsize=7, frameon=False)
    style(dev)
    fig.suptitle(
        "E52b, control (LoRA rank 16 on the loop block, no segments, lr 1e-4, 3080): training"
        " ratio falls below the pool adapters' (left), held-out and val fall with a bump at"
        " 750-1000\n(middle); on dev it gains more than both pool adapters on bedroom and"
        " lobby, slightly less on living (right): the bar E52b's segment models must clear",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(FIGURES / "e52b_control.jpg", dpi=110, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Draw the figure and write the raw results from the control, its sweep and E52a4."""
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("pool", type=Path)
    parser.add_argument("sweep", type=str, help="prefix of the sweep's run dirs")
    args = parser.parse_args()
    run, pool = load(args.run), load(args.pool)
    lora = run["records"]["models"]["lora-none"]
    models = pool["records"]["models"] | {"lora-none": lora}
    assert list(models) == ["pool-gt", "pool-grid", "lora-none"]
    evals = run["records"]["eval"]
    for s in SHORT:
        assert np.isclose(
            evals[s][BASE]["chamfer_cm"], pool["records"]["eval"][s][BASE]["chamfer_cm"]
        )
        evals[s] |= {k: pool["records"]["eval"][s][k] for k in ("pool-gt", "pool-grid")}
    figure(models, evals)
    sweep = {}
    for lr in LRS:
        rec = load(Path(f"{args.sweep}_{lr}"))["records"]["models"]["lora-none"]
        sweep[lr] = {
            k: {int(s): float(v) for s, v in rec[k].items()}
            for k in ("held_out", "val")
        }
    chamfer = {
        label: {SHORT[s]: float(evals[s][label]["chamfer_cm"]) for s in SHORT}
        for label in (BASE, *models)
    }
    for c in chamfer.values():
        c["mean"] = float(np.mean(list(c.values())))
    out = {
        "git": run["git"],
        "pool_git": pool["git"],
        "sweep_500_steps": sweep,
        "models": {
            label: summary(rec)
            | {
                "held_out": {int(k): float(v) for k, v in rec["held_out"].items()},
                "val": {int(k): float(v) for k, v in rec["val"].items()},
                "train_50": [float(x) for x in blocks(rec["train_loss"])],
            }
            for label, rec in models.items()
        },
        "dev_chamfer_cm": chamfer,
        "lora_on_dejaview_cameras_chamfer_cm": {
            SHORT[s]: float(evals[s]["lora-none"]["plain_cameras_chamfer_cm"])
            for s in SHORT
        },
        "lora_ray_deg": {
            SHORT[s]: float(evals[s]["lora-none"]["ray_deg"]) for s in SHORT
        },
    }
    header = "# E52b, control: written by experiments/scripts/e52b_figures.py\n"
    (RESULTS / "e52b_control_train.yaml").write_text(
        header + yaml.safe_dump(out, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
