"""E52a, fourth run, figure and raw results: pool adapters trained with the exact gradient.

Reads the fourth run (`runs/e52a4/e52a.yaml`: pool-gt and pool-grid, kept state chosen on
val) and E52d's arm F (`runs/e52d_F/e52a.yaml`: pool-grid, same draws, same GPU, kept state
chosen on held-out), both from `e52a_segment_adapters.py`.

1. `e52a4_training.jpg`: training ratio (model's error over Déjà View's on the step's draw,
   50-step means); held-out and val ratios every 250 steps, F's held-out next to them; dev
   Chamfer of each kept model minus Déjà View's, per scene.
2. `results/e52a4_train.yaml`: the curves above, the kept steps by val and by held-out, the
   probe's stability rule per model (held-out rise after its minimum, linear trend of the
   training ratio over the last 500 steps) and the dev scores.

uv run python experiments/scripts/e52a4_figures.py runs/e52a4 runs/e52d_F
"""

import argparse
from pathlib import Path

import matplotlib as mpl
import numpy as np
import yaml

mpl.use("Agg")
import matplotlib.pyplot as plt
from e52a2_figures import SHORT
from e52a_figures import FIGURES, style
from e52a_segment_adapters import BASE

RESULTS = Path("experiments/results")
BLOCK = 50  # steps per mean of the training curve
TREND = 500  # steps of the trend in the probe's stability rule
COLOURS = {"pool-gt": "#eb6834", "pool-grid": "#3a7bd5", "F": "0.55"}


def blocks(loss: list[float]) -> np.ndarray:
    """Means of consecutive BLOCK-step windows."""
    x = np.asarray(loss, dtype=np.float64)
    assert x.ndim == 1
    assert len(x) % BLOCK == 0
    return x.reshape(-1, BLOCK).mean(axis=1)


def summary(rec: dict) -> dict:
    """Kept steps by val and held-out, and the probe's stability rule, of one model."""
    held = rec["held_out"]
    best = min(held, key=held.get)
    tail = np.asarray(rec["train_loss"][-TREND:], dtype=np.float64)
    slope = np.polyfit(np.arange(TREND), tail, 1)[0]
    out = {
        "kept_on_held_out": int(best),
        "held_out_min": float(held[best]),
        "held_out_last": float(held[max(held)]),
        "held_out_rise_after_min": float(held[max(held)] - held[best]),
        "train_trend_last_500": float(slope * TREND),
    }
    if "val" in rec:
        val = rec["val"]
        out |= {
            "kept_on_val": int(min(val, key=val.get)),
            "val_min": float(min(val.values())),
        }
    return out


def figure(models: dict, f: dict, evals: dict) -> None:
    """Figure 1 of the module docstring."""
    fig, (train, held, dev) = plt.subplots(1, 3, figsize=(17, 4.6))
    curves = models | {"F": f}
    for label, rec in curves.items():
        b = blocks(rec["train_loss"])
        train.plot(BLOCK * (np.arange(len(b)) + 0.5), b, color=COLOURS[label],
                   lw=1.0, label=label if label != "F" else "E52d arm F (pool-grid)")  # fmt: skip
        steps = sorted(rec["held_out"])
        held.plot(steps, [rec["held_out"][s] for s in steps], "-", color=COLOURS[label],
                  label=f"{label if label != 'F' else 'arm F'} held-out")  # fmt: skip
        if "val" in rec:
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
    for i, label in enumerate(models):
        delta = [evals[s][label]["chamfer_cm"] - evals[s][BASE]["chamfer_cm"]
                 for s in scenes]  # fmt: skip
        dev.bar(np.arange(len(scenes)) + (i - 0.5) * 0.38, delta, 0.38,
                color=COLOURS[label], label=label)  # fmt: skip
    dev.axhline(0, color="0.3", lw=0.8)
    dev.set_xticks(range(len(scenes)), [SHORT[s] for s in scenes])
    dev.set_ylabel("Chamfer minus Déjà View's (cm, 20 views)")
    dev.set_title("dev, kept models; the rule needs gt below grid by 0.05 cm",
                  fontsize=10)  # fmt: skip
    dev.legend(fontsize=7, frameon=False)
    style(dev)
    fig.suptitle(
        "E52a, fourth run (exact gradient, 3080): both training ratios keep falling (left),"
        " both models gain on all 3 dev scenes,\nGT instances no more than grid tiles"
        " (right). pool-grid"
        " reruns arm F on the same draws: its held-out curve departs from F's by as much"
        " as GT's from grid's (middle)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(FIGURES / "e52a4_training.jpg", dpi=110, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Draw the figure and write the raw results from the fourth run and arm F."""
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("f", type=Path)
    args = parser.parse_args()
    run = yaml.safe_load((args.run / "e52a.yaml").read_text(encoding="utf-8"))
    f_run = yaml.safe_load((args.f / "e52a.yaml").read_text(encoding="utf-8"))
    models = run["records"]["models"]
    f = f_run["records"]["models"]["pool-grid"]
    assert list(models) == ["pool-gt", "pool-grid"]
    assert np.allclose(
        f["diagnostics"]["base"], models["pool-grid"]["diagnostics"]["base"]
    )
    evals = run["records"]["eval"]
    figure(models, f, evals)
    scenes = list(SHORT)
    chamfer = {
        label: {SHORT[s]: float(evals[s][label]["chamfer_cm"]) for s in scenes}
        for label in (BASE, *models)
    }
    for c in chamfer.values():
        c["mean"] = float(np.mean(list(c.values())))
    out = {
        "git": run["git"],
        "f_git": f_run["git"],
        "models": {
            label: summary(rec)
            | {
                "held_out": {int(k): float(v) for k, v in rec["held_out"].items()},
                "val": {int(k): float(v) for k, v in rec["val"].items()},
                "train_50": [float(x) for x in blocks(rec["train_loss"])],
            }
            for label, rec in models.items()
        },
        "arm_F": summary(f)
        | {"held_out": {int(k): float(v) for k, v in f["held_out"].items()}},
        "dev_chamfer_cm": chamfer,
    }
    header = "# E52a, fourth run: written by experiments/scripts/e52a4_figures.py\n"
    (RESULTS / "e52a4_train.yaml").write_text(
        header + yaml.safe_dump(out, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
