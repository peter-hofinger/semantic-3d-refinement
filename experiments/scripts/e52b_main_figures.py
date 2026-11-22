"""E52b main arms, rung 5: LoRA + segment pooling (GT instances vs grid tiles) against LoRA alone.

Reads `runs/e52b_main_<arm>_s<seed>` (rung 5 on the 3080, 3000 steps, seeds 0-2, state kept on
val, dev scored) and the control's three seeds `runs/e52b_control{,_s1,_s2}` (LoRA without
segments, same 3080, same script). Dev Chamfer is scored on 20 views per scene after one Umeyama
Sim(3) onto the GT (`src/evaluation/labeled.py`); the plan's margin is the control's seed range.

1. `e52b_main.jpg`: dev Chamfer per scene and mean (seeds as dots, seed mean as bar) for Déjà
   View, LoRA alone and both pooling arms; val ratio to Déjà View per eval step for every run.
2. `results/e52b_main.yaml`: those numbers.

uv run python experiments/scripts/e52b_main_figures.py
"""

from pathlib import Path

import matplotlib as mpl
import numpy as np
import yaml

mpl.use("Agg")
import matplotlib.pyplot as plt
from e52a_figures import FIGURES, style

RUNS = Path("runs")
RESULTS = Path("experiments/results/e52b_main.yaml")
BASE = "Déjà View"
SCENES = {
    "ai_010_007_cam_00": "living room",
    "ai_006_007_cam_00": "bedroom",
    "ai_044_001_cam_00": "lobby",
}
ARMS = {
    "LoRA alone": (["e52b_control", "e52b_control_s1", "e52b_control_s2"], "lora-none", "0.3"),
    "LoRA + pool, GT instances": ([f"e52b_main_gt_s{s}" for s in range(3)], "lora_pool-gt", "#eb6834"),
    "LoRA + pool, grid tiles": ([f"e52b_main_grid_s{s}" for s in range(3)], "lora_pool-grid", "#3a7bd5"),
}  # fmt: skip


def load(run: str, label: str) -> tuple[dict, dict]:
    """Return one run's model record and dev eval, checked against the plan's settings."""
    rec = yaml.safe_load((RUNS / run / "e52a.yaml").read_text(encoding="utf-8"))
    model = rec["records"]["models"][label]
    assert model["selected_on"] == "val", model["selected_on"]
    assert len(model["train_loss"]) == 3000, len(model["train_loss"])
    assert rec["config"]["lr_lora"] == 1e-4, rec["config"]["lr_lora"]
    assert rec["config"]["align"] == "exact", rec["config"]["align"]
    return model, rec["records"]["eval"]


def numbers() -> dict:
    """Dev Chamfer per seed and scene, seed means, ranges and the val curves of every run."""
    base = None
    out: dict = {"arms": {}}
    for arm, (runs, label, _) in ARMS.items():
        seeds = []
        for run in runs:
            model, ev = load(run, label)
            dv = [float(ev[s][BASE]["chamfer_cm"]) for s in SCENES]
            base = dv if base is None else base
            assert np.allclose(dv, base, atol=1e-9), (run, dv, base)
            seeds.append({
                "run": run,
                "best_step": model["best_step"],
                "chamfer_cm": [float(ev[s][label]["chamfer_cm"]) for s in SCENES],
                "ray_deg": [float(ev[s][label]["ray_deg"]) for s in SCENES],
                "val": {int(k): float(v) for k, v in model["val"].items()},
                "held_out": {int(k): float(v) for k, v in model["held_out"].items()},
            })  # fmt: skip
        c = np.array([s["chamfer_cm"] for s in seeds])
        out["arms"][arm] = {
            "seeds": seeds,
            "seed_mean_cm": [float(x) for x in c.mean(0)],
            "mean_cm": float(c.mean()),
            "seed_range_cm": float(np.ptp(c.mean(1))),
        }
    out[BASE] = {"chamfer_cm": base, "mean_cm": float(np.mean(base))}
    mean = {a: np.array(r["seed_mean_cm"]) for a, r in out["arms"].items()}
    ctrl, gt, grid = mean.values()
    out["gaps_cm"] = {
        "gt - grid": [float(x) for x in gt - grid] + [float((gt - grid).mean())],
        "gt - LoRA alone": [float(x) for x in gt - ctrl] + [float((gt - ctrl).mean())],
        "grid - LoRA alone": [float(x) for x in grid - ctrl]
        + [float((grid - ctrl).mean())],
    }
    return out


def figure(res: dict) -> None:
    """Dev Chamfer per scene and mean; val ratio during training."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.0), width_ratios=[1.3, 1])
    ax = axes[0]
    names = [*SCENES.values(), "mean"]
    base = [*res[BASE]["chamfer_cm"], res[BASE]["mean_cm"]]
    width = 0.27
    for k, (arm, (_, _, colour)) in enumerate(ARMS.items()):
        r = res["arms"][arm]
        x = np.arange(4) + (k - 1) * width
        ratio = np.array([*r["seed_mean_cm"], r["mean_cm"]]) / base
        ax.bar(
            x, ratio, width * 0.9, color=colour, alpha=0.75, label=f"{arm}, seed mean"
        )
        for s in r["seeds"]:
            c = np.array([*s["chamfer_cm"], np.mean(s["chamfer_cm"])]) / base
            ax.plot(x, c, "o", color="k", ms=3)
    ax.axhline(1.0, color="0.4", lw=0.8, ls=":", label="Déjà View")
    ax.set_xticks(
        range(4),
        [f"{n}\n(Déjà View {b:.2f} cm)" for n, b in zip(names, base, strict=True)],
    )
    ax.set_ylabel("dev Chamfer / Déjà View's (lower is better)")
    ax.set_ylim(0.8, 1.05)
    ax.set_title("dev Chamfer, 20 views per scene (dots: seeds 0-2)", fontsize=10)
    ax.legend(
        fontsize=7.5,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.2),
        ncol=2,
    )
    ax = axes[1]
    for arm, (_, _, colour) in ARMS.items():
        for k, s in enumerate(res["arms"][arm]["seeds"]):
            ax.plot(list(s["val"]), list(s["val"].values()), color=colour, alpha=0.8,
                    ls=("-", "--", ":")[k], label=f"{arm}, seed {k}")  # fmt: skip
    ax.axhline(1.0, color="0.4", lw=0.8, ls=":")
    ax.set_xlabel("step")
    ax.set_ylabel("val ratio to Déjà View's 3D error (11 scenes)")
    ax.set_title("val during training (kept state = least val)", fontsize=10)
    ax.legend(fontsize=6, frameon=False, ncol=1, loc="upper right")
    for ax in axes:
        style(ax)
    fig.suptitle(
        "Segment pooling on top of LoRA gains nothing: GT instances (3.245 cm) tie grid tiles "
        "(3.224 cm) and neither beats LoRA alone (3.155 cm); 3 seeds per arm, 3080.",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(FIGURES / "e52b_main.jpg", dpi=150)


def main() -> None:
    """Write the raw numbers and the figure."""
    res = numbers()
    RESULTS.write_text(
        yaml.safe_dump(res, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    for arm, r in res["arms"].items():
        print(f"{arm:28s} {np.round(r['seed_mean_cm'], 3)} mean {r['mean_cm']:.3f} "
              f"range {r['seed_range_cm']:.3f} steps {[s['best_step'] for s in r['seeds']]}")  # fmt: skip
    for k, v in res["gaps_cm"].items():
        print(f"{k:18s} {np.round(v, 3)}")
    figure(res)


if __name__ == "__main__":
    main()
