"""E87a rung 5: LoRA + gated segment pooling (GT instances vs grid tiles) against E52b's arms.

Reads `runs/e87a_main_<arm>_s<seed>` (rung 5 on the 3080, 3000 steps, seeds 0-2, lr_gate 1e-3,
state kept on val, dev scored), E52b's ungated arms `runs/e52b_main_<arm>_s<seed>` and the
control's three seeds `runs/e52b_control{,_s1,_s2}` (LoRA without segments); same script, same
3080. Dev Chamfer is scored on 20 views per scene after one Umeyama Sim(3) onto the GT
(`src/evaluation/labeled.py`); the plan's margin is the control's seed range (0.080 cm, E52c).

1. `e87a_main.jpg`: dev Chamfer per scene and mean (seeds as dots, seed mean as bar) for LoRA
   alone and the four pooling arms; val ratio to Déjà View per eval for the gated runs; the gate
   parameters' norm during training.
2. `results/e87a_main.yaml`: those numbers.

uv run python experiments/scripts/e87a_main_figures.py
"""

from pathlib import Path

import matplotlib as mpl
import numpy as np
import yaml

mpl.use("Agg")
import matplotlib.pyplot as plt
from e52a_figures import FIGURES, style

RUNS = Path("runs")
RESULTS = Path("experiments/results/e87a_main.yaml")
BASE = "Déjà View"
SCENES = {
    "ai_010_007_cam_00": "living room",
    "ai_006_007_cam_00": "bedroom",
    "ai_044_001_cam_00": "lobby",
}
ARMS = {
    "LoRA alone": (["e52b_control", "e52b_control_s1", "e52b_control_s2"], "lora-none", "0.3"),
    "LoRA + pool, GT (E52b)": ([f"e52b_main_gt_s{s}" for s in range(3)], "lora_pool-gt", "#f4b393"),
    "LoRA + pool, grid (E52b)": ([f"e52b_main_grid_s{s}" for s in range(3)], "lora_pool-grid", "#a3c4ef"),
    "LoRA + gated pool, GT": ([f"e87a_main_gt_s{s}" for s in range(3)], "lora_pool_gate-gt", "#eb6834"),
    "LoRA + gated pool, grid": ([f"e87a_main_grid_s{s}" for s in range(3)], "lora_pool_gate-grid", "#3a7bd5"),
}  # fmt: skip
GATED = ("LoRA + gated pool, GT", "LoRA + gated pool, grid")
GAPS = {
    "gated GT - gated grid": ("LoRA + gated pool, GT", "LoRA + gated pool, grid"),
    "gated GT - LoRA alone": ("LoRA + gated pool, GT", "LoRA alone"),
    "gated grid - LoRA alone": ("LoRA + gated pool, grid", "LoRA alone"),
    "gated GT - ungated GT": ("LoRA + gated pool, GT", "LoRA + pool, GT (E52b)"),
    "gated grid - ungated grid": (
        "LoRA + gated pool, grid",
        "LoRA + pool, grid (E52b)",
    ),
}


def load(run: str, label: str) -> tuple[dict, dict]:
    """Return one run's model record and dev eval, checked against the plan's settings."""
    rec = yaml.safe_load((RUNS / run / "e52a.yaml").read_text(encoding="utf-8"))
    model = rec["records"]["models"][label]
    assert model["selected_on"] == "val", model["selected_on"]
    assert len(model["train_loss"]) == 3000, len(model["train_loss"])
    assert rec["config"]["lr_lora"] == 1e-4, rec["config"]["lr_lora"]
    assert rec["config"]["align"] == "exact", rec["config"]["align"]
    if "gate" in label:
        assert rec["config"]["lr_gate"] == 1e-3, rec["config"]["lr_gate"]
    return model, rec["records"]["eval"]


def numbers() -> dict:
    """Dev Chamfer per seed and scene, seed means, ranges, val curves and gate norms."""
    base = None
    out: dict = {"arms": {}}
    for arm, (runs, label, _) in ARMS.items():
        seeds = []
        for run in runs:
            model, ev = load(run, label)
            dv = [float(ev[s][BASE]["chamfer_cm"]) for s in SCENES]
            base = dv if base is None else base
            assert np.allclose(dv, base, atol=1e-9), (run, dv, base)
            seed = {
                "run": run,
                "best_step": model["best_step"],
                "chamfer_cm": [float(ev[s][label]["chamfer_cm"]) for s in SCENES],
                "val": {int(k): float(v) for k, v in model["val"].items()},
            }
            if "gate" in label:
                norms = model["diagnostics"]["norm"]["gate"]
                seed["gate_norm"] = {int(k): float(v) for k, v in norms.items()}
            seeds.append(seed)
        c = np.array([s["chamfer_cm"] for s in seeds])
        out["arms"][arm] = {
            "seeds": seeds,
            "seed_mean_cm": [float(x) for x in c.mean(0)],
            "mean_cm": float(c.mean()),
            "seed_range_cm": float(np.ptp(c.mean(1))),
        }
    out[BASE] = {"chamfer_cm": base, "mean_cm": float(np.mean(base))}
    mean = {a: np.array(r["seed_mean_cm"]) for a, r in out["arms"].items()}
    out["gaps_cm"] = {
        k: [float(x) for x in mean[a] - mean[b]] + [float((mean[a] - mean[b]).mean())]
        for k, (a, b) in GAPS.items()
    }
    return out


def figure(res: dict) -> None:
    """Dev Chamfer per scene and mean; val ratio and gate norm during training."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2), width_ratios=[1.4, 1, 0.8])
    ax = axes[0]
    names = [*SCENES.values(), "mean"]
    base = [*res[BASE]["chamfer_cm"], res[BASE]["mean_cm"]]
    width = 0.17
    for k, (arm, (_, _, colour)) in enumerate(ARMS.items()):
        r = res["arms"][arm]
        x = np.arange(4) + (k - 2) * width
        ratio = np.array([*r["seed_mean_cm"], r["mean_cm"]]) / base
        ax.bar(
            x, ratio, width * 0.9, color=colour, alpha=0.85, label=f"{arm}, seed mean"
        )
        for s in r["seeds"]:
            c = np.array([*s["chamfer_cm"], np.mean(s["chamfer_cm"])]) / base
            ax.plot(x, c, "o", color="k", ms=2.5)
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
        ncol=3,
    )
    for arm in GATED:
        colour = ARMS[arm][2]
        for k, s in enumerate(res["arms"][arm]["seeds"]):
            ls = ("-", "--", ":")[k]
            axes[1].plot(list(s["val"]), list(s["val"].values()), color=colour,
                         ls=ls, label=f"{arm}, seed {k}")  # fmt: skip
            axes[2].plot(list(s["gate_norm"]), list(s["gate_norm"].values()),
                         color=colour, ls=ls, label=f"{arm}, seed {k}")  # fmt: skip
    axes[1].axhline(1.0, color="0.4", lw=0.8, ls=":")
    axes[1].set_ylabel("val ratio to Déjà View's 3D error (11 scenes)")
    axes[1].set_title("val during training (kept state = least val)", fontsize=10)
    axes[2].set_ylabel("norm of the gate parameters (w_k, b_k), all loop steps")
    axes[2].set_title("gate parameters during training", fontsize=10)
    for ax in axes[1:]:
        ax.set_xlabel("step")
        ax.legend(fontsize=6, frameon=False, loc="best")
    for ax in axes:
        style(ax)
    g = {a: res["arms"][a]["mean_cm"] for a in ARMS}
    fig.suptitle(
        f"A per-token gate changes nothing: gated GT ({g[GATED[0]]:.3f} cm) ties gated grid "
        f"({g[GATED[1]]:.3f} cm) and the ungated arms; none beats LoRA alone "
        f"({g['LoRA alone']:.3f} cm); 3 seeds per arm, 3080.",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(FIGURES / "e87a_main.jpg", dpi=150)


def main() -> None:
    """Write the raw numbers and the figure."""
    res = numbers()
    RESULTS.write_text(
        yaml.safe_dump(res, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    for arm, r in res["arms"].items():
        print(f"{arm:26s} {np.round(r['seed_mean_cm'], 3)} mean {r['mean_cm']:.3f} "
              f"range {r['seed_range_cm']:.3f} steps {[s['best_step'] for s in r['seeds']]} "
              f"seeds {[round(float(np.mean(s['chamfer_cm'])), 3) for s in r['seeds']]}")  # fmt: skip
    for k, v in res["gaps_cm"].items():
        print(f"{k:26s} {np.round(v, 3)}")
    for arm in GATED:
        for s in res["arms"][arm]["seeds"]:
            n = s["gate_norm"]
            kept = n[min(n, key=lambda t: abs(t - s["best_step"]))]
            print(
                f"{s['run']}: gate norm at 3000 {n[max(n)]:.2f}, near kept step {kept:.2f}"
            )
    figure(res)


if __name__ == "__main__":
    main()
