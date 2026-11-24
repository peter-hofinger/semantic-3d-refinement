"""E94a (issue #94): val Chamfer of the 15 kept models of E52b and E87a, no training.

Reads `runs/e94a_val/e52a.yaml` (`e52a_segment_adapters.py --score`, 3080, 20 views per scene):
LoRA alone (E52b's control, `lora-none`), LoRA + pooling over GT instances / 3 x 3 grid tiles
(E52b) and LoRA + gated pooling (E87a), seeds 0-2 each, Déjà View in the same pass. Every
output is aligned onto the GT by one Umeyama Sim(3) and scored by `src/evaluation/labeled.py`.
The kept states were selected on these 11 val scenes (least training ratio), so the gain over
Déjà View is an upper estimate; arms compare fairly.

Check first: the rescored dev Chamfer of every model must match its record in
`results/e87a_main.yaml` within 0.014 cm (eval-to-eval noise), else val is not read.
Rule (LOG.md, fixed before the run): m = max(0.080 cm, LoRA alone's val seed range); an arm
beats another if its seed-mean val Chamfer is lower by more than m and lower on >= 6 of 11.

1. `e94a_val.jpg`: val Chamfer / Déjà View's per scene and mean (bars: seed mean, dots: seeds);
   val accuracy, completeness, Chamfer per arm (cm).
2. `results/e94a_val.yaml`: those numbers, the dev check and the rule's verdicts.

uv run python experiments/scripts/e94a_val_figures.py
"""

from pathlib import Path

import matplotlib as mpl
import numpy as np
import yaml

mpl.use("Agg")
import matplotlib.pyplot as plt
from e52a_figures import FIGURES, style
from e87a_main_figures import ARMS, BASE, SCENES

RUN = Path("runs/e94a_val/e52a.yaml")
DEV_RECORD = Path("experiments/results/e87a_main.yaml")
RESULTS = Path("experiments/results/e94a_val.yaml")
DEV_TOLERANCE = 0.014  # cm, val moves up to this between evals (E52b rerun)
MARGIN = 0.080  # cm, LoRA alone's dev seed range (E52c)
MIN_SCENES = 6
CONTROL = "LoRA alone"
METRICS = ("accuracy_cm", "completeness_cm", "chamfer_cm")


def dev_check(ev: dict) -> dict:
    """Largest |rescored - recorded| dev Chamfer per model; asserts the tolerance."""
    record = yaml.safe_load(DEV_RECORD.read_text(encoding="utf-8"))["arms"]
    out = {}
    for arm, (runs, label, _) in ARMS.items():
        for run, seed in zip(runs, record[arm]["seeds"], strict=True):
            assert seed["run"] == run, (seed["run"], run)
            now = [ev[s][f"{run}/{label}"]["chamfer_cm"] for s in SCENES]
            out[run] = float(np.abs(np.array(now) - seed["chamfer_cm"]).max())
    worst = max(out.values())
    assert worst <= DEV_TOLERANCE, (
        f"dev rescore off by {worst:.4f} cm: loading is wrong"
    )
    return out


def beats(a: np.ndarray, b: np.ndarray, m: float) -> dict:
    """Seed-mean Chamfer per scene a vs b: mean gap, scenes a is lower, verdict at m."""
    gap, lower = float((a - b).mean()), int((a < b).sum())
    return {"gap_cm": gap, "scenes_lower": lower, "of": len(a),
            "beats": bool(gap < -m and lower >= MIN_SCENES)}  # fmt: skip


def numbers() -> dict:
    """Val metrics per seed and scene, seed means and ranges, the dev check, the rule."""
    rec = yaml.safe_load(RUN.read_text(encoding="utf-8"))
    assert all(k["selected_on"] == "val" for k in rec["records"]["kept"].values())
    assert not any(k["is_zero"] for k in rec["records"]["kept"].values())
    ev = rec["records"]["eval"]
    val = [s for s in ev if s not in SCENES]
    assert len(val) == 11, val
    out: dict = {"git": rec["git"], "val_scenes": val, "dev_check_cm": dev_check(ev)}
    out[BASE] = {k: [float(ev[s][BASE][k]) for s in val] for k in METRICS}
    out["arms"] = {}
    for arm, (runs, label, _) in ARMS.items():
        per = {k: np.array([[ev[s][f"{r}/{label}"][k] for s in val] for r in runs])
               for k in METRICS}  # fmt: skip
        c = per["chamfer_cm"]
        out["arms"][arm] = {
            "runs": runs,
            "kept_steps": [
                rec["records"]["kept"][f"{r}/{label}"]["best_step"] for r in runs
            ],
            "chamfer_cm": c.tolist(),
            "seed_mean_cm": c.mean(0).tolist(),
            "seed_means_cm": c.mean(1).tolist(),
            "seed_range_cm": float(np.ptp(c.mean(1))),
            "mean_cm": {k: float(v.mean()) for k, v in per.items()},
        }
    m = max(MARGIN, out["arms"][CONTROL]["seed_range_cm"])
    mean = {a: np.array(r["seed_mean_cm"]) for a, r in out["arms"].items()}
    dv = np.array(out[BASE]["chamfer_cm"])
    out["rule"] = {"margin_cm": m,
                   "control_vs_dejaview": beats(mean[CONTROL], dv, m),
                   "arm_vs_control": {a: beats(mean[a], mean[CONTROL], m)
                                      for a in ARMS if a != CONTROL},
                   "arm_vs_dejaview": {a: beats(mean[a], dv, m) for a in ARMS}}  # fmt: skip
    return out


def figure(res: dict) -> None:
    """Val Chamfer ratio per scene and mean; val accuracy, completeness, Chamfer per arm."""
    fig, axes = plt.subplots(1, 2, figsize=(18, 5.4), width_ratios=[3, 1])
    ax = axes[0]
    dv = np.array(res[BASE]["chamfer_cm"])
    base = [*dv, dv.mean()]
    width = 0.16
    for k, (arm, (_, _, colour)) in enumerate(ARMS.items()):
        r = res["arms"][arm]
        x = np.arange(len(base)) + (k - 2) * width
        bar = np.array([*r["seed_mean_cm"], np.mean(r["seed_mean_cm"])]) / base
        ax.bar(x, bar, width * 0.9, color=colour, alpha=0.85, label=f"{arm}, seed mean")
        for c in r["chamfer_cm"]:
            ax.plot(x, np.array([*c, np.mean(c)]) / base, "o", color="k", ms=2)
    ax.axhline(1.0, color="0.4", lw=0.8, ls=":", label=BASE)
    ax.set_xticks(
        range(len(base)),
        [f"{n[3:10]}\n{b:.2f}" for n, b in zip(res["val_scenes"], base, strict=False)]
        + [f"mean\n{base[-1]:.2f}"],
        fontsize=7,
    )
    ax.set_xlabel("val scene (Déjà View's Chamfer, cm)")
    ax.set_ylabel("val Chamfer / Déjà View's (lower is better)")
    ax.set_title("val Chamfer, 20 views per scene (dots: seeds 0-2)", fontsize=10)
    ax.legend(fontsize=7.5, frameon=False, loc="upper center",
              bbox_to_anchor=(0.5, -0.2), ncol=6)  # fmt: skip
    ax = axes[1]
    names = ("accuracy", "completeness", "Chamfer")
    for k, (arm, (_, _, colour)) in enumerate(ARMS.items()):
        v = [res["arms"][arm]["mean_cm"][m] for m in METRICS]
        ax.bar(np.arange(3) + (k - 2) * width, v, width * 0.9, color=colour, alpha=0.85)
    ax.plot(np.arange(3), [np.mean(res[BASE][m]) for m in METRICS], "_", color="k",
            ms=40, mew=1.5, label=f"{BASE}")  # fmt: skip
    ax.set_xticks(range(3), names)
    ax.set_ylabel("val mean over 11 scenes, cm (lower is better)")
    ax.set_title("val surface metrics, seed means", fontsize=10)
    ax.legend(fontsize=7.5, frameon=False)
    for a in axes:
        style(a)
    r = res["rule"]
    lora = res["arms"][CONTROL]["mean_cm"]["chamfer_cm"]
    fig.suptitle(
        f"Val (11 scenes, kept states selected on them): LoRA alone {lora:.3f} cm vs "
        f"Déjà View {dv.mean():.3f} cm; margin {r['margin_cm']:.3f} cm; pooling arms vs LoRA "
        + ", ".join(f"{v['gap_cm']:+.3f}" for v in r["arm_vs_control"].values())
        + " cm. 3080.",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(FIGURES / "e94a_val.jpg", dpi=150)


def main() -> None:
    """Write the raw numbers and the figure; print the table and the rule's verdicts."""
    res = numbers()
    RESULTS.write_text(
        yaml.safe_dump(res, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    print(f"dev check, max |diff| {max(res['dev_check_cm'].values()):.4f} cm")
    dv = res[BASE]
    print(f"{BASE:26s} " + " ".join(f"{np.mean(dv[m]):.3f}" for m in METRICS))
    for arm, r in res["arms"].items():
        print(f"{arm:26s} " + " ".join(f"{r['mean_cm'][m]:.3f}" for m in METRICS)
              + f"  seeds {np.round(r['seed_means_cm'], 3)} range {r['seed_range_cm']:.3f}")  # fmt: skip
    print(yaml.safe_dump(res["rule"], sort_keys=False))
    figure(res)


if __name__ == "__main__":
    main()
