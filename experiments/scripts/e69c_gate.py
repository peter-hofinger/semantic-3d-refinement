"""E69c's gates: score the clouds of a rung and decide whether the next rung may run.

`config <rung>` writes `runs/e69c/<rung>_eval.yaml`: Deja View's pixels, the plain density
filter at three quantiles, the oracle and Deja View in the O-Voxel representation and every
model of the rung's run folder as variants, for `cloud_eval`; `gate <rung>` reads the
newest scores and prints each verdict by the rules written in the LOG's plan, and writes
the arms that pass, best first, to `runs/e69c/<rung>.pass` (exit status 1 if none does,
which stops `e69c.sh`). All Chamfer numbers are cm; a difference below BAND is a tie.

- rung0 (formulation): `split` is the smallest of `SPLITS` whose oracle is at most half of
  Deja View's Chamfer on every dev scene; the gate passes if there is one and `dv-ovoxel`
  at that split is within BAND of Deja View on the dev mean. The split is written to
  `runs/e69c/split.yaml`, and whether it also holds on val is printed.
- profile (rung 2): the train step of every arm takes under 1 s (over 0.5 s is a warning)
  and no job (steps times a step with its validation crop) is over the repo's 30 minute
  limit, unless `ALLOW_LONG` is set.
- lr: each arm's rate is the one of the three probe rungs with the least fixed train error.
- overfit1a (rung 3a, the latent): the bedroom's Chamfer is within the larger of BAND and
  10% of the gap from Deja View to `oracle-ovoxel` above the oracle, and child precision
  and recall are at least 0.95 (the numbers `run_model` records).
- overfit1 (rung 3b), also its escalations `overfit1_e1` ...: the arm closes at least half
  of the bedroom's gap from Deja View to `oracle-ovoxel`.
- overfit3 (rung 4): the dev mean beats Deja View's by more than BAND and at least two of
  the three dev scenes are won by more than BAND.
- generalize (rung 5): at the largest train size the gain over Deja View on the 11 val
  scenes exceeds the larger of BAND and twice the standard error of the per-scene
  differences, and the val Chamfer is no worse than at the smallest size by more than BAND.
  The density filter tuned on dev is printed beside it.
- ablation: the best arm beats Deja View on val by more than BAND, twice the standard error
  and the spread of its seeds' val means; every other arm's cost against it is printed,
  and a part helps if removing it costs more than the larger of the two spreads.
"""

import functools
import os
import sys
from pathlib import Path

import numpy as np
import yaml

from experiments.scripts.e69a_gate import BAND, GAP, beats, gap_closed, parse
from src.refinement.generator import load_rung

LIMIT_MIN = 30  # the repo asks before a job longer than this
HEAD = 0.1  # share of the gap that rung 3a's latent may leave
CHILDREN = 0.95  # child precision and recall that rung 3a asks for
BEDROOM = "ai_006_007_cam_00"
RUNS = Path("runs/e69c")
CONFIG = Path("configs/e69c.yaml")
SPLITS = (2, 3, 4)
QUANTILES = (0.5, 0.7, 0.9)
ARMS = ("ovoxel", "ovoxel-dino")
SCORED = {
    "rung0": ["dev", "val"],
    "overfit1a": ["dev"],
    "overfit1": ["dev"],
    "overfit3": ["dev"],
    "generalize": ["val", "dev"],
    "ablation": ["val", "dev"],
}
RAW = {
    "output_dir": "runs/e69c", "root": "data/processed/hypersim",
    "dataset": "configs/hypersim.yaml", "views": "views_n20_uniform_504",
    "backbone": "dejaview_k16", "segmentation": "sam3image-nyu40-t0.3",
    "vocabulary": "hypersim", "aggregated": ["dev", "val"],
}  # fmt: skip


def kind_of(rung: str) -> str:
    """Return the key of `SCORED` and the rules of a rung: its escalations share one."""
    return "overfit1" if rung.startswith("overfit1_e") else rung


def models(rung: str) -> list[str]:
    """Return the labels of the models trained in the rung's run folder, sorted."""
    run = RUNS / rung
    return sorted(
        p.stem.removeprefix(f"e69c-{rung}-") for p in run.glob(f"e69c-{rung}-*.yaml")
    )


def seeds_of(rung: str, arm: str, size: int) -> list[str]:
    """Return the labels of an arm's models of one size, one per seed."""
    return [m for m in models(rung) if parse(m)[:2] == (arm, size)]


def chosen_split() -> int:
    """Return the split rung 0 chose, from `split.yaml`."""
    return int(yaml.safe_load((RUNS / "split.yaml").read_text())["split"])


def write_config(rung: str) -> Path:
    """Write the cloud_eval config of a rung: pixels, references and every model."""
    key = kind_of(rung)
    splits = SPLITS if key == "rung0" else (chosen_split(),)
    variants = [{"name": "Déjà View", "kind": "pixels"}]
    variants += [
        {"name": f"density q{q}", "kind": "cloud", "file": f"e69c-density-q{q}"}
        for q in QUANTILES
    ]
    for k in splits:
        variants += [
            {
                "name": f"oracle-ovoxel-s{k}",
                "kind": "cloud",
                "file": f"e69c-oracle-ovoxel-s{k}",
            },
            {
                "name": f"dv-ovoxel-s{k}",
                "kind": "cloud",
                "file": f"e69c-dv-ovoxel-s{k}",
            },
        ]
    variants += [
        {"name": m, "kind": "cloud", "file": f"e69c-{rung}-{m}"} for m in models(rung)
    ]
    raw = RAW | {"variants": variants}
    if key == "overfit1a":  # the latent is one scene's memory: score the bedroom alone
        raw["dataset"] = str(bedroom_dataset())
    path = RUNS / f"{rung}_eval.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False))
    return path


def bedroom_dataset() -> Path:
    """Write the repo's split file with `dev` cut to the bedroom, for the latent's rung."""
    raw = yaml.safe_load(Path(RAW["dataset"]).read_text())
    raw["splits"]["dev"] = {BEDROOM.partition("_cam")[0]: "cam_00"}
    path = RUNS / "bedroom_hypersim.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False))
    return path


def best_lrs() -> dict[str, float]:
    """Return per arm the learning rate of the probe whose best step has the least error."""
    best: dict[str, tuple[float, float]] = {}
    for run in sorted(RUNS.glob("lr_probe_*")):
        for path in sorted(run.glob("*.yaml")):
            record = yaml.safe_load(path.read_text())
            arm = parse(path.stem.removeprefix(f"e69c-{run.name}-"))[0]
            err = min(e["train"]["chamfer_cells"] for e in record["curve"])
            if arm not in best or err < best[arm][0]:
                best[arm] = (err, record["config"]["lr"])
    return {arm: lr for arm, (_, lr) in best.items()}


def check_profile() -> bool:
    """Print each arm's step times and job estimates; return whether rung 2 passes."""
    profile = yaml.safe_load((RUNS / "profile.yaml").read_text())
    rungs = {
        r: load_rung(CONFIG, r).cfg.steps
        for r in ("lr_probe_1e-3", "overfit1a", "overfit1", "overfit3", "generalize")
    }
    ok = True
    for arm, t in profile.items():
        train = sum(t[k] for k in ("to_device", "grid", "forward_loss", "backward"))
        minutes = {r: steps * t["step_s"] / 60 for r, steps in rungs.items()}
        flag = "FAIL" if train >= 1 else "warn" if train > 0.5 else "ok"
        long = [r for r, m in minutes.items() if m > LIMIT_MIN]
        print(
            f"{arm} | {t['parameters'] / 1e6:.2f} M | train step {train:.2f} s {flag} | "
            f"step with validation {t['step_s']:.2f} s | peak {t['peak_gib']:.1f} GiB | "
            + " ".join(f"{r} {m:.0f} min" for r, m in minutes.items())
        )
        ok &= flag != "FAIL" and (not long or bool(os.environ.get("ALLOW_LONG")))
    return ok


def newest_scores(rung: str) -> dict:
    """Return the splits report of the newest `cloud_eval` run of the rung."""
    runs = sorted(RUNS.glob(f"{rung}_eval-*"))
    assert runs, f"no scores for {rung}: score {rung}_eval.yaml with cloud_eval first"
    return yaml.safe_load((runs[-1] / "run.yaml").read_text())["splits"]


def chamfers(report: dict, split: str, variants: list[str]) -> np.ndarray:
    """Return the Chamfer (cm) of the variants on every scene of a split, averaged."""
    scenes = report[split]["scenes"]
    return np.mean(
        [[scenes[s][v]["chamfer_cm"] for s in sorted(scenes)] for v in variants], axis=0
    )


def choose_split(report: dict) -> int | None:
    """Return the smallest split whose oracle is at most half of Deja View's on every dev scene."""
    dv = chamfers(report, "dev", ["Déjà View"])
    for k in SPLITS:
        if (chamfers(report, "dev", [f"oracle-ovoxel-s{k}"]) <= dv / 2).all():
            return k
    return None


def tuned_density(report: dict) -> tuple[float, float]:
    """Return the quantile with the least dev Chamfer of the density filter, and its val's."""
    best = min(
        QUANTILES, key=lambda q: chamfers(report, "dev", [f"density q{q}"]).mean()
    )
    return best, float(chamfers(report, "val", [f"density q{best}"]).mean())


def rung0(report: dict) -> dict[str, tuple[bool, float]]:
    """Judge the formulation: a split exists and Deja View costs nothing in it."""
    k = choose_split(report)
    for split in ("dev", "val"):
        dv = chamfers(report, split, ["Déjà View"])
        for j in SPLITS:
            oracle = chamfers(report, split, [f"oracle-ovoxel-s{j}"])
            dv_ovoxel = chamfers(report, split, [f"dv-ovoxel-s{j}"])
            print(
                f"rung0 | {split} | s{j} | Déjà View {dv.mean():.3f} | oracle {oracle.mean():.3f} "
                f"(scene ratios {np.round(oracle / dv, 2).tolist()}) | dv-ovoxel {dv_ovoxel.mean():.3f}"
            )
    if k is None:
        return {"split": (False, float("nan"))}
    (RUNS / "split.yaml").write_text(yaml.safe_dump({"split": k}))
    dv = chamfers(report, "dev", ["Déjà View"])
    gap = float((chamfers(report, "dev", [f"dv-ovoxel-s{k}"]) - dv).mean())
    return {f"split {k}": (abs(gap) <= BAND, gap)}


def children_of(rung: str, model: str, scene: str) -> dict[str, float]:
    """Return the child scores `run_model` recorded for a model on a scene."""
    record = yaml.safe_load((RUNS / rung / f"e69c-{rung}-{model}.yaml").read_text())
    return record["children"][scene]


def overfit1a(report: dict) -> dict[str, tuple[bool, float]]:
    """Per latent: passes, and the share of the bedroom's gap to the oracle it closes."""
    scene = report["dev"]["scenes"][BEDROOM]
    k = chosen_split()
    dv, oracle = (
        scene["Déjà View"]["chamfer_cm"],
        scene[f"oracle-ovoxel-s{k}"]["chamfer_cm"],
    )
    out = {}
    for model in models("overfit1a"):
        got = scene[model]["chamfer_cm"]
        kids = children_of("overfit1a", model, BEDROOM)
        print(f"overfit1a | {model} | children {kids}")
        ok = (
            got <= oracle + max(BAND, HEAD * (dv - oracle))
            and kids["precision"] >= CHILDREN
            and kids["recall"] >= CHILDREN
        )
        out[parse(model)[0]] = (ok, gap_closed(dv, oracle, got))
    return out


def overfit1(report: dict, rung: str) -> dict[str, tuple[bool, float]]:
    """Per arm: passes, and the share of the bedroom's gap to `oracle-ovoxel` it closes."""
    scene = report["dev"]["scenes"][BEDROOM]
    oracle = scene[f"oracle-ovoxel-s{chosen_split()}"]["chamfer_cm"]
    out = {}
    for model in models(rung):
        got = gap_closed(
            scene["Déjà View"]["chamfer_cm"], oracle, scene[model]["chamfer_cm"]
        )
        out[parse(model)[0]] = (got >= GAP, got)
    return out


def overfit3(report: dict) -> dict[str, tuple[bool, float]]:
    """Per arm: passes, and the dev-mean gain over Deja View (cm)."""
    dv = chamfers(report, "dev", ["Déjà View"])
    out = {}
    for model in models("overfit3"):
        gain = dv - chamfers(report, "dev", [model])
        out[parse(model)[0]] = (
            gain.mean() > BAND and (gain > BAND).sum() >= 2,
            float(gain.mean()),
        )
    return out


def generalize(report: dict) -> dict[str, tuple[bool, float]]:
    """Per arm: passes, and the val-mean gain over Deja View at the largest size (cm)."""
    dv = chamfers(report, "val", ["Déjà View"])
    q, density = tuned_density(report)
    print(
        f"generalize | density filter tuned on dev: q {q}, val {density:.3f} | Déjà View val {dv.mean():.3f}"
    )
    out = {}
    for arm in sorted({parse(m)[0] for m in models("generalize")}):
        sizes = sorted(
            {parse(m)[1] for m in models("generalize") if parse(m)[0] == arm}
        )
        small, large = (
            chamfers(report, "val", seeds_of("generalize", arm, n))
            for n in (sizes[0], sizes[-1])
        )
        out[arm] = (
            beats(dv, large) and large.mean() <= small.mean() + BAND,
            float((dv - large).mean()),
        )
    return out


def ablation(report: dict) -> dict[str, tuple[bool, float]]:
    """Judge the best arm against Deja View beyond its seeds' spread; print every cost."""
    dv = chamfers(report, "val", ["Déjà View"])
    arms = sorted({parse(m)[0] for m in models("ablation")})
    mean = {a: chamfers(report, "val", seeds_of("ablation", a, 40)) for a in arms}
    spread = {
        a: float(
            np.std(
                [
                    chamfers(report, "val", [m]).mean()
                    for m in seeds_of("ablation", a, 40)
                ],
                ddof=1,
            )
        )
        for a in arms
    }
    best = min(arms, key=lambda a: mean[a].mean())
    for a in arms:
        cost = float(mean[a].mean() - mean[best].mean())
        helps = cost > max(spread[a], spread[best])
        print(
            f"ablation | {a} | val {mean[a].mean():.3f} ± {spread[a]:.3f} | cost against {best} {cost:+.3f} | part helps: {helps}"
        )
    return {
        best: (beats(dv, mean[best], spread[best]), float((dv - mean[best]).mean()))
    }


RULES = {
    "overfit1a": overfit1a,
    "overfit1": overfit1,
    "overfit3": overfit3,
    "generalize": generalize,
    "ablation": ablation,
}


def main() -> None:
    """`config <rung>`, `profile`, `lr`, or `gate <rung>`; see the module docstring."""
    command, *rest = sys.argv[1:]
    rung = rest[0] if rest else ""
    if command == "config":
        print(write_config(rung), *SCORED[kind_of(rung)])
        return
    if command == "profile":
        sys.exit(0 if check_profile() else 1)
    if command == "lr":
        lrs = best_lrs()
        (RUNS / "lr.yaml").write_text(yaml.safe_dump(lrs))
        print(lrs)
        return
    assert command == "gate", command
    report = newest_scores(rung)
    rules = RULES | {"overfit1": functools.partial(overfit1, rung=rung)}
    results = rung0(report) if rung == "rung0" else rules[kind_of(rung)](report)
    ranked = sorted(results.items(), key=lambda kv: -kv[1][1])
    for arm, (ok, value) in ranked:
        print(f"{rung} | {arm} | {'PASS' if ok else 'fail'} | {value:+.3f}")
    passed = [arm for arm, (ok, _) in ranked if ok]
    (RUNS / f"{rung}.pass").write_text("".join(f"{a}\n" for a in passed))
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
