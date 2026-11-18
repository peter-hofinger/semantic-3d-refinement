"""E69a's gates: score the clouds of a rung and decide which arms climb to the next.

`config <rung>` writes `runs/e69a/<rung>_eval.yaml`: Deja View's pixels, the two oracles and
every model of the rung's run folder as variants, for `cloud_eval`; `gate <rung>` reads the
newest scores and prints each arm's verdict by the rules written in the LOG's plan, and
writes the arms that pass, best first, to `runs/e69a/<rung>.pass`, one per line (exit
status 1 if none does, which stops `e69a.sh`). All Chamfer numbers are cm.

- profile (rung 2): the train step (transfer, grid, forward with the loss, backward) of
  every arm takes under 1 s (the ladder; over 0.5 s is a warning), and no job (steps times
  a step with its validation crop, per rung) is over the repo's 30 minute limit, unless
  `ALLOW_LONG` is set; the profile is read from `runs/e69a/profile.yaml`.
- lr: the rate of each arm is the one of the three probe rungs (`lr_probe_*`, 300 steps on
  the bedroom) with the least fixed train error at its best step; written to `lr.yaml`.
- overfit1 (rung 3): the arm closes at least half of the bedroom's gap between Deja View
  and the oracle of its family (`oracle-seeded` for the seeded arms, else `oracle-lattice`).
- overfit3 (rung 4): the dev-mean gain over Deja View exceeds the tie band and at least two
  of the three scenes are won by more than the band.
- generalize (rung 5): at the largest train size the gain over Deja View on the 11 val
  scenes exceeds the larger of the band and twice the standard error of the per-scene
  differences, and the val Chamfer is no worse than at the smallest size by more than the
  band.
- semantics: the arm beats Deja View over its seeds by that margin and the spread of the seeds' val
  means; an arm with classes beats the same arm without them by that margin and by more
  than the spread of the seeds' val means, on each scene's Chamfer averaged over seeds.
"""

import os
import re
import sys
from pathlib import Path

import numpy as np
import yaml

from src.refinement.generator import load_rung

LIMIT_MIN = 30  # the repo asks before a job longer than this
BAND = 0.1  # cm: a difference below it is a tie (X69)
GAP = 0.5  # share of the gap to the oracle that rung 3 asks for
BEDROOM = "ai_006_007_cam_00"
RUNS = Path("runs/e69a")
SCORED = {"overfit1": ["dev"], "overfit3": ["dev"], "generalize": ["val", "dev"]}
SCORED["semantics"] = SCORED["generalize"]
CLASSES = ("sam3", "gt")
RAW = {
    "output_dir": "runs/e69a", "root": "data/processed/hypersim",
    "dataset": "configs/hypersim.yaml", "views": "views_n20_uniform_504",
    "backbone": "dejaview_k16", "segmentation": "sam3image-nyu40-t0.3",
    "vocabulary": "hypersim", "aggregated": ["dev", "val"],
}  # fmt: skip


def gap_closed(dv: float, oracle: float, arm: float) -> float:
    """Return the share of the gap from Deja View's Chamfer to the oracle's the arm closes."""
    assert dv > oracle, (dv, oracle)
    return (dv - arm) / (dv - oracle)


def paired_gain(dv: np.ndarray, arm: np.ndarray) -> tuple[float, float]:
    """Return the mean and standard error of the per-scene gain dv - arm (positive: better)."""
    assert dv.shape == arm.shape
    assert dv.ndim == 1
    assert len(dv) > 1, "a standard error needs two scenes"
    gain = dv - arm
    return float(gain.mean()), float(gain.std(ddof=1) / np.sqrt(len(gain)))


def beats(dv: np.ndarray, arm: np.ndarray, spread: float = 0.0) -> bool:
    """Return whether the gain exceeds the tie band, twice its standard error and spread."""
    mean, se = paired_gain(dv, arm)
    return mean > max(BAND, 2 * se, spread)


def parse(name: str) -> tuple[str, int, int]:
    """Return the arm, train size (0: all) and seed of a label like `seeded-n40-s1`."""
    rest, seed = name.rsplit("-s", 1)
    size = re.search(r"-n(\d+)$", rest)
    return (
        (rest[: size.start()] if size else rest),
        int(size[1] if size else 0),
        int(seed),
    )


def models(rung: str) -> list[str]:
    """Return the labels of the models trained in the rung's run folder, sorted."""
    run = RUNS / rung
    return sorted(
        p.stem.removeprefix(f"e69a-{rung}-") for p in run.glob(f"e69a-{rung}-*.yaml")
    )


def write_config(rung: str) -> Path:
    """Write the cloud_eval config of a rung: pixels, oracles and every model."""
    variants = [
        {"name": "Déjà View", "kind": "pixels"},
        {"name": "Déjà View, kept 0.75", "kind": "pixels", "keep": 0.75},
        {"name": "Déjà View, kept 0.5", "kind": "pixels", "keep": 0.5},
        {"name": "voxel mean, 192", "kind": "voxel_mean", "resolution": 192},
        {
            "name": "voxel mean, densest half",
            "kind": "voxel_mean",
            "resolution": 192,
            "keep": 0.5,
        },
        {"name": "oracle-lattice", "kind": "cloud", "file": "e69a-oracle-lattice"},
        {"name": "oracle-seeded", "kind": "cloud", "file": "e69a-oracle-seeded"},
    ]
    variants += [
        {"name": m, "kind": "cloud", "file": f"e69a-{rung}-{m}"} for m in models(rung)
    ]
    path = RUNS / f"{rung}_eval.yaml"
    path.write_text(yaml.safe_dump(RAW | {"variants": variants}, sort_keys=False))
    return path


def best_lrs() -> dict[str, float]:
    """Return per arm the learning rate of the probe whose best step has the least error."""
    best: dict[str, tuple[float, float]] = {}
    for run in sorted(RUNS.glob("lr_probe_*")):
        for path in sorted(run.glob("*.yaml")):
            record = yaml.safe_load(path.read_text())
            arm = parse(path.stem.removeprefix(f"e69a-{run.name}-"))[0]
            err = min(e["train"]["chamfer_cells"] for e in record["curve"])
            if arm not in best or err < best[arm][0]:
                best[arm] = (err, record["config"]["lr"])
    return {arm: lr for arm, (_, lr) in best.items()}


def check_profile() -> bool:
    """Print each arm's step times and job estimates; return whether rung 2 passes."""
    profile = yaml.safe_load((RUNS / "profile.yaml").read_text())
    rungs = {
        r: load_rung(Path("configs/e69a.yaml"), r).cfg.steps
        for r in SCORED | {"lr_probe_1e-3": 0}
    }
    rungs.pop("semantics")
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


def seeds_of(rung: str, arm: str, size: int) -> list[str]:
    """Return the labels of an arm's models of one size, one per seed."""
    return [m for m in models(rung) if parse(m)[:2] == (arm, size)]


def overfit1(report: dict) -> dict[str, tuple[bool, float]]:
    """Per arm: passes, and the share of the bedroom's gap to its oracle it closes."""
    scene = report["dev"]["scenes"][BEDROOM]
    out = {}
    for model in models("overfit1"):
        arm = parse(model)[0]
        oracle = "oracle-seeded" if arm.startswith("seeded") else "oracle-lattice"
        got = gap_closed(
            scene["Déjà View"]["chamfer_cm"],
            scene[oracle]["chamfer_cm"],
            scene[model]["chamfer_cm"],
        )
        out[arm] = (got >= GAP, got)
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
    out = {}
    for arm in sorted({parse(m)[0] for m in models("generalize")}):
        sizes = sorted(
            {parse(m)[1] for m in models("generalize") if parse(m)[0] == arm}
        )
        small, large = (
            chamfers(report, "val", seeds_of("generalize", arm, n))
            for n in (sizes[0], sizes[-1])
        )
        ok = beats(dv, large) and large.mean() <= small.mean() + BAND
        out[arm] = (ok, float((dv - large).mean()))
    return out


def semantics(report: dict) -> dict[str, tuple[bool, float]]:
    """Per arm: passes against Deja View; per class source: against the arm without."""
    arms = {parse(m)[0] for m in models("semantics")}
    out = {}
    for arm in sorted(
        a for a in arms if not a.endswith(tuple(f"-{c}" for c in CLASSES))
    ):
        dv = chamfers(report, "val", ["Déjà View"])
        none = chamfers(report, "val", seeds_of("semantics", arm, 0))
        spread = float(
            np.std(
                [
                    chamfers(report, "val", [m]).mean()
                    for m in seeds_of("semantics", arm, 0)
                ],
                ddof=1,
            )
        )
        out[arm] = (beats(dv, none, spread), float((dv - none).mean()))
        for source in CLASSES:
            with_classes = chamfers(
                report, "val", seeds_of("semantics", f"{arm}-{source}", 0)
            )
            out[f"{arm}-{source}"] = (
                beats(none, with_classes, spread),
                float((none - with_classes).mean()),
            )
    return out


RULES = {
    "overfit1": overfit1,
    "overfit3": overfit3,
    "generalize": generalize,
    "semantics": semantics,
}


def main() -> None:
    """`config <rung>`, `profile`, `lr` or `gate <rung>`; see the module docstring."""
    command, *rest = sys.argv[1:]
    rung = rest[0] if rest else ""
    if command == "config":
        print(" ".join(SCORED[rung]), write_config(rung))
        return
    if command == "profile":
        sys.exit(0 if check_profile() else 1)
    if command == "lr":
        lrs = best_lrs()
        (RUNS / "lr.yaml").write_text(yaml.safe_dump(lrs))
        print(lrs)
        return
    assert command == "gate", command
    results = RULES[rung](newest_scores(rung))
    for arm, (ok, value) in sorted(results.items(), key=lambda kv: -kv[1][1]):
        print(f"{rung} | {arm} | {'PASS' if ok else 'fail'} | {value:+.3f}")
    passed = [
        arm for arm, (ok, _) in sorted(results.items(), key=lambda kv: -kv[1][1]) if ok
    ]
    (RUNS / f"{rung}.pass").write_text("".join(f"{a}\n" for a in passed))
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
