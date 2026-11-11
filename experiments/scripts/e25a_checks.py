"""E25a checks, added after the first read of the results (not in the plan).

Are the class differences stable?

1. Split half: the 14 scenes split at random into two halves HALVES times; per class with
   >= 2 scenes in each half, the mean log excess of each half; their correlation over
   classes. Stable class effects correlate, noise does not.
2. Without glass: S and its permutation null with window and mirror left out (their GT is
   the glass surface, which E35 already set apart).
3. Scale: the scenes' mean per-pixel error, next to the classes' excess error in cm.

uv run python experiments/scripts/e25a_checks.py runs/e25a_hypersim
"""

import argparse
from pathlib import Path

import numpy as np
import yaml
from e25a_class_split import PERMUTATIONS, log_excess, permuted_spread, spread

HALVES = 1000
GLASS = ("window", "mirror")


def main() -> None:
    """Print the three checks of the module docstring."""
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    run = yaml.safe_load((args.run / "e25a.yaml").read_text(encoding="utf-8"))
    recs = [run["scenes"][n] for n in run["groups"]["all"]]
    rng = np.random.default_rng(0)

    corr, classes = [], []
    for _ in range(HALVES):
        order = rng.permutation(len(recs))
        a = log_excess([recs[i] for i in order[: len(recs) // 2]], "gt")
        b = log_excess([recs[i] for i in order[len(recs) // 2 :]], "gt")
        shared = [c for c in a if c in b and min(len(a[c]), len(b[c])) >= 2]
        if len(shared) >= 3:
            x = [np.mean(a[c]) for c in shared]
            y = [np.mean(b[c]) for c in shared]
            corr.append(np.corrcoef(x, y)[0, 1])
            classes.append(len(shared))
    corr = np.array(corr)
    print(
        f"split half, GT classes: correlation of per-class mean log excess between halves over {len(corr)} splits "
        f"({np.median(classes):.0f} classes median): median {np.median(corr):.2f}, 5-95% {np.quantile(corr, 0.05):.2f} to "
        f"{np.quantile(corr, 0.95):.2f}, share > 0 {np.mean(corr > 0):.2f}"
    )
    for leave in ((), GLASS):
        sub = [
            {"gt": {c: v for c, v in r["gt"].items() if c not in leave}} for r in recs
        ]
        insts = []
        for r in recs:
            keep = [i for i, c in enumerate(r["instances"]["class"]) if c not in leave]
            insts.append({k: [v[i] for i in keep] for k, v in r["instances"].items()})
        s = spread(log_excess(sub, "gt"))
        null = np.array([permuted_spread(insts, rng) for _ in range(PERMUTATIONS)])
        p = (1 + (null >= s).sum()) / (1 + PERMUTATIONS)
        name = "all classes" if not leave else f"without {', '.join(leave)}"
        print(
            f"S, {name}: {s:.3f}; null median {np.median(null):.3f}, 95th percentile {np.quantile(null, 0.95):.3f}; p = {p:.3f}"
        )
    errors = [r["error_cm"] for r in recs]
    print(
        f"scenes' mean per-pixel error: mean {np.mean(errors):.2f} cm, median {np.median(errors):.2f} cm"
    )


if __name__ == "__main__":
    main()
