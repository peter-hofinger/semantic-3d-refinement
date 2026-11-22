"""E84q (issue #84): the dev-trained ResNet keep head against uncut Déjà View and its confidence, per scene.

For each scene of the run the exact Chamfer (fixed alignment, the loss's) of the uncut cloud, of the
head's cut x >= 0, of Déjà View's confidence at the head's kept share (same share, no GT choice), and
of the best top share q of each ranking (GT-chosen: an upper bound, not a method's choice).
The scenes are the training scenes: a rung 4 diagnostic, not a score.

`uv run python experiments/scripts/e84q_dev_sweep.py runs/e84q_... --out experiments/results/e84q_dev_sweep.txt`
"""

import argparse
from functools import partial
from pathlib import Path

import numpy as np

from experiments.scripts.e84m_sweep import SHARES, load_net, logits, sweep
from src.geometry import Geometry
from src.refinement.keep_chamfer import KeepScene, exact_chamfer, standardize


def best(scores: dict[float, dict[str, float]]) -> tuple[float, dict[str, float]]:
    """Return the share with the lowest Chamfer and its scores."""
    q = min(scores, key=lambda k: scores[k]["chamfer_cm"])
    return q, scores[q]


def top(score: np.ndarray, share: float) -> np.ndarray:
    """Return the mask of the `share` highest entries of `score`."""
    kept = np.zeros(len(score), dtype=bool)
    kept[np.argsort(-score, kind="stable")[: round(share * len(score))]] = True
    return kept


def show(label: str, e: dict[str, float]) -> str:
    """Return one table line: Chamfer (A / C)."""
    return (
        f"  {label}: {e['chamfer_cm']:.3f} "
        f"({e['accuracy_cm']:.3f} / {e['completeness_cm']:.3f})"
    )


def main() -> None:
    """Print and save one block per scene."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    net, config = load_net(args.run)
    scenes = [KeepScene.load(name, config, "cuda") for name in config["scenes"]]
    assert config.get("standardize")
    standardize(scenes)
    lines = [
        (
            f"{args.run.name}: exact Chamfer cm (A / C), fixed alignment, training "
            f"scenes; q = kept share, GT-chosen where 'best'; shares {SHARES[0]}...{SHARES[-1]}"
        )
    ]
    for scene in scenes:
        x = logits(net, scene)
        folder = Path(config["root"]) / scene.name
        conf = Geometry.load(
            folder / f"{config['backbone']}_{config['views']}.npz"
        ).confidence[scene.valid.cpu().numpy()]
        assert conf.shape == x.shape
        n = len(x)
        scored = partial(
            exact_chamfer, probes=scene.probes, pred=scene.pred, truth=scene.truth
        )

        cut = x >= 0
        share = float(cut.mean())
        q_head, head = best(sweep(x, scene))
        q_conf, dv = best(sweep(conf, scene))
        lines += [
            scene.name,
            show("uncut", scored(np.ones(n, dtype=bool))),
            show(f"head x >= 0 (kept {share:.2f})", scored(cut)),
            show(
                f"Déjà View confidence, same share {share:.2f}",
                scored(top(conf, share)),
            ),
            show(f"head, best q {q_head}", head),
            show(f"Déjà View confidence, best q {q_conf}", dv),
        ]
    args.out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
