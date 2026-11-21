"""E84m (issue #84): the best Chamfer of each trained head's ranking, over the kept share.

Déjà View's-shaped heads (`runs/<name>/head.pt`) of the overfit rung on the bedroom: the exact
Chamfer (fixed alignment, the loss's) of the top share q of the logit x for q = 5 % ... 100 %,
the best q, and the cloud of x >= 0 (s >= 0.5, c >= 2). The best q is chosen with the GT on the
training scene: an upper bound for a cut, not a method's choice. Run folders are the arguments.

`uv run python experiments/scripts/e84m_sweep.py runs/e84d_... runs/e84i_... --out experiments/results/e84m_sweep.txt`
"""

import argparse
import copy
from pathlib import Path

import numpy as np
import torch
import yaml

from experiments.scripts.e84m_oracle_order import CFG, SCENE
from src.refinement.keep_chamfer import (
    KeepHead,
    KeepNet,
    KeepResNet,
    KeepScene,
    exact_chamfer,
    standardize,
)

SHARES = [round(0.05 * k, 2) for k in range(1, 21)]


def logits(head: KeepNet, scene: KeepScene, chunk: int = 4) -> np.ndarray:
    """Return the head's x on the scene's valid pixels (N,)."""
    with torch.no_grad():
        x = torch.cat(
            [
                head.logits(scene.features[i : i + chunk].float())
                for i in range(0, len(scene.features), chunk)
            ]
        )
    return x[scene.valid].double().cpu().numpy()


def load_net(run: Path) -> tuple[KeepNet, dict]:
    """Return the run's trained network on the GPU and its config."""
    config = yaml.safe_load((run / "run.yaml").read_text())["config"]
    if "resnet" in config:
        net = KeepResNet(**config["resnet"])
    else:
        net = KeepHead(**config.get("head", {}))
    net = net.cuda()
    net.load_state_dict(torch.load(run / "head.pt"))
    return net, config


def sweep(x: np.ndarray, scene: KeepScene) -> dict[float, dict[str, float]]:
    """Return the exact Chamfer scores of the top share q of x, per q in SHARES."""
    order = np.argsort(-x, kind="stable")
    out = {}
    for q in SHARES:
        kept = np.zeros(len(x), dtype=bool)
        kept[order[: round(q * len(x))]] = True
        out[q] = exact_chamfer(kept, scene.probes, scene.pred, scene.truth)
    return out


def main() -> None:
    """Print and save one line per run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path, nargs="+")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    raw_scene = KeepScene.load(SCENE, CFG, "cuda")
    lines = [
        (
            "bedroom ai_006_007, exact Chamfer cm, fixed alignment; uncut 1.846; best over "
            "the top share q = 5 % ... 100 % of the head's logit (GT-chosen), and the "
            "x >= 0 cloud"
        ),
        "run | steps | best q | best Chamfer (A / C) | x >= 0: kept, Chamfer",
    ]
    for run in args.runs:
        net, config = load_net(run)
        scene = copy.copy(raw_scene)
        if config.get("standardize"):
            standardize([scene])
        x = logits(net, scene)
        scores = sweep(x, scene)
        q = min(scores, key=lambda k: scores[k]["chamfer_cm"])
        cut = exact_chamfer(x >= 0, scene.probes, scene.pred, scene.truth)
        lines.append(
            f"{run.name} | {config['steps']} | {q} | {scores[q]['chamfer_cm']:.3f} "
            f"({scores[q]['accuracy_cm']:.3f} / {scores[q]['completeness_cm']:.3f}) | "
            f"{cut['kept']:.2f}, {cut['chamfer_cm']:.3f}"
        )
    args.out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
