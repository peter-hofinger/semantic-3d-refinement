"""E84n (issue #84): where the mimic head stops fitting its target, and why.

The head of E84m (Déjà View's block, random init) trained by `mimic_loss` on 1 000 random pixels,
100 000, all of one view and the whole scene. Reports the excess KL(v || s) = BCE - H(v) over the
trained pixels (0 = fitted; BCE's floor is H(v)'s mean). Also the KL of v to its own k x k local
mean for several k: what a predictor that is smooth at that scale cannot do better than.
`standardize: true` first gives every cached feature channel zero mean and unit variance over the scene.

`uv run python experiments/scripts/e84n_mimic_ladder.py --cfg configs/e84n_mimic_ladder.yaml`
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import yaml
from scipy.stats import spearmanr

from experiments.scripts.e84_dejaview_smoothed_optimum import masked_mean
from src.refinement.keep_chamfer import (
    KeepHead,
    KeepScene,
    logit_backward,
    mimic_loss,
    standardize,
)


def entropy(v: np.ndarray) -> float:
    """Return the mean binary entropy of the soft labels v (the cross-entropy's floor)."""
    c = np.clip(v.astype(np.float64), 1e-9, 1 - 1e-9)
    return float(np.mean(-c * np.log(c) - (1 - c) * np.log(1 - c)))


def cross_entropy(v: np.ndarray, s: np.ndarray) -> float:
    """Return the mean cross-entropy of probabilities s against the soft labels v."""
    c = np.clip(s.astype(np.float64), 1e-9, 1 - 1e-9)
    v = v.astype(np.float64)
    return float(np.mean(-v * np.log(c) - (1 - v) * np.log(1 - c)))


def main() -> None:
    """Train per size, print the table."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cfg", type=Path, required=True)
    raw = yaml.safe_load(parser.parse_args().cfg.read_text())
    scene = KeepScene.load(raw["scene"], raw, "cuda")
    if raw.get("standardize"):
        standardize([scene])
    folder = Path(raw["root"]) / raw["scene"]
    v = np.load(folder / raw["target"])
    target = torch.from_numpy(v).cuda()
    valid = scene.valid.cpu().numpy()
    view_of = np.nonzero(valid)[0]
    rng = np.random.default_rng(raw["seed"])
    print(f"scene: {len(v)} pixels, H(v) {entropy(v):.4f}")
    full = np.full(valid.shape, np.nan)
    full[valid] = v
    for k in raw["smoothing"]:
        local = masked_mean(np.where(valid, full, 0.0), valid, k)[valid] if k > 1 else v
        print(
            f"KL(v || {k}x{k} local mean of v) = "
            f"{cross_entropy(v, np.clip(local, 0, 1)) - entropy(v):.4f}"
        )
    for size in raw["sizes"]:
        if size == "scene":
            idx = np.arange(len(v))
        elif isinstance(size, str):
            idx = np.flatnonzero(view_of == int(size.removeprefix("view")))
        else:
            idx = np.sort(rng.choice(len(v), size, replace=False))
        torch.manual_seed(raw["seed"])
        head = KeepHead().cuda()
        optimizer = torch.optim.Adam(head.parameters(), lr=raw["lr"])
        index = torch.from_numpy(idx).cuda()
        floor = entropy(v[idx])
        for step in range(raw["steps"] + 1):
            optimizer.zero_grad()
            loss, x = logit_backward(
                head,
                scene,
                raw["chunk"],
                1.0,
                lambda x, i=index: mimic_loss(x[i], target[i]),
            )
            if step % raw["eval_every"] == 0:
                s = torch.sigmoid(x[index]).cpu().numpy()
                rho = spearmanr(x[index].cpu().numpy(), v[idx])[0]
                print(
                    f"{size}: {len(idx)} px, step {step}: excess {loss - floor:.4f} "
                    f"(BCE {loss:.4f}, floor {floor:.4f}), Spearman {rho:.3f}, "
                    f"mean s {s.mean():.3f}"
                )
            optimizer.step()


if __name__ == "__main__":
    main()
