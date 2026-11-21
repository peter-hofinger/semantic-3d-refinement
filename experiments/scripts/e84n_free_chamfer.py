"""E84n (issue #84): the expected-Chamfer loss on free per-pixel logits, no head.

The loss of E84d (Poisson-tail charge) minimized over every pixel's x directly: whether the loss
itself, away from any head, leads to a good cloud. Logs the loss, one Bernoulli(s) draw's exact
Chamfer and the best exact Chamfer of a top share of x (GT-chosen).

`uv run python experiments/scripts/e84n_free_chamfer.py --cfg configs/e84n_free_chamfer.yaml`
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import yaml

from experiments.scripts.e84m_sweep import sweep
from src.refinement.keep_chamfer import KeepScene, exact_chamfer, expected_chamfer


def main() -> None:
    """Optimize x and print the curve."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cfg", type=Path, required=True)
    raw = yaml.safe_load(parser.parse_args().cfg.read_text())
    torch.manual_seed(raw["seed"])
    scene = KeepScene.load(raw["scene"], raw, "cuda")
    x = torch.full((len(scene.pred),), raw["init"], device="cuda", requires_grad=True)
    optimizer = torch.optim.Adam([x], lr=raw["lr"])
    generator = torch.Generator("cuda").manual_seed(raw["seed"])
    for step in range(raw["steps"] + 1):
        optimizer.zero_grad()
        s = torch.sigmoid(x)
        loss, acc, comp = expected_chamfer(
            s, *scene.tensors, raw["eps"], tail=raw["tail"]
        )
        loss.backward()
        if step % raw["eval_every"] == 0:
            xn = x.detach().cpu().numpy().astype(np.float64)
            scores = sweep(xn, scene)
            q = min(scores, key=lambda k: scores[k]["chamfer_cm"])
            draw = torch.bernoulli(s.detach(), generator=generator).bool().cpu().numpy()
            bern = exact_chamfer(draw, scene.probes, scene.pred, scene.truth)
            print(
                f"step {step}: loss {loss.item():.4f} (A {acc.item():.3f} C {comp.item():.3f}) "
                f"mean s {s.mean().item():.3f}, undecided {((s > 0.05) & (s < 0.95)).float().mean().item():.2f} | "
                f"Bernoulli {bern['chamfer_cm']:.3f} | best top share {q}: "
                f"{scores[q]['chamfer_cm']:.3f} (A {scores[q]['accuracy_cm']:.3f} "
                f"C {scores[q]['completeness_cm']:.3f})",
                flush=True,
            )
        optimizer.step()


if __name__ == "__main__":
    main()
