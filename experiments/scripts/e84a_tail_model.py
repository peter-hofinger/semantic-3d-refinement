"""E84a diagnostic: a Poisson-tail charge for a GT point whose R probes are all dropped.

The loss charges such a point d_(R+1), a lower bound (E84a), or a scene constant, an upper
bound (E84b). Here the nearest kept prediction beyond the probes is modelled as the nearest
point of a planar Poisson process thinned by the local keep rate: its distance is Rayleigh
(the radial marginal of an isotropic 2D Gaussian) with survival exp(-lambda pi (x^2 - d_R^2))
for x >= d_R and lambda = s_bar R / (pi d_R^2) (R points lie within d_R). With
kappa = sum of s over the R probes the expected distance is
d_R (1 + sqrt(pi) erfcx(sqrt(kappa)) / (2 sqrt(kappa))), capped at the scene's upper bound.
Compared with the exact completeness on uniform keep probabilities and on E84a's saved head
snapshots; nothing is trained.

`uv run python experiments/scripts/e84a_tail_model.py --run runs/<e84a run> --out experiments/results/e84a_tail_model.txt`
"""

import argparse
import math
from pathlib import Path

import numpy as np
import torch

from src.refinement.keep_chamfer import KeepScene, exact_chamfer

R = 8
CFG = {
    "root": "data/processed/hypersim",
    "views": "views_n20_uniform_504",
    "backbone": "dejaview_k16",
    "r": R,
    "miss": "next",
}
SCENE = "ai_006_007_cam_00"


def completeness(s: torch.Tensor, scene: KeepScene, tail: str, upper: float) -> float:
    """Return the loss's C (cm) of probabilities s, the miss charged as `next` or `tail`."""
    _, index, dist = scene.tensors
    sg = s[index]
    drop = torch.cumprod(1 - sg, 1)
    first = torch.cat([torch.ones_like(drop[:, :1]), drop[:, :-1]], 1)
    if tail == "next":
        miss = dist[:, -1]
    else:
        kappa = sg.sum(1).clamp_min(1e-12)
        factor = 1 + math.sqrt(math.pi) * torch.special.erfcx(kappa.sqrt()) / (
            2 * kappa.sqrt()
        )
        miss = (dist[:, R - 1] * factor).clamp(max=upper)
    return ((dist[:, :-1] * sg * first).sum(1) + miss * drop[:, -1]).mean().item()


def main() -> None:
    """Print and save the comparison."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    scene = KeepScene.load(SCENE, CFG, "cuda")
    n = len(scene.pred)
    centre = scene.truth.mean(0)
    upper = 100 * (
        np.linalg.norm(scene.truth - centre, axis=1).max()
        + np.linalg.norm(scene.pred - centre, axis=1).max()
    )
    lines = [
        f"bedroom {SCENE}, R = {R}, completeness in cm; scene upper bound {upper:.0f} cm",
        "uniform keep probability c, exact = mean of 3 Bernoulli draws",
    ]

    def row(label: str, exact: float, s: torch.Tensor) -> str:
        return (
            f"{label} exact {exact:7.3f} | d_(R+1) charge "
            f"{completeness(s, scene, 'next', upper):7.3f} | Poisson tail "
            f"{completeness(s, scene, 'tail', upper):7.3f}"
        )

    for c in (0.5, 0.2, 0.1, 0.05, 0.02, 0.01):
        rng = np.random.default_rng(0)
        exact = np.mean(
            [
                exact_chamfer(rng.random(n) < c, scene.probes, scene.pred, scene.truth)[
                    "completeness_cm"
                ]
                for _ in range(3)
            ]
        )
        lines.append(row(f"c = {c:4.2f}:", exact, torch.full((n,), c, device="cuda")))
    lines.append("E84a head snapshots, one Bernoulli draw (seed = step)")
    for step in (50, 100, 200):
        file = args.run / f"probabilities_{SCENE}_step{step}.npy"
        s = torch.from_numpy(np.load(file).astype(np.float32)).cuda()[scene.valid]
        generator = torch.Generator("cuda").manual_seed(step)
        draw = torch.bernoulli(s, generator=generator).bool().cpu().numpy()
        exact = exact_chamfer(draw, scene.probes, scene.pred, scene.truth)
        lines.append(
            row(
                f"step {step} (mean s {s.mean().item():.3f}):",
                exact["completeness_cm"],
                s,
            )
        )
    args.out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
