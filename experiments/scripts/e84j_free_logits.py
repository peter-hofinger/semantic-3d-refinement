"""E84j (issue #84): Déjà View's ConfLoss on free per-pixel x, no head.

The head-free end of E84i: every valid pixel's x is a parameter, started at Déjà View's own x and
trained by Adam on the loss of E84i (a = the point's distance to the nearest GT point over the
scene scale). The loss is separable, so each x converges to log(alpha / (gamma a) - 1) and the
cut by x is the cut by -a; this logs the loss and the exact Chamfer of the top shares of x.

`uv run python experiments/scripts/e84j_free_logits.py --cfg configs/e84j_free_bedroom.yaml`
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from src.analysis.scene_eval import git_hash
from src.backbone.dejaview import DejaViewBackbone
from src.refinement.keep_chamfer import (
    KeepHead,
    KeepScene,
    confidence_loss,
    exact_chamfer,
)


def main() -> None:
    """Train the free x on the config's scene and write the run folder."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cfg", type=Path, required=True)
    args = parser.parse_args()
    raw = yaml.safe_load(args.cfg.read_text())
    torch.manual_seed(raw["seed"])
    out = Path("runs") / f"{args.cfg.stem}-{time.strftime('%Y%m%d-%H%M%S')}"
    out.mkdir(parents=True)
    scene = KeepScene.load(raw["scene"], raw, "cuda")
    steps = int(raw["backbone"].removeprefix("dejaview_k"))
    head = KeepHead.from_deja_view(DejaViewBackbone(steps=steps).output_block()).cuda()
    with torch.no_grad():
        x0 = torch.cat(
            [
                head.logits(scene.features[i : i + raw["chunk"]].float())
                for i in range(0, len(scene.features), raw["chunk"])
            ]
        )[scene.valid]
    x = x0.clone().requires_grad_()
    a = scene.tensors[0] / (100.0 * scene.scale)
    star = torch.clamp(raw["alpha"] / (raw["gamma"] * a.clamp_min(1e-9)), min=1.0)
    floor = confidence_loss(
        torch.log((star - 1).clamp_min(1e-30)), a, raw["alpha"], raw["gamma"], None
    ).item()
    optimizer = torch.optim.Adam([x], lr=raw["lr"])
    history = []
    for step in range(raw["steps"] + 1):
        optimizer.zero_grad()
        loss = confidence_loss(x, a, raw["alpha"], raw["gamma"], None)
        loss.backward()
        if step % raw["eval_every"] == 0:
            order = torch.argsort(x.detach(), descending=True).cpu().numpy()
            row = {"step": step, "loss": loss.item()}
            for share in raw["shares"]:
                kept = np.zeros(len(order), dtype=bool)
                kept[order[: round(share * len(order))]] = True
                row[f"top{share}"] = exact_chamfer(
                    kept, scene.probes, scene.pred, scene.truth
                )
            history.append(row)
            print(
                f"step {step}: loss {loss.item():.4f} (floor {floor:.4f}) | "
                + " | ".join(
                    f"top{s} {row[f'top{s}']['chamfer_cm']:.3f} "
                    f"(A {row[f'top{s}']['accuracy_cm']:.3f} "
                    f"C {row[f'top{s}']['completeness_cm']:.3f})"
                    for s in raw["shares"]
                )
            )
        optimizer.step()
    (out / "run.yaml").write_text(
        yaml.safe_dump(
            {"config": raw, "git": git_hash(), "floor": floor, "history": history},
            sort_keys=False,
        )
    )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
