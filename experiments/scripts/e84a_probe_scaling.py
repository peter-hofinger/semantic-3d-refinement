"""E84a: where a GT point's R probes come from, and what a larger R costs.

Part 1 (R = 8, bedroom): the share of each probe rank that lies in the view of the GT point's
own pixel, the number of distinct views among a GT point's probes, and how often the rank-1
probe is the GT point's own pixel. The probes are the nearest predictions in 3D over all
20 views (one KD-tree), not the neighbours of a pixel.
Part 2: R = 8 ... 128, the loss's C term over chunks of GT points (index stored as int32,
cast per chunk), timed with 3 fwd+bwd passes, peak GPU memory, KD-tree time. The head's own
forward/backward and the A term are not included.

`uv run python experiments/scripts/e84a_probe_scaling.py --out experiments/results/e84a_probe_scaling.txt`
"""

import argparse
import time
from pathlib import Path

import torch

from src.refinement.keep_chamfer import KeepScene, Probes

CFG = {
    "root": "data/processed/hypersim",
    "views": "views_n20_uniform_504",
    "backbone": "dejaview_k16",
    "r": 8,
    "miss": "next",
}
CHUNK = 250_000  # GT points per chunk


def completeness_backward(
    s: torch.Tensor, index: torch.Tensor, dist: torch.Tensor
) -> None:
    """C of the loss over chunks of GT points; the gradient accumulates in s.grad."""
    n = len(s)
    for i in range(0, n, CHUNK):
        sg = s[index[i : i + CHUNK].long()]
        drop = torch.cumprod(1 - sg, 1)
        first = torch.cat([torch.ones_like(drop[:, :1]), drop[:, :-1]], 1)
        d = dist[i : i + CHUNK]
        c = ((d[:, :-1] * sg * first).sum(1) + d[:, -1] * drop[:, -1]).sum() / n
        c.backward()


def main() -> None:
    """Print and save both parts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    scene = KeepScene.load("ai_006_007_cam_00", CFG, "cuda")
    v, h, w = scene.valid.shape
    view = torch.arange(v, device="cuda")[:, None, None].expand(v, h, w)[scene.valid]
    index = scene.tensors[1]
    same = (view[index] == view[:, None]).float().mean(0)
    own = (index[:, 0] == torch.arange(len(view), device="cuda")).float().mean()
    views_per = (
        torch.stack([(view[index] == k).any(1) for k in range(v)], 1).sum(1).float()
    )
    lines = [
        f"bedroom ai_006_007, N = {len(view)}, R = 8",
        "share of probe rank r in the same view as the GT point's pixel: "
        + " ".join(f"{x:.3f}" for x in same.tolist()),
        f"rank-1 probe is the GT point's own pixel: {own.item():.3f}",
        (
            f"distinct views among the 8 probes: mean {views_per.mean().item():.2f}, "
            f"all 8 in one view {(views_per == 1).float().mean().item():.3f}, "
            f">= 3 views {(views_per >= 3).float().mean().item():.3f}"
        ),
        "",
    ]
    pred, truth = scene.pred, scene.truth
    n = len(pred)
    del scene, index, view
    for r in (8, 32, 64, 128):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        p = Probes.build(pred, truth, r, "next")
        kd = time.perf_counter() - t0
        index, dist = torch.from_numpy(p.index).cuda(), torch.from_numpy(p.dist).cuda()
        s = torch.rand(n, device="cuda").requires_grad_()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(3):
            s.grad = None
            completeness_backward(s, index, dist)
        torch.cuda.synchronize()
        lines.append(
            f"R = {r:3d}: KD probes {kd:4.1f} s, C fwd+bwd {(time.perf_counter() - t0) / 3:.3f} s, "
            f"peak GPU {torch.cuda.max_memory_allocated() / 1e9:.2f} GB, "
            f"probes {(index.numel() + dist.numel()) * 4 / 1e9:.2f} GB (int32 + float32)"
        )
        del index, dist, s, p
    args.out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
