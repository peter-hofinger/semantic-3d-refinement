"""E8b: which of E_mv's matches SAM 3's gate cuts, against a random cut of the same size.

Per scene, on the input depth (before refining): the share of matches the gate cuts,
the share of E_mv's initial energy (sum of squared residuals) they carry, the median
|residual| of cut and kept matches, and the share of cut and of all matches whose source
pixel lies within 3 px of a SAM 3 mask boundary. A random cut removes, in expectation,
the same share of energy as of matches and the same boundary share as all matches.

`uv run python experiments/scripts/e8b_matches.py experiments/configs/e8b_dropout.yaml
--splits dev val`
"""

import argparse
from pathlib import Path

import torch
import yaml

from src.analysis.scene_eval import load_inputs, split_scenes
from src.refinement.multiview import (
    RefineConfig,
    find_matches,
    residuals,
    same_segment_matches,
)

BAND_PX = 3


def near_boundary(ids: torch.Tensor, band: int) -> torch.Tensor:
    """(V, H, W) bool: within band px of a pixel whose 4-neighbour has another id."""
    edge = torch.zeros_like(ids, dtype=torch.bool)
    edge[:, :, 1:] |= ids[:, :, 1:] != ids[:, :, :-1]
    edge[:, :, :-1] |= ids[:, :, 1:] != ids[:, :, :-1]
    edge[:, 1:] |= ids[:, 1:] != ids[:, :-1]
    edge[:, :-1] |= ids[:, 1:] != ids[:, :-1]
    pooled = torch.nn.functional.max_pool2d(
        edge[:, None].float(), 2 * band + 1, 1, band
    )
    return pooled[:, 0] > 0


def main() -> None:
    """Print one line per scene and the mean over each split."""
    parser = argparse.ArgumentParser()
    parser.add_argument("cfg", type=Path)
    parser.add_argument("--splits", nargs="+", default=["dev"])
    args = parser.parse_args()
    raw = yaml.safe_load(args.cfg.read_text(encoding="utf-8"))
    base = RefineConfig(**yaml.safe_load(Path(raw["refine"]).read_text())["refine"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        "scene | cut share of matches | cut share of E_mv energy | median |r| cut /"
        f" kept | share within {BAND_PX} px of a SAM 3 boundary: cut / all"
    )
    for split, names in split_scenes(raw, args.splits).items():
        rows = []
        for name in names:
            inputs = load_inputs(raw, name, device)
            scene = inputs.scene
            scale = scene.depth.median()
            d0 = scale / scene.depth
            m = find_matches(scene, d0, scale, base.neighbors)
            ids = torch.from_numpy(inputs.labels.instance_ids).long().to(device)
            kept = same_segment_matches(m, scene, d0, scale, ids)
            r = residuals(d0, scale, scene, m).abs()
            key = (m.dst * d0.numel() + m.src).cpu()
            is_kept = torch.isin(key, (kept.dst * d0.numel() + kept.src).cpu())
            is_kept = is_kept.to(r.device)
            near = near_boundary(ids, BAND_PX).reshape(-1)[m.src]
            row = [
                1 - is_kept.float().mean().item(),
                (r[~is_kept].square().sum() / r.square().sum()).item(),
                r[~is_kept].median().item(),
                r[is_kept].median().item(),
                near[~is_kept].float().mean().item(),
                near.float().mean().item(),
            ]
            rows.append(row)
            print(
                f"{name} | {row[0]:.3f} | {row[1]:.3f} | {row[2]:.4f} / {row[3]:.4f}"
                f" | {row[4]:.3f} / {row[5]:.3f}",
                flush=True,
            )
        mean = torch.tensor(rows).mean(0).tolist()
        print(
            f"{split} mean | {mean[0]:.3f} | {mean[1]:.3f} | {mean[2]:.4f} /"
            f" {mean[3]:.4f} | {mean[4]:.3f} / {mean[5]:.3f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
