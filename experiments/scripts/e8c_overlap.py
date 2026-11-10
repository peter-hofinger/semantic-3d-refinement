"""E8c: how far SAM 3's gate and a residual test of the same size cut the same matches.

Per scene, on the input depth (before refining): SAM 3's gate cuts k of E_mv's M
matches; the residual test cuts the k of largest |residual|. Printed: k / M, the share
of SAM 3's cut that the residual test also cuts (k / M expected for a random cut), the
share of E_mv's initial energy (sum of squared residuals) each cut carries, and the
share of each cut whose source pixel lies within 3 px of a SAM 3 mask boundary.

`uv run python experiments/scripts/e8c_overlap.py experiments/configs/e8c_residual.yaml
--splits dev val`
"""

import argparse
import sys
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

sys.path.insert(0, str(Path(__file__).parent))
from e8b_matches import BAND_PX, near_boundary


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
        "scene | cut share k/M | residual cut ∩ SAM 3 cut, share of k | E_mv energy"
        f" share: SAM 3 cut / residual cut | within {BAND_PX} px of a SAM 3 boundary:"
        " SAM 3 cut / residual cut"
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
            key = (m.dst * d0.numel() + m.src).cpu()
            sam3_cut = ~torch.isin(key, (kept.dst * d0.numel() + kept.src).cpu())
            sam3_cut = sam3_cut.to(device)
            k = int(sam3_cut.sum())
            r2 = residuals(d0, scale, scene, m).square()
            residual_cut = torch.zeros_like(sam3_cut)
            residual_cut[r2.argsort(descending=True, stable=True)[:k]] = True
            near = near_boundary(ids, BAND_PX).reshape(-1)[m.src]
            row = [
                k / len(m.src),
                (sam3_cut & residual_cut).sum().item() / k,
                (r2[sam3_cut].sum() / r2.sum()).item(),
                (r2[residual_cut].sum() / r2.sum()).item(),
                near[sam3_cut].float().mean().item(),
                near[residual_cut].float().mean().item(),
            ]
            rows.append(row)
            print(
                f"{name} | {row[0]:.3f} | {row[1]:.3f} | {row[2]:.3f} / {row[3]:.3f}"
                f" | {row[4]:.3f} / {row[5]:.3f}",
                flush=True,
            )
        mean = torch.tensor(rows).mean(0).tolist()
        print(
            f"{split} mean | {mean[0]:.3f} | {mean[1]:.3f} | {mean[2]:.3f} /"
            f" {mean[3]:.3f} | {mean[4]:.3f} / {mean[5]:.3f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
