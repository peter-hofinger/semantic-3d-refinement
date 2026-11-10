"""E8a: what each segment variant hands the refiner, before refining (step 1 of the ladder).

Per scene and variant: the share of pixels without a segment, the number of ids, the
mean consistency u over the pixels that have one (E_d's weight; 1 = every view that sees
the point agrees), and the share of E_mv's matches the instance gate keeps.

`uv run python experiments/scripts/e8a_inputs.py experiments/configs/e8a_segments.yaml
--splits dev`
"""

import argparse
from pathlib import Path

import torch
import yaml

from src.analysis.scene_eval import load_inputs, split_scenes, variant_segments
from src.refinement.multiview import (
    RefineConfig,
    consistency,
    find_matches,
    same_segment_matches,
)


def main() -> None:
    """Print one line per scene and segment variant."""
    parser = argparse.ArgumentParser()
    parser.add_argument("cfg", type=Path)
    parser.add_argument("--splits", nargs="+", default=["dev"])
    args = parser.parse_args()
    raw = yaml.safe_load(args.cfg.read_text(encoding="utf-8"))
    base = RefineConfig(**yaml.safe_load(Path(raw["refine"]).read_text())["refine"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for names in split_scenes(raw, args.splits).values():
        for name in names:
            inputs = load_inputs(raw, name, device)
            scene = inputs.scene
            scale = scene.depth.median()
            d0 = scale / scene.depth
            matches = find_matches(scene, d0, scale, base.neighbors)
            for variant in raw["variants"]:
                if variant["refiner"] != "joint":
                    continue
                segments = variant_segments(inputs, variant)
                ids = (
                    torch.full_like(scene.depth, -1, dtype=torch.long)
                    if segments is None
                    else segments.to(device)
                )
                u = consistency(scene, d0, scale, ids)
                kept = same_segment_matches(matches, scene, d0, scale, ids)
                labeled = ids >= 0
                mean_u = u[labeled].mean().item() if labeled.any() else float("nan")
                print(
                    f"{name} | {variant['name']:>9} | no segment"
                    f" {1 - labeled.float().mean().item():.3f} | ids"
                    f" {len(ids.unique()):4d} | mean u {mean_u:.3f} | E_mv kept"
                    f" {len(kept.src) / len(matches.src):.3f}",
                    flush=True,
                )


if __name__ == "__main__":
    main()
