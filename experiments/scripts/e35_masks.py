"""E35: SAM 3 masks of the eval protocol under a new name, to score code versions apart.

    uv run python experiments/scripts/e35_masks.py <tag> <scene folder>...

Runs the protocol's SAM 3 (nyu40 prompts, threshold 0.3) of the checked-out code on
each scene's views and writes `sam3image-nyu40-t0.3-<tag>_<views>.npz` next to them; the
eval's own cache is not touched. Prints the time per scene (model loaded first).
"""

from __future__ import annotations

import argparse
import subprocess
import time
from pathlib import Path

import torch
import yaml

from src.loaders import Views
from src.segmentation.sam3 import Sam3ImageSegmenter, load_prompts

EVAL = yaml.safe_load(Path("configs/eval_hypersim.yaml").read_text())


def main() -> None:
    """Segment every scene with the protocol's SAM 3 and save under the tag."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tag")
    parser.add_argument("scenes", type=Path, nargs="+")
    args = parser.parse_args()
    commit = subprocess.run(
        ["git", "describe", "--always", "--dirty"],  # noqa: S607
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    synonyms = load_prompts([Path("configs/prompts/nyu40.txt")])
    sam3 = Sam3ImageSegmenter(prompts=tuple(synonyms), synonyms=synonyms, threshold=0.3)
    sam3._load()  # noqa: SLF001
    print(f"git {commit}, {torch.cuda.get_device_name()}")
    for folder in args.scenes:
        path = folder / f"{EVAL['views']}.npz"
        views = Views.load(path)
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        seg = sam3.segment(views)
        seconds = time.perf_counter() - t0
        out = path.with_name(f"{EVAL['segmentation']}-{args.tag}_{path.stem}.npz")
        seg.save(out)
        print(
            f"{folder.name}: {len(seg.classes)} masks, {seconds:.1f} s, peak GPU "
            f"{torch.cuda.max_memory_allocated() / 1e9:.2f} GB -> {out.name}"
        )


if __name__ == "__main__":
    main()
