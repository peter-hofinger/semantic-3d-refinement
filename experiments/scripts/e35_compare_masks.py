"""E35: how far two SAM 3 mask sets of the same scenes differ, before any association.

    uv run python experiments/scripts/e35_compare_masks.py <tag a> <tag b> <scene folder>...

A tag names `sam3image-nyu40-t0.3-<tag>_<views>.npz`; `cache` is the eval's own file.
Per scene: identical pixels (same view-local id, ids compared after mapping each mask
of a to the mask of b it shares most pixels with), masks per set, and the largest
score difference between masks matched that way.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import yaml

from src.segmentation.base import Segmentation, count_pairs

EVAL = yaml.safe_load(Path("configs/eval_hypersim.yaml").read_text())


def load(folder: Path, tag: str) -> Segmentation:
    """Load the scene's mask set of that tag."""
    name = EVAL["segmentation"] + ("" if tag == "cache" else f"-{tag}")
    return Segmentation.load(folder / f"{name}_{EVAL['views']}.npz")


def main() -> None:
    """Print one line per scene."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("a")
    parser.add_argument("b")
    parser.add_argument("scenes", type=Path, nargs="+")
    args = parser.parse_args()
    print(
        "scene | masks a | masks b | bit-identical ids | pixels with the matched mask | max score diff"
    )
    print("---|---|---|---|---|---")
    for folder in args.scenes:
        a, b = load(folder, args.a), load(folder, args.b)
        ia, ib = a.instance_ids.ravel(), b.instance_ids.ravel()
        src, dst, n = count_pairs(ia, ib)
        best: dict[int, tuple[int, int]] = {}
        for s, d, c in zip(src.tolist(), dst.tolist(), n.tolist(), strict=True):
            if s >= 0 and c > best.get(s, (0, -2))[0]:
                best[s] = (c, d)
        lookup = {s: d for s, (_, d) in best.items()}
        keys = np.array(sorted(lookup), dtype=np.int64)
        vals = np.array([lookup[k] for k in keys.tolist()], dtype=np.int64)
        mapped = np.full(ia.shape, -1, dtype=np.int64)
        has = ia >= 0
        mapped[has] = vals[np.searchsorted(keys, ia[has])]
        agree = float(np.mean(mapped == ib))
        diffs = [
            abs(a.scores[s] - b.scores[d])
            for s, d in lookup.items()
            if d >= 0 and s in a.scores and d in b.scores
        ]
        print(
            f"{folder.name} | {len(a.classes)} | {len(b.classes)} | "
            f"{bool(np.array_equal(a.instance_ids, b.instance_ids))} | {agree:.5f} | "
            f"{max(diffs, default=0.0):.4f}"
        )


if __name__ == "__main__":
    main()
