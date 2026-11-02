"""E15: are the low-score masks kept by the multi-view filter real, the dropped ones false.

Masks with score in (0.3, 0.5] of a threshold-0.3 run, split into kept (linked to a
mask in another view) and dropped. A mask counts as a match if it has IoU >= 0.5
with a GT instance in its view (as in the scorer, on the per-view masks).

uv run python experiments/scripts/low_score_masks.py <scene> <per-view stem> <assoc tag>
"""

import sys
from collections import Counter
from pathlib import Path

import numpy as np

from src.evaluation.instances import match_view
from src.loaders.scannetpp.gt import GroundTruth
from src.segmentation import Segmentation

scene, stem, tag = sys.argv[1:4]
d = Path("data/processed/scannetpp") / scene
gt = GroundTruth.load(d / "gt_views_n20_uniform_504.npz")
seg = Segmentation.load(d / f"{stem}.npz")
linked = Segmentation.load(d / f"{stem}_assoc-{tag}.npz")

matched = set()
for i in range(len(seg.instance_ids)):
    matched |= {p for _, p, _ in match_view(gt.instance_ids[i], seg.instance_ids[i])}

counts = Counter()
classes = {"kept": Counter(), "dropped": Counter()}
for m, s in seg.scores.items():
    if s > 0.5:
        continue
    view = m // 1000
    kept = bool(np.any(linked.instance_ids[view][seg.instance_ids[view] == m] >= 0))
    k = "kept" if kept else "dropped"
    counts[k, m in matched] += 1
    classes[k][seg.classes[m]] += 1

print(f"{scene} {stem} {tag}: masks with score <= 0.5")
for k in ("kept", "dropped"):
    n = counts[k, True] + counts[k, False]
    print(
        f"  {k:8s} {n:4d}, matching a GT instance (IoU >= 0.5): {counts[k, True]:4d}"
        f" ({counts[k, True] / max(n, 1):.2f})"
    )
    print(
        "    top classes:", ", ".join(f"{c} {n}" for c, n in classes[k].most_common(8))
    )
