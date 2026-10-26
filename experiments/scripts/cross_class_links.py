"""E13: every cross-class link the association made, checked against GT.

A link joins two per-view masks of different classes (overlap >= threshold both
ways) that end up in one object. It is right if both masks mostly lie on the same
GT object (> 50% of their labeled pixels), wrong if on different ones.

uv run python experiments/scripts/cross_class_links.py <scene> <per-view stem> <threshold>
"""

import sys
from collections import Counter
from pathlib import Path

import numpy as np

from src.evaluation.instances import overlap_counts
from src.geometry import Geometry
from src.loaders.scannetpp.gt import GroundTruth
from src.segmentation import Segmentation
from src.segmentation.associate import Associator

scene, stem, threshold = sys.argv[1], sys.argv[2], float(sys.argv[3])
d = Path("data/processed/scannetpp") / scene
geom = "dejaview_k16_views_n20_uniform_504"
gt = GroundTruth.load(d / "gt_views_n20_uniform_504.npz")
seg = Segmentation.load(d / f"{stem}.npz")
linked = Segmentation.load(d / f"{stem}_assoc-xc{threshold:g}-{geom}.npz")
overlap = Associator().overlaps(seg, Geometry.load(d / f"{geom}.npz"))

on, obj = {}, {}  # per-view mask -> dominant GT object / linked object id
for i in range(len(seg.instance_ids)):
    g_ids, p_ids, inter = overlap_counts(gt.instance_ids[i], seg.instance_ids[i])
    lab = inter[g_ids >= 0]
    for col, p in enumerate(p_ids):
        if p >= 0:
            obj[int(p)] = int(np.bincount(
                linked.instance_ids[i][seg.instance_ids[i] == p] + 1
            ).argmax()) - 1  # fmt: skip
            if lab[:, col].max() > lab[:, col].sum() / 2:
                on[int(p)] = int(g_ids[g_ids >= 0][lab[:, col].argmax()])

verdict = Counter()
print(f"{scene} {stem} cross-class {threshold:g}: links that joined masks")
for (a, b), s in sorted(overlap.items()):
    score = min(s, overlap.get((b, a), 0.0))
    ca, cb = seg.classes[a], seg.classes[b]
    if a > b or ca == cb or score < threshold or obj[a] != obj[b]:
        continue
    if a in on and b in on:
        v = "right" if on[a] == on[b] else "WRONG"
        where = f"GT {gt.classes[on[a]]} / {gt.classes[on[b]]}"
    else:
        v, where = "unlabeled", "mask mostly on unlabeled GT"
    verdict[v] += 1
    print(f"  {a:5d} {ca:16s} {b:5d} {cb:16s} {score:.2f}  {v:9s} {where}")
print(
    f"  total {sum(verdict.values())}: "
    + ", ".join(f"{k} {n}" for k, n in verdict.most_common())
)
