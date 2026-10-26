"""E14: the visibility evidence behind every link of one mask.

For mask a and each mask b it overlaps in another view: how many of a's pixels
land inside b's image, how many of those are visible (depth test), how many land
in b, and the same for the reverse direction b -> a's view.

uv run python experiments/scripts/link_evidence.py <scene> <per-view stem> <mask id>...
"""

import sys
from pathlib import Path

import numpy as np

from src.geometry import Geometry
from src.segmentation import Segmentation
from src.segmentation.base import ID_STRIDE, NO_INSTANCE

scene, stem = sys.argv[1], sys.argv[2]
focus = [int(m) for m in sys.argv[3:]]
d = Path("data/processed/scannetpp") / scene
seg = Segmentation.load(d / f"{stem}.npz")
geom = Geometry.load(d / "dejaview_k16_views_n20_uniform_504.npz")
num, height, width = seg.instance_ids.shape
points = geom.world_points()
world_to_cam = np.linalg.inv(geom.cam_to_world)
area = dict(zip(*np.unique(seg.instance_ids, return_counts=True), strict=True))

counts = {}  # (mask, view) -> (in image, visible)
landed = {}  # (a, b) -> visible pixels of a landing in b
for i in range(num):
    labeled = (seg.instance_ids[i] >= 0) & (geom.depth[i] > 0)
    ids, pts = seg.instance_ids[i][labeled], points[i][labeled]
    for j in range(num):
        if i == j:
            continue
        cam = pts @ world_to_cam[j, :3, :3].T + world_to_cam[j, :3, 3]
        z, k = cam[:, 2], geom.intrinsics[j]
        with np.errstate(divide="ignore", invalid="ignore"):
            u = np.floor(k[0, 0] * cam[:, 0] / z + k[0, 2]).astype(np.int64)
            v = np.floor(k[1, 1] * cam[:, 1] / z + k[1, 2]).astype(np.int64)
        inside = (z > 0) & (u >= 0) & (u < width) & (v >= 0) & (v < height)
        u, v, z, src = u[inside], v[inside], z[inside], ids[inside]
        target = geom.depth[j][v, u]
        visible = np.abs(z - target) <= 0.05 * target
        for m in np.unique(src):
            counts[int(m), j] = (int(np.sum(src == m)), int(np.sum(visible[src == m])))
        pairs, n = np.unique(
            np.stack([src[visible], seg.instance_ids[j][v[visible], u[visible]]]),
            axis=1,
            return_counts=True,
        )
        for (a, b), c in zip(pairs.T, n, strict=True):
            if b != NO_INSTANCE:
                landed[int(a), int(b)] = int(c)

for a in focus:
    print(f"{scene} mask {a} ({seg.classes[a]}, {area[a]} px)")
    print("  b      class         | a in b's image: inside visible landed overlap"
          " | b in a's image: inside visible landed overlap visible/inside")  # fmt: skip
    for (x, b), c in sorted(landed.items()):
        if x != a or c < 20:
            continue
        ins, vis = counts[a, b // ID_STRIDE]
        b_ins, b_vis = counts.get((b, a // ID_STRIDE), (0, 0))
        back = landed.get((b, a), 0)
        print(
            f"  {b:5d}  {seg.classes[b]:13s} | {ins:6d} {vis:6d} {c:6d} {c / vis:.2f}"
            f" | {b_ins:6d} {b_vis:6d} {back:6d} {back / max(b_vis, 1):.2f}"
            f" {b_vis / max(b_ins, 1):.2f}"
        )
