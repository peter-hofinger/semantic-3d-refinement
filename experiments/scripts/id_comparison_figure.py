"""Side-by-side ids across views: GT, SAM 3 video mode (E4), final setting (E17).

Picks the views where one GT object is largest (spread over the sequence) and shows
image, GT, video mode and final ids per view. Same color and number = same id, so
a consistent method shows one object in one color in every column.

uv run python experiments/scripts/id_comparison_figure.py <scene> <GT id> [views...]
"""

import sys
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.loaders import Views
from src.loaders.scannetpp.gt import GroundTruth
from src.segmentation import Segmentation

scene, target = sys.argv[1], int(sys.argv[2])
d = Path("data/processed/scannetpp") / scene
views = Views.load(d / "views_n20_uniform_504.npz")
gt = GroundTruth.load(d / "gt_views_n20_uniform_504.npz")
rows = {
    "ground truth": Segmentation(np.where(gt.instance_ids >= 0, gt.instance_ids, -1)),
    "SAM 3 video mode": Segmentation.load(d / "sam3video_views_n20_uniform_504.npz"),
    "final: image mode + 3D linking": Segmentation.load(
        d / "sam3image-top100-t0.3_views_n20_uniform_504_assoc-xc0.7-vf0.25-kl0.5"
        "-dejaview_k16_views_n20_uniform_504.npz"
    ),
}

if len(sys.argv) > 3:
    picked = [int(v) for v in sys.argv[3:]]
else:  # three largest views of the target, at least 3 views apart
    area = (gt.instance_ids == target).sum(axis=(1, 2))
    picked = []
    for v in np.argsort(-area):
        if area[v] > 0 and all(abs(v - p) >= 3 for p in picked):
            picked.append(int(v))
        if len(picked) == 3:
            break
    picked.sort()
print(f"{scene}: views {picked}")

palette = np.random.default_rng(1).integers(40, 255, size=(4096, 3))
min_label = 0.01 * gt.instance_ids[0].size
fig, axes = plt.subplots(
    len(rows) + 1, len(picked), figsize=(4.2 * len(picked), 2.9 * (len(rows) + 1))
)
for c, v in enumerate(picked):
    axes[0, c].imshow(views.images[v])
    axes[0, c].set_title(f"view {v}", fontsize=10)
    for r, (name, seg) in enumerate(rows.items(), start=1):
        ids = seg.instance_ids[v]
        over = views.images[v].astype(float)
        labeled = ids >= 0
        over[labeled] = 0.35 * over[labeled] + 0.65 * palette[ids[labeled] % 4096]
        axes[r, c].imshow(over.astype(np.uint8))
        for obj in np.unique(ids[labeled]):
            ys, xs = np.nonzero(ids == obj)
            if len(xs) >= min_label:
                axes[r, c].text(
                    np.median(xs), np.median(ys), f"{obj}", fontsize=7, color="white",
                    ha="center", va="center",
                    bbox={"facecolor": "black", "alpha": 0.55, "pad": 0.6, "lw": 0},
                )
        on_target = np.unique(ids[gt.instance_ids[v] == target])
        print(f"  view {v} {name}: ids on GT {target}: {on_target[on_target >= 0]}")
        if c == 0:
            axes[r, c].set_ylabel(name, fontsize=10)
for ax in axes.flat:
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
fig.tight_layout()
out = Path("experiments/figures") / f"id_comparison_{scene}.jpg"
fig.savefig(out, dpi=110, pil_kwargs={"quality": 85})
print(f"figure: {out}")
