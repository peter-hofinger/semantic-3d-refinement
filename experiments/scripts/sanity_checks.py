"""Sanity checks behind E4, E6, E8 and E11: scorer, association and Déjà View depth.

uv run python experiments/scripts/sanity_checks.py
"""

from pathlib import Path

import numpy as np

from src.evaluation.instances import STRUCTURE, score
from src.geometry import Geometry
from src.loaders import Views
from src.loaders.scannetpp.gt import GroundTruth
from src.segmentation import Segmentation
from src.segmentation.associate import Associator, gt_geometry
from src.segmentation.base import ID_STRIDE

DATA = Path("data/processed/scannetpp")
SCENES = ("5ee7c22ba0", "7b6477cb95")


def fmt(s: object) -> str:
    """One line per InstanceScores."""
    return "  ".join(
        f"{k} {v:.3f}" if isinstance(v, float) else f"{k} {v}"
        for k, v in vars(s).items()
    )


def per_view_ids(gt: GroundTruth) -> Segmentation:
    """GT masks with fresh ids in every view (what image mode produces)."""
    ids = np.full(gt.instance_ids.shape, -1, np.int32)
    classes = {}
    for i, view in enumerate(gt.instance_ids):
        for k, g in enumerate(np.unique(view[view >= 0])):
            ids[i][view == g] = i * ID_STRIDE + k
            classes[i * ID_STRIDE + k] = gt.classes[int(g)]
    return Segmentation(ids, classes)


for scene in SCENES:
    views = Views.load(DATA / scene / "views_n20_uniform_504.npz")
    gt = GroundTruth.load(DATA / scene / "gt_views_n20_uniform_504.npz")
    ids = np.where(gt.instance_ids >= 0, gt.instance_ids, -1)
    print(f"== {scene}")

    print(" scorer: GT vs GT (expect all 1, leakage 0)")
    print("   ", fmt(score(gt, Segmentation(ids, gt.classes))))
    relabeled = np.where(
        ids >= 0, ids + 100000 * np.arange(len(ids))[:, None, None], -1
    )
    print(" scorer: GT relabeled per view (expect recall 1, consistent 0)")
    print("   ", fmt(score(gt, Segmentation(relabeled))))
    print(" scorer: GT id pairs 2k, 2k+1 merged (expect leakage > 0)")
    print("   ", fmt(score(gt, Segmentation(np.where(ids >= 0, ids // 2, -1)))))

    local = per_view_ids(gt)
    linked = Associator().associate(local, gt_geometry(views, gt))
    print(
        f" association: GT masks per view + GT geometry: {len(local.classes)} masks "
        f"-> {len(linked.classes)} objects"
    )
    for name, exclude in (("all", ()), ("objects", STRUCTURE)):
        print(f"   {name:8s}", fmt(score(gt, linked, exclude)))

    dv = Geometry.load(DATA / scene / "dejaview_k16_views_n20_uniform_504.npz")
    rel = []
    for i in range(len(dv.depth)):
        m = gt.depth[i] > 0
        s = np.median(gt.depth[i][m] / dv.depth[i][m])
        rel.append(
            np.median(np.abs(s * dv.depth[i][m] - gt.depth[i][m]) / gt.depth[i][m])
        )
    print(
        f" Deja View: depth {dv.depth.shape}, finite {np.isfinite(dv.depth).all()}, "
        f"conf {dv.confidence.min():.2f}-{dv.confidence.max():.2f}; per-view scale-aligned "
        f"median abs rel error mean {np.mean(rel):.3f}, worst {np.max(rel):.3f} "
        f"(view {int(np.argmax(rel))})"
    )
