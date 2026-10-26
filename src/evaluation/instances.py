"""Cross-view instance consistency of a segmentation against rendered ground truth.

Per view, a predicted instance matches a GT instance at IoU >= 0.5 (one-to-one by
construction). Across views we then ask: is each GT object found, and does it keep
one predicted id everywhere it is found? For refinement, two pixel-level scores
matter as well: how much GT surface is covered by matched masks (pixel recall), and
how much of the predicted masks spills over into other GT objects (leakage), since
smoothing within a leaking mask crosses a real boundary.
"""

from __future__ import annotations

import argparse
import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl
import numpy as np

mpl.use("Agg")
import matplotlib.pyplot as plt

from src.loaders import Views
from src.loaders.scannetpp.gt import NO_MESH, GroundTruth
from src.segmentation import Segmentation
from src.segmentation.base import NO_INSTANCE

log = logging.getLogger(__name__)

IOU_MATCH = 0.5
MIN_AREA = 0.002  # GT instances smaller than this fraction of a view are not scored
STRUCTURE = ("wall", "floor", "ceiling")
NOT_OBJECTS = ("split",)  # ScanNet++ annotation helper label, scored as unlabeled


@dataclass
class InstanceScores:
    """Detection and id consistency over all views of one scene."""

    recall: float  # GT (view, instance) pairs matched
    pixel_recall: float  # GT pixels of scored instances that belong to a matched one
    precision: float  # predicted (view, instance) pairs matched
    leakage: float  # predicted pixels outside the GT instance each mask mostly covers
    cross_leakage: (
        float  # the part of leakage that lands on a GT object of another class
    )
    consistent: float  # GT instances matched in >= 2 views that keep a single id
    id_switches: int  # sum over GT instances of (distinct predicted ids - 1)
    merges: int  # predicted ids matched to more than one GT instance
    num_gt: int  # GT instances matched in >= 2 views


def overlap_counts(
    gt: np.ndarray, pred: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """GT ids, predicted ids and their pixel intersection counts in one view.

    Pixels where the GT ray missed the mesh are left out.
    """
    valid = gt != NO_MESH
    g, p = gt[valid].astype(np.int64), pred[valid].astype(np.int64)
    g_ids, g_inv = np.unique(g, return_inverse=True)
    p_ids, p_inv = np.unique(p, return_inverse=True)
    inter = np.bincount(
        g_inv * len(p_ids) + p_inv, minlength=len(g_ids) * len(p_ids)
    ).reshape(len(g_ids), len(p_ids))
    return g_ids, p_ids, inter


def match_view(gt: np.ndarray, pred: np.ndarray) -> list[tuple[int, int, float]]:
    """Return (gt id, predicted id, IoU) pairs with IoU >= 0.5 in one view."""
    g_ids, p_ids, inter = overlap_counts(gt, pred)
    union = inter.sum(1, keepdims=True) + inter.sum(0, keepdims=True) - inter
    iou = inter / np.maximum(union, 1)
    return [
        (int(g_ids[a]), int(p_ids[b]), float(iou[a, b]))
        for a, b in zip(*np.nonzero(iou >= IOU_MATCH), strict=True)
        if g_ids[a] >= 0 and p_ids[b] >= 0
    ]


def score(
    gt: GroundTruth, pred: Segmentation, exclude: tuple[str, ...] = ()
) -> InstanceScores:
    """Score a segmentation; GT classes in `exclude` are left out."""
    num, height, width = gt.instance_ids.shape
    min_px = MIN_AREA * height * width
    not_objects = [g for g, c in gt.classes.items() if c.lower() in NOT_OBJECTS]
    gt_ids = np.where(
        np.isin(gt.instance_ids, not_objects), NO_INSTANCE, gt.instance_ids
    )
    pred_ids_of: dict[int, set[int]] = defaultdict(set)
    gt_ids_of: dict[int, set[int]] = defaultdict(set)
    views_of: dict[int, int] = defaultdict(int)
    n_gt = n_pred = n_hit = 0
    px_gt = px_hit = px_pred = px_leak = px_cross = 0
    for i in range(num):
        g_ids, p_ids, inter = overlap_counts(gt_ids[i], pred.instance_ids[i])
        g_area = inter.sum(1)
        scored = {
            int(g)
            for g, a in zip(g_ids, g_area, strict=True)
            if g >= 0 and a >= min_px and gt.classes.get(int(g)) not in exclude
        }
        matches = match_view(gt_ids[i], pred.instance_ids[i])
        pairs = [(g, p) for g, p, _ in matches if g in scored]
        # predictions matching an unscored GT instance are neither right nor wrong
        ignored = {p for g, p, _ in matches if g not in scored}
        # so are predictions lying mostly on unlabeled GT: nothing to judge them by
        if NO_INSTANCE in g_ids:
            on_unlabeled = inter[g_ids == NO_INSTANCE][0] > inter.sum(0) / 2
            ignored |= set(p_ids[on_unlabeled].tolist())
        n_gt += len(scored)
        n_pred += len(set(p_ids.tolist()) - {NO_INSTANCE} - ignored)
        n_hit += len(pairs)
        area = dict(zip(g_ids.tolist(), g_area.tolist(), strict=True))
        px_gt += sum(area[g] for g in scored)
        px_hit += sum(area[g] for g, _ in pairs)
        # leakage over labeled GT pixels, per mask relative to its dominant GT object
        labeled = inter[g_ids >= 0][:, p_ids >= 0]
        names = np.array([gt.classes.get(int(g), "") for g in g_ids[g_ids >= 0]])
        for col in range(labeled.shape[1]):
            dom = names[labeled[:, col].argmax()]
            if dom not in exclude:
                px_pred += labeled[:, col].sum()
                px_leak += labeled[:, col].sum() - labeled[:, col].max()
                px_cross += labeled[names != dom, col].sum()
        for g, p in pairs:
            pred_ids_of[g].add(p)
            gt_ids_of[p].add(g)
            views_of[g] += 1
    multi = [g for g, n in views_of.items() if n >= 2]  # noqa: PLR2004
    return InstanceScores(
        recall=n_hit / max(n_gt, 1),
        pixel_recall=px_hit / max(px_gt, 1),
        precision=n_hit / max(n_pred, 1),
        leakage=px_leak / max(px_pred, 1),
        cross_leakage=px_cross / max(px_pred, 1),
        consistent=sum(len(pred_ids_of[g]) == 1 for g in multi) / max(len(multi), 1),
        id_switches=sum(len(pred_ids_of[g]) - 1 for g in multi),
        merges=sum(len(s) > 1 for s in gt_ids_of.values()),
        num_gt=len(multi),
    )


def contact_sheet(views: Views, seg: Segmentation, title: str, path: Path) -> None:
    """All views with instance overlays and id labels, to eyeball id consistency."""
    num = len(views)
    cols = 5
    rows = -(-num // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 2.9 * rows))
    palette = np.random.default_rng(0).integers(40, 255, size=(4096, 3))
    for i, ax in enumerate(axes.flat):
        ax.axis("off")
        if i >= num:
            continue
        ids = seg.instance_ids[i]
        over = views.images[i].astype(float)
        labeled = ids >= 0
        over[labeled] = 0.45 * over[labeled] + 0.55 * palette[ids[labeled] % 4096]
        ax.imshow(over.astype(np.uint8))
        for obj in np.unique(ids[labeled]):
            ys, xs = np.nonzero(ids == obj)
            ax.text(
                np.median(xs),
                np.median(ys),
                f"{obj}",
                fontsize=6,
                color="white",
                ha="center",
                bbox={"facecolor": "black", "alpha": 0.5, "pad": 0.5},
            )
        ax.set_title(f"view {i}", fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=80, pil_kwargs={"quality": 85})
    plt.close(fig)


def main() -> None:
    """Score a cached segmentation against the rendered GT of the same views."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("views", type=Path, help="prepared views (.npz)")
    parser.add_argument("segmentation", type=Path, help="cached Segmentation (.npz)")
    parser.add_argument("--figures", type=Path, default=Path("outputs/instances"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    views = Views.load(args.views)
    gt = GroundTruth.load(args.views.with_name(f"gt_{args.views.stem}.npz"))
    pred = Segmentation.load(args.segmentation)
    for name, exclude in (("all", ()), ("objects", STRUCTURE)):
        s = score(gt, pred, exclude)
        log.info(
            "%s %-7s recall %.2f  pixel recall %.2f  precision %.2f  leakage %.3f"
            " (cross-class %.3f)  consistent %.2f (%d GT in >=2 views)"
            "  id switches %d  merges %d",
            views.scene_id,
            name,
            s.recall,
            s.pixel_recall,
            s.precision,
            s.leakage,
            s.cross_leakage,
            s.consistent,
            s.num_gt,
            s.id_switches,
            s.merges,
        )
    stem = f"{views.scene_id}_{args.segmentation.stem}"
    contact_sheet(views, pred, stem, args.figures / f"{stem}.jpg")
    gt_seg = Segmentation(np.where(gt.instance_ids >= 0, gt.instance_ids, -1))
    gt_path = args.figures / f"{views.scene_id}_gt.jpg"
    contact_sheet(views, gt_seg, f"{views.scene_id} ground truth", gt_path)
    log.info("figures -> %s", args.figures)


if __name__ == "__main__":
    main()
