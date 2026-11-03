"""Score a labeled per-pixel output against per-view ground truth: surface and labels.

Ground truth is one point per pixel with ground-truth depth, lifted through the
ground-truth cameras (Hypersim's per-pixel maps). Hypersim's meshes are not public
(they need the purchased Evermotion assets), so these pixels are the only 3D ground
truth there is. The method's output is a point and a label per input pixel, in the
first view's frame at its own scale. Pixel i of the output and pixel i of the ground
truth see the same surface, so every score is taken on these correspondences and has
no threshold, cap or grid of its own:

- Alignment: one similarity per output by Umeyama's least squares on every pixel with
  ground-truth depth and a finite output (as VGGT and π³ align point maps; π³ and
  CUT3R add ICP, whose radius would be one more number to choose).
- Surface (7-Scenes / NRGBD protocol of CUT3R, π³, VGGT): accuracy is the mean and
  median distance of each output point to the nearest ground-truth point, completeness
  the other way round, Chamfer their mean. Per class, accuracy takes the output points
  of the pixels whose ground truth has the class, completeness the ground-truth points
  of the class. Neither is capped: an outlier counts with its full distance.
- Labels are scored per ground-truth point on the output's label of the same pixel,
  so they measure the segmentation, not where its points landed. Pixels whose
  ground truth is unlabeled or of a class outside the evaluation are void.

Segments: an instance of an object class is one segment over all views, so an object
split into two ids across views is two segments (the scene-level metrics of Panoptic
Lifting, PanSt3R). The background classes (walls, floor, ceiling) have no instances in
Hypersim's ground truth; there, and in the output, all pixels of such a class form one
segment. Per class, every evaluated class is scored; the aggregates, over the object
classes only.

- mIoU: IoU per class over non-void points, mean over the classes in either.
- PQ, SQ, RQ (Kirillov et al. 2019, scene-level as in Panoptic Lifting): a predicted
  and a ground-truth segment of the same class match if their IoU is above 0.5, which
  makes the matching unique and the same as the IoU-maximising assignment. Void
  points are left out of the IoU; an unmatched prediction lying more than half on
  void is not a false positive (panopticapi).
- AP, AP50, AP25 (ScanNet's `evaluate_semantic_instance.py`, as SegVGGT, IGGT,
  PanSt3R report): per class and IoU threshold, each ground-truth segment takes the
  highest-scoring prediction above the threshold, further matches are false
  positives; an unmatched prediction is a false positive unless more than the
  threshold of it lies on void. AP is averaged over thresholds 0.5:0.95:0.05.
  ScanNet's minimum of 100 mesh vertices per instance has no counterpart on pixels
  and is left out.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import cKDTree

from src.segmentation.base import Segmentation

NONE = -1
BACKGROUND = ("wall", "floor", "ceiling")  # no instances in Hypersim: one segment each
PQ_IOU = 0.5  # PQ's matching threshold, part of its definition
AP_IOUS = np.arange(0.5, 0.951, 0.05)  # ScanNet's AP
AP_IOU_50, AP_IOU_25 = 0.5, 0.25


@dataclass
class LabeledCloud:
    """Ground-truth points with one instance id each (NONE for none) and its class."""

    points: np.ndarray  # (N, 3) float64, ground-truth frame, m
    instance: np.ndarray  # (N,) int64
    classes: dict[int, str] = field(default_factory=dict)
    pixels: np.ndarray | None = None  # (V, H, W) bool, the pixels of the N points

    def __post_init__(self) -> None:
        """Check shapes."""
        assert self.points.ndim == 2 and self.points.shape[1] == 3, self.points.shape  # noqa: PLR2004
        assert self.instance.shape == (len(self.points),), self.instance.shape
        assert np.isfinite(self.points).all()
        assert self.pixels is None or self.pixels.sum() == len(self.points)

    def class_of(self, ids: np.ndarray) -> np.ndarray:
        """Class name of each id, '' for NONE and ids without a class."""
        return class_names(ids, self.classes)


def class_names(ids: np.ndarray, classes: dict[int, str]) -> np.ndarray:
    """Class name of each id, '' for NONE and ids without a class."""
    lut = {NONE: ""} | classes
    unique, inverse = np.unique(ids, return_inverse=True)
    names = np.array([lut.get(int(i), "") for i in unique], dtype=object)
    return names[inverse.reshape(ids.shape)]


def lift(depth: np.ndarray, k: np.ndarray, cam_to_world: np.ndarray) -> np.ndarray:
    """World points (H, W, 3) of a z-depth map through a pinhole; pixel centers +0.5."""
    h, w = depth.shape
    v, u = np.mgrid[0:h, 0:w] + 0.5
    rays = np.stack([u, v, np.ones_like(u)], axis=-1) @ np.linalg.inv(k).T
    return (depth[..., None] * rays) @ cam_to_world[:3, :3].T + cam_to_world[:3, 3]


def ground_truth_cloud(
    depth: np.ndarray,
    instance_ids: np.ndarray,
    classes: dict[int, str],
    intrinsics: np.ndarray,
    cam_to_world: np.ndarray,
) -> LabeledCloud:
    """Every pixel with ground-truth depth, lifted; negative ids: none."""
    assert depth.shape == instance_ids.shape and depth.ndim == 3, depth.shape  # noqa: PLR2004
    pixels = depth > 0
    points = np.stack(
        [
            lift(d.astype(np.float64), k.astype(np.float64), c2w)
            for d, k, c2w in zip(depth, intrinsics, cam_to_world, strict=True)
        ]
    )
    ids = instance_ids[pixels].astype(np.int64)
    return LabeledCloud(
        points[pixels], np.where(ids >= 0, ids, NONE), dict(classes), pixels
    )


def umeyama(
    source: np.ndarray, target: np.ndarray
) -> tuple[float, np.ndarray, np.ndarray]:
    """Least-squares similarity (s, R, t): target ≈ s R source + t (Umeyama 1991)."""
    assert source.shape == target.shape and source.shape[1] == 3, source.shape  # noqa: PLR2004
    mu_s, mu_t = source.mean(0), target.mean(0)
    a, b = source - mu_s, target - mu_t
    u, d, vt = np.linalg.svd(b.T @ a / len(a))
    sign = np.ones(3)
    sign[2] = np.sign(np.linalg.det(u @ vt)) or 1.0
    rotation = (u * sign) @ vt
    scale = float((d * sign).sum() / (a**2).sum(1).mean())
    return scale, rotation, mu_t - scale * rotation @ mu_s


def surface_scores(acc: np.ndarray, comp: np.ndarray) -> dict[str, float]:
    """Accuracy, completeness (mean and median, cm) and Chamfer (their means' mean)."""
    out = {
        "accuracy_cm": 100 * float(acc.mean()) if len(acc) else float("nan"),
        "accuracy_median_cm": 100 * float(np.median(acc)) if len(acc) else float("nan"),
        "completeness_cm": 100 * float(comp.mean()),
        "completeness_median_cm": 100 * float(np.median(comp)),
    }
    return out | {"chamfer_cm": (out["accuracy_cm"] + out["completeness_cm"]) / 2}


def surface_metrics(
    pred: np.ndarray, truth: LabeledCloud, evaluated: list[str]
) -> dict[str, float | dict[str, dict[str, float]]]:
    """Surface scores of the scene and per evaluated class (see module).

    pred: (N, 3) the output point of each ground-truth point's pixel, NaN where none.
    """
    assert pred.shape == truth.points.shape, (pred.shape, truth.points.shape)
    found = np.isfinite(pred).all(1)
    acc = np.full(len(pred), np.nan)
    acc[found] = cKDTree(truth.points).query(pred[found], workers=-1)[0]
    comp = cKDTree(pred[found]).query(truth.points, workers=-1)[0]
    gt_class = truth.class_of(truth.instance)
    per_class = {
        c: surface_scores(acc[found & (gt_class == c)], comp[gt_class == c])
        for c in evaluated
        if (gt_class == c).any()
    }
    return surface_scores(acc[found], comp) | {"surface_per_class": per_class}


def segments(
    ids: np.ndarray, classes: dict[int, str], evaluated: list[str]
) -> tuple[np.ndarray, dict[int, str]]:
    """Segment per point (see module) and the class of each segment.

    Objects keep their instance id; a background class c becomes the segment
    -2 - evaluated.index(c), as does no instance. Classes outside `evaluated`: NONE.
    """
    names = class_names(ids, classes)
    out = np.where(np.isin(names, evaluated), ids, NONE)
    seg_class = {int(i): classes[int(i)] for i in np.unique(out) if i != NONE}
    for c in BACKGROUND:
        if c in evaluated:
            out[names == c] = -2 - evaluated.index(c)
            seg_class[-2 - evaluated.index(c)] = c
    present = set(np.unique(out).tolist())
    seg_class = {s: c for s, c in seg_class.items() if s in present}
    return out, seg_class


@dataclass
class Overlaps:
    """Segment sizes and intersections over all ground-truth points."""

    gt_class: dict[int, str]
    pred_class: dict[int, str]
    gt_size: dict[int, int]
    pred_size: dict[int, int]  # including the points on void
    pred_void: dict[int, int]  # points of the prediction on void
    inter: dict[tuple[int, int], int]  # (pred, gt) of the same class

    def iou(self, p: int, g: int) -> float:
        """IoU of a prediction and a ground-truth segment, void points included."""
        n = self.inter.get((p, g), 0)
        return n / (self.pred_size[p] + self.gt_size[g] - n)


def overlaps(
    gt: np.ndarray,
    gt_class: dict[int, str],
    pred: np.ndarray,
    pred_class: dict[int, str],
) -> Overlaps:
    """Sizes and same-class intersections of segments; NONE in gt is void."""
    pairs, counts = np.unique(np.stack([pred, gt]), axis=1, return_counts=True)
    gt_size = dict(zip(*np.unique(gt[gt != NONE], return_counts=True), strict=True))
    pred_size = dict(
        zip(*np.unique(pred[pred != NONE], return_counts=True), strict=True)
    )
    pred_void: dict[int, int] = {}
    inter: dict[tuple[int, int], int] = {}
    for p, g, n in zip(*pairs.tolist(), counts.tolist(), strict=True):
        if p == NONE:
            continue
        if g == NONE:
            pred_void[p] = n
        elif pred_class[p] == gt_class[g]:
            inter[p, g] = n
    return Overlaps(
        gt_class,
        pred_class,
        {int(k): int(v) for k, v in gt_size.items()},
        {int(k): int(v) for k, v in pred_size.items()},
        pred_void,
        inter,
    )


def semantic_iou(
    gt_names: np.ndarray, pred_names: np.ndarray, evaluated: list[str]
) -> dict[str, float]:
    """IoU per evaluated class over non-void points, for classes in either."""
    scored = np.isin(gt_names, evaluated)
    out = {}
    for c in evaluated:
        g, p = scored & (gt_names == c), scored & (pred_names == c)
        union = (g | p).sum()
        if union:
            out[c] = float((g & p).sum() / union)
    return out


def panoptic_quality(ov: Overlaps, evaluated: list[str]) -> dict[str, dict[str, float]]:
    """PQ, SQ and RQ per class (panopticapi's rules, see module)."""
    matched = {}
    for p, g in ov.inter:
        # IoU without the prediction's void points, as panopticapi
        n = ov.inter[p, g]
        iou = n / (ov.pred_size[p] - ov.pred_void.get(p, 0) + ov.gt_size[g] - n)
        if iou > PQ_IOU:
            matched[p, g] = iou
    out = {}
    for c in evaluated:
        tp = [iou for (p, _), iou in matched.items() if ov.pred_class[p] == c]
        hit_p = {p for p, _ in matched}
        hit_g = {g for _, g in matched}
        fn = sum(1 for g, k in ov.gt_class.items() if k == c and g not in hit_g)
        fp = sum(
            1
            for p, k in ov.pred_class.items()
            if k == c
            and p not in hit_p
            and ov.pred_void.get(p, 0) / ov.pred_size[p] <= PQ_IOU
        )
        if tp or fp or fn:
            rq = len(tp) / (len(tp) + fp / 2 + fn / 2)
            sq = float(np.mean(tp)) if tp else 0.0
            out[c] = {"pq": sq * rq, "sq": sq, "rq": rq}
    return out


def scannet_matches(
    ov: Overlaps, scores: dict[int, float], c: str, threshold: float
) -> tuple[list[float], list[float], int]:
    """Per scored detection: true or false, its score; and the missed ground truths."""
    gts = [g for g, k in ov.gt_class.items() if k == c]
    preds = [p for p, k in ov.pred_class.items() if k == c]
    visited: set[int] = set()
    true, score = [1.0] * len(gts), [-np.inf] * len(gts)
    match = [False] * len(gts)
    missed = 0
    for i, g in enumerate(gts):
        found = False
        for p in preds:
            if (p, g) not in ov.inter or p in visited or ov.iou(p, g) <= threshold:
                continue
            confidence = scores.get(p, 0.0)
            if match[i]:  # a second match: the lower score is a false positive
                true.append(0.0)
                score.append(min(score[i], confidence))
                match.append(True)
                score[i] = max(score[i], confidence)
            else:
                found, match[i], score[i] = True, True, confidence
                visited.add(p)
        missed += not found
    y_true = [t for t, m in zip(true, match, strict=True) if m]
    y_score = [s for s, m in zip(score, match, strict=True) if m]
    for p in preds:
        if any(ov.iou(p, g) > threshold for g in gts if (p, g) in ov.inter):
            continue
        if ov.pred_void.get(p, 0) / ov.pred_size[p] <= threshold:
            y_true.append(0.0)
            y_score.append(scores.get(p, 0.0))
    return y_true, y_score, missed


def scannet_ap(
    ov: Overlaps, scores: dict[int, float], c: str, threshold: float
) -> float:
    """AP of one class at one IoU threshold, as ScanNet's evaluate_semantic_instance."""
    if c not in ov.gt_class.values():
        return float("nan")
    if c not in ov.pred_class.values():
        return 0.0
    y_true, y_score, missed = scannet_matches(ov, scores, c, threshold)
    order = np.argsort(y_score)
    t_sorted, s_sorted = np.array(y_true)[order], np.array(y_score)[order]
    cumsum = np.cumsum(t_sorted)
    _, unique = np.unique(s_sorted, return_index=True)
    total = cumsum[-1] if len(cumsum) else 0
    cumsum = np.append(cumsum, 0)  # cumsum[-1] = 0 for the first unique score
    precision, recall = np.zeros(len(unique) + 1), np.zeros(len(unique) + 1)
    for k, j in enumerate(unique):
        tp = total - cumsum[j - 1]
        fp = len(s_sorted) - j - tp
        fn = cumsum[j - 1] + missed
        precision[k] = tp / (tp + fp)
        recall[k] = tp / (tp + fn)
    precision[-1], recall[-1] = 1.0, 0.0
    widths = np.convolve(np.r_[recall[0], recall, 0.0], [-0.5, 0, 0.5], "valid")
    return float(precision @ widths)


def label_metrics(
    truth: LabeledCloud,
    pred_ids: np.ndarray,
    pred_classes: dict[int, str],
    pred_scores: dict[int, float],
    evaluated: list[str],
) -> dict[str, float | dict]:
    """MIoU, PQ / SQ / RQ and AP / AP50 / AP25, per class and over object classes.

    pred_ids: (N,) the output's instance at each ground-truth point's pixel.
    """
    assert pred_ids.shape == truth.instance.shape, pred_ids.shape
    gt_seg, gt_class = segments(truth.instance, truth.classes, evaluated)
    pred_seg, pred_class = segments(pred_ids, pred_classes, evaluated)
    scores = {
        int(s): pred_scores.get(int(s), 0.0) for s in pred_class if s >= 0
    }  # a background segment is its class's only prediction: its score is moot
    ious = semantic_iou(
        class_names(gt_seg, gt_class), class_names(pred_seg, pred_class), evaluated
    )
    ov = overlaps(gt_seg, gt_class, pred_seg, pred_class)
    pq = panoptic_quality(ov, evaluated)
    ap = {
        c: [scannet_ap(ov, scores, c, t) for t in (*AP_IOUS, AP_IOU_25)]
        for c in evaluated
    }
    objects = [c for c in evaluated if c not in BACKGROUND]

    def mean(values: list[float]) -> float:
        values = [v for v in values if not np.isnan(v)]
        return float(np.mean(values)) if values else float("nan")

    return {
        "miou": mean([ious[c] for c in objects if c in ious]),
        **{k: mean([pq[c][k] for c in objects if c in pq]) for k in ("pq", "sq", "rq")},
        "ap": mean([v for c in objects for v in ap[c][:-1]]),
        "ap50": mean([ap[c][0] for c in objects]),
        "ap25": mean([ap[c][-1] for c in objects]),
        "iou_per_class": ious,
        "pq_per_class": pq,
        "ap_per_class": {
            c: {"ap": mean(v[:-1]), "ap50": v[0], "ap25": v[-1]}
            for c, v in ap.items()
            if not np.isnan(v[0])
        },
    }


def score_output(
    truth: LabeledCloud, points: np.ndarray, labels: Segmentation, evaluated: list[str]
) -> dict:
    """Align a per-pixel output (V, H, W, 3) with its labels, score it (see module).

    The alignment is returned under `align` as a (4, 4) matrix, s R | t.
    """
    assert truth.pixels is not None
    assert points.shape == (*truth.pixels.shape, 3), (points.shape, truth.pixels.shape)
    pred = points[truth.pixels].astype(np.float64)
    found = np.isfinite(pred).all(1)
    s, rotation, t = umeyama(pred[found], truth.points[found])
    pred = s * pred @ rotation.T + t
    align = np.eye(4)
    align[:3, :3], align[:3, 3] = s * rotation, t
    ids = labels.instance_ids[truth.pixels].astype(np.int64)
    ids = np.where(ids >= 0, ids, NONE)
    return {
        "scale": s,
        "align": align,
        **surface_metrics(pred, truth, evaluated),
        **label_metrics(truth, ids, labels.classes, labels.scores, evaluated),
    }
