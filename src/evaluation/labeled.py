"""Score a labeled point cloud against per-view ground truth: surface, classes, objects.

Both clouds are reduced to one point per 1 cm voxel, the majority instance label of
its points, so point density and view overlap do not weigh in. Ground truth is every
pixel of the views with ground-truth depth, lifted through the ground-truth cameras
(Hypersim's per-pixel maps, ScanNet++'s rendered mesh). A prediction comes in its own
frame (Déjà View's, up to scale) and is mapped into the ground truth's by one
similarity (Umeyama 1991) fitted on per-pixel correspondences, the usual protocol for
point maps (VGGT, CUT3R): `align_points`. Camera centers alone do not fix it: Hypersim's
trajectories are close to a line, so the rotation about it is left free.

Surface (Knapitsch et al. 2017, Tanks and Temples): accuracy is the distance of each
predicted point to the nearest ground-truth point, completeness the other way round;
precision and recall are the fractions of either within a tolerance τ, and the F-score
their harmonic mean. Distances are capped at `CAP` (1 m), as DTU's evaluation caps
outliers: a nearest-neighbor query for a point far from the other cloud costs
milliseconds, and a mean should not hinge on a few points outside a window.

Labels are transferred to the ground truth: every ground-truth point takes the
instance of the nearest predicted point within `TRANSFER`, none if there is no such
point. Labels are scored on the ground-truth points, as ScanNet's benchmark scores
mesh vertices, so a well-labeled object in the wrong place still loses its points:
refinement that moves surfaces into place shows in the label scores too.

- Semantics: IoU per class over ground-truth points, the mean over evaluated classes
  (mIoU; ScanNet, ScanNet++). Points whose ground-truth class is not evaluated are
  ignored.
- Instances: average precision (ScanNet's protocol, also ScanNet++): per class,
  predictions by descending score are matched greedily to unmatched ground-truth
  instances at IoU ≥ t; AP is the area under the precision envelope, averaged over t
  = 0.50, 0.55, ..., 0.95 (AP), or taken at t = 0.5 (AP50) and 0.25 (AP25). Walls,
  floors and ceilings are left out, as in ScanNet. A prediction that lies mostly on
  points of classes outside the evaluation is neither a hit nor a false positive.
  An object split into two ids across views fuses into two half-size masks in 3D,
  which miss IoU 0.5: AP also scores cross-view consistency.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import cKDTree

from src.evaluation.depth import lift
from src.evaluation.geometry import similarity

VOXEL = 0.01  # m, grid both clouds are reduced to
TOLERANCES = (0.01, 0.02, 0.05)  # m, precision / recall / F-score thresholds
TRANSFER = 0.05  # m, a ground-truth point takes the label of a prediction this close
CAP = 1.0  # m, distances above count as this
MIN_POINTS = 100  # voxels; smaller instances are not scored (ScanNet: 100 vertices)
IOU_THRESHOLDS = np.round(np.arange(0.5, 0.951, 0.05), 2)
STRUCTURE = ("wall", "floor", "ceiling")  # no instances (ScanNet's AP leaves them out)
NONE = -1
ALIGN_SAMPLES = 200_000  # pixels the similarity is fitted on
ALIGN_TRIM = 3.0  # refit on pixels within this many median residuals of the first fit


@dataclass
class LabeledCloud:
    """Points with one instance id each (NONE for none), its class and score."""

    points: np.ndarray  # (N, 3) float64, ground-truth frame, m
    instance: np.ndarray  # (N,) int64
    classes: dict[int, str] = field(default_factory=dict)
    scores: dict[int, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Check shapes."""
        assert self.points.ndim == 2 and self.points.shape[1] == 3, self.points.shape  # noqa: PLR2004
        assert self.instance.shape == (len(self.points),), self.instance.shape
        assert np.isfinite(self.points).all()

    def class_of(self, ids: np.ndarray) -> np.ndarray:
        """Class name of each id, '' for NONE and ids without a class."""
        lut = {NONE: ""} | self.classes
        return np.array([lut.get(int(i), "") for i in ids], dtype=object)


def align_points(
    pred: np.ndarray, truth: np.ndarray, valid: np.ndarray
) -> tuple[float, np.ndarray, np.ndarray]:
    """Similarity mapping per-pixel points onto ground truth, robust to outliers.

    Fitted on up to `ALIGN_SAMPLES` valid pixels (fixed seed), then refitted on those
    within `ALIGN_TRIM` times the median residual, so a few points far off (glass,
    sky) do not tilt it.
    """
    assert pred.shape == truth.shape and pred.shape[:-1] == valid.shape, pred.shape
    index = np.flatnonzero(valid)
    rng = np.random.default_rng(0)
    index = rng.choice(index, min(ALIGN_SAMPLES, len(index)), replace=False)
    a = pred.reshape(-1, 3)[index].astype(np.float64)
    b = truth.reshape(-1, 3)[index].astype(np.float64)
    s, rotation, t = similarity(a, b)
    residual = np.linalg.norm(s * a @ rotation.T + t - b, axis=1)
    keep = residual <= ALIGN_TRIM * np.median(residual)
    return similarity(a[keep], b[keep])


def voxelize(
    points: np.ndarray, labels: np.ndarray, size: float = VOXEL
) -> tuple[np.ndarray, np.ndarray]:
    """One point per occupied voxel (the mean) and its most frequent label."""
    assert points.shape == (len(labels), 3), (points.shape, labels.shape)
    keys = np.floor(points / size).astype(np.int64)
    _, voxel, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    voxel = voxel.ravel()
    centers = np.zeros((len(counts), 3))
    np.add.at(centers, voxel, points)
    pairs, votes = np.unique(np.stack([voxel, labels]), axis=1, return_counts=True)
    order = np.lexsort((-votes, pairs[0]))  # by voxel, most votes first
    first = np.r_[True, np.diff(pairs[0, order]) > 0]
    majority = np.empty(len(counts), dtype=np.int64)
    majority[pairs[0, order][first]] = pairs[1, order][first]
    return centers / counts[:, None], majority


def ground_truth_cloud(
    depth: np.ndarray,
    instance_ids: np.ndarray,
    classes: dict[int, str],
    intrinsics: np.ndarray,
    cam_to_world: np.ndarray,
) -> LabeledCloud:
    """Every pixel with ground-truth depth, lifted and voxelized; negative ids: none."""
    assert depth.shape == instance_ids.shape and depth.ndim == 3, depth.shape  # noqa: PLR2004
    points, labels = [], []
    for d, ids, k, c2w in zip(
        depth, instance_ids, intrinsics, cam_to_world, strict=True
    ):
        seen = d > 0
        points.append(lift(d.astype(np.float64), k.astype(np.float64), c2w)[seen])
        labels.append(np.where(ids[seen] >= 0, ids[seen], NONE))
    centers, majority = voxelize(np.concatenate(points), np.concatenate(labels))
    return LabeledCloud(centers, majority, dict(classes))


def predicted_cloud(
    points: np.ndarray,
    instance_ids: np.ndarray,
    classes: dict[int, str],
    scores: dict[int, float],
    align: tuple[float, np.ndarray, np.ndarray],
) -> LabeledCloud:
    """Per-pixel points (V, H, W, 3) mapped by the similarity, voxelized."""
    assert points.shape == (*instance_ids.shape, 3), (points.shape, instance_ids.shape)
    s, rotation, t = align
    mapped = s * points.reshape(-1, 3).astype(np.float64) @ rotation.T + t
    ids = instance_ids.reshape(-1).astype(np.int64)
    keep = np.isfinite(mapped).all(1)
    centers, majority = voxelize(
        mapped[keep], np.where(ids[keep] >= 0, ids[keep], NONE)
    )
    return LabeledCloud(centers, majority, dict(classes), dict(scores))


def surface_metrics(
    pred: np.ndarray, truth: np.ndarray, tolerances: tuple[float, ...] = TOLERANCES
) -> dict[str, float]:
    """Accuracy, completeness (mean, cm), Chamfer, precision / recall / F-score at τ."""
    acc, comp = (
        np.minimum(cKDTree(b).query(a, workers=-1, distance_upper_bound=CAP)[0], CAP)
        for a, b in ((pred, truth), (truth, pred))
    )
    out = {
        "accuracy_cm": 100 * float(acc.mean()),
        "completeness_cm": 100 * float(comp.mean()),
        "chamfer_cm": 50 * float(acc.mean() + comp.mean()),
    }
    for tau in tolerances:
        p, r = float((acc < tau).mean()), float((comp < tau).mean())
        tag = f"{100 * tau:g}cm"
        out |= {
            f"precision_{tag}": p,
            f"recall_{tag}": r,
            f"fscore_{tag}": 2 * p * r / (p + r) if p + r > 0 else 0.0,
        }
    return out


def transfer(
    pred: LabeledCloud, truth: LabeledCloud, radius: float = TRANSFER
) -> np.ndarray:
    """Instance of the nearest predicted point within radius per ground-truth point."""
    dist, nearest = cKDTree(pred.points).query(
        truth.points, workers=-1, distance_upper_bound=radius
    )
    found = np.isfinite(dist)
    out = np.full(len(truth.points), NONE, dtype=np.int64)
    out[found] = pred.instance[nearest[found]]
    return out


def semantic_iou(
    truth: LabeledCloud, pred: LabeledCloud, labels: np.ndarray, evaluated: list[str]
) -> dict[str, float]:
    """IoU per evaluated class over ground-truth points of evaluated classes."""
    gt_class, pred_class = truth.class_of(truth.instance), pred.class_of(labels)
    scored = np.isin(gt_class, evaluated)
    out = {}
    for c in evaluated:
        g, p = scored & (gt_class == c), scored & (pred_class == c)
        if g.sum() >= MIN_POINTS:
            out[c] = float((g & p).sum() / (g | p).sum())
    return out


def average_precision(matched: np.ndarray, num_gt: int) -> float:
    """Area under the precision envelope; matched is per prediction by score."""
    if num_gt == 0:
        return float("nan")
    if len(matched) == 0:
        return 0.0
    tp = np.cumsum(matched)
    precision = tp / np.arange(1, len(matched) + 1)
    recall = tp / num_gt
    envelope = np.maximum.accumulate(precision[::-1])[::-1]
    steps = np.diff(np.r_[0.0, recall])
    return float((envelope * steps).sum())


def instance_ap(
    truth: LabeledCloud,
    pred: LabeledCloud,
    labels: np.ndarray,
    evaluated: list[str],
    *,
    class_agnostic: bool = False,
) -> dict[float, float]:
    """Mean AP over classes per IoU threshold (ScanNet's protocol, see module)."""
    things = [c for c in evaluated if c not in STRUCTURE]
    gt_class = truth.class_of(truth.instance)
    ignored = ~np.isin(gt_class, evaluated)
    gt_ids, gt_size = np.unique(truth.instance, return_counts=True)
    gt = {
        int(i): n
        for i, n in zip(gt_ids, gt_size, strict=True)
        if i != NONE and n >= MIN_POINTS and truth.classes.get(int(i)) in things
    }
    pairs, inter = np.unique(
        np.stack([labels, truth.instance]), axis=1, return_counts=True
    )
    pred_ids, pred_size = np.unique(labels, return_counts=True)
    on_ignored = dict(zip(*np.unique(labels[ignored], return_counts=True), strict=True))
    preds = {
        int(i): n
        for i, n in zip(pred_ids, pred_size, strict=True)
        if i != NONE
        and n >= MIN_POINTS
        and pred.classes.get(int(i)) in things
        and on_ignored.get(i, 0) <= n / 2
    }
    overlap: dict[int, dict[int, int]] = {}
    for p, g, n in zip(*pairs, inter, strict=True):
        if int(p) in preds and int(g) in gt:
            overlap.setdefault(int(p), {})[int(g)] = int(n)

    def iou(p: int, g: int) -> float:
        n = overlap.get(p, {}).get(g, 0)
        return n / (preds[p] + gt[g] - n)

    groups = [None] if class_agnostic else things
    out = {}
    for t in (*IOU_THRESHOLDS, 0.25):
        aps = []
        for c in groups:
            gts = {g for g in gt if c is None or truth.classes[g] == c}
            ranked = sorted(
                (p for p in preds if c is None or pred.classes[p] == c),
                key=lambda p: -pred.scores.get(p, 0.0),
            )
            used: set[int] = set()
            hits = []
            for p in ranked:
                best = max(
                    (g for g in overlap.get(p, {}) if g in gts - used),
                    key=lambda g, p=p: iou(p, g),
                    default=None,
                )
                hit = best is not None and iou(p, best) >= t
                if hit:
                    used.add(best)
                hits.append(hit)
            aps.append(average_precision(np.array(hits, dtype=bool), len(gts)))
        out[float(t)] = float(np.nanmean(aps)) if not np.isnan(aps).all() else 0.0
    return out


def label_metrics(
    truth: LabeledCloud, pred: LabeledCloud, evaluated: list[str]
) -> dict[str, float | dict[str, float]]:
    """MIoU, AP / AP50 / AP25 and class-agnostic AP50 after the label transfer."""
    labels = transfer(pred, truth)
    ious = semantic_iou(truth, pred, labels, evaluated)
    ap = instance_ap(truth, pred, labels, evaluated)
    agnostic = instance_ap(truth, pred, labels, evaluated, class_agnostic=True)
    return {
        "miou": float(np.mean(list(ious.values()))) if ious else float("nan"),
        "ap": float(np.mean([ap[float(t)] for t in IOU_THRESHOLDS])),
        "ap50": ap[0.5],
        "ap25": ap[0.25],
        "ap50_class_agnostic": agnostic[0.5],
        "labeled_fraction": float((labels != NONE).mean()),
        "iou_per_class": ious,
    }
