"""Score a labeled point cloud against per-view ground truth: surface, classes, objects.

Both clouds are reduced to one point per 1 cm voxel, the majority instance label of
its points, so point density and view overlap do not weigh in. Ground truth is every
pixel of the views with ground-truth depth, lifted through the ground-truth cameras
(Hypersim's per-pixel maps, ScanNet++'s rendered mesh).

The method is a black box: unposed images in, one labeled cloud out, in the first
view's camera frame at its own scale (Déjà View's convention). Nothing but that cloud
is used: `register` maps it into the ground truth's frame by one similarity found from
the two clouds alone, the usual protocol of point-map evaluations (Umeyama, then ICP:
π³, VGGT on DTU / ETH3D / 7-Scenes / NRGBD). The start is the frame convention: the
ground truth's first camera, and the ratio of the clouds' median distances from it as
scale. Point-to-point ICP that also fits the scale (Umeyama in each step) then refines
it from coarse to fine. Global registration without that start (FPFH features, RANSAC)
failed on Hypersim rooms: features are not scale invariant, and Déjà View's scale is
off by a factor of 3 to 4.

Surface (Knapitsch et al. 2017, Tanks and Temples): accuracy is the mean distance of
each predicted point to the nearest ground-truth point, completeness the other way
round; precision and recall are the fractions of either within `TOLERANCE` (2 cm,
twice the grid), and the F-score their harmonic mean. Distances are capped at `CAP`
(20 cm), as DTU's evaluation caps outliers: the means should not hinge on points
outside a window, and a nearest-neighbor query costs more the farther the point is from
the other cloud (1 m: 19 s per cloud).

Labels are transferred to the ground truth: every ground-truth point takes the
instance of the nearest predicted point within `TRANSFER`, none if there is no such
point. Labels are scored on the ground-truth points, as ScanNet's benchmark scores
mesh vertices, so a well-labeled object in the wrong place still loses its points:
refinement that moves surfaces into place shows in the label scores too.

- Semantics: IoU per class over ground-truth points, the mean over evaluated classes
  (mIoU; ScanNet, ScanNet++). Points whose ground-truth class is not evaluated are
  ignored.
- Instances: AP50 (ScanNet's protocol, also ScanNet++): per class, predictions by
  descending score are matched greedily to unmatched ground-truth instances at IoU ≥
  0.5; AP is the area under the precision envelope, averaged over classes. Walls,
  floors and ceilings are left out, as in ScanNet. A prediction that lies mostly on
  points of classes outside the evaluation is neither a hit nor a false positive.
  An object split into two ids across views fuses into two half-size masks in 3D,
  which miss IoU 0.5: AP also scores cross-view consistency.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

VOXEL = 0.01  # m, grid both clouds are reduced to
TOLERANCE = 0.02  # m, precision / recall / F-score threshold
TRANSFER = 0.05  # m, a ground-truth point takes the label of a prediction this close
CAP = 0.2  # m, distances above count as this
MIN_POINTS = 100  # voxels; smaller instances are not scored (ScanNet: 100 vertices)
MATCH_IOU = 0.5  # an instance is found at this IoU (AP50)
STRUCTURE = ("wall", "floor", "ceiling")  # no instances (ScanNet's AP leaves them out)
NONE = -1
ICP_STAGES = ((0.05, 0.5), (0.05, 0.2), (0.02, 0.05), (0.01, 0.02))  # voxel, radius (m)
ICP_ITERATIONS = 50  # per stage
ICP_POINTS = 30_000  # voxels of the output per ICP stage, at most


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
        unique, inverse = np.unique(ids, return_inverse=True)
        names = np.array([lut.get(int(i), "") for i in unique], dtype=object)
        return names[inverse.reshape(ids.shape)]


def lift(depth: np.ndarray, k: np.ndarray, cam_to_world: np.ndarray) -> np.ndarray:
    """World points (H, W, 3) of a z-depth map through a pinhole; pixel centers +0.5."""
    h, w = depth.shape
    v, u = np.mgrid[0:h, 0:w] + 0.5
    rays = np.stack([u, v, np.ones_like(u)], axis=-1) @ np.linalg.inv(k).T
    return (depth[..., None] * rays) @ cam_to_world[:3, :3].T + cam_to_world[:3, 3]


def as_cloud(points: np.ndarray) -> o3d.geometry.PointCloud:
    """Open3D cloud of (N, 3) points."""
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    return cloud


def register(
    points: np.ndarray, truth: np.ndarray, first_camera: np.ndarray
) -> tuple[float, np.ndarray, np.ndarray]:
    """Similarity (s, R, t) mapping a cloud in the first view's frame onto the truth.

    Start: the ground truth's first camera-to-world (4, 4) and the ratio of median
    distances from that camera. Then scaled point-to-point ICP over `ICP_STAGES`: both
    clouds on the stage's voxel grid, at most `ICP_POINTS` of the output's voxels
    (fixed seed).
    """
    assert points.ndim == 2 and points.shape[1] == 3, points.shape  # noqa: PLR2004
    local = (truth - first_camera[:3, 3]) @ first_camera[:3, :3]
    scale = np.median(np.linalg.norm(local, axis=1)) / np.median(
        np.linalg.norm(points, axis=1)
    )
    transform = first_camera.astype(np.float64).copy()
    transform[:3, :3] *= scale
    # once on a 2 cm grid (at the start's scale): pixels of near surfaces do not weigh
    # more, and the stages below downsample a few hundred thousand points, not millions
    source = as_cloud(points).voxel_down_sample(0.02 / scale)
    target = as_cloud(truth)
    rng = np.random.default_rng(0)
    icp = o3d.pipelines.registration
    for voxel, radius in ICP_STAGES:
        grid = np.asarray(
            copy.deepcopy(source).transform(transform).voxel_down_sample(voxel).points
        )
        grid = grid[rng.choice(len(grid), min(ICP_POINTS, len(grid)), replace=False)]
        result = icp.registration_icp(
            as_cloud(grid),
            target.voxel_down_sample(voxel),
            radius,
            np.eye(4),
            icp.TransformationEstimationPointToPoint(with_scaling=True),
            icp.ICPConvergenceCriteria(max_iteration=ICP_ITERATIONS),
        )
        transform = result.transformation @ transform
    scale = float(np.cbrt(np.linalg.det(transform[:3, :3])))
    return scale, transform[:3, :3] / scale, transform[:3, 3]


def voxelize(
    points: np.ndarray, labels: np.ndarray, size: float = VOXEL
) -> tuple[np.ndarray, np.ndarray]:
    """One point per occupied voxel (the mean) and its most frequent label."""
    assert points.shape == (len(labels), 3), (points.shape, labels.shape)
    keys = np.floor(points / size).astype(np.int64)
    keys -= keys.min(0)
    extent = keys.max(0) + 1
    flat = (keys[:, 0] * extent[1] + keys[:, 1]) * extent[2] + keys[:, 2]
    _, voxel, counts = np.unique(flat, return_inverse=True, return_counts=True)
    centers = np.zeros((len(counts), 3))
    np.add.at(centers, voxel, points)
    offset = labels.min()
    span = int(labels.max() - offset) + 1
    pairs, votes = np.unique(voxel * span + (labels - offset), return_counts=True)
    pair_voxel, pair_label = pairs // span, pairs % span + offset
    order = np.lexsort((-votes, pair_voxel))  # by voxel, most votes first
    first = np.r_[True, np.diff(pair_voxel[order]) > 0]
    majority = np.empty(len(counts), dtype=np.int64)
    majority[pair_voxel[order][first]] = pair_label[order][first]
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
    """Points (..., 3) with ids (...), mapped by the similarity and voxelized."""
    assert points.shape == (*instance_ids.shape, 3), (points.shape, instance_ids.shape)
    s, rotation, t = align
    mapped = s * points.reshape(-1, 3).astype(np.float64) @ rotation.T + t
    ids = instance_ids.reshape(-1).astype(np.int64)
    keep = np.isfinite(mapped).all(1)
    centers, majority = voxelize(
        mapped[keep], np.where(ids[keep] >= 0, ids[keep], NONE)
    )
    return LabeledCloud(centers, majority, dict(classes), dict(scores))


def surface_metrics(pred: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    """Accuracy, completeness (mean, cm), precision / recall / F-score at TOLERANCE."""
    acc, comp = (
        np.minimum(cKDTree(b).query(a, workers=-1, distance_upper_bound=CAP)[0], CAP)
        for a, b in ((pred, truth), (truth, pred))
    )
    p, r = float((acc < TOLERANCE).mean()), float((comp < TOLERANCE).mean())
    return {
        "accuracy_cm": 100 * float(acc.mean()),
        "completeness_cm": 100 * float(comp.mean()),
        "precision": p,
        "recall": r,
        "fscore": 2 * p * r / (p + r) if p + r > 0 else 0.0,
    }


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


def instance_ap50(
    truth: LabeledCloud, pred: LabeledCloud, labels: np.ndarray, evaluated: list[str]
) -> float:
    """Mean AP over classes at IoU 0.5 (ScanNet's protocol, see module)."""
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

    aps = []
    for c in things:
        gts = {g for g in gt if truth.classes[g] == c}
        ranked = sorted(
            (p for p in preds if pred.classes[p] == c),
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
            hit = best is not None and iou(p, best) >= MATCH_IOU
            if hit:
                used.add(best)
            hits.append(hit)
        aps.append(average_precision(np.array(hits, dtype=bool), len(gts)))
    return float(np.nanmean(aps)) if not np.isnan(aps).all() else 0.0


def label_metrics(
    truth: LabeledCloud, pred: LabeledCloud, evaluated: list[str]
) -> dict[str, float | dict[str, float]]:
    """MIoU and AP50 after the label transfer, and the IoU of each class."""
    labels = transfer(pred, truth)
    ious = semantic_iou(truth, pred, labels, evaluated)
    return {
        "miou": float(np.mean(list(ious.values()))) if ious else float("nan"),
        "ap50": instance_ap50(truth, pred, labels, evaluated),
        "iou_per_class": ious,
    }
