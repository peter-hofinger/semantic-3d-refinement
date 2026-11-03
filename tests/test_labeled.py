"""Scores of a labeled per-pixel output: alignment, surface, mIoU, PQ and AP."""

import numpy as np

from src.evaluation.labeled import (
    NONE,
    LabeledCloud,
    ground_truth_cloud,
    label_metrics,
    score_output,
    surface_metrics,
    umeyama,
)
from src.segmentation.base import Segmentation

EVALUATED = ["chair", "floor"]


def room() -> LabeledCloud:
    """A 1 m floor plate (instance 0) with two 20 cm plates on it (chairs 1, 2)."""
    g = np.arange(0, 1, 0.01) + 0.005
    x, y = np.meshgrid(g, g)
    floor = np.c_[x.ravel(), y.ravel(), np.zeros(x.size)]
    s = np.arange(0, 0.2, 0.01) + 0.005
    a, b = np.meshgrid(s, s)
    top = np.c_[a.ravel(), b.ravel(), np.full(a.size, 0.205)]
    boxes = [top + np.array([0.1, 0.1, 0]), top + np.array([0.6, 0.6, 0])]
    points = np.r_[floor, *boxes]
    ids = np.r_[np.zeros(len(floor)), np.ones(len(top)), np.full(len(top), 2)]
    return LabeledCloud(
        points, ids.astype(np.int64), {0: "floor", 1: "chair", 2: "chair"}
    )


def labels(truth: LabeledCloud, ids: np.ndarray, classes: dict, scores: dict) -> dict:
    """Label scores of predicted ids on the truth's points."""
    return label_metrics(truth, ids, classes, scores, EVALUATED)


def test_lifting_ground_truth_puts_pixels_on_the_plane() -> None:
    k = np.array([[[50.0, 0, 16], [0, 50.0, 12], [0, 0, 1]]])
    depth = np.full((1, 24, 32), 2.0, dtype=np.float32)
    depth[0, 0] = 0  # no ground truth on the first row
    ids = np.zeros((1, 24, 32), dtype=np.int32)
    ids[0, :, :16] = -1
    cloud = ground_truth_cloud(depth, ids, {0: "wall"}, k, np.eye(4)[None])
    np.testing.assert_allclose(cloud.points[:, 2], 2.0)
    assert len(cloud.points) == 23 * 32 and cloud.pixels.sum() == len(cloud.points)
    assert set(cloud.instance.tolist()) == {NONE, 0}


def rotation_z(angle: float) -> np.ndarray:
    """Rotation by angle (rad) about z."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def test_umeyama_recovers_a_similarity() -> None:
    rng = np.random.default_rng(0)
    source = rng.normal(size=(500, 3))
    r = rotation_z(0.7) @ rotation_z(0.3)[[2, 0, 1]]
    target = 3.0 * source @ r.T + [1.0, -2.0, 0.5]
    s, rotation, t = umeyama(source, target)
    np.testing.assert_allclose(s, 3.0)
    np.testing.assert_allclose(rotation, r, atol=1e-12)
    np.testing.assert_allclose(t, [1.0, -2.0, 0.5], atol=1e-12)


def test_ground_truth_against_itself_is_perfect() -> None:
    truth = room()
    surface = surface_metrics(truth.points, truth, EVALUATED)
    assert surface["accuracy_cm"] < 1e-9 and surface["completeness_cm"] < 1e-9
    assert set(surface["surface_per_class"]) == {"floor", "chair"}
    scores = labels(truth, truth.instance, truth.classes, {1: 0.9, 2: 0.8})
    assert scores["miou"] == scores["pq"] == scores["ap"] == 1.0
    assert scores["pq_per_class"]["floor"]["pq"] == 1.0  # background: per class only


def test_an_outlier_counts_with_its_full_distance() -> None:
    truth = room()
    pred = truth.points.copy()
    pred[0] -= [10.0, 0, 0]  # one floor point 10 m off, beside the room
    surface = surface_metrics(pred, truth, EVALUATED)
    np.testing.assert_allclose(
        surface["accuracy_cm"], 1000 / len(pred), rtol=1e-6
    )  # no cap
    assert surface["accuracy_median_cm"] < 1e-9


def test_a_missing_point_costs_completeness_only() -> None:
    truth = room()
    pred = truth.points.copy()
    pred[truth.instance == 1] = np.nan  # no output where the first chair is
    surface = surface_metrics(pred, truth, EVALUATED)
    assert surface["accuracy_cm"] < 1e-9 and surface["completeness_cm"] > 0
    per_class = surface["surface_per_class"]
    assert per_class["floor"]["completeness_cm"] < 1e-9
    assert per_class["chair"]["completeness_cm"] > 0


def test_a_shifted_object_costs_its_own_class_only() -> None:
    truth = room()
    pred = truth.points.copy()
    pred[truth.instance == 1] += [0, 0, 0.5]  # one chair floats 50 cm too high
    per_class = surface_metrics(pred, truth, EVALUATED)["surface_per_class"]
    assert per_class["floor"]["accuracy_cm"] < 1e-9
    assert per_class["chair"]["accuracy_cm"] > 10


def test_an_object_split_across_views_is_two_segments() -> None:
    truth = room()
    ids = truth.instance.copy()
    x = truth.points[:, 0]
    ids[(ids == 1) & (x < 0.17)] = 3  # the same chair in thirds, as from three views
    ids[(ids == 1) & (x > 0.23)] = 4
    scores = labels(
        truth, ids, truth.classes | {3: "chair", 4: "chair"},
        {2: 0.95, 1: 0.9, 3: 0.8, 4: 0.7},
    )  # fmt: skip
    assert scores["miou"] == 1.0  # classes are all right
    assert scores["ap50"] == 0.5  # one of two chairs found, at full precision
    pq = scores["pq_per_class"]["chair"]
    assert pq["sq"] == 1.0 and pq["rq"] == 1 / (1 + 3 / 2 + 1 / 2)  # 1 TP, 3 FP, 1 FN


def test_background_pieces_form_one_segment() -> None:
    truth = room()
    ids = truth.instance.copy()
    floor = ids == 0
    ids[floor & (truth.points[:, 0] < 0.5)] = 7  # the floor found as two masks
    scores = labels(truth, ids, truth.classes | {7: "floor"}, {1: 0.9, 2: 0.8})
    assert scores["pq_per_class"]["floor"]["pq"] == 1.0


def test_a_prediction_on_void_is_no_false_positive() -> None:
    truth = room()
    unlabeled = truth.instance.copy()
    unlabeled[truth.instance == 2] = NONE  # the second chair has no ground truth
    truth = LabeledCloud(truth.points, unlabeled, {0: "floor", 1: "chair"})
    scores = labels(truth, room().instance, room().classes, {1: 0.9, 2: 0.8})
    assert scores["pq"] == 1.0 and scores["ap"] == 1.0


def test_a_lower_scored_duplicate_is_a_false_positive() -> None:
    truth = room()
    ids = truth.instance.copy()
    chair = np.flatnonzero(ids == 1)
    ids[chair[: len(chair) // 4]] = 5  # a quarter of chair 1, scored below everything
    ap = labels(truth, ids, truth.classes | {5: "chair"}, {1: 0.9, 2: 0.8, 5: 0.1})
    assert ap["ap50"] == 1.0  # the false positive ranks last: precision envelope full
    ap = labels(truth, ids, truth.classes | {5: "chair"}, {1: 0.9, 2: 0.8, 5: 0.95})
    assert ap["ap50"] < 1.0


def test_score_output_aligns_the_pixels_first() -> None:
    truth = room()
    n = len(truth.points)
    truth.pixels = np.ones((1, 1, n), dtype=bool)
    r = rotation_z(0.4)
    pred = ((truth.points - [0.5, 0.2, 0]) @ r) / 3.0  # other frame, a third the size
    seg = Segmentation(truth.instance[None, None], truth.classes, {1: 0.9, 2: 0.8})
    scores = score_output(truth, pred[None, None], seg, EVALUATED)
    np.testing.assert_allclose(scores["scale"], 3.0)
    assert scores["accuracy_cm"] < 1e-9 and scores["completeness_cm"] < 1e-9
    assert scores["pq"] == 1.0
