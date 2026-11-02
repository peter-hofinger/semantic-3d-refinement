"""Scores of a labeled cloud: voxel majority, surface, mIoU and instance AP."""

import numpy as np

from src.evaluation.labeled import (
    NONE,
    LabeledCloud,
    align_points,
    average_precision,
    ground_truth_cloud,
    label_metrics,
    predicted_cloud,
    surface_metrics,
    voxelize,
)

IDENTITY = (1.0, np.eye(3), np.zeros(3))


def room() -> LabeledCloud:
    """A 1 m floor plate (instance 0) with two 20 cm boxes on it (chairs 1, 2)."""
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


def test_voxels_keep_the_majority_label() -> None:
    points = np.array([[0.001, 0, 0], [0.002, 0, 0], [0.003, 0, 0], [0.5, 0, 0]])
    centers, labels = voxelize(points, np.array([4, 7, 7, 1]), 0.01)
    order = np.argsort(centers[:, 0])
    np.testing.assert_allclose(centers[order, 0], [0.002, 0.5])
    np.testing.assert_array_equal(labels[order], [7, 1])


def test_lifting_ground_truth_puts_pixels_on_the_plane() -> None:
    k = np.array([[[50.0, 0, 16], [0, 50.0, 12], [0, 0, 1]]])
    depth = np.full((1, 24, 32), 2.0, dtype=np.float32)
    ids = np.zeros((1, 24, 32), dtype=np.int32)
    ids[0, :, :16] = -1
    cloud = ground_truth_cloud(depth, ids, {0: "wall"}, k, np.eye(4)[None])
    np.testing.assert_allclose(cloud.points[:, 2], 2.0)
    assert set(cloud.instance.tolist()) == {NONE, 0}


def test_ground_truth_against_itself_is_perfect() -> None:
    truth = room()
    pred = predicted_cloud(
        truth.points[None, None], truth.instance[None, None], truth.classes,
        {1: 0.9, 2: 0.8}, IDENTITY,
    )  # fmt: skip
    surface = surface_metrics(pred.points, truth.points)
    assert surface["accuracy_cm"] < 1e-6 and surface["fscore_1cm"] == 1.0
    scores = label_metrics(truth, pred, ["floor", "chair"])
    assert scores["miou"] == 1.0 and scores["ap"] == 1.0
    assert scores["ap50_class_agnostic"] == 1.0


def test_a_shift_fails_the_tight_tolerance_only() -> None:
    truth = room().points
    surface = surface_metrics(truth + np.array([0, 0, 0.015]), truth)
    assert surface["precision_1cm"] == 0.0 and surface["fscore_2cm"] == 1.0
    np.testing.assert_allclose(surface["chamfer_cm"], 1.5, atol=1e-6)


def test_precision_envelope_ranks_by_score() -> None:
    assert average_precision(np.array([True, False]), 1) == 1.0
    assert average_precision(np.array([False, True]), 1) == 0.5
    assert average_precision(np.array([True]), 2) == 0.5


def test_an_object_split_across_views_loses_its_match() -> None:
    truth = room()
    ids = truth.instance.copy()
    x = truth.points[:, 0]
    ids[(ids == 1) & (x < 0.17)] = 3  # the same chair in thirds, as from three views
    ids[(ids == 1) & (x > 0.23)] = 4
    pred = LabeledCloud(
        truth.points,
        ids,
        truth.classes | {3: "chair", 4: "chair"},
        {2: 0.95, 1: 0.9, 3: 0.8, 4: 0.7},
    )
    scores = label_metrics(truth, pred, ["floor", "chair"])
    assert scores["miou"] == 1.0  # classes are all right
    assert scores["ap50"] == 0.5  # one of two chairs found, at full precision


def test_alignment_recovers_a_similarity_despite_outliers() -> None:
    rng = np.random.default_rng(0)
    truth = rng.uniform(-2, 2, size=(40, 50, 3))
    c, s = np.cos(0.4), np.sin(0.4)
    rotation = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    pred = (truth - [1, 0, 2]) @ rotation / 3.0  # truth = 3 R pred + t
    pred[0, 0] += 50.0  # one point far off, as behind glass
    scale, r, t = align_points(pred, truth, np.ones((40, 50), dtype=bool))
    assert np.isclose(scale, 3.0)
    np.testing.assert_allclose(r, rotation, atol=1e-9)
    np.testing.assert_allclose(t, [1, 0, 2], atol=1e-9)
