"""Scores of a labeled cloud: voxel majority, surface, mIoU and instance AP."""

import numpy as np

from src.evaluation.labeled import (
    NONE,
    LabeledCloud,
    average_precision,
    ground_truth_cloud,
    label_metrics,
    predicted_cloud,
    register,
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
    assert surface["accuracy_cm"] < 1e-6 and surface["fscore"] == 1.0
    scores = label_metrics(truth, pred, ["floor", "chair"])
    assert scores["miou"] == 1.0 and scores["ap50"] == 1.0


def test_a_shift_counts_only_beyond_the_tolerance() -> None:
    truth = room().points
    near = surface_metrics(truth + np.array([0, 0, 0.015]), truth)
    far = surface_metrics(truth + np.array([0, 0, 0.025]), truth)
    assert near["fscore"] == 1.0 and far["fscore"] == 0.0
    np.testing.assert_allclose(near["accuracy_cm"], 1.5, atol=1e-6)


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


def rotation_z(angle: float) -> np.ndarray:
    """Rotation by angle (rad) about z."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def test_registration_recovers_a_similarity_from_the_clouds_alone() -> None:
    rng = np.random.default_rng(0)
    n = 15_000
    walls = [  # an open room, 4 x 5 x 2.5 m, and a table: no symmetry ICP could slip
        np.c_[rng.uniform(0, 4, n), rng.uniform(0, 5, n), np.zeros(n)],
        np.c_[np.zeros(n), rng.uniform(0, 5, n), rng.uniform(0, 2.5, n)],
        np.c_[rng.uniform(0, 4, n), np.full(n, 5.0), rng.uniform(0, 2.5, n)],
        np.c_[rng.uniform(1, 2, n), rng.uniform(1, 1.6, n), np.full(n, 0.75)],
    ]
    truth = np.concatenate(walls)
    first = np.eye(4)
    first[:3, :3] = rotation_z(0.7) @ rotation_z(0.3)[[2, 0, 1]]  # camera looking in
    first[:3, 3] = [3.0, 1.0, 1.5]
    # The output, in the first view's frame at a third of the scale, with a pose error
    # of 2 degrees and 5 cm in that view, which registration must absorb.
    pose_error = rotation_z(np.radians(2.0))
    pred = ((truth - first[:3, 3] - [0.05, 0, 0]) @ first[:3, :3] @ pose_error) / 3.0
    scale, r, t = register(pred, truth, first)
    mapped = scale * pred @ r.T + t
    assert np.isclose(scale, 3.0, rtol=1e-3)
    assert np.abs(mapped - truth).max() < 1e-3
