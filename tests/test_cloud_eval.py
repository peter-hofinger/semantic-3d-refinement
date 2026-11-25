import numpy as np

from src.analysis.cloud_eval import apply_alignment, cloud_scores, transfer_ids
from src.evaluation.labeled import LabeledCloud, surface_metrics, umeyama
from src.segmentation.base import Segmentation

EVALUATED = ["floor"]


def floor() -> LabeledCloud:
    """A 1 m floor plate, one instance."""
    g = np.arange(0, 1, 0.02) + 0.01
    x, y = np.meshgrid(g, g)
    return LabeledCloud(
        np.c_[x.ravel(), y.ravel(), np.zeros(x.size)],
        np.zeros(x.size, dtype=np.int64),
        {0: "floor"},
    )


def labels(truth: LabeledCloud) -> Segmentation:
    """The truth's own instance as the labelling (the points' shape is irrelevant)."""
    return Segmentation(
        np.zeros((1, 1, len(truth.points)), dtype=np.int32), {0: "floor"}
    )


def test_a_shifted_copy_has_the_shift_as_accuracy_and_completeness() -> None:
    truth = floor()
    out = cloud_scores(
        truth.points + np.array([0, 0, 0.03]),
        truth.instance,
        truth,
        labels(truth),
        EVALUATED,
    )
    assert abs(out["accuracy_cm"] - 3) < 1e-9
    assert abs(out["completeness_cm"] - 3) < 1e-9
    assert abs(out["chamfer_cm"] - 3) < 1e-9
    assert out["miou"] == 1.0


def test_a_cloud_of_the_pixels_scores_as_surface_metrics_does() -> None:
    truth = floor()
    pred = truth.points + np.random.default_rng(0).normal(0, 0.01, truth.points.shape)
    ours = cloud_scores(pred, truth.instance, truth, labels(truth), EVALUATED)
    theirs = surface_metrics(pred, truth, EVALUATED)
    for key in ("accuracy_cm", "completeness_cm", "chamfer_cm"):
        assert abs(ours[key] - theirs[key]) < 1e-9


def test_a_cloud_may_have_fewer_points_than_the_truth() -> None:
    truth = floor()
    every_fourth = truth.points[::4]
    out = cloud_scores(
        every_fourth, truth.instance[::4], truth, labels(truth), EVALUATED
    )
    assert out["accuracy_cm"] == 0
    assert out["completeness_cm"] > 0


def test_the_alignment_maps_the_source_onto_the_target() -> None:
    rng = np.random.default_rng(0)
    source = rng.normal(size=(50, 3))
    angle = 0.4
    rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1],
        ]
    )
    target = 2.5 * source @ rotation.T + [1, 2, 3]
    s, r, t = umeyama(source, target)
    align = np.eye(4)
    align[:3, :3], align[:3, 3] = s * r, t
    np.testing.assert_allclose(apply_alignment(source, align), target, atol=1e-9)


def test_ids_come_from_the_nearest_source_point() -> None:
    source = np.array([[0.0, 0, 0], [10, 0, 0]])
    points = np.array([[1.0, 0, 0], [9, 0, 0], [6, 0, 0]])
    assert transfer_ids(points, source, np.array([7, 8])).tolist() == [7, 8, 8]
