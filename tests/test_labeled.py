"""Scores of a labeled per-pixel output: alignment, surface, mIoU, PQ and AP."""

from pathlib import Path

import numpy as np
import pytest
import yaml

from src.analysis.scene_eval import (
    HEADLINE,
    permute_per_view,
    shift_per_view,
    split_report,
    split_scenes,
    tiers,
)
from src.evaluation.labeled import (
    NONE,
    LabeledCloud,
    ground_truth_cloud,
    label_metrics,
    score_output,
    split_scores,
    surface_metrics,
    umeyama,
)
from src.segmentation.base import NO_INSTANCE, Segmentation

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
    assert surface["chamfer_cm"] < 1e-9 and surface["chamfer_median_cm"] < 1e-9
    assert set(surface["surface_per_class"]) == {"floor", "chair"}
    scores = labels(truth, truth.instance, truth.classes, {1: 0.9, 2: 0.8})
    assert scores["miou"] == scores["pq"] == scores["ap"] == 1.0
    assert scores["pq_th"] == scores["pq_st"] == 1.0
    assert scores["sq_st"] == scores["rq_st"] == 1.0


def test_an_outlier_counts_with_its_full_distance() -> None:
    truth = room()
    pred = truth.points.copy()
    pred[0] -= [10.0, 0, 0]  # one floor point 10 m off, beside the room
    surface = surface_metrics(pred, truth, EVALUATED)
    np.testing.assert_allclose(
        surface["accuracy_cm"], 1000 / len(pred), rtol=1e-6
    )  # no cap
    assert surface["accuracy_median_cm"] < 1e-9
    np.testing.assert_allclose(
        surface["chamfer_cm"], (surface["accuracy_cm"] + surface["completeness_cm"]) / 2
    )


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
    assert scores["pq_per_class"]["floor"]["pq"] == scores["pq_st"] == 1.0


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


def test_background_counts_in_the_all_class_means_only() -> None:
    truth = room()
    ids = truth.instance.copy()
    ids[truth.instance == 0] = NONE  # the floor is never found
    scores = labels(truth, ids, truth.classes, {1: 0.9, 2: 0.8})
    assert scores["pq_th"] == 1.0
    assert scores["miou"] == scores["pq"] == 0.5  # floor 0, chair 1
    assert scores["pq_st"] == scores["rq_st"] == 0.0


def test_class_agnostic_ap_ignores_the_object_class() -> None:
    truth = room()
    wrong = truth.classes | {1: "sofa", 2: "sofa"}  # right objects, wrong class
    scores = label_metrics(
        truth, truth.instance, wrong, {1: 0.9, 2: 0.8}, [*EVALUATED, "sofa"]
    )
    assert scores["ap"] == 0.0
    assert scores["ap_agnostic"] == 1.0


def test_class_agnostic_ap_counts_a_missed_object() -> None:
    truth = room()
    ids = truth.instance.copy()
    ids[ids == 2] = NONE
    scores = labels(truth, ids, truth.classes, {1: 0.9})
    assert scores["ap_agnostic"] == 0.5  # one of two chairs at full precision


def scene_scores(ids_gt: np.ndarray, ids_pred: np.ndarray, classes: dict) -> dict:
    """Scores of one scene on points that only differ in their labels."""
    points = np.c_[np.arange(len(ids_gt), dtype=float), np.zeros((len(ids_gt), 2))]
    truth = LabeledCloud(points, ids_gt, classes)
    return {
        "scale": 2.0,
        **surface_metrics(points, truth, EVALUATED),
        **label_metrics(truth, ids_pred, classes, {}, EVALUATED),
    }


def test_split_miou_pools_classes_and_other_scores_average_scenes() -> None:
    classes = {0: "floor", 1: "chair"}
    small = np.array([1, 1, 0, 0])  # chair IoU 0: all of it predicted floor
    large = np.r_[np.ones(96, dtype=np.int64), np.zeros(4, dtype=np.int64)]  # IoU 1
    scenes = [
        scene_scores(small, np.zeros(4, dtype=np.int64), classes),
        scene_scores(large, large, classes),
    ]
    split = split_scores(scenes)
    assert split["iou_counts"]["chair"] == {"intersection": 96, "union": 98}
    assert split["iou_per_class"]["chair"] == 96 / 98
    np.testing.assert_allclose(
        split["miou"], (96 / 98 + split["iou_per_class"]["floor"]) / 2
    )
    assert split["miou"] != (scenes[0]["miou"] + scenes[1]["miou"]) / 2
    assert split["pq"] == (scenes[0]["pq"] + scenes[1]["pq"]) / 2
    assert "scale" not in split


def test_split_averages_a_class_over_the_scenes_that_have_it() -> None:
    classes = {0: "floor", 1: "chair"}
    floor_only = np.zeros(4, dtype=np.int64)
    both = np.array([0, 0, 1, 1])
    split = split_scores(
        [
            scene_scores(floor_only, floor_only, classes),
            scene_scores(both, both, classes),
        ]
    )
    assert split["pq_per_class"]["chair"]["pq"] == 1.0  # not halved by the first scene
    assert split["ap"] == 1.0  # a scene without objects has no AP: it is left out


def test_tiers_follow_the_protocol() -> None:
    truth = room()
    surface = surface_metrics(truth.points, truth, EVALUATED)
    label = labels(truth, truth.instance, truth.classes, {1: 0.9, 2: 0.8})
    sorted_ = tiers({"scale": 1.0, "refinement_s": 2.0, **surface, **label})
    assert tuple(sorted_["headline"]) == HEADLINE
    assert {
        "accuracy_median_cm",
        "chamfer_median_cm",
        "sq",
        "rq",
        "pq_th",
        "rq_st",
    } <= (set(sorted_["supplementary"]))
    assert {"ap50", "ap25", "ap_agnostic", "refinement_s"} <= set(
        sorted_["supplementary"]
    )
    assert {
        "scale",
        "surface_per_class",
        "iou_per_class",
        "pq_per_class",
        "ap_per_class",
    } <= set(sorted_["debug"])


def test_split_report_follows_the_config() -> None:
    classes = {0: "floor", 1: "chair"}
    ids = np.array([0, 0, 1, 1])
    flat = scene_scores(ids, ids, classes) | {"refinement_s": 1.0}
    raw = {
        "variants": [{"name": "v"}],
        "aggregated": ["dev", "val"],
        "per_scene": ["dev", "stress"],
    }
    splits = {"dev": ["a"], "val": ["b"], "stress": ["c"]}
    scores = {(n, "v"): flat for n in "abc"}
    logs = {(n, "v"): {"gn_steps": 1} for n in "abc"}
    report = split_report(raw, splits, scores, logs)
    assert set(report["dev"]) == {"aggregate", "scenes"}
    assert set(report["val"]) == {"aggregate"}
    assert set(report["stress"]) == {"scenes"}
    assert report["stress"]["scenes"]["c"]["variants"]["v"]["log"] == {"gn_steps": 1}
    assert "log" not in report["dev"]["aggregate"]["v"]
    assert yaml.safe_load(yaml.safe_dump(report))["val"]["aggregate"]["v"]["headline"]


def test_a_missing_input_stops_the_run_before_it_starts(tmp_path: Path) -> None:
    dataset = yaml.safe_load(Path("configs/hypersim.yaml").read_text())
    dataset["splits"]["dev"] = {"ai_000_000": "cam_00"}
    path = tmp_path / "hypersim.yaml"
    path.write_text(yaml.safe_dump(dataset))
    raw = {
        "dataset": str(path),
        "root": str(tmp_path),
        "views": "views",
        "backbone": "backbone",
        "segmentation": "masks",
    }
    with pytest.raises(FileNotFoundError, match="ai_000_000_cam_00"):
        split_scenes(raw, ["dev"])
    folder = tmp_path / "ai_000_000_cam_00"
    folder.mkdir()
    for name in ("views", "gt_views", "backbone_views", "masks_views", "dsine_views"):
        (folder / f"{name}.npz").touch()
    assert split_scenes(raw, ["dev"]) == {"dev": ["ai_000_000_cam_00"]}


def ids_two_views() -> np.ndarray:
    """Two 6 x 8 views with three instances and an unlabeled band."""
    ids = np.full((2, 6, 8), NO_INSTANCE, dtype=np.int32)
    ids[:, :, :3], ids[:, :, 3:6], ids[:, :3, 6:] = 4, 7, 9
    return ids


def test_permuting_per_view_keeps_each_views_grouping() -> None:
    ids = ids_two_views()
    out = permute_per_view(ids, np.random.default_rng(0))
    assert ((out == NO_INSTANCE) == (ids == NO_INSTANCE)).all()
    for view, view_ids in zip(out, ids, strict=True):
        pairs = set(zip(view_ids.ravel().tolist(), view.ravel().tolist(), strict=True))
        assert len(pairs) == len(np.unique(view_ids))  # a bijection on the view's ids
        assert set(view.ravel().tolist()) == set(view_ids.ravel().tolist())


def test_permuting_per_view_scrambles_identity_across_views() -> None:
    ids = np.broadcast_to(np.arange(50, dtype=np.int32)[None, None], (2, 1, 50))
    out = permute_per_view(ids.copy(), np.random.default_rng(0))
    assert (out[0] != out[1]).mean() > 0.9


def test_shifting_moves_every_mask_by_the_same_pixels() -> None:
    ids = ids_two_views()
    out = shift_per_view(ids, 2, np.random.default_rng(0))
    for view, view_ids in zip(out, ids, strict=True):
        moved = [
            (dy, dx)
            for dy, dx in ((2, 0), (-2, 0), (0, 2), (0, -2))
            if (np.roll(view_ids, (dy, dx), axis=(0, 1)) == view)[2:-2, 2:-2].all()
        ]
        assert moved, view
        assert (view == NO_INSTANCE).sum() >= 2 * min(view.shape)
