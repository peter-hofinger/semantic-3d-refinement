"""E69c's gates, split choice and damage scorer, on numbers worked by hand."""

from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from scipy.spatial import cKDTree

from experiments.scripts import e69c_gate as gate
from experiments.scripts.e69c_oracle import Frame, children_cloud, surface_chamfer
from src.analysis.cloud_eval import cloud_scores
from src.evaluation.labeled import LabeledCloud
from src.refinement.generator import child_corner, child_targets
from src.segmentation.base import Segmentation


def report(dv: list[float], oracle: dict[int, list[float]], split: str = "dev") -> dict:
    """A cloud_eval report of three scenes: Deja View, an oracle per split, dv +0.05."""
    scenes = {}
    for i, name in enumerate(["a", "b", "c"]):
        row = {"Déjà View": {"chamfer_cm": dv[i]}}
        for k, values in oracle.items():
            row[f"oracle-ovoxel-s{k}"] = {"chamfer_cm": values[i]}
            row[f"dv-ovoxel-s{k}"] = {"chamfer_cm": dv[i] + 0.05}
        scenes[name] = row
    return {split: {"scenes": scenes}}


def test_the_split_is_the_smallest_whose_oracle_is_half_or_less_on_each_dev_scene() -> (
    None
):
    dv = [2.0, 4.0, 8.0]
    # split 2 is over half on the third scene (4.5 > 4.0); split 3 is not
    scores = report(dv, {2: [0.5, 1.0, 4.5], 3: [0.9, 2.0, 4.0], 4: [0.3, 0.6, 1.0]})
    assert gate.choose_split(scores) == 3
    over = [1.1, 1.0, 1.0]  # split 2 and 3 are over half on the first scene
    none = report(dv, {2: over, 3: over, 4: [4.5, 1.0, 1.0]})
    assert gate.choose_split(none) is None
    half = report(dv, {2: [1.0, 2.0, 4.0], 3: over, 4: over})
    assert gate.choose_split(half) == 2  # exactly half passes


def test_rung_zero_passes_when_deja_view_costs_nothing_in_the_representation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(gate, "RUNS", tmp_path)
    dv = [2.0, 4.0, 8.0]
    oracle = {2: [0.5, 1.0, 2.0], 3: [0.4, 0.8, 1.6], 4: [0.3, 0.6, 1.2]}
    scores = report(dv, oracle) | {"val": report(dv, oracle)["dev"]}
    ok, gap = gate.rung0(scores)["split 2"]
    assert ok and gap == pytest.approx(0.05)
    assert yaml.safe_load((tmp_path / "split.yaml").read_text()) == {"split": 2}
    worse = report(dv, oracle)
    for scene in worse["dev"]["scenes"].values():
        scene["dv-ovoxel-s2"]["chamfer_cm"] += 0.2  # 0.25 cm above Deja View
    assert not gate.rung0(worse | {"val": scores["val"]})["split 2"][0]


def write_model(root: Path, rung: str, model: str, **record: object) -> None:
    folder = root / rung
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"e69c-{rung}-{model}.yaml").write_text(yaml.safe_dump(record))


def test_rung_three_b_asks_for_half_of_the_gap_to_the_oracle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(gate, "RUNS", tmp_path)
    (tmp_path / "split.yaml").write_text(yaml.safe_dump({"split": 2}))
    for model in ("ovoxel-s0", "ovoxel-dino-s0"):
        write_model(tmp_path, "overfit1", model)
    scene = {
        "Déjà View": {"chamfer_cm": 1.8},
        "oracle-ovoxel-s2": {"chamfer_cm": 0.4},
        "e69c-x": {"chamfer_cm": 1.0},
        "ovoxel-s0": {"chamfer_cm": 1.1},  # closes (1.8 - 1.1) / 1.4 = 0.5 exactly
        "ovoxel-dino-s0": {"chamfer_cm": 1.11},  # 0.493
    }
    got = gate.overfit1({"dev": {"scenes": {gate.BEDROOM: scene}}}, "overfit1")
    assert got["ovoxel"] == (True, pytest.approx(0.5))
    assert got["ovoxel-dino"] == (False, pytest.approx(0.4929, abs=1e-3))


@pytest.mark.parametrize(
    ("case", "passes"),
    [
        ((0.53, 0.97, 0.97), True),
        ((0.55, 0.97, 0.97), False),  # 0.15 above the oracle; the slack is 0.14
        ((0.53, 0.94, 0.97), False),
        ((0.53, 0.97, 0.949), False),
        ((0.539, 0.95, 0.95), True),  # just inside the slack, scores exactly 0.95
    ],
)
def test_rung_three_a_needs_the_chamfer_and_both_child_scores(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    case: tuple[float, float, float],
    passes: bool,  # noqa: FBT001 - a parametrized expectation
) -> None:
    chamfer, precision, recall = case
    monkeypatch.setattr(gate, "RUNS", tmp_path)
    (tmp_path / "split.yaml").write_text(yaml.safe_dump({"split": 2}))
    write_model(
        tmp_path,
        "overfit1a",
        "latent-s0",
        children={gate.BEDROOM: {"precision": precision, "recall": recall}},
    )
    # the gap is 1.4 cm, so 10% is 0.14 cm, above the 0.1 cm band
    scene = {
        "Déjà View": {"chamfer_cm": 1.8},
        "oracle-ovoxel-s2": {"chamfer_cm": 0.4},
        "latent-s0": {"chamfer_cm": chamfer},
    }
    got = gate.overfit1a({"dev": {"scenes": {gate.BEDROOM: scene}}})
    assert got["latent"][0] is passes


def test_each_arm_takes_the_rate_of_its_probe_with_the_least_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    error = {
        "ovoxel": {"3e-4": 0.5, "1e-3": 0.3, "3e-3": 0.4},
        "ovoxel-dino": {"3e-4": 0.2, "1e-3": 0.25, "3e-3": 0.1},
    }
    for arm, by_rate in error.items():
        for rate, err in by_rate.items():
            curve = [
                {"train": {"chamfer_cells": err + 1}},
                {"train": {"chamfer_cells": err}},
            ]
            run = f"lr_probe_{rate}"
            write_model(
                tmp_path, run, f"{arm}-s0", config={"lr": float(rate)}, curve=curve
            )
    monkeypatch.setattr(gate, "RUNS", tmp_path)
    assert gate.best_lrs() == {"ovoxel": 1e-3, "ovoxel-dino": 3e-3}


def test_the_density_filter_is_tuned_on_dev_and_read_on_val() -> None:
    def scores(values: dict[float, list[float]]) -> dict:
        return {f"density q{q}": {"chamfer_cm": v} for q, v in values.items()}

    dev = {
        "a": scores({0.5: 5.0, 0.7: 3.0, 0.9: 4.0}),
        "b": scores({0.5: 5.0, 0.7: 3.0, 0.9: 4.0}),
    }
    val = {"a": scores({0.5: 1.0, 0.7: 2.0, 0.9: 3.0})}
    assert gate.tuned_density({"dev": {"scenes": dev}, "val": {"scenes": val}}) == (
        0.7,
        pytest.approx(2.0),
    )


def floor() -> LabeledCloud:
    """A 1 m floor plate, one instance."""
    g = np.arange(0, 1, 0.02) + 0.01
    x, y = np.meshgrid(g, g)
    return LabeledCloud(
        np.c_[x.ravel(), y.ravel(), np.zeros(x.size)],
        np.zeros(x.size, dtype=np.int64),
        {0: "floor"},
    )


def test_the_damage_scorer_is_the_surface_half_of_cloud_scores() -> None:
    truth = floor()
    pred = truth.points[::3] + np.random.default_rng(0).normal(
        0, 0.01, truth.points[::3].shape
    )
    labels = Segmentation(
        np.zeros((1, 1, len(truth.points)), dtype=np.int32), {0: "floor"}
    )
    full = cloud_scores(
        pred, np.zeros(len(pred), dtype=np.int64), truth, labels, ["floor"]
    )
    tree = cKDTree(truth.points, balanced_tree=False, compact_nodes=False)
    ours = surface_chamfer(pred, truth.points, tree)
    for key in ("accuracy_cm", "completeness_cm", "chamfer_cm"):
        assert ours[key] == pytest.approx(full[key])


def test_the_oracle_cloud_puts_each_vertex_at_its_target_in_the_scenes_frame() -> None:
    voxel = torch.tensor([[0, 0, 0], [1, 0, 0]])
    points = torch.tensor([[0.1, 0.1, 0.1], [0.3, 0.1, 0.1], [1.75, 0.75, 0.75]])
    occupied, vertex = child_targets(voxel, points, 2, (2, 1, 1))
    frame = Frame(torch.tensor([10.0, 20.0, 30.0]), 0.5, 2.0)
    cloud = children_cloud(voxel, occupied, vertex, 2, frame)
    # the first child's mean is (0.2, 0.1, 0.1) cells: (10 + 0.2 * 0.5) * 2 = 20.2 ...
    expected = np.array([[10.1, 20.05, 30.05], [10.875, 20.375, 30.375]]) * 2
    assert cloud == pytest.approx(expected)
    assert child_corner(2).shape == (8, 3)
