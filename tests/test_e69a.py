"""E69a's gates and oracles: the rules the plan states, on numbers worked by hand."""

from pathlib import Path

import numpy as np
import pytest
import yaml

from experiments.scripts import e69a_gate as gate
from experiments.scripts.e69a_gate import BAND, beats, gap_closed, paired_gain, parse
from experiments.scripts.e69a_oracle import snap


def test_the_gap_closed_is_zero_at_deja_view_and_one_at_the_oracle() -> None:
    assert gap_closed(1.8, 0.3, 1.8) == pytest.approx(0.0)
    assert gap_closed(1.8, 0.3, 0.3) == pytest.approx(1.0)
    assert gap_closed(1.8, 0.3, 1.05) == pytest.approx(0.5)


def test_the_paired_gain_is_the_mean_and_standard_error_of_the_differences() -> None:
    dv = np.array([2.0, 3.0, 4.0, 5.0])
    arm = np.array([1.0, 2.0, 3.0, 4.0])
    assert paired_gain(dv, arm) == pytest.approx((1.0, 0.0))
    mean, se = paired_gain(dv, np.array([1.0, 3.0, 3.0, 5.0]))
    assert mean == pytest.approx(0.5)
    assert se == pytest.approx(np.std([1.0, 0.0, 1.0, 0.0], ddof=1) / 2)


def test_a_gain_beats_only_beyond_the_band_the_error_and_the_spread() -> None:
    dv = np.array([2.0, 3.0, 4.0, 5.0])
    assert beats(dv, dv - 2 * BAND)
    assert not beats(dv, dv - BAND / 2)  # inside the tie band
    assert not beats(dv, dv - 2 * BAND, spread=3 * BAND)  # inside the seeds' spread
    noisy = np.array([-1.0, 2.0, -0.5, 1.7]) + 0.3  # mean 0.85, standard error 0.76
    assert not beats(dv, dv - noisy)


def test_a_model_label_gives_its_arm_size_and_seed() -> None:
    assert parse("seeded-s0") == ("seeded", 0, 0)
    assert parse("seeded-render-sam3-n40-s2") == ("seeded-render-sam3", 40, 2)


def test_the_snap_oracle_keeps_the_starts_within_reach_on_every_axis() -> None:
    target = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
    start = np.array(
        [[0.4, -0.4, 0.4], [0.4, 0.6, 0.0], [9.0, 0.0, 0.0], [5.0, 0.0, 0.0]]
    )
    kept = snap(start, target, 0.5)
    # the first is within 0.5 on every axis of (0, 0, 0); the second is 0.6 off on y;
    # the third is 1.0 off; the fourth is 5 off
    assert kept.tolist() == [[0.0, 0.0, 0.0]]


def test_each_arm_takes_the_rate_of_its_probe_with_the_least_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    error = {
        "lattice": {"3e-4": 0.5, "1e-3": 0.3, "3e-3": 0.4},
        "seeded": {"3e-4": 0.2, "1e-3": 0.25, "3e-3": 0.1},
    }
    for arm, by_rate in error.items():
        for rate, err in by_rate.items():
            run = tmp_path / f"lr_probe_{rate}"
            run.mkdir(exist_ok=True)
            curve = [
                {"train": {"chamfer_cells": err + 1}},
                {"train": {"chamfer_cells": err}},
            ]
            record = {"config": {"lr": float(rate)}, "curve": curve}
            path = run / f"e69a-lr_probe_{rate}-{arm}-s0.yaml"
            path.write_text(yaml.safe_dump(record))
    monkeypatch.setattr(gate, "RUNS", tmp_path)
    assert gate.best_lrs() == {"lattice": 1e-3, "seeded": 3e-3}
