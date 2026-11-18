"""E57c's oracle: the coverage cost of a point and the greedy."""

import numpy as np

from experiments.scripts.e57c_fast import greedy_fast
from experiments.scripts.e57c_oracle_curve import greedy, removal_cost
from src.evaluation.labeled import LabeledCloud, nearest_distances


def cloud(points: np.ndarray) -> LabeledCloud:
    return LabeledCloud(
        points.astype(np.float64), np.zeros(len(points), dtype=np.int64)
    )


def test_a_points_cost_is_the_coverage_only_it_provides() -> None:
    truth = cloud(np.array([[0.0, 0, 0], [1, 0, 0], [2, 0, 0]]))
    pred = np.array(
        [[0.0, 0, 0], [1, 0, 0], [2, 0, 0], [1, 0.1, 0]]
    )  # the last nearly duplicates the middle one
    cost = removal_cost(truth, pred)
    # an end point's GT moves to the next point, 1 m = 100 cm; the middle one's to
    # its duplicate, 10 cm; the duplicate covers nothing
    assert np.allclose(cost, [100, 10, 100, 0])


def test_the_greedy_drops_flying_points_before_points_that_cover_the_gt() -> None:
    grid = np.stack(
        np.meshgrid(np.arange(5.0), np.arange(4.0), [0.0]), axis=-1
    ).reshape(-1, 3)
    truth = cloud(grid)
    pred = np.concatenate(
        [grid, grid[:2] + np.array([0, 0, 5.0])]
    )  # two points 5 m off the surface
    a = 100 * nearest_distances(truth.points, pred)
    kept = greedy(truth, pred, a, [20 / 22])[20 / 22]
    assert not kept[-2:].any()
    assert kept[:-2].all()


def test_the_incremental_greedy_equals_the_reference() -> None:
    rng = np.random.default_rng(0)
    truth = cloud(rng.random((300, 3)))
    pred = np.concatenate(
        [truth.points + 0.02 * rng.standard_normal((300, 3)), rng.random((60, 3)) * 2]
    )
    a = 100 * nearest_distances(truth.points, pred)
    levels = [0.9, 0.7, 0.5, 0.3]
    reference, fast = (
        greedy(truth, pred, a, levels),
        greedy_fast(truth, pred, a, levels),
    )
    assert all((reference[q] == fast[q]).all() for q in levels)
    assert [int(fast[q].sum()) for q in levels] == [
        round(q * len(pred)) for q in levels
    ]
