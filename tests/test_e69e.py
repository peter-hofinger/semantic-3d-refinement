"""E69e rung 0: frames, coverage, halves and the regional score."""

import numpy as np

from experiments.scripts.e69e_rung0 import (
    covered,
    halves,
    score,
    to_grid_frame,
    to_world,
)
from experiments.scripts.e69e_rung0_checks import child_means


def test_grid_frame_inverts_the_world_map() -> None:
    """to_world then to_grid_frame (with scale 1) returns the points."""
    rng = np.random.default_rng(0)
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    rotation = q * np.sign(np.linalg.det(q))
    similarity = (2.5, rotation, np.array([1.0, -2.0, 0.5]))
    points = rng.normal(size=(20, 3))
    back = to_grid_frame(to_world(points, similarity), similarity, 1.0)
    assert np.allclose(back, points)
    assert np.allclose(
        to_grid_frame(to_world(points, similarity), similarity, 2.0), points / 2
    )


def test_covered_reads_the_mask_in_z_y_x_order_and_rejects_outside() -> None:
    """A mask (Z, Y, X) of 2 x 3 x 4 with one True voxel at x 3, y 1, z 0."""
    mask = np.zeros((2, 3, 4), dtype=bool)
    mask[0, 1, 3] = True
    cells = np.array(
        [[3.5, 1.2, 0.9], [1.0, 1.0, 0.0], [4.1, 1.0, 0.0], [-0.1, 1.0, 0.0]]
    )
    assert covered(cells, mask).tolist() == [True, False, False, False]


def test_halves_cut_the_longest_horizontal_axis_at_its_median() -> None:
    """A 6 m x 2 m room, 3 m high: cut on x at its median, never on the up axis."""
    rng = np.random.default_rng(1)
    points = rng.uniform([0, 0, 0], [6, 2, 3], size=(10001, 3))
    axis, cut = halves(points)
    assert axis == 0
    assert cut == np.median(points[:, 0])


def test_score_in_a_region_equals_scoring_the_cropped_clouds() -> None:
    """Scoring half B equals scoring the cropped clouds; a perfect cloud scores 0."""
    rng = np.random.default_rng(2)
    truth = rng.uniform(0, 1, size=(500, 3))
    seen = rng.random(500) < 0.8
    cloud = truth + rng.normal(scale=0.01, size=truth.shape)
    region = (0, 0.5)
    b, keep = cloud[:, 0] >= 0.5, truth[:, 0] >= 0.5
    regional = score(cloud, truth, seen, voxel=False, region=region)
    cropped = score(cloud[b], truth[keep], seen[keep], voxel=False)
    assert regional == cropped
    assert score(truth, truth, seen, voxel=False)["chamfer"] == 0


def test_child_means_average_the_points_of_each_child_and_count_the_outside() -> None:
    """Split 2 of one cell: two points share child 0, one is alone, one is outside."""
    cells = np.array(
        [[0.1, 0.1, 0.1], [0.3, 0.2, 0.4], [0.7, 0.1, 0.1], [1.5, 0.1, 0.1]]
    )
    means, outside = child_means(cells, 2, (1, 1, 1))
    expected = np.array([[0.2, 0.15, 0.25], [0.7, 0.1, 0.1]])
    assert outside == 1
    assert np.allclose(np.sort(means, axis=0), np.sort(expected, axis=0))
