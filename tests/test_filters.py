"""Point filters: the quantile cut and each score's defining property."""

import numpy as np

from src.analysis.filters import (
    both,
    depth_edge,
    facing,
    keep_top,
    log_gradient,
    neighbor_distance,
    pixel_footprint,
    ranks,
    voxel_counts,
    voxel_means,
    voxel_sums,
)


def test_the_cut_keeps_the_top_fraction_and_breaks_ties_without_preference() -> None:
    score = np.arange(100.0).reshape(1, 10, 10)
    keep = keep_top(ranks(score), 0.3)
    assert keep.sum() == 30
    assert (score[keep] >= 70).all()
    assert keep_top(ranks(score), 1.0).all()
    assert not keep_top(ranks(score), 0.0).any()
    flat = np.zeros((1, 10, 10))
    kept = keep_top(ranks(flat), 0.5)
    assert kept.sum() == 50
    np.testing.assert_array_equal(kept, keep_top(ranks(flat), 0.5))  # fixed seed
    assert 10 < kept[0, :5].sum() < 40  # not the first pixels in memory order


def test_both_keeps_what_each_score_ranks_above_one_shared_level() -> None:
    rng = np.random.default_rng(1)
    a, b = rng.random((1, 20, 20)), rng.random((1, 20, 20))
    for q in (0.2, 0.5, 0.8):
        kept = keep_top(ranks(both(a, b)), q)
        level = np.minimum(ranks(a), ranks(b))[kept].min()
        np.testing.assert_array_equal(kept, (ranks(a) >= level) & (ranks(b) >= level))


def test_log_gradient_is_exact_on_exponential_depth_and_scale_free() -> None:
    v, u = np.mgrid[0:8, 0:10].astype(np.float64)
    depth = np.exp(0.03 * u - 0.04 * v)[None]
    np.testing.assert_allclose(log_gradient(depth), 0.05)
    np.testing.assert_allclose(log_gradient(7 * depth), log_gradient(depth))


def test_depth_edge_is_the_log_step_beside_it_and_zero_away_from_it() -> None:
    depth = np.ones((1, 6, 8))
    depth[0, :, 4:] = 3.0
    edge = depth_edge(depth)
    np.testing.assert_allclose(edge[0, :, 3:5], np.log(3.0))
    assert (edge[0, :, :3] == 0).all()
    assert (edge[0, :, 5:] == 0).all()
    np.testing.assert_allclose(depth_edge(5 * depth), edge)


def test_neighbor_distance_singles_out_a_stray_point_and_scales_with_it() -> None:
    g = np.arange(10.0)
    x, y = np.meshgrid(g, g)
    points = np.c_[x.ravel(), y.ravel(), np.zeros(x.size)]
    points = np.r_[points, [[4.5, 4.5, 6.0]]]  # a flying point above the plane
    d = neighbor_distance(points, 4)
    assert d.argmax() == len(points) - 1
    np.testing.assert_allclose(d[11], 1.0)  # an inner grid point: four at distance 1
    np.testing.assert_allclose(neighbor_distance(3 * points, 4), 3 * d)


def test_voxel_counts_are_the_points_sharing_a_cube() -> None:
    points = np.array([[0.1, 0.1, 0.1], [0.9, 0.2, 0.5], [1.5, 0, 0], [-0.1, 0, 0]])
    np.testing.assert_array_equal(voxel_counts(points, 1.0), [2, 2, 1, 1])
    np.testing.assert_array_equal(voxel_counts(points, 2.0), [3, 3, 3, 1])


def test_pixel_footprint_is_depth_over_focal_length() -> None:
    k = np.tile(np.diag([100.0, 100.0, 1.0]), (2, 1, 1))
    assert pixel_footprint(np.full((2, 3, 4), 2.0), k) == 0.02


def test_facing_is_one_along_the_ray_and_zero_across_it() -> None:
    rays = np.array([[0.0, 0, 2], [0, 0, 1], [1, 1, 0]])
    normals = np.array([[0.0, 0, -1], [1, 0, 0], [-1, -1, 0]])
    np.testing.assert_allclose(facing(normals, rays), [1, 0, 1], atol=1e-12)


def test_voxel_sums_add_a_value_per_cube_and_group() -> None:
    points = np.array([[0.1, 0.1, 0.1], [0.9, 0.2, 0.5], [0.5, 0.5, 0.5], [1.5, 0, 0]])
    values = np.array([1.0, 2.0, 4.0, 8.0])
    np.testing.assert_array_equal(voxel_sums(points, 1.0, values), [7, 7, 7, 8])
    groups = np.array([0, 0, 1, 0])
    np.testing.assert_array_equal(voxel_sums(points, 1.0, values, groups), [3, 3, 4, 8])
    np.testing.assert_array_equal(
        voxel_sums(points, 1.0, np.ones(4)), voxel_counts(points, 1.0)
    )


def test_voxel_means_average_a_cube_and_keep_a_lone_point() -> None:
    points = np.array([[0.1, 0.1, 0.1], [0.9, 0.3, 0.5], [1.5, 0.2, 0.7]])
    fused = voxel_means(points, 1.0)
    np.testing.assert_allclose(fused[:2], [[0.5, 0.2, 0.3]] * 2)
    np.testing.assert_array_equal(fused[2], points[2])
    apart = voxel_means(points, 1.0, np.array([0, 1, 0]))
    np.testing.assert_array_equal(apart, points)  # two objects in a cube stay apart
