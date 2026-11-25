"""E69d: voxel centroids and the visibility test of the all-frames ground truth."""

import numpy as np

from experiments.scripts.e69d_all_frames_gt import visible, voxel_centroids


def test_voxel_centroids_are_the_means_of_each_occupied_cube() -> None:
    """Two points in one cube average, a third alone stays; empty cubes give nothing."""
    points = np.array([[0.1, 0.1, 0.1], [0.3, 0.5, 0.7], [1.5, 0.2, 0.2]])
    out = voxel_centroids(points, 1.0)
    expected = np.array([[0.2, 0.3, 0.4], [1.5, 0.2, 0.2]])
    assert out.shape == (2, 3)
    assert np.allclose(np.sort(out, axis=0), np.sort(expected, axis=0))


def test_a_point_is_visible_on_the_surface_and_not_behind_it_or_outside() -> None:
    """Camera at the origin looking down +z at a wall 2 m away (OpenCV axes)."""
    k = np.array([[100.0, 0, 50], [0, 100.0, 40], [0, 0, 1]])
    depth = np.full((80, 100), 2.0)
    depth[:, :10] = 0.0  # no surface in the left columns
    c2w = np.eye(4)
    points = np.array(
        [
            [0.0, 0.0, 2.0],  # on the wall
            [0.0, 0.0, 2.5],  # behind it: occluded
            [0.0, 0.0, -2.0],  # behind the camera
            [5.0, 0.0, 2.0],  # outside the image
            [-0.85, 0.0, 2.0],  # in a column without a surface
            [0.0, 0.0, 2.03],  # within the 2 % relative tolerance
        ]
    )
    assert visible(points, k, c2w, depth, (0.01, 0.02)).tolist() == [
        True,
        False,
        False,
        False,
        False,
        True,
    ]


def test_visibility_is_invariant_to_a_rigid_motion_of_scene_and_camera() -> None:
    """Moving the camera and the points by the same rigid motion changes nothing."""
    rng = np.random.default_rng(0)
    k = np.array([[100.0, 0, 50], [0, 100.0, 40], [0, 0, 1]])
    depth = np.full((80, 100), 2.0)
    points = np.column_stack(
        [rng.uniform(-1, 1, 50), rng.uniform(-0.8, 0.8, 50), rng.uniform(1.5, 2.5, 50)]
    )
    angle = 0.7
    motion = np.eye(4)
    motion[:3, :3] = [
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle), np.cos(angle), 0],
        [0, 0, 1],
    ]
    motion[:3, 3] = [3.0, -1.0, 0.5]
    moved = points @ motion[:3, :3].T + motion[:3, 3]
    assert np.array_equal(
        visible(points, k, np.eye(4), depth, (0.02, 0.02)),
        visible(moved, k, motion, depth, (0.02, 0.02)),
    )
