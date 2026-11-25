"""E69f: dilation, occupancy, space carving and the percentile box."""

import numpy as np

from experiments.scripts.e69f_rung0 import carve, dilate, occupancy


def test_dilate_reaches_chebyshev_distance() -> None:
    """One voxel dilated by 2 fills the 5^3 cube around it and nothing else."""
    mask = np.zeros((9, 9, 9), dtype=bool)
    mask[4, 4, 4] = True
    out = dilate(mask, 2)
    assert out.sum() == 125
    assert out[2:7, 2:7, 2:7].all()


def test_occupancy_reads_x_y_z_and_drops_outside() -> None:
    """A point at x 3.5, y 1.2, z 0.1 sets mask[0, 1, 3]; one outside sets nothing."""
    mask = occupancy(np.array([[3.5, 1.2, 0.1], [9.0, 0.0, 0.0]]), (4, 2, 1))
    assert mask.shape == (1, 2, 4)
    assert mask.sum() == 1
    assert mask[0, 1, 3]


def test_carve_frees_voxels_in_front_of_a_wall_only() -> None:
    """A camera at the origin looks down +z at a wall at z 5: only z < 4 is free."""
    depth = np.full((1, 64, 64), 5.0)
    k = np.array([[[32.0, 0, 31.5], [0, 32.0, 31.5], [0, 0, 1]]])
    lo, eye, origin = np.array([-0.5, -0.5, 0.0]), np.eye(3)[None], np.zeros((1, 3))
    free = carve(lo, 1.0, (1, 1, 8), depth, k, eye, origin)
    assert free[:, 0, 0].tolist() == [True] * 4 + [False] * 4
