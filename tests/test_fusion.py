"""E57c's fusion: the straight-run rule and the reprojected seeds."""

from types import SimpleNamespace

import numpy as np

from experiments.scripts.e57c_reproject import project, seeds_of
from experiments.scripts.e57c_two_fills import BLUE, GREEN, RED, WHITE, runs_on_lines


def test_a_run_of_blue_between_white_and_green_of_one_instance_comes_back() -> None:
    category = np.full((5, 7), RED)
    category[2] = [WHITE, BLUE, BLUE, BLUE, GREEN, RED, RED]
    ids = np.ones((5, 7), dtype=int)
    assert runs_on_lines(category, ids)[2].tolist() == [
        False,
        True,
        True,
        True,
        False,
        False,
        False,
    ]
    assert runs_on_lines(category, ids).sum() == 3


def test_the_run_stops_at_a_pixel_that_is_not_blue_or_of_another_instance() -> None:
    category = np.full((5, 7), RED)
    category[2] = [WHITE, BLUE, RED, BLUE, GREEN, RED, RED]
    ids = np.ones((5, 7), dtype=int)
    assert not runs_on_lines(category, ids).any()
    category[2] = [WHITE, BLUE, BLUE, BLUE, GREEN, RED, RED]
    ids[2, 2] = 2
    assert not runs_on_lines(category, ids).any()
    ids[:] = 1
    category[2] = [WHITE, BLUE, BLUE, BLUE, WHITE, RED, RED]  # no green end
    assert not runs_on_lines(category, ids).any()


def test_a_run_may_go_along_a_diagonal() -> None:
    category = np.full((5, 5), RED)
    for i, value in enumerate([WHITE, BLUE, BLUE, GREEN]):
        category[i, i] = value
    assert runs_on_lines(category, np.zeros((5, 5), dtype=int)).sum() == 2


def test_a_point_on_the_optical_axis_projects_to_the_principal_point() -> None:
    k = np.array([[100.0, 0, 50], [0, 100, 40], [0, 0, 1]])
    uv, front = project(np.array([[0.0, 0, 2], [0, 0, -2]]), np.eye(4), k)
    assert np.allclose(uv[0], [50, 40])
    assert front.tolist() == [True, False]


def test_a_kept_point_seeds_a_dropped_pixel_of_the_same_instance_in_another_view() -> (
    None
):
    k = np.array([[1.0, 0, 2], [0, 1, 2], [0, 0, 1]])
    inputs = SimpleNamespace(
        views=SimpleNamespace(
            cam_to_world=np.stack([np.eye(4)] * 2), intrinsics=np.stack([k] * 2)
        )
    )
    category = np.full((2, 4, 4), GREEN)
    category[0, 1, 1] = (
        WHITE  # the kept point, on the optical axis, lands on pixel (2, 2) of view 1
    )
    category[1, 2, 2] = RED
    aligned = np.zeros((2, 4, 4, 3))
    aligned[0, 1, 1] = [0, 0, 1]
    ids = np.ones((2, 4, 4), dtype=int)
    seeds = seeds_of(category, ids, aligned, inputs)
    assert seeds.sum() == 1
    assert seeds[1, 2, 2]
    ids[1, 2, 2] = 2  # another instance
    assert not seeds_of(category, ids, aligned, inputs).any()
    ids[:] = 1
    category[1, 2, 2] = BLUE  # not a dropped pixel
    assert not seeds_of(category, ids, aligned, inputs).any()
