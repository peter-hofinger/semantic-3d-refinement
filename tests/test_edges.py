import numpy as np

from src.analysis.edges import DEPTH_JUMP, across_edge, depth_edges, slopes

SHAPE = (24, 32)
EVERYWHERE = np.ones(SHAPE, dtype=bool)


def road() -> np.ndarray:
    """Depth of a ground 1.5 m below a camera with f = 100 px, horizon 2 px above."""
    rows = np.arange(SHAPE[0])[:, None] + 0.5
    return np.broadcast_to(100 * 1.5 / (rows + 2), SHAPE).copy()


def step_rule(depth: np.ndarray) -> np.ndarray:
    """E35's rule: a relative depth step above DEPTH_JUMP to a 4-neighbour."""
    edge = np.zeros(depth.shape, dtype=bool)
    v = np.abs(np.diff(depth, axis=0)) / np.minimum(depth[1:], depth[:-1]) > DEPTH_JUMP
    h = np.abs(np.diff(depth, axis=1)) / np.minimum(depth[:, 1:], depth[:, :-1])
    edge[1:] |= v
    edge[:-1] |= v
    edge[:, 1:] |= h > DEPTH_JUMP
    edge[:, :-1] |= h > DEPTH_JUMP
    return edge


def test_a_grazing_plane_has_no_edges_though_its_depth_steps() -> None:
    depth = road()
    assert step_rule(depth)[:10].all()
    assert not depth_edges(depth, EVERYWHERE).any()
    assert np.isnan(across_edge(depth, EVERYWHERE, 3)).all()


def test_slope_of_a_plane_is_its_inverse_depth_gradient() -> None:
    cols, rows = np.meshgrid(np.arange(32.0), np.arange(24.0))
    inverse = 0.2 + 0.003 * cols - 0.002 * rows
    gx, gy = slopes(inverse)
    np.testing.assert_allclose(gx, 0.003, rtol=1e-9)
    np.testing.assert_allclose(gy, -0.002, rtol=1e-9)


def test_fronto_parallel_surfaces_give_the_step_rule() -> None:
    rng = np.random.default_rng(0)
    depth = np.repeat(np.repeat(rng.uniform(2, 3, (6, 8)), 4, axis=0), 4, axis=1)
    np.testing.assert_array_equal(depth_edges(depth, EVERYWHERE), step_rule(depth))


def test_a_step_marks_the_two_columns_at_it_and_finds_the_other_side() -> None:
    depth = np.full(SHAPE, 4.0)
    depth[:, 16:] = 6.0
    edge = depth_edges(depth, EVERYWHERE)
    assert edge[:, 15:17].all()
    assert not edge[:, :15].any()
    assert not edge[:, 17:].any()
    other = across_edge(depth, EVERYWHERE, 3)
    np.testing.assert_allclose(other[:, 13:16], 6.0)
    np.testing.assert_allclose(other[:, 16:19], 4.0)
    assert np.isnan(other[:, :13]).all()
    assert np.isnan(other[:, 19:]).all()


def test_a_box_on_a_grazing_plane_is_outlined_and_nothing_else() -> None:
    depth = road()
    box = np.zeros(SHAPE, dtype=bool)
    box[6:12, 10:20] = True
    depth[box] = depth[11, 0]
    edge = depth_edges(depth, EVERYWHERE)
    outline = np.zeros(SHAPE, dtype=bool)
    outline[5:12, 9:21] = True
    outline[7:11, 11:19] = False
    assert not edge[~outline].any()
    assert edge[5:7, 10:20].all()
    assert edge[6:11, 9:11].all()
    assert edge[6:11, 19:21].all()
    assert not edge[11:13].any()


def test_a_pole_one_pixel_wide_is_an_edge_with_its_neighbours() -> None:
    depth = road()
    depth[:, 15] = 3.0
    edge = depth_edges(depth, EVERYWHERE)
    assert edge[:, 14:17].all()
    assert not edge[:, :14].any()
    assert not edge[:, 17:].any()
    other = across_edge(depth, EVERYWHERE, 3)
    np.testing.assert_allclose(other[:, 14], 3.0)
    np.testing.assert_allclose(other[:, 16], 3.0)
    np.testing.assert_allclose(other[:, 15], depth[:, 14])


def test_pixels_without_ground_truth_are_never_edges() -> None:
    depth = np.full(SHAPE, 4.0)
    depth[:, 16:] = 6.0
    valid = EVERYWHERE.copy()
    valid[:, 16] = False
    depth[~valid] = 0
    assert not depth_edges(depth, valid).any()
    other = across_edge(depth, valid, 3)
    np.testing.assert_allclose(other[:, 14:16], 6.0)
    assert np.isnan(other[~valid]).all()
