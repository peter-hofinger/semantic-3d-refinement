import numpy as np

from src.backbone.segvggt import merge_queries, pinhole_rays, to_cam_to_world
from src.geometry import Geometry
from src.segmentation.base import NO_INSTANCE


def random_world_to_cam(rng: np.random.Generator, num: int) -> np.ndarray:
    q, _ = np.linalg.qr(rng.normal(size=(num, 3, 3)))
    q *= np.sign(np.linalg.det(q))[:, None, None]
    return np.concatenate([q, rng.normal(size=(num, 3, 1))], axis=-1)


def test_cam_to_world_inverts_extrinsics() -> None:
    w2c = random_world_to_cam(np.random.default_rng(0), 3)
    c2w = to_cam_to_world(w2c)
    w2c_h = np.concatenate([w2c, np.tile([[[0, 0, 0, 1.0]]], (3, 1, 1))], axis=1)
    np.testing.assert_allclose(c2w @ w2c_h, np.tile(np.eye(4), (3, 1, 1)), atol=1e-12)


def test_pinhole_rays_reproduce_lifted_points() -> None:
    rng = np.random.default_rng(1)
    k = np.array([[[50.0, 0, 3.0], [0, 55.0, 2.0], [0, 0, 1]]] * 2)
    c2w = to_cam_to_world(random_world_to_cam(rng, 2))
    depth = rng.uniform(1, 3, size=(2, 4, 6))
    origins, directions = pinhole_rays(k, c2w, 4, 6)
    geometry = Geometry(depth, np.ones_like(depth), k, c2w)
    np.testing.assert_allclose(
        origins + depth[..., None] * directions, geometry.world_points(), atol=1e-5
    )


def test_merge_gives_overlap_to_the_more_confident_query() -> None:
    masks = np.zeros((2, 1, 2, 4))
    masks[0, 0, :, :3] = 0.9  # query 0 covers columns 0-2
    masks[1, 0, :, 1:] = 0.9  # query 1 covers columns 1-3, more confident
    ids, labels, scores = merge_queries(
        np.array([0.85, 0.95]), np.array([4, 2]), masks, overlap_threshold=0.0
    )
    np.testing.assert_array_equal(ids[0], [[0, 1, 1, 1], [0, 1, 1, 1]])
    assert labels == {0: 4, 1: 2}
    assert np.isclose(scores[1], 0.95 * 0.9)


def test_merge_drops_a_query_mostly_covered_by_another() -> None:
    masks = np.zeros((2, 1, 2, 4))
    masks[0, 0] = 0.9  # whole image, confident
    masks[1, 0, :, :1] = 0.9  # one column, less confident: wins nothing
    ids, labels, _ = merge_queries(
        np.array([0.95, 0.85]), np.array([0, 1]), masks, overlap_threshold=0.8
    )
    assert set(np.unique(ids)) == {0}
    assert labels == {0: 0}


def test_merge_without_queries_is_empty() -> None:
    ids, labels, scores = merge_queries(
        np.zeros(0), np.zeros(0, int), np.zeros((0, 2, 3, 4)), overlap_threshold=0.8
    )
    assert ids.shape == (2, 3, 4) and np.all(ids == NO_INSTANCE)
    assert labels == {} and scores == {}
