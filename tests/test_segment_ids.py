"""Segment kinds of the ablations: what each keeps of the linked masks (E8a, E86a)."""

import numpy as np

from src.analysis.filters import instance_voxel_density, pixel_footprint, voxel_counts
from src.analysis.scene_eval import segment_ids
from src.segmentation.base import NO_INSTANCE


def ids_pair() -> tuple[np.ndarray, np.ndarray]:
    """Two views of SAM 3 and GT ids, with a no-mesh (-2) GT pixel."""
    sam3 = np.array([[[0, 0, 1], [1, 2, -1]], [[2, 2, 0], [1, 1, -1]]], np.int32)
    gt = np.array([[[5, 5, 6], [6, 7, -2]], [[7, 7, 5], [6, 6, -1]]], np.int32)
    return sam3, gt


def test_sam3_is_the_linked_masks_and_none_is_none() -> None:
    sam3, gt = ids_pair()
    np.testing.assert_array_equal(segment_ids("sam3", sam3, gt, {}), sam3)
    assert segment_ids("none", sam3, gt, {}) is None


def test_gt_marks_unlabeled_and_no_mesh_pixels_as_none() -> None:
    sam3, gt = ids_pair()
    ids = segment_ids("gt", sam3, gt, {})
    assert ids is not None
    assert ids.dtype == np.int32
    np.testing.assert_array_equal(ids, np.where(gt < 0, NO_INSTANCE, gt))


def test_permuted_keeps_each_views_grouping_and_varies_with_the_seed() -> None:
    sam3, gt = ids_pair()
    for seed in range(4):
        ids = segment_ids("permuted", sam3, gt, {"seed": seed})
        assert ids is not None
        for view, own in zip(ids, sam3, strict=True):
            pairs = set(zip(own.ravel().tolist(), view.ravel().tolist(), strict=True))
            assert len(pairs) == len(np.unique(own)), pairs


def test_one_instance_makes_instance_density_plain_voxel_density() -> None:
    rng = np.random.default_rng(0)
    depth = rng.uniform(1.0, 3.0, (2, 4, 5))
    k = np.tile(np.array([[4.0, 0, 2], [0, 4.0, 2], [0, 0, 1]]), (2, 1, 1))
    points = rng.uniform(0.0, 1.0, (2, 4, 5, 3))
    one = np.full(depth.shape, NO_INSTANCE, np.int32)
    side = pixel_footprint(depth, k)
    count = instance_voxel_density(points, depth, k, one)
    np.testing.assert_array_equal(
        count.ravel(), voxel_counts(points.reshape(-1, 3), side)
    )
