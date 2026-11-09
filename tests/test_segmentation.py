from pathlib import Path

import numpy as np
import pytest
import torch

from src.backbone import CachedBackbone, ConstantDepthBackbone
from src.geometry import Geometry
from src.loaders import Views
from src.pipeline import Pipeline
from src.refinement import IdentityRefiner
from src.segmentation import Segmentation
from src.segmentation.base import ID_STRIDE, NO_INSTANCE, count_pairs
from src.segmentation.linked import LinkedSegmenter
from src.segmentation.sam3 import compact, resolve_overlaps


def make_views(num: int = 2, height: int = 20, width: int = 30) -> Views:
    k = np.array([[20.0, 0, width / 2], [0, 20.0, height / 2], [0, 0, 1]])
    return Views(
        scene_id="toy",
        names=[f"{i}.jpg" for i in range(num)],
        images=np.zeros((num, height, width, 3), dtype=np.uint8),
        intrinsics=np.repeat(k[None], num, axis=0).astype(np.float32),
        cam_to_world=np.repeat(np.eye(4)[None], num, axis=0).astype(np.float32),
    )


class HalvesSegmenter:
    """Stand-in for SAM 3: per-view ids for the left and right image halves."""

    def __init__(self, classes: tuple[str, str] = ("table", "chair")) -> None:
        self.classes = classes

    def segment(self, views: Views, geometry: Geometry | None = None) -> Segmentation:  # noqa: ARG002
        num, height, width = views.images.shape[:3]
        ids = np.full((num, height, width), NO_INSTANCE, dtype=np.int32)
        classes, scores = {}, {}
        for i in range(num):
            left, right = i * ID_STRIDE, i * ID_STRIDE + 1
            ids[i, :, : width // 2] = left
            ids[i, :, width // 2 :] = right
            classes |= {left: self.classes[0], right: self.classes[1]}
            scores |= {left: 0.9, right: 0.9}
        return Segmentation(ids, classes, scores)


def test_linked_segmenter_shares_ids_across_views() -> None:
    # two identical cameras looking at a flat wall: each half is one object
    views = make_views()
    geometry = ConstantDepthBackbone().predict(views)
    seg = LinkedSegmenter(HalvesSegmenter()).segment(views, geometry)
    assert len(seg.classes) == 2
    np.testing.assert_array_equal(seg.instance_ids[0], seg.instance_ids[1])
    assert seg.instance_ids[0, 0, 0] != seg.instance_ids[0, 0, -1]
    assert sorted(seg.classes.values()) == ["chair", "table"]


def test_linked_objects_score_the_area_weighted_mean_of_their_masks() -> None:
    views = make_views()
    per_view = HalvesSegmenter().segment(views)
    per_view.scores[ID_STRIDE] = 0.5  # view 1's left half; view 0's is 0.9
    seg = LinkedSegmenter(HalvesSegmenter()).associator.associate(
        per_view, ConstantDepthBackbone().predict(views)
    )
    left, right = seg.instance_ids[0, 0, 0], seg.instance_ids[0, 0, -1]
    assert seg.scores[left] == pytest.approx(0.7)
    assert seg.scores[right] == pytest.approx(0.9)


def test_linked_segmenter_runs_in_the_pipeline() -> None:
    views = make_views()
    segmenter = LinkedSegmenter(HalvesSegmenter())
    result = Pipeline(ConstantDepthBackbone(), segmenter, IdentityRefiner()).run(views)
    assert len(np.unique(result.cloud.instance_ids)) == 2


def test_cached_backbone_rejects_other_resolution(tmp_path: Path) -> None:
    path = tmp_path / "geometry.npz"
    ConstantDepthBackbone().predict(make_views()).save(path)
    assert CachedBackbone(path).predict(make_views()).depth.shape == (2, 20, 30)
    with pytest.raises(ValueError, match="does not match"):
        CachedBackbone(path).predict(make_views(height=10))


def test_compact_numbers_only_visible_masks() -> None:
    """Hidden raw detections get no id; visible ones keep their order."""
    local = torch.tensor([[5, 5, -1], [2, 1500, 1500]])
    present, ids = compact(local, view=3)
    base = 3 * ID_STRIDE
    assert present.tolist() == [2, 5, 1500]
    assert ids.dtype == torch.int32
    assert ids.tolist() == [[base + 1, base + 1, -1], [base, base + 2, base + 2]]


def paint_reference(masks: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """The CPU painting E35 replaced: masks in order, a higher score takes a pixel."""
    ids = np.full(masks.shape[1:], NO_INSTANCE, dtype=np.int32)
    best = np.zeros(masks.shape[1:], dtype=np.float32)
    for obj, (score, mask) in enumerate(zip(scores, masks, strict=True)):
        win = mask & (score > best)
        ids[win] = obj
        best[win] = score
    return ids


def compact_reference(local: np.ndarray, view: int) -> np.ndarray:
    """The CPU id compaction E35 replaced."""
    present = np.unique(local[local >= 0])
    out = np.full(local.shape, NO_INSTANCE, dtype=np.int32)
    keep = local >= 0
    new = view * ID_STRIDE + np.arange(len(present), dtype=np.int32)
    out[keep] = new[np.searchsorted(present, local[keep])]
    return out


@pytest.mark.parametrize("seed", range(5))
def test_gpu_painting_matches_sequential_painting(seed: int) -> None:
    """Highest score wins, the earlier mask on ties, as painting in order did."""
    rng = np.random.default_rng(seed)
    masks = rng.random((12, 15, 20)) < 0.3
    # few distinct scores, so ties between overlapping masks are common
    scores = rng.choice([0.31, 0.5, 0.5, 0.77, 0.9], 12).astype(np.float32)
    local = resolve_overlaps(torch.from_numpy(masks), torch.from_numpy(scores))
    expected = paint_reference(masks, scores)
    np.testing.assert_array_equal(local.numpy(), expected)
    _, ids = compact(local, view=7)
    np.testing.assert_array_equal(ids.numpy(), compact_reference(expected, 7))


def test_gpu_painting_without_masks_leaves_the_view_empty() -> None:
    local = resolve_overlaps(torch.zeros(0, 4, 5, dtype=torch.bool), torch.zeros(0))
    assert local.shape == (4, 5)
    assert bool((local == NO_INSTANCE).all())


def test_count_pairs_matches_unique_rows() -> None:
    rng = np.random.default_rng(0)
    a, b = rng.integers(-3, 5, 1000), rng.integers(-2, 900, 1000)
    pairs, counts = np.unique(np.stack([a, b]), axis=1, return_counts=True)
    got_a, got_b, got_counts = count_pairs(a, b)
    np.testing.assert_array_equal(np.stack([got_a, got_b]), pairs)
    np.testing.assert_array_equal(got_counts, counts)
    assert count_pairs(a[:0], b[:0])[2].size == 0
