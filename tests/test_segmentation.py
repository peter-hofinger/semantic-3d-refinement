from pathlib import Path

import numpy as np
import pytest

from src.backbone import CachedBackbone, ConstantDepthBackbone
from src.geometry import Geometry
from src.loaders import Views
from src.pipeline import Pipeline
from src.refinement import IdentityRefiner
from src.segmentation import Segmentation
from src.segmentation.base import ID_STRIDE, NO_INSTANCE
from src.segmentation.linked import LinkedSegmenter
from src.segmentation.sam3 import _compact
from src.segmentation.trim import trim_mask


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
    local = np.array([[5, 5, -1], [2, 1500, 1500]], dtype=np.int32)
    out = np.full(local.shape, -1, dtype=np.int32)
    mapping = _compact(local, out, view=3)
    base = 3 * ID_STRIDE
    assert mapping == {2: base, 5: base + 1, 1500: base + 2}
    assert out.tolist() == [[base + 1, base + 1, -1], [base, base + 2, base + 2]]


def test_trim_keeps_a_mask_without_depth_jumps_bit_for_bit() -> None:
    v, u = np.mgrid[0:20, 0:30]
    depth = 2.0 + 0.01 * u + 0.005 * v  # a slanted plane: < 1% between neighbours
    mask = (u - 15) ** 2 + (v - 10) ** 2 < 60
    assert np.array_equal(trim_mask(mask, depth), mask)


def test_trim_removes_what_is_seen_through_holes_and_past_the_outline() -> None:
    depth = np.full((20, 30), 3.0)  # wall
    frame = np.zeros((20, 30), dtype=bool)
    frame[4:16, 4:20] = True
    frame[6:14, 6:9] = frame[6:14, 11:14] = frame[6:14, 16:18] = False  # holes
    depth[frame] = 1.0
    mask = np.zeros_like(frame)
    mask[4:16, 4:23] = True  # the frame, its holes and a strip of wall to its right
    assert np.array_equal(trim_mask(mask, depth), frame)


def test_trim_returns_a_nonempty_subset_of_the_mask() -> None:
    rng = np.random.default_rng(0)
    for _ in range(50):
        depth = rng.choice([1.0, 1.5, 2.5], size=(12, 12))
        mask = rng.random((12, 12)) < 0.7
        trimmed = trim_mask(mask, depth)
        assert not (trimmed & ~mask).any()
        assert trimmed.any() == mask.any()
