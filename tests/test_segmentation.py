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
from src.segmentation.associate import Associator, absorb_parts, merge_touching
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


def overlaps_reference(
    associator: Associator, seg: Segmentation, geometry: Geometry
) -> dict[tuple[int, int], float]:
    """The numpy reprojection E35 replaced, one view pair at a time."""
    num, height, width = seg.instance_ids.shape
    points = geometry.world_points()
    world_to_cam = np.linalg.inv(geometry.cam_to_world)
    overlap: dict[tuple[int, int], float] = {}
    for i in range(num):
        labeled = (seg.instance_ids[i] >= 0) & (geometry.depth[i] > 0)
        src_ids = seg.instance_ids[i][labeled]
        src_pts = points[i][labeled]
        for j in range(num):
            if i == j:
                continue
            cam = src_pts @ world_to_cam[j, :3, :3].T + world_to_cam[j, :3, 3]
            z = cam[:, 2]
            k = geometry.intrinsics[j]
            with np.errstate(divide="ignore", invalid="ignore"):
                u = np.floor(k[0, 0] * cam[:, 0] / z + k[0, 2]).astype(np.int64)
                v = np.floor(k[1, 1] * cam[:, 1] / z + k[1, 2]).astype(np.int64)
            inside = (z > 0) & (u >= 0) & (u < width) & (v >= 0) & (v < height)
            u, v, z, ids = u[inside], v[inside], z[inside], src_ids[inside]
            target_depth = geometry.depth[j][v, u]
            known = target_depth > 0
            in_ids, in_count = np.unique(ids[known], return_counts=True)
            in_frame = dict(zip(in_ids.tolist(), in_count.tolist(), strict=True))
            visible = (
                np.abs(z - target_depth) <= associator.depth_tolerance * target_depth
            )
            ids, landed = ids[visible], seg.instance_ids[j][v[visible], u[visible]]
            a_ids, a_inv, a_count = np.unique(
                ids, return_inverse=True, return_counts=True
            )
            sources, targets, n = count_pairs(a_inv, landed)
            enough = (a_count >= associator.min_visible) & (
                a_count
                >= associator.min_visible_fraction
                * np.array([in_frame[m] for m in a_ids.tolist()])
            )
            for a, b, c in zip(sources, targets, n, strict=True):
                if b != NO_INSTANCE and enough[a]:
                    overlap[int(a_ids[a]), int(b)] = c / a_count[a]
    return overlap


def random_scene(seed: int, num: int = 4) -> tuple[Segmentation, Geometry]:
    """Views of a box of depth steps from nearby cameras, with blocky masks."""
    rng = np.random.default_rng(seed)
    height, width = 24, 32
    views = make_views(num, height, width)
    cam_to_world = np.repeat(np.eye(4)[None], num, axis=0)
    for i in range(num):
        angle = rng.uniform(-0.15, 0.15)
        c, s = np.cos(angle), np.sin(angle)
        cam_to_world[i, :3, :3] = [[c, 0, s], [0, 1, 0], [-s, 0, c]]
        cam_to_world[i, :3, 3] = rng.uniform(-0.3, 0.3, 3)
    blocks = rng.uniform(2.0, 4.0, (num, 4, 4))
    depth = np.kron(blocks, np.ones((6, 8))).astype(np.float32)
    depth[rng.random(depth.shape) < 0.05] = 0  # pixels without depth
    labels = rng.integers(-1, 3, (num, 4, 4))
    local = np.kron(labels, np.ones((6, 8), dtype=np.int64))
    ids = np.where(local >= 0, np.arange(num)[:, None, None] * ID_STRIDE + local, -1)
    classes = {int(m): "chair" for m in np.unique(ids[ids >= 0])}
    geometry = Geometry(
        depth=depth,
        confidence=np.ones_like(depth),
        intrinsics=views.intrinsics,
        cam_to_world=cam_to_world.astype(np.float32),
    )
    return Segmentation(ids.astype(np.int32), classes), geometry


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("fraction", [0.0, 0.25])
def test_torch_overlaps_match_the_numpy_reprojection(
    seed: int, fraction: float
) -> None:
    """Same pairs, same shares to float64 rounding, for every directed mask pair."""
    seg, geometry = random_scene(seed)
    associator = Associator(min_visible=5, min_visible_fraction=fraction, device="cpu")
    expected = overlaps_reference(associator, seg, geometry)
    got = associator.overlaps(seg, geometry)
    assert expected  # the scene has overlapping masks to compare
    assert got.keys() == expected.keys()
    for key, share in expected.items():
        assert got[key] == pytest.approx(share, rel=1e-12)


def test_count_pairs_matches_unique_rows() -> None:
    rng = np.random.default_rng(0)
    a, b = rng.integers(-3, 5, 1000), rng.integers(-2, 900, 1000)
    pairs, counts = np.unique(np.stack([a, b]), axis=1, return_counts=True)
    got_a, got_b, got_counts = count_pairs(a, b)
    np.testing.assert_array_equal(np.stack([got_a, got_b]), pairs)
    np.testing.assert_array_equal(got_counts, counts)
    assert count_pairs(a[:0], b[:0])[2].size == 0


def test_fallback_masks_only_take_uncovered_pixels() -> None:
    masks = torch.zeros(2, 1, 4, dtype=torch.bool)
    masks[0, 0, :2] = True  # specific mask, low score
    masks[1, 0, :] = True  # fallback mask, high score
    scores = torch.tensor([0.4, 0.9])
    plain = resolve_overlaps(masks, scores)
    assert plain.tolist() == [[1, 1, 1, 1]]
    weak = resolve_overlaps(masks, scores, torch.tensor([False, True]))
    assert weak.tolist() == [[0, 0, 1, 1]]


def test_touching_objects_of_one_class_merge_and_others_stay_apart() -> None:
    h, w = 8, 40
    k = np.array([[[20.0, 0, w / 2], [0, 20.0, h / 2], [0, 0, 1]]])
    geometry = Geometry(np.full((1, h, w), 2.0), np.ones((1, h, w)), k, np.eye(4)[None])
    ids = np.full((1, h, w), NO_INSTANCE, dtype=np.int32)
    ids[0, :, :10], ids[0, :, 10:20], ids[0, :, 30:] = 0, 1, 2  # 0 | 1, gap, 2
    seg = Segmentation(ids, {0: "chair", 1: "chair", 2: "chair"})
    side = 2.0 / 20  # one pixel's footprint
    merged = merge_touching(seg, geometry, side, 0.05)
    assert merged.instance_ids[0, 0, 0] == merged.instance_ids[0, 0, 15]
    assert merged.instance_ids[0, 0, 0] != merged.instance_ids[0, 0, 35]
    other = Segmentation(ids, {0: "chair", 1: "table", 2: "chair"})
    apart = merge_touching(other, geometry, side, 0.05)
    assert len(np.unique(apart.instance_ids[apart.instance_ids >= 0])) == 3
    assert merged.classes[int(merged.instance_ids[0, 0, 0])] == "chair"


def test_a_small_part_of_another_class_is_absorbed_and_a_large_one_is_not() -> None:
    h, w = 8, 40
    k = np.array([[[20.0, 0, w / 2], [0, 20.0, h / 2], [0, 0, 1]]])
    geometry = Geometry(np.full((1, h, w), 2.0), np.ones((1, h, w)), k, np.eye(4)[None])
    ids = np.zeros((1, h, w), dtype=np.int32)  # sofa
    ids[0, :2, 30:32] = 1  # small "pillow" part inside it
    ids[0, :, :8] = 2  # large "chair" next to it
    seg = Segmentation(ids, {0: "sofa", 1: "pillow", 2: "chair"})
    side = 2.0 / 20
    merged = merge_touching(seg, geometry, side, 0.3)
    out = absorb_parts(merged, geometry, side, (0.5, 0.05))
    assert out.instance_ids[0, 0, 30] == out.instance_ids[0, 5, 20]
    assert out.classes[int(out.instance_ids[0, 0, 30])] == "sofa"
    assert out.instance_ids[0, 0, 0] != out.instance_ids[0, 5, 20]
    assert len(np.unique(merged.instance_ids)) == 3
    kept = absorb_parts(merged, geometry, side, (0.5, 0.05), hosts={"chair"})
    assert len(np.unique(kept.instance_ids)) == 3
