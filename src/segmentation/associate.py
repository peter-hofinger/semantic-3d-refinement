"""Link per-view instance masks across views by reprojecting them with depth.

Mask a in view i and mask b in view j overlap by the fraction of a's pixels that,
lifted with i's depth and projected into j, land inside b, counting only pixels
that are visible in j (inside the image and in front within a depth tolerance).
Optionally a direction only counts if most of a's pixels inside j's image are
visible there: if a is largely hidden in j, the overlap rests on a sliver of it.
Two masks are linked if both directions overlap enough and their classes agree;
optionally masks of different classes are linked too at a stricter overlap
(near-synonym prompts like "monitor" / "tv" name one object differently per view).
Links are added greedily, strongest first, and never put two masks of the same
view into one object. An object takes the class covering most of its pixels.
With detection scores, low-score masks can be dropped before linking, or kept only
if they end up linked to a mask in another view (false detections rarely reappear
at the same 3D place).
"""

from __future__ import annotations

import argparse
import logging
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from src.geometry import Geometry
from src.loaders import Views
from src.loaders.ground_truth import GroundTruth
from src.segmentation.base import ID_STRIDE, NO_INSTANCE, Segmentation

log = logging.getLogger(__name__)


@dataclass
class Associator:
    """Settings for linking masks across views."""

    min_overlap: float = 0.5  # both directions must reach this
    depth_tolerance: float = (
        0.05  # relative depth difference that still counts as visible
    )
    min_visible: int = 50  # pixels of a mask that must be visible in the other view
    # share of a mask's pixels inside the other view's image that must be visible
    min_visible_fraction: float = 0.0
    # overlap needed to link masks of different classes; None: never link them
    cross_class_overlap: float | None = None
    device: str | None = None  # where overlaps are computed; None: CUDA if available
    # scores as in SAM 3's threshold: a mask passes if its score is above the value
    min_score: float = 0.0  # masks not passing are dropped before linking
    # masks not passing are kept only if linked to a mask in another view
    keep_linked_below: float = 0.0

    def overlaps(
        self, seg: Segmentation, geometry: Geometry
    ) -> dict[tuple[int, int], float]:
        """Directed overlap a -> b for every pair of masks in different views.

        Each view's labeled pixels are projected into all other views at once, in
        torch on `device` (CUDA if available), in float64 as the depth's lift is.
        """
        device = torch.device(
            self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        num, height, width = seg.instance_ids.shape
        ids = torch.from_numpy(seg.instance_ids).to(device)
        depth = torch.from_numpy(geometry.depth).to(device, torch.float64)
        points = torch.from_numpy(geometry.world_points()).to(device, torch.float64)
        world_to_cam = torch.from_numpy(np.linalg.inv(geometry.cam_to_world))
        world_to_cam = world_to_cam.to(device, torch.float64)
        k = torch.from_numpy(geometry.intrinsics).to(device, torch.float64)
        stride = num * ID_STRIDE + 1  # ids + 1 of any view, NO_INSTANCE -> 0
        overlap: dict[tuple[int, int], float] = {}
        for i in range(num):
            labeled = (ids[i] >= 0) & (depth[i] > 0)
            src_ids, src_pts = ids[i][labeled].long(), points[i][labeled]
            others = torch.tensor([j for j in range(num) if j != i], device=device)
            rot = world_to_cam[others, :3, :3]
            cam = src_pts @ rot.transpose(1, 2) + world_to_cam[others, None, :3, 3]
            z, kj = cam[..., 2], k[others, :, :, None]
            u = torch.floor(kj[:, 0, 0] * cam[..., 0] / z + kj[:, 0, 2])
            v = torch.floor(kj[:, 1, 1] * cam[..., 1] / z + kj[:, 1, 2])
            inside = (z > 0) & (u >= 0) & (u < width) & (v >= 0) & (v < height)
            row, pixel = inside.nonzero(as_tuple=True)
            j, u, v = others[row], u[row, pixel].long(), v[row, pixel].long()
            target = depth[j, v, u]
            # (target view, source mask) as one key; and with the mask landed on
            source = j * stride + src_ids[pixel] + 1
            known = target > 0
            visible = (z[row, pixel] - target).abs() <= self.depth_tolerance * target
            landed = ids[j, v, u].long() + 1
            in_keys, in_frame = torch.unique(source[known], return_counts=True)
            a_keys, a_count = torch.unique(source[visible], return_counts=True)
            pair_keys, n = torch.unique(
                source[visible] * stride + landed[visible], return_counts=True
            )
            at = torch.searchsorted(in_keys, a_keys)  # visible pixels are known ones
            assert bool((in_keys[at] == a_keys).all())  # noqa: S101
            frame = in_frame[at]
            enough = (a_count >= self.min_visible) & (
                a_count >= self.min_visible_fraction * frame
            )
            a_of_pair = torch.searchsorted(a_keys, pair_keys // stride)
            keep = (pair_keys % stride > 0) & enough[a_of_pair]
            a = (pair_keys[keep] // stride % stride - 1).tolist()
            b = (pair_keys[keep] % stride - 1).tolist()
            shares = (n[keep].double() / a_count[a_of_pair[keep]]).tolist()
            overlap.update(zip(zip(a, b, strict=True), shares, strict=True))
        return overlap

    def _needed(self, same_class: bool) -> float:  # noqa: FBT001
        """Overlap a link needs; inf for different classes unless allowed."""
        if same_class:
            return self.min_overlap
        if self.cross_class_overlap is None:
            return np.inf
        return max(self.cross_class_overlap, self.min_overlap)

    def _drop_low(self, seg: Segmentation) -> Segmentation:
        """Remove masks below `min_score` (their pixels become unlabeled)."""
        if self.min_score > 0 or self.keep_linked_below > 0:
            assert seg.scores, "score options need a segmentation with scores"  # noqa: S101
        if self.min_score <= 0:
            return seg
        drop = [m for m, s in seg.scores.items() if s <= self.min_score]
        keep = {m for m in seg.classes if m not in set(drop)}
        return Segmentation(
            np.where(np.isin(seg.instance_ids, drop), NO_INSTANCE, seg.instance_ids),
            {m: seg.classes[m] for m in keep},
            {m: seg.scores[m] for m in keep},
        )

    def associate(self, seg: Segmentation, geometry: Geometry) -> Segmentation:
        """Give masks that show the same object one shared id."""
        for i, view_ids in enumerate(seg.instance_ids):
            local = np.unique(view_ids[view_ids >= 0]) // ID_STRIDE
            assert np.all(local == i), f"ids in view {i} are not view-local"  # noqa: S101
        seg = self._drop_low(seg)
        overlap = self.overlaps(seg, geometry)
        pairs = {(min(a, b), max(a, b)) for a, b in overlap}
        edges = sorted(
            (
                (score, a, b)
                for a, b in pairs
                if (score := min(overlap.get((a, b), 0.0), overlap.get((b, a), 0.0)))
                >= self._needed(seg.classes.get(a) == seg.classes.get(b))
            ),
            reverse=True,
        )
        parent = {m: m for m in seg.classes}
        views = {m: {m // ID_STRIDE} for m in seg.classes}

        def root(m: int) -> int:
            while parent[m] != m:
                parent[m] = parent[parent[m]]
                m = parent[m]
            return m

        for _, a, b in edges:
            ra, rb = root(a), root(b)
            if ra != rb and not views[ra] & views[rb]:
                parent[rb] = ra
                views[ra] |= views.pop(rb)

        roots = {
            m: r
            for m in seg.classes
            if len(views[r := root(m)]) > 1
            or seg.scores.get(m, 1.0) > self.keep_linked_below
        }
        new_id = {r: n for n, r in enumerate(sorted(set(roots.values())))}
        lookup = np.full(max(seg.classes) + 2, NO_INSTANCE, dtype=np.int32)
        for m, r in roots.items():
            lookup[m] = new_id[r]
        ids = lookup[seg.instance_ids]  # NO_INSTANCE = -1 indexes the last slot
        return Segmentation(
            ids, *object_labels(seg, {m: new_id[r] for m, r in roots.items()})
        )


def object_labels(
    seg: Segmentation, owner: dict[int, int]
) -> tuple[dict[int, str], dict[int, float]]:
    """Class and score of each object from its masks (mask id -> object id).

    The class is the one covering most of the object's pixels over all views, the
    score the masks' detection scores weighted by their area.
    """
    masks, area = np.unique(seg.instance_ids, return_counts=True)
    votes: dict[int, Counter[str]] = defaultdict(Counter)
    weighted: dict[int, list[float]] = defaultdict(lambda: [0.0, 0.0])
    for m, a in zip(masks.tolist(), area.tolist(), strict=True):
        if m not in owner:
            continue
        votes[owner[m]][seg.classes[m]] += a
        if m in seg.scores:
            weighted[owner[m]][0] += a * seg.scores[m]
            weighted[owner[m]][1] += a
    classes = {n: v.most_common(1)[0][0] for n, v in votes.items()}
    return classes, {n: s / a for n, (s, a) in weighted.items()}


def gt_geometry(views: Views, gt: GroundTruth) -> Geometry:
    """Ground-truth depth with the ground-truth cameras: the upper bound."""
    return Geometry(
        depth=gt.depth,
        confidence=(gt.depth > 0).astype(np.float32),
        intrinsics=views.intrinsics,
        cam_to_world=views.cam_to_world,
    )


def main() -> None:
    """Associate a cached per-view segmentation and cache the linked result."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("views", type=Path, help="prepared views (.npz)")
    parser.add_argument("segmentation", type=Path, help="per-view Segmentation (.npz)")
    parser.add_argument(
        "--geometry",
        default="gt",
        help="'gt' for rendered ground truth, or a Geometry .npz (e.g. Déjà View)",
    )
    parser.add_argument(
        "--cross-class",
        type=float,
        help="also link masks of different classes at this overlap (default: never); "
        "output name gets _assoc-xc<value>-<geometry>",
    )
    parser.add_argument(
        "--min-visible-fraction",
        type=float,
        default=0.0,
        help="share of a mask's in-image pixels that must be visible in the other view "
        "for its overlap to count (default 0: off); output name gets vf<value>",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=0.0,
        help="drop masks scoring <= this before linking (needs saved scores); "
        "output name gets ms<value>",
    )
    parser.add_argument(
        "--keep-linked-below",
        type=float,
        default=0.0,
        help="keep masks scoring <= this only if linked to a mask in another view; "
        "output name gets kl<value>",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    views = Views.load(args.views)
    if args.geometry == "gt":
        gt = GroundTruth.load(args.views.with_name(f"gt_{args.views.stem}.npz"))
        geometry, tag = gt_geometry(views, gt), "gt"
    else:
        geometry, tag = Geometry.load(Path(args.geometry)), Path(args.geometry).stem
    seg = Segmentation.load(args.segmentation)
    start = time.perf_counter()
    linked = Associator(
        cross_class_overlap=args.cross_class,
        min_visible_fraction=args.min_visible_fraction,
        min_score=args.min_score,
        keep_linked_below=args.keep_linked_below,
    ).associate(seg, geometry)
    if args.keep_linked_below > 0:
        tag = f"kl{args.keep_linked_below:g}-{tag}"
    if args.min_score > 0:
        tag = f"ms{args.min_score:g}-{tag}"
    if args.min_visible_fraction > 0:
        tag = f"vf{args.min_visible_fraction:g}-{tag}"
    if args.cross_class is not None:
        tag = f"xc{args.cross_class:g}-{tag}"
    out = args.segmentation.with_name(f"{args.segmentation.stem}_assoc-{tag}.npz")
    linked.save(out)
    log.info(
        "%s: %d masks -> %d objects, %.0fs -> %s",
        views.scene_id,
        len(seg.classes),
        len(linked.classes),
        time.perf_counter() - start,
        out,
    )


if __name__ == "__main__":
    main()
