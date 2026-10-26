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
"""

from __future__ import annotations

import argparse
import logging
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.geometry import Geometry
from src.loaders import Views
from src.loaders.scannetpp.gt import GroundTruth
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

    def overlaps(
        self, seg: Segmentation, geometry: Geometry
    ) -> dict[tuple[int, int], float]:
        """Directed overlap a -> b for every pair of masks in different views."""
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
                    np.abs(z - target_depth) <= self.depth_tolerance * target_depth
                )
                ids, landed = ids[visible], seg.instance_ids[j][v[visible], u[visible]]
                a_ids, a_inv, a_count = np.unique(
                    ids, return_inverse=True, return_counts=True
                )
                pairs, n = np.unique(
                    np.stack([a_inv, landed]), axis=1, return_counts=True
                )
                enough = (a_count >= self.min_visible) & (
                    a_count
                    >= self.min_visible_fraction
                    * np.array([in_frame[m] for m in a_ids.tolist()])
                )
                for (a, b), c in zip(pairs.T, n, strict=True):
                    if b != NO_INSTANCE and enough[a]:
                        overlap[int(a_ids[a]), int(b)] = c / a_count[a]
        return overlap

    def _needed(self, same_class: bool) -> float:  # noqa: FBT001
        """Overlap a link needs; inf for different classes unless allowed."""
        if same_class:
            return self.min_overlap
        if self.cross_class_overlap is None:
            return np.inf
        return max(self.cross_class_overlap, self.min_overlap)

    def associate(self, seg: Segmentation, geometry: Geometry) -> Segmentation:
        """Give masks that show the same object one shared id."""
        for i, view_ids in enumerate(seg.instance_ids):
            local = np.unique(view_ids[view_ids >= 0]) // ID_STRIDE
            assert np.all(local == i), f"ids in view {i} are not view-local"  # noqa: S101
        overlap = self.overlaps(seg, geometry)
        edges = sorted(
            (
                (score, a, b)
                for (a, b), s in overlap.items()
                if a < b
                and (score := min(s, overlap.get((b, a), 0.0)))
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

        roots = {m: root(m) for m in seg.classes}
        new_id = {r: n for n, r in enumerate(sorted(set(roots.values())))}
        lookup = np.full(max(seg.classes) + 2, NO_INSTANCE, dtype=np.int32)
        for m, r in roots.items():
            lookup[m] = new_id[r]
        ids = lookup[seg.instance_ids]  # NO_INSTANCE = -1 indexes the last slot
        # class of an object: the one covering most of its pixels over all views
        masks, area = np.unique(seg.instance_ids, return_counts=True)
        votes: dict[int, Counter[str]] = defaultdict(Counter)
        for m, a in zip(masks.tolist(), area.tolist(), strict=True):
            if m in roots:
                votes[new_id[roots[m]]][seg.classes[m]] += a
        classes = {n: v.most_common(1)[0][0] for n, v in votes.items()}
        return Segmentation(ids, classes)


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
    ).associate(seg, geometry)
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
