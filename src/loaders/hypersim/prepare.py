"""Turn Hypersim trajectories into backbone input and ground truth.

Writes, per trajectory, `views_n<N>_<sampling>_<long side>.npz` (Views) and
`gt_<views stem>.npz` (GroundTruth, the format `eval-instances` reads) into
<output_root>/<scene>_<camera>/.

Ground truth comes straight from Hypersim's per-pixel maps, sampled at the centres
of the resized pixels (nearest, no mixing across edges). Hypersim leaves walls,
floors and ceilings without instance ids; every such class becomes one instance
(`STUFF_BASE + NYU40 id`), shared by all views.
"""

from __future__ import annotations

import argparse
import logging
from collections import Counter, defaultdict
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from src.loaders.frames import resize_view, select_frames, target_size
from src.loaders.hypersim.config import check_splits, load_config
from src.loaders.hypersim.scene import (
    NYU40,
    FrameLabels,
    HypersimScene,
    distance_to_depth,
)
from src.loaders.scannetpp.gt import NO_MESH, GroundTruth
from src.loaders.views import Views
from src.segmentation.base import NO_INSTANCE

if TYPE_CHECKING:
    from src.loaders.hypersim.config import HypersimConfig
    from src.loaders.scannetpp.config import PrepareConfig

log = logging.getLogger(__name__)

STUFF_BASE = 1_000_000  # id of a class without Hypersim instances = base + class
UNKNOWN_CLASS = "unknown"  # instance whose pixels carry no NYU40 label


def sample_nearest(array: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Pick the source pixel under each target pixel centre; size is (width, height)."""
    height, width = array.shape[:2]
    new_w, new_h = size
    cols = ((np.arange(new_w) + 0.5) * width / new_w).astype(int)
    rows = ((np.arange(new_h) + 0.5) * height / new_h).astype(int)
    return array[rows[:, None], cols[None, :]]


def frame_ground_truth(
    labels: FrameLabels, k: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Instance ids (stuff ids, NO_MESH), z-depth and NYU40 class of one frame."""
    no_geometry = ~np.isfinite(labels.distance) | (labels.entity < 0)
    ids = labels.instance.copy()
    stuff = (ids < 0) & (labels.semantic > 0)
    ids[stuff] = STUFF_BASE + labels.semantic[stuff]
    ids[ids < 0] = NO_INSTANCE
    ids[no_geometry] = NO_MESH
    depth = distance_to_depth(np.where(no_geometry, 0.0, labels.distance), k)
    return ids, depth, labels.semantic


def instance_classes(ids: np.ndarray, semantic: np.ndarray) -> dict[int, str]:
    """NYU40 name of every instance: the majority class over all its pixels.

    Instances without any NYU40 label on their pixels get UNKNOWN_CLASS.
    """
    counts: dict[int, Counter] = defaultdict(Counter)
    valid = (ids >= 0) & (semantic > 0)
    pairs, n = np.unique(
        np.stack([ids[valid], semantic[valid]]), axis=1, return_counts=True
    )
    for (i, c), count in zip(pairs.T.tolist(), n.tolist(), strict=True):
        counts[i][c] += count
    classes = {i: NYU40[c.most_common(1)[0][0]] for i, c in counts.items()}
    for i in np.unique(ids[ids >= 0]).tolist():
        classes.setdefault(i, UNKNOWN_CLASS)
    return classes


def build(scene: HypersimScene, cfg: PrepareConfig) -> tuple[Views, GroundTruth]:
    """Select frames, resize images and sample the ground truth at the same pixels."""
    k, (width, height) = scene.intrinsics()
    frames = scene.frames()
    centers = np.stack([f.cam_to_world[:3, 3] for f in frames])
    picked = [frames[i] for i in select_frames(centers, cfg.num_frames, cfg.sampling)]
    size = target_size(width, height, cfg.long_side, cfg.patch_size)

    images, intrinsics, ids, depths, semantics = [], [], [], [], []
    for frame in picked:
        image, k_scaled = resize_view(scene.load_image(frame.frame_id), k, size)
        images.append(image)
        intrinsics.append(k_scaled)
        f_ids, f_depth, f_sem = frame_ground_truth(scene.load_labels(frame.frame_id), k)
        ids.append(sample_nearest(f_ids, size))
        depths.append(sample_nearest(f_depth, size))
        semantics.append(sample_nearest(f_sem, size))

    views = Views(
        scene_id=scene.scene_id,
        names=[f.name for f in picked],
        images=np.stack(images),
        intrinsics=np.stack(intrinsics).astype(np.float32),
        cam_to_world=np.stack([f.cam_to_world for f in picked]).astype(np.float32),
    )
    all_ids = np.stack(ids).astype(np.int32)
    gt = GroundTruth(
        instance_ids=all_ids,
        depth=np.stack(depths).astype(np.float32),
        classes=instance_classes(all_ids, np.stack(semantics)),
    )
    return views, gt


def output_path(cfg: HypersimConfig, scene_id: str) -> Path:
    """Where the views of a trajectory are written; the name encodes the settings."""
    p = cfg.prepare
    name = f"views_n{p.num_frames}_{p.sampling}_{p.long_side}.npz"
    return cfg.output_root / scene_id / name


def main() -> None:
    """Prepare the trajectories of the chosen splits after checking the split."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/hypersim.yaml"))
    parser.add_argument(
        "--splits", nargs="+", default=["dev"], help="dev val test stress"
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    cfg = load_config(args.config)
    check_splits(cfg)
    for scene, camera in cfg.scenes(args.splits).items():
        hs = HypersimScene(cfg.data_root, cfg.metadata_root, scene, camera)
        views, gt = build(hs, cfg.prepare)
        path = output_path(cfg, hs.scene_id)
        views.save(path)
        gt.save(path.with_name(f"gt_{path.stem}.npz"))
        ids = gt.instance_ids
        height, width = views.images.shape[1:3]
        log.info(
            "%s: %d views %dx%d; labeled %.0f%% (stuff %.0f%%), unlabeled %.0f%%,"
            " no geometry %.0f%%, %d instances -> %s",
            hs.scene_id,
            len(views),
            width,
            height,
            100 * (ids >= 0).mean(),
            100 * (ids >= STUFF_BASE).mean(),
            100 * (ids == NO_INSTANCE).mean(),
            100 * (ids == NO_MESH).mean(),
            len(np.unique(ids[ids >= 0])),
            path,
        )


if __name__ == "__main__":
    main()
