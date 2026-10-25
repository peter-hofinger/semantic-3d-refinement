"""Turn raw ScanNet++ scenes into backbone input: subsampled, resized views."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from semantic_3d_refinement.data.frames import resize_view, select_frames, target_size
from semantic_3d_refinement.data.scannetpp.config import load_config
from semantic_3d_refinement.data.scannetpp.scene import ScannetppScene
from semantic_3d_refinement.data.views import Views

if TYPE_CHECKING:
    from semantic_3d_refinement.data.scannetpp.config import (
        PrepareConfig,
        ScannetppConfig,
    )

log = logging.getLogger(__name__)


def build_views(scene: ScannetppScene, cfg: PrepareConfig) -> Views:
    """Select frames from a scene and resize them for the backbone."""
    k, (width, height) = scene.intrinsics()
    frames = scene.frames()
    centers = np.stack([f.cam_to_world[:3, 3] for f in frames])
    picked = [frames[i] for i in select_frames(centers, cfg.num_frames, cfg.sampling)]
    size = target_size(width, height, cfg.long_side, cfg.patch_size)

    images, intrinsics = [], []
    for frame in picked:
        image, k_scaled = resize_view(scene.load_image(frame.name), k, size)
        images.append(image)
        intrinsics.append(k_scaled)

    return Views(
        scene_id=scene.scene_id,
        names=[f.name for f in picked],
        images=np.stack(images),
        intrinsics=np.stack(intrinsics).astype(np.float32),
        cam_to_world=np.stack([f.cam_to_world for f in picked]).astype(np.float32),
    )


def output_path(cfg: ScannetppConfig, scene_id: str) -> Path:
    """Where the views of a scene are written; the name encodes the settings."""
    p = cfg.prepare
    name = f"views_n{p.num_frames}_{p.sampling}_{p.long_side}.npz"
    return cfg.output_root / scene_id / name


def main() -> None:
    """Prepare every scene listed in the config."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/scannetpp.yaml"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    cfg = load_config(args.config)
    for scene_id in cfg.scenes:
        views = build_views(ScannetppScene(cfg.data_root, scene_id), cfg.prepare)
        path = output_path(cfg, scene_id)
        views.save(path)
        height, width = views.images.shape[1:3]
        log.info("%s: %d views, %dx%d -> %s", scene_id, len(views), width, height, path)


if __name__ == "__main__":
    main()
