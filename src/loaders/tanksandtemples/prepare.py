"""Turn Tanks and Temples scenes into backbone input and ground truth.

Writes, per scene, `views_n<N>_<sampling>_<long side>.npz` (Views) and
`gt_<views stem>.npz` (GroundTruth) into <output_root>/<scene>/, the layout of the
Hypersim loader. <raw_root>/<scene>/ holds the downloads as they come:
images/<number>.jpg, gt/<Scene>.ply, poses/<Scene>_COLMAP_SfM.log and
alignment/<Scene>_trans.txt. Frames are picked by `select_frames` in capture order,
cut and resized about the image centre and given the laser scan's depth (see
scene.py). Tanks and Temples has no labels: a pixel with depth is unlabeled, one
without is NO_MESH.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
from PIL import Image

from src.loaders.frames import select_frames
from src.loaders.ground_truth import NO_MESH, GroundTruth
from src.loaders.kitti360.scene import read_ply
from src.loaders.tanksandtemples.scene import (
    TanksAndTemplesConfig,
    centred_view,
    load_config,
    read_log,
    render_depth,
    to_scan_frame,
)
from src.loaders.views import Views
from src.segmentation.base import NO_INSTANCE

log = logging.getLogger(__name__)


def scan_points(path: Path) -> np.ndarray:
    """Read the scan's points (P, 3) float32, in metres."""
    vertices = read_ply(path)
    points = np.stack([vertices[axis] for axis in "xyz"], axis=1).astype(np.float32)
    assert np.isfinite(points).all()
    return points


def build(cfg: TanksAndTemplesConfig, scene: str) -> tuple[Views, GroundTruth]:
    """Select frames, cut and resize the images, render the scan into them."""
    root = cfg.raw_root / scene
    name = scene.capitalize()
    poses = to_scan_frame(
        read_log((root / "poses" / f"{name}_COLMAP_SfM.log").read_text()),
        np.loadtxt(root / "alignment" / f"{name}_trans.txt"),
    )
    files = sorted((root / "images").glob("*.jpg"))
    assert len(files) == len(poses), (len(files), len(poses))
    p = cfg.prepare
    picked = select_frames(poses[:, :3, 3], p.num_frames, p.sampling)
    points = scan_points(root / "gt" / f"{name}.ply")

    images, depths, intrinsics = [], [], []
    for i in picked:
        image = Image.open(files[i]).convert("RGB")
        box, size, k = centred_view(image.size, p)
        images.append(np.asarray(image.resize(size, Image.Resampling.LANCZOS, box=box)))
        depths.append(render_depth(points, poses[i], k, size))
        intrinsics.append(k)
    depth = np.stack(depths)
    views = Views(
        scene_id=scene,
        names=[files[i].stem for i in picked],
        images=np.stack(images),
        intrinsics=np.stack(intrinsics).astype(np.float32),
        cam_to_world=poses[picked].astype(np.float32),
    )
    ids = np.where(depth > 0, NO_INSTANCE, NO_MESH).astype(np.int32)
    return views, GroundTruth(instance_ids=ids, depth=depth, classes={})


def output_path(cfg: TanksAndTemplesConfig, scene: str) -> Path:
    """Where the views of a scene are written; the name encodes the settings."""
    p = cfg.prepare
    return (
        cfg.output_root
        / scene
        / f"views_n{p.num_frames}_{p.sampling}_{p.long_side}.npz"
    )


def main() -> None:
    """Prepare the chosen scenes (default: all of the config)."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/tanksandtemples.yaml")
    )
    parser.add_argument("--scenes", nargs="+", help="default: every scene")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    cfg = load_config(args.config)
    scenes = args.scenes or list(cfg.scenes)
    unknown = set(scenes) - set(cfg.scenes)
    if unknown:
        msg = f"not in the config: {sorted(unknown)}"
        raise ValueError(msg)
    for scene in scenes:
        views, gt = build(cfg, scene)
        path = output_path(cfg, scene)
        views.save(path)
        gt.save(path.with_name(f"gt_{path.stem}.npz"))
        height, width = views.images.shape[1:3]
        log.info(
            "%s (%s): %d views %dx%d, depth on %.0f%% of pixels, %.2f-%.2f m -> %s",
            scene,
            cfg.scenes[scene],
            len(views),
            width,
            height,
            100 * (gt.depth > 0).mean(),
            gt.depth[gt.depth > 0].min(),
            gt.depth.max(),
            path,
        )


if __name__ == "__main__":
    main()
