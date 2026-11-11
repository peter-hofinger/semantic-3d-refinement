"""Turn KITTI-360 windows into backbone input and ground truth.

Writes, per scene, `views_n<N>_stride<S>_<long side>.npz` (Views) and
`gt_<views stem>.npz` (GroundTruth) into <output_root>/<scene>/, the layout of the
Hypersim loader. A scene is one accumulated point-cloud window; its views are N of
the window's labelled frames, S apart and centred in it (`pick_frames`). Each image is
cropped about the principal point and resized (see scene.py). The depth comes from
the static points of the window and of its two neighbours, which hold what the last
views see ahead of the window's end.

Instances: KITTI-360's 2D instance map, 1000 x class + instance, the same id in every
frame of a drive. Classes it gives no instances (road, vegetation, ...) become one
instance per class (`STUFF_BASE + class`), as Hypersim's walls; pixels of void
classes with depth are unlabeled.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
from PIL import Image

from src.loaders.ground_truth import NO_MESH, STUFF_BASE, GroundTruth
from src.loaders.kitti360.scene import (
    CLASSES,
    INSTANCE_FACTOR,
    Kitti360Config,
    centred_crop,
    cropped_intrinsics,
    load_config,
    pick_frames,
    read_intrinsics,
    read_ply,
    read_poses,
    render_depth,
    sample_nearest,
    split_scene,
)
from src.loaders.views import Views
from src.segmentation.base import NO_INSTANCE

log = logging.getLogger(__name__)

CAMERA = "image_00"  # the left rectified perspective camera
CENTRE_TOL = 1e-6  # px the prepared principal point may lie off the image centre


def window_points(static: Path, first: int) -> tuple[np.ndarray, np.ndarray]:
    """Points (P, 3), classes (P,) of the window at `first` and its neighbours."""
    windows = sorted(static.glob("*.ply"))
    starts = [int(w.stem.split("_")[0]) for w in windows]
    i = starts.index(first)
    clouds = [read_ply(w) for w in windows[max(i - 1, 0) : i + 2]]
    points = np.concatenate(
        [np.stack([c["x"], c["y"], c["z"]], axis=1) for c in clouds]
    ).astype(np.float64)
    assert np.isfinite(points).all()
    return points, np.concatenate([c["semantic"] for c in clouds])


def instance_ids(
    instance: np.ndarray, depth: np.ndarray
) -> tuple[np.ndarray, dict[int, str]]:
    """Ids (H, W) int32 and their class names from a 2D instance map and the depth."""
    semantic = instance.astype(np.int32) // INSTANCE_FACTOR
    ids = instance.astype(np.int32)
    stuff = ids % INSTANCE_FACTOR == 0
    ids[stuff] = STUFF_BASE + semantic[stuff]
    ids[~np.isin(semantic, list(CLASSES))] = NO_INSTANCE
    ids[depth == 0] = NO_MESH
    labelled = ids >= 0
    pairs = np.unique(np.stack([ids[labelled], semantic[labelled]]), axis=1).T
    classes = {int(i): CLASSES[int(c)] for i, c in pairs}
    assert len(classes) == len(pairs), "an instance id with two classes"
    return ids, classes


def build(cfg: Kitti360Config, scene: str) -> tuple[Views, GroundTruth]:
    """Pick the window's frames, crop and resize them, project the point cloud."""
    drive, first, last = split_scene(scene)
    raw, p = cfg.raw_root, cfg.prepare
    k, size = read_intrinsics((raw / "calibration" / "perspective.txt").read_text())
    poses = read_poses((raw / "data_poses" / drive / "cam0_to_world.txt").read_text())
    labels = raw / "data_2d_semantics" / "train" / drive / CAMERA
    labelled = sorted(int(f.stem) for f in (labels / "semantic").glob("*.png"))
    frames = pick_frames(
        [f for f in labelled if first <= f <= last and f in poses],
        p.num_frames,
        p.stride,
    )
    box, new_size = centred_crop(k, size, p)
    k_new = cropped_intrinsics(k, box, new_size)
    static = raw / "data_3d_semantics" / "train" / drive / "static"
    points, point_classes = window_points(static, first)

    images, depths, ids, classes = [], [], [], {}
    for frame in frames:
        name = f"{frame:010d}.png"
        image = Image.open(raw / "data_2d_raw" / drive / CAMERA / "data_rect" / name)
        assert image.size == size, (image.size, name)
        images.append(
            np.asarray(
                image.convert("RGB").resize(new_size, Image.Resampling.LANCZOS, box=box)
            )
        )
        semantic = np.asarray(Image.open(labels / "semantic" / name))
        instance = np.asarray(Image.open(labels / "instance" / name))
        assert (semantic == instance // INSTANCE_FACTOR).all(), name
        depth = render_depth(
            points,
            point_classes,
            poses[frame],
            k_new,
            sample_nearest(semantic, box, new_size),
        )
        view_ids, view_classes = instance_ids(
            sample_nearest(instance, box, new_size), depth
        )
        assert all(classes.get(i, c) == c for i, c in view_classes.items()), name
        classes.update(view_classes)
        depths.append(depth)
        ids.append(view_ids)

    width, height = new_size
    assert np.allclose(k_new[:2, 2], [width / 2, height / 2], atol=CENTRE_TOL), k_new
    views = Views(
        scene_id=scene,
        names=[f"{f:010d}" for f in frames],
        images=np.stack(images),
        intrinsics=np.repeat(k_new[None], len(frames), axis=0).astype(np.float32),
        cam_to_world=np.stack([poses[f] for f in frames]).astype(np.float32),
    )
    return views, GroundTruth(
        instance_ids=np.stack(ids), depth=np.stack(depths), classes=classes
    )


def output_path(cfg: Kitti360Config, scene: str) -> Path:
    """Where the views of a scene are written; the name encodes the settings."""
    p = cfg.prepare
    name = f"views_n{p.num_frames}_stride{p.stride}_{p.long_side}.npz"
    return cfg.output_root / scene / name


def main() -> None:
    """Prepare the chosen scenes (default: all of the config)."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/kitti360.yaml"))
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
        centers = views.cam_to_world[:, :3, 3].astype(np.float64)
        step = np.linalg.norm(np.diff(centers, axis=0), axis=1)
        log.info(
            "%s (%s): %d views %dx%d, frames %s-%s, %.1f m between views (max %.1f),"
            " depth on %.0f%% of pixels, %.1f-%.1f m, %d instances -> %s",
            scene,
            cfg.scenes[scene],
            len(views),
            width,
            height,
            views.names[0],
            views.names[-1],
            np.median(step),
            step.max(),
            100 * (gt.depth > 0).mean(),
            gt.depth[gt.depth > 0].min(),
            gt.depth.max(),
            len(gt.classes),
            path,
        )


if __name__ == "__main__":
    main()
