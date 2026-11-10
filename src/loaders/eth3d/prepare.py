"""Turn ETH3D scenes into backbone input and ground truth.

Writes, per scene, `views_n<N>_<sampling>_<long side>.npz` (Views) and
`gt_<views stem>.npz` (GroundTruth) into <output_root>/<scene>/, the layout of the
Hypersim loader. Frames are picked by `select_frames` in file-name order (capture
order). Each undistorted image is cropped about its principal point (see scene.py)
and resized; its ground truth is the laser depth under each resized pixel centre,
looked up through the distortion of the original image (nearest, no mixing across
edges). ETH3D has no labels: a pixel with depth is unlabeled, one without is NO_MESH.
"""

from __future__ import annotations

import argparse
import logging
import tempfile
from pathlib import Path

import numpy as np
import py7zr
from PIL import Image

from src.loaders.eth3d.scene import (
    FISHEYE,
    PINHOLE,
    POSE_TOL,
    Eth3dConfig,
    Frame,
    centred_crop,
    load_config,
    read_cameras,
    read_images,
    sample_depth,
)
from src.loaders.frames import resize_view, select_frames, target_size
from src.loaders.ground_truth import NO_MESH, GroundTruth
from src.loaders.views import Views
from src.segmentation.base import NO_INSTANCE

log = logging.getLogger(__name__)

UNDISTORTED_CALIB = "dslr_calibration_undistorted"
DISTORTED_CALIB = "dslr_calibration_jpg"
UNDISTORTED_IMAGES = "images/dslr_images_undistorted"
DEPTH = "ground_truth_depth/dslr_images"


def extract(archive: Path, wanted: list[str], dest: Path) -> dict[str, Path]:
    """Unpack the members whose path below the scene folder is in `wanted`."""
    with py7zr.SevenZipFile(archive) as z:
        by_suffix = {n.split("/", 1)[1]: n for n in z.getnames() if "/" in n}
        missing = sorted(set(wanted) - set(by_suffix))
        if missing:
            msg = f"{archive.name}: missing {missing[:5]}"
            raise FileNotFoundError(msg)
        z.extract(path=dest, targets=[by_suffix[w] for w in wanted])
    return {w: dest / by_suffix[w] for w in wanted}


def same_pose(a: Frame, b: Frame) -> bool:
    """Whether two calibrations give an image the same pose and camera."""
    return a.camera_id == b.camera_id and np.allclose(
        a.cam_to_world, b.cam_to_world, atol=POSE_TOL
    )


def build(cfg: Eth3dConfig, scene: str, tmp: Path) -> tuple[Views, GroundTruth]:
    """Select frames, crop and resize the images, look up their laser depth."""
    calib = [f"{UNDISTORTED_CALIB}/cameras.txt", f"{UNDISTORTED_CALIB}/images.txt"]
    files = extract(cfg.archive(scene, "dslr_undistorted"), calib, tmp)
    pinholes = read_cameras(files[calib[0]].read_text())
    frames = read_images(files[calib[1]].read_text())
    centers = np.stack([f.cam_to_world[:3, 3] for f in frames])
    p = cfg.prepare
    picked = [frames[i] for i in select_frames(centers, p.num_frames, p.sampling)]

    dcalib = [f"{DISTORTED_CALIB}/cameras.txt", f"{DISTORTED_CALIB}/images.txt"]
    dfiles = extract(cfg.archive(scene, "dslr_jpg"), dcalib, tmp)
    fisheyes = read_cameras(dfiles[dcalib[0]].read_text())
    dframes = {f.name: f for f in read_images(dfiles[dcalib[1]].read_text())}

    names = [f.name for f in picked]
    images = extract(
        cfg.archive(scene, "dslr_undistorted"),
        [f"{UNDISTORTED_IMAGES}/{n}" for n in names],
        tmp,
    )
    depths = extract(
        cfg.archive(scene, "dslr_depth"), [f"{DEPTH}/{n}" for n in names], tmp
    )

    out_images, out_k, out_depth = [], [], []
    for frame in picked:
        assert same_pose(frame, dframes[frame.name]), frame.name
        pinhole, fisheye = pinholes[frame.camera_id], fisheyes[frame.camera_id]
        assert pinhole.model == PINHOLE, pinhole.model
        assert fisheye.model == FISHEYE, fisheye.model
        x0, y0, w, h = crop = centred_crop(pinhole)
        image = np.asarray(
            Image.open(images[f"{UNDISTORTED_IMAGES}/{frame.name}"]).convert("RGB")
        )
        assert image.shape == (pinhole.height, pinhole.width, 3), image.shape
        k = pinhole.k
        k[0, 2] -= x0
        k[1, 2] -= y0
        size = target_size(w, h, p.long_side, p.patch_size)
        resized, k_scaled = resize_view(image[y0 : y0 + h, x0 : x0 + w], k, size)
        raw = np.fromfile(depths[f"{DEPTH}/{frame.name}"], dtype=np.float32)
        assert raw.size == fisheye.width * fisheye.height, (raw.size, frame.name)
        depth = raw.reshape(fisheye.height, fisheye.width)
        out_images.append(resized)
        out_k.append(k_scaled)
        out_depth.append(sample_depth(depth, fisheye, pinhole, crop, size))

    depth = np.stack(out_depth)
    views = Views(
        scene_id=scene,
        names=[Path(n).stem for n in names],
        images=np.stack(out_images),
        intrinsics=np.stack(out_k).astype(np.float32),
        cam_to_world=np.stack([f.cam_to_world for f in picked]).astype(np.float32),
    )
    ids = np.where(depth > 0, NO_INSTANCE, NO_MESH).astype(np.int32)
    return views, GroundTruth(instance_ids=ids, depth=depth, classes={})


def output_path(cfg: Eth3dConfig, scene: str) -> Path:
    """Where the views of a scene are written; the name encodes the settings."""
    p = cfg.prepare
    return (
        cfg.output_root
        / scene
        / f"views_n{p.num_frames}_{p.sampling}_{p.long_side}.npz"
    )


def main() -> None:
    """Prepare the chosen scenes (default: all of the config) from their archives."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/eth3d.yaml"))
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
        with tempfile.TemporaryDirectory() as tmp:
            views, gt = build(cfg, scene, Path(tmp))
        path = output_path(cfg, scene)
        views.save(path)
        gt.save(path.with_name(f"gt_{path.stem}.npz"))
        height, width = views.images.shape[1:3]
        k = views.intrinsics
        log.info(
            "%s (%s): %d views %dx%d, depth on %.0f%% of pixels, %.1f-%.1f m, principal"
            " point off centre by <= %.2f px -> %s",
            scene,
            cfg.scenes[scene],
            len(views),
            width,
            height,
            100 * (gt.depth > 0).mean(),
            gt.depth[gt.depth > 0].min(),
            gt.depth[gt.depth > 0].max(),
            max(
                np.abs(k[:, 0, 2] - width / 2).max(),
                np.abs(k[:, 1, 2] - height / 2).max(),
            ),
            path,
        )


if __name__ == "__main__":
    main()
