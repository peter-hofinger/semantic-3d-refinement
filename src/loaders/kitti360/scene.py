"""KITTI-360 scenes: config, calibration, poses, labels and projected LiDAR depth.

KITTI-360 (Liao et al., TPAMI 2022) is a drive through Karlsruhe's suburbs with a
rectified stereo pair, accumulated Velodyne + SICK point clouds per window of frames,
and classes and instances annotated in 3D and transferred to 2D. A scene here is one
window; its views are frames of the left camera.

Camera. `perspective.txt` gives the rectified projection P_rect_00 with pixel centres
at integers; this repo puts pixel corners there (COLMAP), so the principal point
moves by half a pixel. It lies 51 px below the image centre, and Déjà View assumes a
centred one (E30): each image is cropped symmetrically about it, `crop_width` px wide
and as high as the image allows, then resized so the long side is `long_side`.

Ground truth. A pixel's depth is the camera z of the nearest static point that falls
into it and carries the pixel's own class,
  D(p) = min { z_x : x projects into p, class(x) = label(p) },
with label(p) from KITTI-360's 2D semantic map. The class test stands in for an
occlusion test, which a plain z-buffer on a point cloud lacks: without it a pixel of a
moving car takes the road behind it (moving objects are not in the static clouds),
and a pixel of a facade takes whatever was scanned through the gaps of the cloud in
front. Points of the pixel's own class seen through its surface (vegetation behind
vegetation) are not caught. Pixels without such a point have no ground truth.

The class names and which classes are void follow kitti360Scripts' `helpers/labels.py`
(Autonomous Vision Group, MIT).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

DRIVE = "2013_05_28_drive_{}_sync"
MIN_DEPTH = 0.5  # m: nearer points are the car itself or behind the camera
PLY_TYPES = {"float": "<f4", "double": "<f8", "int": "<i4", "uchar": "u1"}
INSTANCE_FACTOR = 1000  # 2D instance map = INSTANCE_FACTOR * class + instance
POSE_COLUMNS = 17  # a row of cam0_to_world.txt: the frame and a 4 x 4 matrix

CLASSES = {
    6: "ground",
    7: "road",
    8: "sidewalk",
    9: "parking",
    10: "rail track",
    11: "building",
    12: "wall",
    13: "fence",
    14: "guard rail",
    15: "bridge",
    16: "tunnel",
    17: "pole",
    18: "polegroup",
    19: "traffic light",
    20: "traffic sign",
    21: "vegetation",
    22: "terrain",
    23: "sky",
    24: "person",
    25: "rider",
    26: "car",
    27: "truck",
    28: "bus",
    29: "caravan",
    30: "trailer",
    31: "train",
    32: "motorcycle",
    33: "bicycle",
    34: "garage",
    35: "gate",
    36: "stop",
    37: "smallpole",
    38: "lamp",
    39: "trash bin",
    40: "vending machine",
    41: "box",
}  # every other id (0-5, 42-44: unlabeled, ego vehicle, unknown object, ...) is void


@dataclass
class PrepareConfig:
    """How a window becomes backbone input."""

    num_frames: int = 20
    stride: int = 2
    crop_width: int = 756
    long_side: int = 504
    patch_size: int = 14


@dataclass
class Kitti360Config:
    """Paths, scene name -> group, and settings."""

    raw_root: Path
    output_root: Path
    scenes: dict[str, str]
    prepare: PrepareConfig = field(default_factory=PrepareConfig)


def load_config(path: Path) -> Kitti360Config:
    """Read a KITTI-360 config file."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return Kitti360Config(
        raw_root=Path(raw["raw_root"]),
        output_root=Path(raw["output_root"]),
        scenes={str(s): str(group) for s, group in raw["scenes"].items()},
        prepare=PrepareConfig(**raw.get("prepare", {})),
    )


def split_scene(scene: str) -> tuple[str, int, int]:
    """Drive folder, first and last frame of a scene named <drive>_<first>_<last>."""
    drive, first, last = scene.split("_")
    assert int(first) < int(last), scene
    return DRIVE.format(drive), int(first), int(last)


def read_intrinsics(text: str) -> tuple[np.ndarray, tuple[int, int]]:
    """K (3, 3) of the left rectified camera, corners at integers; (width, height)."""
    rows = dict(line.split(":", 1) for line in text.splitlines() if ":" in line)
    k = np.array(rows["P_rect_00"].split(), dtype=np.float64).reshape(3, 4)[:, :3]
    assert k[0, 1] == 0 and k[1, 0] == 0 and k[2, 2] == 1, k
    k[:2, 2] += 0.5
    width, height = (round(float(v)) for v in rows["S_rect_00"].split())
    assert 0 < k[0, 2] < width and 0 < k[1, 2] < height, (k, width, height)
    return k, (width, height)


def read_poses(text: str) -> dict[int, np.ndarray]:
    """Frame -> (4, 4) left rectified camera to world, from `cam0_to_world.txt`."""
    table = np.array([line.split() for line in text.splitlines()], dtype=np.float64)
    assert table.shape[1] == POSE_COLUMNS, table.shape
    poses = {int(row[0]): row[1:].reshape(4, 4) for row in table}
    for frame, pose in poses.items():
        r = pose[:3, :3]
        assert np.allclose(r @ r.T, np.eye(3), atol=1e-4), frame
        assert np.allclose(pose[3], [0, 0, 0, 1]), frame
    return poses


def read_ply(path: Path) -> np.ndarray:
    """Vertices of a binary little-endian .ply as a structured array."""
    with path.open("rb") as f:
        head = f.read(4096)
    end = head.index(b"end_header\n") + len(b"end_header\n")
    lines = head[:end].decode("ascii").splitlines()
    assert "format binary_little_endian 1.0" in lines, lines[:3]
    count = next(int(x.split()[2]) for x in lines if x.startswith("element vertex"))
    fields = [x.split()[1:] for x in lines if x.startswith("property")]
    dtype = np.dtype([(name, PLY_TYPES[kind]) for kind, name in fields])
    vertices = np.fromfile(path, dtype=dtype, count=count, offset=end)
    assert len(vertices) == count, (len(vertices), count)
    return vertices


def centred_crop(
    k: np.ndarray, size: tuple[int, int], p: PrepareConfig
) -> tuple[tuple[float, float, float, float], tuple[int, int]]:
    """Crop box (x0, y0, x1, y1) about the principal point and its resized (w, h).

    The box is `crop_width` wide; its height is the largest that stays in the image
    and resizes to a multiple of the patch size. Corners may fall between pixels.
    """
    width, height = size
    cx, cy = k[0, 2], k[1, 2]
    scale = p.long_side / p.crop_width
    assert p.long_side % p.patch_size == 0, p
    assert p.crop_width / 2 <= min(cx, width - cx), (p.crop_width, cx, width)
    room = 2 * min(cy, height - cy)
    new_h = int(room * scale // p.patch_size) * p.patch_size
    assert 0 < new_h <= p.long_side, new_h
    half_w, half_h = p.crop_width / 2, new_h / scale / 2
    return (cx - half_w, cy - half_h, cx + half_w, cy + half_h), (p.long_side, new_h)


def cropped_intrinsics(
    k: np.ndarray, box: tuple[float, float, float, float], size: tuple[int, int]
) -> np.ndarray:
    """K of the crop `box` resized to `size` (width, height)."""
    out = k.copy()
    out[0, 2] -= box[0]
    out[1, 2] -= box[1]
    out[0] *= size[0] / (box[2] - box[0])
    out[1] *= size[1] / (box[3] - box[1])
    return out


def sample_nearest(
    array: np.ndarray, box: tuple[float, float, float, float], size: tuple[int, int]
) -> np.ndarray:
    """Read the source pixel under each centre of the resized crop (no mixing)."""
    new_w, new_h = size
    cols = (box[0] + (np.arange(new_w) + 0.5) * (box[2] - box[0]) / new_w).astype(int)
    rows = (box[1] + (np.arange(new_h) + 0.5) * (box[3] - box[1]) / new_h).astype(int)
    assert cols.min() >= 0 and cols.max() < array.shape[1], (cols.min(), cols.max())
    assert rows.min() >= 0 and rows.max() < array.shape[0], (rows.min(), rows.max())
    return array[rows[:, None], cols[None, :]]


def pick_frames(frames: list[int], num: int, stride: int) -> list[int]:
    """`num` of the sorted `frames`, `stride` apart in list order, centred in it."""
    span = (num - 1) * stride + 1
    assert frames == sorted(frames), "frames must be sorted"
    assert len(frames) >= span, (len(frames), span)
    start = (len(frames) - span) // 2
    return frames[start : start + span : stride]


def render_depth(
    points: np.ndarray,
    classes: np.ndarray,
    cam_to_world: np.ndarray,
    k: np.ndarray,
    labels: np.ndarray,
) -> np.ndarray:
    """Depth (H, W) float32 of the nearest point of each pixel's own class, 0 if none.

    points (P, 3) world, classes (P,) their class ids, labels (H, W) the class of
    every pixel of the view with intrinsics k and pose cam_to_world.
    """
    assert points.shape == (len(points), 3), points.shape
    assert classes.shape == points.shape[:1], (classes.shape, points.shape)
    height, width = labels.shape
    world_to_cam = np.linalg.inv(cam_to_world)
    cam = points @ world_to_cam[:3, :3].T + world_to_cam[:3, 3]
    z = cam[:, 2]
    front = z > MIN_DEPTH
    cam, z, classes = cam[front], z[front], classes[front]
    cols = np.floor(k[0, 0] * cam[:, 0] / z + k[0, 2]).astype(np.int64)
    rows = np.floor(k[1, 1] * cam[:, 1] / z + k[1, 2]).astype(np.int64)
    inside = (cols >= 0) & (cols < width) & (rows >= 0) & (rows < height)
    cols, rows, z, classes = cols[inside], rows[inside], z[inside], classes[inside]
    own = classes == labels[rows, cols]
    depth = np.full(height * width, np.inf)
    np.minimum.at(depth, rows[own] * width + cols[own], z[own])
    depth[~np.isfinite(depth)] = 0
    return depth.reshape(height, width).astype(np.float32)
