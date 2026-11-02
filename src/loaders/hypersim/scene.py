"""Read one Hypersim camera trajectory: tone-mapped images, cameras, pixel labels.

Hypersim conventions (github.com/apple/ml-hypersim, README and contrib/mikeroberts3000):
- camera orientations map camera to world, camera axes x right, y up, z backwards
  (OpenGL); positions are in asset units, `meters_per_asset_unit` converts them;
- `depth_meters` is the Euclidean distance to the camera centre, not z-depth;
- `M_cam_from_uv` maps (u, v, 1), with u and v in [-1, 1] (v up), to a camera ray;
  tilt-shift scenes (most of them) have extra terms: split into K and a camera
  rotation (`camera_from_cam_from_uv`); the few whose K would have skew are rejected;
- `semantic` holds NYU40 class ids, `semantic_instance` instance ids, -1 = none.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING

import h5py
import numpy as np
from PIL import Image

if TYPE_CHECKING:
    from pathlib import Path

OPENGL_TO_OPENCV = np.diag([1.0, -1.0, -1.0, 1.0])
SKEW_TOL = 1e-3  # pixels; larger K[0, 1] = skewed tilt-shift camera, rejected

# NYU40 class names, index = class id (0 unused)
NYU40 = (
    "", "wall", "floor", "cabinet", "bed", "chair", "sofa", "table", "door", "window",
    "bookshelf", "picture", "counter", "blinds", "desk", "shelves", "curtain",
    "dresser", "pillow", "mirror", "floor mat", "clothes", "ceiling", "books",
    "refrigerator", "television", "paper", "towel", "shower curtain", "box",
    "whiteboard", "person", "night stand", "toilet", "sink", "lamp", "bathtub", "bag",
    "otherstructure", "otherfurniture", "otherprop",
)  # fmt: skip


@dataclass
class Frame:
    """One rendered frame and its pose."""

    name: str  # "<camera>/frame.<id>", unique within the scene
    frame_id: int
    cam_to_world: np.ndarray  # (4, 4), OpenCV camera, Hypersim world frame in meters


@dataclass
class FrameLabels:
    """Per-pixel ground truth of one frame at full resolution."""

    distance: np.ndarray  # (H, W) float32, meters to camera centre, nan = none
    semantic: np.ndarray  # (H, W) int32, NYU40 class, -1 = none
    instance: np.ndarray  # (H, W) int32, instance id, -1 = none
    entity: np.ndarray  # (H, W) int32, render entity id, -1 = no geometry


def read_hdf5(path: Path) -> np.ndarray:
    """Read the single dataset of a Hypersim .hdf5 file."""
    with h5py.File(path, "r") as f:
        return f["dataset"][()]


def camera_from_cam_from_uv(
    m: np.ndarray, width: int, height: int
) -> tuple[np.ndarray, np.ndarray]:
    """Split Hypersim's `M_cam_from_uv` into a pinhole K and a camera rotation.

    Pixel (x, y, 1) -> ray (OpenCV axes) is the linear map A = F M S, with S taking
    pixels to (u, v, 1) (corners at 0, v up) and F flipping y and z. QR gives
    A = Q U: U is K^-1 up to scale, Q rotates the camera (tilt-shift scenes tilt the
    image plane; plain scenes give Q = I). Returns (K, Q); cam_to_world gets @ Q.
    Cameras whose K would have skew are rejected (Views has no skew-free match).
    """
    s = np.array([[2 / width, 0, -1], [0, -2 / height, 1], [0, 0, 1]])
    q, u = np.linalg.qr(OPENGL_TO_OPENCV[:3, :3] @ m @ s)
    signs = np.diag(np.sign(np.diag(u)))
    q, u = q @ signs, signs @ u
    k = np.linalg.inv(u)
    k /= k[2, 2]
    if abs(k[0, 1]) > SKEW_TOL or np.linalg.det(q) < 0:
        msg = f"camera with skew is not supported: M_cam_from_uv = {m.tolist()}"
        raise ValueError(msg)
    k[0, 1] = 0.0
    return k, q


def distance_to_depth(distance: np.ndarray, k: np.ndarray) -> np.ndarray:
    """Convert distance to the camera centre into z-depth (pixel centres at +0.5)."""
    height, width = distance.shape
    u, v = np.meshgrid(np.arange(width) + 0.5, np.arange(height) + 0.5)
    pixels = np.stack([u, v, np.ones_like(u)], axis=-1)
    ray_length = np.linalg.norm(pixels @ np.linalg.inv(k).T, axis=-1)  # z = 1 rays
    return (distance / ray_length).astype(np.float32)


class HypersimScene:
    """Paths and loaders for one camera trajectory in the official Hypersim layout."""

    def __init__(
        self, data_root: Path, metadata_root: Path, scene: str, camera: str
    ) -> None:
        """Point at data_root/<scene>; `camera` is e.g. "cam_00"."""
        self.scene = scene
        self.camera = camera
        self.scene_id = f"{scene}_{camera}"
        self.root = data_root / scene
        self.metadata_root = metadata_root

    @property
    def detail_dir(self) -> Path:
        """Scene metadata and camera trajectories."""
        return self.root / "_detail"

    def image_path(self, frame_id: int) -> Path:
        """Tone-mapped colour image (the dataset's own tone mapping, JPEG)."""
        folder = f"scene_{self.camera}_final_preview"
        return self.root / "images" / folder / f"frame.{frame_id:04d}.tonemap.jpg"

    def geometry_path(self, frame_id: int, kind: str) -> Path:
        """Per-pixel geometry/label file, kind e.g. "depth_meters" or "semantic"."""
        folder = f"scene_{self.camera}_geometry_hdf5"
        return self.root / "images" / folder / f"frame.{frame_id:04d}.{kind}.hdf5"

    @cached_property
    def meters_per_asset_unit(self) -> float:
        """Scale from Hypersim asset units to meters."""
        with (self.detail_dir / "metadata_scene.csv").open(encoding="utf-8") as f:
            rows = {
                r["parameter_name"]: r["parameter_value"] for r in csv.DictReader(f)
            }
        return float(rows["meters_per_asset_unit"])

    @cached_property
    def _camera_parameters(self) -> dict[str, str]:
        path = self.metadata_root / "metadata_camera_parameters.csv"
        with path.open(encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row["scene_name"] == self.scene:
                    return row
        msg = f"{self.scene} not in {path}"
        raise KeyError(msg)

    @cached_property
    def _camera(self) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
        row = self._camera_parameters
        width = int(float(row["settings_output_img_width"]))
        height = int(float(row["settings_output_img_height"]))
        m = np.array(
            [[float(row[f"M_cam_from_uv_{i}{j}"]) for j in range(3)] for i in range(3)]
        )
        k, q = camera_from_cam_from_uv(m, width, height)
        return k, q, (width, height)

    def intrinsics(self) -> tuple[np.ndarray, tuple[int, int]]:
        """Return the pinhole matrix K and the image size (width, height)."""
        k, _, size = self._camera
        return k, size

    def public_frame_ids(self) -> list[int]:
        """Frames of this trajectory in the public release, in trajectory order."""
        path = self.metadata_root / "metadata_images_split_scene_v1.csv"
        with path.open(encoding="utf-8") as f:
            return sorted(
                int(r["frame_id"])
                for r in csv.DictReader(f)
                if r["scene_name"] == self.scene
                and r["camera_name"] == self.camera
                and r["included_in_public_release"] == "True"
            )

    def frames(self) -> list[Frame]:
        """Return all public frames in trajectory order with OpenCV poses in meters."""
        cam_dir = self.detail_dir / self.camera
        keyframe_ids = read_hdf5(cam_dir / "camera_keyframe_frame_indices.hdf5")
        rotations = read_hdf5(cam_dir / "camera_keyframe_orientations.hdf5")
        positions = read_hdf5(cam_dir / "camera_keyframe_positions.hdf5")
        keyframe_of = {int(f): i for i, f in enumerate(keyframe_ids)}
        tilt = np.eye(4)
        tilt[:3, :3] = self._camera[1]
        frames = []
        for frame_id in self.public_frame_ids():
            k = keyframe_of[frame_id]
            pose = np.eye(4)
            pose[:3, :3] = rotations[k]
            pose[:3, 3] = positions[k] * self.meters_per_asset_unit
            frames.append(
                Frame(
                    name=f"{self.camera}/frame.{frame_id:04d}",
                    frame_id=frame_id,
                    cam_to_world=pose @ OPENGL_TO_OPENCV @ tilt,
                )
            )
        return frames

    def load_image(self, frame_id: int) -> np.ndarray:
        """Load one tone-mapped image as (H, W, 3) uint8."""
        with Image.open(self.image_path(frame_id)) as img:
            return np.asarray(img.convert("RGB"))

    def load_labels(self, frame_id: int) -> FrameLabels:
        """Load distance, NYU40 class, instance and render entity maps of one frame."""
        return FrameLabels(
            distance=read_hdf5(self.geometry_path(frame_id, "depth_meters")).astype(
                np.float32
            ),
            semantic=read_hdf5(self.geometry_path(frame_id, "semantic")).astype(
                np.int32
            ),
            instance=read_hdf5(
                self.geometry_path(frame_id, "semantic_instance")
            ).astype(np.int32),
            entity=read_hdf5(self.geometry_path(frame_id, "render_entity_id")).astype(
                np.int32
            ),
        )
