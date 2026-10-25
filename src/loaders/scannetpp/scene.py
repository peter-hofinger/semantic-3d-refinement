"""Read one ScanNet++ scene: undistorted DSLR images, intrinsics and poses."""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image

if TYPE_CHECKING:
    from pathlib import Path

# The nerfstudio poses use a world frame with x and y swapped and z flipped
# relative to the laser scan, and OpenGL camera axes. These undo both, giving
# OpenCV cameras in the same frame as the ground-truth mesh.
NERFSTUDIO_TO_SCAN = np.array(
    [[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]], dtype=np.float64
)
OPENGL_TO_OPENCV = np.diag([1.0, -1.0, -1.0, 1.0])


@dataclass
class Frame:
    """One DSLR image and its pose."""

    name: str
    cam_to_world: np.ndarray  # (4, 4), OpenCV camera, laser-scan world frame


class ScannetppScene:
    """Paths and loaders for one scene in the official ScanNet++ layout."""

    def __init__(self, data_root: Path, scene_id: str) -> None:
        """Point at data_root/data/<scene_id>."""
        self.scene_id = scene_id
        self.root = data_root / "data" / scene_id

    @property
    def images_dir(self) -> Path:
        """Undistorted DSLR images."""
        return self.root / "dslr" / "resized_undistorted_images"

    @property
    def mesh_path(self) -> Path:
        """Laser-scan mesh, the ground-truth geometry."""
        return self.root / "scans" / "mesh_aligned_0.05.ply"

    @property
    def semantic_mesh_path(self) -> Path:
        """Mesh with semantic labels on the vertices."""
        return self.root / "scans" / "mesh_aligned_0.05_semantic.ply"

    @property
    def segments_anno_path(self) -> Path:
        """Instance and class annotations on mesh segments."""
        return self.root / "scans" / "segments_anno.json"

    @cached_property
    def _transforms(self) -> dict:
        path = self.root / "dslr" / "nerfstudio" / "transforms_undistorted.json"
        return json.loads(path.read_text())

    def intrinsics(self) -> tuple[np.ndarray, tuple[int, int]]:
        """Return the shared pinhole matrix K and the image size (width, height)."""
        t = self._transforms
        k = np.array([[t["fl_x"], 0, t["cx"]], [0, t["fl_y"], t["cy"]], [0, 0, 1]])
        return k, (int(t["w"]), int(t["h"]))

    def frames(self) -> list[Frame]:
        """Return all usable frames in capture order, skipping frames marked bad."""
        frames = [
            Frame(
                name=f["file_path"],
                cam_to_world=NERFSTUDIO_TO_SCAN.T
                @ np.array(f["transform_matrix"])
                @ OPENGL_TO_OPENCV,
            )
            for f in self._transforms["frames"]
            if not f.get("is_bad", False)
        ]
        return sorted(frames, key=lambda f: f.name)

    def load_image(self, name: str) -> np.ndarray:
        """Load one undistorted image as (H, W, 3) uint8."""
        with Image.open(self.images_dir / name) as img:
            return np.asarray(img.convert("RGB"))
