"""Per-view depth and cameras: what the backbone predicts and the refiner improves."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pathlib import Path

RAY_FIELDS = ("ray_origins", "ray_directions")


@dataclass
class Geometry:
    """Depth maps with cameras, one per view.

    Depth is the distance along the camera's viewing axis (z), at the same
    resolution as the input images. Cameras use OpenCV axes. Backbones that
    predict per-pixel rays also fill the ray fields: a pixel's 3D point is
    origin + depth * direction.
    """

    depth: np.ndarray  # (N, H, W) float32
    confidence: np.ndarray  # (N, H, W) float32, higher is more certain
    intrinsics: np.ndarray  # (N, 3, 3) float32, in pixels
    cam_to_world: np.ndarray  # (N, 4, 4) float32
    ray_origins: np.ndarray | None = None  # (N, H, W, 3) float32
    ray_directions: np.ndarray | None = None  # (N, H, W, 3) float32

    def world_points(self) -> np.ndarray:
        """Lift every pixel to a 3D world point, shape (N, H, W, 3)."""
        _, height, width = self.depth.shape
        # pixel centers; intrinsics follow COLMAP (pixel corners at integers)
        u, v = np.meshgrid(np.arange(width) + 0.5, np.arange(height) + 0.5)
        fx, fy = self.intrinsics[:, 0, 0], self.intrinsics[:, 1, 1]
        cx, cy = self.intrinsics[:, 0, 2], self.intrinsics[:, 1, 2]
        x = (u[None] - cx[:, None, None]) / fx[:, None, None] * self.depth
        y = (v[None] - cy[:, None, None]) / fy[:, None, None] * self.depth
        cam = np.stack([x, y, self.depth], axis=-1)
        rot, trans = self.cam_to_world[:, :3, :3], self.cam_to_world[:, :3, 3]
        return np.einsum("nij,nhwj->nhwi", rot, cam) + trans[:, None, None]

    def save(self, path: Path) -> None:
        """Write to a compressed .npz file."""
        arrays = {
            "depth": self.depth,
            "confidence": self.confidence,
            "intrinsics": self.intrinsics,
            "cam_to_world": self.cam_to_world,
        }
        for name in RAY_FIELDS:
            if getattr(self, name) is not None:
                arrays[name] = getattr(self, name)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **arrays)

    @classmethod
    def load(cls, path: Path) -> Geometry:
        """Read geometry written by `save`."""
        with np.load(path) as f:
            rays = {name: f[name] for name in RAY_FIELDS if name in f}
            return cls(
                depth=f["depth"],
                confidence=f["confidence"],
                intrinsics=f["intrinsics"],
                cam_to_world=f["cam_to_world"],
                **rays,
            )
