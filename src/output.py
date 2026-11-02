"""Turn refined depth and cameras into a labeled point cloud."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pathlib import Path

    from src.geometry import Geometry
    from src.loaders import Views
    from src.segmentation import Segmentation


@dataclass
class PointCloud:
    """Fused points from all views, each with color and instance."""

    points: np.ndarray  # (M, 3) float32, world frame
    colors: np.ndarray  # (M, 3) uint8
    instance_ids: np.ndarray  # (M,) int32, -1 where no object
    classes: dict[int, str] = field(default_factory=dict)  # instance id -> class

    def __len__(self) -> int:
        """Return the number of points."""
        return len(self.points)

    def save_ply(self, path: Path) -> None:
        """Write a binary PLY with xyz, rgb and an instance id per point."""
        vertex = np.empty(
            len(self),
            dtype=[
                ("x", "<f4"),
                ("y", "<f4"),
                ("z", "<f4"),
                ("red", "u1"),
                ("green", "u1"),
                ("blue", "u1"),
                ("instance", "<i4"),
            ],
        )
        vertex["x"], vertex["y"], vertex["z"] = self.points.T
        vertex["red"], vertex["green"], vertex["blue"] = self.colors.T
        vertex["instance"] = self.instance_ids
        header = (
            "ply\nformat binary_little_endian 1.0\n"
            f"element vertex {len(self)}\n"
            "property float x\nproperty float y\nproperty float z\n"
            "property uchar red\nproperty uchar green\nproperty uchar blue\n"
            "property int instance\nend_header\n"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            f.write(header.encode())
            f.write(vertex.tobytes())


def to_point_cloud(
    views: Views,
    geometry: Geometry,
    segmentation: Segmentation,
    min_confidence: float = 0.0,
) -> PointCloud:
    """Lift all pixels with valid depth and enough confidence to 3D."""
    keep = (geometry.depth > 0) & (geometry.confidence >= min_confidence)
    return PointCloud(
        points=geometry.world_points()[keep].astype(np.float32),
        colors=views.images[keep],
        instance_ids=segmentation.instance_ids[keep],
        classes=dict(segmentation.classes),
    )
