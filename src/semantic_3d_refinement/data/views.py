"""A set of posed views: the input format for every pipeline stage."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pathlib import Path


@dataclass
class Views:
    """Images with intrinsics and poses.

    Poses are camera-to-world, OpenCV camera axes (x right, y down, z forward).
    """

    scene_id: str
    names: list[str]
    images: np.ndarray  # (N, H, W, 3) uint8
    intrinsics: np.ndarray  # (N, 3, 3) float32, in pixels
    cam_to_world: np.ndarray  # (N, 4, 4) float32

    def __len__(self) -> int:
        """Return the number of views."""
        return len(self.names)

    def save(self, path: Path) -> None:
        """Write the views to a compressed .npz file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            scene_id=self.scene_id,
            names=np.array(self.names),
            images=self.images,
            intrinsics=self.intrinsics,
            cam_to_world=self.cam_to_world,
        )

    @classmethod
    def load(cls, path: Path) -> Views:
        """Read views written by `save`."""
        with np.load(path) as f:
            return cls(
                scene_id=str(f["scene_id"]),
                names=[str(n) for n in f["names"]],
                images=f["images"],
                intrinsics=f["intrinsics"],
                cam_to_world=f["cam_to_world"],
            )
