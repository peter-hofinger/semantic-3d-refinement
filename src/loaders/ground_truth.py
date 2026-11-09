"""Per-view ground truth shared by the datasets: instance ids, depth, classes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

NO_MESH = -2  # no ground truth at the pixel (no surface hit): ignored in evaluation


@dataclass
class GroundTruth:
    """Per-view instance ids and depth, in the same format for every dataset."""

    instance_ids: np.ndarray  # (N, H, W) int32, NO_INSTANCE unlabeled, NO_MESH no hit
    depth: np.ndarray  # (N, H, W) float32, z-depth in meters, 0 where no hit
    classes: dict[int, str]  # instance id -> label

    def save(self, path: Path) -> None:
        """Write to a compressed .npz file."""
        ids = np.array(sorted(self.classes), dtype=np.int32)
        np.savez_compressed(
            path,
            instance_ids=self.instance_ids,
            depth=self.depth,
            class_ids=ids,
            class_names=np.array([self.classes[i] for i in ids]),
        )

    @classmethod
    def load(cls, path: Path) -> GroundTruth:
        """Read ground truth written by `save`."""
        with np.load(path) as f:
            names = [str(n) for n in f["class_names"]]
            return cls(
                instance_ids=f["instance_ids"],
                depth=f["depth"],
                classes=dict(zip(f["class_ids"].tolist(), names, strict=True)),
            )
