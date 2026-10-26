"""Segmentation result, segmenter interface and a placeholder implementation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

import numpy as np

if TYPE_CHECKING:
    from pathlib import Path

    from src.geometry import Geometry
    from src.loaders import Views

NO_INSTANCE = -1
ID_STRIDE = 1000  # per-view ids: view index * ID_STRIDE + k (before association)


@dataclass
class Segmentation:
    """Instance masks that are consistent across views.

    The same instance id means the same object in every view.
    """

    instance_ids: np.ndarray  # (N, H, W) int32, NO_INSTANCE where nothing was found
    classes: dict[int, str] = field(default_factory=dict)  # instance id -> class

    @classmethod
    def empty(cls, num: int, height: int, width: int) -> Segmentation:
        """No objects anywhere."""
        return cls(np.full((num, height, width), NO_INSTANCE, dtype=np.int32))

    def save(self, path: Path) -> None:
        """Write to a compressed .npz file."""
        ids = np.array(sorted(self.classes), dtype=np.int32)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            instance_ids=self.instance_ids,
            class_ids=ids,
            class_names=np.array([self.classes[i] for i in ids]),
        )

    @classmethod
    def load(cls, path: Path) -> Segmentation:
        """Read a segmentation written by `save`."""
        with np.load(path) as f:
            names = [str(n) for n in f["class_names"]]
            classes = dict(zip(f["class_ids"].tolist(), names, strict=True))
            return cls(f["instance_ids"], classes)


class Segmenter(Protocol):
    """Finds objects in every view and links them across views."""

    def segment(self, views: Views, geometry: Geometry) -> Segmentation:
        """Return instance masks; `geometry` can be used to link views."""


class EmptySegmenter:
    """Placeholder until SAM 3 is wired in: finds no objects."""

    def segment(self, views: Views, geometry: Geometry) -> Segmentation:  # noqa: ARG002
        """Return an empty segmentation."""
        num, height, width = views.images.shape[:3]
        return Segmentation.empty(num, height, width)
