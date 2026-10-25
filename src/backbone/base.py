"""Backbone interface and a placeholder implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import numpy as np

from src.geometry import Geometry

if TYPE_CHECKING:
    from src.loaders import Views


class Backbone(Protocol):
    """Predicts depth and cameras from images."""

    def predict(self, views: Views) -> Geometry:
        """Return one depth map, confidence map and camera per view."""


@dataclass
class ConstantDepthBackbone:
    """Placeholder until Déjà View is wired in: flat depth, ground-truth cameras."""

    depth: float = 2.0

    def predict(self, views: Views) -> Geometry:
        """Return constant depth with the cameras stored in the views."""
        num, height, width = views.images.shape[:3]
        return Geometry(
            depth=np.full((num, height, width), self.depth, dtype=np.float32),
            confidence=np.ones((num, height, width), dtype=np.float32),
            intrinsics=views.intrinsics.copy(),
            cam_to_world=views.cam_to_world.copy(),
        )
