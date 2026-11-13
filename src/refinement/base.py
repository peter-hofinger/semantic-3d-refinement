"""Refiner interface and the identity refiner."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from src.geometry import Geometry
from src.loaders import Views
from src.segmentation import Segmentation


@runtime_checkable
class Refiner(Protocol):
    """Improves the backbone geometry using the segmentation."""

    def refine(
        self, views: Views, geometry: Geometry, segmentation: Segmentation
    ) -> Geometry:
        """Return refined geometry with the same shapes as the input."""


class IdentityRefiner:
    """Return the input unchanged: the baseline every refiner is scored against."""

    def refine(
        self,
        views: Views,  # noqa: ARG002
        geometry: Geometry,
        segmentation: Segmentation,  # noqa: ARG002
    ) -> Geometry:
        """Return the geometry unchanged."""
        return geometry
