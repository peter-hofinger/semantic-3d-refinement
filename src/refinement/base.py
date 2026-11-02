"""Refiner interface and a placeholder implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from src.geometry import Geometry
    from src.loaders import Views
    from src.segmentation import Segmentation


class Refiner(Protocol):
    """Improves the backbone geometry using the segmentation."""

    def refine(
        self, views: Views, geometry: Geometry, segmentation: Segmentation
    ) -> Geometry:
        """Return refined geometry with the same shapes as the input."""


class IdentityRefiner:
    """Placeholder until the optimizer is wired in: changes nothing."""

    def refine(
        self,
        views: Views,  # noqa: ARG002
        geometry: Geometry,
        segmentation: Segmentation,  # noqa: ARG002
    ) -> Geometry:
        """Return the geometry unchanged."""
        return geometry
