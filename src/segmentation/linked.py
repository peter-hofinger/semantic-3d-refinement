"""SAM 3 per view + cross-view association: the pipeline's segmenter.

SAM 3 image mode finds text-prompted instances in every view on its own; the
association then links masks that show the same object, by reprojecting them with
the backbone's depth and cameras. The defaults are the setting chosen in the SAM 3
study (`experiments/LOG.md`, E17): score threshold 0.3, masks scoring <= 0.5 kept
only if linked to another view, cross-class links at overlap >= 0.7, a direction
counts only if >= 25% of the mask is visible in the other view.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

from src.geometry import Geometry
from src.loaders import Views
from src.segmentation.associate import Associator
from src.segmentation.base import Segmentation
from src.segmentation.sam3 import Sam3ImageSegmenter, load_prompts

# ScanNet++'s 100 most frequent semantic classes (not in git: dataset metadata)
TOP100 = Path("data/scannetpp/metadata/semantic_benchmark/top100.txt")


@runtime_checkable
class PerViewSegmenter(Protocol):
    """Segments every view independently (ids view-local, see `ID_STRIDE`)."""

    def segment(self, views: Views, geometry: Geometry | None = None) -> Segmentation:
        """Return per-view masks with view-local ids and detection scores."""


def working_associator() -> Associator:
    """Association settings of the SAM 3 study's working setting (E17)."""
    return Associator(
        cross_class_overlap=0.7, min_visible_fraction=0.25, keep_linked_below=0.5
    )


@dataclass
class LinkedSegmenter:
    """Per-view masks linked across views: the same id is the same object."""

    per_view: PerViewSegmenter
    associator: Associator = field(default_factory=working_associator)

    def segment(self, views: Views, geometry: Geometry) -> Segmentation:
        """Segment every view, then link the masks with the geometry."""
        return self.associator.associate(self.per_view.segment(views), geometry)


def sam3_segmenter(prompt_files: list[Path] | None = None) -> LinkedSegmenter:
    """SAM 3 image mode at threshold 0.3 + working association.

    Prompts come from text files (one per line, `prompt = class` for synonyms);
    default: ScanNet++'s top-100 classes.
    """
    synonyms = load_prompts(prompt_files or [TOP100])
    sam3 = Sam3ImageSegmenter(prompts=tuple(synonyms), synonyms=synonyms, threshold=0.3)
    return LinkedSegmenter(sam3)
