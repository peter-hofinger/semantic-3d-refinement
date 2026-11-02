"""Chain the stages: backbone -> segmentation -> refinement -> point cloud."""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from src.backbone import CachedBackbone, ConstantDepthBackbone
from src.loaders import Views
from src.output import PointCloud, to_point_cloud
from src.refinement import IdentityRefiner
from src.segmentation import EmptySegmenter

if TYPE_CHECKING:
    from src.backbone import Backbone
    from src.geometry import Geometry
    from src.refinement import Refiner
    from src.segmentation import Segmentation, Segmenter

log = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Everything the pipeline produced, kept for evaluation and ablations."""

    geometry: Geometry  # backbone output
    segmentation: Segmentation
    refined: Geometry  # after refinement
    cloud: PointCloud


@dataclass
class Pipeline:
    """The three swappable stages."""

    backbone: Backbone
    segmenter: Segmenter
    refiner: Refiner

    def run(self, views: Views) -> PipelineResult:
        """Run all stages on one set of views."""
        geometry = self.backbone.predict(views)
        segmentation = self.segmenter.segment(views, geometry)
        refined = self.refiner.refine(views, geometry, segmentation)
        cloud = to_point_cloud(views, refined, segmentation)
        return PipelineResult(geometry, segmentation, refined, cloud)


def main() -> None:
    """Run the pipeline on prepared views; stages chosen on the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("views", type=Path, help="prepared views (.npz)")
    parser.add_argument(
        "--backbone",
        choices=["constant", "dejaview"],
        default="constant",
        help="constant: flat depth + GT cameras (placeholder); dejaview: Déjà View",
    )
    parser.add_argument(
        "--geometry",
        type=Path,
        help="cached backbone output (from dejaview-predict) instead of --backbone",
    )
    parser.add_argument(
        "--segmenter",
        choices=["empty", "sam3"],
        default="empty",
        help="empty: no objects (placeholder); sam3: SAM 3 per view + association",
    )
    parser.add_argument(
        "--prompts",
        type=Path,
        action="append",
        help="sam3: prompt file, repeat to combine; default: ScanNet++ top-100 classes",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/cloud.ply"),
        help="point cloud; the segmentation and refined geometry are saved next to "
        "it as <stem>_segmentation.npz and <stem>_geometry.npz",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)

    views = Views.load(args.views)
    pipeline = Pipeline(
        _backbone(args.backbone, args.geometry),
        _segmenter(args.segmenter, args.prompts),
        IdentityRefiner(),
    )
    result = pipeline.run(views)
    result.cloud.save_ply(args.out)
    result.segmentation.save(args.out.with_name(f"{args.out.stem}_segmentation.npz"))
    result.refined.save(args.out.with_name(f"{args.out.stem}_geometry.npz"))
    log.info(
        "%s: %d points, %d instances -> %s",
        views.scene_id,
        len(result.cloud),
        len(result.segmentation.classes),
        args.out,
    )


def _backbone(name: str, cache: Path | None) -> Backbone:
    """Backbone by name; heavy models are imported only when chosen."""
    if cache is not None:
        return CachedBackbone(cache)
    if name == "dejaview":
        from src.backbone.dejaview import DejaViewBackbone  # noqa: PLC0415

        return DejaViewBackbone()
    return ConstantDepthBackbone()


def _segmenter(name: str, prompt_files: list[Path] | None) -> Segmenter:
    """Segmenter by name; SAM 3 is imported only when chosen."""
    if name == "sam3":
        from src.segmentation.linked import sam3_segmenter  # noqa: PLC0415

        return sam3_segmenter(prompt_files)
    return EmptySegmenter()


if __name__ == "__main__":
    main()
