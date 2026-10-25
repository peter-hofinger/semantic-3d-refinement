"""Chain the stages: backbone -> segmentation -> refinement -> point cloud."""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from src.backbone import ConstantDepthBackbone
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
    """Run the pipeline with placeholder stages on prepared views."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("views", type=Path, help="prepared views (.npz)")
    parser.add_argument("--out", type=Path, default=Path("outputs/cloud.ply"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    views = Views.load(args.views)
    pipeline = Pipeline(ConstantDepthBackbone(), EmptySegmenter(), IdentityRefiner())
    result = pipeline.run(views)
    result.cloud.save_ply(args.out)
    log.info("%s: %d points -> %s", views.scene_id, len(result.cloud), args.out)


if __name__ == "__main__":
    main()
