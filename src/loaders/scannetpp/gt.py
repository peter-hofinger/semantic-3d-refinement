"""Render ground-truth instance ids and depth into prepared views from the scan mesh."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import open3d as o3d

from src.loaders.scannetpp.config import load_config
from src.loaders.scannetpp.scene import ScannetppScene
from src.loaders.views import Views
from src.segmentation.base import NO_INSTANCE

if TYPE_CHECKING:
    from collections.abc import Iterator

log = logging.getLogger(__name__)

NO_MESH = -2  # the pixel's ray misses the mesh: no ground truth, ignore in evaluation


@dataclass
class GroundTruth:
    """Per-view instance ids and depth rendered from the annotated laser scan."""

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


def triangle_instances(
    scene: ScannetppScene, triangles: np.ndarray
) -> tuple[np.ndarray, dict[int, str]]:
    """Label every mesh triangle with the instance most of its vertices belong to."""
    seg_of_vertex = np.asarray(
        json.loads(scene.segments_path.read_text())["segIndices"]
    )
    groups = json.loads(scene.segments_anno_path.read_text())["segGroups"]
    inst_of_seg = np.full(seg_of_vertex.max() + 1, NO_INSTANCE, dtype=np.int32)
    for group in groups:
        inst_of_seg[group["segments"]] = group["objectId"]
    a, b, c = inst_of_seg[seg_of_vertex][triangles].T
    # majority of three: a unless only b and c agree
    return np.where(b == c, b, a), {g["objectId"]: g["label"] for g in groups}


def _pixel_rays(views: Views) -> Iterator[np.ndarray]:
    """Yield (H, W, 6) world rays per view, scaled so the hit distance is z-depth."""
    _, height, width = views.images.shape[:3]
    # pixel centers, COLMAP convention as in Geometry.world_points
    u, v = np.meshgrid(np.arange(width) + 0.5, np.arange(height) + 0.5)
    pixels = np.stack([u, v, np.ones_like(u)], axis=-1)
    for k, c2w in zip(views.intrinsics, views.cam_to_world, strict=True):
        directions = pixels @ np.linalg.inv(k).T @ c2w[:3, :3].T
        origins = np.broadcast_to(c2w[:3, 3], directions.shape)
        yield np.concatenate([origins, directions], axis=-1).astype(np.float32)


def render(scene: ScannetppScene, views: Views) -> GroundTruth:
    """Raycast the mesh from every view and read off instance and depth."""
    mesh = o3d.io.read_triangle_mesh(str(scene.mesh_path))
    tri_instance, classes = triangle_instances(scene, np.asarray(mesh.triangles))
    raycaster = o3d.t.geometry.RaycastingScene()
    raycaster.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))

    ids, depths = [], []
    for rays in _pixel_rays(views):
        hit = raycaster.cast_rays(o3d.core.Tensor(rays))
        tri = hit["primitive_ids"].numpy().astype(np.int64)
        missed = tri == raycaster.INVALID_ID
        ids.append(np.where(missed, NO_MESH, tri_instance[np.where(missed, 0, tri)]))
        depths.append(np.where(missed, 0.0, hit["t_hit"].numpy()))
    return GroundTruth(
        instance_ids=np.stack(ids).astype(np.int32),
        depth=np.stack(depths).astype(np.float32),
        classes=classes,
    )


def main() -> None:
    """Render ground truth for prepared views and save it next to them."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("views", type=Path, nargs="+", help="prepared views (.npz)")
    parser.add_argument("--config", type=Path, default=Path("configs/scannetpp.yaml"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    cfg = load_config(args.config)
    for path in args.views:
        views = Views.load(path)
        gt = render(ScannetppScene(cfg.data_root, views.scene_id), views)
        out = path.with_name(f"gt_{path.stem}.npz")
        gt.save(out)
        ids = gt.instance_ids
        log.info(
            "%s: labeled %.0f%%, unlabeled %.0f%%, no mesh %.0f%%, %d instances -> %s",
            views.scene_id,
            100 * (ids >= 0).mean(),
            100 * (ids == NO_INSTANCE).mean(),
            100 * (ids == NO_MESH).mean(),
            len(np.unique(ids[ids >= 0])),
            out,
        )


if __name__ == "__main__":
    main()
