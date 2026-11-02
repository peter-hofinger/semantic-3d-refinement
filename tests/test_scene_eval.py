"""The protocol's ground-truth-camera variant lifts depth as the ground truth does."""

import numpy as np
import torch

from src.analysis.scene_eval import SceneInputs, gt_camera_scene
from src.evaluation.depth import lift
from src.evaluation.labeled import LabeledCloud
from src.geometry import Geometry
from src.loaders.scannetpp.gt import GroundTruth
from src.loaders.views import Views
from src.refinement.monocular import points_from_rays
from src.refinement.multiview import Scene
from src.segmentation.base import Segmentation

H, W = 24, 32


def test_gt_camera_rays_lift_depth_like_the_ground_truth() -> None:
    k = np.array([[[40.0, 0, 13.5], [0, 41.0, 14.0], [0, 0, 1]]], dtype=np.float32)
    c, s = np.cos(0.3), np.sin(0.3)
    c2w = np.eye(4, dtype=np.float32)[None].copy()
    c2w[0, :3, :3] = [[c, 0, s], [0, 1, 0], [-s, 0, c]]
    c2w[0, :3, 3] = [0.5, -1.0, 2.0]
    depth = np.random.default_rng(0).uniform(1, 3, (1, H, W)).astype(np.float32)
    views = Views("toy", ["0"], np.zeros((1, H, W, 3), np.uint8), k, c2w)
    ids = np.zeros((1, H, W), np.int32)
    backbone = torch.from_numpy(depth / 2)  # the backbone's depth, half the scale
    scene = Scene(backbone, torch.zeros(1, H, W, 6), torch.ones(1, H, W), *(
        torch.zeros(1, *n) for n in ((3, 3), (3, 3), (3,))
    ))  # fmt: skip
    inputs = SceneInputs(
        views,
        GroundTruth(ids, depth, {0: "wall"}),
        Geometry(depth, np.ones_like(depth), k, c2w),
        scene,
        Segmentation(ids, {0: "wall"}),
        LabeledCloud(np.zeros((1, 3)), np.zeros(1, dtype=np.int64)),
        (2.0, np.eye(3), np.zeros(3)),
    )
    on_gt = gt_camera_scene(inputs)
    torch.testing.assert_close(on_gt.depth, torch.from_numpy(depth))
    points = points_from_rays(on_gt.depth.double(), on_gt.rays.double()).numpy()
    expected = lift(depth[0].astype(np.float64), k[0].astype(np.float64), c2w[0])
    np.testing.assert_allclose(points[0], expected, atol=1e-5)
