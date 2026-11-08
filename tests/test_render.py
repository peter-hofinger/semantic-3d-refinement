"""Point splatting: projection, z-buffer, splat size."""

import numpy as np
import torch

from src.analysis.render import Camera, intrinsics, look_at, splat
from src.analysis.scene_figures import shown


def test_the_nearest_point_wins_where_it_projects() -> None:
    camera = look_at(np.zeros(3), np.array([0.0, 0, 1]), np.array([0.0, -1, 0]))
    np.testing.assert_allclose(camera[:, :3], np.eye(3), atol=1e-12)
    k = intrinsics(9, 7, 90.0)
    points = torch.tensor([[0.0, 0, 2], [0.0, 0, 1], [1.0, 0, 1]])
    colors = torch.tensor([[1.0, 0, 0], [0.0, 1, 0], [0.0, 0, 1]])
    image = splat(points, colors, torch.zeros(3), Camera(camera, k, (7, 9)), 0.0)
    np.testing.assert_array_equal(image[3, 4], [0, 1, 0])  # green, nearer than red
    np.testing.assert_array_equal(image[3, 8], [0, 0, 1])  # x / z = 1: right edge
    assert image.sum() == 2


def test_a_splat_covers_its_surface_patch() -> None:
    camera = look_at(np.zeros(3), np.array([0.0, 0, 1]), np.array([0.0, -1, 0]))
    k = intrinsics(41, 41, 90.0)  # f = 20
    point, color = torch.tensor([[0.0, 0, 2]]), torch.ones(1, 3)
    near = splat(point, color, torch.tensor([0.2]), Camera(camera, k, (41, 41)), 0.0)
    assert near[..., 0].sum() == 5 * 5  # half-width 20 * 0.2 / 2 = 2 pixels


def test_the_presentation_shows_only_the_chosen_scene_and_views() -> None:
    assert shown("ai_006_007_cam_00") == ("presentation", (18, 0, 12, 6))
    assert shown("ai_044_001_cam_00")[0] == "debug"
