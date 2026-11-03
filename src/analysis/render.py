"""Render point clouds as images: z-buffered square splats from a pinhole camera.

Every point covers the patch of surface its source pixel saw, so its splat grows as the
camera comes closer and the cloud renders without holes where it is dense. Shading is
Lambert under a headlight on a gray albedo; it shows the geometry that image colors
hide.
"""

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

MAX_RADIUS = 4  # pixels; nearer splats are clipped to this half-width
NEAR = 0.05  # m, points nearer to the camera are not drawn


def look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    """World-to-camera (3, 4) of a camera at eye looking at target (OpenCV axes)."""
    forward = (target - eye) / np.linalg.norm(target - eye)
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    rotation = np.stack([right, down, forward])
    return np.concatenate([rotation, -(rotation @ eye)[:, None]], axis=1)


def intrinsics(width: int, height: int, fov_deg: float) -> np.ndarray:
    """Pinhole K with the given horizontal field of view, centered."""
    f = 0.5 * width / np.tan(np.radians(fov_deg) / 2)
    return np.array([[f, 0, (width - 1) / 2], [0, f, (height - 1) / 2], [0, 0, 1]])


@dataclass(frozen=True)
class Camera:
    """A pinhole: world-to-camera (3, 4), K (3, 3), image size (height, width)."""

    world_to_cam: np.ndarray
    k: np.ndarray
    size: tuple[int, int]


def splat(
    points: Tensor,
    colors: Tensor,
    radii: Tensor,
    camera: Camera,
    background: float = 1.0,
) -> np.ndarray:
    """Render points (N, 3) with colors (N, 3) in [0, 1] and world radii (N,) in m.

    A point is drawn as a square of half-width f r / z pixels (at most MAX_RADIUS);
    the nearest fragment of each pixel wins. Returns an (H, W, 3) image.
    """
    k, (h, w) = camera.k, camera.size
    rt = torch.as_tensor(camera.world_to_cam, dtype=points.dtype, device=points.device)
    x = points @ rt[:, :3].T + rt[:, 3]
    z = x[:, 2]
    front = z > NEAR
    x, z, colors, radii = x[front], z[front], colors[front], radii[front]
    f = float(k[0, 0])
    u = (f * x[:, 0] / z + float(k[0, 2])).round().long()
    v = (float(k[1, 1]) * x[:, 1] / z + float(k[1, 2])).round().long()
    half = (f * radii / z).round().long().clamp(0, MAX_RADIUS)
    pixel, depth, color = [], [], []
    for r in range(MAX_RADIUS + 1):
        mine = half == r
        if not mine.any():
            continue
        offsets = torch.arange(-r, r + 1, device=points.device)
        du, dv = torch.meshgrid(offsets, offsets, indexing="xy")
        uu = (u[mine, None] + du.reshape(1, -1)).reshape(-1)
        vv = (v[mine, None] + dv.reshape(1, -1)).reshape(-1)
        n = du.numel()
        inside = (uu >= 0) & (uu < w) & (vv >= 0) & (vv < h)
        pixel.append((vv * w + uu)[inside])
        depth.append(z[mine].repeat_interleave(n)[inside])
        color.append(colors[mine].repeat_interleave(n, dim=0)[inside])
    image = torch.full((h * w, 3), background, dtype=colors.dtype, device=colors.device)
    if pixel:
        pixel_t, depth_t, color_t = torch.cat(pixel), torch.cat(depth), torch.cat(color)
        nearest = torch.full((h * w,), torch.inf, dtype=z.dtype, device=z.device)
        nearest.scatter_reduce_(0, pixel_t, depth_t, "amin")
        won = depth_t == nearest[pixel_t]
        image[pixel_t[won]] = color_t[won]
    return image.view(h, w, 3).cpu().numpy()


def headlight(normals: Tensor, points: Tensor, eye: np.ndarray) -> Tensor:
    """Gray Lambert shading (N, 3) under a light at the camera, two-sided."""
    to_eye = torch.as_tensor(eye, dtype=points.dtype, device=points.device) - points
    to_eye = to_eye / to_eye.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    lit = 0.2 + 0.75 * (normals * to_eye).sum(-1).abs()
    return lit[:, None].expand(-1, 3)
