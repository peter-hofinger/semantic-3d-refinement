"""Tanks and Temples scenes: config, cameras and depth rendered from the laser scan.

Tanks and Temples (Knapitsch et al., SIGGRAPH 2017) gives, per scene, video frames,
a laser scan as a point cloud in metres (about 2 mm between points), camera poses
from COLMAP in COLMAP's own frame and scale (`<Scene>_COLMAP_SfM.log`: per image an
index line and a 4 x 4 camera-to-world matrix) and a similarity A = [sR | t] that
maps that frame onto the scan (`<Scene>_trans.txt`). There are no labels.

Camera. A camera-to-world pose [R_c | c] in COLMAP's frame becomes [R R_c | s R c + t]
in the scan's, with camera coordinates in metres. The dataset ships no intrinsics.
A pinhole with f = `focal` x image width and a centred principal point is assumed and
checked against the scan (E45b: over 8 views of each of the three scenes used, the
scan's projected colours match the image best at f = 0.60 W, to about 0.005 W).

Ground truth. A view's depth is rendered from the points; a point cloud has no
surface, so two things are guarded against. Stray points: with z_1 <= z_2 <= ... the
depths of the points that fall into pixel p, its front is the first depth that
SUPPORT points share,
  z_f(p) = min { z_i : z_{i + SUPPORT - 1} <= (1 + SHELL) z_i }.
Mixing: the pixel takes the depth of one point, the one nearest its centre among
those within the shell [z_f, (1 + SHELL) z_f], never a mean across a depth edge. A
pixel without such a front has no ground truth. What this cannot catch is a surface
the scanner never saw: the points behind it then show through.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

MIN_DEPTH = 0.05  # m: nearer points are behind the camera or on the lens
SUPPORT = 3  # points that must share a depth for it to count as a surface
SHELL = 0.01  # relative depth range those points, and the pixel's sample, lie in
CHUNK = 2_000_000  # points projected at a time: bounds the memory of a large scan
LOG_BLOCK = 5  # lines per camera in a .log file: an index line and a 4 x 4 matrix
SIMILARITY_TOL = 1e-6  # departure from s x rotation allowed in the alignment


@dataclass
class PrepareConfig:
    """How views are picked and resized, and the assumed focal length."""

    num_frames: int = 20
    sampling: str = "uniform"
    long_side: int = 504
    patch_size: int = 14
    focal: float = 0.6


@dataclass
class TanksAndTemplesConfig:
    """Paths, scene name -> group, and settings."""

    raw_root: Path
    output_root: Path
    scenes: dict[str, str]
    prepare: PrepareConfig = field(default_factory=PrepareConfig)


def load_config(path: Path) -> TanksAndTemplesConfig:
    """Read a Tanks and Temples config file."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return TanksAndTemplesConfig(
        raw_root=Path(raw["raw_root"]),
        output_root=Path(raw["output_root"]),
        scenes={str(s): str(group) for s, group in raw["scenes"].items()},
        prepare=PrepareConfig(**raw.get("prepare", {})),
    )


def read_log(text: str) -> np.ndarray:
    """Camera-to-world poses (N, 4, 4) of a .log trajectory, in file order."""
    lines = text.split("\n")
    count = len(lines) // LOG_BLOCK
    assert count > 0, "empty trajectory"
    poses = np.empty((count, 4, 4))
    for i in range(count):
        block = lines[LOG_BLOCK * i : LOG_BLOCK * (i + 1)]
        assert int(block[0].split()[0]) == i, block[0]
        poses[i] = np.array([row.split() for row in block[1:]], dtype=np.float64)
    assert np.isfinite(poses).all()
    assert np.allclose(poses[:, 3], [0, 0, 0, 1]), "last row of a pose"
    return poses


def to_scan_frame(poses: np.ndarray, alignment: np.ndarray) -> np.ndarray:
    """Rigid camera-to-world poses (N, 4, 4) in the scan's frame, in metres."""
    assert poses.shape[1:] == (4, 4), poses.shape
    assert alignment.shape == (4, 4), alignment.shape
    linear = alignment[:3, :3]
    scale = np.cbrt(np.linalg.det(linear))
    assert scale > 0, scale
    rotation = linear / scale
    off = np.abs(rotation @ rotation.T - np.eye(3)).max()
    assert off < SIMILARITY_TOL, f"alignment is not a similarity: {off}"
    out = np.tile(np.eye(4), (len(poses), 1, 1))
    out[:, :3, :3] = rotation @ poses[:, :3, :3]
    out[:, :3, 3] = poses[:, :3, 3] @ linear.T + alignment[:3, 3]
    ortho = np.abs(out[:, :3, :3] @ out[:, :3, :3].transpose(0, 2, 1) - np.eye(3))
    assert ortho.max() < 1e-5, f"pose is not rigid: {ortho.max()}"  # noqa: PLR2004 - the .log prints 12 digits
    return out


def centred_view(
    size: tuple[int, int], p: PrepareConfig
) -> tuple[tuple[float, float, float, float], tuple[int, int], np.ndarray]:
    """Crop box (x0, y0, x1, y1), resized (width, height) and K of the resized crop.

    The full width is kept and resized to `long_side`; the height is the largest
    multiple of the patch size that fits, cut symmetrically about the image centre,
    so one scale applies to both axes and the principal point stays centred.
    """
    width, height = size
    assert width >= height, size
    assert p.long_side % p.patch_size == 0, p
    scale = p.long_side / width
    new_h = int(height * scale // p.patch_size) * p.patch_size
    assert new_h > 0, new_h
    half = new_h / scale / 2
    box = (0.0, height / 2 - half, float(width), height / 2 + half)
    f = p.focal * p.long_side
    k = np.array([[f, 0, p.long_side / 2], [0, f, new_h / 2], [0, 0, 1]])
    return box, (p.long_side, new_h), k


def render_depth(
    points: np.ndarray, cam_to_world: np.ndarray, k: np.ndarray, size: tuple[int, int]
) -> np.ndarray:
    """Depth (H, W) float32 of the supported front point nearest each pixel centre.

    points (P, 3) in the world of the rigid pose cam_to_world; size is (width,
    height) of the view with intrinsics k (pixel corners at integers); 0 = none.
    """
    assert points.shape == (len(points), 3), points.shape
    width, height = size
    world_to_cam = np.linalg.inv(cam_to_world)
    seen = []
    for start in range(0, len(points), CHUNK):
        cam = points[start : start + CHUNK] @ world_to_cam[:3, :3].T
        cam += world_to_cam[:3, 3]
        cam = cam[cam[:, 2] > MIN_DEPTH]
        x = k[0, 0] * cam[:, 0] / cam[:, 2] + k[0, 2]
        y = k[1, 1] * cam[:, 1] / cam[:, 2] + k[1, 2]
        inside = (x >= 0) & (x < width) & (y >= 0) & (y < height)
        seen.append(np.stack([x[inside], y[inside], cam[inside, 2]]))
    x, y, z = np.concatenate(seen, axis=1)
    cols, rows = np.floor(x).astype(np.int64), np.floor(y).astype(np.int64)
    pixel = rows * width + cols
    order = np.lexsort((z, pixel))
    pixel, z = pixel[order], z[order]
    offset = (x[order] - cols[order] - 0.5) ** 2 + (y[order] - rows[order] - 0.5) ** 2

    depth = np.zeros(height * width)
    lead = len(z) - SUPPORT + 1
    if lead > 0:
        last = slice(SUPPORT - 1, None)
        supported = (pixel[last] == pixel[:lead]) & (z[last] <= (1 + SHELL) * z[:lead])
        start = np.flatnonzero(supported)
        start = start[np.unique(pixel[start], return_index=True)[1]]
        front = np.zeros(height * width)
        front[pixel[start]] = z[start]
        shell = (z >= front[pixel]) & (z <= (1 + SHELL) * front[pixel])
        pixel, z, offset = pixel[shell], z[shell], offset[shell]
        nearest = np.lexsort((offset, pixel))
        nearest = nearest[np.unique(pixel[nearest], return_index=True)[1]]
        depth[pixel[nearest]] = z[nearest]
    assert np.isfinite(depth).all()
    assert (depth >= 0).all()
    return depth.reshape(height, width).astype(np.float32)
