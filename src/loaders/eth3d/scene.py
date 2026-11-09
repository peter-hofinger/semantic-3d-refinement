"""ETH3D high-res multi-view scenes: config, COLMAP calibration, DSLR camera model.

ETH3D (Schöps et al., CVPR 2017) ships every DSLR image twice: as taken, calibrated
with COLMAP's THIN_PRISM_FISHEYE model, and undistorted by COLMAP to a PINHOLE camera
with the same pose. Its ground-truth depth maps, rendered from the laser scans, belong
to the pixels of the distorted originals and hold z-depth in metres (checked in E35:
depth over the camera z of the SfM points is 1.000 at every image radius, depth over
ray length falls to 0.82 in the corners). A pixel of the undistorted image therefore
finds its depth through the distortion: pinhole pixel -> normalised (u, v) ->
distorted (u', v') -> distorted pixel.

The distortion is our reimplementation of COLMAP's `ThinPrismFisheyeCameraModel`
(src/colmap/sensor/models.h, BSD-3): with r = |(u, v)| and θ = atan r, the fisheye
radius θ_d = θ (1 + k1 θ² + k2 θ⁴ + k3 θ⁶ + k4 θ⁸) scales (u, v) by θ_d / r, then
  u' = u_f + 2 p1 u_f v_f + p2 (r_f² + 2 u_f²) + sx1 r_f²
  v' = v_f + 2 p2 u_f v_f + p1 (r_f² + 2 v_f²) + sy1 r_f²
on the fisheye coordinates (u_f, v_f), r_f² = u_f² + v_f².

Déjà View assumes a centred principal point (E30); ETH3D's undistorted cameras are about
a pixel off at 504 px, so each image is cropped symmetrically about its principal point.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

from src.loaders.scannetpp.config import PrepareConfig

PINHOLE = "PINHOLE"
FISHEYE = "THIN_PRISM_FISHEYE"
NUM_PARAMS = {PINHOLE: 4, FISHEYE: 12}
ARCHIVE_KINDS = ("dslr_undistorted", "dslr_jpg", "dslr_depth")
POSE_TOL = 1e-6  # both calibrations store the same pose for an image
SMALL_RADIUS = 1e-12  # below this, atan(r) / r = 1 to double precision


@dataclass
class Eth3dConfig:
    """Download URL, paths, scene name -> indoor | outdoor, and settings."""

    base_url: str
    archive_root: Path
    output_root: Path
    scenes: dict[str, str]
    prepare: PrepareConfig = field(default_factory=PrepareConfig)

    def archive(self, scene: str, kind: str) -> Path:
        """Local path of one of a scene's archives, named as on the server."""
        assert kind in ARCHIVE_KINDS, kind
        return self.archive_root / f"{scene}_{kind}.7z"


def load_config(path: Path) -> Eth3dConfig:
    """Read an ETH3D config file; every scene is indoor or outdoor."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    scenes = {str(s): str(kind) for s, kind in raw["scenes"].items()}
    bad = {s: k for s, k in scenes.items() if k not in {"indoor", "outdoor"}}
    if bad:
        msg = f"scenes must be indoor or outdoor: {bad}"
        raise ValueError(msg)
    return Eth3dConfig(
        base_url=str(raw["base_url"]),
        archive_root=Path(raw["archive_root"]),
        output_root=Path(raw["output_root"]),
        scenes=scenes,
        prepare=PrepareConfig(**raw.get("prepare", {})),
    )


@dataclass(frozen=True)
class Camera:
    """A COLMAP camera: model, image size and parameters (focal, centre, distortion)."""

    model: str
    width: int
    height: int
    params: np.ndarray  # (4,) PINHOLE fx fy cx cy, (12,) adds k1 k2 p1 p2 k3 k4 sx1 sy1

    def __post_init__(self) -> None:
        """Check the parameter count of the model."""
        assert self.model in NUM_PARAMS, self.model
        assert self.params.shape == (NUM_PARAMS[self.model],), self.params.shape
        assert np.isfinite(self.params).all()

    @property
    def k(self) -> np.ndarray:
        """3 x 3 intrinsics, pixel corners at integers (COLMAP)."""
        fx, fy, cx, cy = self.params[:4]
        return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])


@dataclass(frozen=True)
class Frame:
    """One registered image: file name, COLMAP camera id and camera-to-world pose."""

    name: str
    camera_id: int
    cam_to_world: np.ndarray  # (4, 4), OpenCV axes


def read_cameras(text: str) -> dict[int, Camera]:
    """Parse COLMAP's cameras.txt."""
    cameras = {}
    for line in text.splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        cid, model, width, height, *params = line.split()
        cameras[int(cid)] = Camera(
            model, int(width), int(height), np.array(params, dtype=np.float64)
        )
    return cameras


def quaternion_to_rotation(q: np.ndarray) -> np.ndarray:
    """Rotation matrix of a unit quaternion (w, x, y, z), Hamilton convention."""
    assert q.shape == (4,)
    assert abs(np.linalg.norm(q) - 1) < 1e-6, q  # noqa: PLR2004 - COLMAP prints 6 digits
    w, x, y, z = q / np.linalg.norm(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def read_images(text: str) -> list[Frame]:
    """Parse COLMAP's images.txt (world-to-camera poses) into frames sorted by name."""
    lines = [line for line in text.splitlines() if not line.startswith("#")]
    frames = []
    for header in lines[0::2]:  # every second line lists the 2D observations
        fields = header.split()
        rotation = quaternion_to_rotation(np.array(fields[1:5], dtype=np.float64))
        translation = np.array(fields[5:8], dtype=np.float64)
        cam_to_world = np.eye(4)
        cam_to_world[:3, :3] = rotation.T
        cam_to_world[:3, 3] = -rotation.T @ translation
        frames.append(Frame(Path(fields[9]).name, int(fields[8]), cam_to_world))
    return sorted(frames, key=lambda f: f.name)


def distort(u: np.ndarray, v: np.ndarray, coeffs: np.ndarray) -> tuple[np.ndarray, ...]:
    """Map normalised pinhole coordinates through THIN_PRISM_FISHEYE's distortion."""
    assert u.shape == v.shape
    assert coeffs.shape == (8,)
    k1, k2, p1, p2, k3, k4, sx1, sy1 = coeffs
    r = np.hypot(u, v)
    theta = np.arctan(r)
    t2 = theta * theta
    theta_d = theta * (1 + t2 * (k1 + t2 * (k2 + t2 * (k3 + t2 * k4))))
    scale = np.where(r > SMALL_RADIUS, theta_d / np.maximum(r, SMALL_RADIUS), 1.0)
    uf, vf = u * scale, v * scale
    rf2 = uf * uf + vf * vf
    ud = uf + 2 * p1 * uf * vf + p2 * (rf2 + 2 * uf * uf) + sx1 * rf2
    vd = vf + 2 * p2 * uf * vf + p1 * (rf2 + 2 * vf * vf) + sy1 * rf2
    return ud, vd


def centred_crop(camera: Camera) -> tuple[int, int, int, int]:
    """Largest window (x0, y0, w, h) centred on the principal point to 0.5 px."""
    assert camera.model == PINHOLE
    cx, cy = camera.params[2:4]
    window = []
    for c, size in ((cx, camera.width), (cy, camera.height)):
        half = int(np.floor(min(c, size - c) - 0.5))
        start = round(c - half)
        assert half > 0
        assert start >= 0
        assert start + 2 * half <= size
        assert abs(start + half - c) <= 0.5  # noqa: PLR2004 - rounding to whole pixels
        window += [start, 2 * half]
    x0, w, y0, h = window
    return x0, y0, w, h


def sample_depth(
    depth: np.ndarray,
    distorted: Camera,
    pinhole: Camera,
    crop: tuple[int, int, int, int],
    size: tuple[int, int],
) -> np.ndarray:
    """Z-depth for each pixel of the cropped, resized pinhole image; 0 = none.

    `depth` belongs to the distorted image; size is (width, height) of the output.
    ETH3D's maps are dotted (laser points splatted on ~1 in 10 pixels), so a pixel
    takes the dot nearest its centre within its footprint, an axis-aligned square
    in the distorted image: one laser sample, never a mix across an edge.
    """
    assert depth.shape == (distorted.height, distorted.width)
    assert distorted.model == FISHEYE
    assert pinhole.model == PINHOLE
    x0, y0, w, h = crop
    new_w, new_h = size
    xs = x0 + (np.arange(new_w) + 0.5) * w / new_w  # pinhole pixel coordinates
    ys = y0 + (np.arange(new_h) + 0.5) * h / new_h
    fx, fy, cx, cy = pinhole.params
    u, v = np.meshgrid((xs - cx) / fx, (ys - cy) / fy)
    ud, vd = distort(u, v, distorted.params[4:])
    dfx, dfy, dcx, dcy = distorted.params[:4]
    xd, yd = ud * dfx + dcx, vd * dfy + dcy  # continuous, pixel corners at integers
    half_x, half_y = 0.5 * w / new_w * dfx / fx, 0.5 * h / new_h * dfy / fy
    cols, rows = np.floor(xd).astype(np.int64), np.floor(yd).astype(np.int64)
    out = np.zeros((new_h, new_w), dtype=np.float32)
    best = np.full((new_h, new_w), np.inf)
    reach_x, reach_y = int(np.ceil(half_x)), int(np.ceil(half_y))
    for dy in range(-reach_y, reach_y + 1):
        for dx in range(-reach_x, reach_x + 1):
            c, r = cols + dx, rows + dy
            ox, oy = c + 0.5 - xd, r + 0.5 - yd  # dot centre relative to the pixel's
            ok = (np.abs(ox) <= half_x) & (np.abs(oy) <= half_y)
            ok &= (c >= 0) & (c < distorted.width) & (r >= 0) & (r < distorted.height)
            value = np.zeros_like(out)
            value[ok] = depth[r[ok], c[ok]]
            dist2 = ox * ox + oy * oy
            take = ok & np.isfinite(value) & (value > 0) & (dist2 < best)
            out[take] = value[take]
            best[take] = dist2[take]
    assert np.isfinite(out).all()
    assert (out >= 0).all()
    return out
