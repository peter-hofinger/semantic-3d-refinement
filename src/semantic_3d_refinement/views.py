"""Loading one cached view (backbone export or synthetic scene) and run bookkeeping."""

import subprocess
from pathlib import Path

import numpy as np
import torch
from torch import Tensor


def check_inputs(depth: Tensor, rays: Tensor, conf: Tensor) -> None:
    """Assert shapes, dtypes and ranges of a single view."""
    assert depth.ndim == 2, depth.shape  # noqa: PLR2004
    h, w = depth.shape
    assert h >= 3 and w >= 3, depth.shape  # noqa: PLR2004
    assert rays.shape == (h, w, 6), rays.shape
    assert conf.shape == (h, w), conf.shape
    assert depth.dtype == rays.dtype == conf.dtype and depth.dtype.is_floating_point
    assert depth.isfinite().all() and (depth > 0).all()
    assert rays.isfinite().all()
    assert conf.isfinite().all() and (conf >= 0).all() and (conf > 0).any()


def fit_pinhole(rays: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    """Replace a predicted ray map by the pinhole that fits it best, keeping the camera's center and rotation.

    Least squares for fx, cx and fy, cy on the camera-frame directions; the directions keep unit z in the camera.
    Déjà View's rays deviate from it by 0.6-0.9 px (median) on ScanNet++, in patch-sized steps that show as a 14 px
    grid in every normal map lifted through them, and by up to 2.7 px on ETH3D office (1.8 px even with a free
    rotation, so the rays themselves are less pinhole-like there).
    """
    h, w = rays.shape[:2]
    d = rays[..., 3:] @ rotation
    x, y = d[..., 0] / d[..., 2], d[..., 1] / d[..., 2]
    v, u = np.mgrid[0:h, 0:w].astype(np.float64)
    fx, cx = np.polyfit(x.ravel(), u.ravel(), 1)
    fy, cy = np.polyfit(y.ravel(), v.ravel(), 1)
    residual = np.hypot(fx * x + cx - u, fy * y + cy - v)
    assert np.median(residual) < 3, np.median(residual)  # noqa: PLR2004
    dirs = np.stack([(u - cx) / fx, (v - cy) / fy, np.ones_like(u)], axis=-1) @ rotation.T
    origin = np.broadcast_to(rays[..., :3].reshape(-1, 3).mean(0), dirs.shape)
    return np.concatenate([origin, dirs], axis=-1)


def load_view(path: Path, view: int | None = None, rays: str = "predicted") -> dict[str, Tensor]:
    """Load one cached view as (H, W[, C]) float tensors: depth, rays, conf, and gt_depth when known.

    Accepts the synthetic single-view format and the Déjà View export (ray_origins, ray_directions, confidence),
    from which `view` picks one frame. rays = "pinhole" replaces its ray map by the best-fitting pinhole.
    """
    assert rays in {"predicted", "pinhole"}, rays
    data = dict(np.load(path))
    if "ray_origins" in data:
        assert view is not None, "a Déjà View export holds several frames; set `view` in the config"
        ray_map = np.concatenate([data["ray_origins"][view], data["ray_directions"][view]], axis=-1)
        if rays == "pinhole":
            ray_map = fit_pinhole(ray_map.astype(np.float64), data["cam_to_world"][view, :3, :3].astype(np.float64))
        data = {"depth": data["depth"][view], "rays": ray_map, "conf": data["confidence"][view]}
    else:
        assert rays == "predicted", "synthetic scenes are already pinhole"
    data.setdefault("conf", np.ones_like(data["depth"]))
    out = {k: torch.from_numpy(np.ascontiguousarray(v)).float() for k, v in data.items()}
    check_inputs(out["depth"], out["rays"], out["conf"])
    return out


def view_from_config(raw: dict) -> dict[str, Tensor]:
    """Load the view a run config names (`input`, `view`, `rays`)."""
    return load_view(Path(raw["input"]), raw.get("view"), raw.get("rays", "predicted"))


def write_ply(path: Path, points: np.ndarray) -> None:
    """Write an (N, 3) float32 point cloud as binary PLY."""
    assert points.ndim == 2 and points.shape[1] == 3, points.shape  # noqa: PLR2004
    header = f"ply\nformat binary_little_endian 1.0\nelement vertex {len(points)}\n"
    header += "property float x\nproperty float y\nproperty float z\nend_header\n"
    with path.open("wb") as f:
        f.write(header.encode())
        f.write(points.astype("<f4").tobytes())


def git_hash() -> str:
    """Return HEAD, marked dirty if the tree has uncommitted changes."""
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()  # noqa: S607
    dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, check=True).stdout.strip()  # noqa: S607
    return head + ("-dirty" if dirty else "")
