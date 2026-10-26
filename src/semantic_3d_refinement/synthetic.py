"""Synthetic piecewise-planar room with known ground truth, in the cached-view format."""

import argparse
from pathlib import Path

import numpy as np

ROOM_PLANES = (
    ((0.0, 1.0, 0.0), 1.0),
    ((0.0, -1.0, 0.0), 1.2),
    ((1.0, 0.0, 0.0), 1.5),
    ((-1.0, 0.0, 0.0), 2.0),
    ((0.0, 0.0, 1.0), 4.0),
)


def pinhole_rays(h: int, w: int, focal: float, yaw: float, origin: tuple[float, float, float]) -> np.ndarray:
    """Return an (H, W, 6) ray map: camera origin and world direction with unit z in the camera frame."""
    v, u = np.mgrid[0:h, 0:w].astype(np.float64)
    dirs_cam = np.stack([(u + 0.5 - w / 2) / focal, (v + 0.5 - h / 2) / focal, np.ones_like(u)], axis=-1)
    c, s = np.cos(yaw), np.sin(yaw)
    rot = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
    dirs = dirs_cam @ rot.T
    origins = np.broadcast_to(np.asarray(origin), dirs.shape)
    return np.concatenate([origins, dirs], axis=-1)


def ray_depth(rays: np.ndarray, planes: tuple[tuple[tuple[float, float, float], float], ...]) -> np.ndarray:
    """Nearest positive intersection with planes n.x = c along origin + t * direction (convex room interior)."""
    o, d = rays[..., :3], rays[..., 3:]
    ts = []
    for n, c in planes:
        n_arr = np.asarray(n)
        denom = d @ n_arr
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (c - o @ n_arr) / denom
        ts.append(np.where(denom > 0, t, np.inf))
    depth = np.min(ts, axis=0)
    assert np.isfinite(depth).all() and (depth > 0).all()
    return depth


def room_scene(  # noqa: PLR0913
    *,
    h: int = 96,
    w: int = 128,
    noise: float = 0.02,
    outlier_frac: float = 0.01,
    planes: tuple[tuple[tuple[float, float, float], float], ...] = ROOM_PLANES,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """Noisy inverse depth plus low-confidence outliers on a known planar scene."""
    assert noise >= 0 and 0 <= outlier_frac < 1
    rng = np.random.default_rng(seed)
    rays = pinhole_rays(h, w, focal=0.8 * w, yaw=0.3, origin=(0.1, -0.1, 0.2))
    gt_depth = ray_depth(rays, planes)
    inv = (1 / gt_depth) * (1 + noise * rng.standard_normal((h, w)))
    outliers = rng.random((h, w)) < outlier_frac
    inv = np.where(outliers, inv * rng.uniform(0.5, 1.5, (h, w)), inv)
    conf = np.where(outliers, 0.1, 1.0)
    depth = 1 / inv
    assert np.isfinite(depth).all() and (depth > 0).all()
    return {
        k: v.astype(np.float32)
        for k, v in {
            "depth": depth,
            "rays": rays,
            "conf": conf,
            "gt_depth": gt_depth,
        }.items()
    }


def main() -> None:
    """Write the room scene: `uv run python -m semantic_3d_refinement.synthetic --out data/synthetic_room.npz`."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("data/synthetic_room.npz"))
    out: Path = parser.parse_args().out
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **room_scene())
    print(f"wrote {out}")  # noqa: T201


if __name__ == "__main__":
    main()
