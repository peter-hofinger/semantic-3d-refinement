"""Synthetic piecewise-planar scenes with known ground truth, as cached views."""

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


def pinhole_rays(
    h: int, w: int, focal: float, yaw: float, origin: tuple[float, float, float]
) -> np.ndarray:
    """Return an (H, W, 6) ray map: origin and world direction, unit z in the camera."""
    v, u = np.mgrid[0:h, 0:w].astype(np.float64)
    dirs_cam = np.stack(
        [(u + 0.5 - w / 2) / focal, (v + 0.5 - h / 2) / focal, np.ones_like(u)], axis=-1
    )
    c, s = np.cos(yaw), np.sin(yaw)
    rot = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
    dirs = dirs_cam @ rot.T
    origins = np.broadcast_to(np.asarray(origin), dirs.shape)
    return np.concatenate([origins, dirs], axis=-1)


def ray_depth(
    rays: np.ndarray, planes: tuple[tuple[tuple[float, float, float], float], ...]
) -> tuple[np.ndarray, np.ndarray]:
    """Nearest positive hit of origin + t * direction with planes n.x = c.

    The planes bound a convex room seen from inside.

    Returns the depth and the unit normal of the plane hit, facing the camera.
    """
    o, d = rays[..., :3], rays[..., 3:]
    ts = []
    for n, c in planes:
        n_arr = np.asarray(n)
        denom = d @ n_arr
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (c - o @ n_arr) / denom
        ts.append(np.where(denom > 0, t, np.inf))
    hit = np.argmin(ts, axis=0)
    depth = np.take_along_axis(np.stack(ts), hit[None], 0)[0]
    assert np.isfinite(depth).all() and (depth > 0).all()
    normals = -np.array([n for n, _ in planes], dtype=np.float64)[hit]
    return depth, normals / np.linalg.norm(normals, axis=-1, keepdims=True)


def room_scene(  # noqa: PLR0913
    *,
    h: int = 96,
    w: int = 128,
    noise: float = 0.02,
    outlier_frac: float = 0.01,
    normal_noise: float = 0.0,
    planes: tuple[tuple[tuple[float, float, float], float], ...] = ROOM_PLANES,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """Noisy inverse depth with low-confidence outliers on a planar scene, and normals.

    The normals are the planes' own, each tilted by isotropic Gaussian noise of
    normal_noise rad per axis.
    """
    assert noise >= 0 and 0 <= outlier_frac < 1 and normal_noise >= 0
    rng = np.random.default_rng(seed)
    rays = pinhole_rays(h, w, focal=0.8 * w, yaw=0.3, origin=(0.1, -0.1, 0.2))
    gt_depth, gt_normals = ray_depth(rays, planes)
    inv = (1 / gt_depth) * (1 + noise * rng.standard_normal((h, w)))
    outliers = rng.random((h, w)) < outlier_frac
    inv = np.where(outliers, inv * rng.uniform(0.5, 1.5, (h, w)), inv)
    conf = np.where(outliers, 0.1, 1.0)
    depth = 1 / inv
    assert np.isfinite(depth).all() and (depth > 0).all()
    normals = gt_normals + normal_noise * np.random.default_rng(
        seed + 1
    ).standard_normal((h, w, 3))
    normals /= np.linalg.norm(normals, axis=-1, keepdims=True)
    return {
        k: v.astype(np.float32)
        for k, v in {
            "depth": depth,
            "rays": rays,
            "conf": conf,
            "gt_depth": gt_depth,
            "normals": normals,
            "gt_normals": gt_normals,
        }.items()
    }


def soften(log_depth: np.ndarray, passes: int = 2) -> np.ndarray:
    """Blur log depth by [1, 2, 1] / 4 per axis (border replicated): steps ramp."""
    for _ in range(passes):
        padded = np.pad(log_depth, 1, mode="edge")
        log_depth = (
            0.25 * padded[1:-1, :-2]
            + 0.5 * padded[1:-1, 1:-1]
            + 0.25 * padded[1:-1, 2:]
        )
        padded = np.pad(log_depth, 1, mode="edge")
        log_depth = (
            0.25 * padded[:-2, 1:-1]
            + 0.5 * padded[1:-1, 1:-1]
            + 0.25 * padded[2:, 1:-1]
        )
    return log_depth


def layered_scene(
    gt_depth: np.ndarray, front: np.ndarray, noise: float, seed: int
) -> dict[str, np.ndarray]:
    """Fronto-parallel layers; prior: the softened, noisy GT; normals: no edge."""
    h, w = gt_depth.shape
    rays = pinhole_rays(h, w, focal=0.8 * w, yaw=0.0, origin=(0.0, 0.0, 0.0))
    depth = np.exp(
        soften(np.log(gt_depth))
        + noise * np.random.default_rng(seed).standard_normal((h, w))
    )
    normals = np.broadcast_to(np.array([0.0, 0.0, -1.0]), (h, w, 3))
    return {
        k: np.ascontiguousarray(x).astype(np.float32)
        for k, x in {
            "depth": depth,
            "rays": rays,
            "gt_depth": gt_depth,
            "normals": normals,
            "front": front,
        }.items()
    }


def occlusion_scene(
    *, h: int = 96, w: int = 128, noise: float = 0.003, seed: int = 0
) -> dict[str, np.ndarray]:
    """Two planes split by a slanted occluding edge, softened into a 2-4 px ramp.

    Feed-forward depth draws occlusion edges as such ramps; the normals (all facing the
    camera) carry no edge.
    """
    v, u = np.mgrid[0:h, 0:w].astype(np.float64)
    front = (u - w / 2) < 0.6 * (v - h / 2)
    return layered_scene(np.where(front, 2.0, 3.0), front, noise, seed)


def junction_scene(
    *, h: int = 96, w: int = 128, noise: float = 0.003, seed: int = 0
) -> dict[str, np.ndarray]:
    """Three layers meeting in a T (two monitors before a wall); prior softened.

    A near plane fills the bottom third, a middle plane the right half above it, a far
    wall the rest; `front` marks the near plane. Two-sided level sets cannot hold three
    layers: the middle one changes side at the T.
    """
    v, u = np.mgrid[0:h, 0:w]
    near, middle = v >= 2 * h // 3, u >= w // 2
    return layered_scene(
        np.where(near, 2.0, np.where(middle, 3.0, 4.0)), near, noise, seed
    )


def main() -> None:
    """Write the room: `uv run python -m src.loaders.synthetic --out <npz>`."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("data/synthetic_room.npz"))
    parser.add_argument(
        "--normal-noise",
        type=float,
        default=0.03,
        help="rad per axis; 0.03 is a median 2° tilt",
    )
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, **room_scene(normal_noise=args.normal_noise))
    print(f"wrote {args.out}")  # noqa: T201


if __name__ == "__main__":
    main()
