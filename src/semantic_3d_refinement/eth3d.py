"""ETH3D high-res multi-view scenes (Schöps et al., CVPR 2017) as backbone input and a ground-truth point cloud.

Reads the undistorted DSLR images with their COLMAP calibration (PINHOLE, pixel centers at +0.5) and the evaluation
laser scans (`dslr_scan_eval`), which the MeshLab project aligns to the calibration's frame, in meters. Views are
picked and resized as for ScanNet++ (`loaders/frames.py` on the `pouya` branch): evenly over the capture order,
Lanczos, intrinsics scaled per axis. ETH3D is CC BY-NC-SA 4.0: everything stays under `data/`, which git ignores.
"""

import argparse
import re
from pathlib import Path

import numpy as np
from PIL import Image


def quat_to_rotation(q: np.ndarray) -> np.ndarray:
    """Rotation matrix of a unit quaternion (w, x, y, z)."""
    w, x, y, z = q / np.linalg.norm(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def read_calibration(calib: Path) -> list[tuple[str, np.ndarray, np.ndarray, tuple[int, int]]]:
    """(image name, K, cam_to_world, (width, height)) per image of a COLMAP text model, sorted by name."""
    cameras = {}
    for line in (calib / "cameras.txt").read_text().splitlines():
        if line and not line.startswith("#"):
            cam_id, model, w, h, fx, fy, cx, cy = line.split()
            assert model == "PINHOLE", model
            k = np.array([[float(fx), 0, float(cx)], [0, float(fy), float(cy)], [0, 0, 1]])
            cameras[cam_id] = (k, (int(w), int(h)))
    lines = [ln for ln in (calib / "images.txt").read_text().splitlines() if not ln.startswith("#")]
    out = []
    for line in lines[::2]:
        _, *qt, cam_id, name = line.split()
        world_to_cam = np.eye(4)
        world_to_cam[:3, :3] = quat_to_rotation(np.array(qt[:4], dtype=float))
        world_to_cam[:3, 3] = np.array(qt[4:], dtype=float)
        k, size = cameras[cam_id]
        out.append((Path(name).name, k, np.linalg.inv(world_to_cam), size))
    return sorted(out, key=lambda r: r[0])


def read_ply_xyz(path: Path) -> np.ndarray:
    """Vertex positions of a binary little-endian PLY whose vertices hold only float x, y, z."""
    with path.open("rb") as f:
        header = b""
        while not header.endswith(b"end_header\n"):
            header += f.readline()
        text = header.decode()
        assert "binary_little_endian" in text, text
        vertex = text.split("element vertex ")[1].split("element")[0]
        assert vertex.split("\n")[1:4] == ["property float x", "property float y", "property float z"], vertex
        assert "property" not in "\n".join(vertex.split("\n")[4:]), vertex
        n = int(vertex.split("\n")[0])
        xyz = np.fromfile(f, dtype="<f4", count=3 * n).reshape(n, 3)
    assert xyz.shape == (n, 3) and np.isfinite(xyz).all()
    return xyz


def load_scan(scan_dir: Path) -> np.ndarray:
    """All evaluation scans of a scene in the calibration's frame, (N, 3) float32 meters."""
    mlp = (scan_dir / "scan_alignment.mlp").read_text()
    parts = []
    for name, matrix in re.findall(r'filename="([^"]+)".*?<MLMatrix44>(.*?)</MLMatrix44>', mlp, flags=re.DOTALL):
        transform = np.array(matrix.split(), dtype=float).reshape(4, 4)
        xyz = read_ply_xyz(scan_dir / name).astype(np.float64)
        parts.append((xyz @ transform[:3, :3].T + transform[:3, 3]).astype(np.float32))
    assert parts, scan_dir
    return np.concatenate(parts)


def target_size(width: int, height: int, long_side: int, patch: int) -> tuple[int, int]:
    """Scale so the long side is about `long_side`, rounded to multiples of `patch`."""
    scale = long_side / max(width, height)
    return max(patch, round(width * scale / patch) * patch), max(patch, round(height * scale / patch) * patch)


def prepare(scene: Path, out: Path, num: int, long_side: int, patch: int) -> None:
    """Write `num` resized views as the backbone's input and the scene's scan as `scan.npy`."""
    calib = read_calibration(scene / "dslr_calibration_undistorted")
    picked = [calib[i] for i in np.unique(np.linspace(0, len(calib) - 1, num).round().astype(int))]
    images, intrinsics = [], []
    for name, k, _, (w, h) in picked:
        size = target_size(w, h, long_side, patch)
        image = Image.open(scene / "images" / "dslr_images_undistorted" / name).convert("RGB")
        assert image.size == (w, h), (image.size, w, h)
        images.append(np.asarray(image.resize(size, Image.Resampling.LANCZOS)))
        intrinsics.append(np.diag([size[0] / w, size[1] / h, 1.0]) @ k)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out / f"views_n{num}_uniform_{long_side}.npz",
        scene_id=scene.name,
        names=np.array([p[0] for p in picked]),
        images=np.stack(images),
        intrinsics=np.stack(intrinsics).astype(np.float32),
        cam_to_world=np.stack([p[2] for p in picked]).astype(np.float32),
    )
    np.save(out / "scan.npy", load_scan(scene / "dslr_scan_eval"))


def main() -> None:
    """Prepare ETH3D scenes: `uv run python -m semantic_3d_refinement.eth3d --scenes office kicker`."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data/eth3d"))
    parser.add_argument("--out", type=Path, default=Path("data/processed/eth3d"))
    parser.add_argument("--scenes", nargs="+", required=True)
    parser.add_argument("--num", type=int, default=20)
    parser.add_argument("--long-side", type=int, default=504)
    parser.add_argument("--patch", type=int, default=14)
    args = parser.parse_args()
    for name in args.scenes:
        prepare(args.root / name, args.out / name, args.num, args.long_side, args.patch)
        print(f"wrote {args.out / name}")  # noqa: T201


if __name__ == "__main__":
    main()
