"""A ScanNet++ scene's laser-scan surface as the point cloud `multiview.load_truth` takes (`scan.npy`).

ETH3D ships its scan as points; ScanNet++ ships a mesh. Its ground-truth depth, rendered into the prepared views from
the laser-scan mesh (z-depth in meters, 0 where the ray misses; Mircea's `scannetpp-render-gt`, cameras and pixel
centers as in the views file), is lifted through those cameras. The cloud covers what the views see, which is all
the evaluation scores.
`uv run python -m src.loaders.scannetpp.scan --views <views npz> --gt <gt views npz> --out <scan.npy>`
"""

import argparse
from pathlib import Path

import numpy as np

from src.evaluation.depth import lift


def scan_from_depth(depth: np.ndarray, intrinsics: np.ndarray, cam_to_world: np.ndarray) -> np.ndarray:
    """World points of every pixel with ground truth over all views, (N, 3) float32."""
    assert depth.ndim == 3 and len(depth) == len(intrinsics) == len(cam_to_world), depth.shape  # noqa: PLR2004
    points = [lift(d, k, c)[d > 0] for d, k, c in zip(depth.astype(np.float64), intrinsics, cam_to_world, strict=True)]
    return np.concatenate(points).astype(np.float32)


def main() -> None:
    """Write the scan cloud of one scene."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--views", type=Path, required=True)
    parser.add_argument("--gt", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    views = np.load(args.views)
    depth = np.load(args.gt)["depth"]
    scan = scan_from_depth(depth, views["intrinsics"].astype(np.float64), views["cam_to_world"].astype(np.float64))
    np.save(args.out, scan)
    print(f"wrote {args.out}: {len(scan)} points")  # noqa: T201


if __name__ == "__main__":
    main()
