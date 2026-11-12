"""E45b: which camera do Tanks and Temples' images have? The data ships no intrinsics.

The laser scan carries colours. For a candidate camera the scan is projected into a
view with the dataset's pose (`to_scan_frame`), the nearest point per pixel gives a
grey image, and its gradient magnitude is correlated with the photo's over the pixels
the scan covers (both smoothed by SIGMA px, the cover eroded by ERODE px so its outline
does not count). A wrong camera shifts every outline, so the correlation peaks at the
right one.

Camera: principal point at the centre, focal length f (as a share of the image
width) and one radial coefficient k1 (OpenCV's: a point at normalised (x, y), r² = x²
+ y², is seen at (x, y) (1 + k1 r²)).

Per scene, VIEWS views uniform over the capture; a view counts in a region if the scan
covers MIN_PIXELS of it for every camera. Printed: with k1 = 0, the mean correlation
per f and the best f over all, the inner and the outer pixels (nearer or farther than
half the half-diagonal from the centre; distortion pulls the two apart); then the
mean correlation over the (k1, f) grid, its best cell, and the best f of the inner
and the outer pixels at that k1 (they should now agree).

uv run python experiments/scripts/e45b_focal_check.py > experiments/results/e45b_focal_check.txt
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

from src.loaders.kitti360.scene import read_ply
from src.loaders.tanksandtemples.scene import load_config, read_log, to_scan_frame

WIDTH, HEIGHT = 960, 540  # px the photos are compared at (half their size)
VIEWS = 8
STRIDE = 3  # every STRIDE-th scan point: still several per pixel
FOCALS = np.arange(0.59, 0.6151, 0.0025)
K1S = np.arange(0.0, -0.0601, -0.01)
SIGMA = 1.5
ERODE = 8
MIN_PIXELS = 5000  # covered pixels of a region a view needs to count there
INNER = 0.5  # of the half-diagonal: where the inner region ends
MIN_DEPTH = 0.05  # m, as the loader
MAX_RADIUS = 1.5  # normalised: beyond the image corner (0.96), before k1 folds back


def gradient(image: np.ndarray) -> np.ndarray:
    """Gradient magnitude (H, W) of the smoothed image."""
    smooth = ndimage.gaussian_filter(image, SIGMA)
    return np.hypot(ndimage.sobel(smooth, 0), ndimage.sobel(smooth, 1))


def scan_image(
    cam: np.ndarray, grey: np.ndarray, focal: float, k1: float
) -> np.ndarray:
    """Grey value (H, W) of the nearest scan point per pixel, NaN where none falls."""
    z = cam[:, 2]
    x, y = cam[:, 0] / z, cam[:, 1] / z
    r2 = x * x + y * y
    near = r2 < MAX_RADIUS**2
    x, y, z, grey, radial = x[near], y[near], z[near], grey[near], 1 + k1 * r2[near]
    cols = np.floor(focal * WIDTH * x * radial + WIDTH / 2).astype(np.int64)
    rows = np.floor(focal * WIDTH * y * radial + HEIGHT / 2).astype(np.int64)
    inside = (cols >= 0) & (cols < WIDTH) & (rows >= 0) & (rows < HEIGHT)
    pixel, z, grey = rows[inside] * WIDTH + cols[inside], z[inside], grey[inside]
    order = np.argsort(-z)  # far first, so the nearest point is written last
    out = np.full(HEIGHT * WIDTH, np.nan)
    out[pixel[order]] = grey[order]
    return out.reshape(HEIGHT, WIDTH)


def row(values: np.ndarray) -> str:
    """Correlations to two decimals."""
    return " ".join(f"{v:.2f}" for v in values)


def main() -> None:
    """Print the camera sweep of every scene of the config."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cfg", type=Path, default=Path("configs/tanksandtemples.yaml")
    )
    args = parser.parse_args()
    cfg = load_config(args.cfg)
    ys, xs = np.mgrid[0:HEIGHT, 0:WIDTH] + 0.5
    radius = np.hypot(xs - WIDTH / 2, ys - HEIGHT / 2) / np.hypot(WIDTH, HEIGHT) * 2
    regions = {"all": radius >= 0, "inner": radius < INNER, "outer": radius >= INNER}
    print(f"f / image width: {' '.join(f'{f:.4f}' for f in FOCALS)}")
    for scene in cfg.scenes:
        root = cfg.raw_root / scene
        name = scene.capitalize()
        poses = to_scan_frame(
            read_log((root / "poses" / f"{name}_COLMAP_SfM.log").read_text()),
            np.loadtxt(root / "alignment" / f"{name}_trans.txt"),
        )
        files = sorted((root / "images").glob("*.jpg"))
        vertices = read_ply(root / "gt" / f"{name}.ply")[::STRIDE]
        points = np.stack([vertices[a] for a in "xyz"], axis=1)
        grey = (
            0.299 * vertices["red"]
            + 0.587 * vertices["green"]
            + 0.114 * vertices["blue"]
        )
        picked = np.linspace(0, len(files) - 1, VIEWS).round().astype(int)
        shape = (VIEWS, len(K1S), len(FOCALS))
        score = {key: np.full(shape, np.nan) for key in regions}
        for v, i in enumerate(picked):
            photo = Image.open(files[i]).convert("L").resize((WIDTH, HEIGHT))
            target = gradient(np.asarray(photo, dtype=np.float64))
            cam = (points - poses[i, :3, 3]) @ poses[i, :3, :3]
            front = cam[:, 2] > MIN_DEPTH
            cam, shade = cam[front], grey[front]
            for a, b in np.ndindex(len(K1S), len(FOCALS)):
                image = scan_image(cam, shade, FOCALS[b], K1S[a])
                cover = np.isfinite(image)
                edges = gradient(np.where(cover, image, 0.0))
                cover = ndimage.binary_erosion(cover, iterations=ERODE)
                for key, region in regions.items():
                    use = cover & region
                    if use.sum() >= MIN_PIXELS:
                        score[key][v, a, b] = np.corrcoef(target[use], edges[use])[0, 1]
        mean = {}
        for key, table in score.items():
            whole = ~np.isnan(table).any(axis=(1, 2))  # views that count everywhere
            assert whole.any(), (scene, key)
            mean[key] = table[whole].mean(axis=0)
        print(f"\n{scene}: views {' '.join(files[i].stem for i in picked)}")
        print(f"  k1 = 0, mean correlation per f: {row(mean['all'][0])}")
        print(
            "  k1 = 0, best f: "
            + ", ".join(f"{key} {FOCALS[mean[key][0].argmax()]:.4f}" for key in regions)
        )
        print("  mean correlation, rows k1, columns f:")
        for k1, line in zip(K1S, mean["all"], strict=True):
            print(f"    {k1:+.2f}: {row(line)}")
        a, b = np.unravel_index(mean["all"].argmax(), mean["all"].shape)
        print(
            f"  best: f = {FOCALS[b]:.4f}, k1 = {K1S[a]:+.2f}"
            f" (correlation {mean['all'][a, b]:.3f}; k1 = 0: {mean['all'][0].max():.3f});"
            f" at that k1 the best f is inner {FOCALS[mean['inner'][a].argmax()]:.4f},"
            f" outer {FOCALS[mean['outer'][a].argmax()]:.4f}"
        )


if __name__ == "__main__":
    main()
