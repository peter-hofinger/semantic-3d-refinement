"""Pick a subset of frames and resize them for the backbone."""

import numpy as np
from PIL import Image


def select_frames(centers: np.ndarray, num: int, method: str = "uniform") -> np.ndarray:
    """Return sorted indices of `num` frames.

    `centers` are camera positions in capture order. "uniform" spaces frames
    evenly over the capture, "farthest" spreads camera positions in space.
    """
    count = len(centers)
    if num >= count:
        return np.arange(count)
    if method == "uniform":
        return np.unique(np.linspace(0, count - 1, num).round().astype(int))
    if method == "farthest":
        chosen = [0]
        dist = np.linalg.norm(centers - centers[0], axis=1)
        for _ in range(num - 1):
            nxt = int(dist.argmax())
            chosen.append(nxt)
            dist = np.minimum(dist, np.linalg.norm(centers - centers[nxt], axis=1))
        return np.sort(np.array(chosen))
    msg = f"unknown sampling method: {method}"
    raise ValueError(msg)


def target_size(width: int, height: int, long_side: int, patch: int) -> tuple[int, int]:
    """Scale so the long side is about `long_side`, rounded to multiples of `patch`."""
    scale = long_side / max(width, height)
    new_w = max(patch, round(width * scale / patch) * patch)
    new_h = max(patch, round(height * scale / patch) * patch)
    return new_w, new_h


def resize_view(
    image: np.ndarray, intrinsics: np.ndarray, size: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray]:
    """Resize an image to `size` (width, height) and scale its intrinsics to match."""
    height, width = image.shape[:2]
    new_w, new_h = size
    resized = Image.fromarray(image).resize((new_w, new_h), Image.Resampling.LANCZOS)
    scaled = intrinsics.astype(np.float64).copy()
    # COLMAP convention (pixel corners at integers), so a plain scale is exact
    scaled[0] *= new_w / width
    scaled[1] *= new_h / height
    return np.asarray(resized), scaled
