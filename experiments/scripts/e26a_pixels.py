"""Map a competitor's per-pixel output back onto our views' pixels (E26a, issue #26).

A competitor sees each view centre-cropped and resized to its own resolution, a map
x' = A x of pixel coordinates (A from its own loader, e.g. MUSt3R's
`get_resize_function`). Our pixel p takes the output at the model pixel its centre
lands in, nearest, so no point is interpolated across a depth edge; pixels the crop
cut off get NaN (no point, which counts against completeness only). numpy only: the
competitors' environments import it next to their runners.
"""

import numpy as np


def back_to_views(out: np.ndarray, to_model: np.ndarray, h: int, w: int) -> np.ndarray:
    """Return out (V, h', w', C) sampled at our (V, h, w) pixels, NaN where cropped off.

    to_model (3, 3) maps our pixel coordinates (corners at 0) to the model's.
    """
    assert out.ndim == 4, out.shape
    assert to_model.shape == (3, 3), to_model.shape
    _, mh, mw, _ = out.shape
    v, u = np.mgrid[0:h, 0:w] + 0.5
    x = to_model @ np.stack([u.ravel(), v.ravel(), np.ones(u.size)])
    col, row = np.floor(x[0] / x[2]).astype(int), np.floor(x[1] / x[2]).astype(int)
    inside = (col >= 0) & (col < mw) & (row >= 0) & (row < mh)
    result = np.full((len(out), h * w, out.shape[-1]), np.nan, np.float32)
    result[:, inside] = out[:, row[inside], col[inside]]
    return result.reshape(len(out), h, w, -1)
