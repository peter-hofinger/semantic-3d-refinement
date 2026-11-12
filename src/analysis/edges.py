"""Depth edges as departures from the local plane, not as depth steps.

E18 and E35 call two neighbouring pixels a depth edge when their depths differ by
more than a share τ of the smaller one (the step rule). A plane seen at a grazing
angle exceeds that without any discontinuity: the road 1.55 m below a camera with
f = 368 px steps by more than 5% from row to row beyond 28 m, and from a pixel to one
3 rows away beyond 10 m (E45a, KITTI-360).

A pinhole camera sees a plane as inverse depth that is linear in the pixel
coordinates, d(p) = a . p + b, so a pixel predicts its neighbour by continuing its own
slope. With m(x, y) = |x - y| / min(x, y), which for inverse depths equals the step
rule's relative depth difference:

Edge. Neighbours p, q along an axis, p' before p and q' after q. They are an edge when
both continuations miss,
  m(d_q, d_p + (d_p - d_p')) > τ  and  m(d_p, d_q - (d_q' - d_q)) > τ,
a missing p' or q' counting as slope 0. A plane passes both. At a crease (a wall
standing on a floor) the floor's continuation meets the wall's foot, so it is no
edge. At a structure one pixel wide both miss, as its own "slope" is the jump.
With zero slopes this is the step rule.

Other surface. For a pixel p with slope g_p (per axis the minmod of its two one-sided
differences: the smaller one, 0 if their signs differ), a pixel q within a radius
lies on another surface when the step rule separates them and p's plane misses q:
  m(d_q, d_p) > τ  and  m(d_q, d_p + g_p . (q - p)) > τ.
"""

from __future__ import annotations

import numpy as np

DEPTH_JUMP = 0.05  # τ: relative miss that makes another surface (E18)


def inverse_depth(depth: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """1 / depth (H, W) where valid, NaN elsewhere."""
    assert depth.shape == valid.shape, (depth.shape, valid.shape)
    assert (depth[valid] > 0).all(), "valid pixels need positive depth"
    out = np.full(depth.shape, np.nan)
    out[valid] = 1.0 / depth[valid]
    return out


def one_sided(inverse: np.ndarray, axis: int) -> tuple[np.ndarray, np.ndarray]:
    """Differences (H, W) to the pixel before and to the pixel after; NaN if missing."""
    pad = [(0, 0), (0, 0)]
    pad[axis] = (1, 1)
    step = np.pad(np.diff(inverse, axis=axis), pad, constant_values=np.nan)
    size = inverse.shape[axis]
    before = np.take(step, np.arange(size), axis=axis)
    after = np.take(step, np.arange(1, size + 1), axis=axis)
    return before, after


def slopes(inverse: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Minmod slope (H, W) of the inverse depth along x and along y, per px."""
    out = []
    for axis in (1, 0):
        before, after = one_sided(inverse, axis)
        smaller = np.where(np.abs(before) <= np.abs(after), before, after)
        both = np.where(before * after > 0, smaller, 0.0)
        single = np.where(np.isnan(before), after, before)
        slope = np.where(np.isnan(before) | np.isnan(after), single, both)
        out.append(np.nan_to_num(slope, nan=0.0))
    return out[0], out[1]


def miss(actual: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    """m(x, y) = |x - y| / min(x, y); NaN where either is, inf for a prediction <= 0."""
    with np.errstate(invalid="ignore", divide="ignore"):
        low = np.minimum(actual, predicted)
        return np.where(low > 0, np.abs(actual - predicted) / low, np.inf + 0 * low)


def depth_edges(
    depth: np.ndarray, valid: np.ndarray, jump: float = DEPTH_JUMP
) -> np.ndarray:
    """Mark valid pixels (H, W) with a valid 4-neighbour off both continuations."""
    inverse = inverse_depth(depth, valid)
    edge = np.zeros(depth.shape, dtype=bool)
    for axis in (1, 0):
        before, after = one_sided(inverse, axis)
        first = [slice(None)] * 2
        second = [slice(None)] * 2
        first[axis], second[axis] = slice(None, -1), slice(1, None)
        a, b = inverse[tuple(first)], inverse[tuple(second)]
        into_a = np.nan_to_num(before[tuple(first)], nan=0.0)
        out_of_b = np.nan_to_num(after[tuple(second)], nan=0.0)
        with np.errstate(invalid="ignore"):
            pair = (miss(b, a + into_a) > jump) & (miss(a, b - out_of_b) > jump)
        edge[tuple(first)] |= pair
        edge[tuple(second)] |= pair
    assert not edge[~valid].any()
    return edge


def across_edge(
    depth: np.ndarray, valid: np.ndarray, radius: int, jump: float = DEPTH_JUMP
) -> np.ndarray:
    """Depth (H, W) of the nearest pixel within `radius` px on another surface.

    NaN where the pixel is invalid or no valid pixel in reach is on another surface.
    """
    inverse = inverse_depth(depth, valid)
    gx, gy = slopes(inverse)
    h, w = depth.shape
    out = np.full(depth.shape, np.nan)
    best = np.full(depth.shape, np.inf)
    padded = np.pad(inverse, radius, constant_values=np.nan)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            dist = np.hypot(dy, dx)
            if dist == 0 or dist > radius:
                continue
            other = padded[radius + dy : radius + dy + h, radius + dx : radius + dx + w]
            with np.errstate(invalid="ignore"):
                apart = miss(other, inverse) > jump
                off = miss(other, inverse + gx * dx + gy * dy) > jump
            take = apart & off & (dist < best)
            out[take], best[take] = 1.0 / other[take], dist
    assert np.isnan(out[~valid]).all()
    return out
