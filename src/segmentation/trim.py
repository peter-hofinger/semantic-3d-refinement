"""Trim a mask at the backbone's depth jumps: drop what is seen through or past it.

A text-prompted mask of a thin object with holes (a bed frame's bars, a chair's back)
also covers the surface seen through the holes; that is exactly where feed-forward
depth bleeds (`experiments/LOG.md`, E31-E32). The mask is split into 4-connected
regions whose links never cross a depth jump, a relative change

    |D(p) - D(q)| / min(D(p), D(q)) > JUMP

between 4-neighbours p, q (the depth-edge definition of E18). Every link from region
r to another region of the same mask crosses a jump and has one far side. A region is
background seen through a hole or past an outline if it lies on the far side of more
than FAR of its links to the rest of the mask; such regions are removed. Since each
cut link has exactly one far side, the far fractions average 1/2 over all links, so
not every region can exceed FAR = 1/2: a trimmed mask is never empty. A mask without
inner jumps is returned unchanged. Both thresholds were fixed before any run (E33
plan), not tuned. The rule only looks at depth order, so it also removes a far region
that is the larger part of the mask (a wall mask that took a hanging lamp in front).
"""

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

JUMP = 0.05  # relative depth change between 4-neighbours that cuts a mask
FAR = 0.5  # a region on the far side of more than this share of its cuts is removed


def relative_jumps(depth: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Relative depth change to the right (H, W-1) and downward (H-1, W) neighbour."""
    assert depth.ndim == 2, depth.shape  # noqa: PLR2004
    assert np.isfinite(depth).all(), "depth must be finite"
    assert (depth > 0).all(), "depth must be positive"
    left, top = depth[:, :-1], depth[:-1]
    right = np.abs(depth[:, 1:] - left) / np.minimum(depth[:, 1:], left)
    down = np.abs(depth[1:] - top) / np.minimum(depth[1:], top)
    return right, down


def trim_mask(mask: np.ndarray, depth: np.ndarray) -> np.ndarray:
    """Remove the regions of `mask` that lie behind the rest of it (H, W) bool."""
    assert mask.shape == depth.shape, (mask.shape, depth.shape)
    assert mask.dtype == bool, mask.dtype
    if not mask.any():
        return mask.copy()
    rows, cols = np.flatnonzero(mask.any(1)), np.flatnonzero(mask.any(0))
    box = np.s_[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]
    m, d = mask[box], depth[box]
    h, w = m.shape
    index = np.arange(h * w).reshape(h, w)
    right, down = relative_jumps(d)
    pairs = [
        (index[:, :-1], index[:, 1:], m[:, :-1] & m[:, 1:], right > JUMP),
        (index[:-1], index[1:], m[:-1] & m[1:], down > JUMP),
    ]
    a = np.concatenate([p[ok & ~cut] for p, _, ok, cut in pairs])
    b = np.concatenate([q[ok & ~cut] for _, q, ok, cut in pairs])
    graph = coo_matrix((np.ones(len(a), np.int8), (a, b)), shape=(h * w, h * w))
    count, region = connected_components(graph, directed=False)
    a = np.concatenate([p[ok & cut] for p, _, ok, cut in pairs])
    b = np.concatenate([q[ok & cut] for _, q, ok, cut in pairs])
    ra, rb = region[a], region[b]
    apart = ra != rb
    a, b, ra, rb = a[apart], b[apart], ra[apart], rb[apart]
    flat = d.ravel()
    far = np.where(flat[a] > flat[b], ra, rb)
    border = np.bincount(np.concatenate([ra, rb]), minlength=count)
    behind = np.bincount(far, minlength=count) > FAR * border
    out = mask.copy()
    out[box] = m & ~behind[region].reshape(h, w)
    assert out.any(), "a trimmed mask keeps at least one region"
    return out
