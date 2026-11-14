"""Point filters for a per-pixel cloud: one score per pixel, cut at the scene's quantiles.

E45 (issue #27) asks whether the joint refiner beats dropping Déjà View's worst points
until its cloud is as complete as the refiner's. Each filter is a score per pixel of all
views, higher meaning keep. `ranks` orders a scene's pixels by it and `keep_top` keeps the
top fraction q, so no filter has a threshold of its own: the cut is a quantile of the
scene's own scores, as VGGT's demo cuts its confidence at a percentile over all frames.
Ties are broken by one fixed random permutation; within a tie a filter has no preference.

- depth gradient: |∇ log D| by central differences per view (`log_gradient`); flying
  pixels sit on steep depth, and the log makes the score relative.
- depth edge: log(max D / min D) over the 3 x 3 window (`depth_edge`), MoGe's `depth_edge`
  (Wang et al. 2025), which drops the pixels where it exceeds log(1 + rtol).
- statistical outlier: the mean distance to the k nearest points of the scene's cloud
  (`neighbor_distance`), PCL's StatisticalOutlierRemoval (Rusu et al. 2008), which drops
  points beyond mean + a std of it; k = 20, Open3D's documented default.
- multi-view consistency: the number of other views whose depth agrees with the pixel's
  point (`consistent_views`), the geometric check of MVS fusion (COLMAP, Schönberger et
  al. 2016; MVSNet, Yao et al. 2018), which keeps pixels consistent in at least n views.
  Agree is E_mv's own test (`multiview.match_candidates`).
- voxel density: the number of the scene's points in the pixel's voxel (`voxel_counts`),
  the voxel side the scene's median pixel footprint D / f (`pixel_footprint`): the
  cloud's own units, no GT scale. Its variants sum a value per voxel instead
  (`voxel_sums`: confidence, or the points of one instance, `instance_voxel_density`),
  and voxel fusion replaces the kept points by their voxel's mean (`voxel_means`), as
  voxel-grid downsampling does (Open3D's `voxel_down_sample`), but one point per pixel,
  so the protocol scores it.
- grazing angle: |cos| between the pixel's ray and its normal (`facing`), the weight
  Curless and Levoy (1996) give a range sample.

Fill (E58a, issue #58): instead of dropping the flagged pixels, each gets a new depth
from its view's kept pixels (`fill_harmonic`), so every pixel keeps a point and the
completeness stays comparable. The fill is the membrane: inverse depth f minimizes
sum_ij w_ij (f_i - f_j)^2 over 4-neighbours with the kept pixels fixed, so it solves
the weighted Laplace equation on the holes. A plane's inverse depth is affine in the
pixel, hence harmonic, so with equal weights a planar hole is filled exactly. With an
image the weights are exp(-|I_i - I_j|^2 / (2 s^2)), s^2 the view's mean squared colour
step, the affinity of Levin et al.'s colorization (2004) that NYU Depth v2 fills its
holes with (Silberman et al. 2012). With instance ids an edge between two instances
keeps only `CUT_WEIGHT`, so a hole fills from its own object. `FILL_ANCHOR` ties every
filled pixel weakly to its own depth, which only matters where no kept pixel reaches.
"""

import numpy as np
import torch
from scipy import sparse
from scipy.ndimage import maximum_filter, minimum_filter
from scipy.sparse.linalg import spsolve
from scipy.spatial import cKDTree
from torch import Tensor

from src.refinement.multiview import Scene, match_candidates, occlusion_threshold

SOR_NEIGHBORS = 20  # Open3D's documented default for remove_statistical_outlier
TIE_SEED = 0  # the one permutation that breaks ties
CUT_WEIGHT = 1e-3  # an edge between two instances, against 1 within one
FILL_ANCHOR = 1e-6  # pull of a filled pixel to its own inverse depth


def ranks(score: np.ndarray) -> np.ndarray:
    """Rank of each pixel's score in [0, N), ascending, ties in a fixed random order."""
    assert np.isfinite(score).all(), "a filter scores every pixel"
    flat = score.ravel()
    tie = np.random.default_rng(TIE_SEED).permutation(flat.size)
    rank = np.empty(flat.size, dtype=np.int64)
    rank[np.lexsort((tie, flat))] = np.arange(flat.size)
    return rank.reshape(score.shape)


def keep_top(rank: np.ndarray, q: float) -> np.ndarray:
    """Return the pixels of the top fraction q of the ranks, round(q N) of them."""
    assert 0 <= q <= 1, q
    return rank >= rank.size - round(q * rank.size)


def keep_top_within(score: np.ndarray, groups: np.ndarray, q: float) -> np.ndarray:
    """Return the top fraction q of each group's pixels by score, round(q n) of n."""
    assert score.shape == groups.shape and 0 <= q <= 1, (score.shape, groups.shape, q)
    flat, g = score.ravel(), groups.ravel()
    tie = np.random.default_rng(TIE_SEED).permutation(flat.size)
    order = np.lexsort((tie, flat, g))
    _, start, count = np.unique(g[order], return_index=True, return_counts=True)
    pos = np.arange(flat.size) - np.repeat(start, count)
    cut = np.repeat(count - np.round(q * count).astype(np.int64), count)
    out = np.empty(flat.size, dtype=bool)
    out[order] = pos >= cut
    return out.reshape(score.shape)


def both(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Score of two filters at once, the lower of a pixel's two ranks.

    Every cut of it keeps the pixels that both scores rank above one shared level.
    """
    return np.minimum(ranks(a), ranks(b)).astype(np.float64)


def log_gradient(depth: np.ndarray) -> np.ndarray:
    """|∇ log D| (V, H, W) per view by central differences, one-sided at the border."""
    assert depth.ndim == 3 and (depth > 0).all(), depth.shape  # noqa: PLR2004
    dy, dx = np.gradient(np.log(depth.astype(np.float64)), axis=(1, 2))
    return np.hypot(dx, dy)


def depth_edge(depth: np.ndarray) -> np.ndarray:
    """log(max / min) of the depth (V, H, W) in each pixel's 3 x 3 window, per view."""
    assert depth.ndim == 3 and (depth > 0).all(), depth.shape  # noqa: PLR2004
    d = depth.astype(np.float64)
    window = (1, 3, 3)
    high = maximum_filter(d, window, mode="nearest")
    return np.log(high / minimum_filter(d, window, mode="nearest"))


def neighbor_distance(points: np.ndarray, k: int = SOR_NEIGHBORS) -> np.ndarray:
    """Mean distance (N,) of each point to its k nearest other points."""
    assert points.ndim == 2 and points.shape[1] == 3 and len(points) > k, points.shape  # noqa: PLR2004
    tree = cKDTree(points, balanced_tree=False, compact_nodes=False)
    d = tree.query(points, k + 1, workers=-1)[0]
    return d[:, 1:].mean(1)


def pixel_footprint(depth: np.ndarray, k: np.ndarray) -> float:
    """Median side D / f of a pixel's patch of surface over the scene's pixels."""
    assert depth.ndim == 3 and k.shape == (len(depth), 3, 3), (depth.shape, k.shape)  # noqa: PLR2004
    return float(np.median(depth / k[:, :1, :1]))


def voxel_index(
    points: np.ndarray, side: float, groups: np.ndarray | None = None
) -> np.ndarray:
    """Index (N,) of each point's cell: its cube of the grid of the given side, and group."""
    assert points.ndim == 2 and points.shape[1] == 3 and side > 0, points.shape  # noqa: PLR2004
    assert groups is None or groups.shape == (len(points),), groups
    cells = np.floor(points / side).astype(np.int64)
    if groups is not None:
        cells = np.column_stack([cells, groups.astype(np.int64)])
    return np.unique(cells, axis=0, return_inverse=True)[1].ravel()


def voxel_counts(points: np.ndarray, side: float) -> np.ndarray:
    """Count the points (N,) in each point's cube of the grid of the given side."""
    index = voxel_index(points, side)
    return np.bincount(index)[index]


def voxel_sums(
    points: np.ndarray,
    side: float,
    values: np.ndarray,
    groups: np.ndarray | None = None,
) -> np.ndarray:
    """Sum the values (N,) of the points in each point's cell (`voxel_index`)."""
    assert values.shape == (len(points),), values.shape
    index = voxel_index(points, side, groups)
    return np.bincount(index, weights=values.astype(np.float64))[index]


def instance_voxel_density(
    points: np.ndarray, depth: np.ndarray, k: np.ndarray, ids: np.ndarray
) -> np.ndarray:
    """Each pixel's (V, H, W) count of its own instance's points in its voxel, >= 1.

    E45's best filter: `voxel_sums` of ones grouped by instance id, the voxel side the
    scene's `pixel_footprint`. Pixels without an instance (-1) count as one instance.
    """
    assert points.shape == (*depth.shape, 3) and ids.shape == depth.shape, ids.shape
    flat = points.reshape(-1, 3)
    side = pixel_footprint(depth, k)
    count = voxel_sums(flat, side, np.ones(len(flat)), ids.ravel()).reshape(depth.shape)
    assert (count >= 1).all(), "a pixel counts itself"
    return count


def voxel_means(
    points: np.ndarray, side: float, groups: np.ndarray | None = None
) -> np.ndarray:
    """Replace each point (N, 3) by the mean of the points in its cell (`voxel_index`)."""
    index = voxel_index(points, side, groups)
    count = np.bincount(index)
    mean = np.stack(
        [np.bincount(index, weights=points[:, a]) / count for a in range(3)], axis=1
    )
    out = mean[index]
    assert np.isfinite(out).all()
    return out


def facing(normals: np.ndarray, directions: np.ndarray) -> np.ndarray:
    """|cos| of the angle between each pixel's normal and its ray (..., 3)."""
    assert normals.shape == directions.shape and normals.shape[-1] == 3, normals.shape  # noqa: PLR2004
    n = normals / np.linalg.norm(normals, axis=-1, keepdims=True)
    r = directions / np.linalg.norm(directions, axis=-1, keepdims=True)
    return np.abs((n * r).sum(-1))


@torch.no_grad()
def consistent_views(
    scene: Scene, d0: Tensor, scale: Tensor, tol: float | None = None
) -> Tensor:
    """Per pixel (V, H, W), how many other views' depth agrees with its point.

    Agree is E_mv's test (`multiview.find_matches`): the point projects into j's image,
    and both its log gap to j's depth and the spread of j's four interpolated pixels
    stay below tol, by default Otsu's threshold on the scene's gaps.
    """
    n, h, w = d0.shape
    candidates = match_candidates(scene, d0, scale)
    if tol is None:
        tol = occlusion_threshold(torch.cat([g for _, g, _ in candidates.values()]))
    count = torch.zeros(n, h * w, device=d0.device)
    for (i, _), (pixels, gap, spread) in candidates.items():
        count[i, pixels[(gap.abs() < tol) & (spread < tol)]] += 1
    assert (count <= n - 1).all()
    return count.view(n, h, w)


def grid_edges(h: int, w: int) -> tuple[np.ndarray, np.ndarray]:
    """Flat pixel indices (E,) of both ends of every 4-neighbour edge of an h x w grid."""
    index = np.arange(h * w).reshape(h, w)
    a = np.concatenate([index[:, :-1].ravel(), index[:-1].ravel()])
    b = np.concatenate([index[:, 1:].ravel(), index[1:].ravel()])
    return a, b


def edge_weights(
    a: np.ndarray, b: np.ndarray, image: np.ndarray | None, ids: np.ndarray | None
) -> np.ndarray:
    """Weight (E,) of each edge: 1, colour affinity with an image, cut across ids."""
    weight = np.ones(len(a))
    if image is not None:
        rgb = image.reshape(-1, 3).astype(np.float64) / 255
        step = ((rgb[a] - rgb[b]) ** 2).sum(1)
        weight *= np.exp(-step / (2 * max(step.mean(), 1e-12)))
    if ids is not None:
        flat = ids.ravel()
        weight *= np.where(flat[a] == flat[b], 1.0, CUT_WEIGHT)
    return weight


def fill_view(
    depth: np.ndarray,
    known: np.ndarray,
    image: np.ndarray | None = None,
    ids: np.ndarray | None = None,
) -> np.ndarray:
    """One view's depth (H, W) with the unknown pixels filled by the weighted membrane."""
    assert depth.ndim == 2 and known.shape == depth.shape and (depth > 0).all()  # noqa: PLR2004
    if known.all():
        return depth
    h, w = depth.shape
    f = 1 / depth.astype(np.float64).ravel()
    a, b = grid_edges(h, w)
    weight = edge_weights(a, b, image, ids)
    n = h * w
    adjacency = sparse.coo_matrix(
        (
            np.concatenate([weight, weight]),
            (np.concatenate([a, b]), np.concatenate([b, a])),
        ),
        shape=(n, n),
    ).tocsr()
    laplacian = sparse.diags(np.asarray(adjacency.sum(1)).ravel()) - adjacency
    unknown = np.flatnonzero(~known.ravel())
    fixed = np.flatnonzero(known.ravel())
    system = laplacian[unknown][:, unknown] + FILL_ANCHOR * sparse.eye(len(unknown))
    rhs = -laplacian[unknown][:, fixed] @ f[fixed] + FILL_ANCHOR * f[unknown]
    out = f.copy()
    out[unknown] = spsolve(system.tocsc(), rhs)
    assert np.isfinite(out).all() and (out > 0).all()
    return (1 / out).reshape(h, w).astype(depth.dtype)


def fill_harmonic(
    depth: np.ndarray,
    known: np.ndarray,
    images: np.ndarray | None = None,
    ids: np.ndarray | None = None,
) -> np.ndarray:
    """Depth (V, H, W) with the pixels not `known` filled per view (`fill_view`)."""
    assert depth.ndim == 3 and known.shape == depth.shape, (depth.shape, known.shape)  # noqa: PLR2004
    return np.stack(
        [
            fill_view(
                depth[v],
                known[v],
                None if images is None else images[v],
                None if ids is None else ids[v],
            )
            for v in range(len(depth))
        ]
    )
