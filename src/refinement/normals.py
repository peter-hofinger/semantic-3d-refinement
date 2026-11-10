"""E_n, the normal term: BiNI's residuals (Cao et al. 2022) on DSINE's normals.

Own matrix-free implementation, generalized from a pinhole K to the ray map: for
neighbors p, q = p + 1 along x or y, x_q lies on the plane of (x_p, n_p) iff, in log
depth z, dz = log|n_p.d_p| - log|n_p.d_q|. Each edge carries two residuals, p's forward
and q's backward one, a (dz - log|n.d_p| + log|n.d_q|), weighted w+ and w-. Unlike BiNI,
which re-weights the sides during its solve, the weights are fixed on the input: 0 on
the input's occlusion contours (`cut_weights`), 1/2 elsewhere, so E_n stays one smooth
sum of squares for Gauss-Newton. The residuals are log-depth steps times the focal
length (`neighbor_terms`), which is why λ_n is 1e-4 (`RefineConfig`).

The normals come from DSINE (Bae and Davison 2024), run once per view on the image and
the backbone's intrinsics and cached next to the views (`dsine-predict`). DSINE is
non-commercial; its code and weights stay outside git, under `data/third_party/`.
"""

import argparse
import heapq
import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from src.geometry import Geometry
from src.loaders.views import Views

DSINE_REPO = Path("data/third_party/dsine_hub")  # hugoycj/DSINE-hub @ a6b7d25
# px, window of the local mid-depth and budget for closing a contour: spans Déjà View's
# 2-4 px occlusion ramps on both sides (fc6c3c4); 3-11 px score the same on dev, and no
# closing loses 0.05 cm of accuracy (E40)
CUT_WINDOW = 7


def normals_from_depth(depth: Tensor, rays: Tensor) -> Tensor:
    """Return unit normals of the lifted points by central differences, toward camera.

    The border replicates its neighbors.
    """
    pts = rays[..., :3] + depth[..., None] * rays[..., 3:]
    n = torch.linalg.cross(
        pts[1:-1, 2:] - pts[1:-1, :-2], pts[2:, 1:-1] - pts[:-2, 1:-1]
    )
    n = torch.nn.functional.pad(
        n.permute(2, 0, 1)[None], (1, 1, 1, 1), mode="replicate"
    )[0].permute(1, 2, 0)
    n = n / n.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return torch.where(((n * rays[..., 3:]).sum(-1) > 0)[..., None], -n, n)


def one_sided(step: Tensor, axis: int) -> tuple[Tensor, Tensor]:
    """Forward differences along an axis (-3: y, -2: x) as each pixel's two tangents.

    Returns (backward, forward), NaN where the neighbor lies outside the image.
    """
    pad = [0, 0, 0, 0] if axis == -2 else [0, 0, 0, 0, 0, 0]  # noqa: PLR2004
    pad_after, pad_before = pad.copy(), pad.copy()
    k = 2 * (-1 - axis)  # pad's pairs run from the last dimension
    pad_after[k + 1], pad_before[k] = 1, 1
    nan = float("nan")
    return (
        torch.nn.functional.pad(step, pad_before, value=nan),
        torch.nn.functional.pad(step, pad_after, value=nan),
    )


def closer_neighbor_normals(points: Tensor) -> Tensor:
    """Return unit normals (..., H, W, 3) of camera points, closer neighbor per axis.

    Peter's proposal (E42): along x the tangent goes to the left or the right neighbor,
    whichever point is closer in 3D, along y to the upper or the lower one; the normal
    is their cross product, facing the camera at the origin (n . x < 0). At an occlusion
    edge the closer neighbor lies on the pixel's own surface, so the normal does not see
    the jump. A neighbor outside the image never wins.
    """
    assert points.shape[-1] == 3 and min(points.shape[-3:-1]) >= 2  # noqa: PLR2004
    tangents = []
    for axis, step in (
        (-2, points[..., 1:, :] - points[..., :-1, :]),
        (-3, points[..., 1:, :, :] - points[..., :-1, :, :]),
    ):
        back, fwd = one_sided(step, axis)
        lb = back.square().sum(-1, keepdim=True).nan_to_num(torch.inf)
        lf = fwd.square().sum(-1, keepdim=True).nan_to_num(torch.inf)
        tangents.append(torch.where(lb < lf, back, fwd))
    n = torch.linalg.cross(*tangents, dim=-1)
    n = n / n.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    out = torch.where((n * points).sum(-1, keepdim=True) > 0, -n, n)
    assert out.isfinite().all()
    return out


def neighbor_terms(
    normals: Tensor, rays: Tensor
) -> tuple[Tensor, list[tuple[Tensor, Tensor, Tensor, Tensor]]]:
    """Per pixel a = s |n.d|; per axis the a, g of both residuals on each edge (p, q).

    p's forward residual and q's backward one.

    A plane through x_p with normal n holds x_q iff z_q - z_p = log|n.d_p| - log|n.d_q|,
    exactly; the residual is a * (z_q - z_p + log|n.d_q| - log|n.d_p|). BiNI uses its
    first-order form n.(d_q - d_p) / n.d_p instead. s = 1 / median |d_q - d_p| (the
    focal length for a pinhole) puts the residuals on BiNI's scale.
    """
    d = rays[..., 3:]
    dx, dy = d[:, 1:] - d[:, :-1], d[1:] - d[:-1]
    s = 1 / torch.cat([dx.norm(dim=-1).ravel(), dy.norm(dim=-1).ravel()]).median()

    def log_cos(n: Tensor, rays_d: Tensor) -> Tensor:
        return (n * rays_d).sum(-1).abs().clamp_min(1e-3).log()

    a = s * (normals * d).sum(-1).abs()
    terms = [
        (
            a[:, :-1],
            a[:, :-1]
            * (
                log_cos(normals[:, :-1], d[:, 1:]) - log_cos(normals[:, :-1], d[:, :-1])
            ),
            a[:, 1:],
            a[:, 1:]
            * (log_cos(normals[:, 1:], d[:, 1:]) - log_cos(normals[:, 1:], d[:, :-1])),
        ),
        (
            a[:-1],
            a[:-1] * (log_cos(normals[:-1], d[1:]) - log_cos(normals[:-1], d[:-1])),
            a[1:],
            a[1:] * (log_cos(normals[1:], d[1:]) - log_cos(normals[1:], d[:-1])),
        ),
    ]
    return a, terms


def diffs(z: Tensor) -> tuple[Tensor, Tensor]:
    """Forward differences along x (H, W-1) and y (H-1, W)."""
    return z[:, 1:] - z[:, :-1], z[1:] - z[:-1]


def cut_weights(
    z0: Tensor, terms: list, tol: float, window: int = CUT_WINDOW
) -> list[tuple[Tensor, Tensor]]:
    """Weights that cut the prior's occlusion edges along closed contours; else 0.5.

    Each pixel lies above or below the local mid-depth (mean of the max and min of log
    depth in a window that spans a soft edge). An edge (p, q) is cut where p and q lie
    on different sides and the prior's step departs from the slope the normals imply by
    more than tol. A level set is closed, so two surfaces cannot leak into each other,
    and the pixels of a soft ramp join their side instead of a break picked row by row
    as BiNI's weights do. Where three layers meet, the middle one changes side and its
    contour stops short; `close_cuts` joins such ends.
    """
    pool = torch.nn.functional.max_pool2d
    hi = pool(z0[None, None], window, stride=1, padding=window // 2)[0, 0]
    lo = -pool(-z0[None, None], window, stride=1, padding=window // 2)[0, 0]
    side = z0 > 0.5 * (hi + lo)
    cuts, steps = [], []
    for (af, gf, ab, gb), dz, crossed in zip(
        terms,
        diffs(z0),
        (side[:, 1:] != side[:, :-1], side[1:] != side[:-1]),
        strict=True,
    ):
        r = (dz + 0.5 * (gf / af.clamp_min(1e-12) + gb / ab.clamp_min(1e-12))).abs()
        cuts.append((crossed & (r > tol)).cpu().numpy())
        steps.append((r / tol).cpu().numpy())
    out = []
    for cut in close_cuts(cuts, steps, budget=float(window)):
        keep = 0.5 * ~torch.from_numpy(cut).to(z0.device)
        out.append((keep, keep.clone()))
    return out


def crack_tips(cut_x: np.ndarray, cut_y: np.ndarray) -> np.ndarray:
    """Corners (H+1, W+1) inside the image where one cut edge ends: open contours.

    The cut between (y, x) and (y, x+1) joins corners (y, x+1) and (y+1, x+1); between
    (y, x) and (y+1, x), corners (y+1, x) and (y+1, x+1).
    """
    h, w = cut_x.shape[0], cut_y.shape[1]
    deg = np.zeros((h + 1, w + 1), int)
    deg[:-1, 1:-1] += cut_x
    deg[1:, 1:-1] += cut_x
    deg[1:-1, :-1] += cut_y
    deg[1:-1, 1:] += cut_y
    tips = deg == 1
    tips[0], tips[-1], tips[:, 0], tips[:, -1] = False, False, False, False
    return tips


Corner = tuple[int, int]
Edge = tuple[int, int, int]  # axis (0: x, 1: y) and the pixel (y, x) on its low side


def corner_moves(c: Corner, h: int, w: int) -> list[tuple[Corner, Edge]]:
    """Neighbors of a pixel corner, each with the pixel edge stepping there cuts."""
    i, j = c
    moves = []
    if 0 < j < w and i < h:
        moves.append(((i + 1, j), (0, i, j - 1)))
    if 0 < j < w and i > 0:
        moves.append(((i - 1, j), (0, i - 1, j - 1)))
    if 0 < i < h and j < w:
        moves.append(((i, j + 1), (1, i - 1, j)))
    if 0 < i < h and j > 0:
        moves.append(((i, j - 1), (1, i - 1, j - 1)))
    return moves


def close_cuts(
    cuts: list[np.ndarray], steps: list[np.ndarray], budget: float
) -> list[np.ndarray]:
    """Extend open contours to the nearest contour or the border, if one is close.

    Shortest path on the pixel corners from each crack tip; cutting edge e costs max(1,
    1 / step_e^2), with step the prior's departure from the normals in units of tol, so
    the path follows the prior's edge and does not cross a smooth surface (noise of tol
    / 4 costs 16). Paths dearer than `budget` are not taken: a contour that ends because
    the occlusion fades out stays open.
    """
    (h, w1), (h1, w) = cuts[0].shape, cuts[1].shape
    assert (h1, w1) == (h - 1, w - 1) and [c.shape for c in cuts] == [
        s.shape for s in steps
    ]
    cuts = [c.copy() for c in cuts]
    costs = [np.maximum(1, np.maximum(s, 1e-6) ** -2) for s in steps]
    for tip in map(tuple, np.argwhere(crack_tips(*cuts)).tolist()):
        if (
            sum(cuts[a][y, x] for _, (a, y, x) in corner_moves(tip, h, w)) == 1
        ):  # not closed by an earlier path
            for a, y, x in shortest_cut(tip, cuts, costs, budget):
                cuts[a][y, x] = True
    return cuts


def cut_moves(c: Corner, cuts: list[np.ndarray], h: int, w: int) -> list[Corner]:
    """Corners one cut edge away from c."""
    return [n for n, (a, y, x) in corner_moves(c, h, w) if cuts[a][y, x]]


def shortest_cut(
    tip: Corner, cuts: list[np.ndarray], costs: list[np.ndarray], budget: float
) -> list[Edge]:
    """Cheapest edges from a crack tip to a contour or the border (Dijkstra).

    Empty if dearer than budget.

    The tip's own contour, up to `budget` corners back along it, is not a target, or
    every tip would close on itself.
    """
    h, w = cuts[1].shape[0] + 1, cuts[0].shape[1] + 1
    own, frontier = {tip}, [tip]
    for _ in range(int(budget)):
        frontier = [
            n for c in frontier for n in cut_moves(c, cuts, h, w) if n not in own
        ]
        own.update(frontier)
    dist, prev, heap = {tip: 0.0}, {}, [(0.0, tip)]
    while heap:
        d, c = heapq.heappop(heap)
        if c not in own and (
            c[0] in {0, h} or c[1] in {0, w} or cut_moves(c, cuts, h, w)
        ):
            path = []
            while c != tip:
                c, edge = prev[c]
                path.append(edge)
            return path
        for n, (a, y, x) in corner_moves(c, h, w):
            nd = d + costs[a][y, x]
            if d == dist[c] and nd <= budget and nd < dist.get(n, np.inf):
                dist[n], prev[n] = nd, (c, (a, y, x))
                heapq.heappush(heap, (nd, n))
    return []


@dataclass(frozen=True)
class NormalTerm:
    """Per axis, BiNI's coefficients (a_p, g_p, a_q, g_q) and weights (w+, w-).

    The weights are 0 on cut edges.
    """

    terms: list[tuple[Tensor, Tensor, Tensor, Tensor]]
    weights: list[tuple[Tensor, Tensor]]


def normal_term(
    normals: Tensor, rays: Tensor, d0: Tensor, edge_tol: float
) -> NormalTerm:
    """BiNI's residuals for the normals, cut on d0's occlusion contours (tol: log)."""
    h, w = d0.shape
    assert normals.shape == (h, w, 3) and normals.isfinite().all(), normals.shape
    assert ((normals.norm(dim=-1) - 1).abs() < 1e-3).all()  # noqa: PLR2004
    _, terms = neighbor_terms(normals.double(), rays.double())
    weights = cut_weights(-d0.double().log(), terms, edge_tol)
    return NormalTerm(
        [tuple(t.to(d0.dtype) for t in axis) for axis in terms],
        [tuple(t.to(d0.dtype) for t in axis) for axis in weights],
    )


DSINE_SIGNS = {
    "as_is": (1, 1, 1),
    "negated": (-1, -1, -1),
    "opengl": (1, -1, -1),
    "flip_xy": (-1, -1, 1),
}


def dsine_normals(images: np.ndarray, intrinsics: np.ndarray, repo: Path) -> np.ndarray:
    """DSINE's camera-frame normals (V, H, W, 3) of uint8 images, run on CUDA."""
    assert images.dtype == np.uint8 and images.shape[-1] == 3, images.shape  # noqa: PLR2004
    assert intrinsics.shape == (len(images), 3, 3), intrinsics.shape
    sys.path.insert(0, str(repo))
    spec = importlib.util.spec_from_file_location("dsine_hubconf", repo / "hubconf.py")
    assert spec is not None and spec.loader is not None
    hub = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(hub)
    predictor = hub.DSINE()
    out = []
    with torch.no_grad():
        for image, k in zip(images, intrinsics, strict=True):
            x = torch.from_numpy(image).float().permute(2, 0, 1)[None] / 255
            n = predictor.infer_tensor(x, torch.from_numpy(k).float()[None].cuda())[0]
            out.append(n.permute(1, 2, 0).cpu().numpy().astype(np.float32))
    normals = np.stack(out)
    assert normals.shape == images.shape and np.isfinite(normals).all()
    return normals


def to_world(
    normals_cam: np.ndarray, cam_to_world: np.ndarray, depth: Tensor, rays: Tensor
) -> tuple[Tensor, str]:
    """Rotate camera-frame normals into the ray frame; axis signs fit to the depth."""
    rot = torch.from_numpy(cam_to_world[:3, :3]).to(rays)
    ref = normals_from_depth(depth, rays)
    cam = torch.from_numpy(normals_cam).to(rays)
    worlds = {
        name: (cam * torch.tensor(sign).to(cam)) @ rot.T
        for name, sign in DSINE_SIGNS.items()
    }
    agree = {
        name: (world * ref).sum(-1).median().item() for name, world in worlds.items()
    }
    best = max(agree, key=agree.__getitem__)
    assert agree[best] > 0.5, agree  # noqa: PLR2004
    world = worlds[best]
    return world / world.norm(dim=-1, keepdim=True), best


def main() -> None:
    """Run DSINE on prepared views with the backbone's intrinsics; cache the normals."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("views", type=Path, nargs="+", help="prepared views (.npz)")
    parser.add_argument(
        "--backbone", default="dejaview_k16", help="geometry next to the views"
    )
    parser.add_argument("--repo", type=Path, default=DSINE_REPO)
    args = parser.parse_args()
    for path in args.views:
        geometry = Geometry.load(path.with_name(f"{args.backbone}_{path.stem}.npz"))
        normals = dsine_normals(Views.load(path).images, geometry.intrinsics, args.repo)
        out = path.with_name(f"dsine_{path.stem}.npz")
        np.savez_compressed(out, normals=normals)
        print(f"{path.parent.name}: {len(normals)} views -> {out}")  # noqa: T201


if __name__ == "__main__":
    main()
