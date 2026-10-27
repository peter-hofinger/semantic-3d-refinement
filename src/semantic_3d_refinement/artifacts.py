"""Does Déjà View show the artifacts our proposal names? Each one measured against ETH3D's laser scans.

The proposal (§1, §4) motivates refinement by five failures of feed-forward reconstruction: depth bleeding across
boundaries (flying pixels), wavy or thick planes, cross-view drift that layers surfaces, failures in reflective or
textureless regions, and confidence that cannot locate the errors. Here each is tested on all 20 views of ETH3D
office and kicker, Déjà View with K = 16 loops, the inputs of `evaluate.py` and `multiview.py`:

- per view, Déjà View's depth is scaled by its median ratio to the scan's depth and compared pixel by pixel,
  e = log(s D / D_gt) (about the relative error), where the scan saw the pixel;
- planes are found in the scan by RANSAC; a view's pixels on one are lifted through ETH3D's camera, and the flatness
  of Déjà View's points is their RMS distance to their own best-fit plane, next to the scan's;
- layering uses the fused cloud of all views in the scan's frame (`multiview.similarity` on the camera centers).

Glass and lamps return no laser, so where they are, the scan has holes and only Déjà View's guess can be shown.
`uv run python -m semantic_3d_refinement.artifacts --cfg configs/artifacts_eth3d.yaml` writes the measurements to
`runs/<cfg>-<time>/run.yaml` and slide-sized figures (16:9), `slides.pdf` and a README to
`figures/<cfg>-<time>/`.
"""

import argparse
import textwrap
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from PIL import Image
from scipy import ndimage
from scipy.stats import spearmanr

from semantic_3d_refinement.evaluate import lift, scan_depth
from semantic_3d_refinement.multiview import (
    MultiviewConfig,
    find_matches,
    load_scene,
    residuals,
    similarity,
    voxel_downsample,
)
from semantic_3d_refinement.normals import grid_score, normals_from_depth
from semantic_3d_refinement.views import git_hash
from semantic_3d_refinement.visualize import (
    BLUE,
    DIVERGING,
    GRID,
    INK,
    INK_2,
    ORANGE,
    SEQUENTIAL,
    SURFACE,
    camera_rays,
    up_frame,
)

SLIDE = (13.33, 7.5)
SCENE_COLORS = {"office": BLUE, "kicker": ORANGE}


@dataclass(frozen=True)
class SceneData:
    """One scene: images, scan depth, Déjà View's per-view scaled depth and confidence, cameras, fused cloud, scan.

    Arrays are (V, H, W[, C]); gt is 0 where the scan saw nothing. fused holds Déjà View's points of every view in
    the scan's frame; rays are Déjà View's pinhole rays in its own frame (`rays: pinhole`), predicted_rays its own.
    """

    images: np.ndarray
    gt: np.ndarray
    pred: np.ndarray
    conf: np.ndarray
    k: np.ndarray
    cam_to_world: np.ndarray
    rays: np.ndarray
    predicted_rays: np.ndarray
    fused: np.ndarray
    scan: np.ndarray
    scales: np.ndarray

    @property
    def err(self) -> np.ndarray:
        """Log ratio of the scaled prediction to the scan's depth, NaN where the scan saw nothing."""
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(self.gt > 0, np.log(self.pred / self.gt), np.nan)


def load_data(paths: dict, device: torch.device) -> SceneData:
    """Déjà View's export, ETH3D's views and the scan of one scene, with the scan's depth in every view."""
    scene = load_scene(Path(paths["input"]))
    cams, scan = np.load(paths["gt_views"]), np.load(paths["scan"])
    k, c2w = cams["intrinsics"].astype(np.float64), cams["cam_to_world"].astype(np.float64)
    h, w = scene.depth.shape[1:]
    scan_gpu = torch.from_numpy(scan).to(device)
    gt = np.stack([
        scan_depth(scan_gpu, torch.from_numpy(ki).to(scan_gpu), torch.from_numpy(ci).to(scan_gpu), h, w).cpu().numpy()
        for ki, ci in zip(k, c2w, strict=True)
    ]).astype(np.float64)  # fmt: skip
    depth = scene.depth.double().numpy()
    scales = np.array([np.median(g[g > 0] / d[g > 0]) for g, d in zip(gt, depth, strict=True)])
    rays = scene.rays.double().numpy()
    s, rotation, t = similarity(scene.center.double().numpy(), c2w[:, :3, 3])
    fused = s * (rays[..., :3] + depth[..., None] * rays[..., 3:]) @ rotation.T + t
    export = np.load(paths["input"])
    predicted = np.concatenate([export["ray_origins"], export["ray_directions"]], axis=-1).astype(np.float64)
    return SceneData(cams["images"], gt, scales[:, None, None] * depth, scene.conf.double().numpy(), k, c2w, rays,
                     predicted, fused, scan, scales)  # fmt: skip


def percent(x: np.ndarray) -> float:
    """Median of |x| in percent, over the finite entries."""
    return float(100 * np.nanmedian(np.abs(x)))


def depth_edges(gt: np.ndarray, jump: float) -> np.ndarray:
    """Pixels with a 4-neighbor the scan also saw whose log depth differs by more than jump: occlusion edges."""
    with np.errstate(divide="ignore"):
        z = np.where(gt > 0, np.log(gt), np.nan)
    edge = np.zeros(gt.shape, dtype=bool)
    for axis in (-1, -2):
        step = np.abs(np.diff(z, axis=axis)) > jump
        lo = [slice(None)] * gt.ndim
        hi = [slice(None)] * gt.ndim
        lo[axis], hi[axis] = slice(None, -1), slice(1, None)
        edge[tuple(lo)] |= step
        edge[tuple(hi)] |= step
    return edge


def bleeding(data: SceneData, jump: float, max_px: int) -> tuple[dict, dict]:
    """Error against distance (px) to the nearest occlusion edge of the scan, and the share of pixels off by > 5 %."""
    edges = depth_edges(data.gt, jump)
    dist = np.stack([ndimage.distance_transform_edt(~e) for e in edges])
    err = np.abs(data.err)
    rows = []
    for b in range(max_px + 1):
        sel = (np.floor(dist) == b) & np.isfinite(err)
        rows.append((100 * float(np.median(err[sel])), 100 * float((err[sel] > 0.05).mean()), int(sel.sum())))  # noqa: PLR2004
    far = np.isfinite(err) & (dist > max_px)
    far_median, far_off = 100 * float(np.median(err[far])), 100 * float((err[far] > 0.05).mean())  # noqa: PLR2004
    summary = {
        "median_at_edge_pct": rows[0][0], "median_at_5px_pct": rows[5][0], "median_far_pct": far_median,
        "off_5pct_at_edge": rows[0][1], "off_5pct_far": far_off, "edge_pixels_pct": 100 * float(edges.mean()),
    }  # fmt: skip
    return summary, {"edges": edges, "dist": dist, "curve": np.array(rows)}


def edge_example(data: SceneData, edges: np.ndarray, view: int, half: int = 40) -> tuple[int, int, str]:
    """Center (row, col) of the crop of a view with the most strong edge pixels, and the profile's axis there."""
    strong = edges[view] & (np.abs(data.err[view]) < np.inf)
    with np.errstate(divide="ignore"):
        z = np.where(data.gt[view] > 0, np.log(data.gt[view]), np.nan)
    gx = np.abs(np.nan_to_num(np.diff(z, axis=1, append=np.nan)))
    gy = np.abs(np.nan_to_num(np.diff(z, axis=0, append=np.nan)))
    strong &= np.maximum(gx, gy) > 0.2  # noqa: PLR2004
    score = ndimage.uniform_filter(strong.astype(float), 2 * half + 1)
    h, w = score.shape
    score[:half], score[-half:], score[:, :half], score[:, -half:] = 0, 0, 0, 0
    r, c = np.unravel_index(np.argmax(score), score.shape)
    axis = (
        "row"
        if gx[r - half : r + half, c - half : c + half].sum() > gy[r - half : r + half, c - half : c + half].sum()
        else "col"
    )
    assert half <= r < h - half and half <= c < w - half
    return int(r), int(c), axis


def ransac_planes(points: np.ndarray, count: int, tol: float, min_frac: float, device: torch.device) -> list:
    """Up to count planes (unit normal n, offset c: n.x = c) holding at least min_frac of the points, largest first."""
    gen = torch.Generator(device=device).manual_seed(0)
    pts = torch.from_numpy(points).to(device)
    total, planes = len(pts), []
    for _ in range(count):
        if len(pts) < 3:  # noqa: PLR2004
            break
        idx = torch.randint(len(pts), (1024, 3), generator=gen, device=device)
        a, b, c = pts[idx].unbind(1)
        n = torch.linalg.cross(b - a, c - a)
        n = n / n.norm(dim=1, keepdim=True).clamp_min(1e-12)
        votes = torch.stack([((pts @ n[i : i + 64].T) - (n[i : i + 64] * a[i : i + 64]).sum(1)).abs().lt(tol).sum(0)
                             for i in range(0, 1024, 64)]).flatten()  # fmt: skip
        best = int(votes.argmax())
        inliers = ((pts @ n[best]) - n[best] @ a[best]).abs() < tol
        for _ in range(3):
            q = pts[inliers]
            normal = torch.linalg.svd(q - q.mean(0), full_matrices=False).Vh[-1]
            offset = normal @ q.mean(0)
            inliers = ((pts @ normal) - offset).abs() < tol
        if inliers.sum() < min_frac * total:
            break
        planes.append((normal.cpu().numpy(), float(offset)))
        pts = pts[~inliers]
    return planes


def plane_residuals(points: np.ndarray) -> np.ndarray:
    """Signed distances of points (N, 3) to their least-squares plane."""
    centered = points - points.mean(0)
    return centered @ np.linalg.svd(centered, full_matrices=False)[2][-1]


def flatness(data: SceneData, planes: list, tol: float, min_px: int) -> tuple[dict, list[dict]]:
    """Per view and scan plane: RMS of Déjà View's points and of the scan's around their own best-fit planes (cm).

    A view's pixels on a plane are those whose scan point lies within tol of it, eroded by 3 px, and only their
    largest connected region: coplanar but separate pieces (lamps on a ceiling) would add offsets, not waviness.
    """
    records = []
    for v in range(len(data.gt)):
        gt_pts = lift(data.gt[v], data.k[v], data.cam_to_world[v])
        pred_pts = lift(data.pred[v], data.k[v], data.cam_to_world[v])
        for p, (normal, offset) in enumerate(planes):
            on = (data.gt[v] > 0) & (np.abs(gt_pts @ normal - offset) < tol)
            labels, count = ndimage.label(ndimage.binary_erosion(on, iterations=3))
            if count == 0:
                continue
            on = labels == 1 + int(np.argmax(ndimage.sum(np.ones_like(labels), labels, range(1, count + 1))))
            if on.sum() < min_px:
                continue
            pred_res, gt_res = plane_residuals(pred_pts[on]), plane_residuals(gt_pts[on])
            records.append({"view": v, "plane": p, "pixels": int(on.sum()), "mask": on,
                            "pred_rms_cm": 100 * float(np.sqrt(np.mean(pred_res**2))),
                            "gt_rms_cm": 100 * float(np.sqrt(np.mean(gt_res**2))), "pred_res": pred_res})  # fmt: skip
    pred = [r["pred_rms_cm"] for r in records]
    gt = [r["gt_rms_cm"] for r in records]
    summary = {"view_planes": len(records), "pred_rms_cm_median": float(np.median(pred)),
               "gt_rms_cm_median": float(np.median(gt)), "pred_rms_cm_p90": float(np.percentile(pred, 90))}  # fmt: skip
    return summary, records


def patch_grid(data: SceneData) -> dict:
    """Grid score (`normals.grid_score`, 1 without a grid) of Déjà View's normals and the scan's, median over views.

    Déjà View's depth is lifted along its predicted rays and along the fitted pinhole, which tells whether the grid
    sits in the depth or in the ray map.
    """
    pred, pred_rays, scan = [], [], []
    for v in range(len(data.gt)):
        depth = torch.from_numpy(data.pred[v])
        pred.append(grid_score(normals_from_depth(depth, torch.from_numpy(data.rays[v]))))
        pred_rays.append(grid_score(normals_from_depth(depth, torch.from_numpy(data.predicted_rays[v]))))
        rays = camera_rays(data.k[v], *data.gt[v].shape)
        scan.append(grid_score(normals_from_depth(torch.from_numpy(np.where(data.gt[v] > 0, data.gt[v], 1.0)), rays)))
    return {"pinhole_grid_score": float(np.median(pred)), "predicted_rays_grid_score": float(np.median(pred_rays)),
            "scan_grid_score": float(np.median(scan))}  # fmt: skip


def layering(data: SceneData, paths: dict, mv: MultiviewConfig, device: torch.device) -> dict:
    """Disagreement between overlapping views, and each view's depth bias in the shared frame against the scan."""
    scene = load_scene(Path(paths["input"])).to(device)
    scale = scene.depth.median()
    m = find_matches(scene, scale / scene.depth, scale, mv.mv_tol, mv.neighbors)
    disagreement = residuals(scale / scene.depth, scale, scene, m).abs().cpu().numpy()
    bias = []
    for v in range(len(data.gt)):
        world_to_cam = np.linalg.inv(data.cam_to_world[v])
        z = (data.fused[v] @ world_to_cam[:3, :3].T + world_to_cam[:3, 3])[..., 2]
        seen = (data.gt[v] > 0) & (z > 0)
        bias.append(100 * float(np.median(np.log(z[seen] / data.gt[v][seen]))))
    return {"disagreement_pct": 100 * float(np.median(disagreement)),
            "disagreement_p90_pct": 100 * float(np.percentile(disagreement, 90)), "view_bias_pct": bias,
            "view_bias_spread_pct": float(np.ptp(bias)),
            "scale_spread_pct": 100 * float(np.ptp(np.log(data.scales)))}  # fmt: skip


def texture(images: np.ndarray, window: int) -> np.ndarray:
    """Local standard deviation of gray level (0-1) in a window: low where a region has no texture."""
    gray = images.astype(np.float64).mean(-1) / 255
    mean = ndimage.uniform_filter(gray, (1, window, window))
    return np.sqrt(np.clip(ndimage.uniform_filter(gray**2, (1, window, window)) - mean**2, 0, None))


def surfaces(data: SceneData, window: int, max_px: int, edges_dist: np.ndarray) -> tuple[dict, dict]:
    """Error by texture decile, away from occlusion edges (so bleeding does not count), and the scan's holes."""
    tex = texture(data.images, window)
    err = np.abs(data.err)
    sel = np.isfinite(err) & (edges_dist > max_px)
    bounds = np.quantile(tex[sel], np.linspace(0, 1, 11))
    decile = np.clip(np.searchsorted(bounds, tex, side="right") - 1, 0, 9)
    medians = [100 * float(np.median(err[sel & (decile == d)])) for d in range(10)]
    holes = data.gt == 0
    summary = {"median_by_texture_decile_pct": medians, "texture_bounds": bounds.tolist(),
               "hole_pct": 100 * float(holes.mean())}  # fmt: skip
    return summary, {"texture": tex, "decile": decile}


def confidence(data: SceneData) -> tuple[dict, dict]:
    """Rank correlation of confidence with error, and the sparsification curve against the oracle (AUSE)."""
    err, conf = np.abs(data.err), data.conf
    sel = np.isfinite(err)
    e, c = err[sel], conf[sel]
    sub = np.random.default_rng(0).choice(len(e), min(500_000, len(e)), replace=False)
    rho = float(spearmanr(c[sub], e[sub]).statistic)
    fractions = np.linspace(0, 0.9, 46)

    def curve(order: np.ndarray) -> np.ndarray:
        remaining = e[order][::-1]
        csum = np.cumsum(remaining[::-1])[::-1]
        keep = (len(e) * (1 - fractions)).astype(int)
        return np.array([csum[len(e) - n] / n for n in keep])

    by_conf = curve(np.argsort(c)[::-1])
    oracle = curve(np.argsort(e))
    ause = float(np.trapezoid(by_conf - oracle, fractions) / np.trapezoid(by_conf[0] - oracle, fractions))
    summary = {"spearman_conf_vs_error": rho, "mean_error_pct": 100 * float(e.mean()),
               "mean_error_pct_top_half_conf": 100 * float(by_conf[25]), "oracle_top_half_pct": 100 * float(oracle[25]),
               "ause_fraction_of_random": ause}  # fmt: skip
    return summary, {"fractions": fractions, "by_conf": by_conf, "oracle": oracle}


def slide(title: str, subtitle: str) -> tuple[Figure, dict]:
    """Start a 16:9 figure with a title and a takeaway; return it and the text style of panel titles."""
    fig = plt.figure(figsize=SLIDE, facecolor=SURFACE)
    fig.text(0.03, 0.955, title, fontsize=22, color=INK, weight="bold", va="top")
    fig.text(0.03, 0.895, textwrap.fill(subtitle, 125), fontsize=13.5, color=INK_2, va="top", linespacing=1.4)
    return fig, {"fontsize": 12, "color": INK}


def save(fig: Figure, out: Path, name: str, source: str) -> None:
    """Write the slide with a source line at the bottom."""
    fig.text(0.03, 0.015, textwrap.fill(source, 190), fontsize=9, color=INK_2, va="bottom")
    fig.savefig(out / f"{name}.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)


SOURCE = (
    "Déjà View (K = 16) on 20 views each of ETH3D office and kicker; ground truth: ETH3D laser scans (CC BY-NC-SA 4.0)."
)


def crop_axes(fig: Figure, rect: tuple[float, float, float, float], n: int) -> list:
    """N image axes side by side in a figure rectangle (left, bottom, width, height)."""
    left, bottom, width, height = rect
    gap = 0.01
    w = (width - gap * (n - 1)) / n
    return [fig.add_axes((left + i * (w + gap), bottom, w, height)) for i in range(n)]


def draw_bleeding(out: Path, data: dict, results: dict, example: dict) -> None:
    """Error against distance to occlusion edges; one edge in close-up with its depth profile."""
    o, k = results["office"]["bleeding"], results["kicker"]["bleeding"]
    fig, kw = slide("1 · Depth bleeding at object boundaries: confirmed",
                    f"Median depth error {o['median_at_edge_pct']:.1f} % (office) and {k['median_at_edge_pct']:.1f} % "
                    f"(kicker) on pixels at a depth edge, {o['median_at_5px_pct']:.1f} / "
                    f"{k['median_at_5px_pct']:.1f} % five pixels away, {o['median_far_pct']:.1f} / "
                    f"{k['median_far_pct']:.1f} % far from edges. "
                    f"{o['off_5pct_at_edge']:.0f} / {k['off_5pct_at_edge']:.0f} % of edge pixels are off by more than "
                    "5 %: they float between foreground and background.")  # fmt: skip
    ax = fig.add_axes((0.06, 0.12, 0.34, 0.62))
    for name, r in results.items():
        curve = r["arrays"]["bleeding"]["curve"]
        ax.plot(np.arange(len(curve)), curve[:, 0], color=SCENE_COLORS[name], lw=2.2, marker="o", ms=4, label=name)
        ax.axhline(r["bleeding"]["median_far_pct"], color=SCENE_COLORS[name], lw=1, ls=":")
    ax.set(xlabel="distance to the nearest depth edge of the scan (px)", ylabel="median depth error (%)",
           ylim=(0, None))  # fmt: skip
    ax.legend(frameon=False, fontsize=12)
    ax.set_title("dotted: median far from edges", fontsize=11, color=INK_2)
    name, view, (r, c, axis), half = example["scene"], example["view"], example["center"], 40
    d = data[name]
    win = (slice(r - half, r + half), slice(c - half, c + half))
    gt, pred = d.gt[view][win], d.pred[view][win]
    lo, hi = np.percentile(gt[gt > 0], [2, 98])
    axes = crop_axes(fig, (0.44, 0.43, 0.54, 0.31), 4)
    panels = [(d.images[view][win], "image", None), (np.where(gt > 0, gt, np.nan), "laser scan depth", "depth"),
              (pred, "Déjà View depth", "depth"),
              (100 * np.abs(d.err[view][win]), "error (0-20 %)", "err")]  # fmt: skip
    for a, (img, title, kind) in zip(axes, panels, strict=True):
        if kind is None:
            a.imshow(img)
        elif kind == "depth":
            a.imshow(img, cmap=SEQUENTIAL.with_extremes(bad=GRID), vmin=lo, vmax=hi)
        else:
            a.imshow(img, cmap="magma", vmin=0, vmax=20)
        if axis == "row":
            a.axhline(half, color="white", lw=1.2, ls="--")
        else:
            a.axvline(half, color="white", lw=1.2, ls="--")
        a.set_title(title, **kw)
        a.set_axis_off()
    ax2 = fig.add_axes((0.47, 0.12, 0.5, 0.22))
    line = (lambda x: x[half]) if axis == "row" else (lambda x: x[:, half])
    px = np.arange(-half, half)
    g = line(gt)
    ax2.plot(px, np.where(g > 0, g, np.nan), color=INK, lw=2.4, drawstyle="steps-mid", label="laser scan")
    ax2.plot(px, line(pred), color=ORANGE, lw=2.2, marker="o", ms=3, label="Déjà View")
    ax2.set(xlabel="pixels along the dashed line", ylabel="depth (m)")
    ax2.legend(frameon=False, fontsize=11, loc="best")
    ax2.set_title(f"{name} view {view}: the scan steps, Déjà View ramps", fontsize=11, color=INK_2)
    save(fig, out, "1_bleeding", SOURCE + " Depth edge: the scan's log depth jumps by more than 5 % between neighbors.")


def draw_planes(out: Path, data: dict, results: dict) -> None:
    """Deviation from the best-fit plane for the median view-plane, Déjà View vs scan, and RMS over all of them."""
    records = [(n, x) for n, r in results.items() for x in r["arrays"]["planes"]]
    pred = np.array([x["pred_rms_cm"] for _, x in records])
    gt = np.array([x["gt_rms_cm"] for _, x in records])
    o, k = results["office"]["grid"], results["kicker"]["grid"]
    fig, kw = slide("2 · Wavy planes: confirmed",
                    f"On walls, floors, ceilings and tables, Déjà View's points lie {np.median(pred):.1f} cm RMS from "
                    f"their best-fit plane (median over {len(pred)} view-planes; worst tenth over "
                    f"{np.percentile(pred, 90):.0f} cm); the scan's {np.median(gt):.2f} cm. Its 14 px patch grid is in "
                    f"the ray map, not the depth: grid score {o['predicted_rays_grid_score']:.1f} / "
                    f"{k['predicted_rays_grid_score']:.1f} along its own rays, {o['pinhole_grid_score']:.1f} / "
                    f"{k['pinhole_grid_score']:.1f} along a pinhole (1 = no grid).")  # fmt: skip
    name, rec = records[int(np.argsort(pred)[len(pred) // 2])]
    d = data[name]
    view, mask = rec["view"], rec["mask"]
    dev = np.full(mask.shape, np.nan)
    dev[mask] = 100 * rec["pred_res"]
    dev_gt = np.full(mask.shape, np.nan)
    dev_gt[mask] = 100 * plane_residuals(lift(d.gt[view], d.k[view], d.cam_to_world[view])[mask])
    axes = crop_axes(fig, (0.03, 0.2, 0.64, 0.52), 3)
    axes[0].imshow(d.images[view])
    axes[0].contour(mask, levels=[0.5], colors=[ORANGE], linewidths=1.5)
    axes[0].set_title(f"{name} view {view}: one plane (outlined)", **kw)
    cmap = DIVERGING.with_extremes(bad=GRID)
    lim = max(2.0, float(np.ceil(np.nanpercentile(np.abs(dev), 95))))
    for a, img, title in [(axes[1], dev, f"Déjà View: RMS {rec['pred_rms_cm']:.1f} cm"),
                          (axes[2], dev_gt, f"laser scan: RMS {rec['gt_rms_cm']:.2f} cm")]:  # fmt: skip
        im = a.imshow(img, cmap=cmap, norm=Normalize(-lim, lim))
        a.set_title(title, **kw)
    for a in axes:
        a.set_axis_off()
    cax = fig.add_axes((0.25, 0.14, 0.4, 0.025))
    fig.colorbar(im, cax=cax, orientation="horizontal", label="distance to its own best-fit plane (cm)")
    ax = fig.add_axes((0.73, 0.14, 0.24, 0.58))
    bins = np.geomspace(0.03, 30, 40)
    ax.hist(gt, bins=bins, color=INK_2, alpha=0.7, label="laser scan")
    ax.hist(pred, bins=bins, color=ORANGE, alpha=0.8, label="Déjà View")
    ax.set(xscale="log", xlabel="RMS distance to plane (cm)", ylabel="view-planes")
    ax.legend(frameon=False, fontsize=11)
    ax.set_title(f"all {len(pred)} view-planes (median example)", fontsize=11, color=INK)
    save(fig, out, "2_planes", SOURCE + " Planes: RANSAC on the scan (1 cm); per view, the largest connected region "
         "within 1 cm, eroded by 3 px.")  # fmt: skip


def draw_layering(out: Path, data: dict, results: dict) -> None:
    """Top-down slab of the fused cloud against the scan, and each view's depth bias in the shared frame."""
    o, k = results["office"]["layering"], results["kicker"]["layering"]
    fig, _ = slide("3 · Cross-view drift layers the surfaces: confirmed",
                   f"Overlapping views disagree by {o['disagreement_pct']:.1f} % (office) and "
                   f"{k['disagreement_pct']:.1f} % (kicker) of depth, median; each view sits at its own offset "
                   f"(spread {o['view_bias_spread_pct']:.1f} / {k['view_bias_spread_pct']:.1f} %). Fused, every wall "
                   "comes out several times. Gray: laser scan; blue: all 20 Déjà View depth maps.")  # fmt: skip
    windows = {"office": [[-1.3, 0.3], [-3.5, -2.5]], "kicker": [[2.9, 4.3], [-1.3, 0.1]]}
    for i, (name, d) in enumerate(data.items()):
        frame = up_frame(d.cam_to_world)
        height = (d.cam_to_world[:, :3, 3] @ frame[2]).mean()
        ax = fig.add_axes((0.05 + 0.31 * i, 0.12, 0.27, 0.6))
        for pts, color in [(d.scan, "#9a9994"), (d.fused.reshape(-1, 3), BLUE)]:
            q = pts @ frame.T
            q = q[np.abs(q[:, 2] - height) < 0.01]  # noqa: PLR2004
            ax.scatter(q[:, 0], q[:, 1], s=0.4, color=color, linewidths=0, rasterized=True)
        ax.set(xlim=windows[name][0], ylim=windows[name][1], aspect="equal", xlabel="m", ylabel="m")
        ax.set_title(f"{name}: 2 cm slab at camera height, top view", fontsize=11, color=INK)
    ax = fig.add_axes((0.69, 0.12, 0.28, 0.6))
    for i, (name, r) in enumerate(results.items()):
        ax.bar(np.arange(20) + 0.4 * i, r["layering"]["view_bias_pct"], width=0.4, color=SCENE_COLORS[name], label=name)
    ax.axhline(0, color=INK_2, lw=0.8)
    ax.set(xlabel="view", ylabel="median depth offset vs scan (%)", xticks=range(0, 20, 2))
    ax.legend(frameon=False, fontsize=11)
    ax.set_title("each view's offset in the shared frame", fontsize=11, color=INK)
    save(fig, out, "3_layering", SOURCE + " Fused cloud mapped to the scan by one similarity on the camera centers.")


def draw_surfaces(out: Path, data: dict, results: dict, glass: dict) -> None:
    """Error by texture decile away from edges; Déjà View's surface where the scan has a hole (glass)."""
    o, k = results["office"]["surfaces"], results["kicker"]["surfaces"]
    fig, kw = slide("4 · Textureless: not confirmed · reflective: not measurable",
                    "Away from depth edges, textureless regions are no worse than textured ones (office: "
                    f"{o['median_by_texture_decile_pct'][0]:.1f} % in the flattest tenth vs "
                    f"{o['median_by_texture_decile_pct'][-1]:.1f} % in the most textured; kicker "
                    f"{k['median_by_texture_decile_pct'][0]:.1f} vs {k['median_by_texture_decile_pct'][-1]:.1f} %). "
                    "Glass and lamps return no laser, so ground truth is missing exactly there.")  # fmt: skip
    ax = fig.add_axes((0.06, 0.14, 0.32, 0.58))
    for name, r in results.items():
        ax.plot(np.arange(1, 11), r["surfaces"]["median_by_texture_decile_pct"], color=SCENE_COLORS[name], lw=2.2,
                marker="o", label=name)  # fmt: skip
    ax.set(xlabel="image texture decile (1 = flattest)", ylabel="median depth error (%)", ylim=(0, None),
           xticks=range(1, 11))  # fmt: skip
    ax.legend(frameon=False, fontsize=11)
    ax.set_title("pixels more than 15 px from depth edges", fontsize=11, color=INK_2)
    d = data[glass["scene"]]
    view = glass["view"]
    axes = crop_axes(fig, (0.43, 0.45, 0.55, 0.27), 3)
    axes[0].imshow(d.images[view])
    axes[0].set_title(f"{glass['scene']} view {view}", **kw)
    axes[1].imshow(d.images[view])
    axes[1].imshow(np.where(d.gt[view] == 0, 1.0, np.nan), cmap="autumn", alpha=0.65)
    axes[1].set_title("no laser return (red)", **kw)
    lo, hi = np.percentile(d.pred[view], [2, 98])
    axes[2].imshow(d.pred[view], cmap=SEQUENTIAL, vmin=lo, vmax=hi)
    axes[2].set_title("Déjà View depth", **kw)
    for a in axes:
        a.set_axis_off()
    ax2 = fig.add_axes((0.47, 0.12, 0.5, 0.22))
    ax2.scatter(*glass["wall"].T, s=0.5, color="#9a9994", linewidths=0, rasterized=True, label="laser scan")
    ax2.scatter(*glass["window"].T, s=0.5, color=ORANGE, linewidths=0, rasterized=True,
                label="Déjà View, no laser return")  # fmt: skip
    ax2.set(aspect="equal", xlabel="along the wall (m)", ylabel="depth (m)")
    ax2.legend(frameon=False, fontsize=10, markerscale=12, loc="lower left")
    ax2.set_title("window, top view: the laser hits only the frame; Déjà View puts the glass in its plane",
                  fontsize=11, color=INK_2)  # fmt: skip
    save(fig, out, "4_surfaces", SOURCE + " Texture: gray-level std in 7 x 7 px, deciles per scene.")


def draw_confidence(out: Path, data: dict, results: dict, example: dict) -> None:
    """Sparsification by confidence against the oracle, and one confidence map next to its error."""
    o, k = results["office"]["confidence"], results["kicker"]["confidence"]

    def removed(key: str) -> str:
        return " / ".join(f"{100 * (1 - s[key] / s['mean_error_pct']):.0f}" for s in (o, k))

    fig, kw = slide("5 · Confidence locates the errors only partly",
                    f"Rank correlation of confidence with error: {o['spearman_conf_vs_error']:+.2f} (office), "
                    f"{k['spearman_conf_vs_error']:+.2f} (kicker). Keeping the most confident half removes "
                    f"{removed('mean_error_pct_top_half_conf')} % of the mean error; knowing the true error, the best "
                    f"half would remove {removed('oracle_top_half_pct')} %.")  # fmt: skip
    ax = fig.add_axes((0.06, 0.14, 0.34, 0.58))
    for name, r in results.items():
        a = r["arrays"]["confidence"]
        ax.plot(
            100 * a["fractions"], 100 * a["by_conf"], color=SCENE_COLORS[name], lw=2.2, label=f"{name}: by confidence"
        )
        ax.plot(100 * a["fractions"], 100 * a["oracle"], color=SCENE_COLORS[name], lw=1.5, ls="--",
                label=f"{name}: by true error")  # fmt: skip
    ax.set(
        xlabel="pixels removed, least confident first (%)", ylabel="mean depth error of the rest (%)", ylim=(0, None)
    )
    ax.legend(frameon=False, fontsize=10)
    name, view = example["scene"], example["view"]
    d = data[name]
    axes = crop_axes(fig, (0.44, 0.3, 0.54, 0.38), 3)
    axes[0].imshow(d.images[view])
    axes[0].set_title(f"{name} view {view}", **kw)
    axes[1].imshow(
        d.conf[view], cmap=SEQUENTIAL, vmin=np.percentile(d.conf[view], 2), vmax=np.percentile(d.conf[view], 98)
    )
    axes[1].set_title("confidence (dark = high)", **kw)
    im = axes[2].imshow(100 * np.abs(d.err[view]), cmap=SEQUENTIAL.with_extremes(bad=GRID), vmin=0, vmax=10)
    axes[2].set_title("true error (gray: no scan)", **kw)
    for a in axes:
        a.set_axis_off()
    cax = fig.add_axes((0.81, 0.24, 0.17, 0.022))
    fig.colorbar(im, cax=cax, orientation="horizontal", label="depth error (%)")
    save(fig, out, "5_confidence", SOURCE)


def draw_summary(out: Path, results: dict) -> None:
    """One slide: each named artifact, what was measured, and the verdict."""
    o, k = results["office"], results["kicker"]
    planes = [x for r in results.values() for x in r["arrays"]["planes"]]
    pred = np.median([x["pred_rms_cm"] for x in planes])
    scan = np.median([x["gt_rms_cm"] for x in planes])
    b, sf, c = (o["bleeding"], k["bleeding"]), (o["surfaces"], k["surfaces"]), (o["confidence"], k["confidence"])

    def pair(values: list[float], fmt: str = ".1f") -> str:
        return " / ".join(f"{v:{fmt}}" for v in values)

    tex = [x["median_by_texture_decile_pct"] for x in sf]
    rows = [
        ("Depth bleeding, flying pixels", "confirmed",
         (f"median error {pair([x['median_at_edge_pct'] for x in b])} % at depth edges vs "
         f"{pair([x['median_far_pct'] for x in b])} % away")),
        ("Wavy or thick planes", "confirmed",
         (f"{pred:.1f} cm RMS from a plane, scan {scan:.2f} cm; 14 px patch grid in the ray map "
         f"(score {pair([r['grid']['predicted_rays_grid_score'] for r in (o, k)])})")),
        ("Cross-view drift, layering", "confirmed",
         (f"views disagree by {pair([r['layering']['disagreement_pct'] for r in (o, k)])} %; per-view offsets spread "
         f"{pair([r['layering']['view_bias_spread_pct'] for r in (o, k)])} %")),
        ("Textureless regions", "not confirmed",
         (f"flattest vs most textured tenth: {tex[0][0]:.1f} vs {tex[0][-1]:.1f} % (office), "
         f"{tex[1][0]:.1f} vs {tex[1][-1]:.1f} % (kicker)")),
        ("Reflective, transparent", "not measurable",
         f"no laser return on {pair([x['hole_pct'] for x in sf], '.0f')} % of pixels (windows, lamps, dark surfaces)"),
        ("Confidence misses errors", "partly",
         (f"Spearman {pair([x['spearman_conf_vs_error'] for x in c], '+.2f')}; the confident half keeps "
         f"{pair([100 * x['mean_error_pct_top_half_conf'] / x['mean_error_pct'] for x in c], '.0f')} % of the error")),
    ]  # fmt: skip
    fig, _ = slide("Which artifacts does Déjà View show?", "Office / kicker, all 20 views each, against ETH3D's laser "
                   "scans; each depth map scaled by its median ratio to the scan.")  # fmt: skip
    colors = {"confirmed": "#b42d2d", "partly": ORANGE, "not confirmed": BLUE, "not measurable": INK_2}
    for i, (name, verdict, evidence) in enumerate(rows):
        y = 0.74 - i * 0.11
        fig.text(0.04, y, name, fontsize=16, color=INK, weight="bold", va="center")
        fig.text(0.34, y, verdict, fontsize=15, color=colors[verdict], weight="bold", va="center")
        fig.text(0.49, y, textwrap.fill(evidence, 72), fontsize=12, color=INK_2, va="center", linespacing=1.3)
    save(fig, out, "0_summary", SOURCE)


def glass_slice(data: SceneData, view: int) -> dict:
    """Top-down 2 cm slab through the biggest scan hole of a view (a window), in metres around the hole.

    Returns the scan's points and Déjà View's points of every view where the scan has no return, along the wall
    and along the view's horizontal direction.
    """
    holes, count = ndimage.label(data.gt[view] == 0)
    sizes = ndimage.sum(np.ones_like(holes), holes, range(1, count + 1))
    hole = holes == 1 + int(np.argmax(sizes))
    frame = up_frame(data.cam_to_world)
    window_pts = data.fused[view][hole] @ frame.T
    height = float(np.median(window_pts[:, 2]))
    center = np.median(window_pts[:, :2], axis=0)
    view_dir = data.cam_to_world[view, :3, 2] @ frame[:2].T
    view_dir /= np.linalg.norm(view_dir)
    across = np.array([-view_dir[1], view_dir[0]])

    def local(p: np.ndarray) -> np.ndarray:
        q = p @ frame.T
        q = q[np.abs(q[:, 2] - height) < 0.01]  # noqa: PLR2004
        xy = q[:, :2] - center
        out = np.stack([xy @ across, xy @ view_dir], 1)
        return out[(np.abs(out[:, 0]) < 1.2) & (np.abs(out[:, 1]) < 0.6)]  # noqa: PLR2004

    holes_all = (data.gt == 0).reshape(-1)
    return {"wall": local(data.scan), "window": local(data.fused.reshape(-1, 3)[holes_all]), "height": height}


def main() -> None:
    """Measure and draw every artifact: `uv run python -m semantic_3d_refinement.artifacts --cfg <cfg>`."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=Path, required=True)
    cfg_path: Path = parser.parse_args().cfg
    raw = yaml.safe_load(cfg_path.read_text())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mv = MultiviewConfig(**raw["multiview"])
    stamp, commit = f"{cfg_path.stem}-{time.strftime('%Y%m%d-%H%M%S')}", git_hash()
    run, out = Path(raw["output_dir"]) / stamp, Path("figures") / stamp
    run.mkdir(parents=True)
    out.mkdir(parents=True)
    plt.rcParams.update({"font.size": 12, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.facecolor": SURFACE})  # fmt: skip

    data, results = {}, {}
    for name, paths in raw["scenes"].items():
        t0 = time.perf_counter()
        d = data[name] = load_data(paths, device)
        bleed, bleed_arrays = bleeding(d, raw["edge_jump"], raw["edge_max_px"])
        sample = d.scan[np.random.default_rng(0).choice(len(d.scan), min(4_000_000, len(d.scan)), replace=False)]
        grid = voxel_downsample(sample.astype(np.float64), raw["ransac_voxel_m"])
        planes = ransac_planes(grid, raw["planes"], raw["plane_tol_m"], raw["plane_min_frac"], device)
        flat, records = flatness(d, planes, raw["plane_tol_m"], raw["plane_min_px"])
        surf, _ = surfaces(d, raw["texture_window"], raw["edge_max_px"], bleed_arrays["dist"])
        conf, conf_arrays = confidence(d)
        results[name] = {
            "bleeding": bleed, "planes": flat | {"planes_found": len(planes)}, "grid": patch_grid(d),
            "layering": layering(d, paths, mv, device), "surfaces": surf, "confidence": conf,
            "arrays": {"bleeding": bleed_arrays, "planes": records, "confidence": conf_arrays},
            "seconds": time.perf_counter() - t0,
        }  # fmt: skip
        print(name, {k: v for k, v in results[name].items() if k != "arrays"}, flush=True)  # noqa: T201

    ex = raw["examples"]
    view = ex["bleeding"]["view"]
    edges = results[ex["bleeding"]["scene"]]["arrays"]["bleeding"]["edges"]
    draw_bleeding(
        out, data, results, ex["bleeding"] | {"center": edge_example(data[ex["bleeding"]["scene"]], edges, view)}
    )
    draw_planes(out, data, results)
    draw_layering(out, data, results)
    glass = glass_slice(data[ex["glass"]["scene"]], ex["glass"]["view"])
    glass |= ex["glass"]
    draw_surfaces(out, data, results, glass)
    draw_confidence(out, data, results, ex["confidence"])
    draw_summary(out, results)

    slides = sorted(out.glob("[0-9]_*.png"))
    pages = [Image.open(s).convert("RGB") for s in slides]
    pages[0].save(out / "slides.pdf", save_all=True, append_images=pages[1:], resolution=150)
    lines = [f"# {stamp}", "", f"git `{commit}`", "", "Slides (16:9, also as `slides.pdf`):", ""]
    lines += [f"![{s.stem}]({s.name})" for s in slides]
    (out / "README.md").write_text("\n".join([*lines, ""]))

    summary = {"config": raw, "git": commit,
               "scenes": {n: {k: v for k, v in r.items() if k != "arrays"} for n, r in results.items()}}  # fmt: skip
    (run / "run.yaml").write_text(yaml.safe_dump(summary, sort_keys=False, allow_unicode=True))
    print(f"wrote {run} and {out}")  # noqa: T201


if __name__ == "__main__":
    main()
