"""Figures for a `scene_eval` run: scores, energy terms, cross sections, views, classes.

`uv run python -m src.analysis.scene_figures --run runs/<run>` writes per scene with a
record each energy term per pixel before and after refinement, a horizontal cross section
of the fused clouds with a 1 m close-up, the views with ground-truth and SAM 3 objects
next to each pixel's distance to the nearest ground-truth point, the fused clouds from
above, surface normals, the scores of every class, and renders of every cloud from the
same cameras (`render_figure`). The shared presentation scene goes to
`figures/<run>/presentation/`, always with the views Mircea chose (issue #7, E31). Every
other scene, and per split the headline scores, go to `figures/<run>/debug/`.
"""

import argparse
import colorsys
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from matplotlib.colors import LinearSegmentedColormap, Normalize
from scipy.spatial import cKDTree

from src.analysis.render import Camera, headlight, intrinsics, look_at, splat
from src.analysis.scene_eval import HEADLINE, SceneInputs, load_inputs
from src.evaluation.labeled import NONE, lift
from src.refinement.multiview import (
    RefineConfig,
    build_problem,
    consistency,
    energy_maps,
    points_from_rays,
)

LABELS = {
    "accuracy_cm": "accuracy (cm) ↓",
    "completeness_cm": "completeness (cm) ↓",
    "chamfer_cm": "Chamfer (cm) ↓",
    "miou": "mIoU ↑",
    "pq": "PQ ↑",
    "ap": "AP ↑",
}
SEQUENTIAL = LinearSegmentedColormap.from_list(
    "seq_blue",
    ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"],
)
DIVERGING = LinearSegmentedColormap.from_list(
    "div", ["#104281", "#6da7ec", "#f0efec", "#ec8a89", "#b42d2d"]
)
SCENE_COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#8a5cc2")
ERROR_MAX_CM = 30.0
PRESENTATION_SCENE = "ai_006_007_cam_00"  # the bedroom, issue #7
PRESENTATION_VIEWS = (18, 0, 12, 6)  # frames 94, 0, 62, 31: most bleeding, then layout
DEBUG_VIEWS = (0, 10)  # two, to keep these figures small in git
TOP_POINTS = 150_000  # per panel of the clouds from above
SLAB = (0.05, 1.5)  # m above the floor shown from above: furniture, no floor or ceiling
SECTION = (0.75, 0.77)  # m above the floor: the cross section, table height
ZOOM = 1.0  # m, side of the cross section's close-up
TERMS = ("E_d", "E_n", "E_mv")
RENDER_SIZE = (480, 640)  # pixels, height x width
RENDER_FOV = 70.0  # degrees, horizontal
PULL_BACK = (0.5, 0.2)  # m back along the camera axis and up: the interior camera
CEILING_CUT = (
    0.3  # m below the ceiling: higher points are left out of the dollhouse view
)


def instance_colors(ids: np.ndarray) -> np.ndarray:
    """RGB in [0, 1] per id, spread by the golden ratio; gray for none."""
    unique, inverse = np.unique(ids, return_inverse=True)
    hues = (np.abs(unique) * 0.618034) % 1
    lut = np.array([colorsys.hsv_to_rgb(h, 0.65, 0.92) for h in hues])
    lut[unique < 0] = 0.85
    return lut[inverse.reshape(ids.shape)]


def shown(scene: str) -> tuple[str, tuple[int, ...]]:
    """Return the figure folder of a scene and the views its figures show."""
    if scene == PRESENTATION_SCENE:
        return "presentation", PRESENTATION_VIEWS
    return "debug", DEBUG_VIEWS


def boundary_overlay(inputs: SceneInputs, view: int) -> np.ndarray:
    """Return one view's image in [0, 1] with its SAM 3 boundaries in red."""
    ids = inputs.labels.instance_ids[view]
    edges = np.zeros(ids.shape, dtype=bool)
    edges[:, 1:] |= ids[:, 1:] != ids[:, :-1]
    edges[1:] |= ids[1:] != ids[:-1]
    overlay = inputs.views.images[view].astype(float) / 255
    overlay[edges] = (1.0, 0.2, 0.2)
    return overlay


def pixel_points(
    inputs: SceneInputs, depth: np.ndarray, align: np.ndarray
) -> np.ndarray:
    """Per-pixel points (V, H, W, 3) of a variant, registered into the truth's frame."""
    points = points_from_rays(
        torch.from_numpy(depth).double(), inputs.scene.rays.cpu().double()
    ).numpy()
    return points @ align[:3, :3].T + align[:3, 3]


def ground_truth_points(inputs: SceneInputs) -> np.ndarray:
    """Per-pixel ground-truth points (V, H, W, 3) in the ground truth's frame."""
    return np.stack(
        [
            lift(d.astype(np.float64), k.astype(np.float64), c.astype(np.float64))
            for d, k, c in zip(
                inputs.gt.depth,
                inputs.views.intrinsics,
                inputs.views.cam_to_world,
                strict=True,
            )
        ]
    )


def summary_figure(run: dict, split: str, out: Path) -> None:
    """Scores of a split: one panel per score, variants as rows, scenes as bars.

    The split's aggregate, if it has one, is the last bar.
    """
    report = run["splits"][split]
    bars = {s: r["variants"] for s, r in report.get("scenes", {}).items()}
    if "aggregate" in report:
        bars[f"{split} aggregate"] = report["aggregate"]
    variants = [v["name"] for v in run["config"]["variants"]]
    fig, axes = plt.subplots(
        1,
        len(HEADLINE),
        figsize=(3.6 * len(HEADLINE), 0.42 * len(variants) + 1.5),
        layout="constrained",
        squeeze=False,
    )
    height = 0.8 / len(bars)
    for ax, key in zip(axes.flat, HEADLINE, strict=True):
        for j, (bar, record) in enumerate(bars.items()):
            values = [record[v]["headline"][key] for v in variants]
            y = np.arange(len(variants)) + (j - (len(bars) - 1) / 2) * height
            ax.barh(y, values, height, color=SCENE_COLORS[j], label=bar)
            ax.axvline(values[0], color=SCENE_COLORS[j], lw=0.8, ls="--")
        ax.set_yticks(np.arange(len(variants)), variants if ax in axes[:, 0] else [])
        ax.invert_yaxis()
        ax.set_title(LABELS[key])
        ax.grid(axis="y", visible=False)
    fig.legend(
        *axes[0, 0].get_legend_handles_labels(),
        loc="outside lower center",
        ncols=len(bars),
    )
    fig.suptitle(
        f"{split}: every variant on every scene; dashed: Déjà View's value per scene",
        x=0.01,
        ha="left",
    )
    fig.savefig(out / f"summary_{split}.png", dpi=130)
    plt.close(fig)


def views_figure(
    inputs: SceneInputs,
    points: dict[str, np.ndarray],
    views: tuple[int, ...],
    name: str,
    out: Path,
) -> None:
    """Draw views: image, GT and SAM 3 objects, distance to the nearest GT point."""
    tree = cKDTree(inputs.truth.points)
    errors = {}
    for k, d in points.items():
        finite = np.isfinite(d).all(-1)
        e = np.full(finite.shape, np.nan)
        cap = ERROR_MAX_CM / 100
        dist = tree.query(d[finite], workers=-1, distance_upper_bound=cap)[0]
        e[finite] = 100 * np.minimum(dist, cap)
        errors[k] = e
    first, *_, last = errors
    titles = ["image", "ground-truth objects", "SAM 3 objects (linked)"]
    titles += [f"{k}\ndistance (cm)" for k in errors] + [f"{last}\nminus {first}"]
    fig, axes = plt.subplots(
        len(views),
        len(titles),
        figsize=(2.3 * len(titles), 1.9 * len(views) + 1.2),
        layout="constrained",
    )
    error_norm = Normalize(0, ERROR_MAX_CM)
    change_norm = Normalize(-10, 10)
    for row, v in zip(axes, views, strict=True):
        gt_ids = np.where(
            inputs.gt.instance_ids[v] >= 0, inputs.gt.instance_ids[v], NONE
        )
        panels = [
            inputs.views.images[v],
            instance_colors(gt_ids),
            instance_colors(inputs.labels.instance_ids[v]),
        ]
        for ax, image in zip(row[:3], panels, strict=True):
            ax.imshow(image)
        for ax, e in zip(row[3:-1], errors.values(), strict=True):
            im = ax.imshow(e[v], cmap=SEQUENTIAL, norm=error_norm)
        ch = row[-1].imshow(
            errors[last][v] - errors[first][v], cmap=DIVERGING, norm=change_norm
        )
        for ax in row:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(visible=False)
        row[0].set_ylabel(f"view {v}")
    for ax, title in zip(axes[0], titles, strict=True):
        ax.set_title(title, fontsize=8)
    fig.colorbar(
        im,
        ax=axes[-1, 3:-1],
        location="bottom",
        shrink=0.7,
        label=f"distance (cm), capped at {ERROR_MAX_CM:g}",
    )
    fig.colorbar(ch, ax=axes[-1, -1], location="bottom", label="cm, blue: closer")
    fig.suptitle(
        f"{name}: distance of each pixel's point to the nearest ground-truth point",
        x=0.01,
        ha="left",
    )
    fig.savefig(out / f"{name}_views.png", dpi=110)
    plt.close(fig)


def topdown_figure(
    inputs: SceneInputs,
    points: dict[str, np.ndarray],
    name: str,
    out: Path,
) -> None:
    """Fused clouds from above (x-y, z up), colored by object and by distance."""
    truth = inputs.truth
    rng = np.random.default_rng(0)
    lo, hi = np.percentile(truth.points[:, :2], [1, 99], axis=0)
    margin = 0.05 * (hi - lo)
    floor = np.percentile(truth.points[:, 2], 2)

    def slab(points: np.ndarray) -> np.ndarray:
        height = points[:, 2] - floor
        return (height > SLAB[0]) & (height < SLAB[1])

    keep = slab(truth.points)
    columns = [("ground truth", truth.points[keep], truth.instance[keep], None)]
    tree = cKDTree(truth.points)
    for k, d in points.items():
        finite = np.isfinite(d).all(-1)
        cloud, ids = d[finite], inputs.labels.instance_ids[finite]
        keep = slab(cloud)
        dist = tree.query(cloud[keep], workers=-1, distance_upper_bound=1.0)[0]
        columns.append((k, cloud[keep], ids[keep], np.minimum(dist, 1.0)))
    fig, axes = plt.subplots(
        2, len(columns), figsize=(4.2 * len(columns), 8.6), layout="constrained"
    )
    for col, (title, shown, ids, dist) in zip(axes.T, columns, strict=True):
        pick = rng.choice(len(shown), min(TOP_POINTS, len(shown)), replace=False)
        p = shown[pick]
        col[0].scatter(p[:, 0], p[:, 1], s=1.0, c=instance_colors(ids[pick]), lw=0)
        col[0].set_title(f"{title}: objects", fontsize=9)
        if dist is not None:
            col[1].scatter(
                p[:, 0], p[:, 1], s=1.0, c=100 * dist[pick], cmap=SEQUENTIAL,
                norm=Normalize(0, ERROR_MAX_CM), lw=0,
            )  # fmt: skip
            col[1].set_title(f"{title}: distance to ground truth", fontsize=9)
        else:
            col[1].scatter(p[:, 0], p[:, 1], s=1.0, c=p[:, 2], cmap="Greys", lw=0)
            col[1].set_title("ground truth: height", fontsize=9)
        for ax in col:
            ax.set_xlim(lo[0] - margin[0], hi[0] + margin[0])
            ax.set_ylim(lo[1] - margin[1], hi[1] + margin[1])
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(visible=False)
    fig.suptitle(
        f"{name}: all 20 views fused, seen from above, points "
        f"{100 * SLAB[0]:g} to {100 * SLAB[1]:g} cm above the floor; distance in cm, "
        f"0 (light) to {ERROR_MAX_CM:g} (dark)",
        x=0.01,
        ha="left",
    )
    fig.savefig(out / f"{name}_topdown.png", dpi=110)
    plt.close(fig)


def unit_normals(points: np.ndarray) -> np.ndarray:
    """Return a point map's unit normals (..., H, W, 3), central differences inside.

    The image border takes one-sided differences: with a zero difference there, the
    border rows and columns had no normal and rendered as dark lines wherever the
    views agreed well enough for them to win the depth test.
    """
    du = np.gradient(points, axis=-2)
    dv = np.gradient(points, axis=-3)
    n = np.cross(du, dv)
    return n / np.linalg.norm(n, axis=-1, keepdims=True).clip(1e-12)


def point_normals(points: np.ndarray) -> np.ndarray:
    """Return a point map's normals (central differences) as RGB (H, W, 3) in [0, 1]."""
    return 0.5 * (1 - unit_normals(points))


def normals_figure(
    inputs: SceneInputs,
    points: dict[str, np.ndarray],
    views: tuple[int, ...],
    name: str,
    out: Path,
) -> None:
    """Draw each view's normals per variant, SAM 3 boundaries, ground truth, a zoom."""
    truth = ground_truth_points(inputs)
    h, w = truth.shape[1:3]
    zoom = (slice(h // 4, 3 * h // 4), slice(w // 4, 3 * w // 4))
    titles = ["image + SAM 3 boundaries", *points, "ground truth"]
    fig, axes = plt.subplots(
        2 * len(views),
        len(titles),
        figsize=(3.0 * len(titles), 2.4 * 2 * len(views)),
        layout="constrained",
    )
    for k, v in enumerate(views):
        panels = [
            boundary_overlay(inputs, v),
            *(point_normals(d[v]) for d in points.values()),
            point_normals(truth[v]),
        ]
        for col, image in zip(axes[2 * k : 2 * k + 2].T, panels, strict=True):
            col[0].imshow(image)
            col[1].imshow(image[zoom])
            for ax in col:
                ax.set_xticks([])
                ax.set_yticks([])
                ax.grid(visible=False)
        axes[2 * k, 0].set_ylabel(f"view {v}")
        axes[2 * k + 1, 0].set_ylabel("center, 2x")
    for ax, title in zip(axes[0], titles, strict=True):
        ax.set_title(title, fontsize=8)
    fig.suptitle(
        f"{name}: surface normals of each variant's points (central differences; "
        "color = direction); red: SAM 3 boundaries",
        x=0.01,
        ha="left",
        fontsize=9,
    )
    fig.savefig(out / f"{name}_normals.png", dpi=120)
    plt.close(fig)


def render_cameras(
    inputs: SceneInputs, views: tuple[int, ...]
) -> dict[str, tuple[Camera, np.ndarray]]:
    """Return the cameras every cloud is rendered from: name -> (camera, eye).

    All are fixed by the ground truth alone (z up). Dollhouse: from above at 60°
    elevation, on the side the input cameras stand, far enough to see the whole room.
    Interior, one per view: its camera moved back and up by `PULL_BACK`, at most half the
    free space behind it, so depth errors show as parallax and the view stays one a
    person took.
    """
    truth = inputs.truth.points
    centers = inputs.views.cam_to_world[:, :3, 3].astype(np.float64)
    forward = inputs.views.cam_to_world[:, :3, 2].astype(np.float64).mean(0)
    floor = np.percentile(truth[:, 2], 2)
    middle = np.append(np.median(truth[:, :2], 0), floor + 1.0)
    side = centers.mean(0)[:2] - middle[:2]
    if np.linalg.norm(side) < 0.1:  # noqa: PLR2004
        side = -forward[:2]
    side /= np.linalg.norm(side)
    reach = np.percentile(np.linalg.norm(truth[:, :2] - middle[:2], axis=1), 95)
    distance = reach / np.tan(np.radians(RENDER_FOV) / 2)
    elevation = np.radians(60)
    above = middle + distance * np.append(np.cos(elevation) * side, np.sin(elevation))
    up = np.array([0.0, 0, 1])
    k = intrinsics(RENDER_SIZE[1], RENDER_SIZE[0], RENDER_FOV)
    cameras = {"dollhouse": (Camera(look_at(above, middle, up), k, RENDER_SIZE), above)}
    for v in views:
        pose = inputs.views.cam_to_world[v].astype(np.float64)
        origin, axis = pose[:3, 3], pose[:3, 2]
        back = (origin - truth) @ axis  # distance behind the camera along its axis
        aside = np.linalg.norm(origin - truth + back[:, None] * axis, axis=1)
        blocked = back[(back > 0) & (aside < 0.3)]  # noqa: PLR2004
        free = blocked.min() if len(blocked) else np.inf
        eye = origin + min(PULL_BACK[0], 0.5 * free) * -axis + PULL_BACK[1] * up
        camera = Camera(look_at(eye, origin + axis, up), k, RENDER_SIZE)
        cameras[f"view {v}, pulled back"] = (camera, eye)
    return cameras


def render_figure(
    inputs: SceneInputs,
    points: dict[str, np.ndarray | tuple[np.ndarray, np.ndarray]],
    views: tuple[int, ...],
    name: str,
    out: Path,
) -> None:
    """Render every cloud from the same cameras, in image colors and shaded.

    All pixels of all views, the ground truth's from its own depth; each point covers
    its source pixel's patch of surface (`render.splat`). The dollhouse view leaves out
    the ceiling. A cloud given as (points, keep) draws only the pixels kept, shaded by
    the normals of all its points: a removed neighbor does not blank a kept pixel.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    truth = ground_truth_points(inputs)
    clouds = {"ground truth": (truth, inputs.gt.depth > 0)}
    clouds |= {
        k: v if isinstance(v, tuple) else (v, np.isfinite(v).all(-1))
        for k, v in points.items()
    }
    centers = inputs.views.cam_to_world[:, None, None, :3, 3]
    focal = inputs.views.intrinsics[:, None, None, 0, 0]
    ceiling = np.percentile(inputs.truth.points[:, 2], 98) - CEILING_CUT
    cameras = render_cameras(inputs, views)
    fig, axes = plt.subplots(
        2 * len(cameras),
        len(clouds),
        figsize=(5.3 * len(clouds), 4.1 * 2 * len(cameras)),
        layout="constrained",
    )
    for col, (title, (p, valid)) in zip(axes.T, clouds.items(), strict=True):
        radii = 0.5 * np.linalg.norm(p - centers, axis=-1) / focal
        normals = unit_normals(p)

        def tensor(a: np.ndarray, keep: np.ndarray) -> torch.Tensor:
            return torch.from_numpy(np.ascontiguousarray(a[keep])).float().to(device)

        for row, (view, (camera, eye)) in enumerate(cameras.items()):
            keep = valid & ((p[..., 2] < ceiling) | (view != "dollhouse"))
            xyz = tensor(p, keep)
            colors = tensor(inputs.views.images / 255, keep)
            shade = headlight(tensor(normals, keep), xyz, eye)
            for i, (mode, c) in enumerate((("colors", colors), ("shaded", shade))):
                ax = col[2 * row + i]
                ax.imshow(splat(xyz, c, tensor(radii, keep), camera))
                ax.set_title(f"{title}: {view}, {mode}", fontsize=9)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.grid(visible=False)
    fig.suptitle(
        f"{name}: every cloud rendered from the same cameras (all 20 views' pixels; "
        "shaded: gray, lit from the camera)",
        x=0.01,
        ha="left",
    )
    fig.savefig(out / f"{name}_render.png", dpi=100)
    plt.close(fig)


def energy_figure(
    inputs: SceneInputs,
    depth: np.ndarray,
    cfg: RefineConfig,
    views: tuple[int, ...],
    out: Path,
) -> None:
    """Each energy term per pixel of each view, on the input and on the refined depth."""
    name = inputs.views.scene_id
    segments = torch.from_numpy(inputs.labels.instance_ids.astype(np.int64))
    p = build_problem(inputs.scene, cfg, inputs.normals, segments)
    rows = {
        "Déjà View": energy_maps(p.d0, cfg, p),
        "Refined": energy_maps(p.scale / torch.from_numpy(depth), cfg, p),
    }
    gate = consistency(inputs.scene, p.d0, p.scale, segments)
    high = {
        key: max(float(np.percentile(m[key].numpy(), 99.5)) for m in rows.values())
        for key in TERMS
    }
    fig, axes = plt.subplots(
        2 * len(views),
        2 + len(TERMS),
        figsize=(2.9 * (2 + len(TERMS)), 2.7 * 2 * len(views)),
        layout="constrained",
    )
    for k, v in enumerate(views):
        overlay = boundary_overlay(inputs, v)
        for row, (title, maps) in zip(
            axes[2 * k : 2 * k + 2], rows.items(), strict=True
        ):
            row[0].imshow(overlay)
            row[0].set_ylabel(f"view {v}, {title}")
            u = row[1].imshow(gate[v], cmap=SEQUENTIAL, vmin=0, vmax=1)
            for ax, key in zip(row[2:], TERMS, strict=True):
                im = ax.imshow(
                    np.log10(maps[key][v].numpy() + 1e-12), cmap=SEQUENTIAL,
                    vmin=np.log10(high[key]) - 4, vmax=np.log10(high[key]),
                )  # fmt: skip
                ax.set_title(f"{key}, mean {maps[key].mean().item():.3g}", fontsize=8)
            for ax in row:
                ax.set_xticks([])
                ax.set_yticks([])
                ax.grid(visible=False)
    axes[0, 0].set_title("image + SAM 3 boundaries", fontsize=8)
    axes[0, 1].set_title("consistency u (E_d's gate)", fontsize=8)
    size = {"location": "bottom", "fraction": 0.3 / fig.get_figheight(), "pad": 0.01}
    fig.colorbar(u, ax=axes[:, 1], shrink=0.8, **size)
    fig.colorbar(
        im, ax=axes[:, 2:], shrink=0.5, **size,
        label="log10 weighted cost per pixel, 4 decades below each term's 99.5th "
        "percentile",
    )  # fmt: skip
    fig.suptitle(
        f"{name}: where each energy term sits, per view on Déjà View's depth, then on the "
        "refined depth; means over all views",
        x=0.01,
        ha="left",
        fontsize=9,
    )
    fig.savefig(out / f"{name}_energy.png", dpi=110)
    plt.close(fig)


def section_figure(
    inputs: SceneInputs, points: dict[str, np.ndarray], name: str, out: Path
) -> None:
    """Fused clouds in a 2 cm slice at table height, colored by view, and a close-up."""
    truth = ground_truth_points(inputs)[inputs.gt.depth > 0]
    floor = np.percentile(truth[:, 2], 2)

    def band(x: np.ndarray) -> np.ndarray:
        height = x[:, 2] - floor
        return (height > SECTION[0]) & (height < SECTION[1])

    gt = truth[band(truth)]
    wall = gt[gt[:, 0] > np.percentile(gt[:, 0], 90)]
    center = np.median(wall[:, :2], axis=0)  # on the wall with the largest x
    lo, hi = np.percentile(truth[:, :2], [1, 99], axis=0)
    view = np.broadcast_to(
        np.arange(len(inputs.gt.depth))[:, None, None], inputs.gt.depth.shape
    )
    colors = plt.get_cmap("tab20")(np.arange(20))
    columns = {"ground truth": None, **points}
    fig, axes = plt.subplots(
        2, len(columns), figsize=(4.6 * len(columns), 8.4), layout="constrained"
    )
    for col, (title, pts) in zip(axes.T, columns.items(), strict=True):
        for ax, (a, b), size in zip(
            col, ((lo, hi), (center - ZOOM / 2, center + ZOOM / 2)), (0.3, 2.0),
            strict=True,
        ):  # fmt: skip
            ax.scatter(*gt[:, :2].T, s=size, c="0.5", lw=0)
            if pts is not None:
                seen = np.isfinite(pts).all(-1)
                flat, ids = pts[seen], view[seen]
                keep = band(flat)
                ax.scatter(*flat[keep, :2].T, s=size, c=colors[ids[keep] % 20], lw=0)
            ax.set_xlim(a[0], b[0])
            ax.set_ylim(a[1], b[1])
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(visible=False)
        col[0].set_title(title, fontsize=9)
        col[1].set_title(f"{title}: {ZOOM:g} m close-up", fontsize=9)
    fig.suptitle(
        f"{name}: {100 * (SECTION[1] - SECTION[0]):g} cm slice "
        f"{100 * SECTION[0]:g} cm above the floor; gray ground truth, colors = view",
        x=0.01,
        ha="left",
        fontsize=9,
    )
    fig.savefig(out / f"{name}_section.png", dpi=110)
    plt.close(fig)


def class_figure(record: dict, names: list[str], name: str, out: Path) -> None:
    """Accuracy, completeness, IoU, PQ and AP of every class in the scene, per variant.

    record: each variant's debug tier, which holds the per-class scores.
    """
    classes = sorted(
        record[names[0]]["surface_per_class"],
        key=lambda c: record[names[0]]["surface_per_class"][c]["accuracy_cm"],
    )
    fig, axes = plt.subplots(
        5, 1, figsize=(max(6.0, 1 + 0.45 * len(classes)), 13.0), sharex=True
    )
    width = 0.8 / len(names)
    for ax, (label, value) in zip(
        axes,
        (
            ("accuracy (cm)", lambda r, c: r["surface_per_class"][c]["accuracy_cm"]),
            (
                "completeness (cm)",
                lambda r, c: r["surface_per_class"][c]["completeness_cm"],
            ),
            ("IoU", lambda r, c: r["iou_per_class"].get(c, 0)),
            ("PQ", lambda r, c: r["pq_per_class"].get(c, {}).get("pq", 0)),
            ("AP", lambda r, c: r["ap_per_class"].get(c, {}).get("ap", 0)),
        ),
        strict=True,
    ):
        for j, n in enumerate(names):
            x = np.arange(len(classes)) + (j - (len(names) - 1) / 2) * width
            ax.bar(x, [value(record[n], c) for c in classes], width,
                   color=SCENE_COLORS[j], label=n)  # fmt: skip
        ax.set_ylabel(label)
    for ax in axes[2:]:
        ax.set_ylim(0, 1)
    axes[-1].set_xticks(np.arange(len(classes)), classes, rotation=60, ha="right")
    axes[0].legend(loc="upper right")
    axes[0].set_title(f"{name}: per class", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / f"{name}_classes.png", dpi=120)
    plt.close(fig)


def main() -> None:
    """Draw the figures of one run; see the module docstring."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument(
        "--show",
        nargs="+",
        default=["Déjà View", "Refined"],
        help="variants in the per-scene figures (first is the reference)",
    )
    args = parser.parse_args()
    run = yaml.safe_load((args.run / "run.yaml").read_text())
    out = Path("figures") / args.run.name
    (out / "debug").mkdir(parents=True, exist_ok=True)
    for split in run["splits"]:
        summary_figure(run, split, out / "debug")
    arrays = np.load(args.run / "results.npz")
    variants = run["config"]["variants"]
    names = [v["name"] for v in variants]
    device = torch.device("cpu")
    cfg = RefineConfig(**run["refine"]["refine"] | {"device": "cpu"})
    refined = names.index("Refined")
    records = {
        scene: record
        for report in run["splits"].values()
        for scene, record in report.get("scenes", {}).items()
    }
    for scene, record in records.items():
        folder, views = shown(scene)
        folder = out / folder
        folder.mkdir(exist_ok=True)
        inputs = load_inputs(run["config"], scene, device)
        assert np.array_equal(arrays[f"{scene}/labels"], inputs.labels.instance_ids)

        def registered(
            n: str, scene: str = scene, inputs: SceneInputs = inputs
        ) -> np.ndarray:
            i = names.index(n)
            return pixel_points(
                inputs, arrays[f"{scene}/{i}"], arrays[f"{scene}/{i}/align"]
            )

        points = {n: registered(n) for n in args.show}
        energy_figure(inputs, arrays[f"{scene}/{refined}"], cfg, views, folder)
        section_figure(inputs, points, scene, folder)
        views_figure(inputs, points, views, scene, folder)
        topdown_figure(inputs, points, scene, folder)
        normals_figure(inputs, points, views, scene, folder)
        render_figure(inputs, points, views, scene, folder)
        debug = {n: r["debug"] for n, r in record["variants"].items()}
        class_figure(debug, args.show, scene, folder)
        print(scene, flush=True)  # noqa: T201
    print(f"wrote {out}")  # noqa: T201


if __name__ == "__main__":
    main()
