"""Accuracy of a scene's fused cloud against a laser scan, and the surface it forms.

All views' points are mapped into the scan's frame by one similarity, fitted on the
camera centers (Umeyama 1991): Déjà View predicts its own cameras, up to scale. Accuracy
is the distance of each sampled pixel's point to the nearest scan point; fidelity is
measured on a 5 mm voxel grid, so point density does not change it: thickness (RMS
distance of the 24 nearest grid points to their plane; layering and noise) and the
angle of that plane to the scan's. Disagreement is the median depth gap between
overlapping views (`multiview.residuals`, on matches fixed from the first geometry),
which needs no ground truth.

`uv run eval-geometry <views npz> <geometry npz>... (--gt <gt npz> | --scan <npy>)`
scores pipeline outputs, e.g. `cloud_backbone.npz` and `cloud_geometry.npz`.
"""

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import yaml
from scipy.spatial import cKDTree

from src.evaluation.depth import accuracy, lift, scan_depth
from src.geometry import Geometry
from src.loaders.scannetpp.gt import GroundTruth
from src.loaders.views import Views
from src.refinement.monocular import points_from_rays
from src.refinement.multiview import (
    Matches,
    MultiviewConfig,
    Scene,
    find_matches,
    residuals,
    scene_from_geometry,
)

VOXEL = 0.005  # grid of the fidelity measures, m
PLANE_K = 24  # grid points per local plane
SAMPLES = 200_000  # scored pixels and grid points


def similarity(a: np.ndarray, b: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Scale, rotation, translation minimizing Σ ‖s R a + t - b‖² (Umeyama 1991)."""
    assert a.shape == b.shape and a.shape[1] == 3 and len(a) >= 3, (a.shape, b.shape)  # noqa: PLR2004
    ma, mb = a.mean(0), b.mean(0)
    u, sv, vt = np.linalg.svd((b - mb).T @ (a - ma) / len(a))
    sign = np.diag([1.0, 1.0, np.sign(np.linalg.det(u @ vt))])
    rotation = u @ sign @ vt
    scale = float((sv * np.diag(sign)).sum() / ((a - ma) ** 2).sum(1).mean())
    return scale, rotation, mb - scale * rotation @ ma


def voxel_downsample(points: np.ndarray, size: float) -> np.ndarray:
    """One point per occupied voxel of the given size, the mean of its points."""
    keys = np.floor(points / size).astype(np.int64)
    _, inverse, counts = np.unique(
        keys, axis=0, return_inverse=True, return_counts=True
    )
    out = np.zeros((len(counts), 3))
    np.add.at(out, inverse.ravel(), points)
    return out / counts[:, None]


def local_planes(
    tree: cKDTree, queries: np.ndarray, k: int
) -> tuple[np.ndarray, np.ndarray]:
    """PCA of each query's k nearest points: unit normal, RMS distance to the plane."""
    _, idx = tree.query(queries, k=k, workers=-1)
    nb = tree.data[idx]
    nb = nb - nb.mean(1, keepdims=True)
    values, vectors = np.linalg.eigh(np.einsum("nki,nkj->nij", nb, nb) / k)
    return vectors[..., 0], np.sqrt(values[:, 0].clip(0))


@dataclass(frozen=True)
class Truth:
    """A scan, its cameras and per-view depth, the similarity from Déjà View's frame.

    `sample` holds the flat indices of 200k pixels (all views) the scan saw, drawn once
    and scored for every variant.
    """

    tree: cKDTree
    grid: cKDTree
    k: np.ndarray
    cam_to_world: np.ndarray
    depth: np.ndarray
    align: tuple[float, np.ndarray, np.ndarray]
    sample: np.ndarray


def load_truth(paths: dict, scene: Scene) -> Truth:
    """Truth for a config's scene: `gt_views` cameras and a `scan` cloud or `gt` depth.

    ETH3D gives its scan as points (`scan`); ScanNet++ as a mesh, rendered into the
    views as depth by `scannetpp-render-gt` (`gt`) and lifted here.
    """
    cams = np.load(paths["gt_views"])
    k, c2w = cams["intrinsics"], cams["cam_to_world"]
    if "gt" in paths:
        scan = scan_from_depth(GroundTruth.load(Path(paths["gt"])).depth, k, c2w)
    else:
        scan = np.load(paths["scan"])
    return truth_for(scan, k, c2w, scene)


def scan_from_depth(
    depth: np.ndarray, intrinsics: np.ndarray, cam_to_world: np.ndarray
) -> np.ndarray:
    """World points of every pixel with ground-truth depth over all views, (N, 3)."""
    assert depth.ndim == 3 and len(depth) == len(intrinsics) == len(cam_to_world), (  # noqa: PLR2004
        depth.shape
    )
    points = [
        lift(d, k, c)[d > 0]
        for d, k, c in zip(
            depth.astype(np.float64),
            intrinsics.astype(np.float64),
            cam_to_world.astype(np.float64),
            strict=True,
        )
    ]
    return np.concatenate(points).astype(np.float32)


def truth_for(
    scan: np.ndarray, intrinsics: np.ndarray, cam_to_world: np.ndarray, scene: Scene
) -> Truth:
    """Scan, its 5 mm grid, cameras and per-view depth; Déjà View aligned to them."""
    scan_gpu = torch.from_numpy(scan).to(scene.depth.device)
    h, w = scene.depth.shape[1:]
    depth = np.stack(
        [
            scan_depth(
                scan_gpu,
                torch.from_numpy(k).to(scan_gpu),
                torch.from_numpy(c).to(scan_gpu),
                h,
                w,
            )
            .cpu()
            .numpy()
            for k, c in zip(intrinsics, cam_to_world, strict=True)
        ]
    )
    centers = cam_to_world[:, :3, 3].astype(np.float64)
    align = similarity(scene.center.cpu().numpy().astype(np.float64), centers)
    grid = cKDTree(voxel_downsample(scan.astype(np.float64), VOXEL))
    k, c2w = intrinsics.astype(np.float64), cam_to_world.astype(np.float64)
    seen = np.flatnonzero(depth > 0)
    sample = np.sort(
        np.random.default_rng(0).choice(seen, min(SAMPLES, len(seen)), replace=False)
    )
    return Truth(cKDTree(scan), grid, k, c2w, depth.astype(np.float64), align, sample)


def fused_metrics(
    depth: np.ndarray, rays: np.ndarray, truth: Truth, tolerances_cm: list[int | float]
) -> dict:
    """Accuracy of all views' points where the scan saw them, and the surface fidelity.

    The cloud is mapped by the camera similarity, the same for every variant. Accuracy
    on the fixed pixel sample (`Truth.sample`): exact distances, but 200k of about 3M,
    since a nearest-neighbor query for points far from the scan costs ~40 µs each (2 min
    per variant for all); the fractions' standard error is under 0.12 points. Fidelity
    on a 5 mm voxel grid of all points, so point density does not change it: at 200k
    sampled grid points, the RMS distance of the 24 nearest grid points to their plane
    (thickness: layering and noise, no ground truth needed), and the angle between that
    plane and the scan's.
    """
    s, rotation, t = truth.align
    points = (
        s
        * points_from_rays(torch.from_numpy(depth), torch.from_numpy(rays)).numpy()
        @ rotation.T
        + t
    )
    dist = truth.tree.query(points.reshape(-1, 3)[truth.sample], workers=-1)[0]
    out = {f"fused_{k}": v for k, v in accuracy(dist, tolerances_cm).items()}
    points = points[truth.depth > 0]
    grid = voxel_downsample(points, VOXEL)
    queries = grid[
        np.random.default_rng(0).choice(
            len(grid), min(SAMPLES, len(grid)), replace=False
        )
    ]
    normals, thickness = local_planes(cKDTree(grid), queries, PLANE_K)
    scan_normals, _ = local_planes(
        truth.grid, truth.grid.data[truth.grid.query(queries, workers=-1)[1]], PLANE_K
    )
    angle = np.degrees(np.arccos(np.abs((normals * scan_normals).sum(-1)).clip(0, 1)))
    return out | {
        "thickness_mm": float(1000 * np.median(thickness)),
        "normal_angle_deg": float(np.median(angle)),
    }


def scan_fidelity(truth: Truth) -> dict[str, float]:
    """Return the scan's own thickness on its grid, the floor of `thickness_mm`."""
    queries = truth.grid.data[
        np.random.default_rng(0).choice(
            truth.grid.n, min(SAMPLES, truth.grid.n), replace=False
        )
    ]
    return {
        "thickness_mm": float(
            1000 * np.median(local_planes(truth.grid, queries, PLANE_K)[1])
        )
    }


def disagreement_pct(depth: torch.Tensor, scene: Scene, matches: Matches) -> float:
    """Median |log depth gap| between overlapping views, in percent."""
    scale = scene.depth.median()
    return 100 * residuals(scale / depth, scale, scene, matches).abs().median().item()


def main() -> None:
    """Score geometries of one scene against its scan; print a Markdown table."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("views", type=Path, help="prepared views (.npz), GT cameras")
    parser.add_argument("geometry", type=Path, nargs="+", help="Geometry .npz files")
    truth_source = parser.add_mutually_exclusive_group(required=True)
    truth_source.add_argument("--gt", type=Path, help="scannetpp-render-gt output")
    truth_source.add_argument("--scan", type=Path, help="scan points (.npy), ETH3D")
    parser.add_argument("--tolerances-cm", type=float, nargs="+", default=[1, 2, 5])
    parser.add_argument("--out", type=Path, help="also write the scores as YAML")
    args = parser.parse_args()

    views = Views.load(args.views)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scenes = [scene_from_geometry(Geometry.load(p)).to(device) for p in args.geometry]
    if args.gt is not None:
        scan = scan_from_depth(
            GroundTruth.load(args.gt).depth, views.intrinsics, views.cam_to_world
        )
    else:
        scan = np.load(args.scan)
    truth = truth_for(scan, views.intrinsics, views.cam_to_world, scenes[0])
    first = scenes[0]
    mv = MultiviewConfig()
    scale = first.depth.median()
    matches = find_matches(first, scale / first.depth, scale, mv.mv_tol, mv.neighbors)

    scores = {}
    for path, scene in zip(args.geometry, scenes, strict=True):
        depth = scene.depth.cpu().numpy().astype(np.float64)
        rays = scene.rays.cpu().numpy().astype(np.float64)
        scores[path.name] = {
            **fused_metrics(depth, rays, truth, args.tolerances_cm),
            "disagreement_pct": disagreement_pct(scene.depth, first, matches),
        }
    keys = ("fused_mean_cm", "fused_median_cm", "thickness_mm", "disagreement_pct")
    print("| geometry | mean (cm) | median (cm) | thickness (mm) | disagreement (%) |")  # noqa: T201
    print("|---|---|---|---|---|")  # noqa: T201
    for name, s in scores.items():
        print(f"| {name} | " + " | ".join(f"{s[k]:.3f}" for k in keys) + " |")  # noqa: T201
    if args.out is not None:
        args.out.write_text(yaml.safe_dump(scores, sort_keys=False))


if __name__ == "__main__":
    main()
