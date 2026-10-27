"""Accuracy of refined monocular depth against a laser scan (ETH3D, `eth3d.py`).

Per view, the scan z-buffered into the calibrated camera gives a ground-truth depth D_gt wherever a scan point lands.
Each depth map is aligned by one scale, s = median(D_gt / D) over those pixels (monocular depth is known up to scale),
lifted through the calibrated camera, X = cam_to_world · s D K⁻¹ (u + ½, v + ½, 1), and scored by each point's distance
to its nearest scan point: accuracy as in ETH3D's benchmark (Schöps et al. 2017), the fraction within τ, plus the mean
and median. Completeness is not measured, since three views see only part of the scan.

The refinement never sees the calibration: its rays are the pinhole fitted to Déjà View's (`rays: pinhole`), along
which D is z-depth. Lifting through the true camera keeps any focal error of Déjà View in the score, the same for
every variant.
"""

import argparse
import dataclasses
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from scipy.spatial import cKDTree
from torch import Tensor

from semantic_3d_refinement.normals import BiniConfig, predicted_normals
from semantic_3d_refinement.refine import RefineConfig, run_ablation
from semantic_3d_refinement.views import git_hash, view_from_config


def scan_depth(points: Tensor, k: Tensor, cam_to_world: Tensor, h: int, w: int) -> Tensor:
    """Depth of the nearest scan point per pixel (a z-buffer), 0 where none lands; pixel centers at +0.5."""
    assert points.ndim == 2 and points.shape[1] == 3, points.shape  # noqa: PLR2004
    world_to_cam, k = torch.linalg.inv(cam_to_world.double()).to(points), k.to(points)
    p = points @ world_to_cam[:3, :3].T + world_to_cam[:3, 3]
    p = p[p[:, 2] > 0]
    uv = (p[:, :2] / p[:, 2:]) @ k[:2, :2].T + k[:2, 2]
    col, row = uv[:, 0].floor().long(), uv[:, 1].floor().long()
    inside = (col >= 0) & (col < w) & (row >= 0) & (row < h)
    depth = torch.full((h * w,), torch.inf, dtype=p.dtype, device=p.device)
    depth = depth.scatter_reduce(0, (row * w + col)[inside], p[inside, 2], reduce="amin")
    depth = torch.where(depth.isinf(), 0, depth).reshape(h, w)
    assert depth.isfinite().all() and (depth >= 0).all()
    return depth


def lift(depth: np.ndarray, k: np.ndarray, cam_to_world: np.ndarray) -> np.ndarray:
    """World points of a z-depth map through a pinhole, (H, W, 3)."""
    h, w = depth.shape
    v, u = np.mgrid[0:h, 0:w] + 0.5
    rays = np.stack([u, v, np.ones_like(u)], axis=-1) @ np.linalg.inv(k).T
    return (depth[..., None] * rays) @ cam_to_world[:3, :3].T + cam_to_world[:3, 3]


def accuracy(dist: np.ndarray, tolerances_cm: list[int | float]) -> dict[str, float]:
    """Mean and median distance in cm, and the fraction of points within each tolerance."""
    assert dist.ndim == 1 and dist.size > 0 and np.isfinite(dist).all(), dist.shape
    cm = 100 * dist
    return {"mean_cm": float(cm.mean()), "median_cm": float(np.median(cm))} | {
        f"acc_{t:g}cm": float((cm <= t).mean()) for t in tolerances_cm
    }


def score(
    depth: np.ndarray, gt_depth: np.ndarray, k: np.ndarray, cam_to_world: np.ndarray, tree: cKDTree
) -> tuple[np.ndarray, float]:
    """Distance to the scan per pixel (NaN where the scan saw nothing) after median scaling; and the scale."""
    valid = gt_depth > 0
    assert valid.mean() > 0.1, valid.mean()  # noqa: PLR2004
    scale = float(np.median(gt_depth[valid] / depth[valid]))
    dist = np.full(depth.shape, np.nan)
    dist[valid] = tree.query(lift(scale * depth, k, cam_to_world)[valid], workers=-1)[0]
    return dist, scale


def evaluate_view(
    raw: dict, cfg: RefineConfig, bini_cfg: BiniConfig, scan: tuple[cKDTree, Tensor]
) -> tuple[dict, dict[str, np.ndarray]]:
    """Refine one view under every variant and score each against the scan; return the record and its arrays."""
    tree, points = scan
    view, cams = raw["view"], np.load(raw["gt_views"])
    data = {key: v.to(points.device) for key, v in view_from_config(raw).items()}
    data["normals"], info = predicted_normals(raw, data)
    variants = run_ablation(data, cfg, raw["variants"], bini_cfg)
    k, c2w = cams["intrinsics"][view].astype(np.float64), cams["cam_to_world"][view].astype(np.float64)
    gt = scan_depth(points, torch.from_numpy(k).to(points), torch.from_numpy(c2w).to(points), *data["depth"].shape)
    gt = gt.cpu().numpy().astype(np.float64)
    fitted = np.load(raw["input"])["intrinsics"][view]
    record = {
        "view": view,
        "image": str(cams["names"][view]),
        "coverage": float((gt > 0).mean()),
        "focal_ratio": float(fitted[0, 0] / k[0, 0]),
        **info,
        "variants": [],
    }
    arrays = {"gt_depth": gt, "image": cams["images"][view]}
    for i, v in enumerate(variants):
        dist, scale = score(v.depth.astype(np.float64), gt, k, c2w, tree)
        arrays |= {f"{i}_depth": v.depth, f"{i}_dist": dist.astype(np.float32)}
        record["variants"].append({
            "label": v.label, "evaluations": v.evaluations, "seconds": v.seconds, "scale": scale,
            **v.metrics, **accuracy(dist[np.isfinite(dist)], raw["tolerances_cm"]),
        })  # fmt: skip
    return record, arrays


def main() -> None:
    """Refine and score every configured view: `uv run python -m semantic_3d_refinement.evaluate --cfg <cfg>`."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=Path, required=True)
    cfg_path: Path = parser.parse_args().cfg
    raw = yaml.safe_load(cfg_path.read_text())
    cfg = RefineConfig(**raw["refine"])
    device = torch.device(cfg.device if cfg.device != "cuda" or torch.cuda.is_available() else "cpu")
    cfg = dataclasses.replace(cfg, device=device.type)
    bini_cfg = BiniConfig(**{**raw["bini"], "device": device.type})

    out = Path(raw["output_dir"]) / f"{cfg_path.stem}-{time.strftime('%Y%m%d-%H%M%S')}"
    out.mkdir(parents=True)
    views, arrays = [], {}
    for scene, paths in raw["scenes"].items():
        t0 = time.perf_counter()
        scan = np.load(paths["scan"])
        tree, scan_gpu = cKDTree(scan), torch.from_numpy(scan).to(device)
        tree_seconds = time.perf_counter() - t0
        for view in raw["views"]:
            record, view_arrays = evaluate_view({**raw, **paths, "view": view}, cfg, bini_cfg, (tree, scan_gpu))
            record["scan_tree_seconds"] = tree_seconds
            views.append({"scene": scene, **record})
            arrays |= {f"{scene}_{view}_{k}": v for k, v in view_arrays.items()}
            print(f"{scene} {view}: " + ", ".join(f"{r['label']} {r['median_cm']:.2f}" for r in record["variants"]))  # noqa: T201

    summary = {"config": raw, "resolved": dataclasses.asdict(cfg), "git": git_hash(), "views": views}
    (out / "run.yaml").write_text(yaml.safe_dump(summary, sort_keys=False, allow_unicode=True))
    np.savez_compressed(out / "results.npz", **arrays)
    print(f"wrote {out}")  # noqa: T201


if __name__ == "__main__":
    main()
