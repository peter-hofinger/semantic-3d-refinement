"""E45a check: where Déjà View's 34% relative error on KITTI-360 comes from.

Added after the first read of E45a's results (not in the plan). The protocol aligns
Déjà View's points onto the GT by one least-squares Sim(3) on the pixels (Umeyama),
which weights a pixel by its squared error in metres. KITTI-360's GT reaches beyond
300 m; Hypersim's rooms end within metres.

Per scene: the error e and relative error r = e / GT depth under the scene's Sim(3);
the same with one Sim(3) per view (cameras taken out); r of the depth alone, each
view's depth scaled by the median of GT depth / Déjà View depth (rays and cameras
taken out); Déjà View's focal length (median over views of the pinhole fit) against
the GT's; and the median of aligned Déjà View depth / GT depth in bins of GT depth
(1 = right, the same value in every bin = one scale fits the scene).
Over all scenes, per bin: share of the GT pixels, share of the summed error, mean e, mean r.

uv run python experiments/scripts/e45a_alignment_check.py > experiments/results/e45a_alignment_check.txt
"""

from pathlib import Path

import numpy as np
import torch

from src.evaluation.labeled import align_output, ground_truth_cloud, umeyama
from src.geometry import Geometry
from src.loaders.ground_truth import GroundTruth
from src.loaders.kitti360.scene import load_config
from src.loaders.views import Views
from src.refinement.multiview import points_from_rays, scene_from_geometry

CFG = Path("configs/kitti360.yaml")
STEM = "views_n20_stride2_504"
BINS = (5, 10, 20, 40, 80)  # m of GT depth
MIN_PX = 100  # GT pixels of a scene in a bin for its ratio to be printed


def main() -> None:
    """Print the per-scene table and the per-bin totals; see the module docstring."""
    config = load_config(CFG)
    labels = [f"{a}-{b} m" for a, b in zip((0, *BINS), (*BINS, "inf"), strict=True)]
    totals = np.zeros((len(labels), 3))
    print(
        "scene | group | scene Sim(3): e m, r % | per-view Sim(3): e m, r % | depth alone r % | "
        f"focal px, Déjà View / GT | aligned depth / GT depth, median in {' | '.join(labels)}"
    )
    for name, group in config.scenes.items():
        folder = config.output_root / name
        views = Views.load(folder / f"{STEM}.npz")
        gt = GroundTruth.load(folder / f"gt_{STEM}.npz")
        geometry = Geometry.load(folder / f"dejaview_k16_{STEM}.npz")
        scene = scene_from_geometry(geometry)
        points = points_from_rays(
            torch.from_numpy(geometry.depth).double(), scene.rays.double()
        ).numpy()
        truth = ground_truth_cloud(
            gt.depth, gt.instance_ids, gt.classes, views.intrinsics, views.cam_to_world
        )
        valid = truth.pixels
        pred, scale, _ = align_output(truth, points)
        d = gt.depth[valid].astype(np.float64)
        e = np.linalg.norm(pred - truth.points, axis=1)
        assert np.isfinite(e).all()
        gt_points = np.zeros((*valid.shape, 3))
        gt_points[valid] = truth.points
        e_view, r_depth = [], []
        for i in range(len(valid)):
            m = valid[i]
            s, rotation, t = umeyama(points[i][m], gt_points[i][m])
            e_view.append(
                np.linalg.norm(
                    s * points[i][m] @ rotation.T + t - gt_points[i][m], axis=1
                )
            )
            z, z_gt = geometry.depth[i][m].astype(np.float64), gt.depth[i][m]
            r_depth.append(np.abs(np.median(z_gt / z) * z - z_gt) / z_gt)
        e_view, r_depth = np.concatenate(e_view), np.concatenate(r_depth)
        ratio = scale * geometry.depth[valid] / d
        which = np.digitize(d, BINS)
        cells = []
        for b in range(len(labels)):
            m = which == b
            totals[b] += [m.sum(), e[m].sum(), (e[m] / d[m]).sum()]
            cells.append(f"{np.median(ratio[m]):.2f}" if m.sum() >= MIN_PX else "-")
        focal = float(np.median(scene.k[:, 0, 0].numpy()))
        print(
            f"{name} | {group} | {e.mean():.2f}, {100 * (e / d).mean():.1f} | "
            f"{e_view.mean():.2f}, {100 * (e_view / d).mean():.1f} | {100 * r_depth.mean():.1f} | "
            f"{focal:.0f} / {views.intrinsics[0, 0, 0]:.0f} | {' | '.join(cells)}"
        )
    print("\nall scenes: GT depth | px share | error share | mean e m | mean r %")
    for label, (px, e_sum, r_sum) in zip(labels, totals, strict=True):
        print(
            f"{label} | {px / totals[:, 0].sum():.3f} | {e_sum / totals[:, 1].sum():.3f} | "
            f"{e_sum / px:.2f} | {100 * r_sum / px:.1f}"
        )


if __name__ == "__main__":
    main()
