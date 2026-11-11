"""E45b check: why the protocol's Sim(3) fails on Tanks and Temples, and what replaces it.

Added after `e45a_alignment_check.py --dataset tanksandtemples` showed Déjà View's aligned
depth at 0.54 (Truck) and 0.25 (Caterpillar) of the GT's in every range bin, while each
view's depth alone, scaled by its median GT / Déjà View depth, is right to 7-9%.

Per scene:
1. Pixels whose depth, after the view's median scale, is off by more than OFF of the GT
   depth: their share of the GT pixels, the share where Déjà View is too far, and where
   they lie: within BAND px of a GT depth edge (plane rule) or of the GT's border (a
   pixel without GT, where the scan ends).
2. One Umeyama Sim(3) for the scene, fitted on (a) all GT pixels, the protocol, and (b)
   the interior: GT pixels farther than BAND px from an edge and from the border. For
   each: its scale, the mean error e and relative error r = e / GT depth over all GT
   pixels, the median r, and the mean r over the interior.

uv run python experiments/scripts/e45b_alignment.py > experiments/results/e45b_alignment.txt
"""

from pathlib import Path

import numpy as np
import torch
from scipy import ndimage

from src.analysis.edges import depth_edges
from src.evaluation.labeled import ground_truth_cloud, umeyama
from src.geometry import Geometry
from src.loaders.ground_truth import GroundTruth
from src.loaders.tanksandtemples.scene import load_config
from src.loaders.views import Views
from src.refinement.multiview import points_from_rays, scene_from_geometry

CFG = Path("configs/tanksandtemples.yaml")
STEM = "views_n20_uniform_504"
BAND = 3  # px, as E25b's band
OFF = (0.1, 0.5, 2.0)  # relative depth errors that count as off


def main() -> None:
    """Print both parts for every scene; see the module docstring."""
    config = load_config(CFG)
    for name, group in config.scenes.items():
        folder = config.output_root / name
        views = Views.load(folder / f"{STEM}.npz")
        gt = GroundTruth.load(folder / f"gt_{STEM}.npz")
        geometry = Geometry.load(folder / f"dejaview_k16_{STEM}.npz")
        points = points_from_rays(
            torch.from_numpy(geometry.depth).double(),
            scene_from_geometry(geometry).rays.double(),
        ).numpy()
        truth = ground_truth_cloud(
            gt.depth, gt.instance_ids, gt.classes, views.intrinsics, views.cam_to_world
        )
        valid = truth.pixels
        edge, border, scaled = [], [], []
        for i in range(len(valid)):
            edges = depth_edges(gt.depth[i].astype(np.float64), valid[i])
            near = ndimage.distance_transform_edt(~edges) <= BAND
            edge.append(
                near[valid[i]] if edges.any() else np.zeros(valid[i].sum(), bool)
            )
            border.append((ndimage.distance_transform_edt(valid[i]) <= BAND)[valid[i]])
            ratio = gt.depth[i][valid[i]] / geometry.depth[i][valid[i]]
            scaled.append(np.median(ratio) / ratio)  # scaled Déjà View depth / GT depth
        edge, border = np.concatenate(edge), np.concatenate(border)
        scaled = np.concatenate(scaled)
        print(
            f"\n{name} ({group}): GT pixels within {BAND} px of an edge {edge.mean():.3f},"
            f" of the border {border.mean():.3f}, of either {(edge | border).mean():.3f}"
        )
        print(
            "  depth off by more than | share of GT px | Déjà View too far | of them"
            " at an edge | at the border | at either"
        )
        for off in OFF:
            out = np.abs(scaled - 1) > off
            print(
                f"  {off:.0%} | {out.mean():.4f} | {(out & (scaled > 1)).mean():.4f} |"
                f" {edge[out].mean():.2f} | {border[out].mean():.2f} |"
                f" {(edge | border)[out].mean():.2f}"
            )
        pred, d = points[valid], gt.depth[valid].astype(np.float64)
        interior = ~(edge | border)
        print(
            "  Sim(3) fitted on | scale | mean e cm | mean r % | median r % |"
            " mean r % over the interior"
        )
        for label, use in (
            ("all GT px", np.ones(len(d), bool)),
            ("interior", interior),
        ):
            s, rotation, t = umeyama(pred[use], truth.points[use])
            e = np.linalg.norm(s * pred @ rotation.T + t - truth.points, axis=1)
            assert np.isfinite(e).all()
            r = 100 * e / d
            print(
                f"  {label} | {s:.3f} | {100 * e.mean():.1f} | {r.mean():.1f} |"
                f" {np.median(r):.1f} | {r[interior].mean():.1f}"
            )


if __name__ == "__main__":
    main()
