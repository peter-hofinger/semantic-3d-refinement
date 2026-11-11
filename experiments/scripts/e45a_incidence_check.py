"""E45a check: the GT normals behind E25a's incidence-angle cell on KITTI-360.

E25a bins each GT pixel by |cos| of the angle between its viewing ray and the surface
normal, the normal being the cross product of the GT point map's image gradients
(`e25a_class_split.incidence`). KITTI-360's GT is projected LiDAR with holes filled
inside an instance, so the gradients run over interpolated and noisy points. Per class
over all scenes: the share of its GT pixels without a normal (a neighbour has no GT)
and in each of E25a's bins. Expected from the scene: road, sidewalk and terrain at a
grazing angle, trunks and poles facing the camera, facades along the road in between.

uv run python experiments/scripts/e45a_incidence_check.py > experiments/results/e45a_incidence_check.txt
writes `experiments/figures/e45a_incidence_check.jpg` (one town scene, four views).
"""

from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from e25a_class_split import ANGLE_BINS, class_map, incidence

from src.evaluation.labeled import ground_truth_cloud
from src.loaders.ground_truth import GroundTruth
from src.loaders.kitti360.scene import load_config
from src.loaders.views import Views

CFG = Path("configs/kitti360.yaml")
STEM = "views_n20_stride2_504"
FIGURE_SCENE = "0010_0000000984_0000001116"
FIGURE_VIEWS = (0, 6, 12, 18)
MIN_PX = 20000  # GT pixels of a class over all scenes for a row
OUT = Path("experiments/figures/e45a_incidence_check.jpg")


def main() -> None:
    """Print the per-class table and draw the figure; see the module docstring."""
    config = load_config(CFG)
    totals: dict[str, np.ndarray] = {}
    for name in config.scenes:
        folder = config.output_root / name
        views = Views.load(folder / f"{STEM}.npz")
        gt = GroundTruth.load(folder / f"gt_{STEM}.npz")
        truth = ground_truth_cloud(
            gt.depth, gt.instance_ids, gt.classes, views.intrinsics, views.cam_to_world
        )
        valid = truth.pixels
        points = np.zeros((*valid.shape, 3))
        points[valid] = truth.points
        cos = incidence(points, valid, views.cam_to_world.astype(np.float64))
        assert np.isnan(cos[~valid]).all()
        assert (
            (cos[valid] >= 0) & (cos[valid] <= 1 + 1e-9) | np.isnan(cos[valid])
        ).all()
        cls = class_map(gt.instance_ids, gt.classes)
        for c in np.unique(cls[valid]):
            v = cos[valid & (cls == c)]
            row = [
                len(v),
                np.isnan(v).sum(),
                (v <= ANGLE_BINS[0]).sum(),
                ((v > ANGLE_BINS[0]) & (v <= ANGLE_BINS[1])).sum(),
                (v > ANGLE_BINS[1]).sum(),
            ]
            assert sum(row[1:]) == row[0]
            totals[c] = totals.get(c, np.zeros(5)) + row
        if name == FIGURE_SCENE:
            fig, axes = plt.subplots(len(FIGURE_VIEWS), 2, figsize=(14, 11))
            for ax, i in zip(axes, FIGURE_VIEWS, strict=True):
                ax[0].imshow(views.images[i])
                ax[0].set_title(f"view {i}")
                im = ax[1].imshow(cos[i], vmin=0, vmax=1, cmap="viridis")
                ax[1].set_title("|cos| of the GT's incidence angle (white: no normal)")
                ax[0].axis("off")
                ax[1].axis("off")
            fig.colorbar(im, ax=axes[:, 1], shrink=0.5, label="|cos| (1 = facing)")
            fig.suptitle(
                f"KITTI-360 {name}: the road should be dark (grazing), trunks bright, facades between"
            )
            fig.savefig(OUT, dpi=110, bbox_inches="tight")
            plt.close(fig)
    lo, hi = ANGLE_BINS
    print(
        f"scenes {len(config.scenes)}, classes with >= {MIN_PX} GT px over all scenes"
    )
    print(
        f"class | GT px | no normal | |cos| <= {lo} | {lo}-{hi} | > {hi} (shares of the class's GT px)"
    )
    for c, t in sorted(totals.items(), key=lambda kv: -kv[1][0]):
        if t[0] >= MIN_PX:
            shares = " | ".join(f"{x / t[0]:.2f}" for x in t[1:])
            print(f"{c or '(unlabeled)'} | {int(t[0])} | {shares}")
    total = sum(totals.values())
    print(
        "all | "
        f"{int(total[0])} | " + " | ".join(f"{x / total[0]:.2f}" for x in total[1:])
    )


if __name__ == "__main__":
    main()
