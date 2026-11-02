"""E1: overlay rendered GT instances and depth on 4 views per scene, plus view spacing.

uv run python experiments/scripts/figures_gt.py
"""

from pathlib import Path

import matplotlib as mpl
import numpy as np

mpl.use("Agg")
import matplotlib.pyplot as plt

from src.loaders import Views
from src.loaders.scannetpp.gt import GroundTruth

DATA = Path("data/processed/scannetpp")
OUT = Path("experiments/figures")
SCENES = ("5ee7c22ba0", "7b6477cb95")
PICKS = (0, 6, 12, 19)

OUT.mkdir(parents=True, exist_ok=True)
palette = np.random.default_rng(0).integers(40, 255, size=(4096, 3)).astype(np.uint8)
for scene in SCENES:
    views = Views.load(DATA / scene / "views_n20_uniform_504.npz")
    gt = GroundTruth.load(DATA / scene / "gt_views_n20_uniform_504.npz")

    steps = np.linalg.norm(np.diff(views.cam_to_world[:, :3, 3], axis=0), axis=1)
    print(
        f"{scene}: camera step between views [m] min {steps.min():.2f} "
        f"median {np.median(steps):.2f} max {steps.max():.2f}"
    )
    ids = gt.instance_ids
    print(
        f"  labeled {100 * (ids >= 0).mean():.0f}%  unlabeled {100 * (ids == -1).mean():.0f}%"
        f"  no mesh {100 * (ids == -2).mean():.0f}%  instances {len(np.unique(ids[ids >= 0]))}"
        f"  depth {gt.depth[gt.depth > 0].min():.2f}-{gt.depth.max():.2f} m"
    )

    fig, ax = plt.subplots(len(PICKS), 3, figsize=(15, 3.4 * len(PICKS)))
    for r, i in enumerate(PICKS):
        img, ids_i, dep = views.images[i], gt.instance_ids[i], gt.depth[i]
        col = palette[np.maximum(ids_i, 0) % 4096].copy()
        col[ids_i == -1] = 0
        col[ids_i == -2] = (255, 0, 255)
        over = (0.5 * img + 0.5 * col).astype(np.uint8)
        edge = np.zeros_like(ids_i, dtype=bool)
        edge[:, 1:] |= ids_i[:, 1:] != ids_i[:, :-1]
        edge[1:] |= ids_i[1:] != ids_i[:-1]
        over[edge] = 255
        ax[r, 0].imshow(img)
        ax[r, 0].set_title(f"{scene} view {i} {views.names[i]}")
        ax[r, 1].imshow(over)
        ax[r, 1].set_title("instances (white edges, black unlabeled, magenta no mesh)")
        im = ax[r, 2].imshow(np.where(dep > 0, dep, np.nan), cmap="viridis")
        ax[r, 2].set_title("GT depth [m]")
        fig.colorbar(im, ax=ax[r, 2], fraction=0.03)
        for a in ax[r]:
            a.axis("off")
    fig.tight_layout()
    fig.savefig(OUT / f"e01_gt_overlay_{scene}.jpg", dpi=70, pil_kwargs={"quality": 85})
    plt.close(fig)
print("figures ->", OUT)
