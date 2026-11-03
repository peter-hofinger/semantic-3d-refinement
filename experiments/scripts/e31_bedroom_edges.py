"""E31: rank the 20 bedroom views (Hypersim ai_006_007) by Déjà View's error at depth edges.

Purpose: pick the views of the shared visual scene that show bleeding most clearly.
Definitions as E18: a GT depth edge is a relative GT depth jump > 5% to a 4-neighbour;
error = |s D_dv - D_gt| / D_gt with s the per-view median of D_gt / D_dv; near-edge
pixels lie within 3 px of a GT depth edge, far pixels more than 10 px away; a near-edge
pixel bleeds if its error is > 5%. Ranked by the share of image pixels that bleed (what
a viewer sees), with the bleeding share of near-edge pixels and the median errors.

uv run python experiments/scripts/e31_bedroom_edges.py > experiments/results/e31_bedroom_edges.txt
"""

from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import distance_transform_edt

from src.geometry import Geometry
from src.loaders import Views
from src.loaders.scannetpp.gt import NO_MESH, GroundTruth

SCENE = Path("data/processed/hypersim/ai_006_007_cam_00")
VIEWS = "views_n20_uniform_504"
DEPTH_JUMP = 0.05  # relative depth change to a neighbour that makes a depth edge (E18)
BLEED_ERROR = 0.05  # relative depth error that counts as bleeding (E18)
BLEED_RANGE = 3  # px from a GT depth edge (E18)
FAR = 10  # px from a GT depth edge
SHOWN = (
    18,
    0,
    12,
    6,
)  # the chosen visual views: ranks 1, 3, 4 (rank 2 = view 19 repeats view 18) + room overview
FIGURE = Path("experiments/figures/e31_bedroom_edges.jpg")


def depth_edges(depth: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Mark valid pixels whose depth differs by more than DEPTH_JUMP from a valid 4-neighbour (as E18)."""
    d = np.maximum(depth, 1e-6)
    h = np.abs(d[:, 1:] - d[:, :-1]) / np.minimum(d[:, 1:], d[:, :-1])
    v = np.abs(d[1:] - d[:-1]) / np.minimum(d[1:], d[:-1])
    h = np.where(valid[:, 1:] & valid[:, :-1], h, 0) > DEPTH_JUMP
    v = np.where(valid[1:] & valid[:-1], v, 0) > DEPTH_JUMP
    edge = np.zeros(depth.shape, dtype=bool)
    edge[:, 1:] |= h
    edge[:, :-1] |= h
    edge[1:] |= v
    edge[:-1] |= v
    return edge


def main() -> None:
    """Print the ranking table and draw the top views."""
    views = Views.load(SCENE / f"{VIEWS}.npz")
    gt = GroundTruth.load(SCENE / f"gt_{VIEWS}.npz")
    dv = Geometry.load(SCENE / f"dejaview_k16_{VIEWS}.npz")
    assert dv.depth.shape == gt.depth.shape, (dv.depth.shape, gt.depth.shape)
    rows, maps = [], {}
    for i, name in enumerate(views.names):
        valid = (gt.instance_ids[i] != NO_MESH) & (gt.depth[i] > 0)
        scale = np.median(gt.depth[i][valid] / dv.depth[i][valid])
        error = np.where(
            valid,
            np.abs(scale * dv.depth[i] - gt.depth[i]) / np.maximum(gt.depth[i], 1e-6),
            np.nan,
        )
        edges = depth_edges(gt.depth[i], valid)
        dist = distance_transform_edt(~edges)
        near, far = valid & (dist <= BLEED_RANGE), valid & (dist > FAR)
        bleed = near & (error > BLEED_ERROR)
        rows.append(
            (
                bleed.sum() / valid.sum(),
                i,
                name,
                bleed.sum() / max(near.sum(), 1),
                np.nanmedian(error[near]),
                np.nanmedian(error[far]),
                near.sum() / valid.sum(),
            )
        )
        maps[i] = (error, edges, bleed)
    rows.sort(reverse=True)
    print(
        "## E31: bedroom ai_006_007, Déjà View K=16, 20 views at 504x378, ranked by bleeding pixels\n"
    )
    print(
        "| rank | view | frame | bleeding px (% of image) | bleeding share of near-edge px | "
        "median error near edges | median error far (> 10 px) | near-edge px (% of image) |"
    )
    print("|---|---|---|---|---|---|---|---|")
    for r, (img, i, name, share, e_near, e_far, n_near) in enumerate(rows, 1):
        print(
            f"| {r} | {i} | {name.split('.')[-1]} | {100 * img:.2f} | {share:.2f} | "
            f"{e_near:.3f} | {e_far:.4f} | {100 * n_near:.1f} |"
        )

    rank = {i: r for r, (_, i, *_) in enumerate(rows, 1)}
    fig, axes = plt.subplots(len(SHOWN), 3, figsize=(12, 2.9 * len(SHOWN)))
    for row, i in zip(axes, SHOWN, strict=True):
        error, edges, bleed = maps[i]
        row[0].imshow(views.images[i])
        row[0].set_title(
            f"view {i} (frame {views.names[i].split('.')[-1]}), rank {rank[i]} of 20: image",
            fontsize=9,
        )
        im = row[1].imshow(100 * error, cmap="magma", vmin=0, vmax=10)
        row[1].set_title("Déjà View depth error (%), per-view median scale", fontsize=9)
        overlay = views.images[i].copy()
        overlay[edges] = (0, 200, 255)
        overlay[bleed] = (255, 40, 40)
        row[2].imshow(overlay)
        row[2].set_title(
            "GT depth edges (cyan), bleeding px > 5% within 3 px (red)", fontsize=9
        )
        for ax in row:
            ax.axis("off")
        fig.colorbar(im, ax=row[1], fraction=0.035)
    fig.suptitle(
        "E31: bedroom ai_006_007, the four views chosen for the shared visuals (rank = by bleeding pixels)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(FIGURE, dpi=110, pil_kwargs={"quality": 88})
    print(f"\nfigure: {FIGURE}")


if __name__ == "__main__":
    main()
