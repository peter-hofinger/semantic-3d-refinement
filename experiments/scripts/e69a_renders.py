"""E69a: the bedroom from views 18, 0, 12, 6 as GT, Déjà View and any generator clouds, in three looks.

Draws the same columns three times: `<stem>_gray.jpg` (gray under a headlight, normals from each cloud's own
neighbours), `<stem>_rgb.jpg` (the image colour of the point) and `<stem>_ids.jpg` (the instance id, one colour per
id, lightly shaded; no id: gray). Generated points have no colour or id of their own: each takes those of the
nearest Déjà View pixel, in Déjà View's frame, as `cloud_eval` does for labels; Déjà View's points have their own
pixel's, and the GT column has the GT's own instance ids (Hypersim's), so its ids differ from the other
columns' (SAM 3's linked masks). Every cloud is capped at 1 M random points; a splat is half the mean distance to the
4 nearest neighbours, at most 3 times the cloud's median.

uv run python experiments/scripts/e69a_renders.py <stem> runs/e69a/<rung>_eval.yaml \
    "seeded=e69a-overfit1-seeded-s0" "oracle=e69a-oracle-seeded"
"""

import sys
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from scipy.spatial import cKDTree

from src.analysis.cloud_eval import apply_alignment
from src.analysis.render import headlight, splat
from src.analysis.scene_eval import load_inputs
from src.analysis.scene_figures import (
    PRESENTATION_SCENE,
    PRESENTATION_VIEWS,
    render_cameras,
)
from src.evaluation.labeled import align_output
from src.refinement.multiview import points_from_rays

CAP = 1_000_000
NEIGHBORS = 8
OUT = Path("experiments/figures")
LOOKS = ("gray", "rgb", "ids")


def id_colors(ids: np.ndarray) -> np.ndarray:
    """One colour per instance id (N, 3) in [0, 1]; negative ids: mid-gray."""
    palette = np.random.default_rng(7).uniform(0.15, 0.95, size=(int(ids.max()) + 2, 3))
    return np.where((ids < 0)[:, None], 0.5, palette[np.maximum(ids, 0)])


def prepare(
    points: np.ndarray,
    colors: np.ndarray,
    ids: np.ndarray,
    rng: np.random.Generator,
    cap: int = CAP,
) -> dict[str, np.ndarray]:
    """Cap the cloud at `cap` points; return points, normals, splat radii and the colours of each look."""
    keep = (
        rng.choice(len(points), cap, replace=False)
        if len(points) > cap
        else slice(None)
    )
    points, colors, ids = points[keep], colors[keep], ids[keep]
    dist, index = cKDTree(points, balanced_tree=False, compact_nodes=False).query(
        points, NEIGHBORS + 1, workers=-1
    )
    offsets = points[index[:, 1:]] - points[:, None]
    normals = np.linalg.eigh(np.einsum("nki,nkj->nij", offsets, offsets))[1][:, :, 0]
    radius = 0.5 * dist[:, 1:5].mean(1)
    return {
        "points": points,
        "normals": normals,
        "radius": np.minimum(radius, 3 * np.median(radius)),
        "rgb": colors,
        "ids": id_colors(ids),
    }


def main() -> None:
    """Draw the three figures."""
    stem, config = sys.argv[1], Path(sys.argv[2])
    pairs = [a.rsplit("=", 1) for a in sys.argv[3:]]
    raw = yaml.safe_load(config.read_text())
    inputs = load_inputs(raw, PRESENTATION_SCENE, torch.device("cpu"))
    pixels = points_from_rays(
        inputs.scene.depth.double(), inputs.scene.rays.double()
    ).numpy()
    flat = pixels.reshape(-1, 3)
    align = align_output(inputs.truth, pixels)[2]
    images = inputs.views.images.reshape(-1, 3) / 255.0
    ids = inputs.labels.instance_ids.reshape(-1).astype(np.int64)
    tree = cKDTree(flat, balanced_tree=False, compact_nodes=False)
    rng = np.random.default_rng(0)
    truth = inputs.truth
    pixel_colors = (
        inputs.views.images[truth.pixels] / 255.0
    )  # the GT's pixels carry the images' colours
    clouds = {
        "ground truth": prepare(truth.points, pixel_colors, truth.instance, rng),
        "Déjà View": prepare(apply_alignment(flat, align), images, ids, rng),
    }
    folder = Path(raw["root"]) / PRESENTATION_SCENE
    for title, name in pairs:
        points = np.load(folder / f"{name}_{raw['views']}_cloud.npz")["points"]
        near = tree.query(points.astype(np.float64), workers=-1)[1]
        clouds[f"{title} ({len(points):,})"] = prepare(
            apply_alignment(points.astype(np.float64), align),
            images[near],
            ids[near],
            rng,
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cameras = {
        v: c
        for v, c in render_cameras(inputs, PRESENTATION_VIEWS).items()
        if v != "dollhouse"
    }
    for look in LOOKS:
        fig, axes = plt.subplots(
            len(cameras),
            len(clouds),
            figsize=(3.6 * len(clouds), 3.0 * len(cameras)),
            layout="constrained",
        )
        for r, (row, (view, (camera, eye))) in enumerate(
            zip(axes, cameras.items(), strict=True)
        ):
            for ax, (title, c) in zip(row, clouds.items(), strict=True):
                xyz = torch.from_numpy(c["points"]).float().to(device)
                shade = headlight(
                    torch.from_numpy(c["normals"]).float().to(device), xyz, eye
                )
                if look == "gray":
                    color = shade
                elif look == "rgb":
                    color = torch.from_numpy(c["rgb"]).float().to(device)
                else:
                    color = torch.from_numpy(c["ids"]).float().to(device) * (
                        0.35 + 0.65 * shade
                    )
                radii = torch.from_numpy(c["radius"]).float().to(device)
                ax.imshow(splat(xyz, color, radii, camera))
                ax.set_xticks([])
                ax.set_yticks([])
                if r == 0:
                    ax.set_title(title, fontsize=9)
            row[0].set_ylabel(view.split(",")[0], fontsize=10)
        fig.suptitle(
            f"Bedroom ai_006_007, {look}; generated points take the colour and id of the nearest Déjà View pixel",
            x=0.01,
            ha="left",
            fontsize=10,
        )
        fig.savefig(OUT / f"{stem}_{look}.jpg", dpi=70)
        plt.close(fig)


if __name__ == "__main__":
    main()
