"""E42: DSINE's normals against normals reconstructed from Déjà View's depth (bedroom).

For views 18, 0, 12, 6 of ai_006_007: the image, DSINE's normals, the normals of
Déjà View's input depth by `closer_neighbor_normals` (Peter's proposal: per axis the
closer of the two neighbors), both in the camera frame and coloured alike (red: right,
green: up, blue: toward the camera), and the angle between the two per pixel, the
cost an angle-based E_n would put there. Prints the angle's quantiles per view.

uv run python experiments/scripts/e42_normals_figure.py \
    > experiments/results/e42_normals_figure.txt
"""

from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from matplotlib.colors import Normalize

from src.analysis.scene_eval import load_inputs
from src.analysis.scene_figures import (
    PRESENTATION_SCENE,
    PRESENTATION_VIEWS,
    SEQUENTIAL,
)
from src.refinement.multiview import points_from_rays
from src.refinement.normals import closer_neighbor_normals

ANGLE_MAX = 60.0  # degrees, top of the colour scale
OUT = Path("experiments/figures/e42_normals_ai_006_007.jpg")


def colours(n: np.ndarray) -> np.ndarray:
    """RGB of camera-frame normals (x right, y down, z forward): right, up, toward."""
    return np.clip((n * np.array([1, -1, -1]) + 1) / 2, 0, 1)


def main() -> None:
    """Draw the figure and print the angle quantiles."""
    raw = yaml.safe_load(Path("configs/eval_hypersim.yaml").read_text())
    inputs = load_inputs(raw, PRESENTATION_SCENE, torch.device("cpu"))
    scene = inputs.scene
    rot = scene.rotation.double()
    world = points_from_rays(scene.depth.double(), scene.rays.double())
    cam = torch.einsum(
        "vhwi,vij->vhwj", world - scene.center.double()[:, None, None], rot
    )
    recon = closer_neighbor_normals(cam)
    dsine = torch.einsum("vhwi,vij->vhwj", inputs.normals.double(), rot)
    angle = torch.rad2deg(
        torch.atan2(
            torch.linalg.cross(recon, dsine, dim=-1).norm(dim=-1),
            (recon * dsine).sum(-1),
        )
    ).numpy()
    print(f"{PRESENTATION_SCENE}: angle between DSINE and reconstructed normals (deg)")
    print("view | median | 75% | 90% | share > 30 deg")
    for v in (*PRESENTATION_VIEWS, "all"):
        a = angle if v == "all" else angle[v]
        q = np.quantile(a, [0.5, 0.75, 0.9])
        print(f"{v} | {q[0]:.1f} | {q[1]:.1f} | {q[2]:.1f} | {(a > 30).mean():.3f}")
    titles = [
        "image",
        "DSINE normals",
        "reconstructed from Déjà View depth\n(closer neighbor per axis)",
        "angle between them (deg)",
    ]
    views = PRESENTATION_VIEWS
    fig, axes = plt.subplots(
        len(views), 4, figsize=(10.5, 1.95 * len(views) + 1.0), layout="constrained"
    )
    for row, v in zip(axes, views, strict=True):
        row[0].imshow(inputs.views.images[v])
        row[1].imshow(colours(dsine[v].numpy()))
        row[2].imshow(colours(recon[v].numpy()))
        im = row[3].imshow(angle[v], cmap=SEQUENTIAL, norm=Normalize(0, ANGLE_MAX))
        for ax in row:
            ax.set_xticks([])
            ax.set_yticks([])
        row[0].set_ylabel(f"view {v}")
    for ax, title in zip(axes[0], titles, strict=True):
        ax.set_title(title, fontsize=8)
    fig.colorbar(
        im, ax=axes[-1, 3], location="bottom", label=f"deg, capped at {ANGLE_MAX:g}"
    )
    fig.suptitle(
        "Bedroom ai_006_007: DSINE vs normals of Déjà View's depth, one colouring "
        "(red right, green up, blue toward the camera)",
        x=0.01,
        ha="left",
        fontsize=9,
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=110)
    plt.close(fig)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
