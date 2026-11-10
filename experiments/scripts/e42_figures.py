"""E42: the bedroom (views 18, 0, 12, 6) refined with BiNI's E_n and with the angle term.

Columns: image; the angle between DSINE's normals and the closer-neighbor normals of
Déjà View's depth, of the depth refined with BiNI's E_n and of the depth refined with
E_a (λ_a = 4e-4), on one 0-60° scale; and each pixel's distance to the nearest GT point
(the protocol's accuracy, after the variant's own Sim(3)), E_a minus BiNI on ±2 cm.
Prints the angle quantiles and the mean distance change over all 20 views.

uv run python experiments/scripts/e42_figures.py runs/<e42_angle run> \
    > experiments/results/e42_figures.txt
"""

import sys
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from matplotlib.colors import Normalize
from scipy.spatial import cKDTree

from src.analysis.scene_eval import load_inputs
from src.analysis.scene_figures import (
    DIVERGING,
    PRESENTATION_SCENE,
    PRESENTATION_VIEWS,
    SEQUENTIAL,
)
from src.refinement.normals import closer_neighbor_normals

sys.path.insert(0, str(Path(__file__).parent))
from e40_figures import distances, variant_points

VARIANTS = {"Déjà View": "Déjà View", "BiNI E_n": "Refined", "angle E_a": "angle 4e-4"}
ANGLE_MAX = 60.0
CHANGE_CM = 2.0
OUT = Path("experiments/figures/e42_bedroom.jpg")


def angles(depth: np.ndarray, rays: torch.Tensor, dsine: torch.Tensor) -> np.ndarray:
    """Angle (deg) between DSINE's normals and the closer-neighbor normals of depth."""
    points = torch.from_numpy(depth).double()[..., None] * rays[..., 3:]
    n = closer_neighbor_normals(points)
    cos = (n * dsine).sum(-1)
    sin = torch.linalg.cross(n, dsine, dim=-1).norm(dim=-1)
    return torch.rad2deg(torch.atan2(sin, cos)).numpy()


def main() -> None:
    """Draw the figure and print the numbers."""
    run_dir = Path(sys.argv[1])
    run = yaml.safe_load((run_dir / "run.yaml").read_text(encoding="utf-8"))
    inputs = load_inputs(run["config"], PRESENTATION_SCENE, torch.device("cpu"))
    names = [v["name"] for v in run["config"]["variants"]]
    arrays = np.load(run_dir / "results.npz")
    rays, dsine = inputs.scene.rays.double(), inputs.normals.double()
    angle = {
        k: angles(arrays[f"{PRESENTATION_SCENE}/{names.index(v)}"], rays, dsine)
        for k, v in VARIANTS.items()
    }
    tree = cKDTree(inputs.truth.points)
    dist = {
        k: distances(tree, variant_points(run_dir, v, inputs))
        for k, v in VARIANTS.items()
        if k != "Déjà View"
    }
    change = dist["angle E_a"] - dist["BiNI E_n"]
    print(
        f"{PRESENTATION_SCENE}, all 20 views: angle to DSINE (deg) median / 90% / > 30°"
    )
    for k, a in angle.items():
        q = np.quantile(a, [0.5, 0.9])
        print(f"{k}: {q[0]:.1f} / {q[1]:.1f} / {(a > 30).mean():.3f}")
    print(
        f"distance to GT, E_a minus BiNI: mean {change.mean():+.4f} cm, "
        f"pixels changed by > 0.5 cm: {(np.abs(change) > 0.5).mean():.4f}"
    )
    titles = ["image"] + [f"angle to DSINE, {k}" for k in angle]
    titles += ["distance to GT,\nE_a minus BiNI (cm)"]
    views = PRESENTATION_VIEWS
    fig, axes = plt.subplots(
        len(views),
        len(titles),
        figsize=(2.3 * len(titles), 1.9 * len(views) + 1.2),
        layout="constrained",
    )
    for row, v in zip(axes, views, strict=True):
        row[0].imshow(inputs.views.images[v])
        for ax, a in zip(row[1:4], angle.values(), strict=True):
            im = ax.imshow(a[v], cmap=SEQUENTIAL, norm=Normalize(0, ANGLE_MAX))
        ch = row[4].imshow(
            change[v], cmap=DIVERGING, norm=Normalize(-CHANGE_CM, CHANGE_CM)
        )
        for ax in row:
            ax.set_xticks([])
            ax.set_yticks([])
        row[0].set_ylabel(f"view {v}")
    for ax, title in zip(axes[0], titles, strict=True):
        ax.set_title(title, fontsize=8)
    fig.colorbar(
        im, ax=axes[-1, 1:4], location="bottom", shrink=0.6, label="deg, capped at 60"
    )
    fig.colorbar(ch, ax=axes[-1, 4], location="bottom", label="cm, blue: E_a closer")
    fig.suptitle(
        "Bedroom ai_006_007: how far each depth's normals are from DSINE's, "
        "and where E_a moves points against BiNI's E_n",
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
