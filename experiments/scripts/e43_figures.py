"""E43: the bedroom (views 18, 0, 12, 6) refined with BiNI, the angle term and tangents.

Columns: image; DSINE's normals; the closer-neighbor normals of the depth refined with
BiNI's E_n, with E42's angle term (λ_a 4e-4, E42's run) and with E43's tangent term
(λ_t 4e-4), one colouring (red right, green up, blue toward the camera); and each
pixel's distance to the nearest GT point, tangent term minus BiNI, on ±2 cm. Prints
the angle to DSINE (median / 90 % / share > 30°) and the mean distance change.

uv run python experiments/scripts/e43_figures.py runs/<e42_angle run> runs/<e43_tangent run> \
    > experiments/results/e43_figures.txt
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
from src.analysis.scene_figures import DIVERGING, PRESENTATION_SCENE, PRESENTATION_VIEWS
from src.refinement.normals import closer_neighbor_normals

sys.path.insert(0, str(Path(__file__).parent))
from e40_figures import distances, variant_points
from e42_normals_after import to_camera
from e42_normals_figure import colours

CHANGE_CM = 2.0
OUT = Path("experiments/figures/e43_bedroom.jpg")


def refined_normals(run_dir: Path, variant: str, rays: torch.Tensor) -> torch.Tensor:
    """Closer-neighbor normals (V, H, W, 3), rays' frame, of one variant's bedroom depth."""
    run = yaml.safe_load((run_dir / "run.yaml").read_text(encoding="utf-8"))
    names = [v["name"] for v in run["config"]["variants"]]
    depth = np.load(run_dir / "results.npz")[
        f"{PRESENTATION_SCENE}/{names.index(variant)}"
    ]
    return closer_neighbor_normals(
        torch.from_numpy(depth).double()[..., None] * rays[..., 3:]
    )


def main() -> None:
    """Draw the figure and print the numbers."""
    e42, e43 = Path(sys.argv[1]), Path(sys.argv[2])
    raw = yaml.safe_load((e43 / "run.yaml").read_text(encoding="utf-8"))["config"]
    inputs = load_inputs(raw, PRESENTATION_SCENE, torch.device("cpu"))
    rays, rot = inputs.scene.rays.double(), inputs.scene.rotation.double()
    dsine = inputs.normals.double()
    normals = {
        "BiNI E_n": refined_normals(e43, "Refined", rays),
        "angle E_a (E42)": refined_normals(e42, "angle 4e-4", rays),
        "tangents E_t (E43)": refined_normals(e43, "tangent 4e-4", rays),
    }
    print(
        f"{PRESENTATION_SCENE}, all 20 views: angle to DSINE (deg) median / 90% / > 30°"
    )
    for k, n in normals.items():
        cos = (n * dsine).sum(-1)
        sin = torch.linalg.cross(n, dsine, dim=-1).norm(dim=-1)
        a = torch.rad2deg(torch.atan2(sin, cos)).numpy()
        q = np.quantile(a, [0.5, 0.9])
        print(f"{k}: {q[0]:.1f} / {q[1]:.1f} / {(a > 30).mean():.3f}")
    tree = cKDTree(inputs.truth.points)
    change = distances(tree, variant_points(e43, "tangent 4e-4", inputs)) - distances(
        tree, variant_points(e43, "Refined", inputs)
    )
    print(
        f"distance to GT, E_t minus BiNI: mean {change.mean():+.4f} cm, "
        f"pixels changed by > 0.5 cm: {(np.abs(change) > 0.5).mean():.4f}"
    )
    maps = {"DSINE": to_camera(dsine, rot)}
    maps |= {k: to_camera(n, rot) for k, n in normals.items()}
    titles = ["image", *maps, "distance to GT,\nE_t minus BiNI (cm)"]
    views = PRESENTATION_VIEWS
    fig, axes = plt.subplots(
        len(views),
        len(titles),
        figsize=(2.3 * len(titles), 1.9 * len(views) + 1.2),
        layout="constrained",
    )
    for row, v in zip(axes, views, strict=True):
        row[0].imshow(inputs.views.images[v])
        for ax, n in zip(row[1:-1], maps.values(), strict=True):
            ax.imshow(colours(n[v]))
        ch = row[-1].imshow(
            change[v], cmap=DIVERGING, norm=Normalize(-CHANGE_CM, CHANGE_CM)
        )
        for ax in row:
            ax.set_xticks([])
            ax.set_yticks([])
        row[0].set_ylabel(f"view {v}")
    for ax, title in zip(axes[0], titles, strict=True):
        ax.set_title(title, fontsize=8)
    fig.colorbar(ch, ax=axes[-1, -1], location="bottom", label="cm, blue: E_t closer")
    fig.suptitle(
        "Bedroom ai_006_007: normals of the refined depth (closer neighbor per axis) "
        "under three normal terms, next to DSINE's",
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
