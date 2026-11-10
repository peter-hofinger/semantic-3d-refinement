"""E42: the normals of the refined depth, BiNI's E_n against the angle term (bedroom).

For views 18, 0, 12, 6 of ai_006_007: the image, DSINE's normals (what both terms pull
toward), and the closer-neighbor normals (`closer_neighbor_normals`) of Déjà View's
depth, of the depth refined with BiNI's E_n (the refiner of E41) and of the depth
refined with the angle term at λ_a = 4e-4, all in the camera frame and coloured alike
(red: right, green: up, blue: toward the camera). The same normal reconstruction for
every depth, so differences come from the depth alone.

uv run python experiments/scripts/e42_normals_after.py runs/<e42_angle run>
"""

import sys
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from src.analysis.scene_eval import load_inputs
from src.analysis.scene_figures import PRESENTATION_SCENE, PRESENTATION_VIEWS
from src.refinement.normals import closer_neighbor_normals

sys.path.insert(0, str(Path(__file__).parent))
from e42_normals_figure import colours

VARIANTS = {
    "Déjà View": "Déjà View",
    "refined, BiNI E_n": "Refined",
    "refined, angle E_a": "angle 4e-4",
}
OUT = Path("experiments/figures/e42_normals_after_ai_006_007.jpg")


def to_camera(n: torch.Tensor, rotation: torch.Tensor) -> np.ndarray:
    """Rotate normals (V, H, W, 3) from the rays' frame into each view's camera."""
    return torch.einsum("vhwi,vij->vhwj", n, rotation).numpy()


def main() -> None:
    """Draw the figure."""
    run_dir = Path(sys.argv[1])
    run = yaml.safe_load((run_dir / "run.yaml").read_text(encoding="utf-8"))
    inputs = load_inputs(run["config"], PRESENTATION_SCENE, torch.device("cpu"))
    names = [v["name"] for v in run["config"]["variants"]]
    arrays = np.load(run_dir / "results.npz")
    rot = inputs.scene.rotation.double()
    dirs = inputs.scene.rays.double()[..., 3:]
    maps = {"DSINE": to_camera(inputs.normals.double(), rot)}
    for title, variant in VARIANTS.items():
        depth = torch.from_numpy(arrays[f"{PRESENTATION_SCENE}/{names.index(variant)}"])
        maps[title] = to_camera(
            closer_neighbor_normals(depth.double()[..., None] * dirs), rot
        )
    titles = ["image", *maps]
    views = PRESENTATION_VIEWS
    fig, axes = plt.subplots(
        len(views),
        len(titles),
        figsize=(2.3 * len(titles), 1.85 * len(views) + 0.6),
        layout="constrained",
    )
    for row, v in zip(axes, views, strict=True):
        row[0].imshow(inputs.views.images[v])
        for ax, n in zip(row[1:], maps.values(), strict=True):
            ax.imshow(colours(n[v]))
        for ax in row:
            ax.set_xticks([])
            ax.set_yticks([])
        row[0].set_ylabel(f"view {v}")
    for ax, title in zip(axes[0], titles, strict=True):
        ax.set_title(title, fontsize=8)
    fig.suptitle(
        "Bedroom ai_006_007: normals of each depth (closer neighbor per axis) next to "
        "DSINE's; red right, green up, blue toward the camera",
        x=0.01,
        ha="left",
        fontsize=9,
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=120)
    plt.close(fig)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
