"""E84q (issue #84): where the dev-trained head keeps and drops, against Déjà View's confidence and the true error.

One row per dev scene, view 0: the image, the head's keep probability s, Déjà View's confidence c (log) and
the point's distance a to the nearest GT point (cm, log, the accuracy term's error). Pixels without GT are blank.

`uv run python experiments/scripts/e84q_figure.py runs/e84q_... --out experiments/figures/e84q_dev_maps.jpg`
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.colors import LogNorm

from src.geometry import Geometry
from src.loaders.views import Views
from src.refinement.keep_chamfer import KeepScene

VIEW = 0


def main() -> None:
    """Write the figure."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load((args.run / "run.yaml").read_text())["config"]
    fig, ax = plt.subplots(
        len(config["scenes"]), 4, figsize=(16, 3.4 * len(config["scenes"]))
    )
    for row, name in enumerate(config["scenes"]):
        folder = Path(config["root"]) / name
        scene = KeepScene.load(name, config, "cpu")
        valid = scene.valid.numpy()
        views = Views.load(folder / f"{config['views']}.npz")
        conf = Geometry.load(
            folder / f"{config['backbone']}_{config['views']}.npz"
        ).confidence
        s = np.load(args.run / f"probabilities_{name}.npy").astype(np.float32)
        err = np.full(valid.shape, np.nan, np.float32)
        err[valid] = scene.probes.a
        blank = ~valid[VIEW]
        panels = [
            ("image", views.images[VIEW], {}),
            (
                "head s (0 drop, 1 keep)",
                np.where(blank, np.nan, s[VIEW]),
                {"vmin": 0, "vmax": 1, "cmap": "viridis"},
            ),
            (
                "Déjà View c (log)",
                np.where(blank, np.nan, conf[VIEW]),
                {"norm": LogNorm(1, 60), "cmap": "viridis"},
            ),
            (
                "distance a to nearest GT point (cm, log)",
                err[VIEW],
                {"norm": LogNorm(0.1, 20), "cmap": "magma"},
            ),
        ]
        for col, (title, image, kw) in enumerate(panels):
            handle = ax[row, col].imshow(image, **kw)
            ax[row, col].set_title(f"{name[:10]}: {title}", fontsize=9)
            ax[row, col].axis("off")
            if col:
                fig.colorbar(handle, ax=ax[row, col], fraction=0.046)
    fig.tight_layout()
    fig.savefig(args.out, dpi=80, pil_kwargs={"quality": 88})


if __name__ == "__main__":
    main()
