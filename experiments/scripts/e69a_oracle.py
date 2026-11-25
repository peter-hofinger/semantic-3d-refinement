"""E69a's references without training: what the grid and the reach allow a generator.

Two oracles per scene, both using the ground truth (the targets the generators train on,
`volume.gt_in_frame`, all 20 views), both dropping a start whose nearest ground-truth point
is further than `offset` cells away on some axis, as a negative logit would:
- `oracle-lattice`: every lattice point of every candidate voxel (`split`, `reach`) moves to
  its nearest ground-truth point; the ceiling of the lattice, gated, implicit and split arms;
- `oracle-seeded`: every Deja View point moves to its nearest ground-truth point; the ceiling
  of the seeded arms (one copy per input point).
The clouds are written as `e69a-<oracle>_<views>_cloud.npz` into the scene folders, for `cloud_eval`.

uv run python experiments/scripts/e69a_oracle.py configs/e69a.yaml overfit1 dev val
"""

import sys
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree

from src.loaders.hypersim.config import load_config
from src.refinement.generator import (
    candidate_mask,
    lattice,
    load_rung,
    load_scene,
)
from src.refinement.volume import voxelize


def snap(start: np.ndarray, target: np.ndarray, reach: float) -> np.ndarray:
    """Return the targets (m, 3) nearest to the starts (n, 3), kept where within reach.

    A start is kept if its nearest target lies within `reach` of it on every axis.
    """
    assert start.ndim == target.ndim == 2
    assert start.shape[1] == target.shape[1] == 3
    nearest = cKDTree(target, balanced_tree=False, compact_nodes=False).query(
        start, workers=-1
    )[1]
    snapped = target[nearest]
    return snapped[np.abs(snapped - start).max(1) <= reach]


def main() -> None:
    """Write both oracles for every scene of the named splits."""
    config, rung, *splits = sys.argv[1:]
    cfg = load_rung(Path(config), rung).cfg
    for scene, cam in load_config(Path("configs/hypersim.yaml")).scenes(splits).items():
        name = f"{scene}_{cam}"
        pts, gt, has, scale = load_scene(cfg, name, "none")
        target = gt[has].numpy().astype(np.float64)
        cell = pts.cell
        grid = voxelize(pts, semantics=False)
        centre = torch.nonzero(candidate_mask(grid, cfg.reach)).flip(1).float() + 0.5
        lattice_cells = (centre[:, None] + lattice(cfg.split)).reshape(-1, 3).numpy()
        starts = {
            "oracle-lattice": (pts.lo.numpy() + lattice_cells * cell).astype(
                np.float64
            ),
            "oracle-seeded": pts.points.numpy().astype(np.float64),
        }
        for oracle, start in starts.items():
            cloud = snap(start, target, cfg.offset * cell) * scale
            np.savez_compressed(
                cfg.root / name / f"{cfg.tag}-{oracle}_{cfg.views}_cloud.npz",
                points=cloud.astype(np.float32),
            )
            print(
                f"{name}: {oracle}: {len(start)} starts, {len(cloud)} kept", flush=True
            )


if __name__ == "__main__":
    main()
