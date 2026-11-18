"""E57c oracle: write every predict scene's true distance to the GT surface as a keep score.

`<out>_<views>.npz` in each scene folder holds `keep` = the negated distance (higher keeps), the
file `scene_eval`'s `keep_by` reads. Cutting by it at keep q is the best any distance head can do.
"""

import argparse
from pathlib import Path

import numpy as np

from src.refinement.volume import VolumeConfig, load_scene


def main() -> None:
    """Write the oracle keep score of every predict scene of the config."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/e57c_volume.yaml"))
    parser.add_argument("--out", default="oracle-distance")
    args = parser.parse_args()
    cfg = VolumeConfig.load(args.config)
    assert cfg.keep is not None, cfg.keep
    assert cfg.keep.target == "distance", cfg.keep
    for scene in cfg.predict:
        _, geometry, _, _, distance = load_scene(cfg, scene, "none")
        assert distance is not None
        score = (-distance).reshape(geometry.depth.shape).astype(np.float32)
        np.savez_compressed(
            cfg.root / scene / f"{args.out}_{cfg.views}.npz", keep=score
        )
        print(f"{scene}: mean distance {distance.mean():.4f} median depths", flush=True)


if __name__ == "__main__":
    main()
