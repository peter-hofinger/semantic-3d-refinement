"""E69's child-vertex head under the regular evaluation: the clouds of the dev scenes.

The slide's numbers use E69f's scorer (all-frames GT, in-box, voxelized, bedroom). The
repo's usual one is `src/analysis/cloud_eval.py`: the 20 views' GT pixels, every output
point, Déjà View's one Umeyama alignment, the mean over dev. This writes what it needs
for each dev scene, in Déjà View's frame: `e2`, the output of the bedroom-trained E2
model (the other two scenes are unseen by it), and the H6 oracle of the candidates with
the targets of either ground truth, `oracle-gt99` and `oracle-gt20`. Nothing is trained.

uv run python -m experiments.scripts.e69_regular
uv run python -m src.analysis.cloud_eval --cfg configs/e69f_eval.yaml --splits dev

`--report` keeps the surface scores of that command's report, per scene and their dev
mean, in `experiments/results/e69f_eval_dev.yaml`.

uv run python -m experiments.scripts.e69_regular --report runs/e69f_eval-<time>/run.yaml
"""

import argparse
import dataclasses
from pathlib import Path

import numpy as np
import torch
import yaml

from experiments.scripts.e69c_oracle import children_cloud
from experiments.scripts.e69e_rung0 import to_grid_frame
from experiments.scripts.e69f_rung0 import load
from experiments.scripts.e69g import CONFIG, masks
from experiments.scripts.e69i import load_model
from src.loaders.hypersim.config import load_config
from src.refinement import generator as gen

E2 = Path("runs/e69f/overfit1_e2/e69f-overfit1_e2-ovoxel-gt99-s0.pt")
REPORT = Path("experiments/results/e69f_eval_dev.yaml")
SURFACE = ("accuracy_cm", "completeness_cm", "chamfer_cm")


def keep_report(run: Path) -> None:
    """Write the surface scores of a `cloud_eval` report to `REPORT`."""
    full = yaml.safe_load(run.read_text())
    dev = full["splits"]["dev"]

    def surface(scores: dict) -> dict:
        return {v: {k: s[k] for k in SURFACE} for v, s in scores.items()}

    scenes = {name: surface(s) for name, s in dev["scenes"].items()}
    kept = {"aggregate": surface(dev["aggregate"]), "scenes": scenes}
    out = {"git": full["git"], "splits": {"dev": kept}}
    REPORT.write_text(yaml.safe_dump(out, sort_keys=False, allow_unicode=True))
    print(f"wrote {REPORT}")


def main() -> None:
    """Write the three clouds of every dev scene, or keep a report (`--report`)."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if args.report:
        keep_report(args.report)
        return
    base = gen.for_arm(gen.load_rung(CONFIG, "overfit1"), "ovoxel-gt99")
    cfg, model, device = load_model(E2)
    scenes = load_config(Path("configs/hypersim.yaml")).scenes(["dev"])
    for scene, cam in scenes.items():
        name = f"{scene}_{cam}"
        pts, _, _, scale = gen.load_scene(cfg, name, cfg.semantics)
        clouds = {"e69j-e2": gen.generate(model, pts, scale, device, cfg)}
        s = load(base, name)
        gt20 = to_grid_frame(s.gt_world[s.has], s.similarity, s.scale)
        for target, cells in (
            ("gt99", s.truth_cells),
            ("gt20", (gt20 - s.lo) / s.cell),
        ):
            held = dataclasses.replace(s, truth_cells=cells)
            voxel = torch.nonzero(
                torch.from_numpy(masks(held, base.coarse_reach)[2])
            ).flip(1)
            occupied, vertex = gen.child_targets(
                voxel, torch.from_numpy(cells).float(), base.split, s.dims
            )
            clouds[f"e69j-oracle-{target}"] = children_cloud(
                voxel, occupied, vertex, base.split, s.frame
            )
        for kind, cloud in clouds.items():
            assert len(cloud), (name, kind)
            assert np.isfinite(cloud).all(), (name, kind)
            np.savez_compressed(
                base.root / name / f"{kind}_{base.views}_cloud.npz", points=cloud
            )
            print(f"{name}: {kind}: {len(cloud):,} points", flush=True)


if __name__ == "__main__":
    main()
