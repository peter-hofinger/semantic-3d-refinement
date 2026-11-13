"""E51a check: is the loss caused by scoring mixed depth on Déjà View's cameras.

E51a scores a mixed run's depth on the plain run's cameras, since mixing bends the rays
past the protocol's pinhole fit. Where the fit still holds (a <= 0.1 on the first 4
steps), each mixed output is also scored on its own cameras and rays; if both lose
against Déjà View by about as much, the loss is the depth's, not the pairing's.
GT instances, Hypersim dev, scored as `e51a_token_mix.score`.

uv run python experiments/scripts/e51a_own_cameras.py
"""

from pathlib import Path

import torch
from e51a_token_mix import fixed_cameras, score

from src.backbone.dejaview import (
    DejaViewBackbone,
    TokenMix,
    patch_segments,
    segment_groups,
)
from src.evaluation.labeled import ground_truth_cloud
from src.loaders.ground_truth import GroundTruth
from src.loaders.views import Views

ROOT = Path("data/processed/hypersim")
SCENES = ("ai_010_007_cam_00", "ai_006_007_cam_00", "ai_044_001_cam_00")
FIRST4 = frozenset(range(4))


def main() -> None:
    """Print Chamfer on fixed and on own cameras per scene and a."""
    backbone = DejaViewBackbone(steps=16)
    for name in SCENES:
        views = Views.load(ROOT / name / "views_n20_uniform_504.npz")
        gt = GroundTruth.load(ROOT / name / "gt_views_n20_uniform_504.npz")
        truth = ground_truth_cloud(
            gt.depth, gt.instance_ids, gt.classes, views.intrinsics, views.cam_to_world
        )
        groups = torch.from_numpy(segment_groups(patch_segments(gt.instance_ids)))
        plain = backbone.predict(views)
        print(f"{name} DV chamfer {score(plain, truth)[0]['chamfer_cm']:.3f}")
        for a in (0.01, 0.03, 0.1):
            mixed = backbone.predict(views, TokenMix(groups, a, FIRST4))
            fixed = score(fixed_cameras(mixed, plain)[0], truth)[0]["chamfer_cm"]
            own = score(mixed, truth)[0]["chamfer_cm"]
            print(f"  a={a} first4: fixed cameras {fixed:.3f}  own cameras {own:.3f}")


if __name__ == "__main__":
    main()
