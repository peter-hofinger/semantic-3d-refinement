"""Solv3R (ubc-vision/solv3r) as a Backbone, plus a command to run it on views.

Solv3R (Kwak, Tagliasacchi, Kagami, Ono, Yi, arXiv 2611.02857) fine-tunes Déjà View's
loop into a stationary solver: one shared block updates the state z of every view per
call, z <- f(z, x0), with no step index, so a solve runs a fixed number of calls or
stops once the state settles, |f(z) - z| / |z| < tau (tau = 0.02, at least 4 and at
most 64 calls, as its offline benchmark). Its image encoder and output heads are Déjà
View's, frozen, so the outputs are the same fields: depth, confidence and a ray map per
view. Cameras are fitted to the rays by DVLT's post-processing, as in `dejaview.py` and
in Solv3R's own offline evaluation, so both backbones give the same kind of Geometry.

Code: Apache-2.0, pinned in the `solv3r` dependency group. Weights (jgkwak/solv3r):
NVIDIA License, research and evaluation only, downloaded into the Hugging Face cache.
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from dvlt.common.constants import DataField, PredictionField
from dvlt.model.dvlt.model import DVLT
from solv3r import Solv3R
from solv3r.model import autocast
from torch import Tensor

from src.backbone.dejaview import PATCH_SIZE, _to_4x4
from src.geometry import Geometry
from src.loaders import Views

log = logging.getLogger(__name__)

CHECKPOINT = "jgkwak/solv3r"
MIN_CALLS, MAX_CALLS = 4, 64  # adaptive solve, Solv3R's eval/offline.py defaults


@dataclass(frozen=True)
class _PostProcess:
    """The fields DVLT's `_postprocess_predictions` reads from its model."""

    patch_size: int = PATCH_SIZE
    use_depth_conf_for_pose: bool = False  # Solv3R's offline evaluation
    world_points_from_rays: bool = False


def relative_change(z: Tensor, z_next: Tensor) -> float:
    """|z_next - z| / |z|, the solver's stopping criterion."""
    return float((z_next.float() - z.float()).norm() / z.float().norm().clamp_min(1e-9))


def run_solver(
    step: Callable[[Tensor], Tensor],
    z: Tensor,
    calls: int,
    adaptive: float | None,
) -> tuple[Tensor, int]:
    """Call `step` from z `calls` times, or until the change is below `adaptive`."""
    if adaptive is None:
        for _ in range(calls):
            z = step(z)
        return z, calls
    for used in range(1, MAX_CALLS + 1):
        z_next = step(z)
        change = relative_change(z, z_next)
        z = z_next
        if used >= MIN_CALLS and change < adaptive:
            return z, used
    return z, MAX_CALLS


@dataclass
class Solv3RBackbone:
    """Runs the released Solv3R weights; weights load on first use."""

    calls: int = 16  # fixed calls, the paper's default; ignored with `adaptive`
    adaptive: float | None = None  # stop at this relative change of the state
    checkpoint: str = CHECKPOINT
    last_calls: int = field(default=0, init=False)
    _model: Any = field(default=None, init=False, repr=False)

    @property
    def tag(self) -> str:
        """Cache name: solv3r_c16 for fixed calls, solv3r_a0.02 for adaptive solves."""
        return (
            f"solv3r_a{self.adaptive:g}" if self.adaptive else f"solv3r_c{self.calls}"
        )

    def predict(self, views: Views) -> Geometry:
        """Predict depth, confidence, rays and cameras for every view."""
        height, width = views.images.shape[1:3]
        if height % PATCH_SIZE or width % PATCH_SIZE:
            msg = f"image size must be a multiple of {PATCH_SIZE}, got {width}x{height}"
            raise ValueError(msg)
        if self._model is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self._model = Solv3R.from_pretrained(self.checkpoint, device=device)
        model = self._model
        images = torch.from_numpy(views.images).permute(0, 3, 1, 2).float() / 255.0
        ctx = model.encode(images, chunk=len(images))
        with torch.no_grad(), autocast():
            z, self.last_calls = run_solver(
                lambda z: model.step(z, ctx),
                model.start(ctx),
                self.calls,
                self.adaptive,
            )
        raw = {k: v[None] for k, v in model.decode(z, ctx).items()}
        batch = {DataField.IMAGES: images[None].to(model.device)}
        preds = DVLT._postprocess_predictions(_PostProcess(), batch, raw)  # noqa: SLF001

        def to_numpy(t: Tensor) -> np.ndarray:
            return t[0].float().cpu().numpy().astype(np.float32)

        cameras = preds[PredictionField.CAMERAS][0]
        rays = to_numpy(raw["rays"])  # (N, H, W, 6): direction, then origin
        geometry = Geometry(
            depth=to_numpy(preds[PredictionField.DEPTHS]),
            confidence=to_numpy(preds[PredictionField.DEPTHS_CONF]),
            intrinsics=to_numpy(cameras.get_intrinsics_matrices()[None]),
            cam_to_world=to_numpy(_to_4x4(cameras.camera_to_worlds)[None]),
            ray_origins=rays[..., 3:],
            ray_directions=rays[..., :3],
        )
        assert np.isfinite(geometry.depth).all()
        assert (geometry.depth > 0).all()
        return geometry


def main() -> None:
    """Run Solv3R on prepared views and save the geometry next to them."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("views", type=Path, nargs="+", help="prepared views (.npz)")
    parser.add_argument("--calls", type=int, default=16, help="fixed solver calls")
    parser.add_argument("--adaptive", type=float, help="stop at this state change")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    backbone = Solv3RBackbone(calls=args.calls, adaptive=args.adaptive)
    for path in args.views:
        views = Views.load(path)
        geometry = backbone.predict(views)
        out = path.with_name(f"{backbone.tag}_{path.stem}.npz")
        geometry.save(out)
        log.info(
            "%s: %d views, %d calls -> %s",
            views.scene_id,
            len(views),
            backbone.last_calls,
            out,
        )


if __name__ == "__main__":
    main()
