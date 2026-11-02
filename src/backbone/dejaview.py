"""Déjà View (nv-tlabs/dvlt) as a Backbone, plus a command to run it on views."""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from accelerate import Accelerator
from dvlt.common.constants import DataField, PredictionField
from dvlt.model.dvlt.model import DVLT

from src.geometry import Geometry
from src.loaders import Views

if TYPE_CHECKING:
    from torch import Tensor

log = logging.getLogger(__name__)

# released inference settings (dvlt/config/experiments/dvlt.yaml)
CHECKPOINT = "nvidia/dvlt"
IMAGE_SIZE = 504
PATCH_SIZE = 14


@dataclass
class DejaViewBackbone:
    """Runs the released Déjà View checkpoint; weights load on first use."""

    steps: int = 16  # loop iterations K, trained range 8 to 16
    checkpoint: str = CHECKPOINT
    _model: Any = field(default=None, init=False, repr=False)
    _accelerator: Any = field(default=None, init=False, repr=False)

    def _load(self) -> tuple[Any, Accelerator]:
        if self._model is None:
            precision = "bf16" if torch.cuda.is_available() else "no"
            self._accelerator = Accelerator(mixed_precision=precision)
            model = DVLT(
                img_size=IMAGE_SIZE,
                inference_steps=self.steps,
                depth_head_type="conv",
            )
            model.load_pretrained(self.checkpoint, strict=True)
            model.setup_test(self._accelerator)
            self._model = model
        return self._model, self._accelerator

    def predict(self, views: Views) -> Geometry:
        """Predict depth, confidence, rays and cameras for every view."""
        height, width = views.images.shape[1:3]
        if height % PATCH_SIZE or width % PATCH_SIZE:
            msg = f"image size must be a multiple of {PATCH_SIZE}, got {width}x{height}"
            raise ValueError(msg)

        model, accelerator = self._load()
        images = torch.from_numpy(views.images).permute(0, 3, 1, 2).float() / 255.0
        batch = {DataField.IMAGES: images[None].to(accelerator.device)}
        with torch.no_grad(), accelerator.autocast():
            raw = model.model(batch)
            # official post-processing: cameras fitted from the predicted rays
            preds = model._postprocess_predictions(batch, raw)  # noqa: SLF001

        def to_numpy(t: Tensor) -> np.ndarray:
            return t[0].float().cpu().numpy().astype(np.float32)

        cameras = preds[PredictionField.CAMERAS][0]
        rays = to_numpy(raw["rays"])  # (N, H, W, 6): direction, then origin
        return Geometry(
            depth=to_numpy(preds[PredictionField.DEPTHS]),
            confidence=to_numpy(preds[PredictionField.DEPTHS_CONF]),
            intrinsics=to_numpy(cameras.get_intrinsics_matrices()[None]),
            cam_to_world=to_numpy(_to_4x4(cameras.camera_to_worlds)[None]),
            ray_origins=rays[..., 3:],
            ray_directions=rays[..., :3],
        )


def _to_4x4(c2w: Tensor) -> Tensor:
    """Append the [0, 0, 0, 1] row to (N, 3, 4) poses."""
    bottom = torch.zeros_like(c2w[:, :1])
    bottom[..., 3] = 1.0
    return torch.cat([c2w, bottom], dim=1)


def main() -> None:
    """Run Déjà View on prepared views and save the geometry next to them."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("views", type=Path, nargs="+", help="prepared views (.npz)")
    parser.add_argument("--steps", type=int, default=16, help="loop iterations K")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    backbone = DejaViewBackbone(steps=args.steps)
    for path in args.views:
        views = Views.load(path)
        geometry = backbone.predict(views)
        out = path.with_name(f"dejaview_k{args.steps}_{path.stem}.npz")
        geometry.save(out)
        log.info("%s: %d views -> %s", views.scene_id, len(views), out)


if __name__ == "__main__":
    main()
