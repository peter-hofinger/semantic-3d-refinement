"""Déjà View (nv-tlabs/dvlt) as a Backbone, plus a command to run it on views.

Déjà View runs one shared `LoopedAABlock` (frame attention, then global attention
over all views) K times on tokens (V, P, C): per view 1 camera and 4 register tokens,
then one token per 14 px patch; the last tokens are decoded into depth and rays.
`predict` can hook a `TokenMix` onto that block (issue #51, E51a): after the chosen
steps each view's patch tokens of a segment s move towards their mean,
t <- (1 - a) t + a mean_s(t), so the loop continues with tokens that agree more
within each segment. With a = 0 the output is Déjà View's bit for bit.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from accelerate import Accelerator
from dvlt.common.constants import DataField, PredictionField
from dvlt.model.dvlt.model import DVLT
from torch import Tensor

from src.geometry import Geometry
from src.loaders import Views
from src.segmentation.base import NO_INSTANCE

log = logging.getLogger(__name__)

# released inference settings (dvlt/config/experiments/dvlt.yaml)
CHECKPOINT = "nvidia/dvlt"
IMAGE_SIZE = 504
PATCH_SIZE = 14
SPECIAL_TOKENS = 5  # per view: 1 camera + 4 register tokens before the patch tokens


def patch_segments(ids: np.ndarray, patch: int = PATCH_SIZE) -> np.ndarray:
    """(V, H, W) ids -> (V, H/p * W/p): the id on > half of each patch, else -1."""
    assert ids.ndim == 3, ids.shape  # noqa: PLR2004
    assert np.issubdtype(ids.dtype, np.integer), ids.dtype
    v, h, w = ids.shape
    assert h % patch == 0, ids.shape
    assert w % patch == 0, ids.shape
    blocks = ids.reshape(v, h // patch, patch, w // patch, patch).transpose(
        0, 1, 3, 2, 4
    )
    blocks = blocks.reshape(v * (h // patch) * (w // patch), patch * patch)
    values, inverse = np.unique(blocks, return_inverse=True)
    inverse = inverse.reshape(blocks.shape)
    rows = np.arange(len(blocks))[:, None]
    counts = np.zeros((len(blocks), len(values)), dtype=np.int64)
    np.add.at(counts, (np.broadcast_to(rows, inverse.shape), inverse), 1)
    best = counts.argmax(axis=1)
    out = values[best].astype(np.int64)
    out[2 * counts[np.arange(len(blocks)), best] <= patch * patch] = NO_INSTANCE
    out[out < 0] = NO_INSTANCE
    return out.reshape(v, -1)


def segment_groups(segments: np.ndarray) -> np.ndarray:
    """(V, N) patch ids -> (V * N,) index 0..G-1 per (view, segment) pair, else -1."""
    assert segments.ndim == 2, segments.shape  # noqa: PLR2004
    view = np.broadcast_to(np.arange(len(segments))[:, None], segments.shape).ravel()
    flat = segments.ravel()
    labeled = flat != NO_INSTANCE
    groups = np.full(flat.shape, -1, dtype=np.int64)
    pairs = np.stack([view[labeled], flat[labeled]], axis=1)
    groups[labeled] = np.unique(pairs, axis=0, return_inverse=True)[1].ravel()
    return groups


def mix_tokens(tokens: Tensor, groups: Tensor, alpha: float) -> Tensor:
    """(V, P, C) tokens: each grouped patch token moves alpha towards its mean."""
    assert tokens.ndim == 3, tokens.shape  # noqa: PLR2004
    v, p, c = tokens.shape
    assert groups.shape == (v * (p - SPECIAL_TOKENS),), (groups.shape, tokens.shape)
    assert groups.dtype == torch.int64
    assert 0.0 <= alpha <= 1.0, alpha
    patches = tokens[:, SPECIAL_TOKENS:].reshape(-1, c).float()
    labeled = groups >= 0
    g = groups[labeled]
    count = int(g.max()) + 1 if len(g) else 0
    sums = patches.new_zeros(count, c).index_add_(0, g, patches[labeled])
    sizes = torch.bincount(g, minlength=count).to(patches.dtype)
    means = sums / sizes[:, None]
    mixed = patches.clone()
    mixed[labeled] = (1.0 - alpha) * patches[labeled] + alpha * means[g]
    out = tokens.clone()
    out[:, SPECIAL_TOKENS:] = mixed.reshape(v, p - SPECIAL_TOKENS, c).to(tokens.dtype)
    assert out.isfinite().all()
    return out


@dataclass
class TokenMix:
    """Forward hook on Déjà View's looped block: `mix_tokens` after the chosen steps."""

    groups: Tensor  # (V * patches,) from `segment_groups`
    alpha: float
    steps: frozenset[int]  # 0-based loop steps after which the tokens are mixed
    calls: int = 0

    def __call__(
        self, _module: torch.nn.Module, _args: tuple, output: Tensor
    ) -> Tensor:
        """Mix the block's output if this call's step is chosen; count the call."""
        step, self.calls = self.calls, self.calls + 1
        if step not in self.steps:
            return output
        return mix_tokens(output, self.groups.to(output.device), self.alpha)


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

    def predict(self, views: Views, mix: TokenMix | None = None) -> Geometry:
        """Predict depth, confidence, rays and cameras per view, optionally mixed."""
        height, width = views.images.shape[1:3]
        if height % PATCH_SIZE or width % PATCH_SIZE:
            msg = f"image size must be a multiple of {PATCH_SIZE}, got {width}x{height}"
            raise ValueError(msg)

        model, accelerator = self._load()
        images = torch.from_numpy(views.images).permute(0, 3, 1, 2).float() / 255.0
        batch = {DataField.IMAGES: images[None].to(accelerator.device)}
        block = model.model.recurrent_blocks
        assert len(block) == 1, "Déjà View's released model shares one looped block"
        assert model.model.patch_start_idx == SPECIAL_TOKENS
        handle = None if mix is None else block[0].register_forward_hook(mix)
        try:
            with torch.no_grad(), accelerator.autocast():
                raw = model.model(batch)
                # official post-processing: cameras fitted from the predicted rays
                preds = model._postprocess_predictions(batch, raw)  # noqa: SLF001
        finally:
            if handle is not None:
                handle.remove()
        assert mix is None or mix.calls == self.steps, (mix.calls, self.steps)

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
