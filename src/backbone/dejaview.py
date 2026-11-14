"""Déjà View (nv-tlabs/dvlt) as a Backbone, plus a command to run it on views.

Déjà View runs one shared `LoopedAABlock` K times on tokens (V, P, C): per view 1
camera and 4 register tokens, then one token per 14 px patch. Each step is frame
attention (the P tokens of one view), then global attention (all V * P tokens); the
last tokens are decoded into depth and rays. `predict` can take an `AttentionBias`
(issue #53, E53a): at the chosen steps the logits of patch pairs with the same
segment id get +beta,

    softmax(q k^T / sqrt(d) + B) v,   B_ij = beta if s_i = s_j >= 0, else 0,

so a token keeps its content but weighs the keys of its own segment e^beta times
more (within its view for frame attention, across views for global attention). Our
runs pass no `block_mask` and use the `auto` backend, so dvlt's `Attention.forward`
is plain SDPA; `AttentionBias.attend` recomputes it (`qkv`, SDPA, projection) with
the mask added. Without a mask it equals dvlt's forward bit for bit.
"""

from __future__ import annotations

import argparse
import functools
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from accelerate import Accelerator
from dvlt.common.constants import DataField, PredictionField
from dvlt.model.dvlt.model import DVLT
from dvlt.model_components import Attention
from dvlt.model_components.layers.attention import _attn_state
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
ALIGN = 8  # the memory-efficient SDPA kernel wants the mask's last dim a multiple of 8
WHERE = ("frame", "global")

Probe = Callable[[int, str, Tensor, Tensor], None]  # (step, where, q, k)


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


def token_segments(segments: np.ndarray) -> Tensor:
    """(V, patches) ids -> (V, SPECIAL_TOKENS + patches) int64, -1 on special tokens."""
    assert segments.ndim == 2, segments.shape  # noqa: PLR2004
    assert segments.min() >= NO_INSTANCE
    special = np.full((len(segments), SPECIAL_TOKENS), NO_INSTANCE, dtype=np.int64)
    return torch.from_numpy(np.concatenate([special, segments], axis=1))


def segment_bias(ids: Tensor, beta: float, dtype: torch.dtype) -> Tensor:
    """(B, N) ids -> (B, 1, N, N) mask: beta where ids match and are >= 0, else 0."""
    assert ids.ndim == 2, ids.shape  # noqa: PLR2004
    assert ids.dtype == torch.int64
    assert np.isfinite(beta), beta
    b, n = ids.shape
    padded = ids.new_zeros((b, 1, n, n + (-n % ALIGN)), dtype=dtype)
    bias = padded[..., :n]
    same = (ids[:, :, None] == ids[:, None, :]) & (ids[:, :, None] >= 0)
    bias[:, 0].masked_fill_(same, beta)
    return bias


def qkv(attn: Attention, x: Tensor, pos: Tensor | None) -> tuple[Tensor, ...]:
    """Dvlt's `Attention.forward` up to SDPA: q, k (normed, RoPE), v as (B, h, N, d)."""
    b, n, _ = x.shape
    out = attn.qkv(x).reshape(b, n, 3, attn.num_heads, attn.head_dim)
    q, k, v = out.permute(2, 0, 3, 1, 4).unbind(0)
    q, k = attn.q_norm(q), attn.k_norm(k)
    if attn.rope is not None and pos is not None:
        q, k = attn.rope(q, pos), attn.rope(k, pos)
    return q, k, v


@dataclass
class AttentionBias:
    """Adds `segment_bias` to frame or global attention at the chosen loop steps."""

    segments: np.ndarray  # (V, patches) from `patch_segments`
    beta: float
    steps: frozenset[int]  # 0-based loop steps whose attention is biased
    where: str  # "frame" or "global"
    probe: Probe | None = None  # sees every step's q and k (analysis only)
    step: int = -1
    _ids: dict[str, Tensor] = field(default_factory=dict, repr=False)
    _bias: dict[str, Tensor] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        """Check the arguments; frame ids are (V, P), global ids (1, V * P)."""
        assert self.where in WHERE, self.where
        ids = token_segments(self.segments)
        self._ids = {"frame": ids, "global": ids.reshape(1, -1)}

    def next_step(self, _module: torch.nn.Module, _args: tuple) -> None:
        """Forward pre-hook on the looped block: count the loop step."""
        self.step += 1

    def attend(
        self, where: str, attn: Attention, x: Tensor, pos: Tensor | None = None
    ) -> Tensor:
        """Dvlt's `Attention.forward` (auto backend, no block mask) plus the bias."""
        b, n, c = x.shape
        assert (b, n) == self._ids[where].shape, (x.shape, self._ids[where].shape)
        q, k, v = qkv(attn, x, pos)
        if self.probe is not None:
            self.probe(self.step, where, q, k)
        mask = None
        if where == self.where and self.step in self.steps:
            if where not in self._bias:
                ids = self._ids[where].to(x.device)
                self._bias[where] = segment_bias(ids, self.beta, v.dtype)
            mask = self._bias[where]
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        out = attn.proj(out.transpose(1, 2).reshape(b, n, c))
        return attn.proj_drop(out)


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

    def predict(self, views: Views, bias: AttentionBias | None = None) -> Geometry:
        """Predict depth, confidence, rays and cameras per view, optionally biased."""
        height, width = views.images.shape[1:3]
        if height % PATCH_SIZE or width % PATCH_SIZE:
            msg = f"image size must be a multiple of {PATCH_SIZE}, got {width}x{height}"
            raise ValueError(msg)

        model, accelerator = self._load()
        images = torch.from_numpy(views.images).permute(0, 3, 1, 2).float() / 255.0
        batch = {DataField.IMAGES: images[None].to(accelerator.device)}
        blocks = model.model.recurrent_blocks
        assert len(blocks) == 1, "Déjà View's released model shares one looped block"
        assert model.model.patch_start_idx == SPECIAL_TOKENS
        attns = {
            "frame": blocks[0].frame_attn.attn,
            "global": blocks[0].global_attn.attn,
        }
        handle = None
        if bias is not None:
            assert _attn_state["backend"] == "auto", _attn_state
            assert bias.step == -1, "an AttentionBias serves one forward pass"
            handle = blocks[0].register_forward_pre_hook(bias.next_step)
            for where, attn in attns.items():
                assert attn.fused_attn
                assert not attn.training
                attn.forward = functools.partial(bias.attend, where, attn)
        try:
            with torch.no_grad(), accelerator.autocast():
                raw = model.model(batch)
                # official post-processing: cameras fitted from the predicted rays
                preds = model._postprocess_predictions(batch, raw)  # noqa: SLF001
        finally:
            if handle is not None:
                handle.remove()
                for attn in attns.values():
                    del attn.forward
        assert bias is None or bias.step == self.steps - 1, (bias.step, self.steps)

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
