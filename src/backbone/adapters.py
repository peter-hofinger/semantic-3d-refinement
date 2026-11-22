"""Learned segment adapters inside Déjà View's frozen loop (E52a, issue #52).

E51a (token averaging) and E53a (one hand-set attention bias) found that the frozen
model does not improve when GT segments are pushed into its loop by hand. Here small
modules are trained while every Déjà View weight stays frozen, so gradient descent, not
a grid, decides per loop step k and head h how segments act:

- bias: logit_ij += β[k, a, h] [s_i = s_j] in frame (a = 0) or global (a = 1) attention.
  Exact without an N x N mask: with e_i the one-hot code of token i's segment (zero for
  camera, register and unlabelled tokens), [q_i, β √d e_i] · [k_j, e_j] / √d =
  q_i · k_j / √d + β [s_i = s_j]; v is padded with zeros and the extra columns dropped.
- pool: after loop step k, x_i += B_k A (m_s(i) - x_i), m_s the mean token of segment s
  over all views (ids are scene-wide), A (dim -> rank) shared and B_k (rank -> dim) zero
  at the start, as LoRA (Hu et al. 2022) initialises its up-projection.
- lora: W x + U D x for the qkv and proj linears of frame and global attention (Hu et
  al. 2022), D (rank x in) random, U (out x rank) zero; one set for all loop steps, as
  the block is shared. With no segments it is plain fine-tuning (E52b's control). Its
  scale (the paper's alpha / r) is 1: with Adam, tuning it is about the same as
  tuning the learning rate (Hu et al. 2022, Sec. 4.1).
- gate (E87a, issue #87): the pooling step becomes
  x_i += g_i B_k A (m_s(i) - x_i), g_i = sigmoid(w_k · x_i + b_k), so each token
  decides from its own state how much of its segment's mean it takes, as Flamingo's
  gated cross-attention (Alayrac et al. 2022) lets a frozen model's layers take new
  input only where it helps, but per token. w_k = 0, b_k = 0 at the start: g = 1/2
  everywhere, never 0, so B_k (zero, as above) still gets a gradient; w_k and b_k get
  one once B_k has moved (a gate at 0 together with B_k = 0 would never train, each
  gradient being proportional to the other).

With every parameter zero and no grad, `forward` is Déjà View's inference bit for bit.
With grad, each loop step is checkpointed (Chen et al. 2016) with its step index as an
argument and patches the attention only while it runs, so the recomputation in
backward (after `forward` has returned) reads the same step's parameters; the frozen
full-resolution decoders run one view at a time, each call checkpointed, so their
activations are held for one view only. The loop's helpers live at module level and get
their per-pass state as a `Loop` argument: beartype decorates and keeps every function
it sees, so closures made per pass kept each pass's codes alive (E52a's job 2738 ran out
of GPU memory after ~6500 steps).
"""

from __future__ import annotations

import functools
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from dvlt.model.dvlt.model import _slice_expand_flatten
from dvlt.model_components import Attention
from dvlt.model_components.layers.attention import _attn_state
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from src.segmentation.base import NO_INSTANCE

PATCH_SIZE = 14
SPECIAL_TOKENS = 5  # per view: 1 camera + 4 register tokens before the patch tokens
MAX_SEGMENTS = 192  # head dim 64 + 192 codes = 256, the flash kernel's limit
ALIGN = 8  # code width padded to a multiple of 8 for the fused SDPA kernels
ATTENTIONS = ("frame", "global")
LORA_TARGETS = ("qkv", "proj")  # each attention's linears that get a LoRA


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


def grid_segments(
    views: int, rows: int, cols: int, tiles: tuple[int, int] = (3, 3)
) -> np.ndarray:
    """(V, rows * cols) ids of tr x tc even patch tiles, apart per view (control)."""
    tr, tc = tiles
    assert 0 < tr <= rows, (tr, rows)
    assert 0 < tc <= cols, (tc, cols)
    assert views > 0, views
    row, col = np.arange(rows)[:, None] * tr // rows, np.arange(cols)[None] * tc // cols
    ids = row * tc + col
    return np.arange(views)[:, None] * tr * tc + ids.reshape(1, -1)


def segment_codes(
    segments: np.ndarray, max_segments: int = MAX_SEGMENTS
) -> tuple[Tensor, int]:
    """(V, P) ids -> (V, 5 + P, S) one-hot codes, S % 8 = 0; and how many were dropped.

    Beyond max_segments the smallest segments (fewest patches) become unlabelled.
    """
    assert segments.ndim == 2, segments.shape  # noqa: PLR2004
    assert segments.min() >= NO_INSTANCE
    ids, counts = np.unique(segments[segments >= 0], return_counts=True)
    kept = np.sort(ids[np.argsort(-counts, kind="stable")[:max_segments]])
    labelled = np.isin(segments, kept)
    width = max(ALIGN, len(kept) + (-len(kept) % ALIGN))
    v, p = segments.shape
    codes = torch.zeros(v, SPECIAL_TOKENS + p, width)
    view, patch = np.nonzero(labelled)
    column = np.searchsorted(kept, segments[view, patch])
    codes[view, SPECIAL_TOKENS + patch, column] = 1.0
    assert (codes.sum(-1) <= 1).all()
    return codes, len(ids) - len(kept)


def qkv(attn: Attention, x: Tensor, pos: Tensor | None) -> tuple[Tensor, ...]:
    """Dvlt's `Attention.forward` up to SDPA: q, k (normed, RoPE), v as (B, h, N, d)."""
    b, n, _ = x.shape
    out = attn.qkv(x).reshape(b, n, 3, attn.num_heads, attn.head_dim)
    q, k, v = out.permute(2, 0, 3, 1, 4).unbind(0)
    q, k = attn.q_norm(q), attn.k_norm(k)
    if attn.rope is not None and pos is not None:
        q, k = attn.rope(q, pos), attn.rope(k, pos)
    return q, k, v


def biased_attention(
    q: Tensor, k: Tensor, v: Tensor, codes: Tensor, beta: Tensor
) -> Tensor:
    """softmax(q kᵀ / √d + β_h [same segment]) v for (B, h, N, d) q, k, v."""
    b, h, n, d = q.shape
    assert codes.shape[:2] == (b, n), (codes.shape, q.shape)
    assert beta.shape == (h,), beta.shape
    e = codes[:, None].to(q.dtype).expand(b, h, n, codes.shape[-1])
    gain = (beta * d**0.5).to(q.dtype)[None, :, None, None]
    out = F.scaled_dot_product_attention(
        torch.cat([q, gain * e], -1),
        torch.cat([k, e], -1),
        torch.cat([v, torch.zeros_like(e)], -1),
        scale=d**-0.5,
    )
    return out[..., :d]


def segment_offsets(x: Tensor, codes: Tensor) -> Tensor:
    """(B, N, C) tokens, (B, N, S) codes -> m_s(i) - x_i per labelled token, else 0."""
    assert x.shape[:2] == codes.shape[:2], (x.shape, codes.shape)
    c, xf = codes.float(), x.float()
    count = c.sum(1)[..., None].clamp(min=1.0)
    means = c @ ((c.transpose(1, 2) @ xf) / count)
    return (means - xf * c.sum(-1, keepdim=True)).to(x.dtype)


class SegmentAdapter(nn.Module):
    """β[k, attention, head], the pooling maps A, B_k and/or LoRA; zero at the start."""

    def __init__(  # noqa: PLR0913
        self,
        *,
        bias: bool,
        pool: bool,
        steps: int = 16,
        heads: int = 12,
        dim: int = 768,
        rank: int = 16,
        lora: int = 0,
        gate: bool = False,
    ) -> None:
        """Bias: (steps, 2, heads) zeros; pool: A random (LoRA's init), B_k zeros.

        lora > 0: D (lora x dim) as nn.Linear's init (LoRA's), U zeros, per target.
        gate (needs pool): w (steps, dim) and b (steps,) zeros.
        """
        super().__init__()
        assert bias or pool or lora > 0
        assert pool or not gate
        self.beta = nn.Parameter(torch.zeros(steps, 2, heads)) if bias else None
        self.down = nn.Linear(dim, rank, bias=False) if pool else None
        self.up = nn.Parameter(torch.zeros(steps, rank, dim)) if pool else None
        self.gate_w = nn.Parameter(torch.zeros(steps, dim)) if gate else None
        self.gate_b = nn.Parameter(torch.zeros(steps)) if gate else None
        self.lora_down = self.lora_up = None
        if lora > 0:
            names = [f"{a}_{t}" for a in ATTENTIONS for t in LORA_TARGETS]
            out = {"qkv": 3 * dim, "proj": dim}
            self.lora_down = nn.ParameterDict(
                {n: nn.Linear(dim, lora, bias=False).weight for n in names}
            )
            self.lora_up = nn.ParameterDict(
                {
                    n: nn.Parameter(torch.zeros(out[n.split("_")[1]], lora))
                    for n in names
                }
            )

    def is_zero(self) -> bool:
        """Tell whether the adapter leaves Déjà View's output unchanged."""
        ups = [self.beta, self.up]
        ups += [] if self.lora_up is None else list(self.lora_up.values())
        return all(p is None or not p.detach().any() for p in ups)


@contextmanager
def replaced(forwards: dict[nn.Module, Callable]) -> Iterator[None]:
    """Give each module the paired forward while inside; restore its own after."""
    for module, fn in forwards.items():
        module.forward = fn
    try:
        yield
    finally:
        for module in forwards:
            module.__dict__.pop("forward", None)


def plain_forward(net: nn.Module, images: Tensor) -> dict[str, Tensor]:
    """Dvlt's `forward_inference` without its no_grad, the image encoder kept frozen."""
    b, s, _, h, w = images.shape
    with torch.no_grad():
        z0 = net._encode_images(images)  # noqa: SLF001
    register = net.register_token.expand(b, s, -1, -1)
    register = register.reshape(b * s, net.num_register_tokens, -1)
    camera = _slice_expand_flatten(net.camera_token, b, s)
    x = torch.cat([camera, register, z0], dim=1)
    rope = net._get_rope_positions(b * s, h, w, images.device)  # noqa: SLF001
    features = net._solve_inference(x, rope, b, s)  # noqa: SLF001
    return net._decode(features, h, w, b, s, rope)  # noqa: SLF001


class Loop:
    """One pass's adapter, looped block, attentions and codes; its step count."""

    def __init__(
        self,
        adapter: SegmentAdapter,
        block: Callable,
        attns: list[Attention],
        frame: Tensor,
        joint: Tensor,
    ) -> None:
        """Codes per view (V, T, S) for frame attention, (1, V T, S) for global."""
        self.adapter, self.block, self.attns = adapter, block, attns
        self.codes = (frame, joint)
        self.step, self.k = -1, -1


def attend(
    loop: Loop, a: int, attn: Attention, x: Tensor, pos: Tensor | None = None
) -> Tensor:
    """Dvlt's `Attention.forward` with loop step k's segment bias in attention a."""
    n, c = x.shape[1:]
    q, k, v = qkv(attn, x, pos)
    assert loop.adapter.beta is not None
    out = biased_attention(q, k, v, loop.codes[a], loop.adapter.beta[loop.k, a])
    return attn.proj_drop(attn.proj(out.transpose(1, 2).reshape(-1, n, c)))


def token_gate(x: Tensor, w: Tensor, b: Tensor) -> Tensor:
    """(B, N, C) tokens, (C,) w, () b -> (B, N, 1) gates sigmoid(w · x_i + b)."""
    assert x.shape[-1] == w.shape[0], (x.shape, w.shape)
    assert b.ndim == 0, b.shape
    g = torch.sigmoid(x.float() @ w.float() + b.float())[..., None]
    assert torch.isfinite(g).all()
    return g


def low_rank(linear: nn.Linear, down: Tensor, up: Tensor, x: Tensor) -> Tensor:
    """W x + b + U D x: the frozen linear plus its LoRA."""
    return F.linear(x, linear.weight, linear.bias) + F.linear(F.linear(x, down), up)


def run(loop: Loop, k: int, *args: Any) -> Tensor:  # noqa: ANN401 (dvlt block args)
    """Loop step k: the block with biased attention and LoRA, then segment pooling."""
    loop.k = k
    adapter = loop.adapter
    biased = [] if adapter.beta is None else loop.attns
    swaps = {m: functools.partial(attend, loop, a, m) for a, m in enumerate(biased)}
    if adapter.lora_down is not None and adapter.lora_up is not None:
        for a, attn in zip(ATTENTIONS, loop.attns, strict=True):
            for t in LORA_TARGETS:
                n = f"{a}_{t}"
                swaps[getattr(attn, t)] = functools.partial(
                    low_rank, getattr(attn, t), adapter.lora_down[n], adapter.lora_up[n]
                )
    with replaced(swaps):
        x = loop.block(*args)
    up, down = loop.adapter.up, loop.adapter.down
    if up is None or down is None:
        return x
    joint = loop.codes[1]
    flat = x.reshape(1, joint.shape[1], -1)
    delta = down(segment_offsets(flat, joint)) @ up[k]
    w, b = adapter.gate_w, adapter.gate_b
    if w is not None and b is not None:
        delta = token_gate(flat, w[k], b[k]) * delta
    return (flat + delta.to(flat.dtype)).reshape(x.shape)


def looped(loop: Loop, *args: Any) -> Tensor:  # noqa: ANN401 (dvlt block args)
    """Run the next loop step, checkpointed with its index when grad is on."""
    loop.step += 1
    if torch.is_grad_enabled():
        return checkpoint(run, loop, loop.step, *args, use_reentrant=False)
    return run(loop, loop.step, *args)


def forward(
    net: nn.Module, images: Tensor, adapter: SegmentAdapter, codes: Tensor
) -> dict[str, Tensor]:
    """Déjà View on (1, V, 3, H, W) images, the adapter in its loop; codes (V, T, S)."""
    b, s, _, h, w = images.shape
    assert b == 1, images.shape
    tokens = net.patch_start_idx + (h // PATCH_SIZE) * (w // PATCH_SIZE)
    assert net.patch_start_idx == SPECIAL_TOKENS
    assert codes.shape[:2] == (s, tokens), (codes.shape, s, tokens)
    if adapter.is_zero() and not torch.is_grad_enabled():
        return net.forward_inference(images)
    blocks = net.recurrent_blocks
    assert len(blocks) == 1, "Déjà View's released model shares one looped block"
    block = blocks[0]
    attns = [block.frame_attn.attn, block.global_attn.attn]
    assert _attn_state["backend"] == "auto", _attn_state
    assert all(attn.fused_attn for attn in attns)
    codes = codes.to(images.device)
    loop = Loop(adapter, block.forward, attns, codes, codes.reshape(1, s * tokens, -1))
    swaps: dict[nn.Module, Callable] = {block: functools.partial(looped, loop)}
    chunk = net.decode_chunk_size
    if torch.is_grad_enabled():  # frozen full-resolution decoders: one view at a time
        net.decode_chunk_size = 1
        for d in (net.ray_decoder, net.depth_decoder):
            swaps[d] = functools.partial(checkpoint, d.forward, use_reentrant=False)
    try:
        with replaced(swaps):
            out = plain_forward(net, images)
    finally:
        net.decode_chunk_size = chunk
    assert loop.step == net.inference_steps - 1, loop.step
    return out


Forward = Callable[[nn.Module, Tensor], dict[str, Tensor]]


def conditioned(adapter: SegmentAdapter, codes: Tensor) -> Forward:
    """`forward` bound to an adapter and codes, as `DejaViewBackbone.predict` takes."""
    return functools.partial(forward, adapter=adapter, codes=codes)
