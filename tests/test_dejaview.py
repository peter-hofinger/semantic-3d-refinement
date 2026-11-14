import numpy as np
import pytest
import torch
from dvlt.model_components import Attention

from src.backbone.dejaview import (
    ALIGN,
    SPECIAL_TOKENS,
    AttentionBias,
    patch_segments,
    qkv,
    segment_bias,
    token_segments,
)
from src.segmentation.base import NO_INSTANCE

DIM, HEADS, VIEWS, PATCHES = 32, 4, 2, 6


def case(seed: int) -> tuple[Attention, np.ndarray, torch.Tensor]:
    torch.manual_seed(seed)
    attn = Attention(DIM, num_heads=HEADS, qk_norm=True).eval()
    segments = np.random.default_rng(seed).integers(-1, 3, size=(VIEWS, PATCHES))
    x = torch.randn(VIEWS, SPECIAL_TOKENS + PATCHES, DIM)
    return attn, segments, x


def test_patch_segments_majority_impure_and_unlabeled() -> None:
    ids = np.zeros((1, 2, 6), dtype=np.int32)
    ids[0, :, 0:2] = 7  # pure
    ids[0, :, 2:4] = [[3, 3], [3, 5]]  # 3 of 4 -> 3
    ids[0, :, 4:6] = [[1, 1], [2, 2]]  # half -> none
    np.testing.assert_array_equal(patch_segments(ids, patch=2), [[7, 3, NO_INSTANCE]])
    ids[0, :, 0:2] = -1
    assert patch_segments(ids, patch=2)[0, 0] == NO_INSTANCE


def test_token_segments_leave_special_tokens_unlabeled() -> None:
    ids = token_segments(np.array([[4, -1], [2, 4]]))
    assert ids.shape == (2, SPECIAL_TOKENS + 2)
    assert (ids[:, :SPECIAL_TOKENS] == NO_INSTANCE).all()
    assert ids[:, SPECIAL_TOKENS:].tolist() == [[4, -1], [2, 4]]


def test_segment_bias_pairs_padding_and_unlabeled() -> None:
    ids = torch.tensor([[3, 3, -1, 5, -1]])
    bias = segment_bias(ids, 2.0, torch.float32)
    assert bias.shape == (1, 1, 5, 5)
    assert bias.stride(-2) % ALIGN == 0
    expected = torch.zeros(5, 5)
    expected[0, 0] = expected[0, 1] = expected[1, 0] = expected[1, 1] = 2.0
    expected[3, 3] = 2.0
    assert torch.equal(bias[0, 0], expected)


def test_global_ids_link_the_same_segment_across_views() -> None:
    ids = token_segments(np.array([[4, 1], [4, -1]])).reshape(1, -1)
    bias = segment_bias(ids, 1.0, torch.float32)[0, 0]
    first, second = SPECIAL_TOKENS, 2 * SPECIAL_TOKENS + 2
    assert bias[first, second] == 1.0  # id 4 in view 0 and view 1
    assert bias[first, first + 1] == 0.0
    assert bias[second + 1].abs().sum() == 0.0


def test_attend_without_mask_is_dvlt_bit_for_bit() -> None:
    attn, segments, x = case(0)
    hook = AttentionBias(segments, 3.0, frozenset(), "frame")
    with torch.no_grad():
        assert torch.equal(hook.attend("frame", attn, x), attn(x))
        flat = x.reshape(1, -1, DIM)
        assert torch.equal(hook.attend("global", attn, flat), attn(flat))


@pytest.mark.parametrize("where", ["frame", "global"])
def test_attend_with_mask_is_the_written_out_softmax(where: str) -> None:
    attn, segments, x = case(1)
    hook = AttentionBias(segments, 1.5, frozenset({0}), where)
    hook.step = 0
    if where == "global":
        x = x.reshape(1, -1, DIM)
    ids = token_segments(segments).reshape(x.shape[:2])
    with torch.no_grad():
        q, k, v = qkv(attn, x, None)
        same = (ids[:, None, :, None] == ids[:, None, None, :]) & (ids >= 0)[
            :, None, :, None
        ]
        logits = q @ k.transpose(-2, -1) / attn.head_dim**0.5 + 1.5 * same
        out = (logits.softmax(-1) @ v).transpose(1, 2).reshape(x.shape)
        torch.testing.assert_close(hook.attend(where, attn, x), attn.proj(out))
        other = "global" if where == "frame" else "frame"
        y = x.reshape(VIEWS, -1, DIM) if other == "frame" else x.reshape(1, -1, DIM)
        assert torch.equal(hook.attend(other, attn, y), attn(y))


def test_bias_only_at_the_chosen_steps() -> None:
    attn, segments, x = case(2)
    hook = AttentionBias(segments, 4.0, frozenset({1}), "frame")
    block = torch.nn.Identity()
    block.register_forward_pre_hook(hook.next_step)
    with torch.no_grad():
        outs = []
        for _ in range(3):
            block(x)
            outs.append(hook.attend("frame", attn, x))
        plain = attn(x)
    assert torch.equal(outs[0], plain)
    assert not torch.allclose(outs[1], plain)
    assert torch.equal(outs[2], plain)
    assert hook.step == 2
