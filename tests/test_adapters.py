import gc
from collections.abc import Callable

import numpy as np
import pytest
import torch
from dvlt.model.dvlt.model import DVLT

from src.backbone import adapters
from src.backbone.adapters import (
    ALIGN,
    ATTENTIONS,
    LORA_TARGETS,
    SPECIAL_TOKENS,
    SegmentAdapter,
    biased_attention,
    forward,
    grid_segments,
    patch_segments,
    plain_forward,
    segment_codes,
    segment_offsets,
    token_gate,
)
from src.segmentation.base import NO_INSTANCE

VIEWS, SIDE = 2, 28  # 2 x 2 patches per view


@pytest.fixture(scope="module")
def net() -> torch.nn.Module:
    torch.manual_seed(0)
    model = DVLT(img_size=504, inference_steps=16, depth_head_type="conv").model
    model.eval().requires_grad_(requires_grad=False)
    return model


def inputs(seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    images = torch.rand(1, VIEWS, 3, SIDE, SIDE)
    codes, _ = segment_codes(np.array([[0, 0, 1, -1], [1, 0, 2, 2]]))
    return images, codes


def test_patch_segments_majority_impure_and_unlabeled() -> None:
    ids = np.zeros((1, 2, 6), dtype=np.int32)
    ids[0, :, 0:2] = 7
    ids[0, :, 2:4] = [[3, 3], [3, 5]]  # 3 of 4 -> 3
    ids[0, :, 4:6] = [[1, 1], [2, 2]]  # half -> none
    np.testing.assert_array_equal(patch_segments(ids, patch=2), [[7, 3, NO_INSTANCE]])


def test_grid_segments_tile_each_view_apart() -> None:
    ids = grid_segments(2, 27, 36, tiles=(3, 3)).reshape(2, 27, 36)
    assert ids[0, 0, 0] == ids[0, 8, 11] != ids[0, 0, 12]
    assert ids[0, 9, 0] != ids[0, 0, 0]
    _, counts = np.unique(ids[0], return_counts=True)
    assert len(counts) == 9
    assert (counts == 9 * 12).all()
    assert not set(ids[0].ravel()) & set(ids[1].ravel())


def test_segment_codes_one_hot_padded_and_drop_smallest() -> None:
    segments = np.array([[5, 5, 5, -1], [9, 5, 2, 2]])
    codes, dropped = segment_codes(segments)
    assert codes.shape == (2, SPECIAL_TOKENS + 4, ALIGN)
    assert dropped == 0
    assert codes[:, :SPECIAL_TOKENS].abs().sum() == 0
    patches = codes[:, SPECIAL_TOKENS:]
    assert patches[0, 3].sum() == 0
    assert torch.equal(patches[0, 0], patches[1, 1])  # id 5 in both views
    assert not torch.equal(patches[1, 0], patches[1, 2])
    codes, dropped = segment_codes(segments, max_segments=2)
    assert dropped == 1
    assert codes[1, SPECIAL_TOKENS].sum() == 0  # id 9, one patch: dropped


def test_biased_attention_is_the_written_out_softmax_per_head() -> None:
    torch.manual_seed(1)
    b, h, n, d = 2, 3, 7, 8
    q, k, v = (torch.randn(b, h, n, d, dtype=torch.float64) for _ in range(3))
    ids = torch.tensor([[0, 0, 1, -1, 1, 2, 0], [3, -1, 3, 3, 1, 1, -1]])
    codes = torch.nn.functional.one_hot(ids.clamp(min=0), 4).double()
    codes[ids < 0] = 0
    beta = torch.tensor([0.5, -1.0, 2.0], dtype=torch.float64)
    same = (ids[:, :, None] == ids[:, None, :]) & (ids >= 0)[:, :, None]
    logits = (
        q @ k.transpose(-2, -1) / d**0.5 + beta[None, :, None, None] * same[:, None]
    )
    expected = logits.softmax(-1) @ v
    torch.testing.assert_close(biased_attention(q, k, v, codes, beta), expected)
    plain = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    torch.testing.assert_close(biased_attention(q, k, v, codes, 0 * beta), plain)


def test_segment_offsets_point_to_the_segment_mean() -> None:
    x = torch.tensor([[[1.0, 0.0], [3.0, 2.0], [5.0, 5.0], [7.0, 1.0]]])
    codes = torch.tensor([[[1.0, 0.0], [1.0, 0.0], [0.0, 0.0], [0.0, 1.0]]])
    expected = torch.tensor([[[1.0, 1.0], [-1.0, -1.0], [0.0, 0.0], [0.0, 0.0]]])
    torch.testing.assert_close(segment_offsets(x, codes), expected)


def test_zero_adapter_is_dejaview_bit_for_bit(net: torch.nn.Module) -> None:
    images, codes = inputs(2)
    with torch.no_grad():
        ref = net.forward_inference(images)
        for key, out in plain_forward(net, images).items():
            assert torch.equal(out, ref[key]), key
        adapter = SegmentAdapter(bias=True, pool=True)
        for key, out in forward(net, images, adapter, codes).items():
            assert torch.equal(out, ref[key]), key
    pool = SegmentAdapter(bias=False, pool=True)
    out = forward(net, images, pool, codes)  # training path: decoded view by view
    torch.testing.assert_close(out["depth"], ref["depth"])
    biased = forward(net, images, SegmentAdapter(bias=True, pool=False), codes)
    torch.testing.assert_close(biased["depth"], ref["depth"])
    lora = SegmentAdapter(bias=False, pool=False, lora=4)
    torch.testing.assert_close(forward(net, images, lora, codes)["depth"], ref["depth"])
    gate = SegmentAdapter(bias=False, pool=True, gate=True)
    torch.testing.assert_close(forward(net, images, gate, codes)["depth"], ref["depth"])


def test_token_gate_is_the_written_out_sigmoid() -> None:
    x = torch.tensor([[[1.0, 2.0], [0.0, -1.0], [3.0, 0.5]]])
    w, b = torch.tensor([0.5, -1.0]), torch.tensor(0.25)
    logits = [0.5 * 1.0 - 2.0 + 0.25, 1.0 + 0.25, 1.5 - 0.5 + 0.25]
    expected = 1 / (1 + torch.exp(-torch.tensor(logits)))
    torch.testing.assert_close(token_gate(x, w, b), expected.reshape(1, 3, 1))
    assert torch.equal(token_gate(x, 0 * w, 0 * b), torch.full((1, 3, 1), 0.5))


def test_open_gate_is_pooling_and_closed_gate_is_dejaview(
    net: torch.nn.Module,
) -> None:
    images, codes = inputs(9)
    torch.manual_seed(10)
    plain = SegmentAdapter(bias=False, pool=True)
    gated = SegmentAdapter(bias=False, pool=True, gate=True)
    with torch.no_grad():
        plain.up.normal_(0, 0.01)
        gated.down.weight.copy_(plain.down.weight)
        gated.up.copy_(plain.up)
        ref = net.forward_inference(images)["depth"]
        pooled = forward(net, images, plain, codes)["depth"]
        assert not torch.allclose(pooled, ref)
        gated.gate_b.fill_(30.0)  # sigmoid(30) = 1 - 1e-13
        torch.testing.assert_close(forward(net, images, gated, codes)["depth"], pooled)
        gated.gate_b.fill_(-30.0)
        torch.testing.assert_close(forward(net, images, gated, codes)["depth"], ref)


def test_gate_trains_once_b_k_moves(net: torch.nn.Module) -> None:
    images, codes = inputs(11)
    adapter = SegmentAdapter(bias=False, pool=True, gate=True)
    forward(net, images, adapter, codes)["depth"].square().mean().backward()
    assert adapter.up.grad.abs().sum() > 0
    assert not adapter.gate_w.grad.any()
    assert not adapter.gate_b.grad.any()
    adapter.zero_grad()
    with torch.no_grad():
        adapter.up.normal_(0, 0.01)
    forward(net, images, adapter, codes)["depth"].square().mean().backward()
    assert adapter.gate_w.grad.abs().sum() > 0
    assert adapter.gate_b.grad.abs().sum() > 0


def test_lora_is_the_merged_weight(net: torch.nn.Module) -> None:
    images, codes = inputs(7)
    torch.manual_seed(8)
    adapter = SegmentAdapter(bias=False, pool=False, lora=4)
    with torch.no_grad():
        for u in adapter.lora_up.values():
            u.normal_(0, 0.01)
        out = forward(net, images, adapter, codes)
        block = net.recurrent_blocks[0]
        attns = dict(
            zip(
                ATTENTIONS, (block.frame_attn.attn, block.global_attn.attn), strict=True
            )
        )
        linears = {
            f"{a}_{t}": getattr(attns[a], t) for a in ATTENTIONS for t in LORA_TARGETS
        }
        saved = {n: m.weight.clone() for n, m in linears.items()}
        try:
            for n, m in linears.items():
                m.weight += adapter.lora_up[n] @ adapter.lora_down[n]
            merged = net.forward_inference(images)
        finally:
            for n, m in linears.items():
                m.weight.copy_(saved[n])
    assert not torch.equal(out["depth"], net.forward_inference(images)["depth"])
    for key in ("depth", "world_points"):
        torch.testing.assert_close(out[key], merged[key], rtol=1e-4, atol=1e-5)


def test_gradients_reach_only_the_adapter(net: torch.nn.Module) -> None:
    images, codes = inputs(3)
    adapter = SegmentAdapter(bias=True, pool=True, lora=4)
    forward(net, images, adapter, codes)["depth"].square().mean().backward()
    assert adapter.beta.grad is not None
    assert adapter.beta.grad.abs().sum() > 0
    assert adapter.up.grad.abs().sum() > 0
    assert all(u.grad.abs().sum() > 0 for u in adapter.lora_up.values())
    assert all(p.grad is None for p in net.parameters())


def test_a_pass_leaves_no_loop_state_alive(net: torch.nn.Module) -> None:
    images, codes = inputs(6)
    adapter = SegmentAdapter(bias=True, pool=True)
    forward(net, images, adapter, codes)["depth"].square().mean().backward()
    with torch.no_grad():
        adapter.up.normal_(0, 0.01)
        forward(net, images, adapter, codes)
    gc.collect()
    assert not [o for o in gc.get_objects() if isinstance(o, adapters.Loop)]


def test_checkpoint_recomputes_with_the_same_step(
    net: torch.nn.Module, monkeypatch: pytest.MonkeyPatch
) -> None:
    images, codes = inputs(4)

    def grads() -> list[torch.Tensor]:
        torch.manual_seed(5)
        adapter = SegmentAdapter(bias=True, pool=True, lora=4)
        with torch.no_grad():
            adapter.beta.normal_(0, 0.5)
            adapter.up.normal_(0, 0.01)
            for u in adapter.lora_up.values():
                u.normal_(0, 0.01)
        forward(net, images, adapter, codes)["depth"].square().mean().backward()
        return [p.grad for p in adapter.parameters()]

    checkpointed = grads()

    def no_checkpoint(
        fn: Callable,
        *args: object,
        use_reentrant: bool,  # noqa: ARG001 (checkpoint's flag; no checkpoint here)
        **kwargs: object,
    ) -> object:
        return fn(*args, **kwargs)

    monkeypatch.setattr(adapters, "checkpoint", no_checkpoint)
    for a, b in zip(checkpointed, grads(), strict=True):
        torch.testing.assert_close(a, b)
