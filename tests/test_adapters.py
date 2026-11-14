import gc
from collections.abc import Callable

import numpy as np
import pytest
import torch
from dvlt.model.dvlt.model import DVLT

from src.backbone import adapters
from src.backbone.adapters import (
    ALIGN,
    SPECIAL_TOKENS,
    SegmentAdapter,
    biased_attention,
    forward,
    grid_segments,
    patch_segments,
    plain_forward,
    segment_codes,
    segment_offsets,
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
    ids = grid_segments(2, 4, 5, block=3).reshape(2, 4, 5)
    assert ids[0, 0, 0] == ids[0, 2, 2] != ids[0, 0, 3]
    assert ids[0, 3, 0] != ids[0, 0, 0]
    assert len(np.unique(ids[0])) == 4
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


def test_gradients_reach_only_the_adapter(net: torch.nn.Module) -> None:
    images, codes = inputs(3)
    adapter = SegmentAdapter(bias=True, pool=True)
    forward(net, images, adapter, codes)["depth"].square().mean().backward()
    assert adapter.beta.grad is not None
    assert adapter.beta.grad.abs().sum() > 0
    assert adapter.up.grad.abs().sum() > 0
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

    def grads() -> tuple[torch.Tensor, torch.Tensor]:
        torch.manual_seed(5)
        adapter = SegmentAdapter(bias=True, pool=True)
        with torch.no_grad():
            adapter.beta.normal_(0, 0.5)
            adapter.up.normal_(0, 0.01)
        forward(net, images, adapter, codes)["depth"].square().mean().backward()
        return adapter.beta.grad, adapter.up.grad

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
