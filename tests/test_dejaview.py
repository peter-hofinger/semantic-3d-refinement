import numpy as np
import torch

from src.backbone.dejaview import (
    SPECIAL_TOKENS,
    TokenMix,
    mix_tokens,
    patch_segments,
    segment_groups,
)
from src.segmentation.base import NO_INSTANCE


def random_case(seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    rng = np.random.default_rng(seed)
    segments = rng.integers(-1, 4, size=(3, 12))
    groups = torch.from_numpy(segment_groups(segments))
    tokens = torch.from_numpy(rng.normal(size=(3, SPECIAL_TOKENS + 12, 8))).float()
    return tokens, groups


def test_patch_segments_majority_impure_and_unlabeled() -> None:
    ids = np.zeros((1, 2, 6), dtype=np.int32)
    ids[0, :, 0:2] = 7  # pure
    ids[0, :, 2:4] = [[3, 3], [3, 5]]  # 3 of 4 -> 3
    ids[0, :, 4:6] = [[1, 1], [2, 2]]  # half -> none
    np.testing.assert_array_equal(patch_segments(ids, patch=2), [[7, 3, NO_INSTANCE]])
    ids[0, :, 0:2] = -1
    assert patch_segments(ids, patch=2)[0, 0] == NO_INSTANCE


def test_segment_groups_split_views_and_skip_unlabeled() -> None:
    groups = segment_groups(np.array([[4, 4, -1], [4, 2, 2]]))
    assert groups[2] == -1
    assert groups[0] == groups[1]
    assert groups[3] not in (groups[0], groups[4])  # same id, other view
    assert groups[4] == groups[5]
    assert sorted(set(groups[groups >= 0])) == [0, 1, 2]


def test_mix_zero_is_identity_bit_for_bit() -> None:
    tokens, groups = random_case(0)
    assert torch.equal(mix_tokens(tokens.bfloat16(), groups, 0.0), tokens.bfloat16())
    assert torch.equal(mix_tokens(tokens, groups, 0.0), tokens)


def test_mix_one_gives_segment_means_and_keeps_the_rest() -> None:
    tokens, groups = random_case(1)
    out = mix_tokens(tokens, groups, 1.0)
    torch.testing.assert_close(out[:, :SPECIAL_TOKENS], tokens[:, :SPECIAL_TOKENS])
    patches, mixed = tokens[:, SPECIAL_TOKENS:].reshape(-1, 8), out[:, SPECIAL_TOKENS:]
    mixed = mixed.reshape(-1, 8)
    torch.testing.assert_close(mixed[groups < 0], patches[groups < 0])
    for g in groups[groups >= 0].unique():
        torch.testing.assert_close(
            mixed[groups == g],
            patches[groups == g].mean(0).expand_as(mixed[groups == g]),
        )


def test_mix_preserves_each_segment_mean() -> None:
    tokens, groups = random_case(2)
    out = mix_tokens(tokens, groups, 0.3)
    before = tokens[:, SPECIAL_TOKENS:].reshape(-1, 8)
    after = out[:, SPECIAL_TOKENS:].reshape(-1, 8)
    for g in groups[groups >= 0].unique():
        torch.testing.assert_close(
            after[groups == g].mean(0), before[groups == g].mean(0)
        )


def test_hook_mixes_only_the_chosen_steps() -> None:
    tokens, groups = random_case(3)
    hook = TokenMix(groups, 1.0, frozenset({1}))
    block = torch.nn.Identity()
    block.register_forward_hook(hook)
    assert torch.equal(block(tokens), tokens)
    assert not torch.equal(block(tokens), tokens)
    assert torch.equal(block(tokens), tokens)
    assert hook.calls == 3
