"""E84a: the expected Chamfer loss, its exact value, gradient caching, the head."""

import itertools

import numpy as np
import pytest
import torch
from dvlt.model.dvlt.heads import DecoderHead

from src.refinement.keep_chamfer import (
    FEATURES,
    HIDDEN,
    KeepHead,
    KeepScene,
    Probes,
    exact_chamfer,
    expected_chamfer,
    loss_backward,
)


def clouds(n: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Aligned predictions and GT points, pixel i of one the pixel i of the other."""
    rng = np.random.default_rng(seed)
    truth = rng.random((n, 3))
    pred = truth + 0.05 * rng.standard_normal((n, 3))
    pred[: n // 5] += 0.5 * rng.standard_normal((n // 5, 3))  # flying points
    return pred, truth


def brute(kept: np.ndarray, pred: np.ndarray, truth: np.ndarray) -> tuple[float, float]:
    """Accuracy and completeness (cm) of the kept predictions by brute force."""
    d = 100 * np.linalg.norm(pred[:, None] - truth[None], axis=-1)  # (pred, truth)
    return float(d.min(1)[kept].mean()), float(d[kept].min(0).mean())


def tensors(p: Probes) -> tuple[torch.Tensor, ...]:
    return (
        torch.from_numpy(p.a),
        torch.from_numpy(p.index.astype(np.int64)),
        torch.from_numpy(p.dist),
    )


def test_the_loss_at_binary_s_with_complete_probes_is_the_exact_chamfer() -> None:
    pred, truth = clouds(60)
    probes = Probes.build(pred, truth, r=4)
    kept = np.ones(60, dtype=bool)
    kept[np.random.default_rng(1).choice(60, 3, replace=False)] = False  # < R dropped
    loss, acc, comp = expected_chamfer(
        torch.from_numpy(kept).float(), *tensors(probes), eps=1e-9
    )
    ref_acc, ref_comp = brute(kept, pred, truth)
    assert acc.item() == pytest.approx(ref_acc, rel=1e-5)
    assert comp.item() == pytest.approx(ref_comp, rel=1e-5)
    assert loss.item() == pytest.approx((ref_acc + ref_comp) / 2, rel=1e-5)
    exact = exact_chamfer(kept, probes, pred, truth)
    assert exact["probe_miss"] == 0
    assert exact["chamfer_cm"] == pytest.approx(loss.item(), rel=1e-5)


def test_the_completeness_is_the_expectation_over_independent_keeps() -> None:
    n = 7  # R = n - 1: the (R + 1)-th probe is the last prediction
    pred, truth = clouds(n, seed=2)
    probes = Probes.build(pred, truth, r=n - 1)
    s = np.random.default_rng(3).random(n)
    expected = 0.0
    for bits in itertools.product([False, True], repeat=n):
        kept = np.array(bits)
        prob = np.prod(np.where(kept, s, 1 - s))
        hit = kept[probes.index]
        first = np.where(
            hit.any(1), hit.argmax(1), n - 1
        )  # no kept probe: the last prediction's distance
        expected += prob * probes.dist[np.arange(n), first].mean()
    _, _, comp = expected_chamfer(torch.from_numpy(s), *tensors(probes), eps=1e-9)
    assert comp.item() == pytest.approx(expected, rel=1e-5)


def test_the_exact_chamfer_needs_no_probe_list() -> None:
    pred, truth = clouds(80, seed=4)
    probes = Probes.build(pred, truth, r=2)
    kept = np.random.default_rng(5).random(80) < 0.3
    exact = exact_chamfer(kept, probes, pred, truth)
    ref_acc, ref_comp = brute(kept, pred, truth)
    assert exact["probe_miss"] > 0  # the fallback query runs
    assert exact["accuracy_cm"] == pytest.approx(ref_acc, rel=1e-5)
    assert exact["completeness_cm"] == pytest.approx(ref_comp, rel=1e-5)
    _, _, comp = expected_chamfer(
        torch.from_numpy(kept).float(), *tensors(probes), eps=1e-9
    )
    assert comp.item() <= ref_comp + 1e-6  # a dropped list is charged a lower bound


def test_chunked_backprop_gives_the_full_scene_gradient() -> None:
    torch.manual_seed(0)
    v, h, w = 5, 6, 7
    pred, truth = clouds(v * h * w - 4, seed=6)
    probes = Probes.build(pred, truth, r=4)
    valid = torch.ones(v, h, w, dtype=torch.bool)
    valid.view(-1)[[3, 40, 77, 120]] = False  # pixels without GT
    scene = KeepScene(
        "toy",
        torch.randn(v, FEATURES, h, w).half(),
        valid,
        pred,
        truth,
        probes,
        tensors(probes),
    )
    reference = KeepHead()
    s = reference(scene.features.float())[valid]
    expected_chamfer(s, *scene.tensors, eps=1e-6)[0].backward()
    for chunk in (1, 2, 5):
        head = KeepHead()
        head.load_state_dict(reference.state_dict())
        loss_backward(head, scene, 1e-6, chunk)
        for p, q in zip(head.parameters(), reference.parameters(), strict=True):
            assert torch.allclose(p.grad, q.grad, rtol=1e-4, atol=1e-7)


def test_the_head_is_the_confidence_branch_of_deja_views_output_block() -> None:
    block = DecoderHead(in_dim=8, out_dim=2, head_type="conv").output_block
    ours = KeepHead().net
    assert [type(m) for m in ours] == [type(m) for m in block]
    for mine, theirs in ((ours[0], block[0]), (ours[2], block[2])):
        assert mine.kernel_size == theirs.kernel_size
        assert mine.padding_mode == theirs.padding_mode
        assert mine.weight.shape[1:] == theirs.weight.shape[1:]
    assert ours[0].weight.shape == (HIDDEN, FEATURES, 3, 3)
    assert ours[2].weight.shape[0] == 1  # the one output: the keep logit


def test_an_upper_bound_miss_charge_makes_the_loss_an_upper_bound() -> None:
    pred, truth = clouds(80, seed=7)
    probes = Probes.build(pred, truth, r=2, miss="bound")
    bound = 100 * (
        np.linalg.norm(truth - truth.mean(0), axis=1).max()
        + np.linalg.norm(pred - truth.mean(0), axis=1).max()
    )
    d = 100 * np.linalg.norm(pred[:, None] - truth[None], axis=-1)
    assert bound >= d.max()  # of every GT-prediction distance
    assert (probes.dist[:, -1] == np.float32(bound)).all()
    kept = np.random.default_rng(8).random(80) < 0.3
    _, _, comp = expected_chamfer(
        torch.from_numpy(kept).float(), *tensors(probes), eps=1e-9
    )
    assert comp.item() >= brute(kept, pred, truth)[1] - 1e-4
    kept[:] = True  # every GT point keeps a probe: exact again
    _, _, comp = expected_chamfer(
        torch.from_numpy(kept).float(), *tensors(probes), eps=1e-9
    )
    assert comp.item() == pytest.approx(brute(kept, pred, truth)[1], rel=1e-5)
