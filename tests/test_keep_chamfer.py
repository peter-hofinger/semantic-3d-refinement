"""E84a: the expected Chamfer loss, its exact value, gradient caching, the head."""

import itertools
import math

import numpy as np
import pytest
import torch
from dvlt.common.constants import DataField
from dvlt.model.dvlt.heads import DecoderHead
from dvlt.model_components.head_activations import activate_head
from dvlt.model_components.loss import DepthLoss
from scipy.integrate import quad

from src.refinement.keep_chamfer import (
    FEATURES,
    HIDDEN,
    KeepHead,
    KeepScene,
    Probes,
    confidence_backward,
    confidence_loss,
    exact_chamfer,
    expected_chamfer,
    loss_backward,
    mimic_backward,
    mimic_loss,
    tail_distance,
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


@pytest.mark.parametrize("tail", [False, True])
def test_chunked_backprop_gives_the_full_scene_gradient(*, tail: bool) -> None:
    torch.manual_seed(0)
    v, h, w = 5, 6, 7
    pred, truth = clouds(v * h * w - 4, seed=6)
    probes = Probes.build(pred, truth, r=4, miss="bound")
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
        1.0,
    )
    reference = KeepHead()
    s = reference(scene.features.float())[valid]
    expected_chamfer(s, *scene.tensors, eps=1e-6, tail=tail)[0].backward()
    for chunk in (1, 2, 5):
        head = KeepHead()
        head.load_state_dict(reference.state_dict())
        loss_backward(head, scene, 1e-6, chunk, tail=tail)
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


@pytest.mark.parametrize(("kappa", "d_r"), [(0.3, 1.5), (2.0, 0.7), (8.0, 2.0)])
def test_the_tail_distance_is_the_mean_of_the_thinned_poisson_nearest_point(
    kappa: float, d_r: float
) -> None:
    survival = lambda x: math.exp(-kappa * (x**2 - d_r**2) / d_r**2)  # noqa: E731
    mean = d_r + quad(survival, d_r, math.inf)[0]
    got = tail_distance(torch.tensor([kappa]), torch.tensor([d_r]), torch.tensor([1e9]))
    assert got.item() == pytest.approx(mean, rel=1e-5)


def test_the_tail_charge_is_capped_grows_as_s_falls_and_has_a_gradient() -> None:
    pred, truth = clouds(80, seed=9)
    probes = Probes.build(pred, truth, r=3, miss="bound")
    a, index, dist = tensors(probes)
    for c in (0.5, 0.1, 0.01):
        s = torch.full((80,), c, requires_grad=True)
        _, _, comp = expected_chamfer(s, a, index, dist, eps=1e-9, tail=True)
        _, _, capped = expected_chamfer(s, a, index, dist, eps=1e-9)
        near = dist.clone()
        near[:, -1] = dist[:, -2]  # a miss charged the R-th probe's distance
        _, _, floor = expected_chamfer(s, a, index, near, eps=1e-9)
        assert floor.item() <= comp.item() <= capped.item()
        comp.backward()
        assert torch.isfinite(s.grad).all()
        assert (s.grad != 0).any()
    low = torch.full((80,), 0.01)
    mid = torch.full((80,), 0.1)
    assert (
        expected_chamfer(low, a, index, dist, eps=1e-9, tail=True)[2]
        > expected_chamfer(mid, a, index, dist, eps=1e-9, tail=True)[2]
    )
    kept = torch.ones(80)  # every probe kept: nothing left for the tail to charge
    assert expected_chamfer(kept, a, index, dist, eps=1e-9, tail=True)[2].item() == (
        pytest.approx(expected_chamfer(kept, a, index, dist, eps=1e-9)[2].item())
    )


def test_the_temperature_is_a_gain_on_the_logit_and_leaves_the_cut_where_it_is() -> (
    None
):
    torch.manual_seed(1)
    head = KeepHead()
    x = torch.randn(2, FEATURES, 5, 6)
    logit = head.net(x)[:, 0]
    for temperature in (1.0, 0.25, 0.05):
        head.temperature = temperature
        s = head(x)
        assert torch.allclose(s, torch.sigmoid(logit / temperature))
        assert ((s >= 0.5) == (logit >= 0)).all()


def test_the_head_from_deja_view_is_its_confidence_as_one_minus_one_over_c() -> None:
    torch.manual_seed(2)
    block = DecoderHead(in_dim=8, out_dim=2, head_type="conv").output_block
    x = torch.randn(2, FEATURES, 5, 6)
    with torch.no_grad():
        _, conf = activate_head(
            block(x), activation="exp_clamped", conf_activation="exp_plus_one"
        )
        s = KeepHead.from_deja_view(block)(x)
    assert (conf >= 1).all()
    assert torch.allclose(s, 1 - 1 / conf, atol=1e-6)
    assert ((s >= 0.5) == (conf >= 2)).all()


def test_the_confidence_loss_is_deja_views_and_minimal_at_alpha_over_gamma_a() -> None:
    alpha, gamma = 0.2, 1.0
    torch.manual_seed(0)
    a = torch.rand(400, dtype=torch.float64) * 0.1 + 0.02  # alpha / a in 2 .. 10
    x = torch.log(alpha / (gamma * a) - 1)  # c = 1 + exp(x) = alpha / (gamma a)
    ours = confidence_loss(x, a, alpha, gamma, None)
    depth = (1.0 + a).reshape(1, 1, 20, 20)
    gt = torch.ones_like(depth)
    batch = {
        DataField.DEPTHS: gt,
        DataField.POINT_MASKS: torch.ones_like(gt, dtype=torch.bool),
        DataField.IS_SYNTHETIC: torch.ones(gt.shape[:2], dtype=torch.bool),
    }
    reference, _ = DepthLoss(
        grad_loss=None, valid_range=-1.0, alpha=alpha, gamma=gamma, disable_conf=False
    )({"depth": depth, "depth_conf": (1 + torch.exp(x)).reshape(depth.shape)}, batch)
    assert torch.allclose(ours, reference)
    x.requires_grad_()
    confidence_loss(x, a, alpha, gamma, None).backward()
    assert x.grad.abs().max() < 1e-9  # the optimum is stationary
    for shift in (-0.1, 0.1):
        assert confidence_loss(x.detach() + shift, a, alpha, gamma, None) > ours


def test_the_quantile_drops_the_largest_losses() -> None:
    x = torch.zeros(100)
    a = torch.arange(100.0) / 100
    kept = confidence_loss(x, a, 0.2, 1.0, 0.9)
    full = 2 * a - 0.2 * math.log(2)
    assert torch.isclose(kept, full[full <= torch.quantile(full, 0.9)].mean())
    assert kept < confidence_loss(x, a, 0.2, 1.0, None)


@pytest.mark.parametrize("quantile", [None, 0.9])
def test_chunked_confidence_backprop_gives_the_full_scene_gradient(
    quantile: float | None,
) -> None:
    torch.manual_seed(0)
    v, h, w = 5, 6, 7
    pred, truth = clouds(v * h * w - 4, seed=6)
    probes = Probes.build(pred, truth, r=4, miss="bound")
    valid = torch.ones(v, h, w, dtype=torch.bool)
    valid.view(-1)[[3, 40, 77, 120]] = False
    scene = KeepScene(
        "toy",
        torch.randn(v, FEATURES, h, w).half(),
        valid,
        pred,
        truth,
        probes,
        tensors(probes),
        0.5,
    )
    reference = KeepHead()
    x = reference.logits(scene.features.float())[valid]
    a = scene.tensors[0] / (100.0 * scene.scale)
    confidence_loss(x, a, 0.2, 1.0, quantile).backward()
    for chunk in (1, 2, 5):
        head = KeepHead()
        head.load_state_dict(reference.state_dict())
        confidence_backward(head, scene, chunk, 1.0, 0.2, 1.0, quantile)
        for p, q in zip(head.parameters(), reference.parameters(), strict=True):
            assert torch.allclose(p.grad, q.grad, rtol=1e-4, atol=1e-7)


def test_extra_layers_widen_the_receptive_field_by_one_pixel_per_side_each() -> None:
    torch.manual_seed(0)
    assert len(KeepHead().net) == len(KeepHead(hidden=32, extra=0).net) == 3
    for extra in (0, 2):
        head = KeepHead(hidden=16, extra=extra)
        x = torch.randn(1, FEATURES, 15, 15, requires_grad=True)
        head.logits(x)[0, 7, 7].backward()
        reach = (x.grad.abs().sum(dim=(0, 1)) > 0).nonzero()
        radius = 1 + extra  # one 3 x 3 conv per layer, each adds one pixel
        assert reach.min() == 7 - radius
        assert reach.max() == 7 + radius


def test_the_mimic_loss_is_minimal_at_its_target_and_chunks_keep_the_gradient() -> None:
    target = torch.tensor([0.0, 0.1, 0.5, 0.9, 1.0], dtype=torch.float64)
    x = torch.logit(target.clamp(0.01, 0.99)).requires_grad_()
    mimic_loss(x, target.clamp(0.01, 0.99)).backward()
    assert x.grad.abs().max() < 1e-12  # s = target is stationary
    torch.manual_seed(0)
    v, h, w = 5, 6, 7
    pred, truth = clouds(v * h * w - 4, seed=6)
    probes = Probes.build(pred, truth, r=4, miss="bound")
    valid = torch.ones(v, h, w, dtype=torch.bool)
    valid.view(-1)[[3, 40, 77, 120]] = False
    scene = KeepScene(
        "toy",
        torch.randn(v, FEATURES, h, w).half(),
        valid,
        pred,
        truth,
        probes,
        tensors(probes),
        1.0,
    )
    goal = torch.rand(int(valid.sum()))
    reference = KeepHead()
    mimic_loss(reference.logits(scene.features.float())[valid], goal).backward()
    for chunk in (1, 2, 5):
        head = KeepHead()
        head.load_state_dict(reference.state_dict())
        mimic_backward(head, scene, goal, chunk, 1.0)
        for p, q in zip(head.parameters(), reference.parameters(), strict=True):
            assert torch.allclose(p.grad, q.grad, rtol=1e-4, atol=1e-7)
