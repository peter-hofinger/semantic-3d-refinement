"""A keep head trained on the expected Chamfer of the kept cloud (E84a, issue #84).

Idea: the oracle of #76 is the subset of Déjà View's per-pixel points with the minimal
Chamfer to the ground truth. Instead of imitating its mask, train on its definition.
A head gives each prediction i a keep probability s_i (sigmoid of a logit) and the loss
is the exact expectation of the Chamfer of the cloud that keeps each prediction
independently with probability s_i, with the predictions aligned once to the ground
truth (distances in cm).

Per scene, with the geometry frozen, these are constants: a_i, the distance of
prediction i to its nearest GT point, and for each GT point j its R nearest predictions
pi_j(1..R) with distances d_j1 <= ... <= d_jR and the (R+1)-th, d_j,R+1.

    A(s) = sum_i s_i a_i / (sum_i s_i + eps)
    C(s) = 1/N sum_j [ sum_{r<=R} d_jr s_pi_j(r) prod_{r'<r} (1 - s_pi_j(r'))
                       + d_j,R+1 prod_{r<=R} (1 - s_pi_j(r)) ]        L = (A + C) / 2

The nearest kept prediction of j is pi_j(r) iff pi_j(r) is kept and the nearer probes
are not, so C is the exact expected completeness whenever one of the R probes is kept;
if all R are dropped it charges d_j,R+1, a lower bound of the true distance (E84a), or
an upper bound of any distance (E84b, `Probes.build`), or, with `tail`, the expected
distance of a thinned planar Poisson process beyond the probes, capped at that bound
(E84d, `tail_distance`). At binary s with every GT point having a kept probe, A is the
mean accuracy and C the completeness of the kept cloud, so L is its Chamfer
(`exact_chamfer` computes it without the probe limit).
A is a ratio of expectations, not the expectation of the ratio.

A and C are whole-scene sums, so they are not minibatched. `loss_backward` computes s of
all pixels without grad, the loss and dL/ds on the full s, and then re-runs the head per
chunk of views with grad, backpropagating each chunk's slice of dL/ds: the gradient is
the full-scene gradient, with no subsampling and no rescaling (gradient caching, Gao et
al. 2021, "Scaling Deep Contrastive Learning Batch Size under Memory Limited Setup").

The head copies the confidence branch of Déjà View's convolutional depth head
(`dvlt.model.dvlt.heads.DecoderHead.output_block`, from MoGe's ConvStack): a 3 x 3
convolution, ReLU and a 1 x 1 convolution on the 66 channels (64 upsampled features and
a UV encoding) that block reads. Only it trains; the features are cached per scene.
Déjà View's confidence is c = 1 + exp(x) >= 1, not a probability: s = sigmoid(x) =
1 - 1/c is its monotone image in (0, 1), and `KeepHead.from_deja_view` starts from the
released weights, so the head's ranking is Déjà View's until the loss moves it (E84g).
"""

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import torch
from dvlt.common.constants import DataField
from scipy.spatial import cKDTree
from torch import Tensor, nn

from src.backbone.dejaview import DejaViewBackbone
from src.evaluation.labeled import align_output, ground_truth_cloud, nearest_distances
from src.geometry import Geometry
from src.loaders.ground_truth import GroundTruth
from src.loaders.views import Views
from src.refinement.multiview import points_from_rays, scene_from_geometry

FEATURES = 66  # what Déjà View's output_block reads: 64 upsampled channels + 2 UV
HIDDEN = 32  # its last_conv_channels
CM = 100.0
DEPTH_TOLERANCE = 1e-2  # median relative depth change of the re-run backbone, bf16


class KeepNet(nn.Module):
    """A keep logit per pixel from the decoder features; a subclass sets `net`."""

    def __init__(self) -> None:
        """Set `temperature`, which scales the logit (E84e)."""
        super().__init__()
        self.temperature = 1.0

    def logits(self, x: Tensor) -> Tensor:
        """Return the raw output x (B, H, W) of features (B, 66, H, W); c = 1 + e^x."""
        assert x.ndim == 4 and x.shape[1] == FEATURES, x.shape  # noqa: PLR2004
        return self.net(x)[:, 0]

    def forward(self, x: Tensor) -> Tensor:
        """Probabilities (B, H, W) in (0, 1) of float features (B, 66, H, W).

        s = sigmoid(logit / temperature), a gain on the logit: the cut at s = 0.5 is the
        logit's sign at any temperature and the loss as a function of s is unchanged.
        """
        assert self.temperature > 0, self.temperature
        s = torch.sigmoid(self.logits(x) / self.temperature)
        assert torch.isfinite(s).all()
        return s


class KeepHead(KeepNet):
    """Déjà View's confidence block: 3 x 3 convolution, ReLU, 1 x 1 convolution."""

    def __init__(self, hidden: int = HIDDEN, extra: int = 0) -> None:
        """Build the convolutions of `net`.

        The defaults are Déjà View's block; `extra` more 3 x 3 convolutions with ReLU of
        `hidden` channels widen the receptive field and the capacity (E84l).
        """
        super().__init__()
        assert hidden >= 1 and extra >= 0, (hidden, extra)
        layers: list[nn.Module] = [
            nn.Conv2d(FEATURES, hidden, 3, padding=1, padding_mode="replicate"),
            nn.ReLU(inplace=True),
        ]
        for _ in range(extra):
            layers += [
                nn.Conv2d(hidden, hidden, 3, padding=1, padding_mode="replicate"),
                nn.ReLU(inplace=True),
            ]
        self.net = nn.Sequential(*layers, nn.Conv2d(hidden, 1, 1))

    @classmethod
    def from_deja_view(cls, block: nn.Module) -> "KeepHead":
        """Return a head at Déjà View's confidence, copied from the block.

        The block's 1 x 1 conv outputs (depth, conf), conf last, with c = 1 + exp(x).
        s = sigmoid(x) = 1 - 1 / c (c - 1 is the odds of s) is a monotone function of c,
        and equal to it at the start.
        """
        head = cls()
        assert [type(m) for m in block] == [type(m) for m in head.net]
        with torch.no_grad():
            head.net[0].load_state_dict(block[0].state_dict())
            head.net[2].weight.copy_(block[2].weight[-1:])
            head.net[2].bias.copy_(block[2].bias[-1:])
        return head


class ResidualBlock(nn.Module):
    """x + conv(relu(conv(x))) with 3 x 3 convolutions of one dilation, then ReLU."""

    def __init__(self, width: int, dilation: int) -> None:
        """Build the two convolutions."""
        super().__init__()
        self.first, self.second = (
            nn.Conv2d(
                width,
                width,
                3,
                padding=dilation,
                dilation=dilation,
                padding_mode="replicate",
            )
            for _ in range(2)
        )

    def forward(self, x: Tensor) -> Tensor:
        """Return the block's output, the shape of x."""
        return torch.relu(x + self.second(torch.relu(self.first(x))))


class KeepResNet(KeepNet):
    """A shallow ResNet on the same features (E84o): stem, residual blocks, 1 x 1 conv.

    A 3 x 3 stem and two blocks of two 3 x 3 convolutions, dilations (2, 4): 5
    convolutions, about 190k parameters, a 27 x 27 receptive field at full resolution,
    no normalization or pooling (the ladder's rung 0).
    """

    def __init__(self, width: int = 64, dilations: Sequence[int] = (2, 4)) -> None:
        """Build `net`."""
        super().__init__()
        assert width >= 1 and len(dilations) >= 1, (width, dilations)
        self.net = nn.Sequential(
            nn.Conv2d(FEATURES, width, 3, padding=1, padding_mode="replicate"),
            nn.ReLU(inplace=True),
            *[ResidualBlock(width, d) for d in dilations],
            nn.Conv2d(width, 1, 1),
        )


@dataclass(frozen=True)
class Probes:
    """A scene's constants (cm): accuracy a, each GT point's R nearest predictions."""

    a: np.ndarray  # (N,) float32, prediction i's distance to the nearest GT point
    index: np.ndarray  # (N, R) int32, GT point j's nearest predictions, nearest first
    dist: np.ndarray  # (N, R + 1) float32, their distances and the (R + 1)-th's

    def __post_init__(self) -> None:
        """Check shapes, finiteness and the order of the probes."""
        n, r = self.index.shape
        assert self.a.shape == (n,) and self.dist.shape == (n, r + 1), self.dist.shape
        assert np.isfinite(self.a).all() and np.isfinite(self.dist).all()
        assert (self.a >= 0).all() and (np.diff(self.dist, axis=1) >= 0).all()
        assert self.index.min() >= 0 and self.index.max() < n

    @classmethod
    def build(
        cls, pred: np.ndarray, truth: np.ndarray, r: int, miss: str = "next"
    ) -> "Probes":
        """Probes of aligned predictions (N, 3) against the GT points (M, 3), M = N.

        `miss` is what the loss charges a GT point whose R probes are all dropped: its
        (R + 1)-th distance, a lower bound of the truth (`next`, E84a), or an upper
        bound of any GT-prediction distance (`bound`, E84b): max_j |t_j - c| +
        max_i |p_i - c| by the triangle inequality, c the GT mean.
        """
        assert pred.shape == truth.shape and pred.shape[1] == 3 and r >= 1, pred.shape  # noqa: PLR2004
        assert miss in ("next", "bound"), miss
        tree = cKDTree(pred, balanced_tree=False, compact_nodes=False)
        dist, index = tree.query(truth, r + 1, workers=-1)
        dist = CM * dist
        if miss == "bound":
            c = truth.mean(0)
            dist[:, r] = CM * (
                np.linalg.norm(truth - c, axis=1).max()
                + np.linalg.norm(pred - c, axis=1).max()
            )
        return cls(
            (CM * nearest_distances(truth, pred)).astype(np.float32),
            index[:, :r].astype(np.int32),
            dist.astype(np.float32),
        )

    def save(self, path: Path) -> None:
        """Write to an .npz file."""
        np.savez(path, a=self.a, index=self.index, dist=self.dist)

    @classmethod
    def load(cls, path: Path) -> "Probes":
        """Read what `save` wrote."""
        with np.load(path) as f:
            return cls(f["a"], f["index"], f["dist"])


def tail_distance(kappa: Tensor, d_r: Tensor, cap: Tensor) -> Tensor:
    """Return the expected distance (cm) to the nearest kept prediction beyond probe R.

    Planar Poisson process thinned by the keep rate: R probes with keep mass kappa lie
    within d_r, so the density is kappa / (pi d_r^2) and the nearest kept point at
    x >= d_r has survival exp(-kappa (x^2 - d_r^2) / d_r^2) (Rayleigh, the radial
    marginal of an isotropic 2D Gaussian). Its mean is d_r (1 + sqrt(pi)
    erfcx(sqrt(kappa)) / (2 sqrt(kappa))), capped at `cap`, an upper bound.
    """
    root = kappa.clamp_min(1e-12).sqrt()
    mean = d_r * (1 + math.sqrt(math.pi) * torch.special.erfcx(root) / (2 * root))
    return mean.minimum(cap)


def expected_chamfer(  # noqa: PLR0913 - the equation's symbols: s, a, index, dist, eps
    s: Tensor, a: Tensor, index: Tensor, dist: Tensor, eps: float, *, tail: bool = False
) -> tuple[Tensor, Tensor, Tensor]:
    """Loss L, accuracy A and completeness C of keep probabilities s (N,).

    `tail` charges a GT point whose probes are all dropped `tail_distance` instead of
    dist[:, -1]; that column must then be the upper bound (`miss="bound"`).
    """
    assert s.shape == a.shape == index.shape[:1] and eps > 0, (s.shape, index.shape)
    assert dist.shape == (len(s), index.shape[1] + 1), dist.shape
    assert (s >= 0).all() and (s <= 1).all()
    acc = (s * a).sum() / (s.sum() + eps)
    kept = s[index]
    drop = torch.cumprod(1 - kept, dim=1)  # nearer probes up to r are all dropped
    first = torch.cat([torch.ones_like(drop[:, :1]), drop[:, :-1]], dim=1)
    miss = tail_distance(kept.sum(1), dist[:, -2], dist[:, -1]) if tail else dist[:, -1]
    comp = ((dist[:, :-1] * kept * first).sum(1) + miss * drop[:, -1]).mean()
    assert torch.isfinite(acc) and torch.isfinite(comp)
    return (acc + comp) / 2, acc, comp


def exact_chamfer(
    kept: np.ndarray, probes: Probes, pred: np.ndarray, truth: np.ndarray
) -> dict[str, float]:
    """Accuracy, completeness and Chamfer (cm) of the kept predictions, same alignment.

    Not limited by R: a GT point without a kept probe is queried against the kept cloud.
    `probe_miss` is their share, where the loss charges a lower bound. Nothing kept:
    NaN scores.
    """
    assert kept.dtype == bool and kept.shape == (len(pred),)
    if not kept.any():
        nan = float("nan")
        return dict.fromkeys(("accuracy_cm", "completeness_cm", "chamfer_cm"), nan) | {
            "kept": 0.0,
            "probe_miss": 1.0,
        }
    hit = kept[probes.index]
    has = hit.any(1)
    rows = np.flatnonzero(has)
    comp = np.empty(len(pred))
    comp[rows] = probes.dist[rows, hit[rows].argmax(1)]
    if not has.all():
        comp[~has] = CM * nearest_distances(pred[kept], truth[~has])
    acc, comp_mean = float(probes.a[kept].mean()), float(comp.mean())
    return {
        "accuracy_cm": acc,
        "completeness_cm": comp_mean,
        "chamfer_cm": (acc + comp_mean) / 2,
        "kept": float(kept.mean()),
        "probe_miss": float(1 - has.mean()),
    }


@dataclass
class KeepScene:
    """One scene's cached features, valid pixels and constants, on the device."""

    name: str
    features: Tensor  # (V, 66, H, W) float16
    valid: Tensor  # (V, H, W) bool, the pixels with GT depth: prediction i of the N
    pred: np.ndarray  # (N, 3) float64 aligned onto the GT, m
    truth: np.ndarray  # (N, 3) float64, m
    probes: Probes
    tensors: tuple[Tensor, Tensor, Tensor]  # a, index (int64), dist on the device
    scale: float  # m, the mean distance of the GT points to the first camera's centre

    @classmethod
    def load(cls, name: str, cfg: dict, device: str) -> "KeepScene":
        """Load a scene's cache, building features and probes on first use.

        `cfg` is a config's root, views, backbone, r and miss (see `Probes.build`).
        """
        features, valid, pred, truth, scale = scene_arrays(name, cfg)
        folder = Path(cfg["root"]) / name
        r, miss = cfg["r"], cfg["miss"]
        probe_file = (
            folder / f"e84a_probes_r{r}_{miss}_{cfg['backbone']}_{cfg['views']}.npz"
        )
        if not probe_file.is_file():
            Probes.build(pred, truth, r, miss).save(probe_file)
        probes = Probes.load(probe_file)
        assert len(pred) == len(probes.a)
        return cls(
            name,
            features.to(device),
            torch.from_numpy(valid).to(device),
            pred,
            truth,
            probes,
            probe_tensors(probes, device),
            scale,
        )

    @classmethod
    def views(
        cls, name: str, cfg: dict, which: Sequence[int], device: str
    ) -> list["KeepScene"]:
        """Return views `which` of a scene as single-view samples, probes cached.

        The scene's probes are never built: each view's are (its GT points, R, miss),
        one file per scene holding all views.
        """
        arrays = scene_arrays(name, cfg)
        _, valid, pred, truth, _ = arrays
        counts = valid.reshape(len(valid), -1).sum(1)
        ends = np.cumsum(counts)
        folder = Path(cfg["root"]) / name
        r, miss = cfg["r"], cfg["miss"]
        file = (
            folder
            / f"e84r_view_probes_r{r}_{miss}_{cfg['backbone']}_{cfg['views']}.npz"
        )
        if not file.is_file():
            parts = {}
            for i in range(len(valid)):
                part = slice(int(ends[i] - counts[i]), int(ends[i]))
                p = Probes.build(pred[part], truth[part], r, miss)
                parts |= {f"a{i}": p.a, f"index{i}": p.index, f"dist{i}": p.dist}
            np.savez(file, **parts)
        out = []
        with np.load(file) as f:
            for i in which:
                probes = Probes(f[f"a{i}"], f[f"index{i}"], f[f"dist{i}"])
                out.append(single_view(name, arrays, i, probes).to(device))
        return out

    def to(self, device: str) -> "KeepScene":
        """Return the scene with its tensors on the device."""
        return replace(
            self,
            features=self.features.to(device),
            valid=self.valid.to(device),
            tensors=tuple(t.to(device) for t in self.tensors),
        )


def single_view(
    name: str,
    arrays: tuple[Tensor, np.ndarray, np.ndarray, np.ndarray, float],
    i: int,
    probes: Probes,
) -> KeepScene:
    """Return view i of a scene's `scene_arrays` alone: its pixels, GT points, probes.

    The ladder's single sample. The alignment is the whole scene's; a view's loss has no
    other view to cover a dropped point, so its optimum keeps more than the scene's.
    """
    features, valid, pred, truth, scale = arrays
    counts = valid.reshape(len(valid), -1).sum(1)
    start = int(counts[:i].sum())
    part = slice(start, start + int(counts[i]))
    assert len(probes.a) == counts[i], (len(probes.a), counts[i])
    return KeepScene(
        f"{name}_view{i}",
        features[i : i + 1].clone(),
        torch.from_numpy(valid[i : i + 1]),
        pred[part],
        truth[part],
        probes,
        probe_tensors(probes, "cpu"),
        scale,
    )


def probe_tensors(p: Probes, device: str) -> tuple[Tensor, Tensor, Tensor]:
    """Return a scene's constants a, index (int64) and dist on the device."""
    return (
        torch.from_numpy(p.a).to(device),
        torch.from_numpy(p.index.astype(np.int64)).to(device),
        torch.from_numpy(p.dist).to(device),
    )


def scene_arrays(
    name: str, cfg: dict
) -> tuple[Tensor, np.ndarray, np.ndarray, np.ndarray, float]:
    """Return a scene's features (V, 66, H, W), valid (V, H, W), predictions, GT, scale.

    Features are captured from Déjà View on first use and cached next to the scene.
    """
    views, backbone = cfg["views"], cfg["backbone"]
    folder = Path(cfg["root"]) / name
    gt = GroundTruth.load(folder / f"gt_{views}.npz")
    v = Views.load(folder / f"{views}.npz")
    geometry = Geometry.load(folder / f"{backbone}_{views}.npz")
    truth = ground_truth_cloud(
        gt.depth, gt.instance_ids, gt.classes, v.intrinsics, v.cam_to_world
    )
    scene = scene_from_geometry(geometry)
    points = points_from_rays(scene.depth.double(), scene.rays.double()).numpy()
    pred, _, _ = align_output(truth, points)
    feature_file = folder / f"e84a_features_{backbone}_{views}.npy"
    if not feature_file.is_file():
        steps = int(backbone.removeprefix("dejaview_k"))
        np.save(feature_file, capture_features(steps, v, geometry))
    features = torch.from_numpy(np.load(feature_file))
    assert truth.pixels is not None and len(pred) == int(truth.pixels.sum())
    assert features.shape[0] == len(v)
    assert features.shape[2:] == truth.pixels.shape[1:]
    scale = np.linalg.norm(truth.points - v.cam_to_world[0][:3, 3], axis=-1).mean()
    return features, truth.pixels, pred, truth.points, float(scale)


def standardize(
    scenes: list[KeepScene], reference: list[KeepScene] | None = None
) -> None:
    """Give each of the 66 feature channels zero mean, unit variance over `reference`.

    A fixed affine map of the input, not a learned normalization: the channels of
    Déjà View's decoder differ in scale and offset by over 40 x (std 0.11-4.6), which
    stalls a head trained on them raw (E84n: 1 000 pixels do not overfit raw, do
    standardized). The statistics are those of `reference` (default `scenes`: the
    training scenes), applied to every scene in `scenes`.
    """
    assert scenes, "no scenes"
    fit = scenes if reference is None else reference
    count = sum(len(s.features) * s.features[0, 0].numel() for s in fit)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    total = torch.zeros(FEATURES, dtype=torch.float64, device=device)
    for s in fit:
        for view in s.features:
            total += view.to(device).double().sum((1, 2))
    mean = (total / count)[:, None, None]
    square = torch.zeros_like(total)
    for s in fit:
        for view in s.features:
            square += ((view.to(device).double() - mean) ** 2).sum((1, 2))
    std = (square / count).sqrt().clamp_min(1e-6)[:, None, None]
    for s in scenes:
        out = torch.empty_like(s.features)
        for i, view in enumerate(s.features):
            out[i] = ((view.to(device).double() - mean) / std).half().to(out.device)
        s.features = out


def capture_features(steps: int, views: Views, geometry: Geometry) -> np.ndarray:
    """Re-run the backbone once and keep what its depth decoder's output_block reads.

    Returns (V, 66, H, W) float16; asserts the re-run's depth is the cached geometry's.
    """
    grabbed: list[Tensor] = []

    def forward(model: nn.Module, images: Tensor) -> dict:
        hook = model.depth_decoder.output_block.register_forward_pre_hook(
            lambda _, args: grabbed.append(args[0].detach().half().cpu())
        )
        try:
            return model({DataField.IMAGES: images})
        finally:
            hook.remove()

    again = DejaViewBackbone(steps=steps).predict(views, forward)
    change = np.median(np.abs(again.depth - geometry.depth) / geometry.depth)
    assert change < DEPTH_TOLERANCE, f"the re-run's depth moved by {change:.2e}"
    out = torch.cat(grabbed).numpy()
    assert out.shape[:2] == (len(views), FEATURES), out.shape
    return out


def probabilities(head: KeepNet, features: Tensor, chunk: int) -> Tensor:
    """Keep probabilities (V, H, W) of all views, `chunk` views at a time, with grad."""
    return torch.cat(
        [head(features[i : i + chunk].float()) for i in range(0, len(features), chunk)]
    )


def loss_backward(  # noqa: PLR0913 - head, scene, the config's eps, chunk, weight, tail
    head: KeepNet,
    scene: KeepScene,
    eps: float,
    chunk: int,
    weight: float = 1.0,
    *,
    tail: bool = False,
) -> dict[str, float]:
    """Backpropagate `weight` x the scene's loss into the head, whole-scene exact.

    Returns L, A, C and the mean s as floats.
    """
    with torch.no_grad():
        s_map = probabilities(head, scene.features, chunk)
    s = s_map[scene.valid].clone().requires_grad_()
    loss, acc, comp = expected_chamfer(s, *scene.tensors, eps, tail=tail)
    (weight * loss).backward()
    grad = torch.zeros_like(s_map)
    grad[scene.valid] = s.grad
    for i in range(0, len(grad), chunk):
        part = head(scene.features[i : i + chunk].float())
        part.backward(grad[i : i + chunk])
    return {
        "loss": loss.item(),
        "accuracy_cm": acc.item(),
        "completeness_cm": comp.item(),
        "mean_s": s.mean().item(),
    }


def confidence_loss(
    x: Tensor, a: Tensor, alpha: float, gamma: float, quantile: float | None
) -> Tensor:
    """Return Déjà View's ConfLoss: the mean of gamma c a - alpha log c, c = 1 + exp(x).

    `a` is the normalized error; a pixel's loss is minimal at c = alpha / (gamma a),
    floor 1. With `quantile`, pixels with a loss above it are left out as in ConfLoss.
    """
    assert x.shape == a.shape and x.ndim == 1 and alpha > 0 and gamma > 0
    c = 1 + torch.exp(x)
    loss = gamma * c * a - alpha * torch.log(c)
    if quantile is not None:
        loss = loss[loss <= torch.quantile(loss.detach(), quantile)]
    return loss.mean()


def mimic_loss(x: Tensor, target: Tensor) -> Tensor:
    """Return the mean cross-entropy of s = sigmoid(x) against a soft target in [0, 1].

    Minimal at s = target, so the head's ranking is learned toward the target's (E84m).
    """
    assert x.shape == target.shape and x.ndim == 1, (x.shape, target.shape)
    assert target.min() >= 0 and target.max() <= 1
    return nn.functional.binary_cross_entropy_with_logits(x, target)


def logit_backward(
    head: KeepNet,
    scene: KeepScene,
    chunk: int,
    weight: float,
    loss_of_x: Callable[[Tensor], Tensor],
) -> tuple[float, Tensor]:
    """Backpropagate `weight` x loss_of_x(x of the valid pixels) into the head.

    The loss is a function of the head's x over the whole scene; its gradient is cached
    and pushed through the head per chunk of views, as `loss_backward` does. Returns the
    loss and x.
    """
    with torch.no_grad():
        x_map = torch.cat(
            [
                head.logits(scene.features[i : i + chunk].float())
                for i in range(0, len(scene.features), chunk)
            ]
        )
    x = x_map[scene.valid].clone().requires_grad_()
    loss = loss_of_x(x)
    (weight * loss).backward()
    grad = torch.zeros_like(x_map)
    grad[scene.valid] = x.grad
    for i in range(0, len(grad), chunk):
        head.logits(scene.features[i : i + chunk].float()).backward(grad[i : i + chunk])
    return loss.item(), x.detach()


def confidence_backward(  # noqa: PLR0913, PLR0917 - head, scene, chunk, weight, the loss's constants
    head: KeepNet,
    scene: KeepScene,
    chunk: int,
    weight: float,
    alpha: float,
    gamma: float,
    quantile: float | None,
) -> dict[str, float]:
    """Backpropagate `weight` x ConfLoss of the head's x, whole-scene exact.

    The error is the point's distance to the nearest GT point (the accuracy term's a_i)
    over the scene scale. Returns the loss and the mean c and s as floats.
    """
    a = scene.tensors[0] / (CM * scene.scale)
    loss, x = logit_backward(
        head,
        scene,
        chunk,
        weight,
        lambda x: confidence_loss(x, a, alpha, gamma, quantile),
    )
    return {
        "loss": loss,
        "mean_c": (1 + torch.exp(x)).mean().item(),
        "mean_s": torch.sigmoid(x).mean().item(),
    }


def mimic_backward(
    head: KeepNet, scene: KeepScene, target: Tensor, chunk: int, weight: float
) -> dict[str, float]:
    """Backpropagate `weight` x `mimic_loss` against the valid pixels' target (N,)."""
    loss, x = logit_backward(
        head, scene, chunk, weight, lambda x: mimic_loss(x, target)
    )
    return {"loss": loss, "mean_s": torch.sigmoid(x).mean().item()}
