"""Learned single-view depth refiner (issue #11): a small residual U-Net.

After BetterDepth (Zhang et al., NeurIPS 2024) and PatchRefiner (Li et al., ECCV
2024): the feed-forward model keeps the global layout and a light network adds what it
misses. Per view the input is the RGB image, the coarse log depth (median 0), the log
confidence (mean 0) and, with `semantics`, two maps from the per-view SAM 3 masks:
instance boundaries and structure (wall, floor, ceiling). The output is a residual r
with log d = log d⁰ + r; the last layer starts at zero, so an untrained refiner returns
its input bit for bit. Cameras, rays and confidence stay the backbone's.

Training follows BetterDepth's global pre-alignment: each view's coarse depth is scaled
onto the ground truth (median ratio over valid pixels) and the network fits the log
error that remains, by L1 plus an L1 on its gradients at four scales (MiDaS, Ranftl et
al., TPAMI 2020). Unlike BetterDepth there is no diffusion and unlike PatchRefiner no
high-resolution patch fusion: the cheapest version that tells whether the masks help a
learned refiner. `learned-refine` trains every model of `configs/learned.yaml` on the
Hypersim train split and writes its depth for the prediction scenes as a backbone
geometry, `<model>_<views>.npz`, which `scene_eval` scores like any other.
"""

from __future__ import annotations

import argparse
import dataclasses
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import Tensor, nn
from torch.nn import functional as F  # noqa: N812

from src.analysis.scene_eval import git_hash
from src.evaluation.labeled import STRUCTURE
from src.geometry import Geometry
from src.loaders.hypersim.config import load_config
from src.loaders.scannetpp.gt import GroundTruth
from src.loaders.views import Views
from src.segmentation.base import Segmentation

LEVELS = 4  # U-Net downsamplings; inputs are padded to a multiple of 2**LEVELS
GRAD_SCALES = 4


@dataclasses.dataclass(frozen=True)
class LearnedConfig:
    """Data, models and training settings (`configs/learned.yaml`)."""

    root: Path
    views: str
    backbone: str
    segmentation: str
    train: list[str]  # scene folders; the last `held_out` only report a loss
    held_out: int
    predict: list[str]
    models: dict[str, bool]  # name -> sees the masks
    width: int
    steps: int
    batch: int
    crop: int
    lr: float
    grad_weight: float
    seed: int

    @classmethod
    def load(cls, path: Path) -> LearnedConfig:
        """Read the config; the train scenes come from the Hypersim split it names."""
        raw = yaml.safe_load(path.read_text())
        split = load_config(Path(raw.pop("split")))
        raw["train"] = [f"{s}_{c}" for s, c in split.scenes(["train"]).items()]
        raw["root"] = Path(raw["root"])
        return cls(**raw)


def boundaries(ids: np.ndarray) -> np.ndarray:
    """1 where a 4-neighbour holds another instance id, else 0 (V, H, W)."""
    edge = np.zeros(ids.shape, dtype=bool)
    dx, dy = ids[:, :, 1:] != ids[:, :, :-1], ids[:, 1:] != ids[:, :-1]
    edge[:, :, 1:] |= dx
    edge[:, :, :-1] |= dx
    edge[:, 1:] |= dy
    edge[:, :-1] |= dy
    return edge.astype(np.float32)


def features(
    images: np.ndarray,
    depth: np.ndarray,
    confidence: np.ndarray,
    masks: Segmentation | None,
) -> np.ndarray:
    """Network input (V, C, H, W): RGB, log depth, log confidence, masks if given."""
    assert images.dtype == np.uint8 and images.shape[:3] == depth.shape
    assert np.all(depth > 0) and np.all(confidence > 0)
    log_d = np.log(depth)
    log_c = np.log(confidence)
    channels = [
        images.transpose(0, 3, 1, 2) / 255,
        (log_d - np.median(log_d, axis=(1, 2), keepdims=True))[:, None],
        (log_c - log_c.mean(axis=(1, 2), keepdims=True))[:, None],
    ]
    if masks is not None:
        structure = [i for i, c in masks.classes.items() if c in STRUCTURE]
        channels += [
            boundaries(masks.instance_ids)[:, None],
            np.isin(masks.instance_ids, structure)[:, None],
        ]
    return np.concatenate(channels, axis=1).astype(np.float32)


def log_error(depth: np.ndarray, gt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Target log(gt) - log(s d) per view, s the median gt / d, and its valid mask."""
    valid = np.isfinite(gt) & (gt > 0)
    target = np.zeros_like(depth)
    for v in range(len(depth)):
        ok = valid[v]
        scale = np.median(gt[v][ok] / depth[v][ok])
        target[v][ok] = np.log(gt[v][ok]) - np.log(scale * depth[v][ok])
    return target, valid.astype(np.float32)


def block(c_in: int, c_out: int) -> nn.Sequential:
    """Two 3x3 convolutions, each with group norm and GELU."""
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1),
        nn.GroupNorm(8, c_out),
        nn.GELU(),
        nn.Conv2d(c_out, c_out, 3, padding=1),
        nn.GroupNorm(8, c_out),
        nn.GELU(),
    )


class Refiner(nn.Module):
    """U-Net from the input maps (B, C, H, W) to a log-depth residual (B, 1, H, W)."""

    def __init__(self, in_channels: int, width: int) -> None:
        """Widths double per level; the head starts at zero (identity refiner)."""
        super().__init__()
        widths = [width * 2**i for i in range(LEVELS + 1)]
        self.down = nn.ModuleList(
            [block(in_channels, widths[0])]
            + [block(widths[i], widths[i + 1]) for i in range(LEVELS)]
        )
        self.up = nn.ModuleList(
            [
                block(widths[i + 1] + widths[i], widths[i])
                for i in reversed(range(LEVELS))
            ]
        )
        self.head = nn.Conv2d(widths[0], 1, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: Tensor) -> Tensor:
        """Pad to a multiple of 2**LEVELS, run, crop back."""
        height, width = x.shape[-2:]
        m = 2**LEVELS
        x = F.pad(x, (0, -width % m, 0, -height % m), mode="replicate")
        skips = []
        for i, down in enumerate(self.down):
            x = down(x if i == 0 else F.avg_pool2d(x, 2))
            skips.append(x)
        for up, skip in zip(self.up, reversed(skips[:-1]), strict=True):
            x = F.interpolate(x, scale_factor=2, mode="bilinear")
            x = up(torch.cat([x, skip], dim=1))
        return self.head(x)[..., :height, :width]


def loss(residual: Tensor, target: Tensor, valid: Tensor, grad_weight: float) -> Tensor:
    """L1 on the log error plus `grad_weight` times its gradient L1 on four scales."""
    err = (residual - target) * valid
    total = err.abs().sum() / valid.sum().clamp(min=1)
    for s in range(GRAD_SCALES):
        e, v = err[..., :: 2**s, :: 2**s], valid[..., :: 2**s, :: 2**s]
        gx = (e[..., 1:] - e[..., :-1]).abs() * v[..., 1:] * v[..., :-1]
        gy = (e[..., 1:, :] - e[..., :-1, :]).abs() * v[..., 1:, :] * v[..., :-1, :]
        total = total + grad_weight * (gx.sum() + gy.sum()) / v.sum().clamp(min=1)
    return total


def load_scene(
    cfg: LearnedConfig, scene: str, *, semantics: bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load one training scene: inputs, target and valid mask."""
    folder = cfg.root / scene
    views = Views.load(folder / f"{cfg.views}.npz")
    geometry = Geometry.load(folder / f"{cfg.backbone}_{cfg.views}.npz")
    gt = GroundTruth.load(folder / f"gt_{cfg.views}.npz")
    masks = (
        Segmentation.load(folder / f"{cfg.segmentation}_{cfg.views}.npz")
        if semantics
        else None
    )
    x = features(views.images, geometry.depth, geometry.confidence, masks)
    target, valid = log_error(geometry.depth, gt.depth)
    return x, target, valid


def train(
    cfg: LearnedConfig, device: torch.device, *, semantics: bool
) -> tuple[Refiner, list[dict]]:
    """Fit one refiner on random crops of the train scenes; log the held-out loss."""
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    data = [load_scene(cfg, s, semantics=semantics) for s in cfg.train]
    fit, held = data[: -cfg.held_out], data[-cfg.held_out :]
    x, target, valid = (np.concatenate(a) for a in zip(*fit, strict=True))
    model = Refiner(x.shape[1], cfg.width).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, cfg.lr, total_steps=cfg.steps)
    height, width = target.shape[1:]
    log = []
    for step in range(cfg.steps + 1):
        if step % (cfg.steps // 10) == 0:
            log.append({"step": step, **held_out_loss(model, held, cfg, device)})
            print(f"step {step}: {log[-1]}", flush=True)  # noqa: T201
        if step == cfg.steps:
            break
        idx = rng.integers(len(x), size=cfg.batch)
        i0 = rng.integers(height - cfg.crop + 1, size=cfg.batch)
        j0 = rng.integers(width - cfg.crop + 1, size=cfg.batch)
        crop = [
            np.s_[k, ..., i : i + cfg.crop, j : j + cfg.crop]
            for k, i, j in zip(idx, i0, j0, strict=True)
        ]
        xb, tb, vb = (
            torch.from_numpy(np.stack([a[c] for c in crop])).to(device)
            for a in (x, target, valid)
        )
        if rng.random() < 0.5:  # noqa: PLR2004
            xb, tb, vb = xb.flip(-1), tb.flip(-1), vb.flip(-1)
        model.train()
        value = loss(model(xb)[:, 0], tb, vb, cfg.grad_weight)
        opt.zero_grad()
        value.backward()
        opt.step()
        sched.step()
    return model, log


def held_out_loss(
    model: Refiner,
    held: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    cfg: LearnedConfig,
    device: torch.device,
) -> dict[str, float]:
    """Loss and mean absolute log error (before and after) on the held-out views."""
    model.eval()
    out = {"loss": 0.0, "abs_log_before": 0.0, "abs_log_after": 0.0}
    with torch.no_grad():
        for x, target, valid in held:
            r = predict_residual(model, x, device)
            t, v = torch.from_numpy(target), torch.from_numpy(valid)
            out["loss"] += float(loss(r, t, v, cfg.grad_weight)) / len(held)
            out["abs_log_before"] += float((t.abs() * v).sum() / v.sum()) / len(held)
            out["abs_log_after"] += float(((r - t).abs() * v).sum() / v.sum()) / len(
                held
            )
    return out


def predict_residual(model: Refiner, x: np.ndarray, device: torch.device) -> Tensor:
    """Residual (V, H, W) on the CPU, one view at a time."""
    model.eval()
    with torch.no_grad():
        return torch.cat(
            [
                model(torch.from_numpy(x[v : v + 1]).to(device))[:, 0].cpu()
                for v in range(len(x))
            ]
        )


def refine(
    model: Refiner, geometry: Geometry, x: np.ndarray, device: torch.device
) -> Geometry:
    """Return the backbone geometry with depth d⁰ · exp(r); all else kept."""
    r = predict_residual(model, x, device).numpy()
    depth = (geometry.depth * np.exp(r)).astype(np.float32)
    assert depth.shape == geometry.depth.shape and np.all(depth > 0)
    return dataclasses.replace(geometry, depth=depth)


def main() -> None:
    """Train every model of the config, save it, write its depth for `predict`."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/learned.yaml"))
    args = parser.parse_args()
    cfg = LearnedConfig.load(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path("runs") / f"learned-{time.strftime('%Y%m%d-%H%M%S')}"
    out.mkdir(parents=True)
    record: dict = {
        "config": yaml.safe_load(args.config.read_text()),
        "git": git_hash(),
    }
    for name, semantics in cfg.models.items():
        t0 = time.perf_counter()
        model, log = train(cfg, device, semantics=semantics)
        record[name] = {"seconds": time.perf_counter() - t0, "held_out": log}
        torch.save(model.state_dict(), out / f"{name}.pt")
        for scene in cfg.predict:
            folder = cfg.root / scene
            views = Views.load(folder / f"{cfg.views}.npz")
            geometry = Geometry.load(folder / f"{cfg.backbone}_{cfg.views}.npz")
            masks = (
                Segmentation.load(folder / f"{cfg.segmentation}_{cfg.views}.npz")
                if semantics
                else None
            )
            x = features(views.images, geometry.depth, geometry.confidence, masks)
            refine(model, geometry, x, device).save(folder / f"{name}_{cfg.views}.npz")
    (out / "run.yaml").write_text(yaml.safe_dump(record, sort_keys=False))
    print(f"wrote {out}")  # noqa: T201


if __name__ == "__main__":
    main()
