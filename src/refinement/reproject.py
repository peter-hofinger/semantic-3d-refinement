"""Reprojection refiner (issue #56): each view's depth refined from the other views.

The 2D refiner (`learned.py`, E36) sees one view and cannot tell which of its pixels the
other views contradict; the volumetric one (`volume.py`, E41) sees all views but only at
the grid's cell size. Here the fused cloud of all other views is projected into each
view, as multi-view stereo's cost volumes sample depth hypotheses along a pixel's ray
(MVSNet, Yao et al. 2018) and depth-map fusion checks a pixel against the other views'
points (Merrell et al. 2007), and a 2D U-Net refines the depth at full resolution.

Per pixel p of view v with depth D, each point x of another view w lands on pixel
round(K_v R_v^T (x - o_v)) at camera depth z; it votes for rho = log(z / D) of the pixel
it lands on. The votes form a soft histogram over rho in [-max_ratio, max_ratio] (each
vote split linearly between its two nearest bins, so it keeps sub-bin position), two
counts for votes in front of and behind that range (occluders, occluded surfaces), all
divided by the number of other views, and the mean log confidence of the votes in range.
With semantics a second histogram takes only the votes whose NYU40 class (SAM 3's
per-view masks or Hypersim's instances) is the pixel's own, so the network can tell
the other views' wall behind a chair from the chair itself.

Input: RGB, log depth and log confidence as in `learned.py`, plus these channels.
Output: r with D exp(r); the U-Net's last layer starts at zero (identity). Target:
`volume.py`'s r* = log(t* / D) along the pinhole ray after the evaluation's Umeyama
alignment, one similarity per scene, so the views keep a common frame (`learned.py`'s
per-view median scaling would remove the disagreement this refiner is meant to see).
Loss: the L1 of `learned.py` with its gradient term. `reproject-refine` trains every
model of the config on the Hypersim train scenes and writes `<model>_<views>.npz` for
the prediction scenes, scored by `scene_eval` like a backbone.
"""

from __future__ import annotations

import argparse
import dataclasses
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import Tensor

from src.analysis.scene_eval import git_hash
from src.device import best_device
from src.geometry import Geometry
from src.loaders.ground_truth import GroundTruth
from src.loaders.hypersim.config import load_config
from src.loaders.views import Views
from src.refinement.learned import Refiner, loss
from src.refinement.multiview import scene_from_geometry
from src.refinement.volume import SEMANTICS, pixel_classes, target_residual
from src.segmentation.base import Segmentation

MIN_DEPTH = 1e-6  # a vote behind its camera's plane is dropped


@dataclasses.dataclass(frozen=True)
class ReprojectConfig:
    """Data, models and training settings (`configs/e56a_reproject.yaml`)."""

    root: Path
    views: str
    backbone: str
    segmentation: str
    train: list[str]  # scene folders of `train_splits`; the last `held_out` only report
    held_out: int
    predict: list[str]
    models: dict[str, str]  # name -> semantics: none, sam3 or gt
    bins: int  # histogram bins over [-max_ratio, max_ratio]
    max_ratio: float
    width: int  # U-Net channels at full resolution
    steps: int
    batch: int  # target views per step
    view_subset: list[int]  # each step fuses a random number of views in this range
    lr: float
    grad_weight: float
    seeds: list[int]  # one model per seed; with several, names end in -s<seed>

    @classmethod
    def load(cls, path: Path) -> ReprojectConfig:
        """Read the config; train and prediction scenes come from Hypersim splits."""
        raw = yaml.safe_load(path.read_text())
        split = load_config(Path(raw.pop("split")))
        train = split.scenes(raw.pop("train_splits"))
        raw["train"] = [f"{s}_{c}" for s, c in train.items()]
        predict = split.scenes(raw.pop("predict_splits"))
        raw["predict"] = [f"{s}_{c}" for s, c in predict.items()]
        raw["root"] = Path(raw["root"])
        assert all(s in SEMANTICS for s in raw["models"].values()), raw["models"]
        return cls(**raw)


@dataclasses.dataclass(frozen=True)
class SceneData:
    """One scene's views on the CPU, in small dtypes; pinhole cameras in DV's frame."""

    images: Tensor  # (V, H, W, 3) uint8
    depth: Tensor  # (V, H, W) float32, camera z
    log_conf: Tensor  # (V, H, W) float16
    classes: Tensor | None  # (V, H, W) uint8, NYU40 index, 0 none; None: no semantics
    k: Tensor  # (V, 3, 3), pixel centers at integers
    rotation: Tensor  # (V, 3, 3) camera to world
    center: Tensor  # (V, 3)
    r: Tensor  # (V, H, W) float16, r* (0 where invalid)
    valid: Tensor  # (V, H, W) bool


def channels(bins: int, *, semantics: bool) -> int:
    """Input channels: RGB, log depth, log conf, votes (and same-class votes)."""
    return 5 + bins + 3 + ((bins + 2) if semantics else 0)


def world_points(depth: Tensor, k: Tensor, rotation: Tensor, center: Tensor) -> Tensor:
    """Pinhole points (V, H, W, 3): o + D R K^-1 (u, v, 1), pixel centers integer."""
    _, h, w = depth.shape
    v, u = torch.meshgrid(
        torch.arange(h, device=depth.device, dtype=depth.dtype),
        torch.arange(w, device=depth.device, dtype=depth.dtype),
        indexing="ij",
    )
    x = (u[None] - k[:, 0, 2, None, None]) / k[:, 0, 0, None, None]
    y = (v[None] - k[:, 1, 2, None, None]) / k[:, 1, 1, None, None]
    cam = torch.stack([x, y, torch.ones_like(x)], -1) * depth[..., None]
    return torch.einsum("vij,vhwj->vhwi", rotation, cam) + center[:, None, None]


def soft_histogram(
    flat: Tensor, rho: Tensor, pixels: int, bins: int, max_ratio: float
) -> Tensor:
    """Votes (pixels, bins + 2): linear soft bins over rho, then in front and behind."""
    out = torch.zeros(pixels * (bins + 2), device=rho.device)
    inside = rho.abs() <= max_ratio
    pos = (rho[inside] + max_ratio) / (2 * max_ratio) * (bins - 1)
    lo = pos.floor().clamp(max=bins - 2)
    frac = pos - lo
    base = flat[inside] * (bins + 2) + lo.long()
    out.index_add_(0, base, 1 - frac)
    out.index_add_(0, base + 1, frac)
    out.index_add_(
        0,
        flat[rho < -max_ratio] * (bins + 2) + bins,
        torch.ones_like(rho[rho < -max_ratio]),
    )
    out.index_add_(
        0,
        flat[rho > max_ratio] * (bins + 2) + bins + 1,
        torch.ones_like(rho[rho > max_ratio]),
    )
    return out.view(pixels, bins + 2)


def reprojection(
    target: int,
    sources: Tensor,
    points: Tensor,
    scene: SceneData,
    cfg: ReprojectConfig,
) -> Tensor:
    """Vote channels (C, H, W) of view `target` from the points of views `sources`."""
    bins, max_ratio = cfg.bins, cfg.max_ratio
    depth, log_conf = scene.depth[target], scene.log_conf
    h, w = depth.shape
    x = points[sources].reshape(-1, 3)
    cam = (x - scene.center[target]) @ scene.rotation[target]
    z = cam[:, 2]
    front = z > MIN_DEPTH
    k = scene.k[target]
    u = (k[0, 0] * cam[:, 0] / z.clamp(min=MIN_DEPTH) + k[0, 2]).round()
    v = (k[1, 1] * cam[:, 1] / z.clamp(min=MIN_DEPTH) + k[1, 2]).round()
    hit = front & (u >= 0) & (u < w) & (v >= 0) & (v < h)
    flat = (v[hit] * w + u[hit]).long()
    rho = torch.log(z[hit] / depth.reshape(-1)[flat])
    others = max(len(sources), 1)
    votes = soft_histogram(flat, rho, h * w, bins, max_ratio) / others
    near = rho.abs() <= max_ratio
    conf_sum = torch.zeros(h * w, device=z.device).index_add_(
        0, flat[near], log_conf[sources].reshape(-1)[hit][near].float()
    )
    conf = conf_sum / votes[:, :bins].sum(1).mul(others).clamp(min=1)
    out = [votes, conf[:, None]]
    if scene.classes is not None:
        own = scene.classes[target].reshape(-1)[flat].long()
        same = (scene.classes[sources].reshape(-1)[hit].long() == own) & (own > 0)
        out.append(
            soft_histogram(flat[same], rho[same], h * w, bins, max_ratio) / others
        )
    feats = torch.cat(out, 1).T.reshape(-1, h, w)
    assert feats.isfinite().all()
    return feats


def inputs(
    scene: SceneData,
    targets: Tensor,
    sources: Tensor,
    cfg: ReprojectConfig,
    device: torch.device,
) -> Tensor:
    """Network input (B, C, H, W) of views `targets`, voted by the other `sources`."""
    on = dataclasses.replace(scene, **{
        f.name: getattr(scene, f.name).to(device)
        for f in dataclasses.fields(scene) if getattr(scene, f.name) is not None
    })  # fmt: skip
    points = world_points(on.depth, on.k, on.rotation, on.center)
    log_d = on.depth.log()
    out = []
    for t in targets.tolist():
        log_dt = log_d[t] - log_d[t].median()
        own = [
            on.images[t].permute(2, 0, 1).float() / 255,
            log_dt[None],
            on.log_conf[t][None].float() - on.log_conf[t].float().mean(),
        ]
        others = sources[sources != t].to(device)
        votes = reprojection(t, others, points, on, cfg)
        out.append(torch.cat([*own, votes]))
    return torch.stack(out)


def load_scene(
    cfg: ReprojectConfig, scene: str, semantics: str
) -> tuple[SceneData, Geometry]:
    """One scene's views, pinhole cameras, classes and r*."""
    folder = cfg.root / scene
    views = Views.load(folder / f"{cfg.views}.npz")
    geometry = Geometry.load(folder / f"{cfg.backbone}_{cfg.views}.npz")
    gt = GroundTruth.load(folder / f"gt_{cfg.views}.npz")
    if semantics == "gt":
        ids, classes = gt.instance_ids, gt.classes
    elif semantics == "sam3":
        masks = Segmentation.load(folder / f"{cfg.segmentation}_{cfg.views}.npz")
        ids, classes = masks.instance_ids, masks.classes
    else:
        ids, classes = np.full(geometry.depth.shape, -1), {}
    pinhole = scene_from_geometry(geometry)
    rays = pinhole.rays.reshape(-1, 6).double().numpy()
    depth = geometry.depth.reshape(-1).astype(np.float64)
    r, valid = target_residual(rays, depth, views, gt)
    shape = geometry.depth.shape
    data = SceneData(
        torch.from_numpy(views.images),
        torch.from_numpy(geometry.depth.astype(np.float32)),
        torch.from_numpy(np.log(geometry.confidence)).half(),
        None
        if semantics == "none"
        else torch.from_numpy(pixel_classes(ids, classes).astype(np.uint8)),
        pinhole.k,
        pinhole.rotation,
        pinhole.center,
        torch.from_numpy(r.reshape(shape)).half(),
        torch.from_numpy(valid.reshape(shape)),
    )
    return data, geometry


def load_train(
    cfg: ReprojectConfig, semantics: str
) -> tuple[list[SceneData], list[str]]:
    """Every train scene; skipped: no pinhole rays or missing inputs."""
    data, skipped = [], []
    for scene in cfg.train:
        try:
            data.append(load_scene(cfg, scene, semantics)[0])
        except AssertionError as err:  # fit_pinhole: Deja View's rays are no pinhole
            skipped.append(f"{scene}: pinhole fit failed ({err})")
            print(f"{scene}: skipped, pinhole fit failed ({err})", flush=True)  # noqa: T201
        except FileNotFoundError as err:  # a scene whose download or caching failed
            skipped.append(f"{scene}: missing {Path(str(err.filename)).name}")
            print(f"{scene}: skipped, missing input ({err})", flush=True)  # noqa: T201
    return data, skipped


def step_loss(
    model: Refiner,
    scene: SceneData,
    cfg: ReprojectConfig,
    rng: np.random.Generator,
    device: torch.device,
) -> Tensor:
    """Loss on `batch` views of a random subset of views, voted by the rest of it."""
    n = len(scene.depth)
    k = int(rng.integers(cfg.view_subset[0], min(cfg.view_subset[1], n) + 1))
    subset = torch.from_numpy(np.sort(rng.choice(n, size=k, replace=False)))
    targets = subset[
        torch.from_numpy(rng.choice(k, size=min(cfg.batch, k), replace=False))
    ]
    x = inputs(scene, targets, subset, cfg, device)
    r = scene.r[targets].to(device).float()
    valid = scene.valid[targets].to(device).float()
    if rng.random() < 0.5:  # noqa: PLR2004
        x, r, valid = x.flip(-1), r.flip(-1), valid.flip(-1)
    return loss(model(x)[:, 0], r, valid, cfg.grad_weight)


def predict_residual(
    model: Refiner,
    scene: SceneData,
    cfg: ReprojectConfig,
    device: torch.device,
) -> Tensor:
    """Residual (V, H, W) on the CPU, each view voted by all others."""
    model.eval()
    every = torch.arange(len(scene.depth))
    with torch.no_grad():
        return torch.cat([
            model(inputs(scene, every[v : v + 1], every, cfg, device))[:, 0].cpu()
            for v in range(len(every))
        ])  # fmt: skip


def held_out_error(
    model: Refiner,
    held: list[SceneData],
    cfg: ReprojectConfig,
    device: torch.device,
) -> dict[str, float]:
    """Mean |r* - r| over valid pixels before (r = 0) and after, held-out scenes."""
    before = after = 0.0
    for scene in held:
        r = predict_residual(model, scene, cfg, device)
        t, v = scene.r.float(), scene.valid
        before += float(t[v].abs().mean()) / len(held)
        after += float((r - t)[v].abs().mean()) / len(held)
    return {"abs_log_before": before, "abs_log_after": after}


def train(
    cfg: ReprojectConfig,
    data: list[SceneData],
    device: torch.device,
    *,
    seed: int,
) -> tuple[Refiner, list[dict]]:
    """Fit one refiner, a train scene per step; keep the least held-out error state."""
    semantics = data[0].classes is not None
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    fit, held = data[: -cfg.held_out], data[-cfg.held_out :]
    model = Refiner(channels(cfg.bins, semantics=semantics), cfg.width).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, cfg.lr, total_steps=cfg.steps)
    log: list[dict] = []
    t0 = time.perf_counter()
    best, best_state = float("inf"), None
    for step in range(cfg.steps + 1):
        if step % (cfg.steps // 20) == 0:
            err = held_out_error(model, held, cfg, device)
            log.append({"step": step, "seconds": time.perf_counter() - t0, **err})
            print(f"step {step}: {log[-1]}", flush=True)  # noqa: T201
            if err["abs_log_after"] < best:
                best = err["abs_log_after"]
                best_state = {
                    k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                }
                log[-1]["best"] = True
        if step == cfg.steps:
            break
        model.train()
        scene = fit[rng.integers(len(fit))]
        value = step_loss(model, scene, cfg, rng, device)
        opt.zero_grad()
        value.backward()
        opt.step()
        sched.step()
    assert best_state is not None
    model.load_state_dict(best_state)
    return model, log


def refine(
    model: Refiner,
    scene: SceneData,
    geometry: Geometry,
    cfg: ReprojectConfig,
    device: torch.device,
) -> Geometry:
    """Return the backbone geometry with depth D exp(r); all else kept."""
    r = predict_residual(model, scene, cfg, device).numpy()
    depth = (geometry.depth * np.exp(r)).astype(np.float32)
    assert depth.shape == geometry.depth.shape
    assert np.all(np.isfinite(depth)) and np.all(depth > 0)
    return dataclasses.replace(geometry, depth=depth)


def main() -> None:
    """Train every model of the config per seed, save it, write its depth for `predict`.

    `--only` restricts the run to the named models (one SLURM task each); every model
    keeps its own record, `<name>.yaml`, so parallel tasks share a run folder. A model
    already saved there is loaded instead of trained.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/e56a_reproject.yaml")
    )
    parser.add_argument("--out", type=Path, help="run folder to create or resume")
    parser.add_argument("--only", nargs="*", help="model names (with -s<seed>)")
    args = parser.parse_args()
    cfg = ReprojectConfig.load(args.config)
    device = best_device()
    out = args.out or Path("runs") / f"reproject-{time.strftime('%Y%m%d-%H%M%S')}"
    out.mkdir(parents=True, exist_ok=True)
    for base, semantics in cfg.models.items():
        sem = semantics != "none"
        names = {s: f"{base}-s{s}" if len(cfg.seeds) > 1 else base for s in cfg.seeds}
        names = {s: n for s, n in names.items() if not args.only or n in args.only}
        if not names:
            continue
        todo = any(not (out / f"{n}.pt").exists() for n in names.values())
        data, skipped = load_train(cfg, semantics) if todo else ([], [])
        for seed, name in names.items():
            saved = out / f"{name}.pt"
            if saved.exists():
                model = Refiner(channels(cfg.bins, semantics=sem), cfg.width)
                model.load_state_dict(torch.load(saved, map_location="cpu"))
                model = model.to(device)
                print(f"{name}: loaded from {saved}")  # noqa: T201
            else:
                t0 = time.perf_counter()
                model, log = train(cfg, data, device, seed=seed)
                torch.save(model.state_dict(), saved)
                record = {
                    "config": yaml.safe_load(args.config.read_text()),
                    "git": git_hash(),
                    "seconds": time.perf_counter() - t0,
                    "skipped": skipped,
                    "held_out": log,
                }
                (out / f"{name}.yaml").write_text(
                    yaml.safe_dump(record, sort_keys=False)
                )
            for scene in cfg.predict:
                sd, geometry = load_scene(cfg, scene, semantics)
                refined = refine(model, sd, geometry, cfg, device)
                refined.save(cfg.root / scene / f"{name}_{cfg.views}.npz")
            print(f"{name}: wrote {len(cfg.predict)} scenes", flush=True)  # noqa: T201
        del data
    print(f"wrote {out}")  # noqa: T201


if __name__ == "__main__":
    main()
