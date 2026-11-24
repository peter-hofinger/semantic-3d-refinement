"""E52a: learned segment adapters inside frozen Déjà View's loop (issue #52).

`--probe` (step 0 of the plan) on one train scene: (1) identity: a zero adapter through
`DejaViewBackbone.predict` equals plain Déjà View bit for bit, and the gap of the
training path at zero (bias: head dim 256 instead of 64); (2) gradients after one step:
finite and non-zero for β and B, none on a frozen weight; (3) peak GPU memory and
seconds per training step at 8, 12 and 20 views; (4) 20 steps of bias only and of pooling
only on that scene's fixed 12 views at each of --overfit-lrs, the loss at every step.

Default: every model of the config (bias, pool, both) with every segment kind (GT
instances; grid: `grid_tiles` even patch tiles per view, the control) is trained on the
train scenes, the state of least held-out error kept, then scored on the eval splits
like a backbone: the protocol (pinhole fit of its own rays, Umeyama Sim(3),
`surface_metrics`) and on Déjà View's plain cameras (as E53a), with how far the rays
turned. The grid never loses a tile to the code cap (asserted).

`--check <run>` after training: the training loss of Déjà View and of every GT model
whose kept state is not zero, per held-out and dev scene at 12 and 20 fixed views (does
the held-out gain hold per scene and at the dev scores' 20 views?), and the bedroom's
per-pixel error maps at views 18, 0, 12, 6 (`<out>/e52a_maps.npz`).

Error of one pass: the conditioned model's points x = o + D d, one Umeyama Sim(3) of a
random pixel subset onto the GT points (`align: detached`, E52a's runs, or `exact`:
differentiable, so the gradient also sees the Sim(3) move with x; E52d's gradient check
found the detached one only partly a descent direction), L1 of |S x - x_gt| / depth_gt
on every GT pixel, clipped at `max_error`. Training loss per step (E52a, second run):
one scene, a random subset of its views, the model's error over Déjà View's on the same
views and pixel subset (one extra no-grad pass), so a scene where Déjà View fails (first
run: error 0.24-0.43 against 0.01-0.06) weighs no more than one where it works; Adam's
step is invariant to that rescaling up to its epsilon. `ratio_floor` > 0 divides by
max(Déjà View's error, floor) instead, which bounds how much more a scene that Déjà View
nearly solves weighs (stability probe). Held-out error: the mean over held-out scenes of
the plain ratio on fixed views (1 at step 0, which is Déjà View).

E52d (why does every adapter diverge after 500-2000 steps?): each step records the
gradient norm per parameter and Déjà View's error, every `diag_every` steps the norm of
each parameter. One change per arm, each off by default: `autocast: false` (fp32 forward);
`weight_decay_up` (AdamW's decoupled decay on B_k alone, A's scale stays LoRA's init);
`clip_factor` (the gradient norm clipped at that multiple of the last 100 steps' median,
scale-free, as the norm's scale is not known before the run); `lr_decay: cosine`.

E52a, fourth run (`docs/experiments.md`): every eval also scores Hypersim's `val` scenes
(`val_splits`, never trained on) the same way as the held-out ones, and `select` names the
set whose least mean ratio picks the kept state (`held_out` as before, or `val`). Every
`probe_every` steps one random val scene's ratio, from its own random stream so that the
training draws stay those of E52d. Every step's losses and gradient norms and every
eval's ratios go to TensorBoard under `<out>/tb/<model>-<segments>`. `train_scenes`
replaces the train splits by named scenes for the ladder's overfit rungs (bedroom, dev),
with `held_out: 0`.

E52b (LoRA, issue #52): a model's third flag adds LoRA of rank `lora_rank` to the shared
loop block's attention linears (`src/backbone/adapters.py`), trained at `lr_lora`.
Segment kind `none` gives every token no segment, so LoRA alone with `none` is plain
fine-tuning of Déjà View on Hypersim (E52b's control).

E87a (gated pooling, issue #87): a model's optional fourth flag adds the per-token gate
sigmoid(w_k · x + b_k) on the pooling branch, w_k and b_k trained at `lr_gate`; three-flag
models (all earlier runs) have none.

uv run python experiments/scripts/e52a_segment_adapters.py --cfg configs/e52a_adapters.yaml --probe --out runs/e52a_probe
uv run python experiments/scripts/e52a_segment_adapters.py --cfg configs/e52a_adapters.yaml --out runs/e52a
uv run python experiments/scripts/e52a_segment_adapters.py --cfg configs/e52a_adapters.yaml --set steps=500 eval_every=50 "segments=[gt]" "models={pool: [false, true, false]}" lr_pool=1e-5 --out runs/e52a_sweep_1e-5
uv run python experiments/scripts/e52a_segment_adapters.py --cfg configs/e52a_adapters.yaml --check runs/e52a --out runs/e52a_check
writes `<out>/e52a.yaml` (records, config, git hash) and, for the default, every
model's adapter and held-out curve `<out>/<model>-<segments>.pt`.
"""

import argparse
import contextlib
import copy
import dataclasses
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter

from src.analysis.scene_eval import git_hash, input_paths
from src.backbone.adapters import (
    PATCH_SIZE,
    SegmentAdapter,
    conditioned,
    forward,
    grid_segments,
    patch_segments,
    segment_codes,
)
from src.backbone.dejaview import DejaViewBackbone
from src.evaluation.labeled import (
    LabeledCloud,
    align_output,
    ground_truth_cloud,
    lift,
    surface_metrics,
    umeyama,
    umeyama_torch,
)
from src.geometry import Geometry
from src.loaders.ground_truth import GroundTruth
from src.loaders.hypersim.config import load_config as hypersim_config
from src.loaders.views import Views
from src.refinement.multiview import points_from_rays, scene_from_geometry
from src.segmentation.base import NO_INSTANCE

PROBE_VIEWS = (8, 12, 20)
OVERFIT_STEPS = 20
CLIP_WINDOW = 100  # gradient norms the clip limit's median is taken over (E52d)
BASE = "Déjà View"
CHECK_VIEWS = (12, 20)  # the training runs' held-out views, the dev scores' views
MAP_SCENE, MAP_VIEWS = "ai_006_007_cam_00", (18, 0, 12, 6)  # bedroom, issue #7


@dataclasses.dataclass(frozen=True)
class Scene:
    """One scene on the CPU: images, GT depth and points, GT patch segments."""

    name: str
    images: np.ndarray  # (V, H, W, 3) uint8
    depth: np.ndarray  # (V, H, W) float32, 0 without GT
    points: np.ndarray  # (V, H, W, 3) float32 GT world points
    segments: np.ndarray  # (V, P) GT instance per patch, -1 none


def load_scene(raw: dict, name: str) -> Scene:
    """Load one scene's views and GT, the GT lifted through the GT cameras."""
    paths = input_paths(raw, name)
    views, gt = Views.load(paths["views"]), GroundTruth.load(paths["gt"])
    points = np.stack(
        [
            lift(d, k, c2w)
            for d, k, c2w in zip(
                gt.depth, views.intrinsics, views.cam_to_world, strict=True
            )
        ]
    ).astype(np.float32)
    return Scene(name, views.images, gt.depth, points, patch_segments(gt.instance_ids))


def segments_of(scene: Scene, kind: str, sel: np.ndarray, cfg: dict) -> np.ndarray:
    """(len(sel), P) segment ids of the chosen views: GT instances, grid or none."""
    assert kind in ("gt", "grid", "none"), kind
    if kind == "gt":
        return scene.segments[sel]
    h, w = scene.images.shape[1:3]
    if kind == "none":
        return np.full((len(sel), (h // PATCH_SIZE) * (w // PATCH_SIZE)), NO_INSTANCE)
    tiles = tuple(cfg["grid_tiles"])
    return grid_segments(len(sel), h // PATCH_SIZE, w // PATCH_SIZE, tiles)


def codes_of(segments: np.ndarray, kind: str) -> tuple[Tensor, int]:
    """Segment codes and how many segments were dropped; the grid drops none."""
    codes, dropped = segment_codes(segments)
    assert kind == "gt" or dropped == 0, f"grid lost {dropped} tiles to the code cap"
    return codes, dropped


def step_loss(
    net: torch.nn.Module,
    adapter: SegmentAdapter,
    scene: Scene,
    sel: np.ndarray,
    kind: str,
    cfg: dict,
    seed: int,
) -> tuple[Tensor, int]:
    """Return one pass's error on the chosen views (pixel subset from seed), drops."""
    device = next(net.parameters()).device
    images = torch.from_numpy(scene.images[sel]).permute(0, 3, 1, 2).float() / 255
    codes, dropped = codes_of(segments_of(scene, kind, sel, cfg), kind)
    with torch.autocast(
        device.type,
        dtype=torch.bfloat16,
        enabled=device.type == "cuda" and cfg["autocast"],
    ):
        out = forward(net, images[None].to(device), adapter, codes)
    pred = out["world_points"][0].float()
    valid = torch.from_numpy(scene.depth[sel] > 0).to(device)
    x = pred[valid]
    y = torch.from_numpy(scene.points[sel]).to(device)[valid]
    depth = torch.from_numpy(scene.depth[sel]).to(device)[valid]
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(x), size=min(cfg["umeyama_pixels"], len(x)), replace=False)
    assert cfg["align"] in ("detached", "exact"), cfg["align"]
    if cfg["align"] == "exact":
        at = torch.from_numpy(pick).to(device)
        s, r, t = umeyama_torch(x[at].double(), y[at].double())
        r_t, t = (s * r.T).to(x.dtype), t.to(x.dtype)
    else:
        s, r, t = umeyama(
            x[pick].detach().double().cpu().numpy(), y[pick].double().cpu().numpy()
        )
        r_t = torch.as_tensor(s * r.T, dtype=x.dtype, device=device)
        t = torch.as_tensor(t, dtype=x.dtype, device=device)
    aligned = x @ r_t + t
    err = (aligned - y).norm(dim=-1) / depth
    assert torch.isfinite(err).all()
    return err.clamp(max=cfg["max_error"]).mean(), dropped


def fixed_error(
    net: torch.nn.Module,
    adapter: SegmentAdapter,
    scenes: list[Scene],
    kind: str,
    cfg: dict,
) -> list[float]:
    """Error per held-out or val scene on its fixed views, no grad."""
    assert all(len(s.images) == 20 for s in scenes), "eval scenes need 20 views"
    sel = np.linspace(0, 19, cfg["held_out_views"]).round().astype(int)
    with torch.no_grad():
        return [
            step_loss(net, adapter, sc, sel, kind, cfg, 0)[0].item() for sc in scenes
        ]


def train(
    net: torch.nn.Module,
    adapter: SegmentAdapter,
    train_scenes: list[Scene],
    sets: dict[str, list[Scene]],
    kind: str,
    cfg: dict,
    writer: SummaryWriter,
) -> dict:
    """AdamW, a learning rate per module, loss over Déjà View's; least `select` kept."""
    assert cfg["select"] in sets, (cfg["select"], list(sets))
    assert not cfg["probe_every"] or "val" in sets
    rng = np.random.default_rng(cfg["seed"])
    probe_rng = np.random.default_rng(cfg["seed"] + 1)  # apart: training draws as E52d
    zero = SegmentAdapter(bias=True, pool=False, rank=cfg["rank"])
    named = {"beta": [adapter.beta], "up": [adapter.up],
             "down": [None if adapter.down is None else adapter.down.weight],
             "lora_down": list((adapter.lora_down or {}).values()),
             "lora_up": list((adapter.lora_up or {}).values()),
             "gate": [adapter.gate_w, adapter.gate_b]}  # fmt: skip
    lrs = {"beta": cfg["lr_bias"], "up": cfg["lr_pool"], "down": cfg["lr_pool"],
           "lora_down": cfg["lr_lora"], "lora_up": cfg["lr_lora"],
           "gate": cfg["lr_gate"]}  # fmt: skip
    groups = [
        {"params": ps, "name": n, "weight_decay": cfg["weight_decay_up"] * (n == "up"),
         "lr": lrs[n]}
        for n, ps in named.items() if ps and ps[0] is not None
    ]  # fmt: skip
    opt = torch.optim.AdamW(groups)
    decay = torch.optim.lr_scheduler.LambdaLR(opt, lambda it: lr_factor(it, cfg))
    assert adapter.is_zero()  # so the reference is Déjà View's own error, bit for bit
    reference = {k: fixed_error(net, adapter, s, kind, cfg) for k, s in sets.items()}
    per_scene = {k: {0: r} for k, r in reference.items()}
    curve: dict[str, dict[int, float]] = {k: {0: 1.0} for k in sets}
    probes: dict[int, float] = {}
    best, best_state = 1.0, copy.deepcopy(adapter.state_dict())
    t0, losses, dropped = time.perf_counter(), [], 0
    diag: dict = {"base": [], "scene": [], "grad": {g["name"]: [] for g in groups}, "clipped": 0,
                  "norm": {g["name"]: {} for g in groups}}  # fmt: skip
    totals: list[float] = []
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    for it in range(1, cfg["steps"] + 1):
        scene = train_scenes[rng.integers(len(train_scenes))]
        lo, hi = cfg["view_subset"]
        k = min(int(rng.integers(lo, hi + 1)), len(scene.images))
        sel = np.sort(rng.choice(len(scene.images), size=k, replace=False))
        seed = int(rng.integers(2**31))
        with torch.no_grad():
            base = step_loss(net, zero, scene, sel, kind, cfg, seed)[0]
        error, d = step_loss(net, adapter, scene, sel, kind, cfg, seed)
        loss = error / base.clamp(min=cfg["ratio_floor"])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        for g in groups:
            diag["grad"][g["name"]].append(group_norm(g["params"], grad=True))
        total = float(np.linalg.norm([diag["grad"][g["name"]][-1] for g in groups]))
        limit = clip_limit(totals, cfg)
        if total > limit:
            for g in groups:
                for p in g["params"]:
                    p.grad.mul_(limit / total)
            diag["clipped"] += 1
        totals.append(total)
        opt.step()
        decay.step()
        losses.append(loss.item())
        diag["base"].append(base.item())
        diag["scene"].append(scene.name)
        writer.add_scalar("train/ratio", losses[-1], it)
        writer.add_scalar("train/dv_error", diag["base"][-1], it)
        for g in groups:
            writer.add_scalar(f"grad_norm/{g['name']}", diag["grad"][g["name"]][-1], it)
        if it % cfg["diag_every"] == 0:
            for g in groups:
                diag["norm"][g["name"]][it] = group_norm(g["params"], grad=False)
                writer.add_scalar(f"norm/{g['name']}", diag["norm"][g["name"]][it], it)
        if cfg["probe_every"] and it % cfg["probe_every"] == 0:
            i = int(probe_rng.integers(len(sets["val"])))
            e = fixed_error(net, adapter, [sets["val"][i]], kind, cfg)[0]
            probes[it] = e / reference["val"][i]
            writer.add_scalar("val/probe_ratio", probes[it], it)
        dropped += d
        if it % cfg["eval_every"] == 0:
            for k, s in sets.items():
                per_scene[k][it] = fixed_error(net, adapter, s, kind, cfg)
                curve[k][it] = float(np.mean(np.array(per_scene[k][it]) / reference[k]))
                writer.add_scalar(f"{k}/ratio", curve[k][it], it)
            if curve[cfg["select"]][it] < best:
                best = curve[cfg["select"]][it]
                best_state = copy.deepcopy(adapter.state_dict())
            print(f"  step {it}: train {np.mean(losses[-cfg['eval_every']:]):.5f}, "
                  + ", ".join(f"{k} {c[it]:.5f}" for k, c in curve.items())
                  + f" ({time.perf_counter() - t0:.0f} s)", flush=True)  # fmt: skip
    writer.flush()
    adapter.load_state_dict(best_state)
    chosen = curve[cfg["select"]]
    return {
        "held_out": curve.get("held_out", {}),
        "held_out_scenes": [s.name for s in sets.get("held_out", [])],
        "held_out_error": per_scene.get("held_out", {}),
        "val": curve.get("val", {}),
        "val_scenes": [s.name for s in sets.get("val", [])],
        "val_error": per_scene.get("val", {}),
        "val_probe": probes,
        "selected_on": cfg["select"],
        "best_step": min(chosen, key=chosen.get),
        "seconds": time.perf_counter() - t0,
        "peak_gb": peak_gb(),
        "segments_dropped": dropped,
        "train_loss": losses,
        "diagnostics": diag,
    }


def group_norm(params: list[Tensor], *, grad: bool) -> float:
    """Euclidean norm of a parameter group's gradients or values, all tensors joined."""
    norms = [(p.grad if grad else p.detach()).norm().item() for p in params]
    return float(np.linalg.norm(norms))


def lr_factor(it: int, cfg: dict) -> float:
    """Learning-rate factor after `it` steps: 1, or cosine from 1 to 0 over the run."""
    assert cfg["lr_decay"] in ("constant", "cosine"), cfg["lr_decay"]
    if cfg["lr_decay"] == "constant":
        return 1.0
    return 0.5 * (1 + float(np.cos(np.pi * min(it, cfg["steps"]) / cfg["steps"])))


def clip_limit(totals: list[float], cfg: dict) -> float:
    """Gradient-norm limit: `clip_factor` x the median of the last 100 norms, or none."""
    if cfg["clip_factor"] is None or len(totals) < CLIP_WINDOW:
        return float("inf")
    assert cfg["clip_factor"] > 1
    return cfg["clip_factor"] * float(np.median(totals[-CLIP_WINDOW:]))


def score(geometry: Geometry, truth: LabeledCloud) -> tuple[dict, np.ndarray]:
    """Score one output as the protocol (cm), as E53a; its per-pixel error map (m)."""
    scene = scene_from_geometry(geometry)
    points = points_from_rays(scene.depth, scene.rays).numpy()
    pred, scale, _ = align_output(truth, points)
    e = np.linalg.norm(pred - truth.points, axis=1)
    assert np.isfinite(e).all()
    surface = surface_metrics(pred, truth, [])
    rec = {
        k: float(surface[k]) for k in ("accuracy_cm", "completeness_cm", "chamfer_cm")
    }
    err = np.full(truth.pixels.shape, np.nan, dtype=np.float32)
    err[truth.pixels] = e
    return rec | {"error_cm": 100 * float(e.mean()), "scale": float(scale)}, err


def on_plain_cameras(out: Geometry, plain: Geometry) -> tuple[Geometry, float]:
    """Put the output's depth and confidence on the plain cameras; its rays' turn."""
    a, b = out.ray_directions, plain.ray_directions
    assert a is not None
    assert b is not None
    cos = (a * b).sum(-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1))
    turn = float(np.degrees(np.median(np.arccos(np.clip(cos, -1, 1)))))
    return dataclasses.replace(plain, depth=out.depth, confidence=out.confidence), turn


def scenes_of(cfg: dict, raw: dict, splits: list[str]) -> list[str]:
    """Scene folders of the splits, without the skipped ones."""
    config = hypersim_config(Path(raw["dataset"]))
    names = [f"{n}_{c}" for n, c in config.scenes(splits).items()]
    return [n for n in names if n not in cfg["skip"]]


def setup(cfg: dict) -> tuple[DejaViewBackbone, torch.nn.Module]:
    """Load the backbone and freeze its network."""
    backbone = DejaViewBackbone()
    model, _ = backbone._load()  # noqa: SLF001
    net = model.model.eval().requires_grad_(requires_grad=False)
    torch.manual_seed(cfg["seed"])
    return backbone, net


def adapter_for(cfg: dict, model: str, device: torch.device) -> SegmentAdapter:
    """Build a zero adapter of the named model; flags (bias, pool, lora[, gate])."""
    flags = cfg["models"][model]
    assert len(flags) in (3, 4), flags
    bias, pool, lora, gate = (*flags, False)[:4]
    lora_rank = cfg["lora_rank"] if lora else 0
    adapter = SegmentAdapter(
        bias=bias, pool=pool, rank=cfg["rank"], lora=lora_rank, gate=gate
    )
    return adapter.to(device)


def peak_gb() -> float:
    """Peak GPU memory since the last reset, GB."""
    return torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0


def probe(args: argparse.Namespace, cfg: dict, raw: dict) -> dict:
    """Step 0; see the module docstring."""
    backbone, net = setup(cfg)
    device = next(net.parameters()).device
    name = args.scene or scenes_of(cfg, raw, cfg["train_splits"])[0]
    scene = load_scene(raw, name)
    views = Views.load(input_paths(raw, name)["views"])
    rec: dict = {"scene": name}

    codes, _ = codes_of(scene.segments, "gt")
    plain = backbone.predict(views)
    zero = adapter_for(cfg, "both", device)
    same = backbone.predict(views, conditioned(zero, codes))
    rec["identity_bit_for_bit"] = all(
        np.array_equal(getattr(plain, f), getattr(same, f))
        for f in ("depth", "confidence", "ray_origins", "ray_directions")
    )
    images = torch.from_numpy(scene.images).permute(0, 3, 1, 2).float()[None] / 255
    with torch.autocast(
        device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
    ):
        ref = net.forward_inference(images.to(device))["depth"].float()
        for model in cfg["models"]:
            out = forward(
                net, images.to(device), adapter_for(cfg, model, device), codes
            )
            gap = (out["depth"].float() - ref).abs() / ref
            rec[f"training_path_at_zero_{model}"] = {
                "max_rel": float(gap.max()), "median_rel": float(gap.median())}  # fmt: skip
    print(rec, flush=True)

    rng = np.random.default_rng(cfg["seed"])
    for v in args.probe_views:
        adapter = adapter_for(cfg, "both", device)
        sel = np.sort(rng.choice(len(scene.images), size=v, replace=False))
        seconds = []
        for _ in range(3):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            loss, _ = step_loss(net, adapter, scene, sel, "gt", cfg, 0)
            loss.backward()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            seconds.append(time.perf_counter() - t0)
        grads = {n: p.grad for n, p in adapter.named_parameters()}
        rec[f"views_{v}"] = {
            "peak_gb": peak_gb(),
            "seconds_per_step": float(np.median(seconds)),
            "grads_finite": all(torch.isfinite(g).all().item() for g in grads.values()),
            "grad_abs_mean": {n: float(g.abs().mean()) for n, g in grads.items()},
            "frozen_without_grad": all(p.grad is None for p in net.parameters()),
        }
        print(f"views {v}: {rec[f'views_{v}']}", flush=True)

    sel = np.linspace(0, 19, args.overfit_views).round().astype(int)
    rec["overfit"] = {}
    for model in ("bias", "pool"):
        for lr in args.overfit_lrs:
            adapter = adapter_for(cfg, model, device)
            opt = torch.optim.AdamW(adapter.parameters(), lr=lr, weight_decay=0.0)
            losses = []
            for _ in range(OVERFIT_STEPS + 1):
                loss, _ = step_loss(net, adapter, scene, sel, "gt", cfg, 0)
                losses.append(loss.item())
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
            rec["overfit"][f"{model} lr={lr:g}"] = losses
            print(f"overfit {model} lr={lr:g}: "
                  + " ".join(f"{x:.5f}" for x in losses[::4]), flush=True)  # fmt: skip
    return rec


def run(args: argparse.Namespace, cfg: dict, raw: dict) -> dict:
    """Train every model and segment kind, then score on the eval splits."""
    backbone, net = setup(cfg)
    device = next(net.parameters()).device
    names = cfg["train_scenes"] or scenes_of(cfg, raw, cfg["train_splits"])
    loaded = []
    for n in names:
        try:
            loaded.append(load_scene(raw, n))
        except FileNotFoundError as err:  # E41b: 15 scenes were never prepared
            print(f"{n}: skipped, missing {Path(str(err.filename)).name}", flush=True)
    cut = len(loaded) - cfg["held_out"]
    train_scenes, held = loaded[:cut], loaded[cut:]
    val_names = scenes_of(cfg, raw, cfg["val_splits"]) if cfg["val_splits"] else []
    assert not set(val_names) & set(names), "val scenes in the training set"
    val = [load_scene(raw, n) for n in val_names]
    sets = {k: s for k, s in (("held_out", held), ("val", val)) if s}
    print(f"{len(train_scenes)} train, {len(held)} held-out, {len(val)} val scenes, "
          f"kept on {cfg['select']}", flush=True)  # fmt: skip
    rec: dict = {"train_scenes": [s.name for s in loaded], "models": {}}
    adapters = {}
    for model in cfg["models"]:
        for kind in cfg["segments"]:
            label = f"{model}-{kind}"
            print(label, flush=True)
            adapter = adapter_for(cfg, model, device)
            with SummaryWriter(args.out / "tb" / label) as writer:
                rec["models"][label] = train(
                    net, adapter, train_scenes, sets, kind, cfg, writer
                )
            torch.save({"state": adapter.state_dict()} | rec["models"][label],
                       args.out / f"{label}.pt")  # fmt: skip
            adapters[label] = adapter
    del loaded, train_scenes, held, val, sets

    rec["eval"] = {}
    for name in scenes_of(cfg, raw, args.splits):
        paths = input_paths(raw, name)
        views, gt = Views.load(paths["views"]), GroundTruth.load(paths["gt"])
        truth = ground_truth_cloud(
            gt.depth, gt.instance_ids, gt.classes, views.intrinsics, views.cam_to_world
        )
        segments = {"gt": patch_segments(gt.instance_ids)}
        for kind in ("grid", "none"):
            segments[kind] = segments_of(
                Scene(name, views.images, gt.depth, np.empty(0), segments["gt"]),
                kind, np.arange(len(views)), cfg)  # fmt: skip
        plain = backbone.predict(views)
        scene_rec = {BASE: score(plain, truth)[0] | {"ray_deg": 0.0}}
        for label, adapter in adapters.items():
            kind = label.split("-")[1]
            codes, _ = codes_of(segments[kind], kind)
            with torch.no_grad():
                out = backbone.predict(views, conditioned(adapter, codes))
            fixed, turn = on_plain_cameras(out, plain)
            scene_rec[label] = score(out, truth)[0] | {"ray_deg": turn} | {
                f"plain_cameras_{k}": v for k, v in score(fixed, truth)[0].items()}  # fmt: skip
            print(f"{name} {label}: Chamfer {scene_rec[label]['chamfer_cm']:.3f} "
                  f"(DV {scene_rec[BASE]['chamfer_cm']:.3f})", flush=True)  # fmt: skip
        rec["eval"][name] = scene_rec
    return rec


def check(args: argparse.Namespace, cfg: dict, raw: dict) -> dict:
    """Training loss of Déjà View and each trained model per scene; bedroom maps."""
    backbone, net = setup(cfg)
    device = next(net.parameters()).device
    adapters = {}
    for path in sorted(args.check.glob("*-gt.pt")):
        adapter = adapter_for(cfg, path.stem.split("-")[0], device)
        adapter.load_state_dict(torch.load(path, map_location=device)["state"])
        if not adapter.is_zero():
            adapters[path.stem] = adapter
    zero = adapter_for(cfg, "bias", device)
    held = scenes_of(cfg, raw, cfg["train_splits"])[-cfg["held_out"] :]
    rec: dict = {"loss": {}}
    for split, names in (("held_out", held), ("dev", scenes_of(cfg, raw, ["dev"]))):
        for name in names:
            scene = load_scene(raw, name)
            for v in CHECK_VIEWS:
                sel = np.linspace(0, 19, v).round().astype(int)
                row = {}
                for label, adapter in ({BASE: zero} | adapters).items():
                    with torch.no_grad():
                        loss = step_loss(net, adapter, scene, sel, "gt", cfg, 0)
                    row[label] = loss[0].item()
                rec["loss"].setdefault(split, {}).setdefault(name, {})[v] = row
                print(f"{split} {name} {v} views: "
                      + ", ".join(f"{k} {x:.5f}" for k, x in row.items()), flush=True)  # fmt: skip
    paths = input_paths(raw, MAP_SCENE)
    views, gt = Views.load(paths["views"]), GroundTruth.load(paths["gt"])
    truth = ground_truth_cloud(
        gt.depth, gt.instance_ids, gt.classes, views.intrinsics, views.cam_to_world
    )
    codes, _ = codes_of(patch_segments(gt.instance_ids), "gt")
    maps = {BASE: score(backbone.predict(views), truth)[1]}
    for label, adapter in adapters.items():
        with torch.no_grad():
            out = backbone.predict(views, conditioned(adapter, codes))
        maps[label] = score(out, truth)[1]
    np.savez_compressed(
        args.out / "e52a_maps.npz",
        **{k: m[list(MAP_VIEWS)].astype(np.float16) for k, m in maps.items()},
    )
    return rec


def main() -> None:
    """Parse the arguments; run the probe, the training or the check."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=Path, required=True)
    parser.add_argument("--splits", nargs="*", default=["dev"], help="none: no scoring")
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--scene", help="probe scene (default: the first train scene)")
    parser.add_argument("--probe-views", type=int, nargs="+", default=list(PROBE_VIEWS))
    parser.add_argument("--overfit-views", type=int, default=12)
    parser.add_argument(
        "--overfit-lrs", type=float, nargs="+", default=[1e-4, 3e-4, 1e-3]
    )
    parser.add_argument(
        "--set", nargs="+", default=[], metavar="KEY=YAML",
        help="override config keys (smoke, lr sweep), recorded with the run")  # fmt: skip
    parser.add_argument("--check", type=Path, help="a training run's dir to check")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(args.cfg.read_text(encoding="utf-8"))
    for item in args.set:
        key, value = item.split("=", 1)
        assert key in cfg, f"unknown config key {key}"
        cfg[key] = yaml.safe_load(value)
        if isinstance(cfg[key], str):  # PyYAML (YAML 1.1) reads 1e-5 as a string
            with contextlib.suppress(ValueError):
                cfg[key] = float(cfg[key])
    raw = yaml.safe_load(Path(cfg["eval"]).read_text(encoding="utf-8"))
    assert raw["backbone"] == "dejaview_k16", raw["backbone"]
    args.out.mkdir(parents=True, exist_ok=False)
    mode = probe if args.probe else check if args.check else run
    rec = mode(args, cfg, raw)
    summary = {"args": {k: str(v) for k, v in vars(args).items()}, "git": git_hash()}
    summary |= {"config": cfg, "records": rec}
    (args.out / "e52a.yaml").write_text(
        yaml.safe_dump(summary, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
