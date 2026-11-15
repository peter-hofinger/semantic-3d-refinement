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

Error of one pass: the conditioned model's points x = o + D d, one Umeyama Sim(3)
(detached) of a random pixel subset onto the GT points, L1 of |S x - x_gt| / depth_gt on
every GT pixel, clipped at `max_error`. Training loss per step (E52a, second run): one
scene, a random subset of its views, the model's error over Déjà View's on the same
views and pixel subset (one extra no-grad pass), so a scene where Déjà View fails
(first run: error 0.24-0.43 against 0.01-0.06) weighs no more than one where it works;
Adam's step is invariant to that rescaling up to its epsilon. Held-out error: the mean
over held-out scenes of the same ratio on fixed views (1 at step 0, which is Déjà View).

uv run python experiments/scripts/e52a_segment_adapters.py --cfg configs/e52a_adapters.yaml --probe --out runs/e52a_probe
uv run python experiments/scripts/e52a_segment_adapters.py --cfg configs/e52a_adapters.yaml --out runs/e52a
uv run python experiments/scripts/e52a_segment_adapters.py --cfg configs/e52a_adapters.yaml --set steps=500 eval_every=50 "segments=[gt]" "models={pool: [false, true]}" lr_pool=1e-5 --out runs/e52a_sweep_1e-5
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
)
from src.geometry import Geometry
from src.loaders.ground_truth import GroundTruth
from src.loaders.hypersim.config import load_config as hypersim_config
from src.loaders.views import Views
from src.refinement.multiview import points_from_rays, scene_from_geometry

PROBE_VIEWS = (8, 12, 20)
OVERFIT_STEPS = 20
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
    """(len(sel), P) segment ids of the chosen views: GT instances or the grid."""
    if kind == "gt":
        return scene.segments[sel]
    h, w = scene.images.shape[1:3]
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
        device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
    ):
        out = forward(net, images[None].to(device), adapter, codes)
    pred = out["world_points"][0].float()
    valid = torch.from_numpy(scene.depth[sel] > 0).to(device)
    x = pred[valid]
    y = torch.from_numpy(scene.points[sel]).to(device)[valid]
    depth = torch.from_numpy(scene.depth[sel]).to(device)[valid]
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(x), size=min(cfg["umeyama_pixels"], len(x)), replace=False)
    s, r, t = umeyama(
        x[pick].detach().double().cpu().numpy(), y[pick].double().cpu().numpy()
    )
    r_t = torch.as_tensor(s * r.T, dtype=x.dtype, device=device)
    aligned = x @ r_t + torch.as_tensor(t, dtype=x.dtype, device=device)
    err = (aligned - y).norm(dim=-1) / depth
    assert torch.isfinite(err).all()
    return err.clamp(max=cfg["max_error"]).mean(), dropped


def held_out_error(
    net: torch.nn.Module,
    adapter: SegmentAdapter,
    scenes: list[Scene],
    kind: str,
    cfg: dict,
) -> list[float]:
    """Error per held-out scene on its fixed views, no grad."""
    assert all(len(s.images) == 20 for s in scenes), "held-out scenes need 20 views"
    sel = np.linspace(0, 19, cfg["held_out_views"]).round().astype(int)
    with torch.no_grad():
        return [
            step_loss(net, adapter, sc, sel, kind, cfg, 0)[0].item() for sc in scenes
        ]


def train(
    net: torch.nn.Module,
    adapter: SegmentAdapter,
    train_scenes: list[Scene],
    held: list[Scene],
    kind: str,
    cfg: dict,
) -> dict:
    """AdamW, a learning rate per module, loss over Déjà View's; least held-out kept."""
    rng = np.random.default_rng(cfg["seed"])
    zero = SegmentAdapter(bias=True, pool=False, rank=cfg["rank"])
    groups = [
        {"params": [q for q in (adapter.beta,) if q is not None], "lr": cfg["lr_bias"]},
        {"params": [q for q in (adapter.up,) if q is not None]
         + ([] if adapter.down is None else list(adapter.down.parameters())),
         "lr": cfg["lr_pool"]},
    ]  # fmt: skip
    opt = torch.optim.AdamW([g for g in groups if g["params"]], weight_decay=0.0)
    assert adapter.is_zero()  # so the reference is Déjà View's own error, bit for bit
    reference = held_out_error(net, adapter, held, kind, cfg)
    per_scene, curve = {0: reference}, {0: 1.0}
    best, best_state = 1.0, copy.deepcopy(adapter.state_dict())
    t0, losses, dropped = time.perf_counter(), [], 0
    for it in range(1, cfg["steps"] + 1):
        scene = train_scenes[rng.integers(len(train_scenes))]
        lo, hi = cfg["view_subset"]
        k = min(int(rng.integers(lo, hi + 1)), len(scene.images))
        sel = np.sort(rng.choice(len(scene.images), size=k, replace=False))
        seed = int(rng.integers(2**31))
        with torch.no_grad():
            base = step_loss(net, zero, scene, sel, kind, cfg, seed)[0]
        error, d = step_loss(net, adapter, scene, sel, kind, cfg, seed)
        loss = error / base
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        losses.append(loss.item())
        dropped += d
        if it % cfg["eval_every"] == 0:
            per_scene[it] = held_out_error(net, adapter, held, kind, cfg)
            curve[it] = float(np.mean(np.array(per_scene[it]) / reference))
            if curve[it] < best:
                best, best_state = curve[it], copy.deepcopy(adapter.state_dict())
            print(f"  step {it}: train {np.mean(losses[-cfg['eval_every']:]):.5f}, "
                  f"held-out {curve[it]:.5f} ({time.perf_counter() - t0:.0f} s)",
                  flush=True)  # fmt: skip
    adapter.load_state_dict(best_state)
    return {
        "held_out": curve,
        "held_out_scenes": [s.name for s in held],
        "held_out_error": per_scene,
        "best_step": min(curve, key=curve.get),
        "seconds": time.perf_counter() - t0,
        "segments_dropped": dropped,
    }


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
    """Build a zero adapter of the named model."""
    bias, pool = cfg["models"][model]
    return SegmentAdapter(bias=bias, pool=pool, rank=cfg["rank"]).to(device)


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
    names = scenes_of(cfg, raw, cfg["train_splits"])
    loaded = []
    for n in names:
        try:
            loaded.append(load_scene(raw, n))
        except FileNotFoundError as err:  # E41b: 15 scenes were never prepared
            print(f"{n}: skipped, missing {Path(str(err.filename)).name}", flush=True)
    held, train_scenes = loaded[-cfg["held_out"] :], loaded[: -cfg["held_out"]]
    print(f"{len(train_scenes)} train + {len(held)} held-out scenes", flush=True)
    rec: dict = {"train_scenes": [s.name for s in loaded], "models": {}}
    adapters = {}
    for model in cfg["models"]:
        for kind in cfg["segments"]:
            label = f"{model}-{kind}"
            print(label, flush=True)
            adapter = adapter_for(cfg, model, device)
            rec["models"][label] = train(net, adapter, train_scenes, held, kind, cfg)
            torch.save({"state": adapter.state_dict()} | rec["models"][label],
                       args.out / f"{label}.pt")  # fmt: skip
            adapters[label] = adapter
    del loaded, train_scenes, held

    rec["eval"] = {}
    for name in scenes_of(cfg, raw, args.splits):
        paths = input_paths(raw, name)
        views, gt = Views.load(paths["views"]), GroundTruth.load(paths["gt"])
        truth = ground_truth_cloud(
            gt.depth, gt.instance_ids, gt.classes, views.intrinsics, views.cam_to_world
        )
        segments = {"gt": patch_segments(gt.instance_ids)}
        segments["grid"] = segments_of(
            Scene(name, views.images, gt.depth, np.empty(0), segments["gt"]),
            "grid", np.arange(len(views)), cfg)  # fmt: skip
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
    parser.add_argument("--splits", nargs="+", default=["dev"])
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
