"""E53a: attention bias within segments inside Déjà View's loop (issue #53).

`--probe` (step 0 of the plan): plain Déjà View on the given scenes; at loop steps 1, 8
and 16, for frame and global attention and per head, the mean share of a labelled
patch query's attention that falls on keys of its own segment (`share`), the share a
uniform attention would give (`uniform`, segment size / sequence length), and the
median over queries of the std of its logits q k / sqrt(d) (`logit_std`). The bias
only adds beta to logits, so `logit_std` sets the scale of beta.

Grid (default): per scene Déjà View runs plainly, once with an `AttentionBias` that
biases no step (the replaced forward without a mask: must equal the plain run bit for
bit), then per attention (frame, global) and step set the same hook at beta = 0 (a
zero mask: SDPA's memory-efficient kernel instead of flash, the variants' baseline)
and at each beta. Segments (--segments): `gt` Hypersim's GT instances; `wrong` the GT
instances of the next dev scene at the same view index (the control); `sam3` the
protocol's SAM 3 masks linked on the plain run's geometry. Scoring as E51a
(`e51a_token_mix.py`): depth and confidence on the plain run's cameras and rays,
pinhole rays, one Umeyama Sim(3), `surface_metrics`, the mean per-pixel 3D error;
plus how far rays and depth moved, the forward time and the peak GPU memory.

uv run python experiments/scripts/e53a_attention_bias.py --cfg configs/eval_hypersim.yaml --probe --scenes ai_006_007_cam_00 --out runs/e53a_probe
uv run python experiments/scripts/e53a_attention_bias.py --cfg configs/eval_hypersim.yaml --segments gt --out runs/e53a_gt
writes `<out>/e53a.yaml` (every record, config, git hash), for the grid also
`<out>/e53a_maps.npz` (the bedroom's per-pixel error at views 18, 0, 12, 6), prints
the report.
"""

import argparse
import dataclasses
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import Tensor

from src.analysis.scene_eval import git_hash, input_paths
from src.backbone.dejaview import (
    WHERE,
    AttentionBias,
    DejaViewBackbone,
    patch_segments,
    token_segments,
)
from src.evaluation.labeled import (
    LabeledCloud,
    align_output,
    ground_truth_cloud,
    surface_metrics,
)
from src.geometry import Geometry
from src.loaders.ground_truth import GroundTruth
from src.loaders.hypersim.config import load_config as hypersim_config
from src.loaders.views import Views
from src.refinement.multiview import points_from_rays, scene_from_geometry
from src.segmentation.base import Segmentation
from src.segmentation.linked import working_associator

K = 16
STEP_SETS = {
    "all": frozenset(range(K)),
    "first4": frozenset(range(4)),
    "last4": frozenset(range(K - 4, K)),
}
BETAS = (0.25, 0.5, 1.0, 2.0, 4.0)
PROBE_STEPS = (0, 7, 15)
CHUNK = 256  # query rows per logit block in the probe (memory)
MAP_SCENE, MAP_VIEWS = "ai_006_007_cam_00", (18, 0, 12, 6)
BASE = "Déjà View"


def run(
    backbone: DejaViewBackbone, views: Views, bias: AttentionBias | None
) -> tuple[Geometry, float, float]:
    """One forward pass, its time in seconds and its peak GPU memory in GB."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    geometry = backbone.predict(views, bias)
    seconds = time.perf_counter() - t0
    peak = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0
    return geometry, seconds, peak


def score(geometry: Geometry, truth: LabeledCloud) -> tuple[dict, np.ndarray]:
    """Score one output as the protocol (cm); return its per-pixel error map (m)."""
    scene = scene_from_geometry(geometry)
    points = points_from_rays(scene.depth, scene.rays).numpy()
    pred, scale, _ = align_output(truth, points)
    e = np.linalg.norm(pred - truth.points, axis=1)
    assert np.isfinite(e).all()
    surface = surface_metrics(pred, truth, [])
    rec = {
        k: float(surface[k]) for k in ("accuracy_cm", "completeness_cm", "chamfer_cm")
    }
    rec |= {"error_cm": 100 * float(e.mean()), "scale": float(scale)}
    err = np.full(truth.pixels.shape, np.nan, dtype=np.float32)
    err[truth.pixels] = e
    return rec, err


def fixed_cameras(biased: Geometry, plain: Geometry) -> tuple[Geometry, dict]:
    """Biased depth and confidence on the plain cameras and rays; how far rays moved."""
    assert biased.ray_directions is not None
    assert plain.ray_directions is not None
    a, b = biased.ray_directions, plain.ray_directions
    cos = (a * b).sum(-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1))
    moved = {
        "ray_deg": float(np.degrees(np.median(np.arccos(np.clip(cos, -1, 1))))),
        "depth_rel": float(np.median(np.abs(biased.depth / plain.depth - 1))),
    }
    geometry = dataclasses.replace(
        plain, depth=biased.depth, confidence=biased.confidence
    )
    return geometry, moved


def same(a: Geometry, b: Geometry) -> bool:
    """Bit-for-bit equal depth, confidence and rays."""
    return all(
        np.array_equal(getattr(a, f), getattr(b, f))
        for f in ("depth", "confidence", "ray_origins", "ray_directions")
    )


@dataclasses.dataclass
class Probe:
    """Collects own-segment attention share and logit spread per step, attention, head."""

    ids: dict[str, Tensor]  # "frame": (V, P), "global": (1, V * P) token ids
    records: dict = dataclasses.field(default_factory=dict)

    def __call__(self, step: int, where: str, q: Tensor, k: Tensor) -> None:
        """Measure at the probe steps; q, k are (B, heads, N, d)."""
        if step not in PROBE_STEPS:
            return
        with torch.autocast(q.device.type, enabled=False):  # logits in float32
            self.measure(step, where, q, k)

    def measure(self, step: int, where: str, q: Tensor, k: Tensor) -> None:
        """Own-segment share per head, uniform share, median logit std (see module)."""
        ids = self.ids[where].to(q.device)
        heads, n, d = q.shape[1], q.shape[2], q.shape[3]
        share = torch.zeros(heads, dtype=torch.float64, device=q.device)
        uniform = torch.zeros((), dtype=torch.float64, device=q.device)
        stds, count = [], 0
        for b in range(q.shape[0]):
            kb = k[b].float()
            for start in range(0, n, CHUNK):
                rows = ids[b, start : start + CHUNK]
                labeled = rows >= 0
                if not labeled.any():
                    continue
                logits = q[b, :, start : start + CHUNK].float() @ kb.transpose(-2, -1)
                logits = logits[:, labeled] / d**0.5
                own = rows[labeled][:, None] == ids[b][None, :]
                share += (logits.softmax(-1) * own).sum(-1).sum(-1).double()
                uniform += own.sum(-1).double().sum() / n
                stds.append(logits.std(-1).flatten().cpu())
                count += int(labeled.sum())
        assert count > 0, where
        share, uniform = share / count, float(uniform / count)
        assert torch.isfinite(share).all()
        self.records.setdefault(f"step {step + 1}", {})[where] = {
            "share_mean": float(share.mean()),
            "share_min_head": float(share.min()),
            "share_max_head": float(share.max()),
            "uniform": uniform,
            "logit_std": float(torch.cat(stds).median()),
        }


def probe(args: argparse.Namespace, raw: dict, scenes: list[str]) -> None:
    """Step 0: plain Déjà View with the probe on each scene; write and print."""
    backbone = DejaViewBackbone(steps=K)
    records = {}
    for name in scenes:
        paths = input_paths(raw, name)
        views, gt = Views.load(paths["views"]), GroundTruth.load(paths["gt"])
        segments = patch_segments(gt.instance_ids)
        ids = token_segments(segments)
        p = Probe({"frame": ids, "global": ids.reshape(1, -1)})
        plain, _, _ = run(backbone, views, None)
        probed, _, _ = run(
            backbone, views, AttentionBias(segments, 0.0, frozenset(), "frame", p)
        )
        assert same(probed, plain), f"{name}: the probe changed Déjà View's output"
        records[name] = p.records
        for step, rec in p.records.items():
            for where, r in rec.items():
                print(
                    f"{name} {step:>8} {where:<6} share {r['share_mean']:.3f} "
                    f"(heads {r['share_min_head']:.3f}-{r['share_max_head']:.3f}) "
                    f"uniform {r['uniform']:.3f} "
                    f"x{r['share_mean'] / r['uniform']:.1f} "
                    f"logit std {r['logit_std']:.2f}",
                    flush=True,
                )
    write(args, raw, records)


def write(args: argparse.Namespace, raw: dict, records: dict) -> None:
    """Save every record with the arguments, config and git hash."""
    summary = {"args": {k: str(v) for k, v in vars(args).items()}, "git": git_hash()}
    summary |= {"config": raw, "scenes": records}
    (args.out / "e53a.yaml").write_text(
        yaml.safe_dump(summary, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def scene_ids(
    args: argparse.Namespace, raw: dict, scenes: list[str], i: int, plain: Geometry
) -> tuple[np.ndarray, str]:
    """Load scene i's (V, H, W) segment ids for --segments, and their source."""
    name = scenes[i]
    if args.segments == "gt":
        return GroundTruth.load(input_paths(raw, name)["gt"]).instance_ids, "gt"
    if args.segments == "wrong":
        other = scenes[(i + 1) % len(scenes)]
        assert other != name
        return GroundTruth.load(input_paths(raw, other)["gt"]).instance_ids, other
    masks = Segmentation.load(input_paths(raw, name)["masks"])
    return working_associator().associate(masks, plain).instance_ids, "sam3"


def grid(args: argparse.Namespace, raw: dict, scenes: list[str]) -> None:
    """Run the grid on the scenes; see the module docstring."""
    backbone = DejaViewBackbone(steps=K)
    records: dict = {}
    maps: dict[str, np.ndarray] = {}
    for i, name in enumerate(scenes):
        paths = input_paths(raw, name)
        views, gt = Views.load(paths["views"]), GroundTruth.load(paths["gt"])
        truth = ground_truth_cloud(
            gt.depth, gt.instance_ids, gt.classes, views.intrinsics, views.cam_to_world
        )
        run(backbone, views, None)  # warm-up: weights, kernels
        plain, t_plain, m_plain = run(backbone, views, None)
        ids, source = scene_ids(args, raw, scenes, i, plain)
        assert ids.shape == plain.depth.shape, (ids.shape, plain.depth.shape)
        segments = patch_segments(ids)
        none, _, _ = run(
            backbone, views, AttentionBias(segments, 1.0, frozenset(), "frame")
        )
        assert same(none, plain), f"{name}: the hook without a mask is not Déjà View"

        rec, err = score(plain, truth)
        scene_rec = {
            "segments_from": source,
            "patch_share": float((segments >= 0).mean()),
            "identity_bit_for_bit": True,
            "variants": {
                BASE: rec
                | {"ray_deg": 0.0, "depth_rel": 0.0, "seconds": t_plain}
                | {"peak_gb": m_plain}
            },
        }
        if name == MAP_SCENE:
            maps[BASE] = err[list(MAP_VIEWS)].astype(np.float16)
        for where in args.where:
            for steps in args.steps:
                for beta in [0.0, *args.betas]:
                    label = f"{where} {steps} b={beta:g}"
                    bias = AttentionBias(segments, beta, STEP_SETS[steps], where)
                    biased, t, m = run(backbone, views, bias)
                    geometry, moved = fixed_cameras(biased, plain)
                    rec, err = score(geometry, truth)
                    scene_rec["variants"][label] = (
                        rec | moved | {"seconds": t, "peak_gb": m}
                    )
                    if name == MAP_SCENE:
                        maps[label] = err[list(MAP_VIEWS)].astype(np.float16)
                    print(
                        f"{name} {label}: Chamfer {rec['chamfer_cm']:.3f} cm",
                        flush=True,
                    )
        records[name] = scene_rec
    write(args, raw, records)
    np.savez_compressed(args.out / "e53a_maps.npz", **maps)
    report(records)


def report(records: dict) -> None:
    """Per variant: mean scores, Chamfer change against its beta = 0 run, wins."""
    names = list(records)
    variants = list(records[names[0]]["variants"])
    keys = ("chamfer_cm", "error_cm", "ray_deg", "depth_rel", "seconds", "peak_gb")
    print(
        f"{'variant':<22}"
        + "".join(f"{k.removesuffix('_cm'):>11}" for k in keys)
        + "  vs DV  vs b=0  wins"
    )
    for v in variants:
        mean = {k: np.mean([records[n]["variants"][v][k] for n in names]) for k in keys}
        zero = v if v == BASE else v.rsplit(" ", 1)[0] + " b=0"

        def delta(ref: str, v: str = v) -> list[float]:
            return [
                records[n]["variants"][v]["chamfer_cm"]
                - records[n]["variants"][ref]["chamfer_cm"]
                for n in names
            ]

        d0 = delta(zero)
        print(
            f"{v:<22}"
            + "".join(f"{mean[k]:>11.3f}" for k in keys)
            + f"  {np.mean(delta(BASE)):+.3f}  {np.mean(d0):+.3f}"
            + f"  {sum(x < 0 for x in d0)}/{len(names)}"
        )


def main() -> None:
    """Parse the arguments; run the probe or the grid."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", default=["dev"])
    parser.add_argument("--scenes", nargs="+", help="subset of the splits' scenes")
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--segments", choices=["gt", "wrong", "sam3"], default="gt")
    parser.add_argument("--betas", type=float, nargs="+", default=list(BETAS))
    parser.add_argument(
        "--steps", nargs="+", default=list(STEP_SETS), choices=STEP_SETS
    )
    parser.add_argument("--where", nargs="+", default=list(WHERE), choices=WHERE)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    raw = yaml.safe_load(args.cfg.read_text())
    assert raw["backbone"] == f"dejaview_k{K}", raw["backbone"]
    config = hypersim_config(Path(raw["dataset"]))
    scenes = [f"{n}_{c}" for n, c in config.scenes(args.splits).items()]
    if args.scenes:
        assert set(args.scenes) <= set(scenes), (args.scenes, scenes)
        scenes = [s for s in scenes if s in args.scenes]
    args.out.mkdir(parents=True, exist_ok=False)
    (probe if args.probe else grid)(args, raw, scenes)


if __name__ == "__main__":
    main()
