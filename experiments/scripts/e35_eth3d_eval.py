"""E35: Déjà View and the joint refiner on ETH3D's 13 high-res training scenes.

ETH3D is out of Déjà View's training domain (paper Table 6). The scenes are scored as
the evaluation protocol scores Hypersim, through the same functions: inputs loaded and
SAM 3 masks linked by `scene_eval.load_inputs`, each variant refined (or not) by
`scene_eval.refine_variant`, its per-pixel points aligned onto the ground truth by one
Umeyama Sim(3) on the pixel pairs and scored by `labeled.surface_metrics`. Only the
label scores are left out: ETH3D has no labels. Kept apart from `scene_eval` so the
protocol module stays Hypersim's (decided 09.11, LOG.md E35).

Writes `runs/eval_eth3d-<time>/run.yaml` (config, git hash, scores per scene and the
indoor / outdoor / all means) and `results.npz` (`<scene>/<i>` depth of variant i,
`<scene>/<i>/align` its s R | t, `<scene>/labels` the linked instances), the inputs
of E35's error split.

uv run python experiments/scripts/e35_eth3d_eval.py --cfg configs/eval_eth3d.yaml
"""

import argparse
import dataclasses
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from src.analysis.scene_eval import git_hash, load_inputs, refine_variant
from src.evaluation.labeled import LabeledCloud, mean, surface_metrics, umeyama
from src.loaders.eth3d.scene import load_config
from src.refinement.multiview import RefineConfig, refine_scene

SURFACE = (
    "accuracy_cm",
    "completeness_cm",
    "chamfer_cm",
    "accuracy_median_cm",
    "completeness_median_cm",
    "chamfer_median_cm",
)


def score(truth: LabeledCloud, points: np.ndarray) -> dict:
    """Align per-pixel points (V, H, W, 3) by Umeyama on the pixels; surface scores in cm."""
    assert truth.pixels is not None
    assert points.shape == (*truth.pixels.shape, 3), (points.shape, truth.pixels.shape)
    pred = points[truth.pixels].astype(np.float64)
    found = np.isfinite(pred).all(1)
    s, rotation, t = umeyama(pred[found], truth.points[found])
    align = np.eye(4)
    align[:3, :3], align[:3, 3] = s * rotation, t
    metrics = surface_metrics(s * pred @ rotation.T + t, truth, [])
    out = {k: float(metrics[k]) for k in SURFACE}
    return out | {"scale": s, "nonfinite": int((~found).sum()), "align": align}


def main() -> None:
    """Refine and score every ETH3D scene; see the module docstring."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=Path, required=True)
    cfg_path: Path = parser.parse_args().cfg
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    refine_raw = yaml.safe_load(Path(raw["refine"]).read_text(encoding="utf-8"))
    base = RefineConfig(**refine_raw["refine"])
    cuda = base.device == "cuda" and torch.cuda.is_available()
    base = dataclasses.replace(base, device="cuda" if cuda else "cpu")
    device = torch.device(base.device)
    kinds = load_config(Path(raw["dataset"])).scenes
    variants = [v["name"] for v in raw["variants"]]

    out = Path(raw["output_dir"]) / f"{cfg_path.stem}-{time.strftime('%Y%m%d-%H%M%S')}"
    out.mkdir(parents=True)
    scenes: dict[str, dict] = {}
    arrays: dict[str, np.ndarray] = {}
    for n, name in enumerate(kinds):
        t0 = time.perf_counter()
        inputs = load_inputs(raw, name, device)
        if n == 0:  # so the first refinement's time is not the CUDA start
            refine_scene(
                inputs.scene,
                dataclasses.replace(base, gn_iters=1, cg_iters=1),
                inputs.normals,
            )
        arrays[f"{name}/labels"] = inputs.labels.instance_ids
        scenes[name] = {"kind": kinds[name], "variants": {}}
        for i, variant in enumerate(raw["variants"]):
            seconds, log, depth, points = refine_variant(inputs, variant, base)
            scores = score(inputs.truth, points)
            arrays[f"{name}/{i}"] = depth.astype(np.float32)
            arrays[f"{name}/{i}/align"] = scores.pop("align")
            scenes[name]["variants"][variants[i]] = scores | {
                "refinement_s": seconds,
                "log": log,
            }
            values = " | ".join(f"{k} {scores[k]:.3f}" for k in SURFACE[:3])
            print(f"{name} | {variants[i]} | {values} | {seconds:.1f} s", flush=True)
        print(f"{name}: {time.perf_counter() - t0:.0f} s in all", flush=True)

    groups = {"indoor": "indoor", "outdoor": "outdoor", "all": None}
    means = {
        group: {
            v: {
                k: mean(
                    [
                        s["variants"][v][k]
                        for s in scenes.values()
                        if kind is None or s["kind"] == kind
                    ]
                )
                for k in SURFACE
            }
            for v in variants
        }
        for group, kind in groups.items()
    }
    for group, by_variant in means.items():
        for v, m in by_variant.items():
            values = " | ".join(f"{k} {m[k]:.3f}" for k in SURFACE[:3])
            print(f"{group} | {v} | {values}", flush=True)
    summary = {
        "config": raw,
        "refine": refine_raw,
        "git": git_hash(),
        "means": means,
        "scenes": scenes,
    }
    (out / "run.yaml").write_text(
        yaml.safe_dump(summary, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    np.savez_compressed(out / "results.npz", **arrays)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
