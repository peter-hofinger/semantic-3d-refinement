"""E26a: the direct competitors' geometry on ETH3D, next to ours with and without semantics.

Our variants are refined and scored exactly as in E35 (`e35_eth3d_eval.score`: one
Umeyama Sim(3) on the pixel pairs, `labeled.surface_metrics`); each competitor's
per-pixel points, written by its runner (`panst3r_predict.py`, `eps3d_predict.py`,
`iggt_predict.py`, `pe3r_predict.py`), are scored by the same function. A competitor
without a file for a scene is reported as missing, not dropped silently.

Prints, per group (indoor, outdoor, all), each method's mean and median over the scenes
of accuracy, completeness and Chamfer (cm); writes `runs/e26a-<time>/run.yaml` (config,
git hash, scores per scene) and `results.npz` (our depths and every alignment).

uv run python experiments/scripts/e26a_eth3d.py --cfg experiments/configs/e26a.yaml
"""

import argparse
import dataclasses
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from e35_eth3d_eval import SURFACE, score

from src.analysis.scene_eval import git_hash, load_inputs, refine_variant
from src.loaders.eth3d.scene import load_config
from src.refinement.multiview import RefineConfig, refine_scene

HEADLINE = SURFACE[:3]


def summary(scenes: dict, methods: list[str]) -> dict:
    """Mean and median over the scenes per group and method; None where one is missing."""
    groups = {"indoor": "indoor", "outdoor": "outdoor", "all": None}
    out = {}
    for group, kind in groups.items():
        out[group] = {}
        for m in methods:
            rows = [
                s["methods"].get(m)
                for s in scenes.values()
                if kind is None or s["kind"] == kind
            ]
            if any(r is None for r in rows):
                out[group][m] = None
                continue
            a = np.array([[r[k] for k in HEADLINE] for r in rows])
            out[group][m] = {
                "scenes": len(rows),
                "mean": dict(zip(HEADLINE, a.mean(0).tolist(), strict=True)),
                "median": dict(zip(HEADLINE, np.median(a, 0).tolist(), strict=True)),
            }
    return out


def main() -> None:
    """Refine and score every ETH3D scene, score every competitor; see the docstring."""
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
    methods = [v["name"] for v in raw["variants"]] + list(raw["competitors"])

    out = Path(raw["output_dir"]) / f"{cfg_path.stem}-{time.strftime('%Y%m%d-%H%M%S')}"
    out.mkdir(parents=True)
    scenes: dict[str, dict] = {}
    arrays: dict[str, np.ndarray] = {}
    for n, name in enumerate(kinds):
        inputs = load_inputs(raw, name, device)
        if n == 0:  # so the first refinement's time is not the CUDA start
            refine_scene(
                inputs.scene,
                dataclasses.replace(base, gn_iters=1, cg_iters=1),
                inputs.normals,
            )
        scenes[name] = {"kind": kinds[name], "methods": {}}
        outputs = {}
        for i, variant in enumerate(raw["variants"]):
            seconds, _, depth, points = refine_variant(inputs, variant, base)
            arrays[f"{name}/{i}"] = depth.astype(np.float32)
            outputs[variant["name"]] = (points, seconds)
        for method, folder in raw["competitors"].items():
            path = Path(folder) / f"{name}.npz"
            if path.is_file():
                with np.load(path) as f:
                    outputs[method] = (
                        f["points"].astype(np.float64),
                        float(f["seconds"]),
                    )
        for method, (points, seconds) in outputs.items():
            scores = score(inputs.truth, points)
            arrays[f"{name}/{method}/align"] = scores.pop("align")
            scenes[name]["methods"][method] = scores | {"seconds": seconds}
            values = " | ".join(f"{k} {scores[k]:.3f}" for k in HEADLINE)
            print(f"{name} | {method} | {values} | {seconds:.1f} s", flush=True)

    table = summary(scenes, methods)
    for group, rows in table.items():
        print(
            f"\n{group}: mean / median over scenes, cm (accuracy | completeness | Chamfer)"
        )
        for m, r in rows.items():
            if r is None:
                print(f"{m:28s} | missing on some scene")
                continue
            cells = " | ".join(
                f"{r['mean'][k]:.2f} / {r['median'][k]:.2f}" for k in HEADLINE
            )
            print(f"{m:28s} | {cells} | {r['scenes']} scenes")
    report = {
        "config": raw,
        "refine": refine_raw,
        "git": git_hash(),
        "summary": table,
        "scenes": scenes,
    }
    (out / "run.yaml").write_text(
        yaml.safe_dump(report, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    np.savez_compressed(out / "results.npz", **arrays)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
