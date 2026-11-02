"""Joint refinement of every scene in a config, once per variant, scored against its scan.

Each variant overrides `refine` and `multiview` of the config; all share one problem
setup per scene (DSINE normals, matches, truth). Every variant is scored on the fused
cloud (`evaluation.geometry`), on the views of the per-view protocol
(`evaluation.depth`), and by the disagreement between views.

`uv run python -m src.analysis.multiview_eval --cfg configs/eval_multiview.yaml` writes
`runs/<cfg>-<time>/{run.yaml, results.npz}`; `visualize --run` draws it.
"""

import argparse
import dataclasses
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import Tensor

from src.evaluation.depth import accuracy, score
from src.evaluation.geometry import (
    Truth,
    disagreement_pct,
    fused_metrics,
    load_truth,
    scan_fidelity,
)
from src.refinement.inputs import git_hash
from src.refinement.monocular import RefineConfig, variant_label
from src.refinement.multiview import (
    MV_KEYS,
    Matches,
    MultiviewConfig,
    Scene,
    find_matches,
    load_scene,
    refine_scene,
)
from src.refinement.normals import predicted_normals


def scene_label(cfg: RefineConfig, mv: MultiviewConfig) -> str:
    """Name the active terms, e.g. "E_d + E_mv"; E_d alone is the input."""
    label = variant_label(cfg) + (" + E_mv" if mv.lambda_mv > 0 else "")
    return label + (" (= input)" if label == "E_d" else "")


def run_variant(
    scene: Scene,
    normals: Tensor,
    cfg: RefineConfig,
    mv: MultiviewConfig,
    problem: tuple[Truth, Matches, dict],
) -> tuple[dict, np.ndarray]:
    """Refine all views under one configuration and score the fused cloud and views."""
    truth, matches, raw = problem
    t0 = time.perf_counter()
    depth_hat, log, counts = refine_scene(scene, cfg, mv, normals)
    if depth_hat.is_cuda:
        torch.cuda.synchronize()
    seconds = time.perf_counter() - t0
    disagreement = disagreement_pct(depth_hat, scene, matches)
    depth = depth_hat.cpu().numpy().astype(np.float64)
    record = {
        "label": scene_label(cfg, mv),
        "seconds": seconds,
        "gn_steps": len(log) - 1,
        **counts,
        "energy_initial": log[0],
        "energy_final": log[-1],
        "disagreement_pct": disagreement,
        **fused_metrics(
            depth,
            scene.rays.cpu().numpy().astype(np.float64),
            truth,
            raw["tolerances_cm"],
        ),
        "per_view": {},
    }
    for v in raw["views"]:
        dist, _ = score(
            depth[v], truth.depth[v], truth.k[v], truth.cam_to_world[v], truth.tree
        )
        record["per_view"][v] = accuracy(dist[np.isfinite(dist)], raw["tolerances_cm"])
    return record, depth_hat.cpu().numpy()


def main() -> None:
    """Refine jointly: `uv run python -m src.analysis.multiview_eval --cfg <cfg>`."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=Path, required=True)
    cfg_path: Path = parser.parse_args().cfg
    raw = yaml.safe_load(cfg_path.read_text())
    base = RefineConfig(**raw["refine"])
    device = torch.device(
        base.device if base.device != "cuda" or torch.cuda.is_available() else "cpu"
    )
    base = dataclasses.replace(base, device=device.type)

    out = Path(raw["output_dir"]) / f"{cfg_path.stem}-{time.strftime('%Y%m%d-%H%M%S')}"
    out.mkdir(parents=True)
    scenes, arrays = {}, {}
    for name, paths in raw["scenes"].items():
        t0 = time.perf_counter()
        scene = load_scene(Path(paths["input"])).to(device)
        normals = torch.stack(
            [
                predicted_normals(
                    {**raw, **paths, "view": v},
                    {"depth": scene.depth[v], "rays": scene.rays[v]},
                )[0]
                for v in range(len(scene.depth))
            ]
        )
        truth = load_truth(paths, scene)
        scale = scene.depth.median()
        base_mv = MultiviewConfig(**raw["multiview"])
        matches = find_matches(
            scene, scale / scene.depth, scale, base_mv.mv_tol, base_mv.neighbors
        )
        s, rotation, t = truth.align
        centers = s * scene.center.cpu().numpy() @ rotation.T + t
        center_residual = np.linalg.norm(centers - truth.cam_to_world[:, :3, 3], axis=1)
        scenes[name] = {
            "setup_seconds": time.perf_counter() - t0,
            "align": {
                "scale": s,
                "rotation": rotation.tolist(),
                "translation": t.tolist(),
            },
            "center_residual_cm": float(100 * np.median(center_residual)),
            "scan": scan_fidelity(truth),
            "variants": [],
        }
        refine_scene(
            scene, base, dataclasses.replace(base_mv, gn_iters=1, cg_iters=1), normals
        )
        for i, o in enumerate(raw["variants"]):
            mv = MultiviewConfig(
                **{**raw["multiview"], **{k: v for k, v in o.items() if k in MV_KEYS}}
            )
            cfg = dataclasses.replace(
                base, **{k: v for k, v in o.items() if k not in MV_KEYS}
            )
            record, depth = run_variant(scene, normals, cfg, mv, (truth, matches, raw))
            scenes[name]["variants"].append(record)
            arrays[f"{name}_{i}_depth"] = depth
            keys = (
                "fused_median_cm",
                "thickness_mm",
                "normal_angle_deg",
                "disagreement_pct",
                "seconds",
            )
            print(  # noqa: T201
                name,
                record["label"],
                {k: round(record[k], 3) for k in keys},
                flush=True,
            )

    summary = {
        "config": raw,
        "resolved": dataclasses.asdict(base),
        "git": git_hash(),
        "scenes": scenes,
    }
    (out / "run.yaml").write_text(
        yaml.safe_dump(summary, sort_keys=False, allow_unicode=True)
    )
    np.savez_compressed(out / "results.npz", **arrays)
    print(f"wrote {out}")  # noqa: T201


if __name__ == "__main__":
    main()
