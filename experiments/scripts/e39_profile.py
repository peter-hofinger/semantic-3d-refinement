"""E39: time every pipeline stage on one scene with warm models, and where SAM 3 spends it.

    uv run python experiments/scripts/e39_profile.py <scene folder>
    uv run python experiments/scripts/e39_profile.py <scene folder> --breakdown

Default: weights load once (timed apart), then Déjà View, SAM 3, association, DSINE and
the joint refiner run `--repeats` times each; the median per stage is reported. The
association links the cached SAM 3 masks with the cached Déjà View geometry, the refiner
uses the cached normals, so every repeat sees the same inputs.
`--breakdown`: one warm SAM 3 run and one association under cProfile with
CUDA_LAUNCH_BLOCKING=1, so GPU time is charged to the call that launched it; the
blocking slows the run, so these shares are not the timings above.
"""

from __future__ import annotations

import argparse
import os
import sys

if "--breakdown" in sys.argv:  # before torch initializes CUDA
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import cProfile
import importlib.util
import io
import pstats
import statistics
import subprocess
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from src.backbone.dejaview import DejaViewBackbone
from src.geometry import Geometry
from src.loaders import Views
from src.refinement.multiview import (
    RefineConfig,
    refine_scene,
    scene_from_geometry,
    scene_normals,
)
from src.refinement.normals import DSINE_REPO
from src.segmentation.base import Segmentation
from src.segmentation.linked import working_associator
from src.segmentation.sam3 import Sam3ImageSegmenter, load_prompts

EVAL = yaml.safe_load(Path("configs/eval_hypersim.yaml").read_text())
PROMPTS = Path("configs/prompts/nyu40.txt")


def timed(fn: Callable[..., Any], *args: Any) -> tuple[float, Any]:  # noqa: ANN401
    """Return (seconds, result) of fn(*args), with the GPU drained before and after."""
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = fn(*args)
    torch.cuda.synchronize()
    return time.perf_counter() - t0, out


def load_dsine() -> Any:  # noqa: ANN401
    """Load DSINE's predictor from the pinned hub repo, as `dsine_normals` does."""
    sys.path.insert(0, str(DSINE_REPO))
    spec = importlib.util.spec_from_file_location(
        "dsine_hubconf", DSINE_REPO / "hubconf.py"
    )
    hub = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(hub)
    return hub.DSINE()


def run_dsine(predictor: Any, views: Views, geometry: Geometry) -> None:  # noqa: ANN401
    """Run DSINE on every view: the loop of `dsine_normals` without its loading."""
    with torch.no_grad():
        for image, k in zip(views.images, geometry.intrinsics, strict=True):
            x = torch.from_numpy(image).float().permute(2, 0, 1)[None] / 255
            n = predictor.infer_tensor(x, torch.from_numpy(k).float()[None].cuda())[0]
            n.permute(1, 2, 0).cpu().numpy()


def sam3_segmenter() -> Sam3ImageSegmenter:
    """Return the eval protocol's SAM 3: nyu40 prompts at threshold 0.3."""
    synonyms = load_prompts([PROMPTS])
    return Sam3ImageSegmenter(prompts=tuple(synonyms), synonyms=synonyms, threshold=0.3)


def owner(path: str, line: int) -> str:
    """`Class.` of the method defined at path:line, or '' for a module function."""
    if not path.endswith(".py") or not Path(path).is_file():
        return ""
    lines = Path(path).read_text(encoding="utf-8").splitlines()[:line]
    if not lines or not lines[-1].startswith("    "):
        return ""
    for text in reversed(lines):
        if text.startswith("class "):
            return text[6:].split("(")[0].split(":")[0] + "."
    return ""


def profile_rows(prof: cProfile.Profile, keys: tuple[str, ...], top: int) -> str:
    """Cumulative time of the profiled functions whose file:name contains a key."""
    stats = pstats.Stats(prof, stream=io.StringIO())
    rows = []
    for (path, line, name), (_, calls, _, cum, _) in stats.stats.items():
        where = f"{Path(path).name}:{owner(path, line)}{name}"
        if any(k in where for k in keys):
            rows.append((cum, calls, where))
    rows.sort(reverse=True)
    return "\n".join(
        f"  {cum:7.2f} s  {calls:6d}x  {w}" for cum, calls, w in rows[:top]
    )


def breakdown(views: Views, masks: Segmentation, geometry: Geometry) -> None:
    """Profile one warm SAM 3 run and one association (CUDA_LAUNCH_BLOCKING=1)."""
    sam3 = sam3_segmenter()
    sam3._load()  # noqa: SLF001
    sam3.segment(views)  # warm-up: kernels, allocator
    prof = cProfile.Profile()
    t0 = time.perf_counter()
    prof.runcall(sam3.segment, views)
    print(f"SAM 3 under cProfile, blocking launches: {time.perf_counter() - t0:.2f} s")
    keys = (
        "sam3.py:",
        "modeling_sam3.py:get_",
        "modeling_sam3.py:forward",
        "image_processing_sam3",
        "processing_sam3.py:__call__",
        "processing_sam3.py:post_process",
        "tokenization",
    )
    print(profile_rows(prof, keys, 25))
    associator = working_associator()
    prof = cProfile.Profile()
    t0 = time.perf_counter()
    prof.runcall(associator.associate, masks, geometry)
    print(f"\nassociation under cProfile: {time.perf_counter() - t0:.2f} s")
    print(profile_rows(prof, ("associate.py:", "base.py:", "torch", "unique"), 15))


def main() -> None:
    """Load the scene's cached inputs, then time or profile the stages."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scene", type=Path, help="e.g. data/processed/hypersim/<scene>")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--breakdown", action="store_true")
    args = parser.parse_args()
    folder: Path = args.scene
    stem = EVAL["views"]
    views = Views.load(folder / f"{stem}.npz")
    geometry = Geometry.load(folder / f"{EVAL['backbone']}_{stem}.npz")
    masks = Segmentation.load(folder / f"{EVAL['segmentation']}_{stem}.npz")
    commit = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],  # noqa: S607
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],  # noqa: S607
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    print(
        f"{folder.name}: {len(views)} views {views.images.shape[2]}x"
        f"{views.images.shape[1]}, git {commit}{' (dirty)' if dirty else ''}, "
        f"{torch.cuda.get_device_name()}, torch {torch.__version__}"
    )
    if args.breakdown:
        breakdown(views, masks, geometry)
        return

    dejaview, sam3 = DejaViewBackbone(), sam3_segmenter()
    loads = {
        "Déjà View": timed(dejaview._load)[0],  # noqa: SLF001
        "SAM 3": timed(sam3._load)[0],  # noqa: SLF001
    }
    loads["DSINE"], dsine = timed(load_dsine)

    refine_raw = yaml.safe_load(Path(EVAL["refine"]).read_text())["refine"]
    cfg = RefineConfig(**{**refine_raw, "device": "cuda"})
    scene = scene_from_geometry(geometry).to(torch.device("cuda"))
    normals = scene_normals(
        np.load(folder / f"dsine_{stem}.npz")["normals"], geometry, scene
    )
    labels = working_associator().associate(masks, geometry)
    segments = torch.from_numpy(labels.instance_ids.astype(np.int64))
    stages = {
        "Déjà View": lambda: dejaview.predict(views),
        "SAM 3 (per-view masks)": lambda: sam3.segment(views),
        "association": lambda: working_associator().associate(masks, geometry),
        "DSINE": lambda: run_dsine(dsine, views, geometry),
        "joint refiner": lambda: refine_scene(scene, cfg, normals, segments),
    }
    for fn in stages.values():  # warm-up: CUDA context, kernels, allocator
        fn()
    runs: dict[str, list[float]] = {name: [] for name in stages}
    for _ in range(args.repeats):
        for name, fn in stages.items():
            runs[name].append(timed(fn)[0])

    medians = {name: statistics.median(t) for name, t in runs.items()}
    total = sum(medians.values())
    print(
        "\nweight loading (once, not in the total): "
        + ", ".join(f"{k} {v:.1f} s" for k, v in loads.items())
    )
    print(f"\nstage | median of {args.repeats} (s) | runs (s) | share")
    print("---|---|---|---")
    for name, t in runs.items():
        share = medians[name] / total
        print(
            f"{name} | {medians[name]:.2f} | {', '.join(f'{x:.2f}' for x in t)} | "
            f"{share:.0%}"
        )
    labeling = medians["SAM 3 (per-view masks)"] + medians["association"]
    print(f"total | {total:.2f} | | 100%")
    print(
        f"\nSAM 3 + association: {labeling:.2f} s = {labeling / total:.0%} of the total"
    )
    print(f"peak GPU memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
