"""E19: arm D, top100 + VLM prompts in one SAM 3 run, vs each list alone.

The union cache is associated like the E17 setups (Déjà View geometry, cross-class
0.7, (a) on, (b) on and off) and saved under the name the associate CLI would give it.
top100 and VLM are reassociated from their caches and must come out identical.

uv run python experiments/scripts/e19_union.py > experiments/results/e19_union.txt
"""

import time
from pathlib import Path

import numpy as np

from src.evaluation.instances import STRUCTURE, score
from src.geometry import Geometry
from src.loaders.scannetpp.gt import GroundTruth
from src.segmentation import Segmentation
from src.segmentation.associate import Associator

DATA = Path("data/processed/scannetpp")
SCENES = {"5ee7c22ba0": "kitchen", "7b6477cb95": "office"}
GEOMETRY = "dejaview_k16_views_n20_uniform_504"
VLM = "vlm-qwen3vl2binstruct_views_n20_uniform_504"
VOCABULARIES = {
    "top100": "sam3image-top100-t0.3_views_n20_uniform_504",
    "VLM": f"sam3image-{VLM}-t0.3_views_n20_uniform_504",
    "top100 + VLM": f"sam3image-top100+{VLM}-t0.3_views_n20_uniform_504",
}

print(
    "| scene | vocabulary | (b) | subset | masks | objects | recall | pixel recall "
    "| precision | leakage | cross-class leakage | consistent | GT in >=2 views "
    "| id switches | merges |"
)
print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
runs = []
for scene, name in SCENES.items():
    d = DATA / scene
    gt = GroundTruth.load(d / "gt_views_n20_uniform_504.npz")
    geometry = Geometry.load(d / f"{GEOMETRY}.npz")
    for vocab, stem in VOCABULARIES.items():
        seg = Segmentation.load(d / f"{stem}.npz")
        for b in (False, True):
            start = time.perf_counter()
            linked = Associator(
                cross_class_overlap=0.7,
                min_visible_fraction=0.25,
                min_score=0.0 if b else 0.5,
                keep_linked_below=0.5 if b else 0.0,
            ).associate(seg, geometry)
            tag = "xc0.7-vf0.25-" + ("kl0.5-" if b else "ms0.5-")
            out = d / f"{stem}_assoc-{tag}{GEOMETRY}.npz"
            check = "new"
            if out.exists():
                old = Segmentation.load(out)
                same = (
                    np.array_equal(old.instance_ids, linked.instance_ids)
                    and old.classes == linked.classes
                )
                assert same, f"{out} differs from the cached result"  # noqa: S101
                check = "same as cache"
            linked.save(out)
            runs.append(
                f"{name} {vocab} b={b:d}: {time.perf_counter() - start:.0f}s, {check}"
            )
            for subset, exclude in (("all", ()), ("objects", STRUCTURE)):
                s = score(gt, linked, exclude)
                print(
                    f"| {name} | {vocab} | {'on' if b else 'off'} | {subset} "
                    f"| {len(seg.classes)} | {len(linked.classes)} | {s.recall:.2f} "
                    f"| {s.pixel_recall:.2f} | {s.precision:.2f} | {s.leakage:.3f} "
                    f"| {s.cross_leakage:.3f} | {s.consistent:.2f} | {s.num_gt} "
                    f"| {s.id_switches} | {s.merges} |"
                )

print()
print("Runs:")
for line in runs:
    print(f"  {line}")
