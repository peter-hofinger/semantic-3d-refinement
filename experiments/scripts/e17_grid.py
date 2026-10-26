"""E17: full grid (a) visible share x (b) multi-view filter x vocabulary, both scenes.

Every setup is associated from the threshold-0.3 SAM 3 cache with Déjà View geometry
and cross-class 0.7, saved under the name the associate CLI would give it, and
scored. Setups cached before (E15 / E16) must come out identical (checked).
Prints the full table and the main effect of each factor (mean change when it is
switched on, over the settings of the other two factors), and saves the figure.

uv run python experiments/scripts/e17_grid.py | tee experiments/results/e17_grid.txt
"""

import itertools
import time
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.evaluation.instances import STRUCTURE, score
from src.geometry import Geometry
from src.loaders.scannetpp.gt import GroundTruth
from src.segmentation import Segmentation
from src.segmentation.associate import Associator

DATA = Path("data/processed/scannetpp")
FIGURE = Path("experiments/figures/e17_main_effects.jpg")
SCENES = {"5ee7c22ba0": "kitchen", "7b6477cb95": "office"}
GEOMETRY = "dejaview_k16_views_n20_uniform_504"
VOCABULARIES = {
    "top100": "sam3image-top100-t0.3_views_n20_uniform_504",
    "VLM": (
        "sam3image-vlm-qwen3vl2binstruct_views_n20_uniform_504-t0.3"
        "_views_n20_uniform_504"
    ),
}
# name, higher is better, format
METRICS = {
    "pixel_recall": ("pixel recall", True, ".2f"),
    "recall": ("recall", True, ".2f"),
    "precision": ("precision", True, ".2f"),
    "cross_leakage": ("cross-class leakage", False, ".3f"),
    "consistent": ("consistent", True, ".2f"),
    "id_switches": ("id switches", False, "d"),
    "merges": ("merges", False, "d"),
}


def run(
    seg: Segmentation, geometry: Geometry, stem: Path, *, a: bool, b: bool
) -> tuple[Segmentation, str]:
    """Associate one setup, compare with an existing cache, save it."""
    linked = Associator(
        cross_class_overlap=0.7,
        min_visible_fraction=0.25 if a else 0.0,
        min_score=0.0 if b else 0.5,
        keep_linked_below=0.5 if b else 0.0,
    ).associate(seg, geometry)
    tag = "xc0.7-" + ("vf0.25-" if a else "") + ("kl0.5-" if b else "ms0.5-")
    out = stem.with_name(f"{stem.name}_assoc-{tag}{GEOMETRY}.npz")
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
    return linked, check


rows = {}  # (scene, vocabulary, a, b) -> {"all": scores, "objects": scores}
print(
    "| scene | vocabulary | (a) vf 0.25 | (b) multi-view filter | subset | recall "
    "| pixel recall | precision | leakage | cross-class leakage | consistent "
    "| GT in >=2 views | id switches | merges |"
)
print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
checks = []
for scene, name in SCENES.items():
    d = DATA / scene
    gt = GroundTruth.load(d / "gt_views_n20_uniform_504.npz")
    geometry = Geometry.load(d / f"{GEOMETRY}.npz")
    for vocab, stem in VOCABULARIES.items():
        seg = Segmentation.load(d / f"{stem}.npz")
        for a, b in itertools.product((False, True), repeat=2):
            start = time.perf_counter()
            linked, check = run(seg, geometry, d / stem, a=a, b=b)
            checks.append(
                f"{name} {vocab} a={a:d} b={b:d}: {len(linked.classes)} objects, "
                f"{time.perf_counter() - start:.0f}s, {check}"
            )
            rows[scene, vocab, a, b] = {}
            for subset, exclude in (("all", ()), ("objects", STRUCTURE)):
                s = score(gt, linked, exclude)
                rows[scene, vocab, a, b][subset] = s
                print(
                    f"| {name} | {vocab} | {'on' if a else 'off'} | "
                    f"{'on' if b else 'off'} | {subset} | {s.recall:.2f} "
                    f"| {s.pixel_recall:.2f} | {s.precision:.2f} | {s.leakage:.3f} "
                    f"| {s.cross_leakage:.3f} | {s.consistent:.2f} | {s.num_gt} "
                    f"| {s.id_switches} | {s.merges} |"
                )

print()
print("Runs:")
for line in checks:
    print(f"  {line}")


def effect(scene: str, factor: str, metric: str, subset: str = "all") -> float:
    """Mean change of a metric when a factor is switched on, over the other two."""
    diffs = []
    for vocab, a, b in itertools.product(VOCABULARIES, (False, True), (False, True)):
        on = {"a": (vocab, True, b), "b": (vocab, a, True), "VLM": ("VLM", a, b)}
        off = {"a": (vocab, False, b), "b": (vocab, a, False), "VLM": ("top100", a, b)}
        diffs.append(
            getattr(rows[(scene, *on[factor])][subset], metric)
            - getattr(rows[(scene, *off[factor])][subset], metric)
        )
    return float(np.mean(diffs))  # each pair counted twice, mean unchanged


FACTORS = {
    "a": "(a) visible share 0.25",
    "b": "(b) multi-view filter",
    "VLM": "(c) VLM instead of top100",
}
for subset in ("all", "objects"):
    print()
    print(f"Main effects ({subset}): mean change when the factor is switched on")
    print("| factor | scene | " + " | ".join(m[0] for m in METRICS.values()) + " |")
    print("|---|---|" + "---|" * len(METRICS))
    for factor, label in FACTORS.items():
        for scene, name in SCENES.items():
            values = [
                f"{effect(scene, factor, m, subset):+.3f}"
                if fmt == ".3f"
                else f"{effect(scene, factor, m, subset):+.2f}"
                for m, (_, _, fmt) in METRICS.items()
            ]
            print(f"| {label} | {name} | " + " | ".join(values) + " |")

# interactions: effect of each factor at each level of each other factor (all)
print()
print("Effect of each factor split by another factor (all, kitchen / office):")
for factor, other in itertools.permutations(("a", "b", "VLM"), 2):
    for level in (False, True):
        parts = []
        for m in ("pixel_recall", "precision", "consistent", "merges"):
            vals = []
            for scene in SCENES:
                diffs = []
                for vocab, a, b in itertools.product(
                    VOCABULARIES, (False, True), (False, True)
                ):
                    setting = {"a": a, "b": b, "VLM": vocab == "VLM"}
                    if setting[other] != level or setting[factor]:
                        continue
                    on = dict(setting, **{factor: True})
                    key = (scene, "VLM" if on["VLM"] else "top100", on["a"], on["b"])
                    base = (scene, vocab, a, b)
                    diffs.append(
                        getattr(rows[key]["all"], m) - getattr(rows[base]["all"], m)
                    )
                vals.append(f"{np.mean(diffs):+.2f}")
            parts.append(f"{m} {'/'.join(vals)}")
        print(f"  {factor} with {other}={'on' if level else 'off'}: " + ", ".join(parts))

# figure: main effects per metric, one panel each, bars per factor, scenes side by side
COLORS = {"kitchen": "#2a78d6", "office": "#eb6834"}
fig, axes = plt.subplots(1, len(METRICS), figsize=(17, 3.6))
x = np.arange(len(FACTORS))
for ax, (m, (label, higher, _)) in zip(axes, METRICS.items(), strict=True):
    for k, (scene, name) in enumerate(SCENES.items()):
        vals = [effect(scene, f, m) for f in FACTORS]
        ax.bar(x + (k - 0.5) * 0.36, vals, 0.34, color=COLORS[name], label=name)
    ax.axhline(0, color="#52514e", linewidth=0.8)
    ax.set_xticks(x, ["(a)", "(b)", "(c) VLM"])
    ax.set_title(f"{label}\n({'higher' if higher else 'lower'} is better)", fontsize=9)
    ax.grid(axis="y", color="#e5e4df", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=8)
axes[0].legend(fontsize=8, frameon=False)
fig.suptitle(
    "E17: mean change when a factor is switched on (over the other two factors), "
    "all classes. (a) visible share 0.25, (b) threshold 0.3 + multi-view filter, "
    "(c) VLM vocabulary instead of top100",
    fontsize=10,
)
fig.tight_layout()
fig.savefig(FIGURE, dpi=130, pil_kwargs={"quality": 90})
print(f"\nfigure: {FIGURE}")