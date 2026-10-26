"""Reference table (markdown) of every cached segmentation with the current scorer.

uv run python experiments/scripts/score_table.py
"""

from pathlib import Path

from src.evaluation.instances import STRUCTURE, score
from src.loaders.scannetpp.gt import GroundTruth
from src.segmentation import Segmentation

DATA = Path("data/processed/scannetpp")
SCENES = {"5ee7c22ba0": "kitchen", "7b6477cb95": "office"}
METHODS = {
    "video mode (E4)": "sam3video_views_n20_uniform_504",
    "image, no association (E6)": "sam3image_views_n20_uniform_504",
    "image + assoc GT geometry (E7)": "sam3image_views_n20_uniform_504_assoc-gt",
    "image + assoc Deja View (E9)": (
        "sam3image_views_n20_uniform_504_assoc-dejaview_k16_views_n20_uniform_504"
    ),
    "image top100 + assoc Deja View (E12)": (
        "sam3image-top100_views_n20_uniform_504_assoc-dejaview_k16_views_n20_uniform_504"
    ),
    # E13: masks of different classes linked at overlap >= 0.7 / >= 0.5
    **{
        f"image{p} + assoc Deja View, cross-class {x} (E13)": (
            f"sam3image{p}_views_n20_uniform_504_assoc-xc{x}"
            "-dejaview_k16_views_n20_uniform_504"
        )
        for p in ("", "-top100")
        for x in ("0.7", "0.5")
    },
    # E14: a direction counts only if >= f of the mask's in-image pixels are visible
    **{
        f"image{p} + assoc Deja View, cross-class 0.7, visible >= {f} (E14)": (
            f"sam3image{p}_views_n20_uniform_504_assoc-xc0.7-vf{f}"
            "-dejaview_k16_views_n20_uniform_504"
        )
        for p in ("", "-top100")
        for f in ("0.5", "0.25")
    },
    # E15: SAM 3 at threshold 0.3; <= 0.5 dropped (= E14) / all kept / kept if linked
    **{
        f"image{p} t0.3 + assoc Deja View, xc 0.7, vf 0.25, {name} (E15)": (
            f"sam3image{p}-t0.3_views_n20_uniform_504_assoc-xc0.7-vf0.25{t}"
            "-dejaview_k16_views_n20_uniform_504"
        )
        for p in ("", "-top100")
        for name, t in (
            ("score > 0.5", "-ms0.5"),
            ("no filter", ""),
            ("multi-view filter", "-kl0.5"),
        )
    },
}

print(
    "| scene | method | subset | recall | pixel recall | precision | leakage "
    "| cross-class leakage | consistent | GT in >=2 views | id switches | merges |"
)
print("|---|---|---|---|---|---|---|---|---|---|---|---|")
for scene, name in SCENES.items():
    gt = GroundTruth.load(DATA / scene / "gt_views_n20_uniform_504.npz")
    for method, stem in METHODS.items():
        pred = Segmentation.load(DATA / scene / f"{stem}.npz")
        for subset, exclude in (("all", ()), ("objects", STRUCTURE)):
            s = score(gt, pred, exclude)
            print(
                f"| {name} | {method} | {subset} | {s.recall:.2f} | {s.pixel_recall:.2f} "
                f"| {s.precision:.2f} | {s.leakage:.3f} | {s.cross_leakage:.3f} "
                f"| {s.consistent:.2f} | {s.num_gt} | {s.id_switches} | {s.merges} |"
            )
