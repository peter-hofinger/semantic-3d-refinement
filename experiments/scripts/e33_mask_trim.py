"""E33: SAM 3 masks cut at Déjà View's depth jumps vs the same masks untrimmed (2D scores).

Both sets: SAM 3 image mode, nyu40 prompts, t0.3 (cluster job 881 for the trimmed set,
824 for the untrimmed one), linked with the working association (xc0.7 vf0.25 kl0.5) on
Déjà View geometry, on Hypersim's three dev scenes, 20 views each. Scores:
`src.evaluation.instances.score` (unchanged scorer; "objects" = without wall, floor,
ceiling) and E32's per-pixel breakdown over all 20 views: right object (the pixel's
linked object mostly covers its GT object), other object, no mask. For the bedroom's
bed (the case that motivated E33) also the bed mask's area on other GT objects.
Figure: the bedroom's E31 views, untrimmed vs trimmed, colours matched to GT objects.

uv run python experiments/scripts/e33_mask_trim.py > experiments/results/e33_mask_trim.txt
"""

from collections import Counter
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from src.evaluation.instances import STRUCTURE, score
from src.geometry import Geometry
from src.loaders import Views
from src.loaders.scannetpp.gt import GroundTruth
from src.segmentation import Segmentation
from src.segmentation.base import NO_INSTANCE
from src.segmentation.linked import working_associator

ROOT = Path("data/processed/hypersim")
SCENES = {
    "ai_010_007_cam_00": "living room",
    "ai_006_007_cam_00": "bedroom",
    "ai_044_001_cam_00": "lobby",
}
VIEWS = "views_n20_uniform_504"
VARIANTS = {"untrimmed": "sam3image-nyu40-t0.3", "trimmed": "sam3image-nyu40-t0.3-trim"}
SHOWN = (18, 0, 12, 6)  # E31's bedroom views
FIGURE = Path("experiments/figures/e33_mask_trim.jpg")
UNLABELED = np.array([255, 255, 255])
CHANGE_COLOURS = {
    "same": (235, 235, 235),
    "lost its mask": (220, 40, 40),
    "other object": (240, 170, 40),
}


def palette(n: int, seed: int) -> np.ndarray:
    """Return n distinct saturated colours (uint8), shuffled reproducibly."""
    hsv = np.stack(
        [np.linspace(0, 1, n, endpoint=False), np.full(n, 0.75), np.full(n, 0.9)],
        axis=1,
    )
    rgb = mpl.colors.hsv_to_rgb(hsv)
    return (255 * rgb[np.random.default_rng(seed).permutation(n)]).astype(np.uint8)


def outlined(colours: np.ndarray, ids: np.ndarray) -> np.ndarray:
    """Draw a dark line where the id changes between 4-neighbours."""
    edge = np.zeros(ids.shape, dtype=bool)
    edge[:, 1:] |= ids[:, 1:] != ids[:, :-1]
    edge[1:] |= ids[1:] != ids[:-1]
    out = colours.copy()
    out[edge] = (30, 30, 30)
    return out


def dominant_gt(pred: np.ndarray, gt: np.ndarray) -> dict[int, int]:
    """Linked object -> the GT object it covers most (on labeled GT pixels)."""
    on = (gt >= 0) & (pred != NO_INSTANCE)
    pairs = Counter(zip(pred[on].tolist(), gt[on].tolist(), strict=True))
    out: dict[int, int] = {}
    for (p, g), _ in sorted(pairs.items(), key=lambda x: -x[1]):
        out.setdefault(p, g)
    return out


def breakdown(
    pred: np.ndarray, gt: np.ndarray, dom: dict[int, int]
) -> dict[str, np.ndarray]:
    """Per labeled GT pixel: right object, other object, no mask."""
    lab = gt >= 0
    mapped = np.vectorize(lambda p: dom.get(int(p), -9))(pred)
    return {
        "right": lab & (pred != NO_INSTANCE) & (mapped == gt),
        "other": lab & (pred != NO_INSTANCE) & (mapped != gt),
        "no mask": lab & (pred == NO_INSTANCE),
    }


def colours_for(
    pred: np.ndarray, gt: np.ndarray, gt_colour: dict[int, np.ndarray]
) -> dict[int, np.ndarray]:
    """E32's matching: each GT object's colour goes to the linked object covering it most."""
    on = (gt >= 0) & (pred != NO_INSTANCE)
    pairs = Counter(zip(pred[on].tolist(), gt[on].tolist(), strict=True))
    dom = dominant_gt(pred, gt)
    owner: dict[int, int] = {}
    for (p, g), _ in sorted(pairs.items(), key=lambda x: -x[1]):
        if dom[p] == g and g not in owner and p not in owner.values():
            owner[g] = p
    extra = [p for p in dom if p not in owner.values()]
    extra_colour = dict(
        zip(
            extra,
            palette(max(len(extra), 1), seed=1)[: len(extra)] // 2 + 100,
            strict=True,
        )
    )
    return {p: gt_colour[g] for g, p in owner.items()} | extra_colour


def paint(ids: np.ndarray, colour: dict[int, np.ndarray]) -> np.ndarray:
    """Colour an id map; NO_INSTANCE white, unmatched ids black."""
    img = np.tile(UNLABELED, (*ids.shape, 1)).astype(np.uint8)
    for k in np.unique(ids[ids != NO_INSTANCE]):
        img[ids == k] = colour.get(int(k), (0, 0, 0))
    return img


def main() -> None:
    """Link both mask sets per scene, score them, draw the bedroom figure."""
    print(
        "## E33: SAM 3 nyu40 t0.3 masks untrimmed vs cut at Déjà View depth jumps, linked (working setting)\n"
    )
    print(
        "| scene | variant | subset | recall | pixel recall | precision | leakage | cross-class leakage | consistent | id switches | merges |"
    )
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    pixel_rows, bed_rows, linked_bedroom = [], [], {}
    for scene, title in SCENES.items():
        folder = ROOT / scene
        gt = GroundTruth.load(folder / f"gt_{VIEWS}.npz")
        geometry = Geometry.load(folder / f"dejaview_k16_{VIEWS}.npz")
        for variant, stem in VARIANTS.items():
            per_view = Segmentation.load(folder / f"{stem}_{VIEWS}.npz")
            linked = working_associator().associate(per_view, geometry)
            for subset, exclude in (("all", ()), ("objects", STRUCTURE)):
                s = score(gt, linked, exclude)
                print(
                    f"| {title} | {variant} | {subset} | {s.recall:.2f} | {s.pixel_recall:.2f} | {s.precision:.2f} | "
                    f"{s.leakage:.3f} | {s.cross_leakage:.3f} | {s.consistent:.2f} | {s.id_switches} | {s.merges} |"
                )
            dom = dominant_gt(linked.instance_ids, gt.instance_ids)
            cls = breakdown(linked.instance_ids, gt.instance_ids, dom)
            lab = (gt.instance_ids >= 0).sum()
            masked = (per_view.instance_ids != NO_INSTANCE).mean()
            pixel_rows.append(
                f"| {title} | {variant} | {cls['right'].sum() / lab:.3f} | {cls['other'].sum() / lab:.3f} | "
                f"{cls['no mask'].sum() / lab:.3f} | {masked:.3f} | {len(linked.classes)} |"
            )
            beds = [g for g, c in gt.classes.items() if c == "bed"]
            if title == "bedroom":
                linked_bedroom[variant] = linked
                for g in beds:
                    on_bed = gt.instance_ids == g
                    bed_pred = np.isin(
                        linked.instance_ids, [p for p, d in dom.items() if d == g]
                    )
                    off = bed_pred & (gt.instance_ids != g)
                    behind = Counter(
                        gt.classes.get(int(x), "unlabeled")
                        for x in gt.instance_ids[off]
                        if x >= 0
                    )
                    bed_rows.append(
                        f"| {g} | {variant} | {cls['right'][on_bed].mean():.3f} | {cls['other'][on_bed].mean():.3f} | "
                        f"{cls['no mask'][on_bed].mean():.3f} | {bed_pred.sum()} | {off.sum() / max(bed_pred.sum(), 1):.3f} | "
                        + ", ".join(f"{k} {v}" for k, v in behind.most_common(4))
                        + " |"
                    )
    print(
        "\nPer labeled GT pixel over all 20 views (E32's breakdown); masked = share of all pixels with a per-view mask\n"
    )
    print(
        "| scene | variant | right object | other object | no mask | masked | linked objects |"
    )
    print("|---|---|---|---|---|---|---|")
    print("\n".join(pixel_rows))
    print(
        "\nBedroom bed (GT instances of class bed), all 20 views; bed masks = linked objects whose dominant GT is the bed\n"
    )
    print(
        "| GT id | variant | right | other | no mask | bed-mask px | bed mask on other GT | there, by GT class (px) |"
    )
    print("|---|---|---|---|---|---|---|---|")
    print("\n".join(bed_rows))

    folder = ROOT / "ai_006_007_cam_00"
    views = Views.load(folder / f"{VIEWS}.npz")
    gt = GroundTruth.load(folder / f"gt_{VIEWS}.npz")
    gt_ids = sorted({int(g) for g in np.unique(gt.instance_ids) if g >= 0})
    gt_colour = dict(zip(gt_ids, palette(len(gt_ids), seed=0), strict=True))
    base, trim = (
        linked_bedroom["untrimmed"].instance_ids,
        linked_bedroom["trimmed"].instance_ids,
    )
    col = {
        v: colours_for(s.instance_ids, gt.instance_ids, gt_colour)
        for v, s in linked_bedroom.items()
    }
    dom_base, dom_trim = (
        dominant_gt(base, gt.instance_ids),
        dominant_gt(trim, gt.instance_ids),
    )
    fig, axes = plt.subplots(len(SHOWN), 4, figsize=(16, 2.9 * len(SHOWN)))
    for row, i in zip(axes, SHOWN, strict=True):
        gb = np.vectorize(lambda p: dom_base.get(int(p), -9))(base[i])
        gtr = np.vectorize(lambda p: dom_trim.get(int(p), -9))(trim[i])
        lost = (base[i] != NO_INSTANCE) & (trim[i] == NO_INSTANCE)
        moved = (base[i] != NO_INSTANCE) & (trim[i] != NO_INSTANCE) & (gb != gtr)
        change = np.tile(np.array(CHANGE_COLOURS["same"]), (*base[i].shape, 1)).astype(
            np.uint8
        )
        change[lost] = CHANGE_COLOURS["lost its mask"]
        change[moved] = CHANGE_COLOURS["other object"]
        frame = views.names[i].split(".")[-1]
        panels = (
            (views.images[i], f"view {i} (frame {frame}): image"),
            (
                outlined(paint(base[i], col["untrimmed"]), base[i]),
                "untrimmed masks, linked",
            ),
            (
                outlined(paint(trim[i], col["trimmed"]), trim[i]),
                "trimmed at depth jumps, linked",
            ),
            (
                change,
                f"changed: lost mask {lost.mean():.3f}, other object {moved.mean():.3f} (of image)",
            ),
        )
        for ax, (img, caption) in zip(row, panels, strict=True):
            ax.imshow(img)
            ax.set_title(caption, fontsize=9)
            ax.axis("off")
    handles = [
        Patch(color=np.array(c) / 255, label=k) for k, c in CHANGE_COLOURS.items()
    ]
    handles.append(Patch(facecolor="white", edgecolor="black", label="no mask"))
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=9, frameon=False)
    fig.suptitle(
        "E33: bedroom ai_006_007, SAM 3 nyu40 t0.3 + 3D linking, untrimmed vs cut at Déjà View depth jumps; "
        "colour = matched GT object (same colour, same object)",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(FIGURE, dpi=110, pil_kwargs={"quality": 88})
    print(f"\nfigure: {FIGURE}")


if __name__ == "__main__":
    main()
