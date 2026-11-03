"""E32: how the four shared bedroom views segment (Hypersim ai_006_007, views from E31).

Segmentation = SAM 3 image mode top100 t0.3 + 3D linking in the working setting
(xc0.7 vf0.25 kl0.5) with Déjà View geometry (E30 cache). Colours are matched, not
numbered: every linked object gets the colour of the GT object it covers most over
all 20 views, one linked object per GT object (the largest overlap wins); further
linked objects on the same GT object (splits) get colours of their own. So the same
colour in the GT and SAM 3 columns means the same object, and a colour change of one
object between views is an id switch. Error map per labeled GT pixel: right object
(the pixel's linked object mostly covers its GT object), other object (the harmful
case for refinement: the mask belongs to something else), no mask.

uv run python experiments/scripts/e32_bedroom_segmentation.py > experiments/results/e32_bedroom_segmentation.txt
"""

from collections import Counter
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from src.loaders import Views
from src.loaders.scannetpp.gt import NO_MESH, GroundTruth
from src.segmentation import Segmentation
from src.segmentation.base import NO_INSTANCE

SCENE = Path("data/processed/hypersim/ai_006_007_cam_00")
VIEWS = "views_n20_uniform_504"
SEGMENTATION = (
    f"e30_cluster/sam3image-top100-t0.3_{VIEWS}_assoc-xc0.7-vf0.25-kl0.5-dejaview_k16"
)
SHOWN = (18, 0, 12, 6)  # E31's chosen views
FIGURE = Path("experiments/figures/e32_bedroom_segmentation.jpg")
UNLABELED = np.array([255, 255, 255])
ERROR_COLOURS = {
    "right": (60, 160, 90),
    "other": (220, 40, 40),
    "no mask": (150, 150, 150),
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


def main() -> None:
    """Match linked objects to GT objects, print per-view shares, draw the figure."""
    views = Views.load(SCENE / f"{VIEWS}.npz")
    gt = GroundTruth.load(SCENE / f"gt_{VIEWS}.npz")
    seg = Segmentation.load(SCENE / f"{SEGMENTATION}.npz")
    assert seg.instance_ids.shape == gt.instance_ids.shape, (
        seg.instance_ids.shape,
        gt.instance_ids.shape,
    )
    labeled = gt.instance_ids >= 0
    pairs = Counter(
        zip(
            seg.instance_ids[labeled & (seg.instance_ids != NO_INSTANCE)].tolist(),
            gt.instance_ids[labeled & (seg.instance_ids != NO_INSTANCE)].tolist(),
            strict=True,
        )
    )
    dominant: dict[int, int] = {}
    for (p, g), _n in sorted(pairs.items(), key=lambda x: -x[1]):
        if p not in dominant:
            dominant[p] = g
    gt_ids = sorted({int(g) for g in np.unique(gt.instance_ids) if g >= 0})
    gt_colour = dict(zip(gt_ids, palette(len(gt_ids), seed=0), strict=True))
    owner: dict[int, int] = {}  # GT id -> the linked object that gets its colour
    for (p, g), _ in sorted(pairs.items(), key=lambda x: -x[1]):
        if dominant[p] == g and g not in owner and p not in owner.values():
            owner[g] = p
    extra = [p for p in dominant if p not in owner.values()]
    extra_colour = dict(zip(extra, palette(len(extra), seed=1) // 2 + 100, strict=True))
    pred_colour = {p: gt_colour[g] for g, p in owner.items()} | extra_colour

    print(
        f"## E32: bedroom ai_006_007, SAM 3 top100 t0.3 + 3D linking (Déjà View geometry), views {SHOWN}\n"
    )
    print(
        f"linked objects on labeled pixels: {len(dominant)}; with the colour of their GT object: {len(owner)}; "
        f"extra (split / second id on a GT object already owned): {len(extra)}\n"
    )
    print(
        "| view | frame | right object | other object | no mask | unlabeled GT (% of image) |"
    )
    print("|---|---|---|---|---|---|")
    fig, axes = plt.subplots(len(SHOWN), 4, figsize=(16, 2.9 * len(SHOWN)))
    for row, i in zip(axes, SHOWN, strict=True):
        g_ids, p_ids = gt.instance_ids[i], seg.instance_ids[i]
        lab = g_ids >= 0
        gt_img = np.tile(UNLABELED, (*g_ids.shape, 1)).astype(np.uint8)
        for g in np.unique(g_ids[lab]):
            gt_img[g_ids == g] = gt_colour[int(g)]
        pred_img = np.tile(UNLABELED, (*p_ids.shape, 1)).astype(np.uint8)
        for p in np.unique(p_ids[p_ids != NO_INSTANCE]):
            pred_img[p_ids == p] = pred_colour.get(int(p), (0, 0, 0))
        mapped = np.vectorize(lambda p: dominant.get(int(p), -9))(p_ids)
        cls = {
            "right": lab & (p_ids != NO_INSTANCE) & (mapped == g_ids),
            "other": lab & (p_ids != NO_INSTANCE) & (mapped != g_ids),
            "no mask": lab & (p_ids == NO_INSTANCE),
        }
        err = np.tile(UNLABELED, (*g_ids.shape, 1)).astype(np.uint8)
        for k, m in cls.items():
            err[m] = ERROR_COLOURS[k]
        share = {k: m.sum() / lab.sum() for k, m in cls.items()}
        unl = 100 * (~lab & (g_ids != NO_MESH)).mean()
        frame = views.names[i].split(".")[-1]
        print(
            f"| {i} | {frame} | {share['right']:.2f} | {share['other']:.2f} | {share['no mask']:.2f} | {unl:.1f} |"
        )
        panels = (
            (views.images[i], f"view {i} (frame {frame}): image"),
            (outlined(gt_img, g_ids), "ground truth instances"),
            (
                outlined(pred_img, p_ids),
                "SAM 3 + 3D linking (colour = matched GT object)",
            ),
            (
                err,
                f"right {share['right']:.2f} / other {share['other']:.2f} / no mask {share['no mask']:.2f}",
            ),
        )
        for ax, (img, title) in zip(row, panels, strict=True):
            ax.imshow(img)
            ax.set_title(title, fontsize=9)
            ax.axis("off")
    handles = [
        Patch(color=np.array(c) / 255, label=k) for k, c in ERROR_COLOURS.items()
    ]
    handles.append(
        Patch(facecolor="white", edgecolor="black", label="unlabeled GT / no mask")
    )
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=9, frameon=False)
    fig.suptitle(
        "E32: bedroom ai_006_007, SAM 3 top100 t0.3 + 3D linking (working setting, Déjà View geometry); "
        "same colour = same object, colour change across views = id switch",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(FIGURE, dpi=110, pil_kwargs={"quality": 88})
    print(f"\nfigure: {FIGURE}")


if __name__ == "__main__":
    main()
