"""E21: class-agnostic SAM 2 masks for the pixels SAM 3 leaves without a mask.

E20: the worst Déjà View pixels near edges have no SAM 3 mask (office 24% of the
edge region), so segment-based fixes cannot touch them. Here SAM 2 automatic masks
(facebook/sam2.1-hiera-small, SAM 2's own automatic-mask defaults: 32 x 32 points,
predicted IoU >= 0.8, stability >= 0.95, no crops; not tuned on these scenes) fill
only those pixels: the working SAM 3 segmentation stays as it is, every automatic
mask keeps its own id per view (no class, no association across views). Overlapping
automatic masks are painted large to small, so the finer mask wins.

Compared: SAM 3 top100 (working setting), automatic masks alone, top100 filled with
automatic masks, top100 filled with the automatic masks of the next view (control:
same fill statistics, unrelated boundaries), GT. Measured on E20's regions: share of
pixels with a mask, E18's which-object label, E20's proxy fix (plane, 21 px).

uv run python experiments/scripts/e21_automatic_masks.py generate   # GPU, ~2 min per scene, cached
uv run python experiments/scripts/e21_automatic_masks.py > experiments/results/e21_automatic_masks.txt
"""

import sys
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import distance_transform_edt

sys.path.insert(0, str(Path(__file__).parent))
from e20_proxy_fix import (  # noqa: E402
    BLEED_ERROR, DATA, FIGURES, RADIUS, REGIONS, SCENES, VIEWS, run, scene_data, summary,
)

from src.loaders import Views  # noqa: E402
from src.segmentation import Segmentation  # noqa: E402
from src.segmentation.base import ID_STRIDE, NO_INSTANCE  # noqa: E402

MODEL = "facebook/sam2.1-hiera-small"
AUTO = f"sam2auto-hiera-small_{VIEWS}"
SETTINGS = {"points_per_crop": 32, "pred_iou_thresh": 0.8, "stability_score_thresh": 0.95}
FILL_OFFSET = 1_000_000  # automatic ids in a filled segmentation, clear of the SAM 3 ids
VARIANTS = ("SAM 3 top100", "automatic only", "top100 + automatic", "top100 + automatic of next view", "GT")


def generate() -> None:
    import torch
    from PIL import Image
    from transformers import pipeline

    total = torch.cuda.get_device_properties(0).total_memory / 2**30
    torch.cuda.set_per_process_memory_fraction(min(3.5 / total, 1.0))
    # float32: in bf16 the pipeline's NMS fails (box and score dtypes differ); the model is small
    generator = pipeline("mask-generation", model=MODEL, device=0)
    for scene, name in SCENES.items():
        views = Views.load(DATA / scene / f"{VIEWS}.npz")
        ids = np.full(views.images.shape[:3], NO_INSTANCE, dtype=np.int32)
        for i, image in enumerate(views.images):
            out = generator(Image.fromarray(image), points_per_batch=32, **SETTINGS)
            masks = sorted((np.asarray(m, dtype=bool) for m in out["masks"]), key=lambda m: -m.sum())
            for k, m in enumerate(masks):
                ids[i][m] = i * ID_STRIDE + k
            print(f"{name} view {i}: {len(masks)} masks, coverage {(ids[i] >= 0).mean():.2f}", flush=True)
        Segmentation(ids).save(DATA / scene / f"{AUTO}.npz")
    print(f"peak GPU memory {torch.cuda.max_memory_allocated() / 2**30:.2f} GB")


def which_object(ids: np.ndarray, gt_ids: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """E18: 1 if a pixel's mask has the pixel's own GT object as its dominant one, 2 other, 0 no mask."""
    side = np.zeros(ids.shape, dtype=np.int8)
    for i in range(len(ids)):
        labeled = valid[i] & (gt_ids[i] >= 0)
        for p in np.unique(ids[i][labeled & (ids[i] >= 0)]):
            in_mask = labeled & (ids[i] == p)
            g, c = np.unique(gt_ids[i][in_mask], return_counts=True)
            side[i][in_mask] = np.where(gt_ids[i][in_mask] == g[c.argmax()], 1, 2)
    return side


def fill(base: np.ndarray, auto: np.ndarray) -> np.ndarray:
    return np.where((base == NO_INSTANCE) & (auto >= 0), auto + FILL_OFFSET, base)


def evaluate() -> None:
    results: dict = {}
    for scene, name in SCENES.items():
        data = scene_data(scene)
        auto = Segmentation.load(DATA / scene / f"{AUTO}.npz").instance_ids
        top100 = data["segs"]["SAM 3 top100"]
        data["segs"] = {
            "SAM 3 top100": top100,
            "automatic only": auto,
            "top100 + automatic": fill(top100, auto),
            "top100 + automatic of next view": fill(top100, np.roll(auto, -1, axis=0)),
            "GT": data["segs"]["GT"],
        }
        gt_ids = data["segs"]["GT"]
        valid = data["valid"]
        print(f"## {name} ({scene})")
        n_auto = [len(np.unique(a[a >= 0])) for a in auto]
        print(
            f"automatic masks per view: median {np.median(n_auto):.0f} (min {min(n_auto)}, max {max(n_auto)}); "
            f"pixels with an automatic mask {(auto >= 0)[valid].mean():.2f}, with a SAM 3 mask "
            f"{(top100 >= 0)[valid].mean():.2f}; SAM 3 no-mask pixels covered by an automatic mask "
            f"{(auto >= 0)[valid & (top100 < 0)].mean():.2f}"
        )
        print()
        sides = {v: which_object(s, gt_ids, valid) for v, s in data["segs"].items() if v != "GT"}
        for region_key, title in REGIONS.items():
            region = (data["band"] if region_key == "realistic" else data["bleed"]) & valid
            labeled = region & (gt_ids >= 0)
            base = data["error"][region]
            print(f"### {title}")
            print(
                "| segments | with a mask | right object | other object | no mask | share > 5% after fix "
                "| mean error after fix | changed px: Déjà View mean | fixed mean |"
            )
            print("|---|---|---|---|---|---|---|---|---|")
            s = summary(base, np.full(base.shape, np.nan))
            print(f"| Déjà View (input) | | | | | {s['over5']:.3f} | {s['mean']:.4f} | | |")
            for v, seg in data["segs"].items():
                fixed = run(data, region_key, v, RADIUS)["plane"][region]
                s = summary(base, fixed)
                results[(name, region_key, v)] = s
                if v == "GT":
                    parts = ["1.00", "-", "-", "-"]
                else:
                    side = sides[v][labeled]
                    parts = [f"{(seg[region] >= 0).mean():.2f}"] + [f"{(side == c).mean():.2f}" for c in (1, 2, 0)]
                    results[(name, region_key, v, "side")] = [(side == c).mean() for c in (1, 2, 0)]
                print(
                    f"| {v} | " + " | ".join(parts) + f" | {s['over5']:.3f} | {s['mean']:.4f} "
                    f"| {s['dv_changed']:.4f} | {s['fix_changed']:.4f} |"
                )
            # only the pixels the fill adds: where top100 has no mask
            gap = (data["segs"]["SAM 3 top100"] < 0)[region]
            filled = run(data, region_key, "top100 + automatic", RADIUS)["plane"][region]
            ctrl = run(data, region_key, "top100 + automatic of next view", RADIUS)["plane"][region]
            for label, f in (("automatic", filled), ("next view's automatic", ctrl)):
                after = np.where(np.isnan(f[gap]), base[gap], f[gap])
                print(
                    f"pixels without a SAM 3 mask ({gap.mean():.2f} of the region), filled with {label}: changed "
                    f"{(~np.isnan(f[gap])).mean():.2f}, mean error {base[gap].mean():.4f} -> {after.mean():.4f}, "
                    f"share > 5% {(base[gap] > BLEED_ERROR).mean():.3f} -> {(after > BLEED_ERROR).mean():.3f}"
                )
            side = sides["top100 + automatic"][labeled]
            gap_l = (data["segs"]["SAM 3 top100"] < 0)[labeled]
            print(
                f"which object on labeled pixels without a SAM 3 mask, after the fill: right "
                f"{(side[gap_l] == 1).mean():.2f}, other {(side[gap_l] == 2).mean():.2f}, still none "
                f"{(side[gap_l] == 0).mean():.2f}"
            )
            # what the gaps are: slivers next to a mask, or whole objects without a mask
            near = np.stack([distance_transform_edt(t < 0) for t in top100])[region][gap]
            print(
                "distance of these pixels to the nearest SAM 3 mask: "
                + ", ".join(f"<= {d} px {(near <= d).mean():.2f}" for d in (1, 3, 10))
            )
            gap_px = region & (top100 < 0)
            classes = np.vectorize(lambda g: data["gt"].classes.get(int(g), "unlabeled"))(gt_ids[gap_px])
            names, counts = np.unique(classes, return_counts=True)
            order = np.argsort(-counts)[:8]
            print(
                "GT class of these pixels: "
                + ", ".join(f"{names[k]} {counts[k] / counts.sum():.2f}" for k in order)
            )
            print()
        results[(name, "data")] = data

    # figure 1: per scene, which object (realistic region) and the proxy fix
    colors = {
        "Déjà View": "#a8a7a2", "SAM 3 top100": "#2a78d6", "automatic only": "#e8a33a",
        "top100 + automatic": "#7a4fc9", "top100 + automatic of next view": "#d6453a", "GT": "#3a9a5b",
    }
    short = {
        "Déjà View": "Déjà View", "SAM 3 top100": "SAM 3\ntop100", "automatic only": "automatic\nonly",
        "top100 + automatic": "top100 +\nautomatic", "top100 + automatic of next view": "top100 +\nnext view's",
        "GT": "GT",
    }
    fig, axes = plt.subplots(2, 4, figsize=(20, 8))
    for row, name in zip(axes, SCENES.values(), strict=True):
        for j, region_key in enumerate(REGIONS):
            ax = row[2 * j]
            names = [v for v in VARIANTS if v != "GT"]
            bottom = np.zeros(len(names))
            for c, (label, color) in enumerate(
                (("right object", "#3a9a5b"), ("other object", "#d6453a"), ("no mask", "#a8a7a2"))
            ):
                vals = np.array([results[(name, region_key, v, "side")][c] for v in names])
                ax.bar(range(len(names)), vals, 0.6, bottom=bottom, color=color, label=label)
                bottom += vals
            ax.set_xticks(range(len(names)), [short[v] for v in names], fontsize=7)
            ax.set_ylim(0, 1)
            ax.legend(fontsize=7, frameon=False)
            ax.set_title(f"{name}, {region_key} region: a pixel's mask belongs to", fontsize=9)
            ax = row[2 * j + 1]
            region = (results[(name, "data")]["band"] if region_key == "realistic" else results[(name, "data")]["bleed"])
            base = results[(name, "data")]["error"][region & results[(name, "data")]["valid"]]
            names = ["Déjà View", *VARIANTS]
            vals = [base.mean()] + [results[(name, region_key, v)]["mean"] for v in VARIANTS]
            ax.bar(range(len(names)), vals, 0.65, color=[colors[n] for n in names])
            for x, val in enumerate(vals):
                ax.text(x, val, f"{val:.3f}", ha="center", va="bottom", fontsize=7)
            ax.set_xticks(range(len(names)), [short[n] for n in names], fontsize=7)
            ax.set_title(f"{name}, {region_key} region: mean error after the E20 proxy fix", fontsize=9)
    for ax in axes.flat:
        ax.grid(axis="y", color="#e5e4df", linewidth=0.6)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(
        "E21: SAM 2 automatic masks (no class, per view) fill the pixels without a SAM 3 mask. Proxy fix as E20 "
        "(plane in inverse depth, 21 px window, same-segment pixels outside the region)",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(FIGURES / "e21_automatic_masks.jpg", dpi=110, pil_kwargs={"quality": 90})

    # figure 2: office view with most bleeding: SAM 3, automatic, filled
    name = "office"
    data = results[(name, "data")]
    i = int(data["bleed"].reshape(len(data["bleed"]), -1).sum(1).argmax())
    image = Views.load(DATA / "7b6477cb95" / f"{VIEWS}.npz").images[i]
    rng = np.random.default_rng(0)
    palette = rng.integers(40, 255, size=(4096, 3))
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.2), layout="constrained")
    for ax, v in zip(axes, ("SAM 3 top100", "automatic only", "top100 + automatic"), strict=True):
        ids = data["segs"][v][i]
        over = image.astype(float) * 0.35
        has = ids >= 0
        over[has] = 0.45 * image[has] + 0.55 * palette[ids[has] % len(palette)]
        over[data["bleed"][i]] = (255, 0, 255)
        ax.imshow(over.astype(np.uint8))
        ax.set_title(f"{name} view {i}: {v} (dark: no mask, magenta: bleeding pixels)", fontsize=9)
        ax.axis("off")
    fig.savefig(FIGURES / "e21_example.jpg", dpi=110, pil_kwargs={"quality": 90})
    print("figures: experiments/figures/e21_automatic_masks.jpg, experiments/figures/e21_example.jpg")


if __name__ == "__main__":
    generate() if sys.argv[1:] == ["generate"] else evaluate()
