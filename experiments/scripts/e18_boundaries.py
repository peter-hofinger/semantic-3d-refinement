"""E18: do SAM 3 mask boundaries sit on the depth edges where Déjà View bleeds?

Per view, on pixels where the GT ray hits the mesh, three edge maps are built
(a pixel is an edge pixel if one of its 4 neighbours differs; both sides marked):
- GT object boundaries: GT instance id changes (unlabeled counts as its own id),
- GT depth edges: relative GT depth jump > 5% between neighbours,
- Déjà View depth edges: the same on the predicted depth.
SAM 3 boundaries are id changes of the working-setting segmentation (association
does not move boundaries). A map B "covers" a map A at tolerance d if an A pixel has
a B pixel within d px. Control: the SAM 3 boundaries of the next view, same density,
unrelated position. Bleeding: Déjà View depth error (per-view median scale) by
distance to the nearest GT depth edge, and how many bleeding pixels (error > 5%
within 3 px of a GT depth edge) have a SAM 3 boundary within 3 px.

Caveat: GT comes from the 5 cm mesh, so GT boundaries and depth edges are only
accurate to a few px (printed as "px per 5 cm").

uv run python experiments/scripts/e18_boundaries.py > experiments/results/e18_boundaries.txt

Optional: a different segmentation stem and a figure prefix (E19 reuses this script):
uv run python experiments/scripts/e18_boundaries.py <segmentation stem> <prefix>
"""

import sys
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import distance_transform_edt

from src.geometry import Geometry
from src.evaluation.instances import match_view
from src.loaders import Views
from src.loaders.scannetpp.gt import NO_MESH, GroundTruth
from src.segmentation import Segmentation
from src.segmentation.base import NO_INSTANCE

DATA = Path("data/processed/scannetpp")
FIGURES = Path("experiments/figures")
SCENES = {"5ee7c22ba0": "kitchen", "7b6477cb95": "office"}
VIEWS = "views_n20_uniform_504"
GEOMETRY = f"dejaview_k16_{VIEWS}"
SEGMENTATION = (
    sys.argv[1]
    if len(sys.argv) > 1
    else f"sam3image-top100-t0.3_{VIEWS}_assoc-xc0.7-vf0.25-kl0.5-{GEOMETRY}"  # E17 best
)
PREFIX = sys.argv[2] if len(sys.argv) > 2 else "e18"  # noqa: PLR2004
DEPTH_JUMP = 0.05  # relative depth change between neighbours that makes a depth edge
BLEED_ERROR = 0.05  # relative depth error that counts as bleeding (as in Peter's README)
BLEED_RANGE = 3  # px from a GT depth edge
CANNY = (50, 150)  # hysteresis thresholds on the 8-bit gray image
IMAGE_EDGE_RANGE = 2  # px: a GT depth edge counts as confirmed with an image edge this close
TOLERANCES = (1, 2, 3, 5, 8)
OFFSET_RANGE = 6  # px; mask/object differences further out are pooled in the last bin
EROSIONS = (0, 1, 2, 3)  # px masks are shrunk by before checking which object a pixel gets
DISTANCE_BINS = ((0, 1), (1, 3), (3, 7), (7, 15), (15, np.inf))


def edges(changed_h: np.ndarray, changed_v: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Mark both pixels of every valid neighbour pair flagged as changed."""
    out = np.zeros_like(valid)
    h = changed_h & valid[:, 1:] & valid[:, :-1]
    v = changed_v & valid[1:] & valid[:-1]
    out[:, 1:] |= h
    out[:, :-1] |= h
    out[1:] |= v
    out[:-1] |= v
    return out


def id_edges(ids: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Pixels where the id changes to a neighbour."""
    return edges(ids[:, 1:] != ids[:, :-1], ids[1:] != ids[:-1], valid)


def depth_jump(depth: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Largest relative depth change from each pixel to a valid 4-neighbour."""
    d = np.maximum(depth, 1e-6)
    h = np.abs(d[:, 1:] - d[:, :-1]) / np.minimum(d[:, 1:], d[:, :-1])
    v = np.abs(d[1:] - d[:-1]) / np.minimum(d[1:], d[:-1])
    h = np.where(valid[:, 1:] & valid[:, :-1], h, 0)
    v = np.where(valid[1:] & valid[:-1], v, 0)
    out = np.zeros(depth.shape)
    for sl_a, sl_b, j in (
        (np.s_[:, 1:], np.s_[:, :-1], h),
        (np.s_[1:], np.s_[:-1], v),
    ):
        out[sl_a] = np.maximum(out[sl_a], j)
        out[sl_b] = np.maximum(out[sl_b], j)
    return out


def depth_edges(depth: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Pixels where the depth jumps by more than DEPTH_JUMP to a neighbour."""
    return depth_jump(depth, valid) > DEPTH_JUMP


def distance_to(edge: np.ndarray) -> np.ndarray:
    """Distance in px from every pixel to the nearest edge pixel."""
    return distance_transform_edt(~edge) if edge.any() else np.full(edge.shape, np.inf)


def scene_maps(scene: str) -> dict[str, np.ndarray]:
    """Edge maps, distances and Déjà View errors of all views of one scene."""
    d = DATA / scene
    views = Views.load(d / f"{VIEWS}.npz")
    gt = GroundTruth.load(d / f"gt_{VIEWS}.npz")
    geometry = Geometry.load(d / f"{GEOMETRY}.npz")
    seg = Segmentation.load(d / f"{SEGMENTATION}.npz")
    split = [g for g, c in gt.classes.items() if c.lower() == "split"]
    gt_ids = np.where(np.isin(gt.instance_ids, split), NO_INSTANCE, gt.instance_ids)
    maps: dict[str, list[np.ndarray]] = {
        k: []
        for k in ("valid", "gt_obj", "gt_depth", "dv_depth", "sam", "error", "px5cm")
    }
    for i in range(len(views)):
        valid = (gt.instance_ids[i] != NO_MESH) & (gt.depth[i] > 0)
        maps["valid"].append(valid)
        maps["gt_obj"].append(id_edges(gt_ids[i], valid))
        maps["gt_depth"].append(depth_edges(gt.depth[i], valid))
        maps["dv_depth"].append(depth_edges(geometry.depth[i], valid))
        maps["sam"].append(id_edges(seg.instance_ids[i], valid))
        scale = np.median(gt.depth[i][valid] / geometry.depth[i][valid])
        error = np.abs(scale * geometry.depth[i] - gt.depth[i]) / np.maximum(
            gt.depth[i], 1e-6
        )
        maps["error"].append(np.where(valid, error, np.nan))
        fx = views.intrinsics[i][0, 0]
        maps["px5cm"].append(np.where(valid, fx * 0.05 / np.maximum(gt.depth[i], 1e-6), np.nan))
    out = {k: np.stack(v) for k, v in maps.items()}
    # GT depth edges confirmed by an image edge: drops mesh artifacts (holes and bumps
    # on dark or glossy surfaces such as screens) without looking at SAM 3 or Déjà View
    image_edges = np.stack(
        [cv2.Canny(cv2.cvtColor(im, cv2.COLOR_RGB2GRAY), *CANNY) > 0 for im in views.images]
    )
    near_image_edge = np.stack([distance_to(e) for e in image_edges]) <= IMAGE_EDGE_RANGE
    out["gt_depth_img"] = out["gt_depth"] & near_image_edge
    out["sam_control"] = np.roll(out["sam"], -1, axis=0) & out["valid"]
    out["sam_or_dv"] = out["sam"] | out["dv_depth"]
    # Déjà View depth edges as dense as the SAM 3 boundaries: same edge budget
    jumps = np.stack([depth_jump(g, v) for g, v in zip(geometry.depth, out["valid"], strict=True)])
    density = out["sam"][out["valid"]].mean()
    out["dv_threshold"] = np.quantile(jumps[out["valid"]], 1 - density)
    out["dv_dense"] = (jumps > out["dv_threshold"]) & out["valid"]
    for k in (
        "gt_obj", "gt_depth", "gt_depth_img", "dv_depth", "dv_dense", "sam", "sam_control",
        "sam_or_dv",
    ):
        out[f"dist_{k}"] = np.stack([distance_to(e) for e in out[k]])
    # side: is a pixel in a SAM 3 mask whose dominant GT object is the pixel's own?
    # 0 no mask, 1 right object, 2 other; also with masks eroded by k px (the mask's
    # object is decided before eroding)
    for k in EROSIONS:
        side = np.zeros(gt_ids.shape, dtype=np.int8)
        for i in range(len(views)):
            ids, own = seg.instance_ids[i], gt_ids[i]
            labeled = out["valid"][i] & (own >= 0)
            for p in np.unique(ids[labeled & (ids >= 0)]):
                in_mask = labeled & (ids == p)
                g, n = np.unique(own[in_mask], return_counts=True)
                keep = in_mask
                if k:
                    keep = in_mask & (distance_transform_edt(ids == p) > k)
                side[i][keep] = np.where(own[keep] == g[n.argmax()], 1, 2)
        out[f"side_eroded{k}"] = side
    # signed boundary offset of matched masks (IoU >= 0.5): pixels the SAM 3 mask adds
    # outside its GT object vs pixels of the object it misses, by distance to the GT outline
    offsets = np.zeros((2, OFFSET_RANGE + 1))  # [outside, inside] x distance 1..range
    for i in range(len(views)):
        for g, p, _ in match_view(gt_ids[i], seg.instance_ids[i]):
            obj = (gt_ids[i] == g) & out["valid"][i]
            mask = (seg.instance_ids[i] == p) & out["valid"][i]
            outside = distance_transform_edt(~obj)  # distance to the object from outside
            inside = distance_transform_edt(obj)  # distance to the background from inside
            for row, sel, dist in ((0, mask & ~obj, outside), (1, obj & ~mask, inside)):
                d = np.minimum(np.ceil(dist[sel]).astype(int), OFFSET_RANGE)
                offsets[row] += np.bincount(d, minlength=OFFSET_RANGE + 1)
    out["offsets"] = offsets
    out["side"] = out["side_eroded0"]
    out["labeled"] = out["valid"] & (gt_ids >= 0)
    out["gt_class"] = np.vectorize(lambda g: gt.classes.get(int(g), "-"))(gt_ids)
    out["images"] = views.images
    return out


def cover(m: dict[str, np.ndarray], target: str, by: str, d: float) -> float:
    """Share of `target` edge pixels with a `by` edge pixel within d px."""
    return float((m[f"dist_{by}"][m[target]] <= d).mean())


def in_bin(m: dict[str, np.ndarray], key: str, lo: float, hi: float) -> np.ndarray:
    """Valid pixels at distance (lo, hi] from a `key` edge; the first bin includes 0."""
    dist = m[f"dist_{key}"]
    return m["valid"] & (dist <= hi) & ((dist > lo) | (lo == 0))


SOURCES = {
    "gt_obj": "GT object boundaries",
    "sam": "SAM 3 boundaries",
    "dv_depth": "Déjà View depth edges",
    "dv_dense": "Déjà View depth edges, as dense as SAM 3",
    "sam_or_dv": "SAM 3 or Déjà View",
    "sam_control": "control: SAM 3 of next view",
}
EDGE_SETS = {
    "gt_depth": "All GT depth edges",
    "gt_depth_img": f"GT depth edges confirmed by an image edge (Canny, <= {IMAGE_EDGE_RANGE} px)",
}
results = {}
for scene, name in SCENES.items():
    m = scene_maps(scene)
    results[name] = m
    valid = m["valid"]
    print(f"## {name} ({scene})")
    print(
        f"evaluated pixels {valid.sum()} ({valid.mean():.0%} of all); "
        f"px per 5 cm at GT depth: median {np.nanmedian(m['px5cm']):.1f} "
        f"(10-90%: {np.nanpercentile(m['px5cm'], 10):.1f}-{np.nanpercentile(m['px5cm'], 90):.1f})"
    )
    print(
        "edge density (share of evaluated pixels): "
        + ", ".join(f"{SOURCES.get(k, k)} {m[k][valid].mean():.3f}"
                    for k in ("gt_depth", "gt_obj", "sam", "dv_depth"))
        + f"; dense Déjà View edges: jump > {m['dv_threshold']:.3f}"
    )
    print(
        f"GT depth edges on a GT object boundary (<= 1 px): "
        f"{cover(m, 'gt_depth', 'gt_obj', 1):.2f} (rest: inside one object or unlabeled)"
    )
    print(
        f"GT depth edges confirmed by an image edge (<= {IMAGE_EDGE_RANGE} px): "
        f"{m['gt_depth_img'].sum() / m['gt_depth'].sum():.2f}; of these on a GT object "
        f"boundary: {cover(m, 'gt_depth_img', 'gt_obj', 1):.2f}"
    )
    print()
    print("SAM 3 boundaries vs GT object boundaries (precision / recall / F):")
    print("| d | precision | recall | F | control precision | control recall |")
    print("|---|---|---|---|---|---|")
    for d in TOLERANCES:
        p, r = cover(m, "sam", "gt_obj", d), cover(m, "gt_obj", "sam", d)
        cp, cr = cover(m, "sam_control", "gt_obj", d), cover(m, "gt_obj", "sam_control", d)
        print(f"| {d} | {p:.2f} | {r:.2f} | {2 * p * r / max(p + r, 1e-9):.2f} | {cp:.2f} | {cr:.2f} |")
    print()
    off = m["offsets"]
    print(
        "Matched masks vs their GT object (IoU >= 0.5), pixels by distance to the GT "
        "outline (last bin: >= that distance):"
    )
    print("| | " + " | ".join(f"{d} px" for d in range(1, OFFSET_RANGE + 1)) + " | total |")
    print("|---|" + "---|" * (OFFSET_RANGE + 1))
    for row, label in ((0, "mask outside the object"), (1, "object outside the mask")):
        vals = " | ".join(f"{int(v)}" for v in off[row, 1:])
        print(f"| {label} | {vals} | {int(off[row].sum())} |")
    print(
        f"ratio outside / missed within 3 px: {off[0, 1:4].sum() / max(off[1, 1:4].sum(), 1):.2f}"
    )
    print()
    # which edge does SAM 3 follow: the GT depth edge or the Déjà View depth edge?
    near = m["sam"] & ((m["dist_gt_depth"] <= 5) | (m["dist_dv_depth"] <= 5))
    print(
        "Edge positions, median distance in px: "
        f"Déjà View depth edges -> GT depth edge {np.median(m['dist_gt_depth'][m['dv_depth']]):.1f}; "
        f"SAM 3 boundaries near a depth edge (<= 5 px, n={near.sum()}) -> GT depth edge "
        f"{np.median(m['dist_gt_depth'][near]):.1f}, -> Déjà View depth edge "
        f"{np.median(m['dist_dv_depth'][near]):.1f}; mean {m['dist_gt_depth'][near].mean():.2f} / "
        f"{m['dist_dv_depth'][near].mean():.2f}"
    )
    print()
    for key, edge_name in EDGE_SETS.items():
        print(f"### {edge_name}")
        print("Share of these edges covered (edge within d px):")
        print("| edges | " + " | ".join(f"d={d}" for d in TOLERANCES) + " |")
        print("|---|" + "---|" * len(TOLERANCES))
        for k, label in SOURCES.items():
            vals = " | ".join(f"{cover(m, key, k, d):.2f}" for d in TOLERANCES)
            print(f"| {label} | {vals} |")
        print()
        print("Déjà View depth error by distance to the nearest of these edges:")
        print("| distance px | pixels | median error | share > 5% |")
        print("|---|---|---|---|")
        err, dist = m["error"], m[f"dist_{key}"]
        for lo, hi in DISTANCE_BINS:
            e = err[in_bin(m, key, lo, hi)]
            print(
                f"| {lo}-{hi} | {e.size} | {np.median(e):.3f} "
                f"| {(e > BLEED_ERROR).mean():.2f} |"
            )
        bleed = valid & (dist <= BLEED_RANGE) & (err > BLEED_ERROR)
        print(
            f"bleeding pixels (error > 5%, <= {BLEED_RANGE} px from such an edge): "
            f"{bleed.sum()} = {bleed.sum() / max((valid & (err > BLEED_ERROR)).sum(), 1):.0%}"
            " of all pixels with error > 5%"
        )
        print("Share of bleeding pixels with an edge within d px:")
        print("| edges | " + " | ".join(f"d={d}" for d in TOLERANCES) + " |")
        print("|---|" + "---|" * len(TOLERANCES))
        for k, label in SOURCES.items():
            vals = " | ".join(
                f"{(m[f'dist_{k}'][bleed] <= d).mean():.2f}" for d in TOLERANCES
            )
            print(f"| {label} | {vals} |")
        m[f"bleed_{key}"] = bleed
        print()
        print(
            "Which object does SAM 3 assign a pixel to? (labeled GT pixels; right = the "
            "mask's dominant GT object is the pixel's own)"
        )
        print("| pixels | n | right object | other object | no mask |")
        print("|---|---|---|---|---|")
        for label, sel in (
            ("bleeding", bleed),
            (f"all <= {BLEED_RANGE} px from such an edge", valid & (dist <= BLEED_RANGE)),
            ("> 15 px from such an edge", valid & (dist > 15)),  # noqa: PLR2004
        ):
            s = m["side"][sel & m["labeled"]]
            m[f"side_{key}_{label.split()[0]}"] = s
            print(
                f"| {label} | {s.size} | {(s == 1).mean():.2f} | {(s == 2).mean():.2f} "
                f"| {(s == 0).mean():.2f} |"
            )
        print()
        print("Masks eroded by k px before assigning (labeled pixels):")
        print("| k | bleeding: right | other | no mask | > 15 px: right | other | no mask |")
        print("|---|---|---|---|---|---|---|")
        for k in EROSIONS:
            parts = []
            for sel in (bleed, valid & (dist > 15)):  # noqa: PLR2004
                s = m[f"side_eroded{k}"][sel & m["labeled"]]
                parts += [f"{(s == c).mean():.2f}" for c in (1, 2, 0)]
            print(f"| {k} | " + " | ".join(parts) + " |")
        print()
        print("Bleeding pixels by GT class (top 10 by count):")
        print("| GT class | pixels | share of bleeding | right object | other object | no mask |")
        print("|---|---|---|---|---|---|")
        sel = bleed & m["labeled"]
        names, counts = np.unique(m["gt_class"][sel], return_counts=True)
        for c in names[np.argsort(-counts)][:10]:
            s = m["side"][sel & (m["gt_class"] == c)]
            print(
                f"| {c} | {s.size} | {s.size / sel.sum():.2f} | {(s == 1).mean():.2f} "
                f"| {(s == 2).mean():.2f} | {(s == 0).mean():.2f} |"
            )
        print()

# figure 1: coverage curves and error vs distance, one row per scene
COLORS = {
    "gt_obj": "#52514e",
    "sam": "#2a78d6",
    "dv_depth": "#eb6834",
    "dv_dense": "#e8a33a",
    "sam_or_dv": "#3a9a5b",
    "sam_control": "#a8a7a2",
}
KEY = "gt_depth_img"  # figure 1 uses the image-confirmed GT depth edges
fig, axes = plt.subplots(2, 4, figsize=(20, 8))
for row, (name, m) in zip(axes, results.items(), strict=True):
    for ax, target, title in (
        (row[0], KEY, "confirmed GT depth edges covered"),
        (row[1], f"bleed_{KEY}", "bleeding pixels covered"),
    ):
        for k, label in SOURCES.items():
            vals = [(m[f"dist_{k}"][m[target]] <= d).mean() for d in TOLERANCES]
            style = "--" if k == "sam_control" else "-"
            ax.plot(TOLERANCES, vals, style, marker="o", color=COLORS[k], label=label)
        ax.set_ylim(0, 1)
        ax.set_xlabel("tolerance d (px)")
        ax.set_title(f"{name}: {title}", fontsize=9)
    ax = row[2]
    centers, medians, shares = [], [], []
    for lo, hi in DISTANCE_BINS:
        sel = in_bin(m, KEY, lo, hi)
        centers.append(f"{lo}-{hi:g}")
        medians.append(np.median(m["error"][sel]))
        shares.append((m["error"][sel] > BLEED_ERROR).mean())
    x = np.arange(len(centers))
    ax.bar(x - 0.18, medians, 0.36, color="#2a78d6", label="median error")
    ax.bar(x + 0.18, shares, 0.36, color="#eb6834", label="share > 5%")
    ax.set_xticks(x, centers)
    ax.set_xlabel("distance to nearest confirmed GT depth edge (px)")
    ax.set_title(f"{name}: Déjà View depth error vs distance to the edge", fontsize=9)
    ax = row[3]
    groups = [
        ("bleeding,\nall edges", m["side_gt_depth_bleeding"]),
        ("bleeding,\nconfirmed", m[f"side_{KEY}_bleeding"]),
        ("<= 3 px,\nconfirmed", m[f"side_{KEY}_all"]),
        ("> 15 px,\nconfirmed", m[f"side_{KEY}_>"]),
    ]
    bottom = np.zeros(len(groups))
    for code, label, color in (
        (1, "right object", "#3a9a5b"),
        (2, "other object", "#d6453a"),
        (0, "no mask", "#a8a7a2"),
    ):
        vals = np.array([(s == code).mean() for _, s in groups])
        ax.bar(range(len(groups)), vals, 0.6, bottom=bottom, color=color, label=label)
        bottom += vals
    ax.set_xticks(range(len(groups)), [g for g, _ in groups], fontsize=7)
    ax.set_ylim(0, 1)
    ax.set_title(f"{name}: SAM 3 mask of a pixel belongs to", fontsize=9)
for ax in axes.flat:
    ax.grid(axis="y", color="#e5e4df", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=7, frameon=False)
fig.suptitle(
    f"{PREFIX.upper()}: SAM 3 boundaries vs depth edges. GT from the 5 cm mesh; depth edges = relative "
    "jump > 5%, confirmed = image edge within 2 px; bleeding = Déjà View error > 5% within "
    f"3 px of an edge. Segmentation: {SEGMENTATION.split('_')[0]} + a + b",
    fontsize=10,
)
fig.tight_layout()
fig.savefig(FIGURES / f"{PREFIX}_boundaries.jpg", dpi=110, pil_kwargs={"quality": 90})

# figure 2: the view with the most bleeding pixels per scene, edges overlaid
fig, axes = plt.subplots(2, 2, figsize=(14, 7))
for row, (name, m) in zip(axes, results.items(), strict=True):
    bleed = m["bleed_gt_depth"]
    i = int(bleed.reshape(len(bleed), -1).sum(1).argmax())
    img = m["images"][i].astype(float) * 0.6
    over = img.copy()
    over[m["gt_depth"][i]] = (255, 150, 150)
    over[m["gt_depth_img"][i]] = (255, 40, 40)
    over[m["sam"][i]] = (40, 170, 255)
    over[m["gt_depth"][i] & m["sam"][i]] = (255, 255, 255)
    row[0].imshow(over.astype(np.uint8))
    row[0].set_title(
        f"{name} view {i}: GT depth edges (red; pink: no image edge), SAM 3 (blue), both (white)",
        fontsize=9,
    )
    over = img.copy()
    over[m["dv_depth"][i]] = (255, 170, 40)
    over[bleed[i]] = (255, 40, 255)
    row[1].imshow(over.astype(np.uint8))
    row[1].set_title(
        f"{name} view {i}: Déjà View depth edges (orange), bleeding pixels (magenta)",
        fontsize=9,
    )
for ax in axes.flat:
    ax.axis("off")
fig.tight_layout()
fig.savefig(FIGURES / f"{PREFIX}_example.jpg", dpi=110, pil_kwargs={"quality": 90})
print(f"figures: {FIGURES / f'{PREFIX}_boundaries.jpg'}, {FIGURES / f'{PREFIX}_example.jpg'}")
