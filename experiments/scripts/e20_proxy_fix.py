"""E20: can segments fix Déjà View's bleeding? A crude proxy for the refiner.

Region pixels get their Déjà View depth replaced by a fit to nearby pixels of the
same segment that lie outside the region ("interior"): a plane in inverse depth
(1/z is linear in pixel coordinates on a plane) or the mean inverse depth, over a
(2R+1)^2 window, only with >= MIN_NEIGHBOURS interior pixels; otherwise the pixel
keeps its Déjà View depth (as does every pixel with no segment).

Regions:
- realistic: within BAND px of Déjà View's own depth jumps (E18's dense threshold,
  as dense as the SAM 3 boundaries); uses no GT, contains correct pixels too,
- oracle: E18's bleeding pixels (error > 5% within 3 px of a GT depth edge); the
  fit then uses pixels outside both regions.
Segments: none (one segment for the whole image = plain local smoothing), SAM 3
top100 and top100 + VLM union (working setting), GT instances, shuffled (top100
masks of the next view: same statistics, unrelated position).
Error as in E18: |s * depth - GT| / GT with the per-view median scale s of Déjà View.

Caveat: GT depth and GT instances come from the same 5 cm mesh, so GT segments are
aligned with the GT depth edges by construction (favours the GT row).

uv run python experiments/scripts/e20_proxy_fix.py > experiments/results/e20_proxy_fix.txt
"""

from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import distance_transform_edt, uniform_filter

from src.geometry import Geometry
from src.loaders.scannetpp.gt import NO_MESH, GroundTruth
from src.segmentation import Segmentation
from src.segmentation.base import NO_INSTANCE

DATA = Path("data/processed/scannetpp")
FIGURES = Path("experiments/figures")
SCENES = {"5ee7c22ba0": "kitchen", "7b6477cb95": "office"}
VIEWS = "views_n20_uniform_504"
GEOMETRY = f"dejaview_k16_{VIEWS}"
ASSOC = f"_assoc-xc0.7-vf0.25-kl0.5-{GEOMETRY}"
TOP100 = f"sam3image-top100-t0.3_{VIEWS}{ASSOC}"
UNION = f"sam3image-top100+vlm-qwen3vl2binstruct_{VIEWS}-t0.3_{VIEWS}{ASSOC}"
DEPTH_JUMP = 0.05  # GT depth edge, as E18
BLEED_ERROR = 0.05
BLEED_RANGE = 3
BAND = 3  # px around Déjà View's dense depth jumps
RADIUS = 10  # fit window half size, px
MIN_NEIGHBOURS = 20
RADII = (5, 10, 20)  # sensitivity, plane fit only
VARIANTS = ("none", "SAM 3 top100", "SAM 3 union", "GT", "shuffled")
FITS = ("plane", "mean")


def depth_jump(depth: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Largest relative depth change from each pixel to a valid 4-neighbour (as E18)."""
    d = np.maximum(depth, 1e-6)
    h = np.abs(d[:, 1:] - d[:, :-1]) / np.minimum(d[:, 1:], d[:, :-1])
    v = np.abs(d[1:] - d[:-1]) / np.minimum(d[1:], d[:-1])
    h = np.where(valid[:, 1:] & valid[:, :-1], h, 0)
    v = np.where(valid[1:] & valid[:-1], v, 0)
    out = np.zeros(depth.shape)
    for sl_a, sl_b, j in ((np.s_[:, 1:], np.s_[:, :-1], h), (np.s_[1:], np.s_[:-1], v)):
        out[sl_a] = np.maximum(out[sl_a], j)
        out[sl_b] = np.maximum(out[sl_b], j)
    return out


def id_edges(ids: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Pixels where the id changes to a valid 4-neighbour (as E18)."""
    out = np.zeros_like(valid)
    h = (ids[:, 1:] != ids[:, :-1]) & valid[:, 1:] & valid[:, :-1]
    v = (ids[1:] != ids[:-1]) & valid[1:] & valid[:-1]
    out[:, 1:] |= h
    out[:, :-1] |= h
    out[1:] |= v
    out[:-1] |= v
    return out


def distance_to(edge: np.ndarray) -> np.ndarray:
    return distance_transform_edt(~edge) if edge.any() else np.full(edge.shape, np.inf)


def fit(
    inv: np.ndarray, region: np.ndarray, interior: np.ndarray, ids: np.ndarray, radius: int,
) -> dict[str, np.ndarray]:
    """Inverse depth of region pixels fitted to same-segment interior pixels nearby.

    Window sums come from box filters per segment (cropped to the segment's region
    pixels plus the window); the plane is solved in coordinates centred on each
    target pixel, so its value there is the constant term. NaN = not fitted.
    """
    out = {f: np.full(inv.shape, np.nan) for f in FITS}
    size = 2 * radius + 1
    h, w = inv.shape
    for s in np.unique(ids[region]):
        if s == NO_INSTANCE:
            continue
        target = region & (ids == s)
        rows, cols = np.nonzero(target)
        r0, r1 = max(rows.min() - radius, 0), min(rows.max() + radius + 1, h)
        c0, c1 = max(cols.min() - radius, 0), min(cols.max() + radius + 1, w)
        wgt = (interior & (ids == s))[r0:r1, c0:c1].astype(float)
        if not wgt.any():
            continue
        y = inv[r0:r1, c0:c1]
        v, u = np.mgrid[r0:r1, c0:c1].astype(float) / radius
        terms = {
            "1": wgt, "u": wgt * u, "v": wgt * v, "uu": wgt * u * u, "uv": wgt * u * v,
            "vv": wgt * v * v, "y": wgt * y, "uy": wgt * u * y, "vy": wgt * v * y,
        }
        tr, tc = rows - r0, cols - c0
        m = {k: uniform_filter(t, size, mode="constant")[tr, tc] * size**2 for k, t in terms.items()}
        n = m["1"]
        ok = n >= MIN_NEIGHBOURS
        out["mean"][rows[ok], cols[ok]] = m["y"][ok] / n[ok]
        # centre on the target pixel (u0, v0): moments of (u - u0), (v - v0)
        u0, v0 = cols / radius, rows / radius
        su, sv = m["u"] - u0 * n, m["v"] - v0 * n
        suu = m["uu"] - 2 * u0 * m["u"] + u0**2 * n
        svv = m["vv"] - 2 * v0 * m["v"] + v0**2 * n
        suv = m["uv"] - u0 * m["v"] - v0 * m["u"] + u0 * v0 * n
        suy, svy = m["uy"] - u0 * m["y"], m["vy"] - v0 * m["y"]
        a = np.stack(
            [np.stack([suu, suv, su], -1), np.stack([suv, svv, sv], -1), np.stack([su, sv, n], -1)],
            -2,
        )
        b = np.stack([suy, svy, m["y"]], -1)
        det = np.linalg.det(a)
        good = ok & (np.abs(det) > 1e-6 * np.maximum(n, 1) ** 3)
        plane = np.full(len(rows), np.nan)
        if good.any():
            plane[good] = np.linalg.solve(a[good], b[good][..., None])[:, 2, 0]
        # degenerate neighbourhoods (a line of pixels): fall back to the mean
        plane = np.where(ok & ~good, m["y"] / np.maximum(n, 1), plane)
        out["plane"][rows, cols] = plane
    return out


def scene_data(scene: str) -> dict:
    d = DATA / scene
    gt = GroundTruth.load(d / f"gt_{VIEWS}.npz")
    geometry = Geometry.load(d / f"{GEOMETRY}.npz")
    top100 = Segmentation.load(d / f"{TOP100}.npz").instance_ids
    union = Segmentation.load(d / f"{UNION}.npz").instance_ids
    split = [g for g, c in gt.classes.items() if c.lower() == "split"]
    gt_ids = np.where(np.isin(gt.instance_ids, split), NO_INSTANCE, gt.instance_ids)
    n = len(gt_ids)
    valid = (gt.instance_ids != NO_MESH) & (gt.depth > 0)
    # dense Déjà View threshold exactly as E18: as dense as the top100 SAM 3 boundaries
    sam_edges = np.stack([id_edges(top100[i], valid[i]) for i in range(n)])
    jumps_valid = np.stack([depth_jump(geometry.depth[i], valid[i]) for i in range(n)])
    threshold = np.quantile(jumps_valid[valid], 1 - sam_edges[valid].mean())
    everywhere = np.ones(gt_ids.shape[1:], dtype=bool)
    band = np.stack(
        [distance_to(depth_jump(g, everywhere) > threshold) <= BAND for g in geometry.depth]
    )
    scale = np.array([np.median(gt.depth[i][valid[i]] / geometry.depth[i][valid[i]]) for i in range(n)])
    gt_edges = np.stack([distance_to(depth_jump(gt.depth[i], valid[i]) > DEPTH_JUMP) for i in range(n)])
    error = np.abs(scale[:, None, None] * geometry.depth - gt.depth) / np.maximum(gt.depth, 1e-6)
    bleed = valid & (gt_edges <= BLEED_RANGE) & (error > BLEED_ERROR)
    segs = {
        "none": np.zeros_like(gt_ids),
        "SAM 3 top100": top100,
        "SAM 3 union": union,
        "GT": gt_ids,
        "shuffled": np.roll(top100, -1, axis=0),
    }
    # E18's "which object": does the pixel's top100 mask have the pixel's own GT object as its dominant one?
    side = np.zeros(gt_ids.shape, dtype=np.int8)
    for i in range(n):
        labeled = valid[i] & (gt_ids[i] >= 0)
        for p in np.unique(top100[i][labeled & (top100[i] >= 0)]):
            in_mask = labeled & (top100[i] == p)
            g, c = np.unique(gt_ids[i][in_mask], return_counts=True)
            side[i][in_mask] = np.where(gt_ids[i][in_mask] == g[c.argmax()], 1, 2)
    return {
        "gt": gt, "depth": geometry.depth, "valid": valid, "scale": scale, "error": error,
        "band": band, "bleed": bleed, "segs": segs, "threshold": threshold, "side": side,
        "images": None,
    }


def run(data: dict, region_key: str, variant: str, radius: int) -> dict[str, np.ndarray]:
    """Fixed relative errors (NaN = unchanged) per fit, all views stacked."""
    region = data["band"] if region_key == "realistic" else data["bleed"]
    blocked = data["band"] | data["bleed"]  # the oracle fit must not use bleeding pixels
    out = {f: np.full(region.shape, np.nan) for f in FITS}
    for i in range(len(region)):
        interior = ~(region[i] if region_key == "realistic" else blocked[i])
        fitted = fit(1 / np.maximum(data["depth"][i], 1e-6), region[i], interior, data["segs"][variant][i], radius)
        gt_depth = data["gt"].depth[i]
        for f in FITS:
            depth = data["scale"][i] / np.where(fitted[f] > 0, fitted[f], np.nan)
            out[f][i] = np.abs(depth - gt_depth) / np.maximum(gt_depth, 1e-6)
    return out


def summary(base: np.ndarray, fixed: np.ndarray) -> dict[str, float]:
    changed = ~np.isnan(fixed)
    after = np.where(changed, fixed, base)
    return {
        "changed": changed.mean(),
        "median": np.median(after),
        "mean": after.mean(),
        "over5": (after > BLEED_ERROR).mean(),
        "dv_changed": base[changed].mean() if changed.any() else np.nan,
        "fix_changed": fixed[changed].mean() if changed.any() else np.nan,
        "better": (fixed[changed] < base[changed] - 0.01).mean() if changed.any() else np.nan,
        "worse": (fixed[changed] > base[changed] + 0.01).mean() if changed.any() else np.nan,
    }


REGIONS = {
    "realistic": f"realistic region: <= {BAND} px from Déjà View's own dense depth jumps (no GT)",
    "oracle": "oracle region: E18 bleeding pixels (error > 5% within 3 px of a GT depth edge)",
}
results: dict = {}
examples: dict = {}
for scene, name in SCENES.items():
    data = scene_data(scene)
    valid = data["valid"]
    print(f"## {name} ({scene})")
    print(
        f"dense Déjà View jump threshold {data['threshold']:.4f}; realistic region "
        f"{data['band'][valid].mean():.1%} of evaluated pixels, contains "
        f"{(data['band'] & data['bleed']).sum() / data['bleed'].sum():.0%} of the bleeding pixels; "
        f"oracle region {data['bleed'][valid].mean():.1%}"
    )
    print()
    for region_key, title in REGIONS.items():
        region = (data["band"] if region_key == "realistic" else data["bleed"]) & valid
        base = data["error"][region]
        for f in FITS:
            print(f"### {title}; fit: {f}, window {2 * RADIUS + 1} px")
            print(
                "| segments | changed | median error | mean error | share > 5% "
                "| changed px: Déjà View mean | fixed mean | better by > 1 pp | worse by > 1 pp |"
            )
            print("|---|---|---|---|---|---|---|---|---|")
            s = summary(base, np.full(base.shape, np.nan))
            print(f"| Déjà View (input) | 0.00 | {s['median']:.4f} | {s['mean']:.4f} | {s['over5']:.3f} | | | | |")
            for variant in VARIANTS:
                key = (name, region_key, variant, RADIUS)
                if key not in results:
                    results[key] = run(data, region_key, variant, RADIUS)
                s = summary(base, results[key][f][region])
                results[key + (f, "summary")] = s
                print(
                    f"| {variant} | {s['changed']:.2f} | {s['median']:.4f} | {s['mean']:.4f} "
                    f"| {s['over5']:.3f} | {s['dv_changed']:.4f} | {s['fix_changed']:.4f} "
                    f"| {s['better']:.2f} | {s['worse']:.2f} |"
                )
            print()
        # same pixels: where none and SAM 3 top100 both change something
        none = results[(name, region_key, "none", RADIUS)]["plane"][region]
        sam = results[(name, region_key, "SAM 3 top100", RADIUS)]["plane"][region]
        gtv = results[(name, region_key, "GT", RADIUS)]["plane"][region]
        both = ~np.isnan(none) & ~np.isnan(sam)
        allthree = both & ~np.isnan(gtv)
        print(
            f"Plane fit, pixels changed by none and SAM 3 top100 ({both.mean():.2f} of the region): "
            f"mean error Déjà View {base[both].mean():.4f}, none {none[both].mean():.4f}, "
            f"SAM 3 {sam[both].mean():.4f}; share > 5%: {(base[both] > BLEED_ERROR).mean():.3f} / "
            f"{(none[both] > BLEED_ERROR).mean():.3f} / {(sam[both] > BLEED_ERROR).mean():.3f}; "
            f"also by GT ({allthree.mean():.2f}): GT {gtv[allthree].mean():.4f} vs SAM 3 "
            f"{sam[allthree].mean():.4f} vs none {none[allthree].mean():.4f}"
        )
        side = data["side"][region]
        print("Plane fit with SAM 3 top100 by E18's which-object label (labeled GT pixels):")
        print("| SAM 3 mask belongs to | pixels | changed | Déjà View mean | after mean | share > 5% before | after |")
        print("|---|---|---|---|---|---|---|")
        labeled = (data["gt"].instance_ids >= 0)[region] | (side > 0)
        for code, label in ((1, "right object"), (2, "other object"), (0, "no mask")):
            sel = labeled & (side == code)
            after = np.where(np.isnan(sam[sel]), base[sel], sam[sel])
            print(
                f"| {label} | {sel.sum()} | {(~np.isnan(sam[sel])).mean():.2f} | {base[sel].mean():.4f} "
                f"| {after.mean():.4f} | {(base[sel] > BLEED_ERROR).mean():.3f} | {(after > BLEED_ERROR).mean():.3f} |"
            )
        print()
        print(f"Window sensitivity (plane fit, share > 5% in the region; Déjà View {(base > BLEED_ERROR).mean():.3f}):")
        print("| segments | " + " | ".join(f"{2 * r + 1} px" for r in RADII) + " |")
        print("|---|" + "---|" * len(RADII))
        for variant in VARIANTS:
            vals = []
            for r in RADII:
                key = (name, region_key, variant, r)
                if key not in results:
                    results[key] = run(data, region_key, variant, r)
                vals.append(f"{summary(base, results[key]['plane'][region])['over5']:.3f}")
            print(f"| {variant} | " + " | ".join(vals) + " |")
        print()
    examples[name] = data

# figure 1: share > 5% and mean error per variant, plane fit
colors = {
    "Déjà View": "#a8a7a2", "none": "#52514e", "SAM 3 top100": "#2a78d6",
    "SAM 3 union": "#7fb1ea", "GT": "#3a9a5b", "shuffled": "#d6453a",
}
fig, axes = plt.subplots(2, 4, figsize=(20, 8))
for row, name in zip(axes, SCENES.values(), strict=True):
    data = examples[name]
    for j, region_key in enumerate(REGIONS):
        region = (data["band"] if region_key == "realistic" else data["bleed"]) & data["valid"]
        base = data["error"][region]
        for k, (metric, label) in enumerate((("over5", "share of pixels with error > 5%"), ("mean", "mean relative error"))):
            ax = row[2 * j + k]
            names = ["Déjà View", *VARIANTS]
            vals = [summary(base, np.full(base.shape, np.nan))[metric]] + [
                results[(name, region_key, v, RADIUS, "plane", "summary")][metric] for v in VARIANTS
            ]
            ax.bar(range(len(names)), vals, 0.65, color=[colors[n] for n in names])
            for x, val in enumerate(vals):
                ax.text(x, val, f"{val:.3f}", ha="center", va="bottom", fontsize=7)
            ax.set_xticks(range(len(names)), [n.replace("SAM 3 ", "SAM 3\n") for n in names], fontsize=7)
            ax.set_title(f"{name}, {region_key} region: {label}", fontsize=9)
for ax in axes.flat:
    ax.grid(axis="y", color="#e5e4df", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
fig.suptitle(
    "E20: proxy fix of Déjà View depth near edges: plane in inverse depth fitted to same-segment pixels outside the "
    f"region ({2 * RADIUS + 1} px window). Realistic region: <= {BAND} px from Déjà View depth jumps; oracle: GT bleeding pixels",
    fontsize=10,
)
fig.tight_layout()
fig.savefig(FIGURES / "e20_proxy_fix.jpg", dpi=110, pil_kwargs={"quality": 90})

# figure 2: error maps of the office view with most bleeding, realistic region
name = "office"
data = examples[name]
i = int(data["bleed"].reshape(len(data["bleed"]), -1).sum(1).argmax())
panels = [("Déjà View", data["error"][i])]
for v in ("none", "SAM 3 top100", "GT"):
    fixed = results[(name, "realistic", v, RADIUS)]["plane"][i]
    panels.append((v, np.where(np.isnan(fixed), data["error"][i], fixed)))
fig, axes = plt.subplots(1, 4, figsize=(20, 3.6), layout="constrained")
for ax, (title, err) in zip(axes, panels, strict=True):
    im = ax.imshow(np.where(data["valid"][i], err, np.nan), cmap="magma", vmin=0, vmax=0.1)
    ax.set_title(f"{name} view {i}: {title} (share > 5%: {(err[data['valid'][i]] > BLEED_ERROR).mean():.3f})", fontsize=9)
    ax.axis("off")
fig.colorbar(im, ax=axes, fraction=0.012, label="relative depth error")
fig.savefig(FIGURES / "e20_example.jpg", dpi=110, pil_kwargs={"quality": 90})
print("figures: experiments/figures/e20_proxy_fix.jpg, experiments/figures/e20_example.jpg")
