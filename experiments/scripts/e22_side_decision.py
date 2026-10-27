"""E22: side decision for edge pixels: which nearby segment's surface does a pixel take?

E20 fitted each region pixel only to its own mask, so pixels with no mask (slivers
along SAM 3 outlines, E21) or in the wrong mask could not be fixed. Here every
segment with interior pixels in the pixel's window is a candidate: its plane in
inverse depth is fitted exactly as E20 (same-segment pixels outside the region,
21 px window, >= 20 pixels), and the pixel takes the candidate plane whose value
is closest to its own Déjà View inverse depth. No-mask pixels get candidates too;
pixels with no candidate keep their Déjà View depth.

Rows per segmentation (none, SAM 3 top100, SAM 3 union, GT, shuffled):
- own mask: E20's rule (reference),
- side decision: candidate plane closest to the pixel's Déjà View inverse depth
  (absolute difference; registered before the first run, no GT used),
- side decision, log: closest by |log(plane / Déjà View)|, i.e. relative depth
  difference (added after the first run: the absolute difference in inverse depth
  favours far planes, see LOG E22),
- oracle pick: candidate with the lowest evaluated error (uses GT; the best any side
  decision among these candidates can do).
Regions and error exactly as E20. With "none" there is one candidate, so all three
rules give E20's "none" row (check).

Caveat: picking the closest plane can just snap back to the Déjà View value; the
script reports how far the depth moves, the share of pixels that change, and the
error on pixels whose own top100 mask belongs to another object.

uv run python experiments/scripts/e22_side_decision.py > experiments/results/e22_side_decision.txt
"""

import sys
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import uniform_filter

sys.path.insert(0, str(Path(__file__).parent))
from e20_proxy_fix import (  # noqa: E402
    BLEED_ERROR, FIGURES, MIN_NEIGHBOURS, RADIUS, REGIONS, SCENES, VARIANTS, fit, run, scene_data, summary,
)

from src.segmentation.base import NO_INSTANCE  # noqa: E402

RULES = ("own mask", "side decision", "side decision, log", "oracle pick")
SIDES = {"side decision": "side", "side decision, log": "side_log", "oracle pick": "oracle"}


def candidates(
    inv: np.ndarray, gt_inv: np.ndarray, region: np.ndarray, interior: np.ndarray, ids: np.ndarray, radius: int,
) -> dict[str, np.ndarray]:
    """Closest candidate plane to Déjà View (side) and to GT (oracle) for each region pixel."""
    size = 2 * radius + 1
    shape = inv.shape
    out = {k: np.full(shape, np.nan) for k in SIDES.values()}
    out |= {f"{k}_seg": np.full(shape, NO_INSTANCE) for k in SIDES.values()}
    out["count"] = np.zeros(shape, dtype=int)
    best = {k: np.full(shape, np.inf) for k in SIDES.values()}
    for s in np.unique(ids[interior]):
        if s == NO_INSTANCE:
            continue
        support = uniform_filter((interior & (ids == s)).astype(float), size, mode="constant") * size**2
        near = region & (support >= MIN_NEIGHBOURS - 0.5)
        if not near.any():
            continue
        # E20's fit with every nearby region pixel relabelled to s: region pixels are never interior,
        # so the support of s is unchanged and the plane of s is evaluated at all of them
        plane = fit(inv, near, interior, np.where(near, s, ids), radius)["plane"]
        ok = ~np.isnan(plane) & (plane > 0)
        out["count"] += ok
        safe = np.where(ok, plane, 1.0)
        dist = {
            "side": np.abs(safe - inv),  # as registered before the run
            "side_log": np.abs(np.log(safe / inv)),  # scale-free: relative depth difference either way
            "oracle": np.abs(gt_inv / safe - 1),  # exactly the evaluated relative depth error
        }
        for k, d in dist.items():
            better = ok & (d < best[k])
            best[k][better] = d[better]
            out[k][better] = plane[better]
            out[f"{k}_seg"][better] = s
    return out


def run_e22(data: dict, region_key: str, variant: str) -> dict[str, np.ndarray]:
    """Relative errors (NaN = unchanged) per rule, chosen segment, candidate count, depth change."""
    region = data["band"] if region_key == "realistic" else data["bleed"]
    blocked = data["band"] | data["bleed"]
    own = run(data, region_key, variant, RADIUS)["plane"]
    out = {"own mask": own} | {rule: np.full(region.shape, np.nan) for rule in SIDES}
    out |= {f"{rule} seg": np.full(region.shape, NO_INSTANCE) for rule in SIDES}
    out |= {f"{rule} moved": np.full(region.shape, np.nan) for rule in SIDES}
    out["count"] = np.zeros(region.shape, dtype=int)
    for i in range(len(region)):
        interior = ~(region[i] if region_key == "realistic" else blocked[i])
        inv = 1 / np.maximum(data["depth"][i], 1e-6)
        gt_depth = data["gt"].depth[i]
        gt_inv = data["scale"][i] / np.maximum(gt_depth, 1e-6)
        c = candidates(inv, gt_inv, region[i], interior, data["segs"][variant][i], RADIUS)
        for rule, key in SIDES.items():
            depth = data["scale"][i] / c[key]
            out[rule][i] = np.abs(depth - gt_depth) / np.maximum(gt_depth, 1e-6)
            out[f"{rule} seg"][i] = c[f"{key}_seg"]
            out[f"{rule} moved"][i] = np.abs(inv / c[key] - 1)  # relative depth change
        out["count"][i] = c["count"]
    return out


def dominant_object(seg: np.ndarray, gt_ids: np.ndarray, labeled: np.ndarray) -> dict[int, int]:
    """Segment id -> GT instance it mostly covers (labeled pixels), as E18's which-object label."""
    out = {}
    for s in np.unique(seg[labeled & (seg >= 0)]):
        g, c = np.unique(gt_ids[labeled & (seg == s)], return_counts=True)
        out[int(s)] = int(g[c.argmax()])
    return out


def main() -> None:
    results: dict = {}
    examples: dict = {}
    for scene, name in SCENES.items():
        data = scene_data(scene)
        valid = data["valid"]
        gt_ids = data["segs"]["GT"]
        print(f"## {name} ({scene})")
        print()
        for region_key, title in REGIONS.items():
            region = (data["band"] if region_key == "realistic" else data["bleed"]) & valid
            base = data["error"][region]
            dv = summary(base, np.full(base.shape, np.nan))
            print(f"### {title}; plane fit, window {2 * RADIUS + 1} px")
            print(
                "| segments | rule | changed | median error | mean error | share > 5% "
                "| changed px: Déjà View mean | fixed mean | better by > 1 pp | worse by > 1 pp |"
            )
            print("|---|---|---|---|---|---|---|---|---|---|")
            print(f"| Déjà View (input) | | 0.00 | {dv['median']:.4f} | {dv['mean']:.4f} | {dv['over5']:.3f} | | | | |")
            for variant in VARIANTS:
                r = results[(name, region_key, variant)] = run_e22(data, region_key, variant)
                for rule in RULES:
                    s = summary(base, r[rule][region])
                    results[(name, region_key, variant, rule)] = s
                    print(
                        f"| {variant} | {rule} | {s['changed']:.2f} | {s['median']:.4f} | {s['mean']:.4f} "
                        f"| {s['over5']:.3f} | {s['dv_changed']:.4f} | {s['fix_changed']:.4f} "
                        f"| {s['better']:.2f} | {s['worse']:.2f} |"
                    )
            print()

            # does the side decision just snap back to the Déjà View value?
            print("Candidates, depth change, chosen object (labeled GT pixels, E18 dominant-object rule):")
            print(
                "| segments | rule | median candidates | >= 2 candidates | median depth change | depth change > 1% "
                "| chosen = own mask | chosen right object | own mask right object |"
            )
            print("|---|---|---|---|---|---|---|---|---|")
            for variant in VARIANTS:
                r = results[(name, region_key, variant)]
                seg = data["segs"][variant]
                count = r["count"][region]
                doms = [dominant_object(seg[i], gt_ids[i], valid[i] & (gt_ids[i] >= 0)) for i in range(len(seg))]

                def right_share(ids: np.ndarray) -> float:
                    hits, total = 0, 0
                    for i, dom in enumerate(doms):
                        sel = region[i] & (gt_ids[i] >= 0) & (ids[i] >= 0)
                        mapped = np.array([dom.get(int(x), -9) for x in ids[i][sel]])
                        hits, total = hits + (mapped == gt_ids[i][sel]).sum(), total + sel.sum()
                    return hits / max(total, 1)

                own_right = right_share(seg)
                for rule in SIDES:
                    changed = ~np.isnan(r[rule][region])
                    moved = r[f"{rule} moved"][region][changed]
                    same = (r[f"{rule} seg"][region] == seg[region])[changed & (seg[region] >= 0)].mean()
                    print(
                        f"| {variant} | {rule} | {np.median(count):.0f} | {(count >= 2).mean():.2f} "
                        f"| {np.median(moved):.4f} | {(moved > 0.01).mean():.2f} | {same:.2f} "
                        f"| {right_share(r[f'{rule} seg']):.2f} | {own_right:.2f} |"
                    )
            print()

            # by E18's which-object label of the top100 mask (labeled pixels), no-mask = gap pixels
            for variant in ("SAM 3 top100", "GT"):
                r = results[(name, region_key, variant)]
                side = data["side"][region]
                labeled = (data["gt"].instance_ids >= 0)[region] | (side > 0)
                print(f"{variant}, by the which-object label of the top100 mask (changed share, mean error / share > 5%):")
                print("| top100 mask belongs to | pixels | Déjà View | " + " | ".join(RULES) + " |")
                print("|---|---|---|" + "---|" * len(RULES))
                for code, label in ((1, "right object"), (2, "other object"), (0, "no mask (gap)")):
                    sel = labeled & (side == code)
                    cells = []
                    for rule in RULES:
                        f = r[rule][region][sel]
                        after = np.where(np.isnan(f), base[sel], f)
                        cells.append(f"{(~np.isnan(f)).mean():.2f}: {after.mean():.4f} / {(after > BLEED_ERROR).mean():.3f}")
                    print(
                        f"| {label} | {sel.sum()} | {base[sel].mean():.4f} / {(base[sel] > BLEED_ERROR).mean():.3f} | "
                        + " | ".join(cells) + " |"
                    )
                print()
            # all gap pixels of the region (also on unlabeled GT)
            gap = data["segs"]["SAM 3 top100"][region] < 0
            print(f"Gap pixels (no top100 mask, {gap.mean():.2f} of the region; changed share, mean error / share > 5%):")
            print("| segments | Déjà View | " + " | ".join(SIDES) + " |")
            print("|---|---|" + "---|" * len(SIDES))
            for variant in ("SAM 3 top100", "SAM 3 union", "GT", "shuffled"):
                cells = []
                for rule in SIDES:
                    f = results[(name, region_key, variant)][rule][region][gap]
                    after = np.where(np.isnan(f), base[gap], f)
                    cells.append(f"{(~np.isnan(f)).mean():.2f}: {after.mean():.4f} / {(after > BLEED_ERROR).mean():.3f}")
                print(f"| {variant} | {base[gap].mean():.4f} / {(base[gap] > BLEED_ERROR).mean():.3f} | " + " | ".join(cells) + " |")
            print()
        examples[name] = data

    # figure 1: mean error and share > 5% per segmentation and rule
    colors = {"own mask": "#7fb1ea", "side decision": "#2a78d6", "side decision, log": "#1c4f8f", "oracle pick": "#3a9a5b"}
    width = 0.8 / len(RULES)
    fig, axes = plt.subplots(2, 4, figsize=(20, 8))
    for row, name in zip(axes, SCENES.values(), strict=True):
        data = examples[name]
        for j, region_key in enumerate(REGIONS):
            region = (data["band"] if region_key == "realistic" else data["bleed"]) & data["valid"]
            dv = summary(data["error"][region], np.full(region.sum(), np.nan))
            for k, (metric, label) in enumerate((("over5", "share of pixels with error > 5%"), ("mean", "mean relative error"))):
                ax = row[2 * j + k]
                for m, rule in enumerate(RULES):
                    vals = [results[(name, region_key, v, rule)][metric] for v in VARIANTS]
                    x = np.arange(len(VARIANTS)) + (m - (len(RULES) - 1) / 2) * width
                    ax.bar(x, vals, width, color=colors[rule], label=rule)
                    for xx, val in zip(x, vals, strict=True):
                        ax.text(xx, val, f"{val:.3f}", ha="center", va="bottom", fontsize=5, rotation=90)
                ax.axhline(dv[metric], color="#52514e", linestyle="--", linewidth=1, label="Déjà View (input)")
                ax.set_xticks(range(len(VARIANTS)), [v.replace("SAM 3 ", "SAM 3\n") for v in VARIANTS], fontsize=7)
                ax.set_title(f"{name}, {region_key} region: {label}", fontsize=9)
    axes[0, 0].legend(fontsize=7, frameon=False)
    for ax in axes.flat:
        ax.grid(axis="y", color="#e5e4df", linewidth=0.6)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(
        "E22: side decision near edges: each pixel takes the nearby segment plane closest to its Déjà View value (abs. inverse "
        f"depth / log ratio; {2 * RADIUS + 1} px window) vs E20's own-mask rule; oracle pick = candidate with the lowest GT error",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(FIGURES / "e22_side_decision.jpg", dpi=110, pil_kwargs={"quality": 90})

    # figure 2: office view with most bleeding, realistic region, SAM 3 top100 and GT
    name = "office"
    data = examples[name]
    i = int(data["bleed"].reshape(len(data["bleed"]), -1).sum(1).argmax())
    panels = [("Déjà View", data["error"][i])]
    for v, rule in (("SAM 3 top100", "own mask"), ("SAM 3 top100", "side decision, log"), ("GT", "side decision, log")):
        fixed = results[(name, "realistic", v)][rule][i]
        panels.append((f"{v}, {rule}", np.where(np.isnan(fixed), data["error"][i], fixed)))
    fig, axes = plt.subplots(1, 4, figsize=(20, 3.6), layout="constrained")
    for ax, (title, err) in zip(axes, panels, strict=True):
        im = ax.imshow(np.where(data["valid"][i], err, np.nan), cmap="magma", vmin=0, vmax=0.1)
        ax.set_title(f"{name} view {i}: {title} (share > 5%: {(err[data['valid'][i]] > BLEED_ERROR).mean():.3f})", fontsize=9)
        ax.axis("off")
    fig.colorbar(im, ax=axes, fraction=0.012, label="relative depth error")
    fig.savefig(FIGURES / "e22_example.jpg", dpi=110, pil_kwargs={"quality": 90})
    print("figures: experiments/figures/e22_side_decision.jpg, experiments/figures/e22_example.jpg")


if __name__ == "__main__":
    main()
