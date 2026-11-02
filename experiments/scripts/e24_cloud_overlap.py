"""E24: 3D point-cloud overlap (Replicate Any Scene, Eqs. 1-2) vs our reprojection.

Same per-view masks (top100, threshold 0.3), same linking (cross-class 0.7,
multi-view filter kl 0.5, greedy, one mask per view), only the overlap measure
changes. Setups fixed before running:
  reproject + vf0.25          working setting (must equal the E17 cache)
  reproject                   no visibility rule (must equal the E17 cache)
  cloud                       RAS overlap, both directions >= threshold (ours)
  cloud, either direction     RAS overlap with RAS's own rule (reference only)
  ... + GT geometry           reproject + vf0.25 and cloud with rendered GT depth
For every setup: the scorer, the merges by GT class, and the mask pairs linked
into one object although they lie mostly on different GT objects (wrong links).

uv run python experiments/scripts/e24_cloud_overlap.py | tee experiments/results/e24_cloud_overlap.txt
"""

import time
from collections import defaultdict
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree

from src.evaluation.instances import STRUCTURE, match_view, overlap_counts, score
from src.geometry import Geometry
from src.loaders import Views
from src.loaders.scannetpp.gt import GroundTruth
from src.segmentation import Segmentation
from src.segmentation.associate import Associator, gt_geometry

DATA = Path("data/processed/scannetpp")
FIGURE = Path("experiments/figures/e24_cloud_overlap.jpg")
SCENES = {"5ee7c22ba0": "kitchen", "7b6477cb95": "office"}
VIEWS = "views_n20_uniform_504"
GEOMETRY = "dejaview_k16_views_n20_uniform_504"
STEM = "sam3image-top100-t0.3_views_n20_uniform_504"
# label -> (Associator settings, geometry, cache tag)
SETUPS = {
    "reproject + vf0.25": (
        {"min_visible_fraction": 0.25},
        "dv",
        "xc0.7-vf0.25-kl0.5",
    ),
    "reproject": ({}, "dv", "xc0.7-kl0.5"),
    "cloud": ({"overlap_rule": "cloud"}, "dv", "xc0.7-kl0.5-pc"),
    "cloud, either direction": (
        {"overlap_rule": "cloud", "direction_rule": "max"},
        "dv",
        "xc0.7-kl0.5-pc-or",
    ),
    "reproject + vf0.25, GT geometry": (
        {"min_visible_fraction": 0.25},
        "gt",
        "xc0.7-vf0.25-kl0.5",
    ),
    "cloud, GT geometry": ({"overlap_rule": "cloud"}, "gt", "xc0.7-kl0.5-pc"),
}


def dominant_gt(gt: GroundTruth, seg: Segmentation) -> dict[int, int]:
    """Per-view mask -> GT object covering > 50% of its labeled pixels (or -1)."""
    out = {}
    for i in range(len(gt.instance_ids)):
        g_ids, p_ids, inter = overlap_counts(gt.instance_ids[i], seg.instance_ids[i])
        labeled = inter[g_ids >= 0]
        for col, p in enumerate(p_ids.tolist()):
            if p < 0:
                continue
            total = labeled[:, col].sum()
            best = labeled[:, col].argmax() if total else 0
            ok = total and labeled[best, col] > total / 2
            out[p] = int(g_ids[g_ids >= 0][best]) if ok else -1
    return out


def wrong_links(
    seg: Segmentation, linked: Segmentation, dom: dict[int, int], gt: GroundTruth
) -> list[str]:
    """Objects joining masks that lie mostly on different GT objects."""
    members: dict[int, set[int]] = defaultdict(set)
    for i in range(len(seg.instance_ids)):
        a, b = seg.instance_ids[i].ravel(), linked.instance_ids[i].ravel()
        keep = (a >= 0) & (b >= 0)
        for m, o in set(zip(a[keep].tolist(), b[keep].tolist(), strict=True)):
            members[o].add(m)
    out = []
    for o, ms in sorted(members.items()):
        gts = {dom[m] for m in ms if dom.get(m, -1) >= 0}
        if len(gts) > 1:
            names = sorted(f"{gt.classes.get(g, '?')} {g}" for g in gts)
            out.append(f"obj {o} ({linked.classes[o]}): " + ", ".join(names))
    return out


def merges(gt: GroundTruth, linked: Segmentation) -> list[str]:
    """Predicted ids matched (IoU >= 0.5) to more than one GT instance."""
    gts: dict[int, set[int]] = defaultdict(set)
    for i in range(len(gt.instance_ids)):
        for g, p, _ in match_view(gt.instance_ids[i], linked.instance_ids[i]):
            gts[p].add(g)
    return [
        f"{linked.classes[p]} {p} -> "
        + ", ".join(sorted(f"{gt.classes.get(g, '?')} {g}" for g in gs))
        for p, gs in sorted(gts.items())
        if len(gs) > 1
    ]


rows = {}
details = []
print(
    "| scene | overlap | subset | recall | pixel recall | precision | cross-class "
    "leakage | consistent | GT in >=2 views | id switches | merges | wrong links |"
)
print("|---|---|---|---|---|---|---|---|---|---|---|---|")
for scene, name in SCENES.items():
    d = DATA / scene
    gt = GroundTruth.load(d / f"gt_{VIEWS}.npz")
    geoms = {
        "dv": Geometry.load(d / f"{GEOMETRY}.npz"),
        "gt": gt_geometry(Views.load(d / f"{VIEWS}.npz"), gt),
    }
    seg = Segmentation.load(d / f"{STEM}.npz")
    dom = dominant_gt(gt, seg)
    for label, (settings, geom, tag) in SETUPS.items():
        assoc = Associator(cross_class_overlap=0.7, keep_linked_below=0.5, **settings)
        start = time.perf_counter()
        linked = assoc.associate(seg, geoms[geom])
        seconds = time.perf_counter() - start
        geom_tag = GEOMETRY if geom == "dv" else "gt"
        # cache names as the associate CLI gives them
        parts = tag.split("-")
        cli = [p for p in parts if p.startswith("xc")]
        cli += [p for p in parts if p.startswith("vf")]
        cli += [p for p in parts if p.startswith("kl")]
        cli += [p for p in parts if p == "pc"] + [p for p in parts if p == "or"]
        out = d / f"{STEM}_assoc-{'-'.join(cli)}-{geom_tag}.npz"
        check = "new"
        if out.exists():
            old = Segmentation.load(out)
            assert np.array_equal(old.instance_ids, linked.instance_ids), out
            assert old.classes == linked.classes, out
            check = "same as cache"
        linked.save(out)
        wrong = wrong_links(seg, linked, dom, gt)
        rows[scene, label] = {}
        for subset, exclude in (("all", ()), ("objects", STRUCTURE)):
            s = score(gt, linked, exclude)
            rows[scene, label][subset] = (s, len(wrong))
            print(
                f"| {name} | {label} | {subset} | {s.recall:.2f} | "
                f"{s.pixel_recall:.2f} | {s.precision:.2f} | {s.cross_leakage:.3f} | "
                f"{s.consistent:.2f} | {s.num_gt} | {s.id_switches} | {s.merges} | "
                f"{len(wrong) if subset == 'all' else ''} |"
            )
        details.append(
            (
                (
                    f"{name} / {label}: {len(linked.classes)} objects, "
                    f"{seconds:.0f}s, {check} ({out.name})"
                ),
                merges(gt, linked),
                wrong,
            )
        )

print("\nRuns, merges (by IoU match) and wrong links (masks on different GT objects):")
for head, ms, ws in details:
    print(f"\n{head}")
    for m in ms:
        print(f"  merge: {m}")
    for w in ws:
        print(f"  wrong link: {w}")

# tau of the cloud rule, to put the distances in context
print("\nCloud rule: tau = 3 x mean nearest-neighbour distance inside the mask's cloud")
for scene, name in SCENES.items():
    d = DATA / scene
    geometry = Geometry.load(d / f"{GEOMETRY}.npz")
    seg = Segmentation.load(d / f"{STEM}.npz")

    points = geometry.world_points()
    taus, depths = [], []
    for i in range(len(seg.instance_ids)):
        for m in np.unique(seg.instance_ids[i][seg.instance_ids[i] >= 0]):
            sel = (seg.instance_ids[i] == m) & (geometry.depth[i] > 0)
            if sel.sum() < 50:
                continue
            cloud = points[i][sel]
            nn = cKDTree(cloud).query(cloud, k=2, workers=-1)[0][:, 1]
            taus.append(3 * nn.mean())
            depths.append(np.median(geometry.depth[i][sel]))
    taus, depths = np.array(taus), np.array(depths)
    # Déjà View is up to scale: report tau relative to the mask's depth too
    print(
        f"  {name}: {len(taus)} masks, tau median {np.median(taus):.4f} "
        f"(10-90%: {np.percentile(taus, 10):.4f}-{np.percentile(taus, 90):.4f}) "
        f"scene units, tau / median mask depth {np.median(taus / depths):.4f} "
        f"(90%: {np.percentile(taus / depths, 90):.4f})"
    )

# figure: key metrics per setup, scenes side by side
METRICS = {
    "pixel_recall": "pixel recall",
    "consistent": "consistent (objects in >= 2 views, one id)",
    "merges": "merges (lower is better)",
    "wrong": "wrong links (lower is better)",
}
COLORS = {"kitchen": "#2a78d6", "office": "#eb6834"}
labels = list(SETUPS)
fig, axes = plt.subplots(1, len(METRICS), figsize=(17, 4.4))
y = np.arange(len(labels))
for ax, (m, title) in zip(axes, METRICS.items(), strict=True):
    for k, (scene, name) in enumerate(SCENES.items()):
        vals = [
            rows[scene, lab]["all"][1]
            if m == "wrong"
            else getattr(rows[scene, lab]["all"][0], m)
            for lab in labels
        ]
        ax.barh(y + (k - 0.5) * 0.38, vals, 0.36, color=COLORS[name], label=name)
        for yy, v in zip(y + (k - 0.5) * 0.38, vals, strict=True):
            ax.text(
                v,
                yy,
                f" {v:.2f}" if isinstance(v, float) else f" {v}",
                va="center",
                fontsize=7,
            )
    ax.set_yticks(y, labels if ax is axes[0] else [""] * len(labels), fontsize=8)
    ax.invert_yaxis()
    ax.set_title(title, fontsize=9)
    ax.grid(axis="x", color="#e5e4df", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=8)
axes[-1].legend(fontsize=8, frameon=False, loc="lower right")
fig.suptitle(
    "E24: 3D point-cloud overlap (Replicate Any Scene) vs reprojection overlap. "
    "top100 t0.3, cross-class 0.7, multi-view filter 0.5, all classes",
    fontsize=10,
)
fig.tight_layout()
fig.savefig(FIGURE, dpi=130, pil_kwargs={"quality": 90})
print(f"\nfigure: {FIGURE}")
