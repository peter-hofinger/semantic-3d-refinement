"""E28: Qwen3.8-27B with answers capped at 64 tokens (greedy) vs E26, E27, E16/E25, top100.

SAM 3 caches (threshold 0.3), all on the cluster (RTX 5090): top100 and the two 2B lists
from job 658 (`<scene>/e25_cluster/`), the two 27B lists from job 664
(`<scene>/e26_cluster/`), the two rp1.1 lists from job 667
(`experiments/scripts/e27_vlm_rp.sbatch`, `<scene>/e27_cluster/`), the two
64-token lists from job 676 (`experiments/scripts/e28_vlm_cap.sbatch`, `<scene>/e28_cluster/`). Association uses the working setting
(cross-class 0.7, vf 0.25, kl 0.5) with the Déjà View K=16 cache; the E25-E27 rows must
reproduce their association caches exactly.

uv run python experiments/scripts/e28_vlm_cap.py > experiments/results/e28_vlm_cap.txt
"""

import json
from collections import defaultdict
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.evaluation.instances import STRUCTURE, match_view, overlap_counts, score
from src.geometry import Geometry
from src.loaders.scannetpp.gt import GroundTruth
from src.segmentation import Segmentation
from src.segmentation.associate import Associator

DATA = Path("data/processed/scannetpp")
FIGURE = Path("experiments/figures/e28_vlm_cap.jpg")
SCENES = {"5ee7c22ba0": "kitchen", "7b6477cb95": "office"}
VIEWS = "views_n20_uniform_504"
GEOMETRY = "dejaview_k16_views_n20_uniform_504"
LISTS = {  # label -> prompt-file stem (the SAM 3 cache is sam3image-<stem>-t0.3_<views>)
    "top100": "top100",
    "2B union (E16)": f"vlm-qwen3vl2binstruct_{VIEWS}",
    "2B progressive (E25)": f"vlmprog-qwen3vl2binstruct_{VIEWS}",
    "27B union": f"vlm-qwen3827b_{VIEWS}",
    "27B progressive": f"vlmprog-qwen3827b_{VIEWS}",
    "27B union rp1.1": f"vlm-qwen3827b-rp1.1_{VIEWS}",
    "27B progressive rp1.1": f"vlmprog-qwen3827b-rp1.1_{VIEWS}",
    "27B union cap64": f"vlm-qwen3827b-mt64_{VIEWS}",
    "27B progressive cap64": f"vlmprog-qwen3827b-mt64_{VIEWS}",
}
FOLDER = {
    label: "e28_cluster"
    if "cap" in label
    else "e27_cluster"
    if "rp" in label
    else "e26_cluster"
    if "27B" in label
    else "e25_cluster"
    for label in LISTS
}
VLM = [label for label in LISTS if label != "top100"]
TAG = f"xc0.7-vf0.25-kl0.5-{GEOMETRY}"


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


def members(seg: Segmentation, linked: Segmentation) -> dict[int, set[int]]:
    """Linked object -> its per-view masks."""
    out: dict[int, set[int]] = defaultdict(set)
    for i in range(len(seg.instance_ids)):
        a, b = seg.instance_ids[i].ravel(), linked.instance_ids[i].ravel()
        keep = (a >= 0) & (b >= 0)
        for m, o in set(zip(a[keep].tolist(), b[keep].tolist(), strict=True)):
            out[o].add(m)
    return out


def merges(gt: GroundTruth, linked: Segmentation) -> list[str]:
    """Predicted ids matched (IoU >= 0.5) to more than one GT instance."""
    gts: dict[int, set[int]] = defaultdict(set)
    for i in range(len(gt.instance_ids)):
        for g, p, _ in match_view(gt.instance_ids[i], linked.instance_ids[i]):
            gts[p].add(g)
    return [
        f"{linked.classes[p]} -> "
        + ", ".join(sorted(f"{gt.classes.get(g, '?')} {g}" for g in gs))
        for p, gs in sorted(gts.items())
        if len(gs) > 1
    ]


def prompt_names(path: Path) -> list[str]:
    """Names in a prompt file (comments and `prompt = class` handled as in SAM 3)."""
    lines = path.read_text(encoding="utf-8").splitlines()
    return [ln.split("=")[0].strip() for ln in lines if ln.strip() and ln[0] != "#"]


rows = {}
notes = []
print(
    "| scene | vocabulary | machine | names | subset | recall | pixel recall | "
    "precision | cross-class leakage | consistent | GT in >=2 views | id switches | "
    "merges | wrong links | mixed-class objects |"
)
print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for scene, name in SCENES.items():
    d = DATA / scene
    gt = GroundTruth.load(d / f"gt_{VIEWS}.npz")
    geometry = Geometry.load(d / f"{GEOMETRY}.npz")
    runs = [(label, "cluster", d / FOLDER[label]) for label in LISTS]
    for label, machine, folder in runs:
        stem = LISTS[label]
        prompts = (
            Path("data/scannetpp/metadata/semantic_benchmark/top100.txt")
            if stem == "top100"
            else d / f"{stem}.txt"
        )
        seg = Segmentation.load(folder / f"sam3image-{stem}-t0.3_{VIEWS}.npz")
        out = folder / f"sam3image-{stem}-t0.3_{VIEWS}_assoc-{TAG}.npz"
        linked = Associator(
            cross_class_overlap=0.7, min_visible_fraction=0.25, keep_linked_below=0.5
        ).associate(seg, geometry)
        if out.exists():  # E25-E27 association caches must be reproduced exactly
            old = Segmentation.load(out)
            assert np.array_equal(old.instance_ids, linked.instance_ids), out
            assert old.classes == linked.classes, out
        linked.save(out)
        dom = dominant_gt(gt, seg)
        objs = members(seg, linked)
        wrong = [
            o
            for o, ms in objs.items()
            if len({dom[m] for m in ms if dom.get(m, -1) >= 0}) > 1
        ]
        mixed = [o for o, ms in objs.items() if len({seg.classes[m] for m in ms}) > 1]
        n_names = len(prompt_names(prompts))
        rows[scene, label, machine] = {}
        for subset, exclude in (("all", ()), ("objects", STRUCTURE)):
            s = score(gt, linked, exclude)
            rows[scene, label, machine][subset] = s
            extra = f"{len(wrong)} | {len(mixed)}" if subset == "all" else " | "
            print(
                f"| {name} | {label} | {machine} | {n_names} | {subset} | "
                f"{s.recall:.2f} | {s.pixel_recall:.2f} | {s.precision:.2f} | "
                f"{s.cross_leakage:.3f} | {s.consistent:.2f} | {s.num_gt} | "
                f"{s.id_switches} | {s.merges} | {extra} |"
            )
        rows[scene, label, machine]["names"] = n_names
        if machine == "cluster":
            notes.append((name, label, merges(gt, linked), len(seg.classes)))

print("\nMasks and merges (cluster runs):")
for name, label, ms, n_masks in notes:
    print(f"\n{name} / {label}: {n_masks} per-view masks")
    for m in ms:
        print(f"  merge: {m}")

print("\nVocabularies:")
for scene, name in SCENES.items():
    d = DATA / scene
    for label in VLM:
        stem = LISTS[label]
        answers = json.loads((d / f"{stem}.json").read_text(encoding="utf-8"))
        names = prompt_names(d / f"{stem}.txt")
        truncated = sum(a["truncated"] for a in answers.values())
        seconds = [a["seconds"] for a in answers.values() if "seconds" in a]
        timing = f", VLM {sum(seconds):.0f}s" if seconds else ""
        print(
            f"\n{name} / {label}: {len(names)} names, truncated answers "
            f"{truncated}/{len(answers)}{timing}"
        )
        if "new" in next(iter(answers.values())):
            per_view = [len(a["new"]) for a in answers.values()]
            print(f"  new names per view: {per_view}")
        print("  " + ", ".join(names))

# figure: key metrics per vocabulary (cluster runs), scenes side by side
METRICS = {
    "names": "names in the list",
    "pixel_recall": "pixel recall",
    "precision": "precision",
    "consistent": "consistent (one id)",
    "merges": "merges (lower is better)",
}
COLORS = {"kitchen": "#2a78d6", "office": "#eb6834"}
labels = list(LISTS)
fig, axes = plt.subplots(1, len(METRICS), figsize=(17, 5))
y = np.arange(len(labels))
for ax, (m, title) in zip(axes, METRICS.items(), strict=True):
    for k, (scene, name) in enumerate(SCENES.items()):
        r = [rows[scene, lab, "cluster"] for lab in labels]
        vals = [x["names"] if m == "names" else getattr(x["all"], m) for x in r]
        ys = y + (k - 0.5) * 0.38
        ax.barh(ys, vals, 0.36, color=COLORS[name], label=name)
        for yy, v in zip(ys, vals, strict=True):
            txt = f" {v:.2f}" if isinstance(v, float) else f" {v}"
            ax.text(v, yy, txt, va="center", fontsize=7)
    ax.set_yticks(y, labels if ax is axes[0] else [""] * len(labels), fontsize=8)
    ax.invert_yaxis()
    ax.set_title(title, fontsize=9)
    ax.grid(axis="x", color="#e5e4df", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=8)
handles, names = axes[0].get_legend_handles_labels()
fig.legend(handles, names, fontsize=8, frameon=False, ncol=2, loc="upper right")
fig.suptitle(
    "E28: VLM vocabulary, Qwen3.8-27B greedy, answers capped at 64 tokens vs E26 (256) and E27 (rp1.1), "
    "Qwen3-VL-2B, top100. SAM 3 t0.3 on the cluster, association xc0.7 vf0.25 kl0.5, all classes",
    fontsize=10,
)
fig.tight_layout(rect=(0, 0, 1, 0.97))
fig.savefig(FIGURE, dpi=130, pil_kwargs={"quality": 90})
print(f"\nfigure: {FIGURE}")
