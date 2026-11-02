"""E30: first Hypersim run of the segmenter (six val scenes, 20 views at 504x378).

Three scenes with tilt-shift cameras (ts, off-centre principal point) and three with plain
pinhole cameras. Cluster jobs `experiments/scripts/e30_hypersim.sbatch` (ts scenes) and
`e30_plain.sbatch` (plain scenes + the restaurant VLM run that hit the ID_STRIDE bug): Déjà View K=16, Qwen3.8-27B union
vocabulary capped at 64 tokens (E28 setting), SAM 3 image mode t0.3 with top100 and with
that list. Copied back: Déjà View cache + VLM lists into `<scene>/`, SAM 3 caches into
`<scene>/e30_cluster/`. Here (CPU): Déjà View checks (intrinsics vs GT for the off-centre
principal points, per-view-scale depth error as E8, the paper's point-map metric as E23),
then association with the working setting (xc0.7 vf0.25 kl0.5) on Déjà View and on GT
geometry, scored with eval-instances.

uv run python experiments/scripts/e30_hypersim.py > experiments/results/e30_hypersim.txt
"""

import json
from collections import defaultdict
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from e23_pointmap_metric import metrics, umeyama

from src.evaluation.instances import STRUCTURE, match_view, overlap_counts, score
from src.geometry import Geometry
from src.loaders import Views
from src.loaders.scannetpp.gt import NO_MESH, GroundTruth
from src.segmentation import Segmentation
from src.segmentation.associate import gt_geometry
from src.segmentation.linked import working_associator

DATA = Path("data/processed/hypersim")
FIGURE = Path("experiments/figures/e30_hypersim.jpg")
SCENES = {  # first three: tilt-shift cameras (job 760), last three: plain pinhole (job 776)
    "ai_015_001_cam_00": "kitchen (ts)",
    "ai_047_009_cam_00": "conference (ts)",
    "ai_052_001_cam_00": "restaurant (ts)",
    "ai_010_007_cam_00": "living room",
    "ai_006_007_cam_00": "bedroom",
    "ai_044_001_cam_00": "lobby",
}
VIEWS = "views_n20_uniform_504"
DV = f"dejaview_k16_{VIEWS}"
TOP100 = Path("data/scannetpp/metadata/semantic_benchmark/top100.txt")
LISTS = {"top100": "top100", "27B union cap64": f"vlm-qwen3827b-mt64_{VIEWS}"}


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


def depth_checks() -> None:
    """Déjà View vs GT: intrinsics, per-view-scale depth error, point-map metric."""
    print("## Déjà View vs GT\n")
    print(
        "| scene | fx DV/GT | fy DV/GT | cx DV - GT (px) | cy DV - GT (px) | "
        "depth abs rel (median, per-view scale) | worst view | Rel. L2 scene Sim(3) | "
        "inliers < 3% | Rel. L2 per-view Sim(3) |"
    )
    print("|---|---|---|---|---|---|---|---|---|---|")
    for scene, name in SCENES.items():
        d = DATA / scene
        views = Views.load(d / f"{VIEWS}.npz")
        gt = GroundTruth.load(d / f"gt_{VIEWS}.npz")
        dv = Geometry.load(d / f"{DV}.npz")
        k_dv, k_gt = dv.intrinsics.astype(np.float64), views.intrinsics.astype(np.float64)
        fx = np.median(k_dv[:, 0, 0] / k_gt[:, 0, 0])
        fy = np.median(k_dv[:, 1, 1] / k_gt[:, 1, 1])
        cx = np.median(k_dv[:, 0, 2] - k_gt[:, 0, 2])
        cy = np.median(k_dv[:, 1, 2] - k_gt[:, 1, 2])
        valid = (gt.instance_ids != NO_MESH) & (gt.depth > 0) & np.isfinite(dv.depth)
        per_view_rel = []
        for i in range(len(views)):
            v = valid[i]
            s = np.median(gt.depth[i][v] / dv.depth[i][v])
            per_view_rel.append(np.abs(s * dv.depth[i][v] - gt.depth[i][v]) / gt.depth[i][v])
        rel_view = [np.median(r) for r in per_view_rel]
        rel_all = np.concatenate(per_view_rel)
        gt_pts = Geometry(gt.depth, np.ones_like(gt.depth), views.intrinsics, views.cam_to_world).world_points()
        dv_pts = dv.world_points()
        centres = views.cam_to_world[:, :3, 3]
        ray = np.linalg.norm(gt_pts - centres[:, None, None], axis=-1)
        src, dst = dv_pts[valid].astype(np.float64), gt_pts[valid].astype(np.float64)
        s, rot, t = umeyama(src, dst)
        err = np.linalg.norm(src @ (s * rot).T + t - dst, axis=-1)
        rel_l2, _, inl = metrics(err, ray[valid])
        errs, norms = [], []
        for i in range(len(views)):
            v = valid[i]
            si, ri, ti = umeyama(dv_pts[i][v].astype(np.float64), gt_pts[i][v].astype(np.float64))
            errs.append(np.linalg.norm(dv_pts[i][v] @ (si * ri).T + ti - gt_pts[i][v], axis=-1))
            norms.append(ray[i][v])
        rel_l2_v, _, _ = metrics(np.concatenate(errs), np.concatenate(norms))
        worst = int(np.argmax(rel_view))
        print(
            f"| {name} | {fx:.3f} | {fy:.3f} | {cx:+.1f} | {cy:+.1f} | {np.median(rel_all):.4f} | "
            f"{worst}: {rel_view[worst]:.4f} | {rel_l2:.4f} | {inl:.1%} | {rel_l2_v:.4f} |"
        )
        print(f"  GT principal points {name}: cx {np.median(k_gt[:, 0, 2]):.1f}, cy {np.median(k_gt[:, 1, 2]):.1f}"
              f" (image {views.images.shape[2]}x{views.images.shape[1]}); DV cx {np.median(k_dv[:, 0, 2]):.1f},"
              f" cy {np.median(k_dv[:, 1, 2]):.1f}")
    print()


def main() -> None:
    depth_checks()
    rows = {}
    notes = []
    print("## Segmentation (SAM 3 t0.3, association xc0.7 vf0.25 kl0.5)\n")
    print(
        "| scene | vocabulary | geometry | names | subset | recall | pixel recall | precision | "
        "cross-class leakage | consistent | GT in >=2 views | id switches | merges | wrong links |"
    )
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for scene, name in SCENES.items():
        d = DATA / scene
        views = Views.load(d / f"{VIEWS}.npz")
        gt = GroundTruth.load(d / f"gt_{VIEWS}.npz")
        geometries = {"Déjà View": Geometry.load(d / f"{DV}.npz"), "GT": gt_geometry(views, gt)}
        for label, stem in LISTS.items():
            prompts = TOP100 if stem == "top100" else d / f"{stem}.txt"
            seg = Segmentation.load(d / "e30_cluster" / f"sam3image-{stem}-t0.3_{VIEWS}.npz")
            dom = dominant_gt(gt, seg)
            for geo_name, geometry in geometries.items():
                linked = working_associator().associate(seg, geometry)
                tag = "dejaview_k16" if geo_name == "Déjà View" else "gt"
                linked.save(d / "e30_cluster" / f"sam3image-{stem}-t0.3_{VIEWS}_assoc-xc0.7-vf0.25-kl0.5-{tag}.npz")
                wrong = [
                    o
                    for o, ms in members(seg, linked).items()
                    if len({dom[m] for m in ms if dom.get(m, -1) >= 0}) > 1
                ]
                n_names = len(prompt_names(prompts))
                for subset, exclude in (("all", ()), ("objects", STRUCTURE)):
                    s = score(gt, linked, exclude)
                    rows[name, label, geo_name, subset] = s
                    extra = len(wrong) if subset == "all" else ""
                    print(
                        f"| {name} | {label} | {geo_name} | {n_names} | {subset} | {s.recall:.2f} | "
                        f"{s.pixel_recall:.2f} | {s.precision:.2f} | {s.cross_leakage:.3f} | "
                        f"{s.consistent:.2f} | {s.num_gt} | {s.id_switches} | {s.merges} | {extra} |"
                    )
                notes.append((name, label, geo_name, merges(gt, linked), len(seg.classes)))

    print("\nMasks and merges:")
    for name, label, geo_name, ms, n_masks in notes:
        print(f"\n{name} / {label} / {geo_name}: {n_masks} per-view masks")
        for m in ms:
            print(f"  merge: {m}")

    print("\nVLM vocabularies:")
    for scene, name in SCENES.items():
        stem = LISTS["27B union cap64"]
        answers = json.loads((DATA / scene / f"{stem}.json").read_text(encoding="utf-8"))
        names = prompt_names(DATA / scene / f"{stem}.txt")
        truncated = sum(a["truncated"] for a in answers.values())
        print(f"\n{name}: {len(names)} names, truncated answers {truncated}/{len(answers)}")
        print("  " + ", ".join(names))

    metrics_ = {
        "pixel_recall": "pixel recall",
        "precision": "precision",
        "consistent": "consistent (one id)",
        "merges": "merges (lower is better)",
    }
    configs = [(lab, geo) for lab in LISTS for geo in ("Déjà View", "GT")]
    colors = ["#2a78d6", "#7fb1ea", "#eb6834", "#f4b08a"]
    fig, axes = plt.subplots(1, len(metrics_), figsize=(18, 4.3))
    x = np.arange(len(SCENES))
    for ax, (m, title) in zip(axes, metrics_.items(), strict=True):
        for k, ((lab, geo), c) in enumerate(zip(configs, colors, strict=True)):
            vals = [getattr(rows[n, lab, geo, "all"], m) for n in SCENES.values()]
            xs = x + (k - 1.5) * 0.2
            ax.bar(xs, vals, 0.19, color=c, label=f"{lab}, {geo} geometry")
            for xx, v in zip(xs, vals, strict=True):
                ax.text(xx, v, f"{v:.2f}" if isinstance(v, float) else str(v), ha="center", va="bottom", fontsize=6)
        ax.set_xticks(x, list(SCENES.values()), fontsize=7, rotation=30, ha="right")
        ax.set_title(title, fontsize=9)
        ax.grid(axis="y", color="#e5e4df", linewidth=0.6)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=8, frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 0.93))
    fig.suptitle(
        "E30: Hypersim, SAM 3 t0.3 + association xc0.7 vf0.25 kl0.5, all classes, 20 views at 504x378",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    fig.savefig(FIGURE, dpi=130, pil_kwargs={"quality": 90})
    print(f"\nfigure: {FIGURE}")


if __name__ == "__main__":
    main()
