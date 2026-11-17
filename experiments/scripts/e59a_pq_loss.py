"""E59a: where SAM 3 + linking lose panoptic quality on Hypersim (issue #59).

The labels are scored per ground-truth pixel (`src/evaluation/labeled.py`), so this
needs no geometry beyond the linking. Per scene:

1. Oracles, each scored with `label_metrics` as the evaluation does:
   - `per-view`: SAM 3's masks unlinked, every mask its own segment (what linking adds);
   - `linked`: the evaluation's segmentation (working association, E17);
   - `class oracle`: linked segments, each given the GT class most of its pixels have;
   - `link oracle`: every per-view mask gets the GT segment most of its pixels lie on,
     masks of one GT segment in different views share an id; within a view the
     largest keeps it and the others stay apart (perfect linking, splits kept);
   - `group oracle`: as `link oracle`, but masks of one GT segment in the same view are
     merged too (perfect grouping of SAM 3's masks);
   - `boundary k`: linked, pixels within k px of a GT or a predicted segment border
     made void in both (what sharper borders could give at most);
   - `all`: group oracle + class oracle + boundary 2.
2. A cause for every unmatched GT segment (false negative) of `linked`, first that
   applies: `class` (a prediction of another class matches it, IoU > 0.5), `merged`
   (one prediction covers more than half of it but is too large), `split across
   views` (more than half covered by several predictions, but in each view mostly by
   one: the linking), `split within views`, `missed`. And for every unmatched
   prediction (false positive): `class`, `fragment` (more than half of it lies in one GT
   segment), `merger` (it covers more than half of a GT segment), `spurious`.

uv run python experiments/scripts/e59a_pq_loss.py --splits dev val

`--segmentation <name>` scores another cached segmentation (E59b); its results get the
name as suffix. With the catch-all prompts (`catchall` in the name) the catch-alls
count as prompted. `--merge <contact> ...` adds the linked labels after
`merge_touching` (E59c) at each contact share, cubes of `MERGE_SIDE` pixel footprints;
`--absorb <contact>:<ratio> ...` adds each merge with small other-class parts absorbed
too (E59d), once into any object and once into things only (not wall, floor, ceiling).
"""

import argparse
from collections import Counter
from pathlib import Path

import matplotlib as mpl
import numpy as np
import yaml
from scipy.ndimage import binary_dilation

mpl.use("Agg")
import matplotlib.pyplot as plt

from src.analysis.filters import pixel_footprint
from src.analysis.scene_eval import VOCABULARIES, git_hash
from src.evaluation.labeled import (
    BACKGROUND,
    NONE,
    PQ_IOU,
    LabeledCloud,
    ground_truth_cloud,
    label_metrics,
    overlaps,
    segments,
)
from src.geometry import Geometry
from src.loaders.ground_truth import GroundTruth
from src.loaders.hypersim.config import load_config
from src.loaders.views import Views
from src.segmentation.associate import absorb_parts, merge_touching
from src.segmentation.base import Segmentation, count_pairs
from src.segmentation.linked import working_associator

ROOT = Path("data/processed/hypersim")
VIEWS = "views_n20_uniform_504"
SEGMENTATION = ["sam3image-nyu40-t0.3"]  # set by main
EVALUATED = list(VOCABULARIES["hypersim"])
FIGURES = Path("experiments/figures")
RESULTS = Path("experiments/results")
BEDROOM, BEDROOM_VIEWS = "ai_006_007_cam_00", (18, 0, 12, 6)
CATCH_ALL = ("otherstructure", "otherfurniture", "otherprop")
MERGE_SIDE = 4  # cube side of merge_touching, in pixel footprints
MERGES: list[float] = []  # contact shares, set by main
ABSORBS: list[tuple[float, float]] = []  # (contact, size ratio), set by main
UNPROMPTED = set(CATCH_ALL)  # emptied by main for a segmentation that prompts them
MIN_VIEW_PIXELS, MIN_VIEW_SHARE = (
    20,
    0.01,
)  # a segment is in a view with this much of it
SCORES = ("pq", "pq_th", "pq_st", "sq", "rq", "ap", "miou")
FN_CAUSES = (
    "not prompted",
    "class",
    "merged",
    "split across views",
    "split within views",
    "missed",
)
FP_CAUSES = ("class", "fragment", "merger", "spurious")
COLOURS = {  # cause map of the figure
    "matched": "#d9e8d0",
    "not prompted": "#b8b4a8",
    "class": "#e1a43a",
    "merged": "#8e6bbf",
    "split across views": "#d1495b",
    "split within views": "#3a7bd5",
    "missed": "#52514e",
}


def majority(src: np.ndarray, dst: np.ndarray) -> dict[int, int]:
    """For each src id, the dst id (not NONE) it overlaps most."""
    keep = (src != NONE) & (dst != NONE)
    a, b, n = count_pairs(src[keep], dst[keep])
    out: dict[int, int] = {}
    best: dict[int, int] = {}
    for i, j, k in zip(a.tolist(), b.tolist(), n.tolist(), strict=True):
        if k > best.get(i, 0):
            best[i], out[i] = k, j
    return out


def relabel(ids: np.ndarray, lut: dict[int, int]) -> np.ndarray:
    """Map ids through lut; ids not in it, and NONE, stay."""
    out = ids.astype(np.int64).copy()
    for i, j in lut.items():
        out[ids == i] = j
    return out


def link_oracle(
    labels: Segmentation, pred_px: np.ndarray, view_of: np.ndarray, gt_seg: np.ndarray,
    *, within: bool,
) -> Segmentation:  # fmt: skip
    """Merge linked segments of one majority GT segment (see module).

    Without `within`, only segments that never share a view are merged.
    """
    target = majority(pred_px, gt_seg)
    size = Counter(pred_px[pred_px != NONE].tolist())
    a, b, n = count_pairs(pred_px[pred_px != NONE], view_of[pred_px != NONE])
    seen: dict[int, set[int]] = {}
    for p, v, k in zip(a.tolist(), b.tolist(), n.tolist(), strict=True):
        if k >= max(MIN_VIEW_PIXELS, MIN_VIEW_SHARE * size[p]):
            seen.setdefault(p, set()).add(v)
    groups: dict[int, list[tuple[int, set[int]]]] = {}
    lut: dict[int, int] = {}
    for p in sorted(size, key=lambda p: -size[p]):
        g, views = target.get(p), seen.get(p, set())
        if g is None:
            continue
        for i, (rep, used) in enumerate(groups.setdefault(g, [])):
            if within or not used & views:
                lut[p] = rep
                groups[g][i] = (rep, used | views)
                break
        else:
            groups[g].append((p, set(views)))
    votes: dict[int, Counter] = {}
    for p in size:
        votes.setdefault(lut.get(p, p), Counter())[labels.classes[p]] += size[p]
    classes = {i: c.most_common(1)[0][0] for i, c in votes.items()}
    return Segmentation(relabel(labels.instance_ids, lut).astype(np.int32), classes)


def class_oracle(labels: Segmentation, pred_px: np.ndarray, gt_seg: np.ndarray,
                 gt_class: dict[int, str], *, prompted: bool) -> Segmentation:  # fmt: skip
    """Give each predicted segment the GT class most of its pixels have.

    With `prompted`, only if SAM 3 was asked for that class (not a catch-all).
    """
    target = majority(pred_px, gt_seg)
    classes = {
        i: gt_class[target[i]]
        if i in target and not (prompted and gt_class[target[i]] in UNPROMPTED)
        else c
        for i, c in labels.classes.items()
    }
    return Segmentation(labels.instance_ids, classes, labels.scores)


def borders(ids: np.ndarray) -> np.ndarray:
    """Return True where a 4-neighbour holds another id (V, H, W)."""
    edge = np.zeros(ids.shape, dtype=bool)
    dx, dy = ids[:, :, 1:] != ids[:, :, :-1], ids[:, 1:] != ids[:, :-1]
    edge[:, :, 1:] |= dx
    edge[:, :, :-1] |= dx
    edge[:, 1:] |= dy
    edge[:, :-1] |= dy
    return edge


def band(gt_full: np.ndarray, pred_full: np.ndarray, k: int) -> np.ndarray:
    """Pixels within k px (chessboard) of a GT or a predicted segment border."""
    edge = borders(gt_full) | borders(pred_full)
    if k > 1:
        edge = binary_dilation(edge, np.ones((1, 3, 3), bool), iterations=k - 1)
    return edge


def scores(
    truth: LabeledCloud, labels: Segmentation, keep: np.ndarray | None = None
) -> dict:
    """label_metrics of a segmentation at the GT pixels, optionally on a subset."""
    ids = labels.instance_ids[truth.pixels].astype(np.int64)
    ids = np.where(ids >= 0, ids, NONE)
    if keep is not None:
        pixels = truth.pixels.copy()
        pixels[truth.pixels] = keep
        truth = LabeledCloud(
            truth.points[keep], truth.instance[keep], truth.classes, pixels
        )
        ids = ids[keep]
    m = label_metrics(truth, ids, labels.classes, labels.scores, EVALUATED)
    gt_segments = len(segments(truth.instance, truth.classes, EVALUATED)[1])
    return {k: m[k] for k in SCORES} | {"gt_segments": gt_segments}


def causes(
    truth: LabeledCloud, labels: Segmentation, view_of: np.ndarray
) -> tuple[dict, dict, dict]:
    """Cause per unmatched GT segment and per unmatched prediction (see module)."""
    ids = labels.instance_ids[truth.pixels].astype(np.int64)
    ids = np.where(ids >= 0, ids, NONE)
    gt_seg, gt_class = segments(truth.instance, truth.classes, EVALUATED)
    pred_seg, pred_class = segments(ids, labels.classes, EVALUATED)
    ov = overlaps(gt_seg, gt_class, pred_seg, pred_class)
    same = {
        (p, g)
        for (p, g), n in ov.inter.items()
        if n / (ov.pred_size[p] - ov.pred_void.get(p, 0) + ov.gt_size[g] - n) > PQ_IOU
    }
    hit_p, hit_g = {p for p, _ in same}, {g for _, g in same}
    a, b, n = count_pairs(pred_seg, gt_seg)
    inter: dict[tuple[int, int], int] = {
        (p, g): k
        for p, g, k in zip(a.tolist(), b.tolist(), n.tolist(), strict=True)
        if NONE not in (p, g)
    }
    by_g: dict[int, list[tuple[int, int]]] = {}
    by_p: dict[int, list[tuple[int, int]]] = {}
    for (p, g), k in inter.items():
        by_g.setdefault(g, []).append((p, k))
        by_p.setdefault(p, []).append((g, k))

    def iou(p: int, g: int, k: int) -> float:
        return k / (ov.pred_size[p] - ov.pred_void.get(p, 0) + ov.gt_size[g] - k)

    keep = (gt_seg != NONE) & (pred_seg != NONE)
    va, _, vn = count_pairs(gt_seg[keep] * 64 + view_of[keep], pred_seg[keep])
    per_view: dict[tuple[int, int], int] = Counter()  # (g, v) -> largest single pred
    for gv, k in zip(va.tolist(), vn.tolist(), strict=True):
        key = (gv // 64, gv % 64)
        per_view[key] = max(per_view[key], k)
    fn: dict[int, str] = {}
    for g in gt_class:
        if g in hit_g:
            continue
        cov = sorted(
            ((k / ov.gt_size[g], p, k) for p, k in by_g.get(g, [])), reverse=True
        )
        if gt_class[g] in UNPROMPTED:
            fn[g] = "not prompted"
        elif any(iou(p, g, k) > PQ_IOU for _, p, k in cov):
            fn[g] = "class"
        elif cov and cov[0][0] > PQ_IOU:
            fn[g] = "merged"
        elif sum(c for c, _, _ in cov if c >= 0.05) > PQ_IOU:
            single = sum(k for (gg, _), k in per_view.items() if gg == g)
            fn[g] = (
                "split across views"
                if single / ov.gt_size[g] > PQ_IOU
                else "split within views"
            )
        else:
            fn[g] = "missed"
    fp: dict[int, str] = {}
    for p in pred_class:
        if p in hit_p or ov.pred_void.get(p, 0) / ov.pred_size[p] > PQ_IOU:
            continue
        on = sorted(by_p.get(p, []), key=lambda t: -t[1])
        size = ov.pred_size[p] - ov.pred_void.get(p, 0)
        if any(iou(p, g, k) > PQ_IOU for g, k in on):
            fp[p] = "class"
        elif on and on[0][1] / size > PQ_IOU:
            fp[p] = "fragment"
        elif any(k / ov.gt_size[g] > PQ_IOU for g, k in on):
            fp[p] = "merger"
        else:
            fp[p] = "spurious"
    status = {g: fn.get(g, "matched") for g in gt_class}
    return fn, fp, status


def scene_inputs(name: str) -> tuple:
    """Load views, GT cloud, per-view masks and the linked segmentation."""
    folder = ROOT / name
    views = Views.load(folder / f"{VIEWS}.npz")
    gt = GroundTruth.load(folder / f"gt_{VIEWS}.npz")
    geometry = Geometry.load(folder / f"dejaview_k16_{VIEWS}.npz")
    masks = Segmentation.load(folder / f"{SEGMENTATION[0]}_{VIEWS}.npz")
    linked = working_associator().associate(masks, geometry)
    truth = ground_truth_cloud(
        gt.depth, gt.instance_ids, gt.classes, views.intrinsics, views.cam_to_world
    )
    return views, truth, masks, linked, geometry


def run_scene(name: str) -> tuple[dict, dict]:
    """All oracle scores and the cause counts of one scene; maps for the figure."""
    views, truth, masks, linked, geometry = scene_inputs(name)
    view_of = np.nonzero(truth.pixels)[0]
    gt_seg, gt_class = segments(truth.instance, truth.classes, EVALUATED)
    mask_px = masks.instance_ids[truth.pixels].astype(np.int64)
    mask_px = np.where(mask_px >= 0, mask_px, NONE)
    pred_px = linked.instance_ids[truth.pixels].astype(np.int64)
    pred_px = np.where(pred_px >= 0, pred_px, NONE)
    gt_full = np.full(truth.pixels.shape, NONE, dtype=np.int64)
    gt_full[truth.pixels] = gt_seg
    grouped = link_oracle(linked, pred_px, view_of, gt_seg, within=True)
    grouped_px = grouped.instance_ids[truth.pixels].astype(np.int64)
    best = class_oracle(grouped, np.where(grouped_px >= 0, grouped_px, NONE), gt_seg,
                        gt_class, prompted=False)  # fmt: skip
    variants = {
        "per-view": scores(truth, masks),
        "linked": scores(truth, linked),
        "class oracle, prompted": scores(
            truth, class_oracle(linked, pred_px, gt_seg, gt_class, prompted=True)
        ),
        "class oracle": scores(
            truth, class_oracle(linked, pred_px, gt_seg, gt_class, prompted=False)
        ),
        "link oracle": scores(
            truth, link_oracle(linked, pred_px, view_of, gt_seg, within=False)
        ),
        "group oracle": scores(truth, grouped),
    }
    side = MERGE_SIDE * pixel_footprint(geometry.depth, geometry.intrinsics)
    for contact in MERGES:
        merged = merge_touching(linked, geometry, side, contact)
        variants[f"merged {contact:g}"] = scores(truth, merged)
        things = {c for c in EVALUATED if c not in BACKGROUND}
        for need, ratio in ABSORBS:
            for into, hosts in (("", None), (" into things", things)):
                absorbed = absorb_parts(merged, geometry, side, (need, ratio), hosts)
                label = f"merged {contact:g}, absorbed {need:g}:{ratio:g}{into}"
                variants[label] = scores(truth, absorbed)
    for k in (2, 5):
        keep = ~band(gt_full, linked.instance_ids, k)[truth.pixels]
        variants[f"boundary {k}"] = scores(truth, linked, keep)
    keep = ~band(gt_full, best.instance_ids, 2)[truth.pixels]
    variants["all"] = scores(truth, best, keep)
    fn, fp, status = causes(truth, linked, view_of)
    out = {
        "scores": variants,
        "gt_segments": len(gt_class),
        "fn": dict(Counter(fn.values())),
        "fp": dict(Counter(fp.values())),
        "fn_by_class": {
            c: dict(Counter(fn[g] for g in fn if gt_class[g] == c))
            for c in {gt_class[g] for g in fn}
        },
    }
    maps = {}
    if name == BEDROOM:
        cause = np.full(truth.pixels.shape, -1, dtype=np.int64)
        index = {c: i for i, c in enumerate(COLOURS)}
        cause[truth.pixels] = [
            index[status[g]] if g in status else -1 for g in gt_seg.tolist()
        ]
        maps = {
            "images": views.images,
            "gt": gt_full,
            "linked": linked.instance_ids,
            "cause": cause,
        }
    return out, maps


def id_colours(ids: np.ndarray, seed: int) -> np.ndarray:
    """Return a random colour per id (V, H, W, 3), black for NONE."""
    rng = np.random.default_rng(seed)
    unique, inverse = np.unique(ids, return_inverse=True)
    lut = rng.uniform(0.15, 1.0, (len(unique), 3))
    lut[unique == NONE] = 0
    return lut[inverse.reshape(ids.shape)]


def figure(maps: dict, path: Path) -> None:
    """Bedroom views 18, 0, 12, 6: image, GT segments, SAM 3 linked, cause per GT segment."""
    rows = (
        "image",
        "GT segments",
        "SAM 3 + linking",
        "cause if unmatched",
    )
    fig, axes = plt.subplots(4, 4, figsize=(13, 8.6))
    gt_c, pred_c = id_colours(maps["gt"], 0), id_colours(maps["linked"], 1)
    palette = np.array([mpl.colors.to_rgb(c) for c in COLOURS.values()])
    for j, v in enumerate(BEDROOM_VIEWS):
        cause = np.where(
            maps["cause"][v][..., None] >= 0, palette[maps["cause"][v].clip(0)], 1.0
        )
        for i, img in enumerate((maps["images"][v], gt_c[v], pred_c[v], cause)):
            axes[i, j].imshow(img)
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])
            if j == 0:
                axes[i, j].set_ylabel(rows[i], fontsize=9)
        axes[0, j].set_title(f"view {v}", fontsize=10)
    handles = [mpl.patches.Patch(color=c, label=k) for k, c in COLOURS.items()]
    fig.legend(handles=handles, loc="lower center", ncol=7, fontsize=9, frameon=False)
    fig.suptitle(
        "Bedroom ai_006_007: where SAM 3 + linking lose PQ. Bottom row: each GT segment coloured by why\n"
        "no prediction matches it (IoU > 0.5, same class); white: no GT. Rows 2 and 3: one colour per id, not shared",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    fig.savefig(path, dpi=110)
    plt.close(fig)


def table(results: dict) -> str:
    """Split means of every variant, and the cause counts."""
    lines = []
    for split, scenes in results.items():
        lines.append(f"## {split} ({len(scenes)} scenes), mean over scenes")
        lines.append("| variant | " + " | ".join(SCORES) + " | GT segments |")
        lines.append("|---|" + "---|" * (len(SCORES) + 1))
        for v in next(iter(scenes.values()))["scores"]:
            vals = [
                np.nanmean([s["scores"][v][k] for s in scenes.values()]) for k in SCORES
            ]
            n = sum(s["scores"][v]["gt_segments"] for s in scenes.values())
            lines.append(
                f"| {v} | " + " | ".join(f"{x:.3f}" for x in vals) + f" | {n} |"
            )
        gts = sum(s["gt_segments"] for s in scenes.values())
        fn = Counter()
        fp = Counter()
        for s in scenes.values():
            fn.update(s["fn"])
            fp.update(s["fp"])
        lines.append(
            f"\nGT segments {gts}, unmatched {sum(fn.values())}: "
            + ", ".join(f"{c} {fn[c]}" for c in FN_CAUSES)
        )
        lines.append(
            f"unmatched predictions {sum(fp.values())}: "
            + ", ".join(f"{c} {fp[c]}" for c in FP_CAUSES)
            + "\n"
        )
    return "\n".join(lines)


def main() -> None:
    """Run the splits, write the results, the table and the bedroom figure."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", nargs="+", default=["dev"])
    parser.add_argument("--segmentation", default=SEGMENTATION[0])
    parser.add_argument("--merge", type=float, nargs="*", default=[])
    parser.add_argument("--absorb", nargs="*", default=[])
    args = parser.parse_args()
    default = args.segmentation == SEGMENTATION[0]
    SEGMENTATION[0] = args.segmentation
    MERGES.extend(args.merge)
    ABSORBS.extend(tuple(map(float, a.split(":"))) for a in args.absorb)
    if "catchall" in args.segmentation:
        UNPROMPTED.clear()
    config = load_config(Path("configs/hypersim.yaml"))
    results: dict[str, dict] = {}
    for split in args.splits:
        results[split] = {}
        for s, c in config.scenes([split]).items():
            name = f"{s}_{c}"
            out, maps = run_scene(name)
            results[split][name] = out
            pq = out["scores"]
            print(
                f"{split} {name}: PQ linked {pq['linked']['pq']:.3f}, all {pq['all']['pq']:.3f}, fn {out['fn']}",
                flush=True,
            )
            if maps and default:
                FIGURES.mkdir(parents=True, exist_ok=True)
                figure(maps, FIGURES / "e59a_bedroom.jpg")
    RESULTS.mkdir(parents=True, exist_ok=True)
    tag = "_".join(args.splits) + ("" if default else f"_{args.segmentation}")
    tag += "_merge" if args.merge else ""
    (RESULTS / f"e59a_{tag}.yaml").write_text(
        yaml.safe_dump({"git": git_hash(), "results": results}, sort_keys=False)
    )
    text = table(results)
    (RESULTS / f"e59a_{tag}.txt").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
