"""Error breakdowns of one cached segmentation against GT (E4, E10, E11, E12).

uv run python experiments/scripts/breakdowns.py ids <scene> <segmentation stem>
uv run python experiments/scripts/breakdowns.py recall <scene> <segmentation stem> [prompt files...]
uv run python experiments/scripts/breakdowns.py leakage <scene> <segmentation stem>
uv run python experiments/scripts/breakdowns.py precision <scene> <segmentation stem>

ids: GT objects split over several predicted ids, predicted ids merging GT objects.
recall: found/total and share of scored pixels per GT class, and whether it is prompted.
leakage: leaked pixels by predicted class and by (predicted class -> GT class).
precision: unmatched predicted masks by class and by the GT object they lie on.
"""

import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from src.evaluation.instances import MIN_AREA, NOT_OBJECTS, match_view, overlap_counts
from src.loaders.scannetpp.gt import GroundTruth
from src.segmentation import Segmentation
from src.segmentation.sam3 import PROMPTS, load_prompts

kind, scene, stem = sys.argv[1:4]
# prompt files the segmentation used (default: the 16 built-in prompts)
prompts = tuple(load_prompts([Path(p) for p in sys.argv[4:]])) or PROMPTS
d = Path("data/processed/scannetpp") / scene
gt = GroundTruth.load(d / "gt_views_n20_uniform_504.npz")
# same as the scorer: annotation helper labels count as unlabeled
not_objects = [g for g, c in gt.classes.items() if c.lower() in NOT_OBJECTS]
gt.instance_ids[np.isin(gt.instance_ids, not_objects)] = -1
pred = Segmentation.load(d / f"{stem}.npz")
num, height, width = gt.instance_ids.shape
min_px = MIN_AREA * height * width

if kind == "ids":
    preds_of, gts_of, views_of = defaultdict(set), defaultdict(set), defaultdict(list)
    for i in range(num):
        for g, p, _ in match_view(gt.instance_ids[i], pred.instance_ids[i]):
            if (gt.instance_ids[i] == g).sum() >= min_px:
                preds_of[g].add(p)
                gts_of[p].add(g)
                views_of[g].append(i)
    print("GT objects split over several predicted ids:")
    mixed = 0
    for g, ps in sorted(preds_of.items(), key=lambda x: -len(x[1])):
        if len(ps) > 1:
            names = sorted({pred.classes.get(p) for p in ps})
            mixed += len(names) > 1
            print(
                f"  GT {g:3d} {gt.classes[g]:18s} in {len(views_of[g]):2d} views "
                f"-> {len(ps)} ids {sorted(ps)} classes {names}"
            )
    splits = sum(len(ps) > 1 for ps in preds_of.values())
    print(f"  {mixed} of {splits} split objects have ids with different classes")
    print("predicted ids covering several GT objects (merges):")
    for p, gs in sorted(gts_of.items(), key=lambda x: -len(x[1])):
        if len(gs) > 1:
            names = ", ".join(f"{g}:{gt.classes[g]}" for g in sorted(gs))
            print(f"  id {p:5d} ({pred.classes.get(p)}) -> {names}")

elif kind == "recall":
    tot, hit, px_tot, px_hit = Counter(), Counter(), Counter(), Counter()
    for i in range(num):
        matched = {
            g for g, _, _ in match_view(gt.instance_ids[i], pred.instance_ids[i])
        }
        g_ids, area = np.unique(gt.instance_ids[i], return_counts=True)
        for g, a in zip(g_ids, area, strict=True):
            if g < 0 or a < min_px:
                continue
            c = gt.classes[int(g)]
            tot[c] += 1
            px_tot[c] += a
            if g in matched:
                hit[c] += 1
                px_hit[c] += a
    total_px = sum(px_tot.values())
    print(
        f"instance recall {sum(hit.values()) / sum(tot.values()):.2f}, "
        f"pixel-weighted recall {sum(px_hit.values()) / total_px:.2f}"
    )
    print(f"  {'GT class':22s} found/total  % of scored px  prompted")
    for c in sorted(tot, key=lambda c: -px_tot[c]):
        prompted = "yes" if any(p in c or c in p for p in prompts) else "NO"
        print(
            f"  {c:22s} {hit[c]:4d}/{tot[c]:<4d}   {100 * px_tot[c] / total_px:10.1f}%"
            f"     {prompted}"
        )

elif kind == "leakage":
    leak, size, into = Counter(), Counter(), Counter()
    for i in range(num):
        g_ids, p_ids, inter = overlap_counts(gt.instance_ids[i], pred.instance_ids[i])
        lab = inter[g_ids >= 0][:, p_ids >= 0]
        g_lab = g_ids[g_ids >= 0]
        for col, p in enumerate(p_ids[p_ids >= 0]):
            c = pred.classes[int(p)]
            dom = lab[:, col].argmax()
            size[c] += lab[:, col].sum()
            leak[c] += lab[:, col].sum() - lab[dom, col]
            for r in np.nonzero(lab[:, col])[0]:
                if r != dom:
                    into[c, gt.classes[int(g_lab[r])]] += lab[r, col]
    total = sum(leak.values())
    print("leaked pixels by predicted class (share of all leakage, rate within class):")
    for c, v in leak.most_common(8):
        print(f"  {c:12s} {100 * v / total:5.1f}%   rate {v / size[c]:.2f}")
    print("top (predicted class -> GT class it leaks into):")
    for (c, g), v in into.most_common(10):
        print(f"  {c:12s} -> {g:18s} {100 * v / total:5.1f}%")

elif kind == "precision":
    # unmatched predicted masks, with the scorer's ignore rules
    fp, fp_into, n_pred = Counter(), Counter(), 0
    for i in range(num):
        g_ids, p_ids, inter = overlap_counts(gt.instance_ids[i], pred.instance_ids[i])
        g_area = dict(zip(g_ids.tolist(), inter.sum(1).tolist(), strict=True))
        matches = match_view(gt.instance_ids[i], pred.instance_ids[i])
        hit = {p for g, p, _ in matches if g >= 0 and g_area[g] >= min_px}
        ignored = {p for g, p, _ in matches if p not in hit}
        if -1 in g_ids:
            on_unlabeled = inter[g_ids == -1][0] > inter.sum(0) / 2
            ignored |= set(p_ids[on_unlabeled].tolist())
        for col, p in enumerate(p_ids):
            if p < 0 or p in ignored:
                continue
            n_pred += 1
            if p in hit:
                continue
            c = pred.classes[int(p)]
            fp[c] += 1
            labeled = inter[:, col] * (g_ids >= 0)
            dom = g_ids[labeled.argmax()]
            share = labeled.max() / inter[:, col].sum()
            where = "inside" if share > 0.9 else "spans"  # part of one object or not
            fp_into[c, gt.classes.get(int(dom), "?"), where] += 1
    total = sum(fp.values())
    print(f"{total} of {n_pred} counted masks unmatched (precision {1 - total / n_pred:.2f})")
    print("unmatched masks by predicted class:")
    for c, v in fp.most_common(15):
        print(f"  {c:16s} {v:4d}")
    print("top (predicted class -> dominant GT class, >90% inside it or spanning):")
    for (c, g, w), v in fp_into.most_common(15):
        print(f"  {c:16s} -> {g:18s} {w:6s} {v:4d}")

else:
    sys.exit(__doc__)
