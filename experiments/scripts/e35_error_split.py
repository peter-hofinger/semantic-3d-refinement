"""E35: split Déjà View's error, and the joint refiner's completeness loss, by source.

Reads a run of `src.analysis.scene_eval` (Hypersim) or of `e35_eth3d_eval.py` (ETH3D):
both store per scene the depth of each variant, its alignment s R | t onto the ground
truth (one Umeyama Sim(3) on the pixel pairs, the protocol's) and the linked SAM 3
instances. Every number below is on the GT pixels the protocol scores; per-pixel error
is the 3D distance of a pixel's aligned point to its own GT point (cm).

Error split, per variant (LOG.md E35 plan; thresholds fixed before running):
1. Camera / global: the mean error with one Sim(3) per view instead of one per scene.
2. Cross-view: for pixel pairs that the GT says see one surface (view i's GT point,
   projected into view j's GT camera, lands within VISIBLE of j's GT depth), the
   predicted depth gap minus the GT's own gap (pixel discretisation), relative to j's
   GT depth; median and share above VISIBLE (double surfaces).
3. Edges: error by distance to the nearest GT depth edge (a relative jump above
   DEPTH_JUMP to a 4-neighbour, E18 / E31), bins of EDGE_BINS px: pixel share and
   contribution to the mean error (contributions sum to the mean).
4. Objects: on GT instances (where the dataset has labels) and on SAM 3's linked
   instances (every scene, so the datasets compare), of at least MIN_OBJECT px, each
   fitted with its own isotropic scale and 3D shift in closed form; the error that
   stays is shape, the rest placement. The fit is least squares, so the mean distance
   can rise slightly where placement is already right.
5. Glass and mirrors (NYU40 window, mirror): reported apart and left out of 1-4.

Completeness, Déjà View -> refined: each GT point's distance to the nearest predicted
point, as the scorer computes it; its change summed by class, edge bin, number of views
whose GT sees the point (VIEW_BINS), Déjà View confidence quartile and |depth change|
quartile (per scene), as contributions to the scene's completeness change. Also
non-finite points, and near edges (<= COLLAPSE px) the share of pixels whose depth lies
closer to the GT depth across the edge than to their own (two surfaces pulled into one).
Scenes are averaged with equal weight, as the protocol's split means.

uv run python experiments/scripts/e35_error_split.py runs/<run> [--maps <scene> ...] [--variants A B] > experiments/results/e35_error_split_<data>.txt
also writes `runs/<run>/e35_split.yaml` (every record) and, for the `--maps` scenes,
`runs/<run>/e35_maps.npz` (`<scene>/comp/<i>`: each GT pixel's completeness distance, m,
NaN without GT), the inputs of `e35_figures.py`. The split compares the run's first
and last variant; `--variants A B` (E37) compares variants A and B instead and writes
`e35_split_A_B.yaml` and `e35_maps_A_B.npz`.
"""

import argparse
import multiprocessing
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
import yaml
from scipy import ndimage

from src.analysis.scene_eval import input_paths
from src.evaluation.labeled import lift, nearest_distances, umeyama
from src.geometry import Geometry
from src.loaders.ground_truth import GroundTruth
from src.loaders.views import Views
from src.refinement.multiview import points_from_rays, scene_from_geometry

DEPTH_JUMP = (
    0.05  # relative depth step to a 4-neighbour that makes a GT depth edge (E18)
)
VISIBLE = (
    0.05  # relative depth gap below which view j's GT sees view i's GT point (E34)
)
EDGE_BINS = (1, 3, 10)  # px from a GT depth edge: <= 1, 1-3, 3-10, > 10
VIEW_BINS = (1, 3, 7)  # views whose GT sees a point, itself included: 1, 2-3, 4-7, 8+
MIN_OBJECT = 200  # px over all views
COLLAPSE = 3  # px from a GT depth edge where surfaces can be pulled into one
GLASS = ("window", "mirror")
WORKERS = 6


def depth_edges(depth: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Mark valid pixels (H, W) whose depth differs by more than DEPTH_JUMP from a valid 4-neighbour."""
    with np.errstate(
        divide="ignore", invalid="ignore"
    ):  # pixels without GT, masked below
        h = np.abs(depth[:, 1:] - depth[:, :-1]) / np.minimum(
            depth[:, 1:], depth[:, :-1]
        )
        v = np.abs(depth[1:] - depth[:-1]) / np.minimum(depth[1:], depth[:-1])
    h = np.where(valid[:, 1:] & valid[:, :-1], h, 0) > DEPTH_JUMP
    v = np.where(valid[1:] & valid[:-1], v, 0) > DEPTH_JUMP
    edge = np.zeros(depth.shape, dtype=bool)
    edge[:, 1:] |= h
    edge[:, :-1] |= h
    edge[1:] |= v
    edge[:-1] |= v
    return edge


def edge_distance(depth: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Distance (V, H, W) in px of each pixel to the nearest GT depth edge of its view."""
    out = np.full(depth.shape, np.inf)
    for i in range(len(depth)):
        edge = depth_edges(depth[i], valid[i])
        if edge.any():
            out[i] = ndimage.distance_transform_edt(~edge)
    return out


def across_edge(depth: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """GT depth (V, H, W) of the nearest pixel within COLLAPSE px on another surface, NaN if none."""
    out = np.full(depth.shape, np.nan)
    best = np.full(depth.shape, np.inf)
    _, h, w = depth.shape
    r = COLLAPSE
    padded = np.pad(
        np.where(valid, depth, np.nan), ((0, 0), (r, r), (r, r)), constant_values=np.nan
    )
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            dist = np.hypot(dy, dx)
            if dist == 0 or dist > r:
                continue
            other = padded[:, r + dy : r + dy + h, r + dx : r + dx + w]
            with np.errstate(divide="ignore", invalid="ignore"):
                apart = np.abs(other - depth) / np.minimum(other, depth) > DEPTH_JUMP
            take = valid & apart & (dist < best)
            out[take], best[take] = other[take], dist
    return out


def to_camera(points: np.ndarray, cam_to_world: np.ndarray) -> np.ndarray:
    """World points (..., 3) in a camera's frame."""
    return (points - cam_to_world[:3, 3]) @ cam_to_world[:3, :3]


def cross_view(
    gt_pts: np.ndarray,
    gt_depth: np.ndarray,
    valid: np.ndarray,
    use: np.ndarray,
    cams: tuple[np.ndarray, np.ndarray],
    preds: list[np.ndarray],
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Count the views whose GT sees each GT pixel (V, H, W); per variant the disagreement of pairs from `use`."""
    k, c2w = cams
    n, h, w = gt_depth.shape
    seen = valid.astype(np.int64)
    gaps: list[list[np.ndarray]] = [[] for _ in preds]
    for i in range(n):
        idx = np.flatnonzero(valid[i])
        g = gt_pts[i].reshape(-1, 3)[idx]
        for j in range(n):
            if j == i:
                continue
            cam = to_camera(g, c2w[j])
            front = cam[:, 2] > 0
            pix = cam @ k[j].T
            with np.errstate(divide="ignore", invalid="ignore"):
                x = np.floor(pix[:, 0] / pix[:, 2]).astype(np.int64, copy=False)
                y = np.floor(pix[:, 1] / pix[:, 2]).astype(np.int64, copy=False)
            inside = front & (x >= 0) & (x < w) & (y >= 0) & (y < h)
            src, x, y = np.flatnonzero(inside), x[inside], y[inside]
            z_gt = gt_depth[j, y, x]
            rel_gt = (cam[src, 2] - z_gt) / np.where(z_gt > 0, z_gt, np.nan)
            hit = (z_gt > 0) & (np.abs(rel_gt) < VISIBLE)
            src, x, y, z_gt, rel_gt = src[hit], x[hit], y[hit], z_gt[hit], rel_gt[hit]
            seen[i].reshape(-1)[idx[src]] += 1
            keep = use[i].reshape(-1)[idx[src]]
            src, x, y, z_gt, rel_gt = (
                src[keep],
                x[keep],
                y[keep],
                z_gt[keep],
                rel_gt[keep],
            )
            for v, pred in enumerate(preds):
                z_i = to_camera(pred[i].reshape(-1, 3)[idx[src]], c2w[j])[:, 2]
                z_j = to_camera(pred[j, y, x], c2w[j])[:, 2]
                gap = np.abs((z_i - z_j) / z_gt - rel_gt)
                gaps[v].append(gap[np.isfinite(gap)])
    return seen, [np.concatenate(g) if g else np.zeros(0) for g in gaps]


def per_view_error(pred: np.ndarray, gt_pts: np.ndarray, use: np.ndarray) -> np.ndarray:
    """Per-pixel error (V, H, W) after one Sim(3) per view on its own pixel pairs, NaN off `use`."""
    out = np.full(use.shape, np.nan)
    for i in range(len(use)):
        if use[i].sum() < MIN_OBJECT:
            continue
        p, g = pred[i][use[i]], gt_pts[i][use[i]]
        s, rotation, t = umeyama(p, g)
        out[i][use[i]] = np.linalg.norm(s * p @ rotation.T + t - g, axis=1)
    return out


def object_error(
    pred: np.ndarray, gt: np.ndarray, ids: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Error before and after fitting each object's own scale and shift: min Σ|s p + t - g|²."""
    before, after = [], []
    for o in np.unique(ids[ids >= 0]):
        m = ids == o
        if m.sum() < MIN_OBJECT:
            continue
        p, g = pred[m], gt[m]
        pc, gc = p - p.mean(0), g - g.mean(0)
        s = float((pc * gc).sum() / (pc**2).sum())
        before.append(np.linalg.norm(p - g, axis=1))
        after.append(np.linalg.norm(s * pc - gc, axis=1))
    if not before:
        return np.zeros(0), np.zeros(0)
    return np.concatenate(before), np.concatenate(after)


def object_record(
    fit: tuple[np.ndarray, np.ndarray] | None, pixels: int
) -> dict[str, float]:
    """Pixel share and mean error before and after the object fit, NaN without objects."""
    if fit is None or not len(fit[0]):
        return dict.fromkeys(("object_px", "object_cm", "fit_cm"), float("nan"))
    before, after = fit
    return {
        "object_px": float(len(before) / pixels),
        "object_cm": 100 * float(before.mean()),
        "fit_cm": 100 * float(after.mean()),
    }


def bins(values: np.ndarray, edges: tuple[float, ...]) -> np.ndarray:
    """Bin index of each value: <= edges[0], ..., > edges[-1]."""
    return np.searchsorted(np.asarray(edges, dtype=float), values, side="left")


def contributions(
    change: np.ndarray, group: np.ndarray, labels: list[str]
) -> dict[str, tuple[float, float]]:
    """Per group: pixel share and contribution (cm) to the mean of `change`."""
    n = len(change)
    return {
        lab: (float((group == g).sum() / n), float(100 * change[group == g].sum() / n))
        for g, lab in enumerate(labels)
    }


COLUMNS = (
    ("error_cm", ".2f"),
    ("error_per_view_cm", ".2f"),
    ("cross_view_median_pct", ".2f"),
    ("cross_view_over_5pct", ".3f"),
    ("gt_object_px", ".2f"),
    ("gt_object_cm", ".2f"),
    ("gt_fit_cm", ".2f"),
    ("sam3_object_px", ".2f"),
    ("sam3_object_cm", ".2f"),
    ("sam3_fit_cm", ".2f"),
    ("glass_px_share", ".3f"),
    ("glass_error_cm", ".2f"),
    ("nonfinite_px", ".0f"),
    ("completeness_cm", ".2f"),
    ("scale", ".4f"),
)
EDGE_LABELS = ["<=1 px", "1-3 px", "3-10 px", ">10 px"]
VIEW_LABELS = ["1 view", "2-3", "4-7", "8+"]
QUART_LABELS = ["Q1 (low)", "Q2", "Q3", "Q4 (high)"]


def quartile(values: np.ndarray) -> np.ndarray:
    """Quartile index 0-3 of each value within its own array."""
    return bins(values, tuple(np.quantile(values, [0.25, 0.5, 0.75])))


def analyse(
    raw: dict,
    name: str,
    arrays: dict[str, np.ndarray],
    picked: list[int],
    *,
    maps: bool,
) -> dict:
    """Every number of the module docstring for one scene, on the picked variants."""
    paths = input_paths(raw, name)
    views, gt = Views.load(paths["views"]), GroundTruth.load(paths["gt"])
    geometry = Geometry.load(paths["geometry"])
    rays = scene_from_geometry(geometry).rays.double()
    k, c2w = views.intrinsics.astype(np.float64), views.cam_to_world.astype(np.float64)
    depth = gt.depth.astype(np.float64)
    valid = depth > 0
    gt_pts = np.stack(
        [lift(d, ki, ci) for d, ki, ci in zip(depth, k, c2w, strict=True)]
    )
    lut = np.full(max(int(gt.instance_ids.max()), 0) + 1, "", dtype=object)
    for i, c in gt.classes.items():
        lut[i] = c
    cls = np.where(
        gt.instance_ids >= 0, lut[np.maximum(gt.instance_ids, 0)], ""
    ).astype(str)
    glass = np.isin(cls, GLASS) & valid
    use = valid & ~glass

    preds, depths = [], []
    for v in picked:
        d = arrays[f"{name}/{v}"].astype(np.float64)
        a = arrays[f"{name}/{v}/align"]
        p = points_from_rays(torch.from_numpy(d), rays).numpy()
        preds.append(p @ a[:3, :3].T + a[:3, 3])
        depths.append(d)
    seen, gaps = cross_view(gt_pts, depth, valid, use, (k, c2w), preds)
    dist = edge_distance(depth, valid)
    ebin = bins(dist, EDGE_BINS)
    objects = {"gt": gt.instance_ids, "sam3": arrays[f"{name}/labels"]}
    if not gt.classes:
        del objects["gt"]

    out: dict = {"variants": []}
    comps = []
    for v, pred in enumerate(preds):
        err = np.linalg.norm(pred - gt_pts, axis=-1)
        finite = np.isfinite(err)
        u = use & finite
        view_err = per_view_error(pred, gt_pts, u)
        fits = {
            o: object_error(pred[u], gt_pts[u], ids[u]) for o, ids in objects.items()
        }
        found = np.isfinite(pred[valid]).all(1)
        comp = nearest_distances(pred[valid][found], gt_pts[valid])
        comps.append(comp)
        out["variants"].append(
            {
                "error_cm": 100 * float(err[u].mean()),
                "error_per_view_cm": 100 * float(np.nanmean(view_err[u])),
                "cross_view_median_pct": 100 * float(np.median(gaps[v])),
                "cross_view_over_5pct": float((gaps[v] > VISIBLE).mean()),
                "edges": contributions(err[u], ebin[u], EDGE_LABELS),
                **{
                    f"{o}_{key}": value
                    for o in ("gt", "sam3")
                    for key, value in object_record(fits.get(o), u.sum()).items()
                },
                "glass_px_share": float(glass.sum() / valid.sum()),
                "glass_error_cm": 100 * float(err[glass & finite].mean())
                if (glass & finite).any()
                else float("nan"),
                "nonfinite_px": int((~finite[valid]).sum()),
                "completeness_cm": 100 * float(comp.mean()),
                "scale": float(
                    np.cbrt(np.linalg.det(arrays[f"{name}/{picked[v]}/align"][:3, :3]))
                ),
            }
        )

    change = comps[-1] - comps[0]  # refined minus Déjà View, per GT pixel (m)
    conf = geometry.confidence[valid]
    moved = np.abs(depths[-1] - depths[0])[valid] / depths[0][valid]
    vbin = bins(seen[valid], VIEW_BINS)
    classes = sorted(set(cls[valid].tolist()))
    cidx = np.searchsorted(np.array(classes), cls[valid])
    out["completeness"] = {
        "change_cm": float(100 * change.mean()),
        "by_edge": contributions(change, ebin[valid], EDGE_LABELS),
        "by_views": contributions(change, vbin, VIEW_LABELS),
        "by_confidence": contributions(change, quartile(conf), QUART_LABELS),
        "by_moved": contributions(change, quartile(moved), QUART_LABELS),
        "by_class": contributions(change, cidx, [c or "unlabeled" for c in classes]),
    }
    near = valid & (dist <= COLLAPSE)
    other = across_edge(depth, valid)
    has = near & np.isfinite(other)
    collapse = []
    for pred in preds:
        z = np.stack([to_camera(pred[i], c2w[i])[..., 2] for i in range(len(pred))])
        ok = has & np.isfinite(z)
        collapse.append(float((np.abs(z - other) < np.abs(z - depth))[ok].mean()))
    out["collapse_near_edges"] = collapse
    out["near_edge_px_share"] = float(has.sum() / valid.sum())
    if maps:
        for v, comp in enumerate(comps):
            full = np.full(valid.shape, np.nan, dtype=np.float32)
            full[valid] = comp
            out[f"map/{v}"] = full
    return out


def groups_of(summary: dict) -> dict[str, list[str]]:
    """Scene groups of a run: Hypersim's splits with per-scene records, or ETH3D's kinds."""
    if "splits" in summary:
        return {
            s: list(r["scenes"]) for s, r in summary["splits"].items() if "scenes" in r
        }
    kinds = defaultdict(list)
    for name, record in summary["scenes"].items():
        kinds[record["kind"]].append(name)
    return dict(kinds) | {"all": list(summary["scenes"])}


def scored_completeness(summary: dict, name: str, variant: str) -> float:
    """Return the completeness the run itself reported for a scene and variant (cm)."""
    if "splits" in summary:
        for record in summary["splits"].values():
            if name in record.get("scenes", {}):
                return record["scenes"][name]["variants"][variant]["headline"][
                    "completeness_cm"
                ]
    return summary["scenes"][name]["variants"][variant]["completeness_cm"]


def mean_of(records: list[dict], path: tuple) -> float:
    """Mean over scenes of one nested value."""
    values = []
    for record in records:
        value = record
        for key in path:
            value = value[key]
        values.append(value)
    return float(np.nanmean(values))


def report(group: str, records: list[dict], variants: list[str]) -> None:
    """Print one group's split, averaged over its scenes."""
    print(f"\n## {group} ({len(records)} scenes)")
    print(
        "variant | error cm | per-view Sim(3) cm | cross-view median % | >5% | GT object px | GT object cm | after fit cm | SAM 3 object px | SAM 3 object cm | after fit cm | glass px | glass cm | non-finite px | comp cm | Sim(3) scale"
    )
    for v, name in enumerate(variants):
        cells = [
            f"{mean_of(records, ('variants', v, key)):{fmt}}" for key, fmt in COLUMNS
        ]
        print(f"{name} | " + " | ".join(cells))
    for v, name in enumerate(variants):
        print(f"edges, {name}: bin | px share | contribution to error cm")
        for lab in EDGE_LABELS:
            print(
                f"  {lab} | {mean_of(records, ('variants', v, 'edges', lab, 0)):.3f} | {mean_of(records, ('variants', v, 'edges', lab, 1)):.2f}"
            )
    change = mean_of(records, ("completeness", "change_cm"))
    print(
        f"completeness change, {variants[-1]} - {variants[0]}: {change:+.2f} cm; group | px share | contribution cm"
    )
    for key in ("by_edge", "by_views", "by_confidence", "by_moved"):
        labs = list(records[0]["completeness"][key])
        cells = [
            f"{lab} {mean_of(records, ('completeness', key, lab, 0)):.3f} / {mean_of(records, ('completeness', key, lab, 1)):+.2f}"
            for lab in labs
        ]
        print(f"  {key}: " + " | ".join(cells))
    pooled: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])
    for r in records:
        for lab, (share, contrib) in r["completeness"]["by_class"].items():
            pooled[lab][0] += share / len(records)
            pooled[lab][1] += contrib / len(records)
    top = sorted(pooled.items(), key=lambda kv: -kv[1][1])[:8]
    print(
        "  by_class (largest 8): "
        + " | ".join(f"{lab} {s:.3f} / {c:+.2f}" for lab, (s, c) in top)
    )
    collapse = [
        mean_of(records, ("collapse_near_edges", v)) for v in range(len(variants))
    ]
    share = mean_of(records, ("near_edge_px_share",))
    print(
        f"pulled across the edge (<= {COLLAPSE} px, {share:.3f} of px): "
        + ", ".join(f"{n} {c:.3f}" for n, c in zip(variants, collapse, strict=True))
    )


def main() -> None:
    """Split every scene of a run; see the module docstring."""
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--maps", nargs="*", default=[], help="scenes to save maps of")
    parser.add_argument("--variants", nargs=2, type=int, help="the two to compare")
    args = parser.parse_args()
    run: Path = args.run
    summary = yaml.safe_load((run / "run.yaml").read_text(encoding="utf-8"))
    raw = summary["config"]
    every = [v["name"] for v in raw["variants"]]
    picked = args.variants or [0, len(every) - 1]
    assert all(0 <= v < len(every) for v in picked), picked
    variants = [every[v] for v in picked]
    suffix = f"_{picked[0]}_{picked[1]}" if args.variants else ""
    groups = groups_of(summary)
    names = list(dict.fromkeys(n for g in groups.values() for n in g))
    with np.load(run / "results.npz") as f:
        per_scene = {
            n: {k: f[k] for k in f.files if k.startswith(f"{n}/")} for n in names
        }
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(WORKERS, mp_context=context) as pool:
        jobs = {
            n: pool.submit(analyse, raw, n, per_scene[n], picked, maps=n in args.maps)
            for n in names
        }
        records = {n: job.result() for n, job in jobs.items()}
    maps = {
        f"{n}/comp/{k.split('/')[1]}": records[n].pop(k)
        for n in names
        for k in list(records[n])
        if k.startswith("map/")
    }
    if maps:
        np.savez_compressed(run / f"e35_maps{suffix}.npz", **maps)
    (run / f"e35_split{suffix}.yaml").write_text(
        yaml.safe_dump(
            {
                "run": str(run),
                "git": summary["git"],
                "groups": groups,
                "scenes": records,
            },
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    print(f"run {run} (git {summary['git']}); variants {variants}")
    print(
        "check, completeness recomputed vs the run's (cm): scene | variant | here | run"
    )
    for n in names:
        for v, name in enumerate(variants):
            print(
                f"  {n} | {name} | {records[n]['variants'][v]['completeness_cm']:.3f} | {scored_completeness(summary, n, name):.3f}"
            )
    for group, members in groups.items():
        report(group, [records[n] for n in members], variants)
    print(
        "\n## per scene: error cm, per-view cm, cross-view >5%, comp cm (variant 0 -> 1), comp change by edge bin cm"
    )
    for n in names:
        r = records[n]
        a, b = r["variants"]
        edge = " ".join(f"{c[1]:+.2f}" for c in r["completeness"]["by_edge"].values())
        print(
            f"{n} | {a['error_cm']:.2f} -> {b['error_cm']:.2f} | {a['error_per_view_cm']:.2f} -> {b['error_per_view_cm']:.2f} | "
            f"{a['cross_view_over_5pct']:.3f} -> {b['cross_view_over_5pct']:.3f} | {a['completeness_cm']:.2f} -> {b['completeness_cm']:.2f} | {edge}"
        )


if __name__ == "__main__":
    main()
