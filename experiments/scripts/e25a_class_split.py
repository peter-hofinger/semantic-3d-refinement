"""E25a: Déjà View's error per class, beyond what depth, edges and viewing angle explain.

Déjà View's output per scene, aligned onto the GT by the protocol's one Umeyama Sim(3)
on the pixels (`align_output`). Per GT pixel: error e = 3D distance of its aligned point
to its own GT point, relative error r = e / GT depth. Expected relative error r̂ = the
scene's mean r over the GT pixels in the same geometric cell: distance to a GT depth
edge (E35's bins; --edges picks E35's step rule or the plane rule of
`src/analysis/edges.py`, which a plane at a grazing angle passes) x the scene's depth quartile x incidence angle of the GT surface
(cos bins ANGLE_BINS, normals from the GT point map; no normal is its own bin). By
construction the mean of r̂ over a scene equals the mean of r.

Per class and scene (>= MIN_PX pixels): excess = Σr / Σr̂ over its pixels (1 = as hard as
its geometry predicts), excess error = Σ(e - ê) per pixel of the scene (cm), ê the
scene's mean e in the pixel's cell (sums to 0 over a scene: the class's share of the
mean error beyond its geometry), the share of its error within EDGE_PX of a depth edge,
the cross-view gap of E35 (median over the
pairs the GT says see one surface, sources subsampled by STRIDE), the share of error
removed by fitting each object instance's own scale and shift (E35; Hypersim's walls,
floors and ceilings are one instance per class and are left out), and the scorer's
per-class accuracy and completeness.

Do the classes differ? S = standard deviation over classes (>= MIN_SCENES scenes) of the
mean log excess over scenes. Null: per scene the instances' class labels are permuted
among the instances, PERMUTATIONS times; pixels, geometry and grouping stay.

SAM 3: per-view masks, unlinked (a class needs no identity), each pixel within ERODE px
of a change of mask id dropped. On Hypersim the GT and SAM 3 classes are compared on the
same pixels (SAM 3's kept ones): slope and correlation of the per-class log excess. On
ETH3D, which has no labels, SAM 3's classes are the only ones. KITTI-360 (E45a) has GT
classes and is read as Hypersim; its groups are the drives (country, town). A config's
`max_gt_depth` (m) drops the GT beyond it before the alignment and every score.

uv run python experiments/scripts/e25a_class_split.py --cfg configs/eval_hypersim.yaml --splits dev val --out runs/e25a_hypersim
uv run python experiments/scripts/e25a_class_split.py --cfg configs/eval_eth3d.yaml --out runs/e25a_eth3d
uv run python experiments/scripts/e25a_class_split.py --cfg configs/eval_kitti360.yaml --edges plane --maps 0010_0000000984_0000001116 --out runs/e45a_kitti360
prints the report and writes `<out>/e25a.yaml` (every record) and `<out>/e25a_maps.npz`
(the --maps scenes: r, r̂ and the class of each pixel), the inputs of `e25a_figures.py`.
"""

import argparse
import multiprocessing
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
import yaml
from e35_error_split import EDGE_BINS, bins, quartile, to_camera
from e35_error_split import depth_edges as step_edges
from scipy import ndimage

from src.analysis.edges import depth_edges as plane_edges
from src.analysis.scene_eval import VOCABULARIES, git_hash, input_paths
from src.evaluation.labeled import align_output, ground_truth_cloud, surface_metrics
from src.geometry import Geometry
from src.loaders.eth3d.scene import load_config as eth3d_config
from src.loaders.ground_truth import STUFF_BASE, GroundTruth
from src.loaders.hypersim.config import load_config as hypersim_config
from src.loaders.kitti360.scene import load_config as kitti360_config
from src.loaders.views import Views
from src.refinement.multiview import points_from_rays, scene_from_geometry
from src.segmentation.base import Segmentation

ANGLE_BINS = (0.4, 0.7)  # cos of the incidence angle: <= 0.4, 0.4-0.7, > 0.7, none
EDGE_PX = 3  # px from a GT depth edge that count as "at the edge"
ERODE = 3  # px around a change of SAM 3 mask id that are dropped
MIN_PX = 2000  # pixels of a class in a scene for it to count there
MIN_SCENES = 3  # scenes a class needs for the class table and S
MIN_OBJECT = 200  # px of an instance for its scale + shift fit (E35)
VISIBLE = (
    0.05  # relative depth gap below which view j's GT sees view i's GT point (E35)
)
STRIDE = 4  # every STRIDE-th source pixel in the cross-view gaps
FAILS = 1.5  # excess beyond FAILS (or below 1 / FAILS) in >= 2/3 of a class's scenes
PERMUTATIONS = 1000
WORKERS = 6
EDGE_RULES = {"step": step_edges, "plane": plane_edges}


def edge_distance(depth: np.ndarray, valid: np.ndarray, rule: str) -> np.ndarray:
    """Distance (V, H, W) in px of each pixel to the nearest GT depth edge of its view, by `rule`."""
    out = np.full(depth.shape, np.inf)
    for i in range(len(depth)):
        edge = EDGE_RULES[rule](depth[i], valid[i])
        if edge.any():
            out[i] = ndimage.distance_transform_edt(~edge)
    return out


def incidence(points: np.ndarray, valid: np.ndarray, c2w: np.ndarray) -> np.ndarray:
    """|cos| between each GT pixel's surface normal and its viewing ray (V, H, W), NaN without one."""
    out = np.full(valid.shape, np.nan)
    for i in range(len(points)):
        n = np.cross(np.gradient(points[i], axis=1), np.gradient(points[i], axis=0))
        ray = points[i] - c2w[i, :3, 3]
        with np.errstate(
            divide="ignore", invalid="ignore"
        ):  # pixels without GT, dropped below
            cos = np.abs((n * ray).sum(-1)) / (
                np.linalg.norm(n, axis=-1) * np.linalg.norm(ray, axis=-1)
            )
        inner = ndimage.binary_erosion(valid[i], border_value=0) & np.isfinite(cos)
        out[i][inner] = cos[inner]
    return out


def expected(r: np.ndarray, cell: np.ndarray) -> np.ndarray:
    """Mean of r over the pixels of each one's cell, per pixel."""
    sums = np.bincount(cell, weights=r)
    counts = np.bincount(cell)
    return sums[cell] / counts[cell]


def eroded(ids: np.ndarray) -> np.ndarray:
    """Pixels (V, H, W) farther than ERODE px from a change of mask id in their view."""
    out = np.zeros(ids.shape, dtype=bool)
    for i in range(len(ids)):
        change = np.zeros(ids[i].shape, dtype=bool)
        h = ids[i][:, 1:] != ids[i][:, :-1]
        v = ids[i][1:] != ids[i][:-1]
        change[:, 1:] |= h
        change[:, :-1] |= h
        change[1:] |= v
        change[:-1] |= v
        out[i] = (
            ndimage.distance_transform_edt(~change) > ERODE if change.any() else True
        )
    return out


def class_map(ids: np.ndarray, classes: dict[int, str]) -> np.ndarray:
    """Class name of each pixel's instance (V, H, W), "" without one."""
    out = np.full(ids.shape, "", dtype=object)
    for i, c in classes.items():
        out[ids == i] = c
    return out.astype(str)


def pair_gaps(
    gt_pts: np.ndarray,
    depth: np.ndarray,
    valid: np.ndarray,
    cams: tuple[np.ndarray, np.ndarray],
    pred: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Flat source pixel and |predicted - GT relative depth gap| of each pair that sees one surface (E35)."""
    k, c2w = cams
    n, h, w = depth.shape
    sources, gaps = [], []
    for i in range(n):
        idx = np.flatnonzero(valid[i])[::STRIDE]
        g = gt_pts[i].reshape(-1, 3)[idx]
        p_i = pred[i].reshape(-1, 3)[idx]
        for j in range(n):
            if j == i:
                continue
            cam = to_camera(g, c2w[j])
            pix = cam @ k[j].T
            with np.errstate(
                divide="ignore", invalid="ignore"
            ):  # behind the camera, dropped
                x = np.floor(pix[:, 0] / pix[:, 2])
                y = np.floor(pix[:, 1] / pix[:, 2])
            inside = (cam[:, 2] > 0) & (x >= 0) & (x < w) & (y >= 0) & (y < h)
            src = np.flatnonzero(inside)
            x, y = x[inside].astype(np.int64), y[inside].astype(np.int64)
            z_gt = depth[j, y, x]
            with np.errstate(
                divide="ignore", invalid="ignore"
            ):  # no GT in view j, dropped
                rel_gt = (cam[src, 2] - z_gt) / z_gt
            hit = (z_gt > 0) & (np.abs(rel_gt) < VISIBLE)
            src, x, y, z_gt, rel_gt = src[hit], x[hit], y[hit], z_gt[hit], rel_gt[hit]
            z_i = to_camera(p_i[src], c2w[j])[:, 2]
            z_j = to_camera(pred[j, y, x], c2w[j])[:, 2]
            gap = np.abs((z_i - z_j) / z_gt - rel_gt)
            ok = np.isfinite(gap)
            sources.append(i * h * w + idx[src[ok]])
            gaps.append(gap[ok])
    return np.concatenate(sources), np.concatenate(gaps)


def object_fit(
    pred: np.ndarray, gt: np.ndarray, ids: np.ndarray, cls: np.ndarray
) -> dict[str, list[float]]:
    """Per class: error summed before and after each object instance's own scale + shift fit (E35)."""
    out: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])
    for o in np.unique(ids[(ids >= 0) & (ids < STUFF_BASE)]):
        m = ids == o
        if m.sum() < MIN_OBJECT:
            continue
        p, g = pred[m], gt[m]
        pc, gc = p - p.mean(0), g - g.mean(0)
        s = float((pc * gc).sum() / (pc**2).sum())
        out[str(cls[m][0])][0] += float(np.linalg.norm(p - g, axis=1).sum())
        out[str(cls[m][0])][1] += float(np.linalg.norm(s * pc - gc, axis=1).sum())
    return dict(out)


def per_class(
    cls: np.ndarray, keys: dict[str, np.ndarray], total: int
) -> dict[str, dict[str, float]]:
    """Per class with >= MIN_PX px: px share and the sums in `keys` (all per-pixel arrays over the same pixels)."""
    out = {}
    for c in np.unique(cls):
        m = cls == c
        if not c or m.sum() < MIN_PX:
            continue
        out[str(c)] = {"px": int(m.sum()), "px_share": float(m.sum() / total)} | {
            k: float(v[m].sum()) for k, v in keys.items()
        }
    return out


def analyse(raw: dict, name: str, *, edges: str, maps: bool) -> dict:
    """Every number of the module docstring for one scene."""
    gt_classes = "vocabulary" in raw
    paths = input_paths(raw, name)
    views, gt = Views.load(paths["views"]), GroundTruth.load(paths["gt"])
    gt_depth = np.where(gt.depth <= raw.get("max_gt_depth", np.inf), gt.depth, 0)
    geometry = Geometry.load(paths["geometry"])
    masks = Segmentation.load(paths["masks"])
    scene = scene_from_geometry(geometry)
    points = points_from_rays(
        torch.from_numpy(geometry.depth).double(), scene.rays.double()
    ).numpy()
    truth = ground_truth_cloud(
        gt_depth, gt.instance_ids, gt.classes, views.intrinsics, views.cam_to_world
    )
    pred, scale, _ = align_output(truth, points)
    valid = truth.pixels
    assert pred.shape == truth.points.shape
    assert valid is not None
    k, c2w = views.intrinsics.astype(np.float64), views.cam_to_world.astype(np.float64)
    depth = gt_depth.astype(np.float64)
    gt_pts = np.zeros((*valid.shape, 3))
    gt_pts[valid] = truth.points
    pred_full = np.full((*valid.shape, 3), np.nan)
    pred_full[valid] = pred

    e = np.linalg.norm(pred - truth.points, axis=1)
    finite = np.isfinite(e)
    assert finite.all(), f"{name}: {int((~finite).sum())} non-finite Déjà View points"
    d = depth[valid]
    r = e / d
    dist = edge_distance(depth, valid, edges)
    cos = incidence(gt_pts, valid, c2w)
    angle = np.where(
        np.isfinite(cos[valid]), bins(np.nan_to_num(cos[valid]), ANGLE_BINS), 3
    )
    cell = bins(dist[valid], EDGE_BINS) * 16 + quartile(d) * 4 + angle
    r_hat = expected(r, cell)
    assert np.isclose(r_hat.sum(), r.sum(), rtol=1e-9), (r_hat.sum(), r.sum())
    keys = {
        "r": r,
        "r_hat": r_hat,
        "e": e,
        "excess_e": e - expected(e, cell),
        "e_edge": np.where(dist[valid] <= EDGE_PX, e, 0.0),
    }
    assert abs(keys["excess_e"].sum()) < 1e-9 * e.sum(), keys["excess_e"].sum()
    sources, gaps = pair_gaps(gt_pts, depth, valid, (k, c2w), pred_full)
    flat = np.full(valid.size, -1, dtype=np.int64)
    flat[np.flatnonzero(valid)] = np.arange(valid.sum())
    gap_px = flat[sources]

    sam3 = class_map(masks.instance_ids, masks.classes)
    keep = (eroded(masks.instance_ids) & (sam3 != ""))[valid]
    out: dict = {
        "scale": scale,
        "px": int(valid.sum()),
        "error_cm": 100 * float(e.mean()),
        "relative_error_pct": 100 * float(r.mean()),
        "cell_px": np.bincount(cell, minlength=64).tolist(),
        "gap_median_pct": 100 * float(np.median(gaps)),
        "sam3_kept_share": float(keep.mean()),
        "sam3": per_class(
            sam3[valid][keep], {k: v[keep] for k, v in keys.items()}, len(r)
        ),
    }
    out["sam3_gaps"] = {
        c: 100 * float(np.median(gaps[sam3[valid][gap_px] == c])) for c in out["sam3"]
    }
    if gt_classes:
        cls = class_map(gt.instance_ids, gt.classes)[valid]
        out["gt"] = per_class(cls, keys, len(r))
        out["gt_on_sam3_px"] = per_class(
            cls[keep], {k: v[keep] for k, v in keys.items()}, len(r)
        )
        out["gt_gaps"] = {
            c: 100 * float(np.median(gaps[cls[gap_px] == c])) for c in out["gt"]
        }
        out["gt_fit"] = object_fit(pred, truth.points, gt.instance_ids[valid], cls)
        surface = surface_metrics(pred, truth, list(VOCABULARIES[raw["vocabulary"]]))
        out["gt_surface"] = {
            c: {
                "accuracy_cm": s["accuracy_cm"],
                "completeness_cm": s["completeness_cm"],
            }
            for c, s in surface["surface_per_class"].items()
        }
        ids = gt.instance_ids[valid]
        inst, where = np.unique(ids[cls != ""], return_inverse=True)
        labelled = cls != ""
        out["instances"] = {
            "class": [str(cls[labelled][where == n][0]) for n in range(len(inst))],
            "px": np.bincount(where).tolist(),
            "r": np.bincount(where, weights=r[labelled]).tolist(),
            "r_hat": np.bincount(where, weights=r_hat[labelled]).tolist(),
        }
    if maps:
        full = {
            key: np.full(valid.shape, np.nan, dtype=np.float32)
            for key in ("r", "r_hat")
        }
        full["r"][valid], full["r_hat"][valid] = r, r_hat
        out["maps"] = full | {
            "gt_class": class_map(gt.instance_ids, gt.classes) if gt_classes else sam3,
            "sam3_class": np.where(eroded(masks.instance_ids), sam3, ""),
        }
    return out


def log_excess(records: list[dict], key: str) -> dict[str, list[float]]:
    """Per class, log excess in each scene where it counts."""
    out: dict[str, list[float]] = defaultdict(list)
    for rec in records:
        for c, s in rec[key].items():
            out[c].append(float(np.log(s["r"] / s["r_hat"])))
    return dict(out)


def spread(per_scene: dict[str, list[float]]) -> float:
    """S: standard deviation over classes with >= MIN_SCENES scenes of their mean log excess."""
    means = [np.mean(v) for v in per_scene.values() if len(v) >= MIN_SCENES]
    return float(np.std(means)) if len(means) > 1 else 0.0


def permuted_spread(instances: list[dict], rng: np.random.Generator) -> float:
    """S after permuting each scene's instance class labels among its instances."""
    out: dict[str, list[float]] = defaultdict(list)
    for inst in instances:
        cls = rng.permutation(np.array(inst["class"]))
        px, r, r_hat = (np.array(inst[k]) for k in ("px", "r", "r_hat"))
        for c in np.unique(cls):
            m = cls == c
            if px[m].sum() >= MIN_PX:
                out[str(c)].append(float(np.log(r[m].sum() / r_hat[m].sum())))
    return spread(out)


def class_table(
    records: list[dict], key: str, extra: dict[str, dict[str, str]]
) -> list[str]:
    """Rows: class | scenes | px share | excess (geo mean, range) | scenes beyond FAILS | excess error cm | edge share | ...extra."""
    per = log_excess(records, key)
    rows = []
    for c, logs in sorted(per.items(), key=lambda kv: -np.mean(kv[1])):
        if len(logs) < MIN_SCENES:
            continue
        hit = [rec[key][c] for rec in records if c in rec[key]]
        share = np.mean([h["px_share"] for h in hit])
        excess_cm = 100 * np.mean(
            [
                h["excess_e"] / rec["px"]
                for rec in records
                for cc, h in rec[key].items()
                if cc == c
            ]
        )
        edge = sum(h["e_edge"] for h in hit) / sum(h["e"] for h in hit)
        beyond = sum(abs(v) > np.log(FAILS) for v in logs)
        same = max(
            sum(v > np.log(FAILS) for v in logs), sum(v < -np.log(FAILS) for v in logs)
        )
        flag = " *" if same >= 2 * len(logs) / 3 else ""
        cells = [extra.get(c, {}).get(k, "-") for k in ("gap", "fit", "acc", "comp")]
        rows.append(
            f"{c}{flag} | {len(logs)} | {share:.3f} | {np.exp(np.mean(logs)):.2f} "
            f"({np.exp(min(logs)):.2f}-{np.exp(max(logs)):.2f}) | {beyond} | {excess_cm:+.3f} | {edge:.2f} | "
            + " | ".join(cells)
        )
    return rows


def gt_extra(records: list[dict]) -> dict[str, dict[str, str]]:
    """Per GT class: median cross-view gap %, error share the object fit removes, scorer accuracy / completeness cm."""
    out: dict[str, dict[str, str]] = defaultdict(dict)
    gaps, fits, acc, comp = (
        defaultdict(list),
        defaultdict(lambda: [0.0, 0.0]),
        defaultdict(list),
        defaultdict(list),
    )
    for rec in records:
        for c, g in rec["gt_gaps"].items():
            gaps[c].append(g / rec["gap_median_pct"])
        for c, (before, after) in rec["gt_fit"].items():
            fits[c][0] += before
            fits[c][1] += after
        for c, s in rec["gt_surface"].items():
            acc[c].append(s["accuracy_cm"])
            comp[c].append(s["completeness_cm"])
    for c in gaps:
        out[c]["gap"] = f"{np.median(gaps[c]):.2f}"
    for c, (before, after) in fits.items():
        out[c]["fit"] = f"{1 - after / before:.2f}"
    for c in acc:
        out[c]["acc"] = f"{np.nanmean(acc[c]):.2f}"
        out[c]["comp"] = f"{np.nanmean(comp[c]):.2f}"
    return dict(out)


def calibration(records: list[dict]) -> tuple[float, float, int]:
    """Slope and correlation of SAM 3's per-class mean log excess on GT's, same pixels; classes compared."""
    gt, sam3 = log_excess(records, "gt_on_sam3_px"), log_excess(records, "sam3")
    shared = [
        c for c in gt if c in sam3 and min(len(gt[c]), len(sam3[c])) >= MIN_SCENES
    ]
    x = np.array([np.mean(gt[c]) for c in shared])
    y = np.array([np.mean(sam3[c]) for c in shared])
    slope = float((x - x.mean()) @ (y - y.mean()) / ((x - x.mean()) ** 2).sum())
    return slope, float(np.corrcoef(x, y)[0, 1]), len(shared)


HEADER = "class | scenes | px share | excess (range) | scenes beyond 1.5x | excess error cm | error share at edges | cross-view gap / scene's | fit removes | accuracy cm | completeness cm"


def report(
    groups: dict[str, list[str]], records: dict[str, dict], *, gt_classes: bool
) -> None:
    """Print the class tables, the permutation test and the calibration."""
    for group, names in groups.items():
        recs = [records[n] for n in names]
        print(
            f"\n## {group} ({len(recs)} scenes); * = beyond {FAILS}x the same way in >= 2/3 of its scenes"
        )
        if gt_classes:
            print(f"\n### GT classes\n{HEADER}")
            print("\n".join(class_table(recs, "gt", gt_extra(recs))))
        sam3_extra = defaultdict(dict)
        for rec in recs:
            for c, g in rec["sam3_gaps"].items():
                sam3_extra[c].setdefault("_", []).append(g / rec["gap_median_pct"])
        sam3_extra = {
            c: {"gap": f"{np.median(v['_']):.2f}"} for c, v in sam3_extra.items()
        }
        print(
            f"\n### SAM 3 classes, eroded (kept {np.mean([r['sam3_kept_share'] for r in recs]):.3f} of the GT pixels)\n{HEADER}"
        )
        print("\n".join(class_table(recs, "sam3", sam3_extra)))


def main() -> None:
    """Analyse every scene, then report; see the module docstring."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=Path, required=True)
    parser.add_argument("--splits", nargs="*", default=["dev", "val"])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--maps", nargs="*", default=["ai_006_007_cam_00"])
    parser.add_argument("--edges", choices=sorted(EDGE_RULES), default="step")
    parser.add_argument(
        "--scales",
        type=Path,
        help="a scene_eval run.yaml to check the alignment scales against",
    )
    args = parser.parse_args()
    raw = yaml.safe_load(args.cfg.read_text(encoding="utf-8"))
    gt_classes = "vocabulary" in raw
    if raw.get("vocabulary") == "hypersim":
        config = hypersim_config(Path(raw["dataset"]))
        groups = {
            s: [f"{n}_{c}" for n, c in config.scenes([s]).items()] for s in args.splits
        }
        groups["all"] = [n for s in args.splits for n in groups[s]]
    else:
        load = kitti360_config if gt_classes else eth3d_config
        kinds = load(Path(raw["dataset"])).scenes
        groups = {
            kind: [n for n, k in kinds.items() if k == kind]
            for kind in sorted(set(kinds.values()))
        }
        groups["all"] = list(kinds)
    names = groups["all"]
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(WORKERS, mp_context=context) as pool:
        jobs = {
            n: pool.submit(analyse, raw, n, edges=args.edges, maps=n in args.maps)
            for n in names
        }
        records = {n: job.result() for n, job in jobs.items()}
    maps = {f"{n}/{k}": v for n in names for k, v in records[n].pop("maps", {}).items()}
    args.out.mkdir(parents=True, exist_ok=True)
    if maps:
        np.savez_compressed(args.out / "e25a_maps.npz", **maps)
    git = git_hash()
    print(f"config {args.cfg}, git {git}, scenes {len(names)}, edges {args.edges}")
    print("check, excess over all pixels = 1 by construction (asserted per scene)")
    if args.scales:
        run = yaml.safe_load(args.scales.read_text(encoding="utf-8"))
        print("check, alignment scale here vs the run's Déjà View: scene | here | run")
        for split in run["splits"].values():
            for n, rec in split.get("scenes", {}).items():
                if n in records:
                    print(
                        f"  {n} | {records[n]['scale']:.6f} | {rec['variants']['Déjà View']['debug']['scale']:.6f}"
                    )
    report(groups, records, gt_classes=gt_classes)
    if gt_classes:
        recs = [records[n] for n in names]
        s = spread(log_excess(recs, "gt"))
        rng = np.random.default_rng(0)
        null = np.array(
            [
                permuted_spread([r["instances"] for r in recs], rng)
                for _ in range(PERMUTATIONS)
            ]
        )
        p = float((1 + (null >= s).sum()) / (1 + PERMUTATIONS))
        print(f"\n## Do the GT classes differ? (all {len(recs)} scenes)")
        print(
            f"S = {s:.3f}; null (labels permuted among instances, {PERMUTATIONS}x): median {np.median(null):.3f}, 95th percentile {np.quantile(null, 0.95):.3f}; p = {p:.3f}"
        )
        slope, corr, n = calibration(recs)
        print(
            f"SAM 3 vs GT classes on the same pixels, per-class mean log excess over {n} classes: slope {slope:.2f}, correlation {corr:.2f}"
        )
        print(
            f"S on those pixels: GT {spread(log_excess(recs, 'gt_on_sam3_px')):.3f}, SAM 3 {spread(log_excess(recs, 'sam3')):.3f}"
        )
        summary = {
            "S": s,
            "null": null.tolist(),
            "p": p,
            "slope": slope,
            "correlation": corr,
        }
    else:
        summary = {}
    (args.out / "e25a.yaml").write_text(
        yaml.safe_dump(
            {
                "config": str(args.cfg),
                "git": git,
                "edges": args.edges,
                "groups": groups,
                "summary": summary,
                "scenes": records,
            },
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
