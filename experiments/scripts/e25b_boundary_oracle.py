"""E25b: how much of Déjà View's error at depth edges could segments remove? An oracle bound.

Déjà View's output per scene, aligned onto the GT by the protocol's one Umeyama Sim(3) on
the pixels (`align_output`) and scored by `surface_metrics`. The band is the GT pixels
within BAND px of a GT depth edge; --edges picks E35's step rule (the default, E25b as
reported) or the plane rule of `src/analysis/edges.py`, which a plane at a grazing angle
passes, for the band, the surface across the edge and both depth components.

1. What the edge error is: for each band pixel with a GT pixel on the other surface within
   E35's COLLAPSE px (`across_edge`), t = (z_p - z_g) / (z_o - z_g) with z_g its GT depth,
   z_o the other surface's and z_p the aligned prediction's, all in the GT camera. Bins
   T_BINS: away from the other surface, near its own, between the two (bleeding), on or
   past the other (the edge drawn in the wrong place); pixel and error share per bin.
2. Oracle snap: per view and segment, the pixels within BAND px of a change of segment id
   take the inverse depth of a plane over pixel coordinates (a 3D plane through a pinhole
   is linear in inverse depth), least squares on Déjà View's inverse depth at the same
   segment's interior pixels (farther than BAND px from a change of id) in the
   WINDOW x WINDOW window around them. The fit is clamped to the range [lo, hi] of those
   pixels' inverse depth widened by its own ratio, [lo^2 / hi, hi^2 / lo]: a plane may
   extrapolate past its interior about as far as the interior spans, and stays positive.
   (Added after E25b's first look at dev: unclamped, a few planes extrapolated up to 464x
   the depth and dragged the scene's least-squares Sim(3) off; the plain range clamp
   cut a sloped plane's legitimate 4 px extrapolation in the synthetic check.) Fewer
   than MIN_FIT such pixels or interior pixels on one line: Déjà View's depth stays.

Segments: GT instances; GT depth components (the GT pixels cut at GT depth edges, each
edge pixel given its nearest component's id: the edges without semantics); SAM 3's
linked masks (the protocol's); Déjà View depth components (the same on its own depth).
Each segment is split into its 4-connected pieces per view (`pieces`; added after E25b's
first val run: Hypersim's walls are one instance per scene, so a window could hold two
walls at different depths, e.g. through a door, and the plane through both dragged the
Sim(3) of 3 of the 11 val scenes; a local fit should see one connected surface).

A config without `vocabulary` is Tanks and Temples (E45b): no labels, so no GT-instance
snap, and its groups are the config's (outdoor, indoor), not splits.

uv run python experiments/scripts/e25b_boundary_oracle.py --cfg configs/eval_hypersim.yaml --splits dev val --out runs/e25b --check runs/e8a_segments-20261110-130124/run.yaml
uv run python experiments/scripts/e25b_boundary_oracle.py --cfg configs/eval_tanksandtemples.yaml --edges plane --maps truck --out runs/e45b_tanksandtemples
prints the report and writes `<out>/e25b.yaml` (every record) and `<out>/e25b_maps.npz`
(the --maps scenes: each variant's per-pixel error and Déjà View's t), the inputs of
`e25b_figures.py`.
"""

import argparse
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
import yaml
from e25a_class_split import EDGE_RULES, edge_distance
from e35_error_split import across_edge as step_across
from e35_error_split import bins, to_camera
from scipy import ndimage

from src.analysis.edges import across_edge as plane_across
from src.analysis.scene_eval import git_hash, input_paths
from src.evaluation.labeled import align_output, ground_truth_cloud, surface_metrics
from src.geometry import Geometry
from src.loaders.ground_truth import GroundTruth
from src.loaders.hypersim.config import load_config as hypersim_config
from src.loaders.tanksandtemples.scene import load_config as tanksandtemples_config
from src.loaders.views import Views
from src.refinement.multiview import points_from_rays, scene_from_geometry
from src.segmentation.base import Segmentation
from src.segmentation.linked import working_associator

BAND = 3  # px from a change of segment id (snap) or from a GT depth edge (score)
WINDOW = 31  # px, side of the square window a snapped pixel's plane is fitted in
MIN_FIT = 30  # interior pixels of the same segment in the window
MIN_SPREAD = (
    1.0  # px^4, det of the interior pixels' coordinate covariance (not one line)
)
T_BINS = (0.0, 0.25, 0.75)  # t: <= 0 away, 0-0.25 own, 0.25-0.75 between, > 0.75 other
T_LABELS = ("away", "own", "between", "other")
SOURCES = ("GT instances", "GT depth comp.", "SAM 3", "DV depth comp.")
MARGIN = 0.05  # cm Chamfer, the plan's margin between two snaps
WORKERS = 4


def box_sum(a: np.ndarray, r: int) -> np.ndarray:
    """Sum of `a` (H, W) over the (2r + 1)^2 window around each pixel, zero outside the image."""
    c = np.pad(a, ((r + 1, r), (r + 1, r))).cumsum(0).cumsum(1)
    return (
        c[2 * r + 1 :, 2 * r + 1 :]
        - c[: -2 * r - 1, 2 * r + 1 :]
        - c[2 * r + 1 :, : -2 * r - 1]
        + c[: -2 * r - 1, : -2 * r - 1]
    )


def id_change(ids: np.ndarray) -> np.ndarray:
    """Pixels (H, W) whose id differs from a 4-neighbour's."""
    out = np.zeros(ids.shape, dtype=bool)
    h = ids[:, 1:] != ids[:, :-1]
    v = ids[1:] != ids[:-1]
    out[:, 1:] |= h
    out[:, :-1] |= h
    out[1:] |= v
    out[:-1] |= v
    return out


def plane_snap(inv: np.ndarray, ids: np.ndarray) -> np.ndarray:
    """Inverse depth (H, W) with each segment's boundary band replaced by local planes fitted to its interior."""
    assert inv.shape == ids.shape, (inv.shape, ids.shape)
    assert inv.ndim == 2, inv.shape
    assert np.isfinite(inv).all()
    assert (inv > 0).all()
    change = id_change(ids)
    if not change.any():
        return inv.copy()
    dist = ndimage.distance_transform_edt(~change)
    band, interior = (ids >= 0) & (dist <= BAND), (ids >= 0) & (dist > BAND)
    out = inv.copy()
    r = WINDOW // 2
    for s in np.unique(ids[band]):
        ys, xs = np.nonzero(band & (ids == s))
        y0, x0 = max(ys.min() - r, 0), max(xs.min() - r, 0)
        y1, x1 = (
            min(ys.max() + r + 1, inv.shape[0]),
            min(xs.max() + r + 1, inv.shape[1]),
        )
        m = (interior[y0:y1, x0:x1] & (ids[y0:y1, x0:x1] == s)).astype(np.float64)
        if m.sum() < MIN_FIT:
            continue
        v, u = np.mgrid[0 : y1 - y0, 0 : x1 - x0].astype(np.float64)
        w = inv[y0:y1, x0:x1]
        sums = [
            box_sum(m * f, r)[ys - y0, xs - x0]
            for f in (np.ones_like(u), u, v, u * u, u * v, v * v, w, u * w, v * w)
        ]
        n, su, sv, suu, suv, svv, sw, suw, svw = sums
        ok = n >= MIN_FIT
        n = np.where(ok, n, 1.0)
        mu, mv, mw = su / n, sv / n, sw / n
        cuu, cuv, cvv = suu / n - mu * mu, suv / n - mu * mv, svv / n - mv * mv
        cuw, cvw = suw / n - mu * mw, svw / n - mv * mw
        det = cuu * cvv - cuv * cuv
        ok &= det > MIN_SPREAD
        det = np.where(ok, det, 1.0)
        a = (cvv * cuw - cuv * cvw) / det
        b = (cuu * cvw - cuv * cuw) / det
        fit = mw + a * (xs - x0 - mu) + b * (ys - y0 - mv)
        lo = ndimage.minimum_filter(
            np.where(m > 0, w, np.inf), WINDOW, mode="constant", cval=np.inf
        )
        hi = ndimage.maximum_filter(
            np.where(m > 0, w, -np.inf), WINDOW, mode="constant", cval=-np.inf
        )
        lo, hi = lo[ys[ok] - y0, xs[ok] - x0], hi[ys[ok] - y0, xs[ok] - x0]
        out[ys[ok], xs[ok]] = np.clip(fit[ok], lo * lo / hi, hi * hi / lo)
    assert np.isfinite(out).all()
    assert (out > 0).all()
    return out


def snap_depth(depth: np.ndarray, ids: np.ndarray) -> np.ndarray:
    """Depth (V, H, W) after `plane_snap` in each view."""
    return np.stack([1 / plane_snap(1 / d, i) for d, i in zip(depth, ids, strict=True)])


def across_edge(depth: np.ndarray, valid: np.ndarray, rule: str) -> np.ndarray:
    """GT depth (V, H, W) of the nearest pixel within BAND px on another surface by `rule`, NaN if none."""
    if rule == "step":
        return step_across(depth, valid)
    return np.stack(
        [plane_across(d, v, BAND) for d, v in zip(depth, valid, strict=True)]
    )


def depth_components(depth: np.ndarray, valid: np.ndarray, rule: str) -> np.ndarray:
    """Ids (V, H, W) of connected components cut at `rule`'s depth edges, edge pixels joined to their nearest; -1 off `valid`."""
    out = np.full(depth.shape, -1, dtype=np.int64)
    for i in range(len(depth)):
        free = valid[i] & ~EDGE_RULES[rule](depth[i], valid[i])
        lab, n = ndimage.label(free)
        if not n:
            continue
        idx = ndimage.distance_transform_edt(
            lab == 0, return_distances=False, return_indices=True
        )
        lab = lab[idx[0], idx[1]]
        out[i] = np.where(valid[i], lab - 1, -1)
    return out


def pieces(ids: np.ndarray) -> np.ndarray:
    """Ids (V, H, W) with each segment split into its 4-connected pieces per view; negative ids become -1."""
    out = np.full(ids.shape, -1, dtype=np.int64)
    for i in range(len(ids)):
        start = 0
        for s in np.unique(ids[i][ids[i] >= 0]):
            lab, n = ndimage.label(ids[i] == s)
            out[i][lab > 0] = lab[lab > 0] - 1 + start
            start += n
    return out


def check() -> None:
    """Identity and known solution of the snap (the plan's checks), on CPU in well under a second."""
    rng = np.random.default_rng(0)
    v, u = np.mgrid[0:60, 0:80].astype(np.float64)
    plane = 0.2 + 0.001 * u - 0.0005 * v
    noisy = plane * (1 + 0.05 * rng.standard_normal(plane.shape))
    noisy = np.abs(noisy)
    assert np.array_equal(plane_snap(noisy, np.full(plane.shape, -1)), noisy)
    assert np.array_equal(
        plane_snap(noisy, np.zeros(plane.shape, dtype=np.int64)), noisy
    )
    two = np.where((u < 20) | (u >= 60), 5, -1)[None]
    assert np.array_equal(np.unique(pieces(two)), [-1, 0, 1])
    ids = (u >= 40).astype(np.int64)
    snapped = plane_snap(np.where(ids == 0, plane, noisy), ids)
    left = (ids == 0) & (np.abs(u - 39.5) < BAND + 1)
    assert np.allclose(snapped[left], plane[left], rtol=1e-9), np.abs(snapped - plane)[
        left
    ].max()


def scores(
    pred: np.ndarray,
    truth_points: np.ndarray,
    band: np.ndarray,
    ref_band_cm: float | None,
) -> dict:
    """Mean error over all pixels and in the band, and the protocol's surface scores (cm)."""
    e = np.linalg.norm(pred - truth_points, axis=1)
    assert np.isfinite(e).all()
    return {
        "error_cm": 100 * float(e.mean()),
        "band_error_cm": 100 * float(e[band].mean()),
        "band_removed": float("nan")
        if ref_band_cm is None
        else 1 - 100 * float(e[band].mean()) / ref_band_cm,
    }


def analyse(raw: dict, name: str, *, edges: str, maps: bool) -> dict:
    """Every number of the module docstring for one scene."""
    paths = input_paths(raw, name)
    views, gt = Views.load(paths["views"]), GroundTruth.load(paths["gt"])
    geometry = Geometry.load(paths["geometry"])
    labels = working_associator().associate(Segmentation.load(paths["masks"]), geometry)
    rays = scene_from_geometry(geometry).rays.double()
    truth = ground_truth_cloud(
        gt.depth, gt.instance_ids, gt.classes, views.intrinsics, views.cam_to_world
    )
    valid = truth.pixels
    assert valid is not None
    c2w = views.cam_to_world.astype(np.float64)
    depth = gt.depth.astype(np.float64)
    dv = geometry.depth.astype(np.float64)
    assert np.isfinite(dv).all()
    assert (dv > 0).all()
    band = (edge_distance(depth, valid, edges) <= BAND)[valid]
    other = across_edge(depth, valid, edges)
    everywhere = np.ones(dv.shape, dtype=bool)
    sources = {
        "GT instances": gt.instance_ids,
        "GT depth comp.": depth_components(depth, valid, edges),
        "SAM 3": labels.instance_ids,
        "DV depth comp.": depth_components(dv, everywhere, edges),
    }
    if not gt.classes:
        del sources["GT instances"]
    segments = {k: pieces(v) for k, v in sources.items()}
    out: dict = {
        "px": int(valid.sum()),
        "band_px_share": float(band.mean()),
        "variants": {},
    }
    maps_out: dict[str, np.ndarray] = {}
    ref_band = None
    for variant, ids in [("Déjà View", None), *segments.items()]:
        d = dv if ids is None else snap_depth(dv, ids)
        points = points_from_rays(torch.from_numpy(d), rays).numpy()
        pred, scale, _ = align_output(truth, points)
        rec = scores(pred, truth.points, band, ref_band)
        ref_band = rec["band_error_cm"] if ids is None else ref_band
        surface = surface_metrics(pred, truth, [])
        rec |= {k: surface[k] for k in ("accuracy_cm", "completeness_cm", "chamfer_cm")}
        rec |= {"scale": scale, "snapped_px": float((d != dv)[valid].mean())}
        full = np.full((*valid.shape, 3), np.nan)
        full[valid] = pred
        z = np.stack([to_camera(full[i], c2w[i])[..., 2] for i in range(len(full))])[
            valid
        ]
        o = other[valid]
        has = band & np.isfinite(o)
        t = (z[has] - depth[valid][has]) / (o[has] - depth[valid][has])
        e = np.linalg.norm(pred - truth.points, axis=1)[has]
        tb = bins(t, T_BINS)
        rec["partner_share"] = float(has.sum() / band.sum())
        rec["t_px"] = {lab: float((tb == k).mean()) for k, lab in enumerate(T_LABELS)}
        rec["t_error"] = {
            lab: float(e[tb == k].sum() / e.sum()) for k, lab in enumerate(T_LABELS)
        }
        out["variants"][variant] = rec
        if maps:
            err = np.full(valid.shape, np.nan, dtype=np.float32)
            err[valid] = np.linalg.norm(pred - truth.points, axis=1)
            maps_out[f"error/{variant}"] = err
            if ids is None:
                tmap = np.full(valid.shape, np.nan, dtype=np.float32)
                flat = np.flatnonzero(valid)[has]
                tmap.reshape(-1)[flat] = t
                maps_out["t"] = tmap
                maps_out["band"] = np.zeros(valid.shape, dtype=bool)
                maps_out["band"][valid] = band
    if maps:
        out["maps"] = maps_out
    return out


def split_mean(records: list[dict], variant: str, key: str) -> float:
    """Mean over scenes of one variant's score."""
    return float(np.mean([r["variants"][variant][key] for r in records]))


def wins(records: list[dict], a: str, b: str) -> tuple[int, float]:
    """Scenes where a's Chamfer is below b's by more than MARGIN, and the mean difference a - b (cm)."""
    diff = np.array(
        [
            r["variants"][a]["chamfer_cm"] - r["variants"][b]["chamfer_cm"]
            for r in records
        ]
    )
    return int((diff < -MARGIN).sum()), float(diff.mean())


def report(groups: dict[str, list[str]], records: dict[str, dict]) -> None:
    """Print the split tables, the t decomposition and the plan's comparisons."""
    for group, names in groups.items():
        recs = [records[n] for n in names]
        variants = [v for v in ("Déjà View", *SOURCES) if v in recs[0]["variants"]]
        band = np.mean([r["band_px_share"] for r in recs])
        print(f"\n## {group} ({len(recs)} scenes; band = {band:.3f} of the GT pixels)")
        print(
            "variant | snapped px | error cm | band error cm | band error removed | accuracy cm | completeness cm | Chamfer cm"
        )
        for v in variants:
            cells = [
                f"{split_mean(recs, v, k):{f}}"
                for k, f in (
                    ("snapped_px", ".3f"),
                    ("error_cm", ".2f"),
                    ("band_error_cm", ".2f"),
                )
            ]
            removed = (
                "-"
                if v == "Déjà View"
                else f"{split_mean(recs, v, 'band_removed'):.3f}"
            )
            cells += [removed] + [
                f"{split_mean(recs, v, k):.3f}"
                for k in ("accuracy_cm", "completeness_cm", "chamfer_cm")
            ]
            print(f"{v} | " + " | ".join(cells))
        partner = np.mean([r["variants"]["Déjà View"]["partner_share"] for r in recs])
        print(
            f"t of the band pixels with a GT pixel across the edge ({partner:.3f} of the band): variant | "
            + " | ".join(f"{lab} px / error" for lab in T_LABELS)
        )
        for v in variants:
            cells = [
                f"{np.mean([r['variants'][v]['t_px'][lab] for r in recs]):.3f} / {np.mean([r['variants'][v]['t_error'][lab] for r in recs]):.3f}"
                for lab in T_LABELS
            ]
            print(f"  {v} | " + " | ".join(cells))
        print(
            f"Chamfer, a vs b: scenes where a is lower by > {MARGIN} cm / {len(recs)}, mean a - b cm"
        )
        for a, b in (
            ("GT instances", "Déjà View"),
            ("GT instances", "GT depth comp."),
            ("SAM 3", "GT instances"),
            ("SAM 3", "DV depth comp."),
            ("DV depth comp.", "Déjà View"),
            ("SAM 3", "Déjà View"),
        ):
            if a in variants and b in variants:
                n, mean = wins(recs, a, b)
                print(f"  {a} vs {b}: {n} | {mean:+.3f}")


def main() -> None:
    """Analyse every scene, then report; see the module docstring."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=Path, required=True)
    parser.add_argument("--splits", nargs="*", default=["dev", "val"])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--maps", nargs="*", default=["ai_006_007_cam_00"])
    parser.add_argument("--edges", choices=sorted(EDGE_RULES), default="step")
    parser.add_argument(
        "--check",
        type=Path,
        help="a scene_eval run.yaml whose Déjà View scores must be reproduced",
    )
    args = parser.parse_args()
    check()
    raw = yaml.safe_load(args.cfg.read_text(encoding="utf-8"))
    if raw.get("vocabulary") == "hypersim":
        config = hypersim_config(Path(raw["dataset"]))
        groups = {
            s: [f"{n}_{c}" for n, c in config.scenes([s]).items()] for s in args.splits
        }
        groups["all"] = [n for s in args.splits for n in groups[s]]
    else:
        kinds = tanksandtemples_config(Path(raw["dataset"])).scenes
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
        np.savez_compressed(args.out / "e25b_maps.npz", **maps)
    git = git_hash()
    print(f"config {args.cfg}, git {git}, scenes {len(names)}, edges {args.edges}")
    print(
        "check, snap: identity with no segment and with one segment, a pinhole plane recovered to 1e-9 (passed)"
    )
    if args.check:
        run = yaml.safe_load(args.check.read_text(encoding="utf-8"))
        print(
            "check, Déjà View here vs the run's: scene | scale here | run | Chamfer cm here | run"
        )
        for split in run["splits"].values():
            for n, rec in split.get("scenes", {}).items():
                if n in records:
                    theirs = rec["variants"]["Déjà View"]
                    mine = records[n]["variants"]["Déjà View"]
                    print(
                        f"  {n} | {mine['scale']:.6f} | {theirs['debug']['scale']:.6f} | {mine['chamfer_cm']:.3f} | {theirs['headline']['chamfer_cm']:.3f}"
                    )
    report(groups, records)
    (args.out / "e25b.yaml").write_text(
        yaml.safe_dump(
            {
                "config": str(args.cfg),
                "git": git,
                "edges": args.edges,
                "groups": groups,
                "scenes": records,
            },
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
