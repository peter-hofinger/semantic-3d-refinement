"""E38: does completeness reward the spread of Déjà View's views, which the refiner merges.

Reads a run of E37 (`scene_eval` on Hypersim, `e35_eth3d_eval.py` on ETH3D): per scene
the depth of each variant and its scene Sim(3) onto the ground truth. Per variant and
GT point (every pixel with GT depth, all views, as the protocol's completeness):

- fused completeness C_f: distance to the nearest predicted point of any view;
- own-view completeness C_o: distance to the nearest predicted point of its own view;
- fusion bonus B = C_o - C_f >= 0: how much the other views cover for its own.

Averaged over the same GT points, the completeness change from variant A to B splits
exactly into ΔC_f = ΔC_o - ΔB: each view moving off the GT, and the views no longer
covering for each other. Also the spread: views with a predicted point within 2 and
5 cm of a GT point, and the share of GT points whose nearest point is another view's
(C_f < C_o). ΔC_o and -ΔB are split by Déjà View's confidence quartile (per scene, as
E35). The predicted cloud is the protocol's: the finite points of the pixels with GT.

uv run python experiments/scripts/e38_spread.py runs/<run> [--variants A B] [--maps <scene> ...] > experiments/results/e38_spread_<data>.txt
also writes `runs/<run>/e38_spread_A_B.yaml` and, for the `--maps` scenes,
`runs/<run>/e38_maps_A_B.npz` (`<scene>/{own,bonus}/<i>`: ΔC_o and -ΔB per GT pixel, m,
NaN without GT), the inputs of `e38_figures.py`.
"""

import argparse
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
import yaml
from e35_error_split import (
    QUART_LABELS,
    contributions,
    groups_of,
    mean_of,
    quartile,
    scored_completeness,
)
from scipy.spatial import cKDTree

from src.analysis.scene_eval import input_paths
from src.evaluation.labeled import lift, nearest_distances
from src.geometry import Geometry
from src.loaders.ground_truth import GroundTruth
from src.loaders.views import Views
from src.refinement.multiview import points_from_rays, scene_from_geometry

RADII = (0.02, 0.05)  # m: a view "covers" a GT point with a predicted point this close
MAX_VIEWS = 20
WORKERS = 6


def views_within(clouds: list[np.ndarray], queries: np.ndarray) -> np.ndarray:
    """Per query and radius (N, len(RADII)): the number of clouds with a point within it."""
    count = np.zeros((len(queries), len(RADII)), dtype=np.int64)
    for cloud in clouds:
        tree = cKDTree(cloud, balanced_tree=False, compact_nodes=False)
        d = tree.query(queries, distance_upper_bound=max(RADII), workers=-1)[0]
        count += d[:, None] < np.asarray(RADII)
    return count


def coverage(
    pred: np.ndarray, gt_pts: np.ndarray, valid: np.ndarray
) -> dict[str, np.ndarray]:
    """C_f, C_o (m) and views within RADII per GT point, in `valid`'s C order."""
    clouds = []
    for i in range(len(valid)):
        p = pred[i][valid[i]]
        clouds.append(p[np.isfinite(p).all(1)])
    assert all(len(c) for c in clouds), [len(c) for c in clouds]
    queries = gt_pts[valid]
    fused = nearest_distances(np.concatenate(clouds), queries)
    own = np.concatenate(
        [nearest_distances(c, gt_pts[i][valid[i]]) for i, c in enumerate(clouds)]
    )
    assert (own >= fused - 1e-9).all(), float((fused - own).max())
    return {"fused": fused, "own": own, "within": views_within(clouds, queries)}


def analyse(
    raw: dict,
    name: str,
    arrays: dict[str, np.ndarray],
    picked: list[int],
    *,
    maps: bool,
) -> dict:
    """Every number of the module docstring for one scene, on the two picked variants."""
    paths = input_paths(raw, name)
    views, gt = Views.load(paths["views"]), GroundTruth.load(paths["gt"])
    geometry = Geometry.load(paths["geometry"])
    rays = scene_from_geometry(geometry).rays.double()
    k, c2w = views.intrinsics.astype(np.float64), views.cam_to_world.astype(np.float64)
    depth = gt.depth.astype(np.float64)
    valid = depth > 0
    assert len(valid) <= MAX_VIEWS, len(valid)
    gt_pts = np.stack(
        [lift(d, ki, ci) for d, ki, ci in zip(depth, k, c2w, strict=True)]
    )
    cov = []
    for v in picked:
        d = arrays[f"{name}/{v}"].astype(np.float64)
        a = arrays[f"{name}/{v}/align"]
        p = points_from_rays(torch.from_numpy(d), rays).numpy()
        cov.append(coverage(p @ a[:3, :3].T + a[:3, 3], gt_pts, valid))

    out: dict = {"variants": []}
    for c in cov:
        bonus = c["own"] - c["fused"]
        hist = np.bincount(c["within"][:, 0], minlength=MAX_VIEWS + 1)
        out["variants"].append(
            {
                "fused_cm": 100 * float(c["fused"].mean()),
                "own_cm": 100 * float(c["own"].mean()),
                "bonus_cm": 100 * float(bonus.mean()),
                "other_view_share": float((c["fused"] < c["own"]).mean()),
                **{
                    f"views_within_{round(100 * r)}cm": float(c["within"][:, j].mean())
                    for j, r in enumerate(RADII)
                },
                "views_within_2cm_hist": (hist / hist.sum()).tolist(),
            }
        )
    d_own = cov[1]["own"] - cov[0]["own"]
    d_bonus = -((cov[1]["own"] - cov[1]["fused"]) - (cov[0]["own"] - cov[0]["fused"]))
    q = quartile(geometry.confidence[valid])
    out["change"] = {
        "fused_cm": float(100 * (d_own + d_bonus).mean()),
        "own_cm": float(100 * d_own.mean()),
        "bonus_cm": float(100 * d_bonus.mean()),
        "own_by_confidence": contributions(d_own, q, QUART_LABELS),
        "bonus_by_confidence": contributions(d_bonus, q, QUART_LABELS),
    }
    if maps:
        for key, change in (("own", d_own), ("bonus", d_bonus)):
            full = np.full(valid.shape, np.nan, dtype=np.float32)
            full[valid] = change
            for i in range(len(full)):
                out[f"map/{key}/{i}"] = full[i]
    return out


def report(group: str, records: list[dict], variants: list[str]) -> None:
    """Print one group's numbers, averaged over its scenes."""
    print(f"\n## {group} ({len(records)} scenes)")
    print(
        "variant | C_f cm | C_o cm | B cm | nearest from another view | views within 2 cm | within 5 cm"
    )
    for v, name in enumerate(variants):
        cells = [
            f"{mean_of(records, ('variants', v, key)):{fmt}}"
            for key, fmt in (
                ("fused_cm", ".2f"),
                ("own_cm", ".2f"),
                ("bonus_cm", ".2f"),
                ("other_view_share", ".3f"),
                ("views_within_2cm", ".2f"),
                ("views_within_5cm", ".2f"),
            )
        ]
        print(f"{name} | " + " | ".join(cells))
    f, o, b = (
        mean_of(records, ("change", key)) for key in ("fused_cm", "own_cm", "bonus_cm")
    )
    share = b / f if f else float("nan")
    print(
        f"change {variants[1]} - {variants[0]}: ΔC_f {f:+.2f} = ΔC_o {o:+.2f} + (-ΔB) {b:+.2f} cm; -ΔB share {share:.2f}"
    )
    for key in ("own_by_confidence", "bonus_by_confidence"):
        cells = [
            f"{lab} {mean_of(records, ('change', key, lab, 1)):+.2f}"
            for lab in QUART_LABELS
        ]
        print(f"  {key} (cm): " + " | ".join(cells))
    for v, name in enumerate(variants):
        hist = np.mean(
            [r["variants"][v]["views_within_2cm_hist"] for r in records], axis=0
        )
        print(
            f"  views within 2 cm, {name}: "
            + " ".join(f"{n}:{h:.3f}" for n, h in enumerate(hist) if h >= 0.0005)
        )


def main() -> None:
    """Analyse every scene of a run; see the module docstring."""
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--variants", nargs=2, type=int, default=[0, 1])
    parser.add_argument("--maps", nargs="*", default=[], help="scenes to save maps of")
    args = parser.parse_args()
    run: Path = args.run
    summary = yaml.safe_load((run / "run.yaml").read_text(encoding="utf-8"))
    raw = summary["config"]
    every = [v["name"] for v in raw["variants"]]
    picked = args.variants
    assert all(0 <= v < len(every) for v in picked), picked
    variants = [every[v] for v in picked]
    suffix = f"_{picked[0]}_{picked[1]}"
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
        f"{n}/{k.removeprefix('map/')}": records[n].pop(k)
        for n in names
        for k in list(records[n])
        if k.startswith("map/")
    }
    if maps:
        np.savez_compressed(run / f"e38_maps{suffix}.npz", **maps)
    (run / f"e38_spread{suffix}.yaml").write_text(
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
    print("check, C_f vs the run's completeness (cm): scene | variant | here | run")
    for n in names:
        for v, name in enumerate(variants):
            print(
                f"  {n} | {name} | {records[n]['variants'][v]['fused_cm']:.3f} | {scored_completeness(summary, n, name):.3f}"
            )
    for group, members in groups.items():
        report(group, [records[n] for n in members], variants)
    print("\n## per scene: C_f, C_o, B cm (variant 0 -> 1) | ΔC_o, -ΔB cm")
    for n in names:
        a, b = records[n]["variants"]
        c = records[n]["change"]
        print(
            f"{n} | {a['fused_cm']:.2f} -> {b['fused_cm']:.2f} | {a['own_cm']:.2f} -> {b['own_cm']:.2f} | "
            f"{a['bonus_cm']:.2f} -> {b['bonus_cm']:.2f} | {c['own_cm']:+.2f} {c['bonus_cm']:+.2f}"
        )


if __name__ == "__main__":
    main()
