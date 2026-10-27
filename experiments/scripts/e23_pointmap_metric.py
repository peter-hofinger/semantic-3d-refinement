"""E23: the Déjà View paper's point-map metric on our two ScanNet++ scenes.

The paper (arXiv 2605.30215) reports on ScanNet++ Rel. L2 0.015 and inlier ratio
93.3% (point error < 3%), after one Sim(3) alignment for the whole scene. Here:
Déjà View world points (its own frame, K=16, 20 views at 504 px) against GT world
points (GT depth rendered from the 5 cm mesh, lifted with the GT cameras), on every
pixel where the GT ray hits the mesh. Alignment: Umeyama Sim(3) least squares over
all these pixels (no outlier rejection).

The normaliser of "relative" is not stated in our notes, so two are reported:
- per-ray: |aligned - GT| / distance of the GT point from its own GT camera centre
  (independent of the world origin),
- first camera: |aligned - GT| / |GT| with points in the first view's GT camera
  frame (common in point-map papers).
Also per view: one Sim(3) per view instead of one for the scene; the gap to the
scene-level numbers is the cross-view drift.

uv run python experiments/scripts/e23_pointmap_metric.py > experiments/results/e23_pointmap_metric.txt
"""

from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.geometry import Geometry
from src.loaders import Views
from src.loaders.scannetpp.gt import NO_MESH, GroundTruth

DATA = Path("data/processed/scannetpp")
FIGURES = Path("experiments/figures")
SCENES = {"5ee7c22ba0": "kitchen", "7b6477cb95": "office"}
VIEWS = "views_n20_uniform_504"
GEOMETRY = f"dejaview_k16_{VIEWS}"
INLIER = 0.03
PAPER = (0.015, 0.933)


def umeyama(src: np.ndarray, dst: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Similarity (s, R, t) minimising |s R src + t - dst|^2."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    a, b = src - mu_s, dst - mu_d
    u, sig, vt = np.linalg.svd(b.T @ a / len(src))
    d = np.eye(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        d[2, 2] = -1
    rot = u @ d @ vt
    scale = np.trace(np.diag(sig) @ d) / a.var(0).sum()
    return scale, rot, mu_d - scale * rot @ mu_s


def metrics(err: np.ndarray, norm: np.ndarray) -> tuple[float, float, float]:
    rel = err / norm
    return rel.mean(), np.median(rel), (rel < INLIER).mean()


def main() -> None:
    rows = {}
    per_view = {}
    for scene, name in SCENES.items():
        d = DATA / scene
        views = Views.load(d / f"{VIEWS}.npz")
        gt = GroundTruth.load(d / f"gt_{VIEWS}.npz")
        geometry = Geometry.load(d / f"{GEOMETRY}.npz")
        valid = (gt.instance_ids != NO_MESH) & (gt.depth > 0) & np.isfinite(geometry.depth)
        gt_geom = Geometry(gt.depth, np.ones_like(gt.depth), views.intrinsics, views.cam_to_world)
        gt_pts = gt_geom.world_points()
        dv_pts = geometry.world_points()
        centres = views.cam_to_world[:, :3, 3]
        ray = np.linalg.norm(gt_pts - centres[:, None, None], axis=-1)
        # GT points in the first view's camera frame, for the first-camera normaliser
        w2c0 = np.linalg.inv(views.cam_to_world[0].astype(np.float64))
        first = np.linalg.norm(gt_pts @ w2c0[:3, :3].T + w2c0[:3, 3], axis=-1)

        src, dst = dv_pts[valid].astype(np.float64), gt_pts[valid].astype(np.float64)
        s, rot, t = umeyama(src, dst)
        err = np.linalg.norm(src @ (s * rot).T + t - dst, axis=-1)
        rows[(name, "scene Sim(3)", "per-ray")] = metrics(err, ray[valid])
        rows[(name, "scene Sim(3)", "first camera")] = metrics(err, first[valid])
        err_map = np.full(valid.shape, np.nan)
        err_map[valid] = err / ray[valid]

        # one Sim(3) per view
        errs, norms, firsts, view_rel = [], [], [], []
        for i in range(len(views)):
            v = valid[i]
            si, ri, ti = umeyama(dv_pts[i][v].astype(np.float64), gt_pts[i][v].astype(np.float64))
            e = np.linalg.norm(dv_pts[i][v] @ (si * ri).T + ti - gt_pts[i][v], axis=-1)
            errs.append(e)
            norms.append(ray[i][v])
            firsts.append(first[i][v])
            view_rel.append((np.nanmean(err_map[i]), (e / ray[i][v]).mean()))
        err_v, ray_v, first_v = np.concatenate(errs), np.concatenate(norms), np.concatenate(firsts)
        rows[(name, "per-view Sim(3)", "per-ray")] = metrics(err_v, ray_v)
        rows[(name, "per-view Sim(3)", "first camera")] = metrics(err_v, first_v)
        per_view[name] = np.array(view_rel)
        print(
            f"{name} ({scene}): {valid.sum()} points ({valid.mean():.1%} of the pixels), scene Sim(3) scale {s:.4f}, "
            f"mean ray length {ray[valid].mean():.2f} m"
        )

    print()
    print(f"Paper (ScanNet++, whole-scene Sim(3)): Rel. L2 {PAPER[0]}, inlier ratio (< {INLIER:.0%}) {PAPER[1]:.1%}")
    print()
    print("| scene | alignment | normaliser | Rel. L2 (mean) | median | inlier ratio < 3% |")
    print("|---|---|---|---|---|---|")
    for (name, align, norm), (mean, med, inl) in rows.items():
        print(f"| {name} | {align} | {norm} | {mean:.4f} | {med:.4f} | {inl:.1%} |")
    print()
    print("Per view, mean relative error (per-ray normaliser): scene Sim(3) / own Sim(3)")
    for name, vals in per_view.items():
        print(f"{name}: " + ", ".join(f"{i}: {a:.3f}/{b:.3f}" for i, (a, b) in enumerate(vals)))

    fig, axes = plt.subplots(1, 2, figsize=(13, 3.8), sharey=True)
    for ax, (name, vals) in zip(axes, per_view.items(), strict=True):
        x = np.arange(len(vals))
        ax.bar(x - 0.2, vals[:, 0], 0.4, color="#2a78d6", label="one Sim(3) for the scene")
        ax.bar(x + 0.2, vals[:, 1], 0.4, color="#7fb1ea", label="one Sim(3) per view")
        ax.axhline(PAPER[0], color="#52514e", linestyle="--", linewidth=1, label="paper, ScanNet++ (scene)")
        ax.set_xticks(x)
        ax.set_xlabel("view")
        ax.set_title(f"{name}: mean relative point error per view", fontsize=10)
        ax.grid(axis="y", color="#e5e4df", linewidth=0.6)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("|aligned - GT| / GT ray length")
    axes[0].legend(fontsize=8, frameon=False)
    fig.suptitle("E23: Déjà View point maps vs GT (5 cm mesh), 20 views at 504 px, K=16", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIGURES / "e23_pointmap_metric.jpg", dpi=110, pil_kwargs={"quality": 90})
    print("figure: experiments/figures/e23_pointmap_metric.jpg")


if __name__ == "__main__":
    main()
