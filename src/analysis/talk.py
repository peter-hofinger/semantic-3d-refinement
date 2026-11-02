"""Draw two class-talk slides: DSINE + BiNI on one ScanNet++ view, and the joint refinement of all views.

Both are drawn from finished runs, nothing is re-optimized.

DSINE + BiNI: from a BiNI run (`normals.py`), nothing is re-optimized: BiNI integrates DSINE's normals into depth,
anchored to Déjà View's depth, and nothing else touches it. Ground truth is ScanNet++'s laser-scan mesh rendered
into the same view (z-depth, 0 where the ray misses), which the run did not see. Normals are compared in the camera
frame: Déjà View's lifted through its pinhole fit and rotated by its camera, the scan's through ScanNet++'s intrinsics.

Joint refinement: the ETH3D kicker wall from above before and after E_d + E_mv (`multiview.py`), and its runtime
against L-BFGS, both jointly and per view (`evaluate.py`), on office, where all of them ran on the same 20 views.

`uv run python -m src.analysis.talk --bini runs/<bini run> --gt <gt views npz> --eval runs/<eval run>
--multiview runs/<multiview run>` writes `figures/talk-<time>/` with both slides as PNG, `slides.pdf` and a README.
"""

import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from matplotlib.patches import Rectangle
from PIL import Image

from src.analysis.artifacts import SLIDE
from src.analysis.visualize import (
    BLUE,
    GRID,
    INK,
    INK_2,
    MV_VIEWS,
    ORANGE,
    SURFACE,
    camera_rays,
    fused_clouds,
    up_frame,
)
from src.refinement.inputs import git_hash, view_from_config
from src.refinement.normals import normals_from_depth

NOTES = """Our energy has a normal term, so first: how much can a normal estimator add on its own? DSINE predicts
sharp per-pixel normals; BiNI integrates them into a depth map held near Déjà View's, and nothing else runs. On this
ScanNet++ kitchen the result is visibly crisper than Déjà View: flat faces, clean edges, no texture noise. Against
the laser scan its normals are clearly closer, even closer than DSINE's own. But the depth is no more accurate: the
normals fix orientation, not offsets. Normals buy visual fidelity; accuracy has to come from elsewhere."""
SOURCE = (
    "ScanNet++ 5ee7c22ba0, view 13 (Déjà View K = 16, 20 views); ground truth: ScanNet++ laser-scan mesh rendered "
    "into the view. Angles: median over pixels with ground truth, away from its holes."
)
JOINT_NOTES = """The views disagree by about half a percent, so the fused cloud has every wall several times. We
minimize one energy over all 20 views, where neighboring views have to agree in depth. With L-BFGS that took a
minute. The energy is a sum of squares, so Gauss-Newton with conjugate gradients fits it, as it does BiNI: now 2.5
seconds for all 20 views, faster than L-BFGS refines a single view. Six copies of the wall become one line and the
surface gets 3 to 5 times thinner. The distance to the scan improves less, 8 to 10 percent: agreement removes what
the views disagree on, not the error they share."""
CROP = (96, 144)
# Joint L-BFGS on E_d + E_mv, 20 views of ETH3D office, 500 iterations (commit 8800b34); the run was replaced by
# Gauss-Newton before it was kept, so its time is quoted here rather than read from a run.
LBFGS_JOINT_SECONDS = 60.0
INPUT, AGREE, MONO = "E_d (= input)", "E_d + E_mv", "E_d + E_bd + E_n"
TIMES, MINUS = "×", "−"  # noqa: RUF001


def detail_window(n: np.ndarray, size: tuple[int, int]) -> tuple[slice, slice]:
    """Return the window of the given (rows, cols) with the largest summed normal gradient."""
    g = np.linalg.norm(np.diff(n, axis=0)[:, :-1], axis=-1) + np.linalg.norm(np.diff(n, axis=1)[:-1], axis=-1)
    c = np.cumsum(np.cumsum(np.pad(g, ((1, 0), (1, 0))), 0), 1)
    h, w = size
    sums = c[h:, w:] - c[:-h, w:] - c[h:, :-w] + c[:-h, :-w]
    r, col = np.unravel_index(sums.argmax(), sums.shape)
    return slice(r, r + h), slice(col, col + w)


def measure(bini_run: Path, gt_path: Path) -> dict:
    """Camera-frame normals of Déjà View, DSINE, BiNI and the scan, their angles to the scan, and depth errors."""
    raw = yaml.safe_load((bini_run / "run.yaml").read_text())["config"]
    view = raw["view"]
    data = view_from_config(raw)
    refined = dict(np.load(bini_run / "refined.npz"))
    rotation = np.load(raw["input"])["cam_to_world"][view, :3, :3].astype(np.float64)
    k = np.load(raw["gt_views"])["intrinsics"][view].astype(np.float64)
    gt = np.load(gt_path)["depth"][view].astype(np.float64)
    rays = data["rays"].double()

    def lifted(depth: np.ndarray) -> np.ndarray:
        return normals_from_depth(torch.as_tensor(depth).double(), rays).numpy() @ rotation

    normals = {
        "Déjà View": lifted(data["depth"].numpy()),
        "DSINE": refined["normals"].astype(np.float64) @ rotation,
        "BiNI": lifted(refined["depth"]),
        "scan": normals_from_depth(torch.from_numpy(np.where(gt > 0, gt, 1.0)), camera_rays(k, *gt.shape)).numpy(),
    }
    holes = np.lib.stride_tricks.sliding_window_view(np.pad(gt == 0, 2, constant_values=True), (5, 5)).any((-1, -2))
    angles = {name: float(np.median(np.degrees(np.arccos(np.clip((n * normals["scan"]).sum(-1), -1, 1)))[~holes]))
              for name, n in normals.items() if name != "scan"}  # fmt: skip
    errors = {}
    for name, depth in [("Déjà View", data["depth"].numpy().astype(np.float64)), ("BiNI", refined["depth"])]:
        d, g = depth[~holes], gt[~holes]
        errors[name] = 100 * float(np.mean(np.abs(np.log(np.median(g / d) * d / g))))
    image = np.load(raw["gt_views"])["images"][view]
    return {"normals": normals, "holes": holes, "angles": angles, "errors": errors, "image": image,
            "seconds": yaml.safe_load((bini_run / "run.yaml").read_text())["refine_seconds"], "raw": raw}  # fmt: skip


def bini_slide(out: Path, m: dict) -> None:
    """One slide: image and four normal maps, full and zoomed, with the angle to the scan and the depth error."""
    a, e = m["angles"], m["errors"]
    fig = plt.figure(figsize=SLIDE, facecolor=SURFACE)
    fig.text(0.03, 0.955, "DSINE + BiNI: sharper normals, not more accurate depth", fontsize=26, color=INK,
             weight="bold", va="top")  # fmt: skip
    fig.text(0.03, 0.875, "BiNI integrates DSINE's per-pixel normals into depth, held near Déjà View's; no other "
             "optimization.", fontsize=15, color=INK_2, va="top")  # fmt: skip
    rgb = {"scan": np.where(m["holes"][..., None], 0.9, m["normals"]["scan"] * 0.5 + 0.5)}
    rgb |= {name: n * 0.5 + 0.5 for name, n in m["normals"].items() if name != "scan"}
    win = detail_window(m["normals"]["DSINE"], CROP)
    panels = [
        ("image", m["image"] / 255.0, ""),
        ("Déjà View", rgb["Déjà View"], f"{a['Déjà View']:.1f}°"),
        ("DSINE", rgb["DSINE"], f"{a['DSINE']:.1f}°"),
        ("BiNI (DSINE → depth)", rgb["BiNI"], f"{a['BiNI']:.1f}°"),
        ("laser scan", rgb["scan"], "ground truth"),
    ]
    w, gap = 0.176, 0.0175
    for i, (name, img, note) in enumerate(panels):
        x = 0.03 + i * (w + gap)
        for y, h, window in [(0.5, 0.22, (slice(None), slice(None))), (0.2, 0.25, win)]:
            ax = fig.add_axes((x, y, w, h))
            ax.imshow(np.clip(img[window], 0, 1), interpolation="nearest")
            ax.set_axis_off()
            ax.set_anchor("N")
            if window is win:
                continue
            ax.set_title(name, fontsize=14, color=INK, pad=24)
            color = BLUE if name.startswith("BiNI") else INK_2
            ax.text(0.5, 1.03, note, transform=ax.transAxes, ha="center", va="bottom", fontsize=13, color=color,
                    weight="bold" if name.startswith("BiNI") else "normal")  # fmt: skip
            if i == 0:
                ax.add_patch(Rectangle((win[1].start, win[0].start), CROP[1], CROP[0], fill=False, color=ORANGE,
                                       lw=2))  # fmt: skip
    fig.text(0.5, 0.18, "zoom (orange box)   ·   ° = median angle to the scan's normals", fontsize=12, color=INK_2,
             ha="center", va="top")  # fmt: skip
    fig.text(0.03, 0.075, f"Normals: {a['Déjà View']:.1f}° → {a['BiNI']:.1f}° to the scan (DSINE alone "
             f"{a['DSINE']:.1f}°).   Depth: mean error {e['Déjà View']:.1f} % → {e['BiNI']:.1f} %, no gain.   "
             f"BiNI: {m['seconds']:.1f} s.", fontsize=14, color=INK, va="center")  # fmt: skip
    fig.text(0.03, 0.02, SOURCE, fontsize=9, color=INK_2)
    fig.savefig(out / "dsine_bini.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)


def slab(ax: plt.Axes, scan: np.ndarray, cloud: np.ndarray, c2w: np.ndarray, window: list) -> None:
    """Plot a 2 cm horizontal slab at camera height from above: scan gray, fused cloud blue."""
    frame = up_frame(c2w)
    height = (c2w[:, :3, 3] @ frame[2]).mean()
    for pts, color, size in [(scan, "#9a9994", 1.2), (cloud, BLUE, 1.0)]:
        q = pts @ frame.T
        q = q[np.abs(q[:, 2] - height) < 0.01]  # noqa: PLR2004
        ax.scatter(q[:, 0], q[:, 1], s=size, color=color, linewidths=0, rasterized=True)
    ax.set(xlim=window[0], ylim=window[1], aspect="equal", anchor="N")
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ax.spines.values():
        side.set_color(GRID)


def joint_slide(out: Path, ev: dict, mv: dict, mv_run: Path) -> None:
    """One slide: the kicker wall before and after joint refinement, and the solve's runtime against L-BFGS."""
    rows = {n: {v["label"]: v for v in sc["variants"]} for n, sc in mv["scenes"].items()}
    office = rows["office"]
    labels = [v["label"] for v in mv["scenes"]["kicker"]["variants"]]
    per_view = [next(v for v in r["variants"] if v["label"] == MONO)["seconds"] for r in ev["views"]
                if r["scene"] == "office"]  # fmt: skip
    fig = plt.figure(figsize=SLIDE, facecolor=SURFACE)
    fig.text(0.03, 0.955, f"20 views refined together in {office[AGREE]['seconds']:.1f} s: one wall instead of many",
             fontsize=26, color=INK, weight="bold", va="top")  # fmt: skip
    fig.text(0.03, 0.875, "One energy over all 20 depth maps: each stays near Déjà View's prediction, and overlapping "
             "views must agree.", fontsize=15, color=INK_2, va="top")  # fmt: skip

    paths = mv["config"]["scenes"]["kicker"]
    c2w = np.load(paths["gt_views"])["cam_to_world"].astype(np.float64)
    fused = fused_clouds(paths, mv["scenes"]["kicker"]["align"], dict(np.load(mv_run / "results.npz")), "kicker",
                         len(labels))  # fmt: skip
    scan, window = np.load(paths["scan"]).astype(np.float64), MV_VIEWS["kicker"]["slice_window"]
    for i, (label, text) in enumerate([(INPUT, "before: Déjà View"), (AGREE, f"after: {AGREE}")]):
        ax = fig.add_axes((0.03 + i * 0.235, 0.22, 0.225, 0.55))
        slab(ax, scan, fused[labels.index(label)], c2w, window)
        ax.set_title(text, fontsize=15, color=INK, weight="bold", pad=8)
        if i == 0:
            x0, y0 = window[0][0] + 0.06, window[1][0] + 0.06
            ax.plot([x0, x0 + 0.2], [y0, y0], color=INK, lw=3)
            ax.text(x0 + 0.1, y0 + 0.03, "20 cm", ha="center", va="bottom", fontsize=11, color=INK)
    fig.text(0.258, 0.355, "ETH3D kicker, a 2 cm horizontal slice seen from above:\ngray laser scan, blue all 20 "
             "depth maps fused", fontsize=11.5, color=INK_2, ha="center", va="top", linespacing=1.3)  # fmt: skip
    fig.text(0.03, 0.25, "Energy terms", fontsize=12, color=INK, weight="bold", va="top")
    fig.text(0.03, 0.215, "E_d: stay near Déjà View's depth    E_bd: bending (smoothness)\nE_n: follow DSINE's "
             "normals    E_mv: overlapping views agree in depth", fontsize=12, color=INK_2, va="top",
             linespacing=1.5)  # fmt: skip

    speedup = LBFGS_JOINT_SECONDS / office[AGREE]["seconds"]
    groups = [
        (6.35, "Same energy, two solvers"),
        (2.95, "With bending and normals: 20 views together vs one view alone"),
    ]
    bars = [
        (5.3, "L-BFGS  ·  20 views  ·  E_d + E_mv", LBFGS_JOINT_SECONDS, INK_2, ""),
        (4.1, "Gauss-Newton + CG  ·  20 views  ·  E_d + E_mv", office[AGREE]["seconds"], BLUE,
         f"   {speedup:.0f}{TIMES} faster"),
        (1.9, f"L-BFGS  ·  1 view  ·  {MONO}", float(np.mean(per_view)), INK_2, ""),
        (0.7, f"Gauss-Newton + CG  ·  20 views  ·  {MONO} + E_mv", office["E_d + E_bd + E_n + E_mv"]["seconds"],
         BLUE, ""),
    ]  # fmt: skip
    ax = fig.add_axes((0.53, 0.33, 0.44, 0.44))
    ax.barh([b[0] for b in bars], [b[2] for b in bars], height=0.6, color=[b[3] for b in bars], zorder=2)
    ax.grid(axis="y", visible=False)
    for yi, text, sec, color, note in bars:
        ax.text(sec + 1, yi, f"{sec:.1f} s{note}", va="center", fontsize=14, color=color, weight="bold")
        ax.text(0, yi + 0.36, text, va="bottom", fontsize=12, color=INK)
    for yi, text in groups:
        ax.text(0, yi, text, va="bottom", fontsize=12.5, color=INK_2, style="italic")
    ax.set(ylim=(0.2, 6.9), xlim=(0, LBFGS_JOINT_SECONDS * 1.18), yticks=[])
    ax.set_xlabel("seconds, including building the cross-view matches", fontsize=11, color=INK_2)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.set_title("Runtime (RTX 3080, ETH3D office, 336 x 504 px per view)", fontsize=13.5, color=INK,
                 weight="bold", loc="left", pad=6)  # fmt: skip

    thin = [rows[n][INPUT]["thickness_mm"] / rows[n][AGREE]["thickness_mm"] for n in rows]
    closer = [100 * (1 - rows[n][AGREE]["fused_median_cm"] / rows[n][INPUT]["fused_median_cm"]) for n in rows]
    agree = [rows[n][INPUT]["disagreement_pct"] / rows[n][AGREE]["disagreement_pct"] for n in rows]
    fig.text(0.53, 0.235, f"{AGREE} against Déjà View, ranges over office and kicker:", fontsize=12, color=INK,
             weight="bold", va="top")  # fmt: skip
    items = [
        (f"{min(agree):.0f}-{max(agree):.0f}{TIMES}", "better agreement\nbetween views", BLUE),
        (f"{min(thin):.0f}-{max(thin):.0f}{TIMES}", "thinner surfaces\nin the fused cloud", BLUE),
        (f"{min(closer):.0f}-{max(closer):.0f} %", "closer to the\nlaser scan, only", INK_2),
    ]  # fmt: skip
    for x, (number, label, color) in zip([0.61, 0.76, 0.9], items, strict=True):
        fig.text(x, 0.19, number, fontsize=22, color=color, weight="bold", ha="center", va="top")
        fig.text(x, 0.125, label, fontsize=11.5, color=INK_2, ha="center", va="top", linespacing=1.3)
    fig.text(0.03, 0.02, "Déjà View (K = 16) on 20 views each of ETH3D office and kicker; ground truth: ETH3D laser "
             "scans (CC BY-NC-SA 4.0). L-BFGS on 20 views: 500 iterations; Gauss-Newton: 10 steps of 20 CG "
             "iterations.", fontsize=9, color=INK_2)  # fmt: skip
    fig.savefig(out / "joint_solve.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    """Draw both slides from a BiNI run with its ground truth, an ETH3D evaluation run and a multi-view run."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--bini", type=Path, required=True)
    parser.add_argument("--gt", type=Path, required=True)
    parser.add_argument("--eval", type=Path, required=True)
    parser.add_argument("--multiview", type=Path, required=True)
    args = parser.parse_args()
    stamp, commit = f"talk-{time.strftime('%Y%m%d-%H%M%S')}", git_hash()
    out = Path("figures") / stamp
    out.mkdir(parents=True)
    plt.rcParams.update({"font.size": 12, "axes.facecolor": SURFACE})
    m = measure(args.bini, args.gt)
    bini_slide(out, m)
    ev = yaml.safe_load((args.eval / "run.yaml").read_text())
    mv = yaml.safe_load((args.multiview / "run.yaml").read_text())
    joint_slide(out, ev, mv, args.multiview)
    pages = [Image.open(out / f"{name}.png").convert("RGB") for name in ("dsine_bini", "joint_solve")]
    pages[0].save(out / "slides.pdf", save_all=True, append_images=pages[1:], resolution=150)
    runs = ", ".join(f"`{p.name}`" for p in (args.bini, args.eval, args.multiview))
    lines = [f"# {stamp}", "", f"git `{commit}`; from {runs}.", "",
             ("Images and ground truth: ScanNet++ (Yeshwanth et al., ICCV 2023), used under its terms of use, and "
              "ETH3D (Schöps et al., CVPR 2017), CC BY-NC-SA 4.0; the figures that show ETH3D are adapted from it "
              "under the same licence. The data itself is not in this repository."), "",
             "## DSINE + BiNI", "",
             "| | median angle to the scan's normals (°) | mean depth error (%) |", "|---|---|---|"]  # fmt: skip
    lines += [f"| {name} | {m['angles'][name]:.2f} | {m['errors'][name]:.2f} |" if name in m["errors"]
              else f"| {name} | {m['angles'][name]:.2f} | |" for name in m["angles"]]  # fmt: skip
    lines += [
        "",
        "![dsine_bini](dsine_bini.png)",
        "",
        NOTES,
        "",
        "## joint refinement",
        "",
        "![joint_solve](joint_solve.png)",
        "",
        JOINT_NOTES,
    ]
    (out / "README.md").write_text("\n".join([*lines, ""]))
    print(f"wrote {out}", m["angles"], m["errors"])  # noqa: T201


if __name__ == "__main__":
    main()
