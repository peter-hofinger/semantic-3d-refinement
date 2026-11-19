"""E57c: the figures of the issue-#76 report: every algorithm side by side, how the oracle decides, how it peels, and the fusion step by step.

All on the bedroom `ai_006_007` (the presentation scene), the cloud from view 18 pulled back 0.5 m (`scene_figures.render_cameras`); numbers are the dev means of
`results/e57c_summary_dev.txt`. Palette and layout: `e57c_style`.

uv run python -m experiments.scripts.e57c_figures_issue76 [--only algorithms anatomy peeling steps mechanism]
"""

import argparse
import re
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from matplotlib.colors import ListedColormap, PowerNorm, SymLogNorm
from matplotlib.lines import Line2D

from experiments.scripts.e57c_oracle_curve import window_mean
from experiments.scripts.e57c_style import (
    CATEGORY_COLORS,
    ERROR_MAP,
    GROUND,
    GROUP,
    GROUP_NAME,
    HAIR,
    INK,
    MUTED,
    NEED_MAP,
    NET_MAP,
    ORDER_MAP,
    STEP,
    Layout,
    apply_style,
    chips,
    cloud_image,
    to_rgb,
)
from experiments.scripts.e57c_summary import scene_rows
from experiments.scripts.e57c_two_fills import GOLD, GREEN, WHITE
from src.analysis.filters import keep_top, ranks
from src.analysis.scene_figures import (
    CEILING_CUT,
    PRESENTATION_SCENE,
    PRESENTATION_VIEWS,
    ground_truth_points,
    render_cameras,
)

OUT = Path("experiments/figures")
VIEW = 18
BASELINE = 3.500  # Déjà View's dev Chamfer
SECONDS = {  # optimized variants, s per scene (results/e57c_summary_dev.txt); accuracy oracle derived
    "confidence": 0.01,
    "density": 3.7,
    "greedy": 7.6,
    "marginal": 2.9,
    "accuracy": 1.9,
    "union": 7.6,
    "product": 9.4,
    "mean": 9.4,
    "fusion ids": 9.8,
    "fusion one": 9.8,
    "A ids": 12.5,
    "A one": 12.7,
}
ZOO = {
    "accuracy": (0.7, 2.479),
    "pixel": (0.8, 2.713),
    "smooth": (0.8, 2.769),
}  # q, dev Chamfer: results/e57c_oracle_zoo_dev.txt


def dev_table() -> dict[str, tuple[float, float]]:
    """Return (kept, Chamfer) per approach from the dev summary."""
    out = {}
    for line in (
        Path("experiments/results/e57c_summary_dev.txt")
        .read_text()
        .split("dev: mean over")[1]
        .splitlines()
    ):
        m = re.match(
            r"^  (\S.*?)\s{2,}(\d\.\d{3})\s+[\d.]+ / [\d.]+ / ([\d.]+)\s", line
        )
        if m:
            out[m.group(1).replace(" (best on this split)", "")] = (
                float(m.group(2)),
                float(m.group(3)),
            )
    return out


def greedy_curve() -> dict[float, tuple[float, float, float]]:
    """Return the greedy oracle's dev (accuracy, completeness, Chamfer) per q from the zoo run."""
    block = (
        Path("experiments/results/e57c_oracle_zoo_dev.txt")
        .read_text()
        .split("dev: mean over 3 scenes")[1]
    )
    part = block.split("  greedy: accuracy / completeness / Chamfer (cm)")[1].split(
        "  greedy-own"
    )[0]
    return {
        float(q): (float(a), float(c), float(h))
        for q, a, c, h in re.findall(
            r"q ([\d.]+): ([\d.]+) / ([\d.]+) / ([\d.]+)", part
        )
    }


def setup() -> dict:
    """Compute the bedroom's state once."""
    raw = yaml.safe_load(Path("experiments/configs/e57c_oracle.yaml").read_text())
    st: dict = {}
    scene_rows(raw, PRESENTATION_SCENE, state=st, score=False)
    inputs, pixels = st["inputs"], st["inputs"].truth.pixels

    def full(kept: np.ndarray) -> np.ndarray:
        mask = np.ones(pixels.shape, dtype=bool)  # a pixel without GT is kept
        mask[pixels] = kept
        return mask

    st["full"] = full
    a, pred, truth = st["a"], st["pred"], inputs.truth
    e = 100 * np.linalg.norm(pred - truth.points, axis=1)
    masks = st["masks"]
    masks["accuracy"] = full(keep_top(ranks(-a), ZOO["accuracy"][0]))
    masks["pixel"] = full(keep_top(ranks(-e), ZOO["pixel"][0]))
    masks["smooth"] = full(
        keep_top(ranks(-window_mean(a, pixels, 7)), ZOO["smooth"][0])
    )
    st["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    st["camera"], st["eye"] = render_cameras(inputs, PRESENTATION_VIEWS)[
        "view 18, pulled back"
    ]
    return st


def algorithms(st: dict, camera_key: str, out: str) -> None:
    """Every approach as a cloud, in groups, with its dev Chamfer on a common bar."""
    inputs, aligned, masks, device = (
        st["inputs"],
        st["aligned"],
        st["masks"],
        st["device"],
    )
    camera = render_cameras(inputs, PRESENTATION_VIEWS)[camera_key][0]
    ceiling = None
    if camera_key == "dollhouse":
        ceiling = np.percentile(inputs.truth.points[:, 2], 98) - CEILING_CUT
    table = dev_table()
    gt_points = ground_truth_points(inputs)
    spec = [  # key, name, sub-note, group, dev (kept, Chamfer), seconds
        (
            "gt",
            "Ground truth",
            "GT depth of the same 20 views",
            "reference",
            None,
            None,
        ),
        (
            "Déjà View",
            "Déjà View, whole",
            "the backbone's cloud, 1 point per pixel",
            "reference",
            table["Déjà View"],
            None,
        ),
        (
            "confidence only q 0.5",
            "Confidence only",
            "keep the more confident half",
            "reference",
            table["confidence only q 0.5"],
            SECONDS["confidence"],
        ),
        (
            "density q 0.6",
            "Instance voxel density",
            "#27, the best filter without the GT",
            "reference",
            table["density q 0.6"],
            SECONDS["density"],
        ),
        (
            "greedy oracle q 0.5",
            "Greedy oracle",
            "reverse greedy on the exact Chamfer gain",
            "oracle",
            table["greedy oracle q 0.5"],
            SECONDS["greedy"],
        ),
        (
            "marginal oracle q 0.6 (one round)",
            "Marginal oracle",
            "the same gain, one round",
            "oracle",
            table["marginal oracle q 0.6 (one round)"],
            SECONDS["marginal"],
        ),
        (
            "accuracy",
            "Accuracy oracle",
            "rank by the true error",
            "oracle",
            (ZOO["accuracy"][0], ZOO["accuracy"][1]),
            SECONDS["accuracy"],
        ),
        (
            "pixel",
            "Same-pixel error oracle",
            "what the confidence loss targets",
            "oracle",
            (ZOO["pixel"][0], ZOO["pixel"][1]),
            None,
        ),
        (
            "smooth",
            "Smoothed-accuracy oracle",
            "error averaged over 7 x 7 pixels",
            "oracle",
            (ZOO["smooth"][0], ZOO["smooth"][1]),
            None,
        ),
        (
            "union, no gate",
            "Union",
            "confident or kept by the oracle",
            "fusion",
            table["union, no gate"],
            SECONDS["union"],
        ),
        (
            "product q 0.3",
            "Product of two gradients",
            "soft confidence x oracle, best q",
            "fusion",
            table["product q 0.3"],
            SECONDS["product"],
        ),
        (
            "mean q 0.2",
            "Mean of two gradients",
            "soft confidence + oracle, best q",
            "fusion",
            table["mean q 0.2"],
            SECONDS["mean"],
        ),
        (
            "fusion, no seeds, ids",
            "Fusion, no seeds",
            "8 cm gate + straight runs, SAM 3 ids",
            "fusion",
            table["fusion, no seeds, ids"],
            SECONDS["fusion ids"],
        ),
        (
            "fusion, no seeds, one instance",
            "Fusion, no seeds",
            "same, no instance information",
            "fusion",
            table["fusion, no seeds, one instance"],
            SECONDS["fusion one"],
        ),
        (
            "A: seeds, ids",
            "Fusion A with seeds",
            "reprojected seeds, SAM 3 ids",
            "fusion",
            table["A: seeds, ids"],
            SECONDS["A ids"],
        ),
        (
            "A: seeds, one instance",
            "Fusion A with seeds",
            "reprojected seeds, no instance information",
            "fusion",
            table["A: seeds, one instance"],
            SECONDS["A one"],
        ),
    ]
    cols, margin, gap, width = 4, 0.45, 0.22, 17.0
    pw = (width - 2 * margin - (cols - 1) * gap) / cols
    ih = pw * 3 / 4
    title_h, bar_h, row_h = 0.78, 0.07, 0.78 + pw * 3 / 4 + 0.07 + 0.30
    header, footer = 1.45, 0.6
    layout = Layout(width, header + 4 * row_h + footer)
    layout.text(
        margin, 0.28, "Which points should we keep?", fontsize=27, fontweight="bold"
    )
    layout.text(
        margin,
        0.88,
        "Fifteen ways to pick a subset of Déjà View's cloud. Number: Chamfer, mean over the 3 dev scenes (lower is better; Déjà View whole = 3.500 cm). Cloud: bedroom ai_006_007.",
        fontsize=11,
        color=MUTED,
    )
    for k, group in enumerate(("reference", "oracle", "fusion")):
        chips(
            layout, margin + k * 3.6, 1.17, [(GROUP[group], GROUP_NAME[group])], size=10
        )
    for i, (key, name, note, group, dev, seconds) in enumerate(spec):
        r, c = divmod(i, cols)
        x0, y0 = margin + c * (pw + gap), header + r * row_h
        if key == "gt":
            image = cloud_image(
                inputs, gt_points, inputs.gt.depth > 0, camera, device, ceiling=ceiling
            )
        else:
            image = cloud_image(
                inputs, aligned, masks[key], camera, device, ceiling=ceiling
            )
        layout.fig.patches.append(
            plt.Rectangle(
                (x0 / layout.width, 1 - (y0 + 0.66) / layout.height),
                0.055 / layout.width,
                0.66 / layout.height,
                transform=layout.fig.transFigure,
                facecolor=GROUP[group],
                edgecolor="none",
            )
        )
        layout.text(x0 + 0.15, y0 - 0.02, name, fontsize=12, fontweight="bold")
        sub = ""
        if key == "Déjà View":
            sub = "3.500 cm  ·  kept 1.00"
        elif dev is not None:
            kept, chamfer = dev
            sub = (
                f"{chamfer:.3f} cm  ·  {100 * (chamfer - BASELINE) / BASELINE:+.0f} %  ·  kept {kept:.2f}".replace(
                    "-", "\u2212"
                )
                + (f"  ·  {seconds:g} s" if seconds else "")
            )
        layout.text(
            x0 + 0.15,
            y0 + 0.215,
            sub or note,
            fontsize=10,
            color=INK if sub else MUTED,
            fontweight="bold" if sub else "normal",
        )
        if sub:
            layout.text(x0 + 0.15, y0 + 0.43, note, fontsize=9, color=MUTED)
        ax = layout.place(x0, y0 + title_h, pw, ih)
        ax.imshow(image)
        if dev is not None:
            track = layout.place(x0, y0 + title_h + ih + 0.08, pw, bar_h)
            track.add_patch(
                plt.Rectangle((0, 0), 1, 1, transform=track.transAxes, color=HAIR)
            )
            track.add_patch(
                plt.Rectangle(
                    (0, 0),
                    min(dev[1], BASELINE) / BASELINE,
                    1,
                    transform=track.transAxes,
                    color=GROUP[group],
                )
            )
        if key != "gt" and dev is not None and dev[1] > BASELINE:
            layout.text(
                x0 + pw - 0.95,
                y0 + title_h + ih + 0.2,
                "bar cut at 3.5 cm",
                fontsize=7.5,
                color=MUTED,
            )
    layout.text(
        margin,
        header + 4 * row_h + 0.1,
        "Bar = Chamfer on a 0-3.5 cm scale (full bar = Déjà View whole). Clouds: all 20 views, one point per pixel, aligned onto the GT, seen "
        + (
            "from above, ceiling left out"
            if camera_key == "dollhouse"
            else "from view 18 pulled back 0.5 m"
        )
        + "; gray = nothing drawn. Seconds: optimized variant, CPU, per scene, without loading and scoring.",
        fontsize=8.8,
        color=MUTED,
    )
    layout.save(str(OUT / out), dpi=105)


def view_maps(st: dict) -> dict[str, np.ndarray]:
    """Return view 18's maps of the oracle's quantities."""
    pixels = st["inputs"].truth.pixels

    def scatter(values: np.ndarray) -> np.ndarray:
        out = np.full(pixels.shape, np.nan)
        out[pixels] = values
        return out[VIEW]

    a, m = st["a"], st["m"]
    return {
        "error": scatter(a - a.mean()),
        "need": scatter(m),
        "net": scatter((a - a.mean()) - m),
        "order": scatter(st["v_oracle"]),
        "keep": scatter(st["oracle"].astype(float)),
    }


def anatomy(st: dict, out: str) -> None:
    """How the oracle decides: the error, the coverage a point provides, their difference, the greedy order and the kept set, in view 18."""
    inputs = st["inputs"]
    maps = view_maps(st)
    cols, margin, gap, width = 3, 0.45, 0.3, 17.4
    pw = (width - 2 * margin - (cols - 1) * gap) / cols
    ih = pw * 3 / 4
    row_h = 0.95 + ih + 0.55
    header = 1.5
    layout = Layout(width, header + 2 * row_h + 0.4)
    layout.text(
        margin,
        0.28,
        "How the oracle decides, pixel by pixel",
        fontsize=27,
        fontweight="bold",
    )
    layout.text(
        margin,
        0.88,
        "Dropping a point lowers the accuracy term by how much worse than average it is, and raises the completeness term by the GT coverage that only it provides. Keep it if the second is larger.",
        fontsize=11,
        color=MUTED,
    )
    layout.text(
        margin,
        1.14,
        "Bedroom ai_006_007, view 18, all 20 views in the cloud; first round of the greedy (every point still present); cm.",
        fontsize=10,
        color=MUTED,
    )
    ground = to_rgb(GROUND)
    panels = [
        (
            "1  The view",
            "the 504 x 378 pixels, one 3D point each",
            None,
            None,
            None,
            None,
        ),
        (
            "2  Error",
            "how much farther from the GT than the average point",
            maps["error"],
            ERROR_MAP,
            PowerNorm(0.6, 0, 12),
            "cm above the mean error",
        ),
        (
            "3  Coverage it provides",
            "GT that would lose its nearest point (sum of d2 - d1)",
            maps["need"],
            NEED_MAP,
            PowerNorm(0.5, 0, 4),
            "cm of added completeness cost",
        ),
        (
            "4  Net change if dropped",
            "error minus coverage: dark = dropping helps, teal = keep",
            maps["net"],
            NET_MAP,
            SymLogNorm(1, vmin=-6, vmax=12),
            "cm; > 0: drop",
        ),
        (
            "5  The greedy order",
            "the q below which a pixel is dropped; dark teal survives longest",
            maps["order"],
            ORDER_MAP,
            mpl.colors.Normalize(0, 1),
            "q (kept fraction) at which it is dropped",
        ),
        (
            "6  Kept at q = 0.5",
            f"{100 * st['oracle'].mean():.0f} % of the pixels, Chamfer 2.048 cm (dev)",
            maps["keep"],
            ListedColormap([STEP["dropped"], STEP["oracle"]]),
            mpl.colors.Normalize(0, 1),
            None,
        ),
    ]
    for i, (name, note, data, cmap, norm, label) in enumerate(panels):
        r, c = divmod(i, cols)
        x0, y0 = margin + c * (pw + gap), header + r * row_h
        layout.text(x0, y0, name, fontsize=13.5, fontweight="bold")
        layout.text(x0, y0 + 0.33, note, fontsize=10, color=MUTED)
        ax = layout.place(x0, y0 + 0.75, pw, ih)
        if data is None:
            ax.imshow(inputs.views.images[VIEW] / 255)
        else:
            cm = cmap.copy() if hasattr(cmap, "copy") else cmap
            cm.set_bad(ground)
            image = ax.imshow(data, cmap=cm, norm=norm, interpolation="nearest")
            if label is not None:
                cax = layout.place(x0 + 0.1 * pw, y0 + 0.75 + ih + 0.12, 0.8 * pw, 0.1)
                bar = layout.fig.colorbar(image, cax=cax, orientation="horizontal")
                bar.outline.set_visible(False)
                bar.ax.tick_params(labelsize=8, length=2, color=HAIR)
                if isinstance(norm, SymLogNorm):
                    bar.set_ticks([-5, -1, 0, 1, 5, 10])
                    bar.set_ticklabels(["-5", "-1", "0", "1", "5", "10"])
                bar.set_label(label, fontsize=8.5, color=MUTED, labelpad=2)
    layout.save(str(OUT / out), dpi=105)


def peeling(st: dict, out: str) -> None:
    """Draw the greedy peeling: clouds at six kept fractions above the Chamfer curve (dev)."""
    inputs, aligned, path, device, camera = (
        st["inputs"],
        st["aligned"],
        st["path"],
        st["device"],
        st["camera"],
    )
    curve = greedy_curve()
    qs = [1.0, 0.8, 0.6, 0.5, 0.3, 0.1]
    width, margin, gap = 17.4, 0.45, 0.18
    tw = (width - 2 * margin - 5 * gap) / 6
    ih = tw * 3 / 4
    header = 1.5
    top = header + 0.45
    curve_y = top + ih + 0.75
    layout = Layout(width, curve_y + 4.5)
    layout.text(
        margin,
        0.28,
        "Peeling the cloud: one point at a time, cheapest first",
        fontsize=27,
        fontweight="bold",
    )
    layout.text(
        margin,
        0.88,
        "The greedy drops the points whose error is worth more than the coverage they provide. Accuracy keeps improving; completeness only starts to pay once the redundancy is gone.",
        fontsize=11,
        color=MUTED,
    )
    layout.text(
        margin,
        1.14,
        "Clouds: bedroom ai_006_007 from view 18 pulled back. Curves: dev mean over 3 scenes, cm.",
        fontsize=10,
        color=MUTED,
    )
    ax = layout.place(margin + 0.55, curve_y, width - 2 * margin - 0.9, 3.6)
    ax.spines["left"].set_visible(True)
    ax.spines["bottom"].set_visible(True)
    q_axis = sorted(curve)
    for idx, color, label in (
        (0, STEP["error"], "accuracy"),
        (1, STEP["oracle"], "completeness"),
        (2, INK, "Chamfer"),
    ):
        ax.plot(
            q_axis,
            [curve[q][idx] for q in q_axis],
            "-",
            color=color,
            lw=3.2 if idx == 2 else 1.8,
            solid_capstyle="round",
        )
        ax.text(
            0.085,
            curve[0.1][idx],
            f"  {label}",
            color=color,
            fontsize=11.5,
            fontweight="bold",
            va="center",
            ha="left",
        )
    ax.axhline(BASELINE, color=MUTED, lw=1, ls=(0, (4, 3)))
    ax.text(
        0.9,
        BASELINE + 0.1,
        "Déjà View whole: 3.500 cm",
        color=MUTED,
        fontsize=9.5,
        ha="left",
    )
    best = min(curve, key=lambda q: curve[q][2])
    ax.scatter([best], [curve[best][2]], s=95, color=INK, zorder=5)
    ax.annotate(
        f"minimum {curve[best][2]:.3f} cm\nat q = {best} ({100 * (curve[best][2] - BASELINE) / BASELINE:+.0f} %)",
        (best, curve[best][2]),
        (best, curve[best][2] - 0.55),
        fontsize=11,
        fontweight="bold",
        ha="center",
        arrowprops={"arrowstyle": "-", "color": INK, "lw": 1},
        bbox={
            "boxstyle": "round,pad=0.25",
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.92,
        },
        zorder=7,
    )
    ax.set_xlim(1.06, -0.12)
    ax.set_ylim(0.8, 5.0)
    ax.set_xticks([1.0, 0.8, 0.6, 0.4, 0.2])
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_xlabel(
        "kept fraction q  (the greedy removes pixels from left to right)", fontsize=10.5
    )
    ax.set_ylabel("cm", fontsize=10.5)
    ax.grid(axis="y", color=HAIR, lw=0.6)
    ax.set_axisbelow(True)
    inv = layout.fig.transFigure.inverted()
    for i, q in enumerate(qs):
        x0 = margin + i * (tw + gap)
        keep = (
            st["full"](np.ones(len(st["a"]), dtype=bool))
            if q == 1.0
            else st["full"](path[q])
        )
        layout.text(x0, header, f"q = {q}", fontsize=13, fontweight="bold")
        chamfer = curve[q][2]
        layout.text(x0, header + 0.25, f"{chamfer:.3f} cm", fontsize=10, color=MUTED)
        ax_t = layout.place(x0, top, tw, ih)
        ax_t.imshow(cloud_image(inputs, aligned, keep, camera, device))
        ax.scatter([q], [chamfer], s=45, color=INK, zorder=4)
        target = inv.transform(ax.transData.transform((q, chamfer)))
        start = ((x0 + tw / 2) / layout.width, 1 - (top + ih) / layout.height)
        layout.fig.add_artist(
            Line2D(
                [start[0], target[0]],
                [start[1], target[1]],
                color=HAIR,
                lw=1.2,
                zorder=6,
                transform=layout.fig.transFigure,
            )
        )
    layout.save(str(OUT / out), dpi=105)


def steps(st: dict, out: str) -> None:
    """Fusion A with seeds and no instance information, as four clouds and a zoom on the seam."""
    inputs, aligned, masks, device, camera = (
        st["inputs"],
        st["aligned"],
        st["masks"],
        st["device"],
        st["camera"],
    )
    category = st["category"]
    table = dev_table()
    kept_gate = np.isin(category, [WHITE, GREEN])
    stages = [
        (
            "1",
            "Déjà View",
            "every pixel of the 20 views",
            masks["Déjà View"],
            f"{table['Déjà View'][1]:.3f} cm · kept 1.00",
        ),
        (
            "2",
            "Threshold by confidence",
            "keep the more confident half",
            masks["confidence only q 0.5"],
            f"{table['confidence only q 0.5'][1]:.1f} cm · kept 0.50",
        ),
        (
            "3",
            "Add what the oracle needs",
            "oracle-kept pixels farther than 8 cm from what is kept",
            kept_gate,
            f"kept {kept_gate.mean():.2f} · a seam is left",
        ),
        (
            "4",
            "Bridge the seam",
            "reprojected seeds + straight runs, no instance ids",
            masks["A: seeds, one instance"],
            f"{table['A: seeds, one instance'][1]:.3f} cm · kept {table['A: seeds, one instance'][0]:.2f}",
        ),
    ]
    margin, gap, width = 0.45, 0.25, 17.4
    pw = (width - 2 * margin - 3 * gap) / 4
    ih = pw * 3 / 4
    crop = (slice(20, 240), slice(380, 560))
    zh = 3.7
    zw = zh * 180 / 220
    header = 1.5
    layout = Layout(width, header + 0.82 + ih + 0.55 + zh + 0.3)
    layout.text(margin, 0.28, "Fusion A, step by step", fontsize=27, fontweight="bold")
    layout.text(
        margin,
        0.88,
        "The confidence cut removes whole surfaces; the oracle's pixels bring them back; the gate between them leaves a seam; seeds reprojected from other views let a straight run of pixels close it.",
        fontsize=11,
        color=MUTED,
    )
    layout.text(
        margin,
        1.14,
        "Bedroom ai_006_007 from view 18 pulled back; numbers: dev means (3 scenes), Chamfer lower is better.",
        fontsize=10,
        color=MUTED,
    )
    for i, (num, name, note, mask, sub) in enumerate(stages):
        x0 = margin + i * (pw + gap)
        y0 = header
        layout.text(
            x0,
            y0 - 0.02,
            f"{num}",
            fontsize=22,
            fontweight="bold",
            color=STEP["bridged"] if num == "4" else STEP["confidence"],
        )
        layout.text(x0 + 0.42, y0 - 0.02, name, fontsize=13.5, fontweight="bold")
        layout.text(x0 + 0.42, y0 + 0.31, note, fontsize=9.6, color=MUTED)
        layout.text(x0 + 0.42, y0 + 0.52, sub, fontsize=10, color=INK)
        image = cloud_image(inputs, aligned, mask, camera, device)
        ax = layout.place(x0, y0 + 0.82, pw, ih)
        ax.imshow(image)
        ax.add_patch(
            plt.Rectangle(
                (crop[1].start, crop[0].start),
                crop[1].stop - crop[1].start,
                crop[0].stop - crop[0].start,
                fill=False,
                edgecolor=STEP["seed"],
                lw=1.6,
            )
        )
        zoom = layout.place(x0 + (pw - zw) / 2, y0 + 0.82 + ih + 0.55, zw, zh)
        zoom.imshow(image[crop])
        for spine in zoom.spines.values():
            spine.set_visible(True)
            spine.set_color(STEP["seed"])
            spine.set_linewidth(1.6)
    layout.text(
        margin,
        header + 0.82 + ih + 0.2,
        "Zoom on the wall right of the door (violet frame):",
        fontsize=10,
        color=STEP["seed"],
        fontweight="bold",
    )
    layout.save(str(OUT / out), dpi=105)


def mechanism(st: dict, out: str) -> None:
    """Draw the bridge in image space: view 12 before, with the seeds, and after."""
    category, seeds, back, back_seeds = (
        st["category"],
        st["seeds"]["single"],
        st["back"]["single"],
        st["back_seeds"]["single"],
    )
    view = 12
    palette = np.array([to_rgb(c) for c in CATEGORY_COLORS])
    before = palette[np.where(back[view], GOLD, category[view])]
    final = palette[np.where(back_seeds[view], GOLD, category[view])]
    dim = 0.45 * before + 0.55 * 1.0
    with_seeds = np.where(seeds[view][..., None], to_rgb(STEP["seed"]), dim)
    r0, r1, c0, c1 = (
        30,
        210,
        30,
        210,
    )  # around the strip's source pixels of view 12 (e57c_view_maps)
    margin, gap, width = 0.45, 0.3, 17.4
    pw = (width - 2 * margin - 2 * gap) / 3
    ih = pw * 3 / 4
    zh = 4.2
    zw = zh * (c1 - c0) / (r1 - r0)
    header = 1.7
    layout = Layout(width, header + 0.8 + ih + 0.5 + zh + 0.5)
    layout.text(
        margin,
        0.28,
        "Why the seam closes: seeds from other views",
        fontsize=27,
        fontweight="bold",
    )
    layout.text(
        margin,
        0.88,
        "Seam pixels are ones the oracle wants but the 8 cm gate held back. In a source view they sit between dropped and oracle-added pixels, with no kept pixel next to them; the kept patch that flanks them lives in another view.",
        fontsize=11,
        color=MUTED,
    )
    layout.text(
        margin,
        1.14,
        "Bedroom ai_006_007, view 12 in image space. A kept point of another view, reprojected onto a pixel that was dropped, becomes a seed (one pixel per point, no radius).",
        fontsize=10,
        color=MUTED,
    )
    chips(
        layout,
        margin,
        1.45,
        [
            (STEP["confidence"], "kept by confidence"),
            (STEP["oracle"], "added by the oracle"),
            (STEP["seam"], "seam: oracle-kept, held back by the gate"),
            (STEP["bridged"], "bridged"),
            (STEP["seed"], "seed"),
            (STEP["dropped"], "dropped"),
        ],
        gap=0.35,
        size=10,
    )
    titles = [
        ("Confidence cut and 8 cm gate", "the seam (rose) has no kept pixel beside it"),
        ("Seeds from the other views", "kept points reprojected onto dropped pixels"),
        (
            "Straight runs bridge the seam",
            "seed (or kept) - seam pixels - oracle-added pixel",
        ),
    ]
    for i, ((name, note), image) in enumerate(
        zip(titles, (before, with_seeds, final), strict=True)
    ):
        x0 = margin + i * (pw + gap)
        layout.text(x0, header, f"{i + 1}  {name}", fontsize=13.5, fontweight="bold")
        layout.text(x0, header + 0.33, note, fontsize=10, color=MUTED)
        ax = layout.place(x0, header + 0.8, pw, ih)
        ax.imshow(image, interpolation="nearest")
        ax.add_patch(
            plt.Rectangle((c0, r0), c1 - c0, r1 - r0, fill=False, edgecolor=INK, lw=1.4)
        )
        zoom = layout.place(x0 + (pw - zw) / 2, header + 0.8 + ih + 0.5, zw, zh)
        zoom.imshow(image[r0:r1, c0:c1], interpolation="nearest")
        for spine in zoom.spines.values():
            spine.set_visible(True)
            spine.set_color(INK)
    layout.save(str(OUT / out), dpi=105)


def main() -> None:
    """Draw the requested figures."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--only",
        nargs="*",
        default=["algorithms", "anatomy", "peeling", "steps", "mechanism"],
    )
    args = parser.parse_args()
    apply_style()
    st = setup()
    if "algorithms" in args.only:
        algorithms(st, "view 18, pulled back", "e57c_issue76_algorithms.jpg")
        algorithms(st, "dollhouse", "e57c_issue76_algorithms_dollhouse.jpg")
    if "anatomy" in args.only:
        anatomy(st, "e57c_issue76_oracle_anatomy.jpg")
    if "peeling" in args.only:
        peeling(st, "e57c_issue76_oracle_peeling.jpg")
    if "steps" in args.only:
        steps(st, "e57c_issue76_fusion_steps.jpg")
    if "mechanism" in args.only:
        mechanism(st, "e57c_issue76_fusion_mechanism.jpg")


if __name__ == "__main__":
    main()
