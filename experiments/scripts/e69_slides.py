"""E69's slide-1 visuals in the TUM style: coverage, 3a against 3b, a backup, a trio.

Every number is read from E69f / E69g / E69i outputs already on disk; nothing is trained.
`rung0`: in-box unseen GT99v covered by the candidates, Reach-4 against two-level H6, per
scene on E69f's locked box (`e69f_lock.txt`). `rung3`: the bedroom rendered from one
dollhouse camera, rung 3a (the latent table, candidate-centred) and rung 3b (E2), each
point coloured by its distance to GT99v; E2's gate and child recall come from one forward
pass of the whole grid, as `generate` runs it. `backup`: GT99v with the in-box unseen points
8-13 cells from Déjà View in orange, next to the H6 oracle in the error colours.
`trio`: Déjà View, the H6 oracle with the 20 views' GT as targets and E2 side by side with
their names and dev means, under the regular evaluation (`cloud_eval`): every output point,
drawn at one splat size and coloured by its distance to the 20 views' GT pixels. The
images are the bedroom (clouds of `e69_regular`); the number under each name is the mean
Chamfer over the dev scenes (`e69f_eval_dev.yaml`).

uv run python -m experiments.scripts.e69_slides rung0|rung3|backup|trio [--light]
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
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import distance_transform_edt

from experiments.scripts.e69a_renders import prepare
from experiments.scripts.e69e_rung0 import CM, to_world
from experiments.scripts.e69f_rung0 import Loaded, load, oracle
from experiments.scripts.e69g import BEDROOM, CONFIG, masks
from experiments.scripts.e69i import load_model, occupancy, pooled
from src.analysis.render import Camera, headlight, intrinsics, look_at, splat
from src.evaluation.labeled import nearest_distances
from src.refinement import generator as gen
from src.refinement.generator import (
    child_targets,
    for_arm,
    load_rung,
    two_level,
)
from src.refinement.volume import voxelize

OUT = Path("experiments/figures")
RESULTS = Path("experiments/results")
RUNS = Path("runs/e69f")
BLUE, LIGHT, ORANGE = "#0065BD", "#64A0C8", "#E37222"
INK, DARK, GRAY, GRID, ZONE = "#222222", "#555555", "#B3B3B3", "#E8E8E8", "#F5F6F8"
MUTED = "#808080"  # the slide's footnote grey
MAGMA = plt.get_cmap("magma")(
    np.linspace(0, 0.88, 256)
)  # 0.88: the top stays off-white
ERROR = LinearSegmentedColormap.from_list("magma088", MAGMA)
SUFFIX = ""  # "_light" for the reversed scale
VMAX = 5.0  # cm, top of the shared error scale
PX = 1000  # render side, pixels, before the shared crop
GATE = 80.0  # % in-box unseen coverage, E69f's pass gate
LATENT_3A = "e69f-overfit1a_c-latent-gt99-s0"
E2 = "e69f-overfit1_e2-ovoxel-gt99-s0"

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans"],
        "pdf.fonttype": 42,
        "axes.edgecolor": INK,
        "savefig.facecolor": "#FFFFFF",
    }
)


def coverage() -> dict[str, tuple[float, float]]:
    """Return scene -> (Reach-4, H6) in-box unseen coverage (%) of E69f's lock."""
    text = (RESULTS / "e69f_lock.txt").read_text()
    rows = {}
    for rule in ("R4", "H6(coarse)"):
        line = next(
            x for x in text.splitlines() if x.startswith(rule + " ") and "ai_" in x
        )
        rows[rule] = dict(re.findall(r"(ai_\d{3}_\d{3}) ([\d.]+)/", line))
    return {k: (float(rows["R4"][k]), float(rows["H6(coarse)"][k])) for k in rows["R4"]}


def reach4_dv_gt20() -> tuple[float, float]:
    """Return the median in-box unseen coverage (%) at reach 4 from DV's and GT20's points.

    E69f's C2 (`e69f_diagnose.txt`, measured before the box was locked): candidates
    grown from the input views' GT depth cover no more than from Déjà View's.
    """
    pairs = re.findall(
        r"r4 +([\d.]+) / +([\d.]+)", (RESULTS / "e69f_diagnose.txt").read_text()
    )
    dv, gt20 = np.array(pairs, dtype=float).T
    assert len(dv) == 6
    return float(np.median(dv)), float(np.median(gt20))


def rung0() -> list[str]:
    """Draw the coverage dumbbell (5.4 x 4.6 in)."""
    data = sorted(coverage().items(), key=lambda kv: kv[1][1])
    r4m = float(np.median([v[0] for _, v in data]))
    h6m = float(np.median([v[1] for _, v in data]))
    passed = sum(v[1] >= GATE for _, v in data)
    fig = plt.figure(figsize=(5.4, 4.6))
    ax = fig.add_axes((0.155, 0.255, 0.68, 0.56))
    lo, hi = 40.0, 101.0
    ax.axvspan(lo, GATE, color=ZONE, zorder=0, lw=0)
    ax.axvline(GATE, color="#333333", lw=1.2, ls=(0, (4, 3)), zorder=1)
    ax.text(
        GATE - 0.8,
        7.55,
        "fail < 80 %",
        ha="right",
        va="center",
        fontsize=8.5,
        color="#666666",
    )
    ys = [*range(len(data) + 1, 1, -1), 0.0]
    rows = [(k, *v) for k, v in data] + [("Median", r4m, h6m)]
    for y, (_, r4, h6) in zip(ys, rows, strict=True):
        median = y == 0
        size = 95 if median else 60
        ax.annotate(
            "",
            xy=(h6 - 1.1, y),
            xytext=(r4 + 1.1, y),
            arrowprops={
                "arrowstyle": "-|>",
                "color": LIGHT,
                "lw": 2.2 if median else 1.6,
                "mutation_scale": 11,
                "shrinkA": 0,
                "shrinkB": 0,
            },
            zorder=2,
        )
        ax.scatter([r4], [y], s=size, color=GRAY, zorder=3, lw=0)
        ax.scatter([h6], [y], s=size, color=BLUE, zorder=3, lw=0)
        ax.text(
            r4 - 1.6,
            y,
            f"{r4:.1f}",
            ha="right",
            va="center",
            fontsize=10.5,
            fontweight="bold",
            color="#8C8C8C",
        )
        ax.text(
            1.03,
            y,
            f"+{h6 - r4:.1f}",
            transform=ax.get_yaxis_transform(),
            ha="left",
            va="center",
            fontsize=10.5,
            fontweight="bold",
            color=BLUE,
        )
        if median:
            ax.text(
                h6,
                y + 0.42,
                f"{h6:.1f}",
                ha="center",
                va="bottom",
                fontsize=10.5,
                fontweight="bold",
                color=BLUE,
            )
    ax.axhline(1.0, color="#CCCCCC", lw=0.8, zorder=1)
    ax.text(
        1.03,
        len(data) + 1.95,
        "Δ pp",
        transform=ax.get_yaxis_transform(),
        ha="left",
        va="center",
        fontsize=10.5,
        fontweight="bold",
        color=BLUE,
    )
    ax.text(
        1.03,
        -0.68,
        f"{passed}/6 pass",
        transform=ax.get_yaxis_transform(),
        ha="left",
        va="center",
        fontsize=9.5,
        fontweight="bold",
        color=BLUE,
    )
    ax.set_yticks(
        ys, [n.replace("ai_", "") if n.startswith("ai_") else n for n, *_ in rows]
    )
    for label in ax.get_yticklabels():
        label.set_fontsize(9.5)
        label.set_color("#555555")
    ax.get_yticklabels()[-1].set_fontweight("bold")
    ax.get_yticklabels()[-1].set_color(INK)
    ax.get_yticklabels()[-1].set_fontsize(10.5)
    ax.set_xlim(lo, hi)
    ax.set_ylim(-1.0, len(data) + 1.6)
    ax.set_xticks(range(40, 101, 10))
    ax.tick_params(axis="x", labelsize=9.5, colors="#555555", length=3)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_linewidth(1.2)
    ax.set_xlabel(
        "In-box unseen GT inside the candidates (%)", fontsize=10.5, color="#333333"
    )
    fig.text(
        0.04,
        0.955,
        "Rung 0: candidates reach the missing surface",
        fontsize=12,
        fontweight="bold",
        color=INK,
        va="center",
    )
    handles = [
        plt.Line2D(
            [], [], marker="o", ls="", color=GRAY, markersize=7, label="Reach-4"
        ),
        plt.Line2D(
            [], [], marker="o", ls="", color=BLUE, markersize=7, label="Two-level H6"
        ),
        plt.Line2D([], [], color="#333333", lw=1.2, ls=(0, (4, 3)), label="80 % gate"),
    ]
    fig.legend(
        handles=handles,
        loc="center left",
        bbox_to_anchor=(0.03, 0.885),
        ncol=3,
        frameon=False,
        fontsize=10.5,
        labelcolor="#333333",
        handletextpad=0.4,
        columnspacing=1.4,
    )
    dv, gt20 = reach4_dv_gt20()
    fig.text(
        0.04,
        0.035,
        f"GT depth ≈ DV at reach 4 ({gt20:.1f} vs {dv:.1f} %, median): the rule, "
        "not the backbone, is the ceiling",
        fontsize=8.5,
        color="#666666",
        va="center",
    )
    fig.text(
        0.04,
        0.105,
        "Bedroom: H6 oracle 0.39 cm  ·  DV 3.37 cm",
        fontsize=9.5,
        fontweight="bold",
        color=BLUE,
        ha="left",
        va="center",
        bbox={"boxstyle": "round,pad=0.35", "fc": "#FFFFFF", "ec": BLUE, "lw": 0.8},
    )
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"e69_slide_rung0.{ext}", dpi=300)
    return [
        f"rung0: Reach-4 median {r4m:.1f} -> H6 {h6m:.1f} %, {passed}/6 >= {GATE:.0f} %"
    ]


def dollhouse(s: Loaded) -> tuple[Camera, np.ndarray, float]:
    """Return a camera 72 degrees above the room on the input cameras' side, and its eye.

    Also the height above which points are cut (the ceiling), so the room is seen from above.
    """
    truth = s.truth
    floor, ceiling = np.percentile(truth[:, 2], [1, 99])
    middle = np.append(np.median(truth[:, :2], 0), floor + 0.4 * (ceiling - floor))
    centers = s.views.cam_to_world[:, :3, 3].astype(np.float64)
    side = centers.mean(0)[:2] - middle[:2]
    side /= np.linalg.norm(side)
    reach = np.percentile(np.linalg.norm(truth[:, :2] - middle[:2], axis=1), 98)
    fov, elevation = 40.0, np.radians(72)
    distance = 1.25 * reach / np.tan(np.radians(fov) / 2)
    eye = middle + distance * np.append(np.cos(elevation) * side, np.sin(elevation))
    camera = Camera(
        look_at(eye, middle, np.array([0.0, 0, 1])), intrinsics(PX, PX, fov), (PX, PX)
    )
    return camera, eye, ceiling - 0.2


def render(
    cloud: np.ndarray,
    color: np.ndarray,
    camera: Camera,
    eye: np.ndarray,
    top: float,
    radius: float | None = None,
) -> np.ndarray:
    """Splat a world cloud (N, 3) with colours (N, 3), lightly shaded; above `top` cut.

    radius (m): every point is drawn, at this one splat size; None caps the cloud and
    sizes each point by its neighbours' distance, which draws a stray point several
    times larger than a surface point.
    """
    keep = cloud[:, 2] < top
    c = prepare(
        cloud[keep],
        color[keep],
        np.zeros(int(keep.sum()), dtype=np.int64),
        np.random.default_rng(0),
        **({} if radius is None else {"cap": int(keep.sum())}),
    )
    xyz = torch.from_numpy(c["points"]).float()
    shade = headlight(torch.from_numpy(c["normals"]).float(), xyz, eye)
    rgb = torch.from_numpy(c["rgb"]).float() * (0.55 + 0.45 * shade / 0.95)
    size = torch.from_numpy(c["radius"]).float()
    if radius is not None:
        size = torch.full_like(size, radius)
    return splat(xyz, rgb.clamp(0, 1), size, camera)


def error_colors(truth: np.ndarray, cloud: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return each point's distance to the GT `truth` (cm) and its shared-scale colour."""
    d = CM * nearest_distances(truth, cloud)
    return d, ERROR(np.clip(d / VMAX, 0, 1))[:, :3]


def crop_box(images: list[np.ndarray], pad: int = 12) -> tuple[slice, slice]:
    """Return one square crop around everything drawn in any of the images."""
    drawn = np.any([(im < 0.999).any(2) for im in images], 0)
    rows, cols = np.nonzero(drawn)
    r0, r1, c0, c1 = rows.min(), rows.max(), cols.min(), cols.max()
    side = max(r1 - r0, c1 - c0) + 2 * pad
    rc, cc = (r0 + r1) // 2, (c0 + c1) // 2
    r = int(np.clip(rc - side // 2, 0, PX - side))
    c = int(np.clip(cc - side // 2, 0, PX - side))
    return slice(r, r + side), slice(c, c + side)


def cloud_of(s: Loaded, name: str) -> np.ndarray:
    """Return a model's saved bedroom cloud in the world frame."""
    folder = Path("data/processed/hypersim") / s.name
    pts = np.load(folder / f"{name}_views_n20_uniform_504_cloud.npz")["points"]
    return to_world(pts.astype(np.float64), s.similarity)


def stages(s: Loaded) -> tuple[float, float]:
    """Return E2's gate recall (GT-holding coarse candidates it opens) and child recall."""
    cfg, model, device = load_model(RUNS / "overfit1_e2" / f"{E2}.pt")
    pts = gen.load_scene(cfg, BEDROOM, cfg.semantics)[0]
    grid = voxelize(pts, semantics=model.semantics).to(device)
    with torch.no_grad():
        out, candidates, _, wide = two_level(model, grid, cfg.coarse_reach, None)
    held = pooled(occupancy(s)).to(device)
    opened = pooled(candidates)
    gate = float((opened & held & wide).sum() / (held & wide).sum())
    voxel = torch.nonzero(candidates).flip(1).cpu()
    occ, _ = child_targets(
        voxel, torch.from_numpy(s.truth_cells).float(), cfg.split, s.dims
    )
    child = float(((out.logit.cpu() > 0) & occ).sum() / occ.sum())
    return gate, child


def colorbar(fig: plt.Figure, rect: tuple[float, float, float, float]) -> None:
    """Draw the shared horizontal error colour bar in `rect` (figure fractions)."""
    cax = fig.add_axes(rect)
    bar = fig.colorbar(
        mpl.cm.ScalarMappable(mpl.colors.Normalize(0, VMAX), ERROR),
        cax=cax,
        orientation="horizontal",
        extend="max",
        extendfrac=0.04,
    )
    bar.outline.set_linewidth(0.6)
    bar.outline.set_edgecolor("#999999")
    bar.set_ticks(range(int(VMAX) + 1))
    cax.tick_params(labelsize=9.5, colors="#555555", length=2.5, width=0.6)
    bar.set_label(
        "Distance to ground truth (cm)", fontsize=10.5, color="#333333", labelpad=3
    )


def rung3(s: Loaded) -> list[str]:
    """Draw rung 3a against rung 3b on one camera and one error scale (6.4 x 4.6 in)."""
    camera, eye, top = dollhouse(s)
    gate, child = stages(s)
    shots, means = [], []
    for name in (LATENT_3A, E2):
        cloud = cloud_of(s, name)
        d, color = error_colors(s.truth, cloud)
        means.append(d.mean())
        shots.append(render(cloud, color, camera, eye, top))
    box = crop_box(shots)
    fig = plt.figure(figsize=(6.4, 4.6))
    w, h, gap, left, bottom = 3.05, 3.05, 0.14, 0.07, 0.72
    texts = [
        (
            "Rung 3a: Child-Vertex Head Fits",
            BLUE,
            "Free latent, candidate-centred crops",
            "0.57 cm in-box  ·  DV 3.37 cm",
            None,
        ),
        (
            "Rung 3b: Child Head Bottleneck",
            ORANGE,
            f"0.82 cm acc  ·  {100 * gate:.0f}% coarse → {100 * child:.0f}% child",
            None,
            "Unseen children at chance",
        ),
    ]
    for i, (shot, (head, tone, line1, line2, cause)) in enumerate(
        zip(shots, texts, strict=True)
    ):
        x = left + i * (w + gap)
        ax = fig.add_axes((x / 6.4, bottom / 4.6, w / 6.4, h / 4.6))
        ax.imshow(shot[box], interpolation="lanczos")
        ax.set_axis_off()
        cx = (x + w / 2) / 6.4
        fig.text(
            cx,
            4.43 / 4.6,
            head,
            ha="center",
            va="center",
            fontsize=12,
            fontweight="bold",
            color=tone,
        )
        fig.text(
            cx,
            4.20 / 4.6,
            line1,
            ha="center",
            va="center",
            fontsize=9.5,
            color="#555555",
        )
        if line2:
            fig.text(
                cx,
                4.00 / 4.6,
                line2,
                ha="center",
                va="center",
                fontsize=9.5,
                color=BLUE,
                fontweight="bold",
            )
        if cause:
            fig.text(
                cx,
                4.00 / 4.6,
                cause,
                ha="center",
                va="center",
                fontsize=9.5,
                color=ORANGE,
                fontweight="bold",
            )
    colorbar(fig, (1.7 / 6.4, 0.42 / 4.6, 3.0 / 6.4, 0.11 / 4.6))
    fig.savefig(OUT / f"e69_slide_rung3{SUFFIX}.png", dpi=300)
    return [
        (
            f"rung3: mean distance 3a {means[0]:.2f}, 3b {means[1]:.2f} cm; "
            f"E2 gate recall {100 * gate:.1f} %, child recall {100 * child:.1f} %"
        )
    ]


def backup(s: Loaded) -> list[str]:
    """Draw in-box GT99v with the far unseen surface in orange next to the H6 oracle (5.5 x 4.2 in)."""
    camera, eye, top = dollhouse(s)
    occupied = (s.grid[0] > 0).numpy()
    dist = distance_transform_edt(~occupied)
    index = np.floor(s.truth_cells).astype(np.int64)
    inside = s.inside & ((index >= 0) & (index < np.array(s.dims))).all(1)
    far = np.zeros(len(s.truth), dtype=bool)
    d = dist[tuple(index[inside][:, ::-1].T)]
    far[inside] = (~s.seen[inside]) & (d >= 8) & (d <= 13)
    gray = np.full((len(s.truth), 3), 0.80)
    gray[far] = mpl.colors.to_rgb(ORANGE)
    h6 = oracle(s, masks(s, 6)[2])[0]
    err, color = error_colors(s.truth, h6)
    shots = [
        render(s.truth[s.inside], gray[s.inside], camera, eye, top),
        render(h6, color, camera, eye, top),
    ]
    box = crop_box(shots)
    fig = plt.figure(figsize=(5.5, 4.2))
    w, gap, left, bottom = 2.6, 0.1, 0.1, 0.72
    heads = [
        (
            "Missing surface",
            ORANGE,
            f"Unseen, 8\u201313 cells from DV  ·  {100 * far[s.inside].mean():.1f}% of GT",
        ),
        ("Locked H6 oracle", BLUE, "0.39 cm in-box  ·  DV 3.37 cm"),
    ]
    for i, (shot, (head, tone, sub)) in enumerate(zip(shots, heads, strict=True)):
        x = left + i * (w + gap)
        ax = fig.add_axes((x / 5.5, bottom / 4.2, w / 5.5, w / 4.2))
        ax.imshow(shot[box], interpolation="lanczos")
        ax.set_axis_off()
        cx = (x + w / 2) / 5.5
        fig.text(
            cx,
            3.86 / 4.2,
            head,
            ha="center",
            va="center",
            fontsize=12,
            fontweight="bold",
            color=tone,
        )
        fig.text(
            cx, 3.64 / 4.2, sub, ha="center", va="center", fontsize=9.5, color="#555555"
        )
    colorbar(fig, (2.9 / 5.5, 0.42 / 4.2, 2.3 / 5.5, 0.11 / 4.2))
    fig.text(
        0.1 / 5.5,
        0.47 / 4.2,
        "Bedroom (E69f C3): 96 % of the unseen surface\nbeyond reach 4 lies outside every input frustum",
        fontsize=8.5,
        color="#666666",
        va="center",
    )
    fig.savefig(OUT / f"e69_slide_rung0_render{SUFFIX}.png", dpi=300)
    return [
        f"backup: far unseen {int(far.sum()):,} GT points; H6 oracle mean distance {err.mean():.2f} cm"
    ]


def trio(s: Loaded) -> list[str]:
    """Draw Déjà View, the GT20-target oracle and E2 under their `cloud_eval` dev means."""
    report = yaml.safe_load((RESULTS / "e69f_eval_dev.yaml").read_text())
    scores = report["splits"]["dev"]["aggregate"]
    camera, eye, top = dollhouse(s)
    panels = [
        ("Déjà View", "Déjà View", s.dv),
        ("Oracle ceiling", "Oracle GT20 targets", cloud_of(s, "e69j-oracle-gt20")),
        ("Trained today", "Trained E2", cloud_of(s, "e69j-e2")),
    ]
    gt20 = s.gt_world[s.has]
    shots = [
        render(c, error_colors(gt20, c)[1], camera, eye, top, 0.6 * s.child_m)
        for _, _, c in panels
    ]
    box = crop_box(shots)
    width, height, side, gap = 12.2, 4.4, 3.5, 0.2  # inches, at the slide's size
    base = 0.15  # room under the images for the colour bar's lowest label
    fig = plt.figure(figsize=(width, height))
    lines = []
    for i, (shot, (title, variant, _)) in enumerate(zip(shots, panels, strict=True)):
        x = 0.05 + i * (side + gap)
        ax = fig.add_axes((x / width, base / height, side / width, side / height))
        ax.imshow(shot[box], interpolation="lanczos")
        ax.set_axis_off()
        dev = f"dev {scores[variant]['chamfer_cm']:.2f} cm"
        for y, text, size, tone in ((4.18, title, 16, INK), (3.86, dev, 12, MUTED)):
            fig.text(
                (x + side / 2) / width,
                y / height,
                text,
                ha="center",
                va="center",
                fontsize=size,
                color=tone,
            )
        lines.append(f"trio: {title}: {dev}")
    left = 0.05 + 3 * side + 2 * gap + 0.2
    bar = fig.colorbar(
        mpl.cm.ScalarMappable(mpl.colors.Normalize(0, VMAX), ERROR),
        cax=fig.add_axes((left / width, base / height, 0.14 / width, side / height)),
    )
    bar.outline.set_visible(False)
    bar.set_ticks(
        [0, VMAX / 2, VMAX],
        labels=["0 cm", f"{VMAX / 2:g} cm", f"\u2265 {VMAX:.0f} cm"],
    )
    bar.ax.tick_params(labelsize=12, colors=INK, length=4, width=0.8)
    fig.savefig(OUT / f"e69_slide_trio{SUFFIX}.png", dpi=300)
    return lines


def main() -> None:
    """Draw one figure and print the numbers it shows."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("figure", choices=["rung0", "rung3", "backup", "trio"])
    parser.add_argument(
        "--light", action="store_true", help="reversed scale: pale is close to the GT"
    )
    args = parser.parse_args()
    if args.light:
        global ERROR, SUFFIX
        ERROR, SUFFIX = (
            LinearSegmentedColormap.from_list("magma088_r", MAGMA[::-1]),
            "_light",
        )
    if args.figure == "rung0":
        lines = rung0()
    else:
        s = load(for_arm(load_rung(CONFIG, "overfit1"), "ovoxel-gt99"), BEDROOM)
        lines = {"rung3": rung3, "backup": backup, "trio": trio}[args.figure](s)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
