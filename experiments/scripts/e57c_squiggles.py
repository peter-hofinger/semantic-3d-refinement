"""E57c: where the white squiggles of the straight-run fill come from, bedroom, pulled-back view 18.

Renders the cloud of `e57c_two_fills` rule 2 (confidence cut q_c 0.5, oracle q_g 0.5, gate 8 cm,
straight-run fill) from the camera of view 18 pulled back, twice with an id buffer (`render.splat`
with each point's flat pixel index as its color; ids < 2^24 are exact in float32): the kept pixels
and all pixels. A gap pixel of the render shows no kept splat but has a dropped pixel visible
behind it; the id buffer names that source pixel (view, y, x) and its category. Figure 1: the
render with the gaps by category (blue: the oracle keeps it, the fill did not; red: the oracle
drops it too) and by source view, and a zoom on `ZONE`. Figure 2: the zone's source views in image space (category
map, SAM 3 borders, confidence percentile). The printed counts say where the gap pixels lie
(image border, SAM 3 border, isolated, one pixel wide, confidence percentile) and why the blue
ones were not filled: the ends of their shortest straight run of blue pixels of one instance.

uv run python -m experiments.scripts.e57c_squiggles > experiments/results/e57c_squiggles.txt
"""

from collections import Counter
from itertools import pairwise
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from matplotlib.colors import ListedColormap
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree

from experiments.scripts.e57c_hybrid import far_from
from experiments.scripts.e57c_oracle_curve import greedy, rank_scores
from experiments.scripts.e57c_seams import DELTA, LEVELS, drop_order
from experiments.scripts.e57c_two_fills import (
    BLUE,
    COLORS,
    DIRECTIONS,
    GOLD,
    GREEN,
    RED,
    WHITE,
    runs_on_lines,
)
from src.analysis.render import Camera, splat
from src.analysis.scene_eval import SceneInputs, load_inputs
from src.analysis.scene_figures import (
    PRESENTATION_SCENE,
    PRESENTATION_VIEWS,
    render_cameras,
)

ZONE = (
    slice(30, 260),
    slice(400, 560),
)  # render rows, columns of the right-wall squiggle
BORDER = 8  # px: "at the image border"
ENDS = {
    WHITE: "kept",
    RED: "dropped",
    GREEN: "green",
    GOLD: "gold",
    5: "image border",
    6: "other instance",
}


def render_ids(
    inputs: SceneInputs,
    aligned: np.ndarray,
    keep: np.ndarray,
    camera: Camera,
    device: torch.device,
) -> np.ndarray:
    """Return (H, W) flat pixel ids of the nearest splat of the pixels in `keep`, -1 where none."""
    centers = inputs.views.cam_to_world[:, None, None, :3, 3]
    focal = inputs.views.intrinsics[:, None, None, 0, 0]
    idx = np.flatnonzero((keep & np.isfinite(aligned).all(-1)).ravel())
    assert len(idx) < 2**24
    radii = (0.5 * np.linalg.norm(aligned - centers, axis=-1) / focal).reshape(-1)[idx]
    xyz = torch.from_numpy(aligned.reshape(-1, 3)[idx]).float().to(device)
    color = (
        torch.from_numpy(idx.astype(np.float32))[:, None]
        .expand(-1, 3)
        .contiguous()
        .to(device)
    )
    image = splat(
        xyz, color, torch.from_numpy(radii).float().to(device), camera, background=-1.0
    )
    return np.rint(image[..., 0]).astype(np.int64)


def line_starts(h: int, w: int, dy: int, dx: int) -> list[tuple[int, int]]:
    """Start pixels of the lines of direction (dy, dx) that cover an (h, w) image."""
    if dy == 0:
        return [(y, 0) for y in range(h)]
    if dx == 0:
        return [(0, x) for x in range(w)]
    edge = 0 if dx > 0 else w - 1
    return [(0, x) for x in range(w)] + [(y, edge) for y in range(1, h)]


def run_info(category: np.ndarray, ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Length (4, H, W) and the two end codes (4, H, W, 2) of each blue pixel's run per direction."""
    h, w = category.shape
    length = np.zeros((4, h, w), dtype=int)
    ends = np.full((4, h, w, 2), -1, dtype=int)
    for d, (dy, dx) in enumerate(DIRECTIONS):
        for y0, x0 in line_starts(h, w, dy, dx):
            t = np.arange(max(h, w))
            ys, xs = y0 + dy * t, x0 + dx * t
            ok = (ys < h) & (xs >= 0) & (xs < w)
            n = len(ok) if ok.all() else int(ok.argmin())
            ys, xs = ys[:n], xs[:n]
            c, i = category[ys, xs], ids[ys, xs]
            key = np.where(c == BLUE, i, -2)
            cuts = np.flatnonzero(np.diff(np.concatenate([[-3], key, [-3]])))
            for s, e in pairwise(cuts):
                if key[s] == -2:
                    continue

                code = [
                    5 if not 0 <= k < n else (6 if i[k] != i[s] else int(c[k]))
                    for k in (s - 1, e)
                ]
                length[d, ys[s:e], xs[s:e]] = e - s
                ends[d, ys[s:e], xs[s:e]] = code
    return length, ends


def main() -> None:
    """Draw the two figures and print the counts."""
    raw = yaml.safe_load(Path("experiments/configs/e57c_oracle.yaml").read_text())
    inputs = load_inputs(raw, PRESENTATION_SCENE, torch.device("cpu"))
    truth, pixels = inputs.truth, inputs.truth.pixels
    assert pixels is not None
    points, align, pred, a, _, ranking = rank_scores(inputs)
    v_conf, v_oracle = drop_order(
        ranking["confidence"].astype(np.float64), greedy(truth, pred, a, LEVELS)
    )
    percentile = 1 - v_conf
    base, oracle = percentile > 0.5, v_oracle <= 0.5
    gate = np.where(
        base,
        WHITE,
        np.where(oracle, np.where(far_from(pred, base, DELTA), GREEN, BLUE), RED),
    )
    category = np.full(pixels.shape, WHITE)
    category[pixels] = gate
    ids = inputs.labels.instance_ids
    category = np.where(
        np.stack([runs_on_lines(category[v], ids[v]) for v in range(len(ids))]),
        GOLD,
        category,
    )
    conf = np.full(pixels.shape, np.nan)
    conf[pixels] = percentile
    aligned = points @ align[:3, :3].T + align[:3, 3]
    kept = np.isin(category, [WHITE, GREEN, GOLD])
    camera, _ = render_cameras(inputs, PRESENTATION_VIEWS)["view 18, pulled back"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ids_kept = render_ids(inputs, aligned, kept, camera, device)
    ids_all = render_ids(inputs, aligned, np.ones_like(kept), camera, device)
    gap = (ids_kept < 0) & (ids_all >= 0)
    hole = (ids_kept < 0) & (ids_all < 0)
    _, h, w = category.shape
    oracle_full = np.ones(kept.shape, dtype=bool)  # a pixel without GT is kept
    oracle_full[pixels] = oracle
    ids_oracle = render_ids(inputs, aligned, oracle_full, camera, device)
    ids_blue = render_ids(inputs, aligned, category == BLUE, camera, device)
    covered = gap & (
        ids_blue >= 0
    )  # a blue pixel (oracle keeps it, not brought back) would cover it
    source = np.where(covered, ids_blue, 0)
    cat_gap = np.where(gap, np.where(covered, BLUE, RED), -1)
    view_gap = np.where(covered, source // (h * w), -1)
    zone = np.zeros(gap.shape, dtype=bool)
    zone[ZONE] = True
    print(
        f"render {gap.shape}: gap pixels {gap.sum()}, holes (nothing behind either) {hole.sum()}; zone {ZONE}: gap {(gap & zone).sum()}"
    )
    for name, sel in (("render", gap), ("zone", gap & zone)):
        print(
            f"{name}: {sel.sum()} gap pixels; covered by the oracle-alone cloud {(sel & (ids_oracle >= 0)).sum()}, "
            f"by its blue pixels (oracle keeps, the fill did not bring back) {(sel & covered).sum()}, "
            f"by neither {(sel & ~covered & (ids_oracle < 0)).sum()}"
        )

    # figure 1
    kept_colors = torch.from_numpy((inputs.views.images / 255).reshape(-1, 3)).float()
    rgb = np.where(
        (ids_kept >= 0)[..., None], kept_colors.numpy()[np.maximum(ids_kept, 0)], 1.0
    )
    shown = rgb.copy()
    shown[hole] = 0.0
    shown[gap & (cat_gap == BLUE)] = [0.12, 0.47, 0.71]
    shown[gap & (cat_gap == RED)] = [0.84, 0.15, 0.16]
    cmap = plt.get_cmap("tab20")
    by_view = np.where(
        gap[..., None], cmap(np.maximum(view_gap, 0) % 20)[..., :3], 0.92
    )
    fig, axes = plt.subplots(2, 2, figsize=(14, 11), layout="constrained")
    for ax, image, title in (
        (axes[0, 0], rgb, "the straight-run cloud (kept pixels, image colors)"),
        (
            axes[0, 1],
            shown,
            "gaps: a blue pixel (oracle keeps, not brought back) would cover it / red: no blue pixel does; black: nothing behind",
        ),
        (axes[1, 0], by_view, "gaps by source view (tab20 color = view mod 20)"),
        (axes[1, 1], shown[ZONE], "zoom on the zone of the right-wall squiggle"),
    ):
        ax.imshow(image, interpolation="nearest")
        ax.set_title(title, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in axes.ravel()[:3]:
        ax.add_patch(
            plt.Rectangle(
                (ZONE[1].start, ZONE[0].start),
                ZONE[1].stop - ZONE[1].start,
                ZONE[0].stop - ZONE[0].start,
                fill=False,
                edgecolor="k",
                linewidth=1,
            )
        )
    fig.savefig("experiments/figures/e57c_squiggles_render.jpg", dpi=80)
    plt.close(fig)

    # the zone's source pixels
    src = np.unique(source[covered & zone])
    sv, sy, sx = src // (h * w), (src % (h * w)) // w, src % w
    print(
        f"zone: {len(src)} source pixels in {len(np.unique(sv))} views; most: {Counter(sv.tolist()).most_common(6)}"
    )
    tallies: dict[str, int] = Counter()
    pair_count: Counter = Counter()
    run_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for v, y, x in zip(sv, sy, sx, strict=True):
        cat = category[v]
        border = np.zeros((h, w), dtype=bool)
        border[:-1] |= ids[v][:-1] != ids[v][1:]
        border[:, :-1] |= ids[v][:, :-1] != ids[v][:, 1:]
        d_inst = distance_transform_edt(~border)[y, x]
        tallies["category blue"] += cat[y, x] == BLUE
        tallies["category red"] += cat[y, x] == RED
        tallies["within 2 px of a SAM 3 border"] += d_inst <= 2
        tallies[f"within {BORDER} px of the image border"] += (
            min(y, h - 1 - y, x, w - 1 - x) < BORDER
        )
        tallies["instance -1"] += ids[v][y, x] < 0
        around = [
            cat[min(max(y + dy, 0), h - 1), min(max(x + dx, 0), w - 1)]
            for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0))
        ]
        isol = all(k in (WHITE, GREEN, GOLD) for k in around)
        thin = (
            around[0] in (WHITE, GREEN, GOLD) and around[1] in (WHITE, GREEN, GOLD)
        ) or (around[2] in (WHITE, GREEN, GOLD) and around[3] in (WHITE, GREEN, GOLD))
        tallies["all 4 neighbors kept (isolated)"] += isol
        tallies["kept on both sides in a row or column (one pixel wide)"] += thin
        tallies["confidence percentile c > 0.3"] += conf[v, y, x] > 0.3
        if cat[y, x] == BLUE:
            if v not in run_cache:
                run_cache[v] = run_info(category[v], ids[v])
            length, ends = run_cache[v]
            best = int(np.argmin(length[:, y, x]))
            pair_count[tuple(sorted(ENDS[k] for k in ends[best, y, x]))] += 1
    print(
        "zone source pixels:",
        {k: int(val) for k, val in tallies.items()},
        "of",
        len(src),
    )
    borders = []
    for v in sorted(set(sv.tolist())):
        edge = np.pad(ids[v][:-1] != ids[v][1:], ((0, 1), (0, 0))) | np.pad(
            ids[v][:, :-1] != ids[v][:, 1:], ((0, 0), (0, 1))
        )
        borders.append(distance_transform_edt(~edge) <= 2)
    print(
        f"base rate: {np.mean(borders):.3f} of all pixels of those views lie within 2 px of a SAM 3 border"
    )
    flat = aligned.reshape(-1, 3)
    tree = cKDTree(
        flat[np.flatnonzero(kept.ravel())], balanced_tree=False, compact_nodes=False
    )
    near = 100 * tree.query(flat[src], workers=-1)[0]
    print(
        f"3D distance of the zone's source pixels to the nearest kept pixel (cm): median {np.median(near):.2f}, 90th percentile {np.percentile(near, 90):.2f}, max {near.max():.2f}"
    )
    accuracy = np.full(pixels.shape, np.nan)
    accuracy[pixels] = a
    print(
        f"true accuracy a_i of the zone's source pixels (cm): median {np.nanmedian(accuracy.reshape(-1)[src]):.2f}, "
        f"90th percentile {np.nanpercentile(accuracy.reshape(-1)[src], 90):.2f}; all pixels: median {np.median(a):.2f}"
    )
    pairs: Counter = Counter()
    for r, c in zip(*np.nonzero(gap & zone), strict=True):
        sides = (
            (ids_kept[r, max(c - 4, 0) : c][::-1], ids_kept[r, c + 1 : c + 5]),
            (ids_kept[max(r - 4, 0) : r, c][::-1], ids_kept[r + 1 : r + 5, c]),
        )
        for before, after in sides:
            first, second = before[before >= 0], after[after >= 0]
            if len(first) and len(second):
                same = first[0] // (h * w) == second[0] // (h * w)
                pairs[
                    "same source view on both sides"
                    if same
                    else "different source views"
                ] += 1
                break
        else:
            pairs["no kept pixel within 4 px on both sides"] += 1
    print(
        "what lies on the two sides of a gap pixel of the zone (kept render pixels within 4 px):",
        dict(pairs),
    )
    print(
        "blue ones: ends of the shortest straight run (why no run was closed by white and green):"
    )
    for pair, n in pair_count.most_common():
        print(f"  {pair}: {n}")

    # figure 2
    top = [v for v, _ in Counter(sv.tolist()).most_common(3)]
    fig, axes = plt.subplots(
        len(top), 3, figsize=(15, 4.6 * len(top)), layout="constrained"
    )
    axes = np.atleast_2d(axes)
    palette = ListedColormap(COLORS)
    for r, v in enumerate(top):
        mine = sv == v
        border = np.zeros((h, w), dtype=bool)
        border[:-1] |= ids[v][:-1] != ids[v][1:]
        border[:, :-1] |= ids[v][:, :-1] != ids[v][:, 1:]
        y0, y1, x0, x1 = (
            max(sy[mine].min() - 25, 0),
            min(sy[mine].max() + 26, h),
            max(sx[mine].min() - 25, 0),
            min(sx[mine].max() + 26, w),
        )
        for c, (title, data, cmap_, window) in enumerate(
            [
                (
                    f"view {v}: categories (dots: the zone's source pixels)",
                    category[v],
                    palette,
                    (slice(None), slice(None)),
                ),
                (
                    "zoom: categories + SAM 3 borders (black)",
                    category[v],
                    palette,
                    (slice(y0, y1), slice(x0, x1)),
                ),
                (
                    "zoom: confidence percentile c + borders",
                    conf[v],
                    plt.get_cmap("viridis"),
                    (slice(y0, y1), slice(x0, x1)),
                ),
            ]
        ):
            ax = axes[r, c]
            ax.imshow(
                data[window],
                cmap=cmap_,
                vmin=-0.5 if c < 2 else 0,
                vmax=4.5 if c < 2 else 1,
                interpolation="nearest",
            )
            oy, ox = window[0].start or 0, window[1].start or 0
            if c > 0:
                overlay = np.where(border[window], 1.0, np.nan)
                ax.imshow(
                    overlay,
                    cmap=ListedColormap(["k"]),
                    alpha=0.7,
                    interpolation="nearest",
                )
            ax.plot(
                sx[mine] - ox,
                sy[mine] - oy,
                ".",
                color="magenta",
                markersize=2 if c == 0 else 4,
            )
            ax.set_title(title, fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
    fig.savefig("experiments/figures/e57c_squiggles_sources.jpg", dpi=75)
    plt.close(fig)


if __name__ == "__main__":
    main()
