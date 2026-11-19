"""E57c: two ways to close the gate's seam on the bedroom, plotted on view 18.

Starting from `e57c_four_colors` (confidence cut q_c 0.5, greedy oracle q_g 0.5, gate delta 8 cm):
1. band: confidence v <= 0.5 is kept; in the band 0.5 < v <= `BAND` Déjà View's pixel is kept iff the
   oracle keeps it (no distance gate); beyond the band everything is dropped;
2. lines: the gate's blue pixels (conf-dropped, oracle-kept, within delta of the kept set)
   come back when they lie on a straight run, in one of four directions (row, column, both
   diagonals) of one view's image, that only passes through blue pixels of one SAM 3 instance
   and whose two ends are a white (confidence-kept) and a green (oracle-reintroduced) pixel of
   that instance; one pass, the pixels that come back do not seed others.
Colors: white kept by confidence; green reintroduced by the oracle (rule 1: in the band; rule 2: the
gate's green); gold, rule 2's reintroduced blue; blue, oracle-kept but not reintroduced; red,
dropped. Titles: kept share and Chamfer of the bedroom over all 20 views (cm).

uv run python -m experiments.scripts.e57c_two_fills > experiments/results/e57c_two_fills.txt
"""

from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from matplotlib.colors import ListedColormap

from experiments.scripts.e57c_hybrid import far_from
from experiments.scripts.e57c_oracle_curve import evaluate, greedy, rank_scores
from experiments.scripts.e57c_seams import DELTA, LEVELS, drop_order
from src.analysis.scene_eval import load_inputs
from src.analysis.scene_figures import PRESENTATION_SCENE

VIEW = 18
BAND = 0.7  # the figure's band; Peter's reading: confidence percentile 0.3-0.5, v in (0.5, 0.7]
BANDS = (0.6, 0.7, 0.8, 0.9, 1.0)  # swept for the Chamfer: 1.0 is the union
COLORS = ["white", "tab:red", "tab:green", "tab:blue", "gold"]
WHITE, RED, GREEN, BLUE, GOLD = range(5)
DIRECTIONS = ((0, 1), (1, 0), (1, 1), (1, -1))


def runs_on_lines(category: np.ndarray, ids: np.ndarray) -> np.ndarray:
    """Mask (H, W) of the blue pixels on a straight run closed by white and green of the run's instance."""
    h, w = category.shape
    out = np.zeros((h, w), dtype=bool)
    for dy, dx in DIRECTIONS:
        starts = [(0, x) for x in range(w)] + [(y, 0) for y in range(1, h)]
        if dx < 0:
            starts = [(0, x) for x in range(w)] + [(y, w - 1) for y in range(1, h)]
        for y0, x0 in starts:
            n = min(
                (h - y0 + dy - 1) // dy if dy else h,
                (w - x0 + dx - 1) // dx if dx > 0 else (x0 // -dx + 1 if dx < 0 else w),
            )
            ys, xs = y0 + dy * np.arange(n), x0 + dx * np.arange(n)
            c, i = category[ys, xs], ids[ys, xs]
            blue = c == BLUE
            if not blue.any():
                continue
            edges = np.flatnonzero(
                np.diff(np.concatenate([[0], blue.astype(int), [0]]))
            )
            for s, e in zip(edges[::2], edges[1::2], strict=True):  # the run is [s, e)
                if s == 0 or e == n or len(set(i[s:e])) != 1:
                    continue
                if i[s - 1] == i[s] == i[e] and {c[s - 1], c[e]} == {WHITE, GREEN}:
                    out[ys[s:e], xs[s:e]] = True
    return out


def main() -> None:
    """Draw view 18 for both rules and print the bedroom scores."""
    raw = yaml.safe_load(Path("experiments/configs/e57c_oracle.yaml").read_text())
    inputs = load_inputs(raw, PRESENTATION_SCENE, torch.device("cpu"))
    truth, pixels = inputs.truth, inputs.truth.pixels
    assert pixels is not None
    points, _, pred, a, _, ranking = rank_scores(inputs)
    v_conf, v_oracle = drop_order(
        ranking["confidence"].astype(np.float64), greedy(truth, pred, a, LEVELS)
    )
    base, oracle = v_conf <= 0.5, v_oracle <= 0.5
    far = far_from(pred, base, DELTA)

    def full(values: np.ndarray, fill: int) -> np.ndarray:
        out = np.full(pixels.shape, fill)
        out[pixels] = values
        return out

    for band_edge in BANDS:
        kept = base | ((v_conf <= band_edge) & oracle)
        print(
            f"rule 1 with the band to {band_edge}: kept {kept.mean():.3f}, Chamfer {evaluate(truth, points, full(kept, 1).astype(bool))[2]:.3f} cm"
        )
    # rule 1
    band = ~base & (v_conf <= BAND)
    cat1 = np.where(
        base, WHITE, np.where(band & oracle, GREEN, np.where(oracle, BLUE, RED))
    )
    # rule 2
    gate = np.where(base, WHITE, np.where(oracle, np.where(far, GREEN, BLUE), RED))
    gate_full = full(gate, WHITE)
    ids = inputs.labels.instance_ids
    back = np.stack([runs_on_lines(gate_full[v], ids[v]) for v in range(len(ids))])
    cat2_full = np.where(back, GOLD, gate_full)
    cat1_full = full(cat1, WHITE)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), layout="constrained")
    for ax, cat, title in (
        (
            axes[0],
            cat1_full,
            f"1. transition band 0.5 < v <= {BAND}: the oracle decides",
        ),
        (
            axes[1],
            cat2_full,
            "2. blue pixels on a straight run between white and green of one instance",
        ),
    ):
        kept = np.isin(cat, [WHITE, GREEN, GOLD])
        chamfer = evaluate(truth, points, kept)[2]
        print(
            f"{title}: kept {kept.mean():.3f}, Chamfer {chamfer:.3f} cm; view {VIEW}: "
            + ", ".join(
                f"{n} {(cat[VIEW] == k).mean():.3f}"
                for k, n in enumerate(["white", "red", "green", "blue", "gold"])
            )
        )
        ax.imshow(
            cat[VIEW],
            cmap=ListedColormap(COLORS),
            vmin=-0.5,
            vmax=4.5,
            interpolation="nearest",
        )
        ax.set_title(
            f"{title}\nkept {kept.mean():.2f}, Chamfer {chamfer:.2f} cm (all views)",
            fontsize=9,
        )
        ax.set_xticks([])
        ax.set_yticks([])
    names = [
        "kept by confidence",
        "dropped",
        "oracle, reintroduced",
        "oracle-kept, not reintroduced",
        "rule 2: blue brought back",
    ]
    order = [WHITE, GREEN, GOLD, BLUE, RED]
    fig.legend(
        [
            plt.Rectangle((0, 0), 1, 1, facecolor=COLORS[k], edgecolor="0.5")
            for k in order
        ],
        [names[k] for k in order],
        loc="outside lower center",
        ncols=5,
        fontsize=8,
    )
    fig.suptitle(
        f"{PRESENTATION_SCENE}, view {VIEW}: confidence q 0.5, greedy oracle q 0.5",
        fontsize=10,
    )
    fig.savefig("experiments/figures/e57c_two_fills_view18.jpg", dpi=90)
    plt.close(fig)


if __name__ == "__main__":
    main()
