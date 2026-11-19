"""E57c: the segmentation masks as a verifier of the dropped low-confidence pixels.

Cut Déjà View's pixels by confidence at the kept fraction q_c (`known`). Per view, the weighted
Laplace membrane of E58a (`filters.fill_view`, the edges between SAM 3 instances cut) fills the
inverse depth of the other pixels from the known ones; it is exact for planes and carries a
wall's confident pixels across the wall's low-confidence hole. E58a replaced the dropped pixels
by this fill and lost (Déjà View's own depth is better there); here Déjà View's own point is
kept as it is and the fill only checks it: a dropped pixel comes back iff
|fill depth / Déjà View depth - 1| <= eps. A pixel that is known stays known, so the surfaces
confidence already covers cannot get worse; a point off the extrapolated surface (a flying
point) stays dropped.

`--splits dev` scores each (q_c, eps) like `e57c_hybrid` (cm, realigned, pixels with GT), and
the mean accuracy of the pixels that come back, and their share inside the greedy oracle's
kept set (q 0.5). `--figure` draws the bedroom's views: image with the SAM 3 borders; confidence
with the cut as an isoline; the true accuracy a_i (cm); the completeness cost m_i of dropping the
pixel alone (cm); the greedy oracle's keep (q 0.5); and the membrane's residual at the dropped
pixels, the isoline in every panel.

uv run python -m experiments.scripts.e57c_verify --splits dev
uv run python -m experiments.scripts.e57c_verify --figure
"""

import argparse
import time
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from matplotlib.colors import LogNorm

from experiments.scripts.e57c_oracle_curve import evaluate, greedy, rank_scores
from src.analysis.filters import fill_harmonic, keep_top, keep_top_within, ranks
from src.analysis.scene_eval import SceneInputs, load_inputs
from src.analysis.scene_figures import PRESENTATION_SCENE, PRESENTATION_VIEWS
from src.loaders.hypersim.config import load_config

QC = (  # the confidence cut: (name, q, grouping); 0.5 is dvlt's viewer default
    ("global", 0.5, None),
    ("global", 0.7, None),
    ("per view", 0.5, "view"),
    ("per instance", 0.5, "instance"),
)
EPS = (0.005, 0.01, 0.02, 0.04)  # relative depth tolerance of the verifier
QG = 0.5  # the greedy oracle's kept fraction


def residual(inputs: SceneInputs, known: np.ndarray) -> np.ndarray:
    """|membrane depth / Déjà View depth - 1| (V, H, W); 0 at the known pixels."""
    depth = inputs.scene.depth.numpy().astype(np.float64)
    filled = fill_harmonic(depth, known, ids=inputs.labels.instance_ids)
    out = np.abs(filled / depth - 1)
    assert np.isfinite(out).all()
    assert (out[known] < 1e-9).all()  # the float round trip 1 / (1 / d)
    out[known] = 0
    return out


def scatter(pixels: np.ndarray, values: np.ndarray, fill: float = np.nan) -> np.ndarray:
    """Values (N,) of the GT pixels as a (V, H, W) map, `fill` elsewhere."""
    out = np.full(pixels.shape, fill)
    out[pixels] = values
    return out


def scene_rows(cfg: dict, scene: str, *, figure: bool = False) -> dict:
    """Return the cuts' (share, accuracy, completeness, Chamfer, mean a back, share in oracle)."""
    inputs = load_inputs(cfg, scene, torch.device("cpu"))
    truth, pixels = inputs.truth, inputs.truth.pixels
    assert pixels is not None
    points, _, pred, a, m, ranking = rank_scores(inputs)
    oracle = greedy(truth, pred, a, [0.9, 0.8, 0.7, 0.6, QG])[QG]
    conf = ranking["confidence"].astype(np.float64)
    rows = {}
    views = np.broadcast_to(np.arange(pixels.shape[0])[:, None, None], pixels.shape)
    groups = {"view": views[pixels], "instance": inputs.labels.instance_ids[pixels]}
    for name, q, grouping in QC:
        base = (
            keep_top(ranks(conf), q)
            if grouping is None
            else keep_top_within(ranks(conf), groups[grouping], q)
        )
        known = scatter(pixels, base, fill=True).astype(bool)  # no GT: not scored, kept
        res = residual(inputs, known)
        for eps in (0.0, *EPS):
            kept_full = known | (res <= eps)
            kept = kept_full[pixels]
            back = kept & ~base
            score = evaluate(truth, points, kept_full)
            rows[(name, q, eps)] = np.array(
                [
                    kept_full.mean(),
                    *score,
                    a[back].mean() if back.any() else np.nan,
                    oracle[back].mean() if back.any() else np.nan,
                ]
            )
        if figure and (name, q) == ("global", 0.5):
            draw(inputs, conf, a, m, oracle, res, known, q)
    return rows


def draw(
    inputs: SceneInputs,
    conf: np.ndarray,
    a: np.ndarray,
    m: np.ndarray,
    oracle: np.ndarray,
    res: np.ndarray,
    known: np.ndarray,
    qc: float,
) -> None:
    """Draw the bedroom's presentation views, one row each, to experiments/figures/."""
    pixels = inputs.truth.pixels
    assert pixels is not None
    conf_map = inputs.geometry.confidence
    level = np.sort(conf)[round((1 - qc) * len(conf))]  # kept: conf >= level
    ids = inputs.labels.instance_ids
    keep_map = scatter(pixels, oracle.astype(float))
    panels = [
        ("image, SAM 3 borders", None, None),
        ("Déjà View confidence (log)", conf_map, LogNorm(1, conf_map.max())),
        ("true accuracy a_i (cm)", scatter(pixels, a), LogNorm(0.1, 30)),
        ("completeness cost m_i (cm)", scatter(pixels, m), LogNorm(0.01, 10)),
        ("greedy oracle keeps (q 0.5)", keep_map, None),
        (
            "membrane residual, dropped pixels",
            np.where(known, np.nan, res),
            LogNorm(1e-3, 0.3),
        ),
    ]
    views = PRESENTATION_VIEWS
    fig, axes = plt.subplots(
        len(views),
        len(panels),
        figsize=(3.4 * len(panels), 3.4 * len(views)),
        layout="constrained",
    )
    for r, v in enumerate(views):
        border = np.zeros(ids[v].shape, dtype=bool)
        border[:-1] |= ids[v][:-1] != ids[v][1:]
        border[:, :-1] |= ids[v][:, :-1] != ids[v][:, 1:]
        for c, (title, data, norm) in enumerate(panels):
            ax = axes[r, c]
            if data is None:
                image = inputs.views.images[v] / 255
                ax.imshow(np.where(border[..., None], [1, 0.9, 0], image))
            else:
                cmap = plt.get_cmap("RdYlGn" if "keeps" in title else "viridis").copy()
                cmap.set_bad("0.85")
                ax.imshow(
                    data[v],
                    norm=norm,
                    cmap=cmap,
                    vmin=0 if norm is None else None,
                    vmax=1 if norm is None else None,
                )
            ax.contour(conf_map[v], levels=[level], colors="red", linewidths=0.8)
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(title, fontsize=9)
        axes[r, 0].set_ylabel(f"view {v}")
    fig.suptitle(
        f"{PRESENTATION_SCENE}: the confidence cut at q {qc} (red isoline) against the oracle's values; "
        "gray: no GT / known pixel",
        x=0.01,
        ha="left",
    )
    fig.savefig("experiments/figures/e57c_verify_bedroom.jpg", dpi=70)
    plt.close(fig)


def main() -> None:
    """Print the dev means or draw the bedroom."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", nargs="*", default=[])
    parser.add_argument("--figure", action="store_true")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path("experiments/configs/e57c_oracle.yaml").read_text())
    if args.figure:
        rows = scene_rows(cfg, PRESENTATION_SCENE, figure=True)
        for (name, q, eps), row in rows.items():
            print(f"bedroom {name} {q} eps {eps}: " + " ".join(f"{x:.3f}" for x in row))
    split = load_config(Path(cfg["dataset"]))
    for name in args.splits:
        scenes = [f"{s}_{c}" for s, c in split.scenes([name]).items()]
        runs = []
        for scene in scenes:
            t0 = time.perf_counter()
            runs.append(scene_rows(cfg, scene))
            print(f"{scene}: {time.perf_counter() - t0:.0f} s", flush=True)
        print(f"{name}: mean over {len(runs)} scenes")
        print(
            "  q_c    eps   kept   accuracy / completeness / Chamfer (cm)   a back  in oracle"
        )
        for key in runs[0]:
            s, acc, comp, ch, back, inor = np.nanmean([r[key] for r in runs], axis=0)
            print(
                f"  {key[0]:>12} {key[1]:.1f}  {key[2]:.3f}  {s:.3f}  {acc:.3f} / {comp:.3f} / {ch:.3f}   {back:.2f}   {inor:.3f}"
            )
        for cut, q, _ in QC:
            per = [f"{r[(cut, q, 0.02)][3]:.2f}" for r in runs]
            print(f"  per scene Chamfer, {cut} {q}, eps 0.02: {' / '.join(per)}")


if __name__ == "__main__":
    main()
