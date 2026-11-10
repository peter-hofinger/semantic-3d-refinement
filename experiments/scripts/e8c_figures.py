"""E8c: the bedroom (views 18, 0, 12, 6): where SAM 3's gate and the residual test cut.

Columns: image; per pixel, the share of its E_mv matches SAM 3's gate cuts and the
share the residual test of the same size cuts (grey: no match); the distance to the
GT refined with the gate only; then the residual test and SAM 3, each minus the gate
only, on ±2 cm (blue: closer to the GT than the gate only). One run of
`experiments/configs/e8c_residual.yaml`.

uv run python experiments/scripts/e8c_figures.py runs/<run> \
    > experiments/results/e8c_figures.txt
"""

import sys
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from matplotlib.colors import Normalize
from scipy.spatial import cKDTree

from src.analysis.scene_eval import load_inputs
from src.analysis.scene_figures import (
    DIVERGING,
    PRESENTATION_SCENE,
    PRESENTATION_VIEWS,
    SEQUENTIAL,
)
from src.refinement.multiview import (
    RefineConfig,
    find_matches,
    residuals,
    same_segment_matches,
)

sys.path.insert(0, str(Path(__file__).parent))
from e40_figures import distances, variant_points

CHANGE_CM = 2.0
REFERENCE = "gate only"
COMPARED = ("residual @ SAM 3", "SAM 3", "Off", "random @ SAM 3, seed 0")
OUT = Path("experiments/figures/e8c_bedroom.jpg")


def cut_shares(inputs: object, neighbors: int) -> tuple[np.ndarray, np.ndarray]:
    """(V, H, W) twice: share of each pixel's matches cut by SAM 3, by residual; NaN."""
    scene = inputs.scene
    scale = scene.depth.median()
    d0 = scale / scene.depth
    m = find_matches(scene, d0, scale, neighbors)
    ids = torch.from_numpy(inputs.labels.instance_ids).long()
    kept = same_segment_matches(m, scene, d0, scale, ids)
    key = m.dst * d0.numel() + m.src
    sam3_cut = ~torch.isin(key, kept.dst * d0.numel() + kept.src)
    order = (
        residuals(d0, scale, scene, m).square().argsort(descending=True, stable=True)
    )
    residual_cut = torch.zeros_like(sam3_cut)
    residual_cut[order[: int(sam3_cut.sum())]] = True
    total = torch.bincount(m.src, minlength=d0.numel()).double()
    out = []
    for cut in (sam3_cut, residual_cut):
        n = torch.bincount(m.src[cut], minlength=d0.numel()).double()
        share = torch.where(total > 0, n / total.clamp_min(1), torch.nan)
        out.append(share.reshape(d0.shape).numpy())
    return out[0], out[1]


def main() -> None:
    """Draw the figure and print each variant's mean change on the bedroom."""
    run_dir = Path(sys.argv[1])
    raw = yaml.safe_load((run_dir / "run.yaml").read_text(encoding="utf-8"))["config"]
    base = RefineConfig(**yaml.safe_load(Path(raw["refine"]).read_text())["refine"])
    inputs = load_inputs(raw, PRESENTATION_SCENE, torch.device("cpu"))
    tree = cKDTree(inputs.truth.points)
    ref = distances(tree, variant_points(run_dir, REFERENCE, inputs))
    changes = {}
    for name in COMPARED:
        changes[name] = distances(tree, variant_points(run_dir, name, inputs)) - ref
        c = changes[name]
        print(
            f"{PRESENTATION_SCENE}, all 20 views, {name} minus {REFERENCE}: mean"
            f" {c.mean():+.4f} cm, pixels changed by > 0.5 cm {(np.abs(c) > 0.5).mean():.4f}"
        )
    sam3, resid = cut_shares(inputs, base.neighbors)
    matched = np.isfinite(sam3)
    print(
        f"{PRESENTATION_SCENE}: matched pixels {matched.mean():.3f}, of them with any"
        f" match cut by SAM 3 {(sam3[matched] > 0).mean():.3f}, by residual"
        f" {(resid[matched] > 0).mean():.3f}, by both {((sam3 > 0) & (resid > 0))[matched].mean():.3f}"
    )
    shown = COMPARED[:2]
    titles = [
        "image",
        "share of E_mv matches\nSAM 3's gate cuts",
        "share the residual test\nof the same size cuts",
        "gate only\ndistance to GT (cm)",
        *(f"{n}\nminus gate only" for n in shown),
    ]
    views = PRESENTATION_VIEWS
    fig, axes = plt.subplots(
        len(views),
        len(titles),
        figsize=(2.0 * len(titles), 1.6 * len(views) + 1.2),
        layout="constrained",
    )
    grey = plt.get_cmap("magma").copy()
    grey.set_bad("0.6")
    for row, v in zip(axes, views, strict=True):
        row[0].imshow(inputs.views.images[v])
        sh = row[1].imshow(sam3[v], cmap=grey, norm=Normalize(0, 1))
        row[2].imshow(resid[v], cmap=grey, norm=Normalize(0, 1))
        im = row[3].imshow(ref[v], cmap=SEQUENTIAL, norm=Normalize(0, 10))
        for ax, name in zip(row[4:], shown, strict=True):
            ch = ax.imshow(
                changes[name][v], cmap=DIVERGING, norm=Normalize(-CHANGE_CM, CHANGE_CM)
            )
        for ax in row:
            ax.set_xticks([])
            ax.set_yticks([])
        row[0].set_ylabel(f"view {v}")
    for ax, title in zip(axes[0], titles, strict=True):
        ax.set_title(title, fontsize=8)
    fig.colorbar(
        sh, ax=axes[-1, 1:3].tolist(), location="bottom", label="share, grey: no match"
    )
    fig.colorbar(im, ax=axes[-1, 3], location="bottom", label="cm, capped at 10")
    fig.colorbar(
        ch,
        ax=axes[-1, 4:].tolist(),
        location="bottom",
        label="cm, blue: closer than the gate only",
    )
    fig.suptitle(
        "Bedroom ai_006_007: where SAM 3's gate and a residual test of the same size cut"
        " E_mv's matches, and what each does to the surface",
        x=0.01,
        ha="left",
        fontsize=9,
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=110)
    plt.close(fig)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
