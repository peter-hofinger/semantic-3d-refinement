"""E8b: the bedroom (views 18, 0, 12, 6): where SAM 3's gate cuts E_mv, and what it does.

Columns: image; SAM 3's segments (one colour per linked id); per pixel, the share of its
E_mv matches the gate cuts (grey: no match); the distance to the GT refined with
the random cut of SAM 3's size (seed 0); then SAM 3 and the gate alone, each minus that
random cut, on ±2 cm (blue: closer to the GT than the random cut). One run of
`experiments/configs/e8b_dropout.yaml`.

uv run python experiments/scripts/e8b_figures.py runs/<run> \
    > experiments/results/e8b_figures.txt
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
from src.refinement.multiview import RefineConfig, find_matches, same_segment_matches

sys.path.insert(0, str(Path(__file__).parent))
from e8a_figures import id_colours
from e40_figures import distances, variant_points

CHANGE_CM = 2.0
REFERENCE = "random @ SAM 3, seed 0"
COMPARED = ("SAM 3", "gate only", "Off", "random @ permuted")
OUT = Path("experiments/figures/e8b_bedroom.jpg")


def cut_share(inputs: object, neighbors: int) -> np.ndarray:
    """(V, H, W): share of each pixel's E_mv matches SAM 3's gate cuts, NaN unmatched."""
    scene = inputs.scene
    scale = scene.depth.median()
    d0 = scale / scene.depth
    m = find_matches(scene, d0, scale, neighbors)
    ids = torch.from_numpy(inputs.labels.instance_ids).long()
    kept = same_segment_matches(m, scene, d0, scale, ids)
    total = torch.bincount(m.src, minlength=d0.numel()).double()
    left = torch.bincount(kept.src, minlength=d0.numel()).double()
    share = torch.where(total > 0, 1 - left / total.clamp_min(1), torch.nan)
    return share.reshape(d0.shape).numpy()


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
    cut = cut_share(inputs, base.neighbors)
    print(
        f"{PRESENTATION_SCENE}: matched pixels {np.isfinite(cut).mean():.3f}, of them"
        f" with any match cut {(cut[np.isfinite(cut)] > 0).mean():.3f}"
    )
    shown = COMPARED[:2]
    titles = [
        "image",
        "SAM 3 segments\n(linked ids)",
        "share of E_mv matches\nSAM 3's gate cuts",
        "random cut, SAM 3's size\ndistance to GT (cm)",
        *(f"{n}\nminus random cut" for n in shown),
    ]
    views = PRESENTATION_VIEWS
    fig, axes = plt.subplots(
        len(views),
        len(titles),
        figsize=(2.0 * len(titles), 1.6 * len(views) + 1.2),
        layout="constrained",
    )
    colours = id_colours(inputs.labels.instance_ids)
    grey = plt.get_cmap("magma").copy()
    grey.set_bad("0.6")
    for row, v in zip(axes, views, strict=True):
        row[0].imshow(inputs.views.images[v])
        row[1].imshow(colours[v])
        sh = row[2].imshow(cut[v], cmap=grey, norm=Normalize(0, 1))
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
    fig.colorbar(sh, ax=axes[-1, 2], location="bottom", label="share, grey: no match")
    fig.colorbar(im, ax=axes[-1, 3], location="bottom", label="cm, capped at 10")
    fig.colorbar(
        ch,
        ax=axes[-1, 4:].tolist(),
        location="bottom",
        label="cm, blue: closer than the random cut",
    )
    fig.suptitle(
        "Bedroom ai_006_007: SAM 3's gate cuts E_mv at mask edges and on regions"
        " linked differently across views (ceiling); a random cut of the same size"
        " acts like no cut",
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
