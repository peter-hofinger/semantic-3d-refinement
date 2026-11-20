"""E84a figures: the loss against the exact Chamfer over the steps, and s per view.

`<prefix>_curves.jpg`: per step the loss L next to the exact Chamfer (cm, fixed alignment) of
one Bernoulli(s) draw and of the thresholded cloud, the exact completeness, and the mean s.
`<prefix>_s_views.jpg` (bedroom only, views 18, 0, 12, 6 of issue #7): the image, the
prediction's distance to the GT a_i (cm), and s at each saved step, all s on [0, 1].

`uv run python experiments/scripts/e84a_figures.py --run runs/<run> --out experiments/figures --prefix e84a`
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

from src.loaders.views import Views
from src.refinement.keep_chamfer import KeepScene

SCENE = "ai_006_007_cam_00"
VIEWS = (18, 0, 12, 6)
A_RANGE = 5.0  # cm, the colour scale of a_i


def curves(run: Path, out: Path, prefix: str) -> None:
    """Loss and exact scores over the steps."""
    record = yaml.safe_load((run / "run.yaml").read_text())["history"]
    step = [h["step"] for h in record]
    exact = [h["exact"][SCENE] for h in record]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].plot(step, [h["loss"] for h in record], label="loss L (expected Chamfer)")
    ax[0].plot(
        step,
        [e["bernoulli"]["chamfer_cm"] for e in exact],
        label="exact Chamfer, one Bernoulli(s) draw",
    )
    ax[0].axhline(1.846, color="gray", ls="--", label="Déjà View uncut 1.846")
    ax[0].set(
        xlabel="step",
        ylabel="Chamfer (cm)",
        yscale="log",
        title="loss and the exact Chamfer of its cloud",
    )
    ax[1].plot(step, [h["completeness_cm"] for h in record], label="C of the loss")
    ax[1].plot(
        step,
        [e["bernoulli"]["completeness_cm"] for e in exact],
        label="exact completeness, Bernoulli draw",
    )
    ax[1].plot(
        step,
        [e["threshold"]["completeness_cm"] for e in exact],
        label="exact completeness, s >= 0.5",
    )
    ax[1].axhline(1.172, color="gray", ls="--", label="Déjà View uncut 1.172")
    ax[1].set(
        xlabel="step",
        ylabel="completeness (cm)",
        yscale="log",
        title="completeness: the loss' term and the exact one",
    )
    ax[2].plot(step, [h["mean_s"] for h in record], label="mean s")
    ax[2].plot(
        step,
        [e["s"]["above_half"] for e in exact],
        label="share of pixels with s >= 0.5",
    )
    ax[2].set(xlabel="step", ylabel="share", title="how much of the cloud is kept")
    for a in ax:
        a.legend(fontsize=7)
        a.grid(alpha=0.3)
    fig.suptitle(f"{prefix} rung 3, bedroom {SCENE}: {run.name}", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / f"{prefix}_curves.jpg", dpi=130)
    plt.close(fig)


def maps(run: Path, out: Path, prefix: str) -> None:
    """Image, a_i and s at the saved steps for the four presentation views."""
    # E84a's runs predate the key `miss`, their loss charged d_(R+1)
    cfg = {"miss": "next"} | yaml.safe_load((run / "run.yaml").read_text())["config"]
    scene = KeepScene.load(SCENE, cfg, "cpu")
    images = Views.load(Path(cfg["root"]) / SCENE / f"{cfg['views']}.npz").images
    a_map = np.zeros(scene.valid.shape, dtype=np.float32)
    a_map[scene.valid.numpy()] = scene.probes.a
    steps = sorted(
        int(p.stem.rsplit("step", 1)[1])
        for p in run.glob(f"probabilities_{SCENE}_step*.npy")
    )
    rows = (
        [("image", None), ("a_i (cm)", a_map)]
        + [
            (
                f"s, step {k}",
                np.load(run / f"probabilities_{SCENE}_step{k}.npy").astype(np.float32),
            )
            for k in steps
        ]
        + [
            (
                f"s, step {cfg['steps']}",
                np.load(run / f"probabilities_{SCENE}.npy").astype(np.float32),
            )
        ]
    )
    fig, ax = plt.subplots(
        len(rows), len(VIEWS), figsize=(3.2 * len(VIEWS), 2.5 * len(rows))
    )
    for i, (label, data) in enumerate(rows):
        for j, v in enumerate(VIEWS):
            a = ax[i, j]
            if data is None:
                a.imshow(images[v])
            elif label.startswith("a_i"):
                shown = a.imshow(data[v], cmap="magma", vmin=0, vmax=A_RANGE)
            else:
                im = a.imshow(data[v], cmap="viridis", vmin=0, vmax=1)
            a.set(xticks=[], yticks=[])
            if j == 0:
                a.set_ylabel(label)
            if i == 0:
                a.set_title(f"view {v}")
    fig.colorbar(shown, ax=ax[1], fraction=0.02, label="cm (clipped at 5)")
    fig.colorbar(im, ax=ax[2:], fraction=0.02, label="keep probability s")
    fig.suptitle(
        f"{prefix} rung 3, bedroom: a_i and the head's s at the saved steps (shared scales, s = 1 keep)",
        fontsize=10,
    )
    fig.savefig(out / f"{prefix}_s_views.jpg", dpi=110, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Draw both figures."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--prefix", required=True, help="figure names, e.g. e84b")
    args = parser.parse_args()
    curves(args.run, args.out, args.prefix)
    maps(args.run, args.out, args.prefix)


if __name__ == "__main__":
    main()
