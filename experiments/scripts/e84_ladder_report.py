"""E84 ladder report figures (issue #84): loss curves of rungs 3-5 and what the net outputs against the optimum.

`<name>_curves.jpg`: one column per rung, train loss of every step (blue) and the exact Chamfer of the net's cut at the evals (orange), val loss (val smoothed over 50 steps, raw faint),
the sample's uncut loss and its free-logit optima (cells of 1, 2, 4 pixels) as lines.
`<name>_maps.jpg`: view 18 of the bedroom, the rung 3 net's keep probability against the optimum's (no network), full and zoomed.

`uv run python experiments/scripts/e84_ladder_report.py --rung3 runs/... --rung4 runs/... --rung5 runs/... --out experiments/figures/ [--name e84_cut]`
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from experiments.scripts.e84_ladder import free_fit, load, uncut
from src.refinement.keep_chamfer import KeepResNet, KeepScene, standardize

WINDOW = 50


def scalar(run: Path, tag: str) -> np.ndarray:
    """Return the per-step values of a TensorBoard scalar of the run."""
    acc = EventAccumulator(str(run / "tb"), size_guidance={"scalars": 0})
    acc.Reload()
    return np.array([e.value for e in acc.Scalars(tag)])


def smooth(y: np.ndarray) -> np.ndarray:
    """Return the trailing mean over WINDOW steps."""
    c = np.cumsum(np.insert(y, 0, 0.0))
    n = np.minimum(np.arange(1, len(y) + 1), WINDOW)
    return (c[1:] - c[np.arange(len(y)) + 1 - n]) / n


def reference(run: Path) -> dict[str, float]:
    """Return the mean over the run's train samples of uncut and the three optima."""
    ref = yaml.safe_load((run / "run.yaml").read_text())["reference"]
    return {
        k: float(np.mean([v[k] for v in ref.values()]))
        for k in next(iter(ref.values()))
    }


def maps(run: Path, out: Path, name: str) -> None:
    """Write the rung 3 net's s against the optimum's on view 18, full and a zoom."""
    cfg = yaml.safe_load((run / "run.yaml").read_text())["config"]
    sample = KeepScene.views("ai_006_007_cam_00", cfg, [18], "cuda")[0]
    standardize([sample])
    net = KeepResNet(**cfg["resnet"]).cuda()
    net.load_state_dict(torch.load(run / "net.pt"))
    with torch.no_grad():
        mine = torch.sigmoid(net.logits(sample.features.float()))[0].cpu().numpy()
    _, optimum = free_fit(sample, cfg, 1)
    star = optimum.reshape(mine.shape).cpu().numpy()
    box = (slice(150, 230), slice(200, 280))
    fig, ax = plt.subplots(1, 4, figsize=(17, 3.6))
    for a, (title, img) in zip(
        ax,
        [
            ("optimum s*, free logits (no network)", star),
            ("network's s after 3000 steps", mine),
            ("optimum, 80 x 80 zoom", star[box]),
            ("network, same zoom", mine[box]),
        ],
        strict=True,
    ):
        im = a.imshow(img, vmin=0, vmax=1, cmap="viridis", interpolation="nearest")
        a.set_title(title, fontsize=9)
        a.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.012, label="keep probability s")
    fig.savefig(out / f"{name}_maps.jpg", dpi=80, pil_kwargs={"quality": 90})


def val_uncut(run: Path) -> float:
    """Return the keep-everything loss of the run's eval val views (the run's own if logged)."""
    saved = yaml.safe_load((run / "run.yaml").read_text())
    if "val_uncut_mean" in saved:
        return saved["val_uncut_mean"]
    cfg = saved["config"]
    val = load(cfg["val"], cfg)
    return float(
        np.mean([uncut(s, cfg) for s in val[:: max(1, len(val) // cfg["n_eval"])]])
    )


def curves(runs: dict[str, list[Path]], out: Path, name: str) -> None:
    """Write the loss curves, one column per rung: train against the optima, val below if logged.

    The first run of a column is the rung's official run, the others are thin repeats.
    """
    rows = 2 if any("rung 5" in title for title in runs) else 1
    fig, ax = plt.subplots(
        rows, len(runs), figsize=(6.2 * len(runs), 3.8 * rows), squeeze=False
    )
    for col, (title, paths) in enumerate(runs.items()):
        run = paths[0]
        top = ax[0, col]
        for extra in paths[1:]:
            top.plot(scalar(extra, "loss/train"), color="tab:gray", lw=0.6, alpha=0.8)
        train = scalar(run, "loss/train")
        top.plot(train, color="tab:blue", lw=1.0)
        hist = yaml.safe_load((run / "run.yaml").read_text())["history"]
        cut = [(h["step"], h["train_cut"]) for h in hist if "train_cut" in h]
        top.plot(*zip(*cut, strict=True), "o-", color="tab:orange", ms=3, lw=0.8)
        ref = reference(run)
        for key, style, label in [
            ("uncut", ":", "keep everything"),
            ("optimum_block4", "--", "optimum, 4 x 4 cells"),
            ("optimum_block2", "--", "optimum, 2 x 2 cells"),
            ("optimum_block1", "-.", "optimum, free pixels"),
        ]:
            if key in ref:
                top.axhline(ref[key], color="k", ls=style, lw=0.8)
                top.text(len(train), ref[key], f" {label}", fontsize=7, va="center")
        top.set(title=title, xlim=(0, len(train) * 1.3))
        top.set_ylim(
            min(ref.get("optimum_block1", 0.0), train.min()) * 0.97, ref["uncut"] * 1.05
        )
        if "rung 5" not in title:
            top.set_xlabel("step")
            top.grid(alpha=0.3, which="both")
            continue
        bottom = ax[1, col]
        val = scalar(run, "loss/val")
        bottom.plot(val, color="tab:orange", alpha=0.2, lw=0.6)
        bottom.plot(smooth(val), color="tab:orange")
        bottom.axhline(val_uncut(run), color="k", ls=":", lw=0.8)
        bottom.set(yscale="log", xlabel="step", xlim=(0, len(val) * 1.3))
        for a in (top, bottom):
            a.grid(alpha=0.3, which="both")
    ax[0, 0].set_ylabel("train loss: expected Chamfer (cm)")
    if rows == 2:
        ax[1, 0].set_ylabel("val loss (cm, log)")
    fig.tight_layout()
    fig.savefig(out / f"{name}_curves.jpg", dpi=80, pil_kwargs={"quality": 90})


def main() -> None:
    """Write both figures."""
    parser = argparse.ArgumentParser(description=__doc__)
    for rung in ("rung3", "rung4", "rung5"):
        parser.add_argument(f"--{rung}", type=Path, nargs="*")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--name", default="e84_ladder", help="figure file prefix")
    args = parser.parse_args()
    runs = {
        name: paths
        for name, paths in [
            ("rung 3: one view (train batch)", args.rung3),
            (
                "rung 4: 3 views (blue: official run, grey: repeats)",
                args.rung4,
            ),
            ("rung 5: growing training sets", args.rung5),
        ]
        if paths
    }
    curves(runs, args.out, args.name)
    if args.rung3:
        maps(args.rung3[0], args.out, args.name)


if __name__ == "__main__":
    main()
