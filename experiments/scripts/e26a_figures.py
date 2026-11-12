"""E26a: the table and the figure of the competitors' geometry on ETH3D.

Reads one `e26a_eth3d.py` run and, if given, the PE3R scores the cluster wrote
(`e26a_score_points.py`, `scores.yaml`); a PE3R scene is used only if its views'
fingerprint matches ours.
- stdout: per group (indoor, outdoor, all), every method's mean / median over the scenes
  of accuracy, completeness and Chamfer (cm); then ours against Déjà View, no segments
  and permuted segments: mean and median difference and the scenes it is better on.
- `experiments/figures/e26a_eth3d.png`: (a) Chamfer per scene and method, log scale;
  (b) each refined variant's Chamfer minus Déjà View's, per scene.

uv run python experiments/scripts/e26a_figures.py runs/e26a-<time> [pe3r scores.yaml] \
    > experiments/results/e26a_eth3d.txt
"""

import hashlib
import sys
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

from src.loaders.views import Views

OUT = Path("experiments/figures/e26a_eth3d.png")
METRICS = ("accuracy_cm", "completeness_cm", "chamfer_cm")
OURS = "Refined, SAM 3 (ours)"
CONTROLS = ("Déjà View", "Refined, no segments", "Refined, permuted SAM 3")
# the reference palette's categorical slots in fixed order, plus a shape per method
STYLE = {
    "Déjà View": ("#2a78d6", "o"),
    OURS: ("#eb6834", "D"),
    "PanSt3R v2": ("#1baf7a", "s"),
    "EPS3D": ("#eda100", "^"),
    "IGGT": ("#e87ba4", "v"),
    "PE3R": ("#008300", "P"),
}
DELTAS = {
    "Refined, no segments": ("#6250d6", "o"),
    OURS: ("#eb6834", "D"),
    "Refined, permuted SAM 3": ("#e34948", "s"),
}


def fingerprint(scene: str) -> str:
    """SHA-1 of a scene's view images, as `e26a_score_points.py` computes it."""
    views = Views.load(Path(f"data/processed/eth3d/{scene}/views_n20_uniform_504.npz"))
    return hashlib.sha1(  # noqa: S324 - a fingerprint, not security
        np.ascontiguousarray(views.images).tobytes()
    ).hexdigest()


def merged(run: Path, pe3r: Path | None) -> tuple[dict, list[str]]:
    """Scenes with every method's scores, PE3R's added where its views match ours."""
    scenes = yaml.safe_load((run / "run.yaml").read_text())["scenes"]
    methods = list(next(iter(scenes.values()))["methods"])
    if pe3r is not None:
        remote = yaml.safe_load(pe3r.read_text())
        for name, s in remote.items():
            if s["views_sha1"] == fingerprint(name):
                scenes[name]["methods"]["PE3R"] = s
            else:
                print(f"PE3R {name}: other views, not used")
        methods.append("PE3R")
    return scenes, methods


def table(scenes: dict, methods: list[str]) -> None:
    """Print each group's mean / median and the paired comparisons of ours."""
    for group in ("indoor", "outdoor", "all"):
        names = [n for n, s in scenes.items() if group in ("all", s["kind"])]
        print(f"\n{group} ({len(names)} scenes): mean / median over scenes, cm")
        print(f"{'method':26s} | accuracy      | completeness  | Chamfer")
        for m in methods:
            rows = [scenes[n]["methods"].get(m) for n in names]
            if any(r is None for r in rows):
                print(f"{m:26s} | on {sum(r is not None for r in rows)} scenes only")
                continue
            a = np.array([[r[k] for k in METRICS] for r in rows])
            cells = " | ".join(
                f"{x:6.2f} / {y:5.2f}"
                for x, y in zip(a.mean(0), np.median(a, 0), strict=True)
            )
            print(f"{m:26s} | {cells}")
    print(
        "\nours minus each control, Chamfer cm: mean | median | scenes ours is better"
    )
    for c in CONTROLS:
        d = np.array(
            [s["methods"][OURS]["chamfer_cm"] - s["methods"][c]["chamfer_cm"]
             for s in scenes.values()]
        )  # fmt: skip
        print(
            f"{c:26s} | {d.mean():+.3f} | {np.median(d):+.3f} | {(d < 0).sum()} of {len(d)}"
        )


def figure(scenes: dict, methods: list[str]) -> None:
    """Draw (a) Chamfer per scene and method, (b) refined minus Déjà View per scene."""
    order = sorted(scenes, key=lambda n: (scenes[n]["kind"], n))
    y = np.arange(len(order))
    fig, (a, b) = plt.subplots(
        1, 2, figsize=(13, 6.5), sharey=True, layout="constrained",
        gridspec_kw={"width_ratios": [1.5, 1]},
    )  # fmt: skip
    for m, (color, marker) in STYLE.items():
        if m not in methods:
            continue
        x = [scenes[n]["methods"].get(m, {}).get("chamfer_cm", np.nan) for n in order]
        a.scatter(x, y, s=46, color=color, marker=marker, label=m, zorder=3,
                  edgecolors="#fcfcfb", linewidths=1)  # fmt: skip
    a.set_xscale("log")
    a.set_xlabel("Chamfer distance (cm, log scale; lower is better)")
    a.set_title("(a) every method, per scene", loc="left")
    a.legend(loc="lower right", fontsize=9, frameon=False)
    for (m, (color, marker)), off in zip(DELTAS.items(), (-0.2, 0, 0.2), strict=True):
        d = [
            scenes[n]["methods"][m]["chamfer_cm"]
            - scenes[n]["methods"]["Déjà View"]["chamfer_cm"]
            for n in order
        ]
        b.scatter(d, y + off, s=40, color=color, marker=marker, label=m, zorder=3,
                  edgecolors="#fcfcfb", linewidths=1)  # fmt: skip
    b.axvline(0, color="#888888", lw=1)
    b.set_xscale("symlog", linthresh=1)
    b.set_xticks([0, 0.2, 0.5, 1, 2], ["0", "0.2", "0.5", "1", "2"])
    b.set_xlabel("Chamfer minus Déjà View's (cm, symlog; < 0 is better)")
    b.set_title("(b) our refinement against its input", loc="left")
    b.legend(loc="lower right", fontsize=9, frameon=False)
    a.set_yticks(y, [f"{n} ({scenes[n]['kind']})" for n in order])
    for ax in (a, b):
        ax.grid(axis="x", color="#e5e5e2", lw=0.8)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(
        "ETH3D, 13 high-res training scenes, 20 views each, each output aligned by its "
        "own Sim(3). Look at (b): every refined variant sits right of 0.",
        x=0.01, ha="left", fontsize=11,
    )  # fmt: skip
    fig.savefig(OUT, dpi=110)
    plt.close(fig)


def main() -> None:
    """Print the table, draw the figure."""
    run = Path(sys.argv[1])
    pe3r = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    scenes, methods = merged(run, pe3r)
    print(
        f"run {run.name} (git {yaml.safe_load((run / 'run.yaml').read_text())['git']})"
    )
    table(scenes, methods)
    figure(scenes, methods)


if __name__ == "__main__":
    main()
