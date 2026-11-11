"""E27a: flag, refine, filter against the filter alone; the table and the bedroom render.

Reads one `scene_eval --cfg experiments/configs/e27a.yaml` run.
- stdout: per split, each variant's accuracy / completeness / Chamfer as mean / median
  over the scenes, then each refined and cut output minus `Déjà View, kept 0.5`, which
  keeps the same pixels: mean, median and the scenes on which the refined one is better.
- `e27a_bedroom.jpg`: the bedroom from views 18, 0, 12, 6 pulled back, gray under a
  headlight: the ground truth, Déjà View, Déjà View cut at q 0.5 and the tangent output
  cut at q 0.5. The last two draw the same pixels, so they differ only by the refinement.

uv run python experiments/scripts/e27a_figures.py runs/<run> \
    > experiments/results/e27a_figures.txt
"""

import sys
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from src.analysis.render import headlight, splat
from src.analysis.scene_eval import load_inputs
from src.analysis.scene_figures import (
    PRESENTATION_SCENE,
    PRESENTATION_VIEWS,
    ground_truth_points,
    pixel_points,
    render_cameras,
    unit_normals,
)

OUT = Path("experiments/figures")
METRICS = ("accuracy_cm", "completeness_cm", "chamfer_cm")
REFERENCE = "Déjà View, kept 0.5"
REFINED = ("Refined, kept 0.5", "tangent 4e-2, kept 0.5")
SHOWN = ("Déjà View", REFERENCE, "tangent 4e-2, kept 0.5")
TITLES = {
    "Déjà View": "Déjà View",
    REFERENCE: "Déjà View, instance voxel density (q 0.5)",
    "tangent 4e-2, kept 0.5": "flag-refine-filter: E_t 4e-2 (q 0.5)",
}


def table(report: dict) -> None:
    """Print each split's scores and the refined cuts against the filter alone."""
    for split, record in report.items():
        scenes = record["scenes"]
        names = list(scenes)
        variants = list(scenes[names[0]]["variants"])

        def values(v: str, sc: dict = scenes, ns: list = names) -> np.ndarray:
            rows = [sc[n]["variants"][v]["headline"] for n in ns]
            return np.array([[r[m] for m in METRICS] for r in rows])

        print(f"\n{split} ({len(names)} scenes), cm, accuracy | completeness | Chamfer")
        print("variant | mean / median over scenes")
        for v in variants:
            a = values(v)
            cells = " | ".join(
                f"{a[:, i].mean():.3f} / {np.median(a[:, i]):.3f}" for i in range(3)
            )
            print(f"{v:24s} | {cells}")
        print(f"minus {REFERENCE} (same pixels): mean | median | scenes better")
        for v in REFINED:
            d = values(v) - values(REFERENCE)
            mean = " / ".join(f"{x:+.3f}" for x in d.mean(0))
            median = " / ".join(f"{x:+.3f}" for x in np.median(d, 0))
            better = " / ".join(str(int(x)) for x in (d < 0).sum(0))
            print(f"{v:24s} | {mean} | {median} | {better} of {len(names)}")


def render(run: Path, raw: dict) -> None:
    """Draw the bedroom's four clouds from the presentation cameras."""
    inputs = load_inputs(raw, PRESENTATION_SCENE, torch.device("cpu"))
    arrays = np.load(run / "results.npz")
    variants = [v["name"] for v in raw["variants"]]
    clouds = {"ground truth": (ground_truth_points(inputs), inputs.gt.depth > 0)}
    for name in SHOWN:
        key = f"{PRESENTATION_SCENE}/{variants.index(name)}"
        points = pixel_points(inputs, arrays[key], arrays[f"{key}/align"])
        clouds[TITLES[name]] = (points, arrays[f"{key}/kept"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    centers = inputs.views.cam_to_world[:, None, None, :3, 3]
    focal = inputs.views.intrinsics[:, None, None, 0, 0]
    cameras = {
        view: camera
        for view, camera in render_cameras(inputs, PRESENTATION_VIEWS).items()
        if view != "dollhouse"
    }
    fig, axes = plt.subplots(
        len(cameras),
        len(clouds),
        figsize=(4.6 * len(clouds), 3.7 * len(cameras)),
        layout="constrained",
    )
    for r, (row, (view, (camera, eye))) in enumerate(
        zip(axes, cameras.items(), strict=True)
    ):
        for ax, (name, (p, kept)) in zip(row, clouds.items(), strict=True):
            keep = kept & np.isfinite(p).all(-1)

            def t(a: np.ndarray, k: np.ndarray = keep) -> torch.Tensor:
                return torch.from_numpy(np.ascontiguousarray(a[k])).float().to(device)

            radii = 0.5 * np.linalg.norm(p - centers, axis=-1) / focal
            xyz = t(p)
            shade = headlight(t(unit_normals(p)), xyz, eye)
            ax.imshow(splat(xyz, shade, t(radii), camera))
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(name, fontsize=11)
        row[0].set_ylabel(view.split(",")[0], fontsize=11)
    fig.suptitle(
        "Bedroom ai_006_007, gray, lit from the camera. The last two columns keep the "
        "same pixels (flagged on Déjà View); they differ only by the refinement",
        x=0.01,
        ha="left",
        fontsize=12,
    )
    fig.savefig(OUT / "e27a_bedroom.jpg", dpi=80)
    plt.close(fig)


def main() -> None:
    """Print the table, then draw the bedroom."""
    run = Path(sys.argv[1])
    summary = yaml.safe_load((run / "run.yaml").read_text())
    print(f"run {run.name} (git {summary['git']})")
    table(summary["splits"])
    render(run, summary["config"])


if __name__ == "__main__":
    main()
