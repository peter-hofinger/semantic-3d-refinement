"""E34: the evaluation protocol on the 11 Hypersim val scenes, Déjà View vs Refined.

Reads the `run.yaml` that `src.analysis.scene_eval` wrote for
`configs/eval_hypersim_val.yaml` (cluster job 969) and prints the per-scene table, the
count of scenes each metric improves on and the means over the valid scenes. Also checks
every camera rotation of the prepared views and of Hypersim's raw keyframes for
orthonormality, which is what makes ai_031_004 invalid (its keyframe 0 is 10 x a rotation).
Figure: F@5cm, accuracy and completeness per scene, both variants on shared axes.

uv run python experiments/scripts/e34_val_eval.py runs/eval_hypersim_val-20261105-094825/run.yaml \
    > experiments/results/e34_val_eval.txt
"""

import sys
from pathlib import Path

import h5py
import matplotlib as mpl
import numpy as np
import yaml

mpl.use("Agg")
import matplotlib.pyplot as plt

PROCESSED = Path("data/processed/hypersim")
RAW = Path("data/hypersim/raw")
VIEWS = "views_n20_uniform_504.npz"
INVALID = {
    "ai_031_004_cam_00": "keyframe 0 rotation is 10 x orthonormal (raw Hypersim)"
}
VARIANTS = ("Déjà View", "Refined")
METRICS = ("accuracy_cm", "completeness_cm", "fscore", "miou", "ap50")
LOWER_IS_BETTER = {"accuracy_cm", "completeness_cm"}
FIGURE = Path("experiments/figures/e34_val_eval.jpg")


def rotation_error(rotations: np.ndarray) -> np.ndarray:
    """Max |R R^T - I| per rotation of an (N, 3, 3) stack."""
    eye = np.eye(3)
    return np.abs(rotations @ rotations.transpose(0, 2, 1) - eye).max(axis=(1, 2))


def check_rotations(scenes: list[str]) -> None:
    """Print the worst orthonormality error of prepared views and raw keyframes."""
    print("rotation check, max |R R^T - I| (prepared views; raw keyframes, bad ids)")
    for scene in scenes:
        prepared = np.load(PROCESSED / scene / VIEWS)["cam_to_world"][:, :3, :3]
        name, camera = scene.rsplit("_cam_", 1)
        raw_path = RAW / name / "_detail" / f"cam_{camera}"
        with h5py.File(raw_path / "camera_keyframe_orientations.hdf5") as f:
            raw = f["dataset"][:].astype(np.float64)
        bad = np.flatnonzero(rotation_error(raw) > 1e-3).tolist()
        print(
            f"  {scene}: {rotation_error(prepared).max():.1e}; "
            f"{rotation_error(raw).max():.1e}, {bad}"
        )


def main() -> None:
    """Print the table and the rotation check, write the figure."""
    run = yaml.safe_load(Path(sys.argv[1]).read_text(encoding="utf-8"))
    scenes = run["config"]["scenes"]
    print(f"run {sys.argv[1]}, git {run['git'][:7]}")
    print("scene | variant | scale | " + " | ".join(METRICS))
    for scene in scenes:
        for variant in VARIANTS:
            row = run["scenes"][scene]["variants"][variant]
            values = " | ".join(f"{row[m]:.3f}" for m in METRICS)
            print(f"{scene} | {variant} | {row['scale']:.2f} | {values}")
    valid = [s for s in scenes if s not in INVALID]
    print(f"\nvalid scenes: {len(valid)}; excluded: {INVALID}")
    table = {
        v: np.array(
            [[run["scenes"][s]["variants"][v][m] for m in METRICS] for s in valid]
        )
        for v in VARIANTS
    }
    before, after = table[VARIANTS[0]], table[VARIANTS[1]]
    for j, metric in enumerate(METRICS):
        gain = (
            before[:, j] - after[:, j]
            if metric in LOWER_IS_BETTER
            else after[:, j] - before[:, j]
        )
        print(
            f"{metric}: mean {before[:, j].mean():.3f} -> {after[:, j].mean():.3f}, "
            f"better on {(gain > 0).sum()} / {len(valid)}, worse on {(gain < 0).sum()}"
        )
    print()
    check_rotations(scenes)

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    x = np.arange(len(scenes))
    shown = (
        ("fscore", "F@5cm (higher is better)"),
        ("accuracy_cm", "accuracy (cm, lower is better)"),
        ("completeness_cm", "completeness (cm, lower is better)"),
    )
    for ax, (metric, label) in zip(axes, shown, strict=True):
        for k, variant in enumerate(VARIANTS):
            values = [run["scenes"][s]["variants"][variant][metric] for s in scenes]
            ax.bar(x + (k - 0.5) * 0.4, values, 0.4, label=variant)
        for i, scene in enumerate(scenes):
            if scene in INVALID:
                ax.axvspan(i - 0.5, i + 0.5, color="0.85", zorder=0)
        ax.set_ylabel(label)
    axes[0].set_ylim(0, 1.15)
    axes[0].legend(ncol=2, loc="upper center")
    axes[0].set_title(
        "Hypersim val (11 scenes, 20 views, 504 px): Déjà View vs joint refiner, every weight 1. "
        "Grey: ai_031_004 invalid (bad GT camera)",
        fontsize=9,
    )
    axes[-1].set_xticks(x, [s.removesuffix("_cam_00") for s in scenes], rotation=30)
    fig.tight_layout()
    fig.savefig(FIGURE, dpi=80, pil_kwargs={"quality": 85})


if __name__ == "__main__":
    main()
