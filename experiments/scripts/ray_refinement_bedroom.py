"""Quick check: solving the ray maps (cameras) with the depth, on the bedroom; PanSt3R.

The default joint refiner (`configs/refine.yaml`, BiNI's E_n) with the cameras fixed,
against the same refiner after `camera_rounds` solves of the cameras on E_mv
(`multiview.refine_scene_and_cameras`), and the camera solve alone (no depth steps).
Inputs and scoring are E27a's (`experiments/configs/e27a.yaml`): 20 views at 504 px,
Déjà View K = 16, linked SAM 3 masks, DSINE; each output aligned onto the ground truth by
one Umeyama Sim(3) on its pixels. Besides the surface scores it measures the cameras
themselves against Hypersim's, in that same alignment: rotation (deg), center (cm) and
focal length (%), median over the 20 views.

Given PanSt3R's prediction (`panst3r_predict.py`) it is scored too, with its own
points and labels; every other row is labeled by the linked SAM 3 masks. The labels are
scored per pixel, apart from the geometry, so mIoU, PQ and AP compare only the labels
and accuracy, completeness and Chamfer only the geometry.

Bedroom only, no dev or val: a look, not a result.
- stdout: the table; `<out>/scores.yaml`: it and every variant's log.
- `<out>/bedroom.jpg`: the clouds from views 18, 0, 12, 6 pulled back, gray under a
  headlight, as `e27a_figures.py` draws them; `<out>/labels.jpg` the instances in those
  views: ground truth, linked SAM 3, PanSt3R.

uv run python experiments/scripts/ray_refinement_bedroom.py runs/ray_refinement \
    [runs/panst3r/ai_006_007_cam_00.npz]
"""

import dataclasses
import sys
import time
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from src.analysis.render import headlight, splat
from src.analysis.scene_eval import (
    HEADLINE,
    VOCABULARIES,
    git_hash,
    load_inputs,
)
from src.analysis.scene_figures import (
    PRESENTATION_SCENE,
    PRESENTATION_VIEWS,
    ground_truth_points,
    instance_colors,
    render_cameras,
    unit_normals,
)
from src.evaluation.labeled import score_output
from src.refinement.multiview import (
    RefineConfig,
    Scene,
    points_from_rays,
    refine_scene_and_cameras,
)
from src.segmentation.base import Segmentation

CAMERA_KEYS = ("rotation_deg", "center_cm", "focal_pct")
CONFIG = Path("experiments/configs/e27a.yaml")
VARIANTS = {
    "Déjà View": None,
    "Refined, cameras fixed": {},
    "cameras only": {"camera_rounds": 1, "gn_iters": 0},
    "Refined + cameras, 1 round": {"camera_rounds": 1},
    "Refined + cameras, 2 rounds": {"camera_rounds": 2},
}


def camera_errors(scene: Scene, align: np.ndarray, views: object) -> dict[str, float]:
    """Median rotation (deg), center (cm) and focal (%) error against the truth."""
    s = np.cbrt(np.linalg.det(align[:3, :3]))
    turn = align[:3, :3] / s
    rotation = turn @ scene.rotation.double().cpu().numpy()
    center = s * scene.center.double().cpu().numpy() @ turn.T + align[:3, 3]
    gt = views.cam_to_world.astype(np.float64)
    relative = np.einsum("vji,vjk->vik", gt[:, :3, :3], rotation)
    angle = np.degrees(np.arccos(np.clip((np.trace(relative, 0, 1, 2) - 1) / 2, -1, 1)))
    focal = scene.k[:, 0, 0].double().cpu().numpy() / views.intrinsics[:, 0, 0]
    return {
        "rotation_deg": float(np.median(angle)),
        "center_cm": float(
            np.median(100 * np.linalg.norm(center - gt[:, :3, 3], axis=1))
        ),
        "focal_pct": float(np.median(100 * np.abs(focal - 1))),
    }


def print_row(name: str, row: dict, seconds: float) -> None:
    """One variant's headline scores, its camera errors if measured, its runtime."""
    cams = [f"{row[k]:.3f}" if k in row else "-" for k in CAMERA_KEYS]
    cells = [f"{row[k]:.3f}" for k in HEADLINE] + cams + [f"{seconds:.1f}"]
    print(f"{name:28s} | " + " | ".join(cells), flush=True)


def render(inputs: object, clouds: dict, out: Path) -> None:
    """Draw every cloud from the presentation views, pulled back."""
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
        for ax, (name, p) in zip(row, clouds.items(), strict=True):
            keep = np.isfinite(p).all(-1)

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
        "Bedroom ai_006_007, gray, lit from the camera, every output aligned onto the "
        "ground truth by its own Sim(3). Look at wall corners and the bed's edges.",
        x=0.01,
        ha="left",
        fontsize=12,
    )
    fig.savefig(out / "bedroom.jpg", dpi=80)
    plt.close(fig)


def labels_figure(inputs: object, panst3r: Segmentation, out: Path) -> None:
    """Draw the instances of the presentation views: truth, linked SAM 3, PanSt3R."""
    rows = {
        "ground truth": inputs.gt.instance_ids,
        "SAM 3, linked": inputs.labels.instance_ids,
        "PanSt3R v2": panst3r.instance_ids,
    }
    fig, axes = plt.subplots(
        len(rows),
        len(PRESENTATION_VIEWS),
        figsize=(4 * len(PRESENTATION_VIEWS), 3.2 * len(rows)),
        layout="constrained",
    )
    for row, (name, ids) in zip(axes, rows.items(), strict=True):
        for ax, view in zip(row, PRESENTATION_VIEWS, strict=True):
            ax.imshow(instance_colors(ids[view]))
            ax.set_xticks([])
            ax.set_yticks([])
        row[0].set_ylabel(name, fontsize=11)
    for ax, view in zip(axes[0], PRESENTATION_VIEWS, strict=True):
        ax.set_title(f"view {view}", fontsize=11)
    fig.suptitle(
        "Bedroom ai_006_007, one color per instance id (shared across views within a "
        "row; colors are not matched across rows), gray: no instance",
        x=0.01,
        ha="left",
        fontsize=12,
    )
    fig.savefig(out / "labels.jpg", dpi=80)
    plt.close(fig)


def load_panst3r(path: Path) -> tuple[np.ndarray, Segmentation, float]:
    """PanSt3R's per-pixel points, its labels as a Segmentation, its runtime."""
    with np.load(path) as f:
        ids = f["class_ids"].tolist()
        labels = Segmentation(
            f["instance_ids"].astype(np.int32),
            dict(zip(ids, map(str, f["class_names"]), strict=True)),
            dict(zip(ids, f["scores"].tolist(), strict=True)),
        )
        return f["points"].astype(np.float64), labels, float(f["seconds"])


def main() -> None:
    """Refine the bedroom per variant, score it, measure its cameras, draw it."""
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    raw = yaml.safe_load(CONFIG.read_text())
    refine_raw = yaml.safe_load(Path(raw["refine"]).read_text())
    base = RefineConfig(**refine_raw["refine"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base = dataclasses.replace(base, device=device.type)
    inputs = load_inputs(raw, PRESENTATION_SCENE, device)
    evaluated = list(VOCABULARIES[raw["vocabulary"]])
    segments = torch.from_numpy(inputs.labels.instance_ids.astype(np.int64))
    refine_scene_and_cameras(
        inputs.scene, dataclasses.replace(base, gn_iters=1), inputs.normals
    )
    clouds = {"ground truth": ground_truth_points(inputs)}
    report = {"git": git_hash(), "config": str(CONFIG), "variants": {}}
    print("variant | " + " | ".join(HEADLINE) + " | rot deg | ctr cm | f % | s")
    for name, change in VARIANTS.items():
        scene, depth, log = inputs.scene, inputs.scene.depth, []
        t0 = time.perf_counter()
        if change is not None:
            cfg = dataclasses.replace(base, **change)
            scene, depth, log, _ = refine_scene_and_cameras(
                scene, cfg, inputs.normals, segments
            )
        if device.type == "cuda":
            torch.cuda.synchronize()
        seconds = time.perf_counter() - t0
        points = points_from_rays(depth.double(), scene.rays.double()).cpu().numpy()
        scores = score_output(inputs.truth, points, inputs.labels, evaluated)
        align = scores.pop("align")
        cams = camera_errors(scene, align, inputs.views)
        clouds[name] = points @ align[:3, :3].T + align[:3, 3]
        row = {k: float(scores[k]) for k in HEADLINE} | cams
        report["variants"][name] = row | {"seconds": seconds, "energy": log[-1:]}
        print_row(name, row, seconds)
    panst3r = None
    if len(sys.argv) > 2:
        points, panst3r, seconds = load_panst3r(Path(sys.argv[2]))
        scores = score_output(inputs.truth, points, panst3r, evaluated)
        align = scores.pop("align")
        clouds["PanSt3R v2"] = points @ align[:3, :3].T + align[:3, 3]
        row = {k: float(scores[k]) for k in HEADLINE}
        report["variants"]["PanSt3R v2"] = row | {"seconds": seconds}
        print_row("PanSt3R v2", row, seconds)
    (out / "scores.yaml").write_text(yaml.safe_dump(report, allow_unicode=True))
    render(inputs, clouds, out)
    if panst3r is not None:
        labels_figure(inputs, panst3r, out)


if __name__ == "__main__":
    main()
