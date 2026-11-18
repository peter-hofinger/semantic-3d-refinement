"""E52c, probe before any training: does the cross-view distance see Déjà View's error.

The term of `cross_view_distance` (src/backbone/adapters.py) on each dev scene's 12 fixed
views, 6000 random GT-valid pixels (seed 1), cap 0.05, on two point sets: GT points
(the floor: what perfect geometry scores) and Déjà View's points after one exact Sim(3)
onto GT (20000 pixels, seed 0), as the training loss aligns them. Two distances: point to
point (|x_p - x_q|, the first draft) and point to plane (|n_q · (x_p - x_q)|, normals from
each map at central-difference strides 2 and 7 px); partners within one GT instance
(gate) or any. A term that scores GT as high as Déjà View measures the sample spacing,
not the geometry; the gated / ungated gap on Déjà View bounds how much E52c's two arms
can differ in what they ask of the model.

uv run python experiments/scripts/e52c_probe.py
writes `experiments/results/e52c_probe.yaml` and `experiments/figures/e52c_probe.jpg`.
"""

from pathlib import Path

import matplotlib as mpl
import numpy as np
import torch
import yaml

mpl.use("Agg")
import matplotlib.pyplot as plt
from e52a_figures import FIGURES, style
from e52a_segment_adapters import load_scene, setup

from src.analysis.scene_eval import git_hash
from src.backbone.adapters import (
    SegmentAdapter,
    cross_view_distance,
    forward,
    map_normals,
    segment_codes,
)
from src.evaluation.labeled import umeyama_torch
from src.segmentation.base import NO_INSTANCE

SCENES = {"ai_010_007_cam_00": "living room", "ai_006_007_cam_00": "bedroom",
          "ai_044_001_cam_00": "lobby"}  # fmt: skip
VIEWS, PIXELS, CAP, STRIDES = 12, 6000, 0.05, (2, 7)
RESULTS = Path("experiments/results/e52c_probe.yaml")


def point_distance(
    x: torch.Tensor,
    view: torch.Tensor,
    instance: torch.Tensor,
    depth: torch.Tensor,
    *,
    gate: bool,
    cap: float,
) -> float:
    """`cross_view_distance` with |x_p - x_q| in place of the plane distance."""
    pairs = torch.unique(torch.stack([instance, view]), dim=1)
    ids, seen = torch.unique(pairs[0], return_counts=True)
    keep = (instance >= 0) & torch.isin(instance, ids[seen > 1])
    x, view, instance, depth = x[keep], view[keep], instance[keep], depth[keep]
    other = view[:, None] != view[None]
    if gate:
        other &= instance[:, None] == instance[None]
    d = torch.cdist(x[None], x[None])[0].masked_fill(~other, torch.inf).amin(1)
    return (d / depth).clamp(max=cap).mean().item()


def probe_scene(net: torch.nn.Module, raw: dict, name: str) -> dict:
    """Distances of GT and aligned Déjà View points on one scene's fixed views."""
    scene = load_scene(raw, name)
    sel = np.linspace(0, 19, VIEWS).round().astype(int)
    device = next(net.parameters()).device
    images = torch.from_numpy(scene.images[sel]).permute(0, 3, 1, 2).float() / 255
    codes, _ = segment_codes(np.full((VIEWS, scene.segments.shape[1]), NO_INSTANCE))
    zero = SegmentAdapter(bias=True, pool=False, rank=16)
    with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16):
        pred = forward(net, images[None].to(device), zero, codes)["world_points"][0]
    pred = pred.float()
    valid = torch.from_numpy(scene.depth[sel] > 0).to(device)
    gt = torch.from_numpy(scene.points[sel]).to(device)
    n = int(valid.sum())
    fit = torch.from_numpy(np.random.default_rng(0).choice(n, 20000, replace=False))
    s, r, t = umeyama_torch(pred[valid][fit].double(), gt[valid][fit].double())
    dv = (pred.double() @ (s * r.T) + t).float()
    at = torch.from_numpy(np.random.default_rng(1).choice(n, PIXELS, replace=False))
    at = at.to(device)
    view = torch.nonzero(valid)[:, 0][at]
    inst = torch.from_numpy(scene.instances[sel]).to(device)[valid].long()[at]
    depth = torch.from_numpy(scene.depth[sel]).to(device)[valid][at]
    out: dict = {}
    for label, points in (("gt", gt), ("dejaview", dv)):
        x = points[valid][at]
        for gate in (True, False):
            key = f"{label}_{'gate' if gate else 'any'}"
            out[f"point_{key}"] = point_distance(
                x, view, inst, depth, gate=gate, cap=CAP
            )
            for stride in STRIDES:
                normal = map_normals(points, stride)[valid][at]
                out[f"plane_s{stride}_{key}"] = cross_view_distance(
                    x, normal, view, inst, depth, gate=gate, cap=CAP
                ).item()
    return out


def figure(rows: dict[str, dict]) -> None:
    """Point and plane distance (stride 2), GT vs Déjà View, gate vs any, per scene."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    bars = [("gt_gate", "GT, gate", "#9db8d9"), ("gt_any", "GT, any", "#d0dcea"),
            ("dejaview_gate", "Déjà View, gate", "#eb6834"),
            ("dejaview_any", "Déjà View, any", "#f4b38f")]  # fmt: skip
    for ax, kind, title in ((axes[0], "point", "point to point (first draft)"),
                            (axes[1], "plane_s2", "point to plane (E52c)")):  # fmt: skip
        for b, (key, label, colour) in enumerate(bars):
            values = [rows[s][f"{kind}_{key}"] for s in SCENES]
            ax.bar(
                np.arange(3) + (b - 1.5) * 0.2, values, 0.2, label=label, color=colour
            )
        ax.set_xticks(np.arange(3), list(SCENES.values()))
        ax.set_title(title, fontsize=10)
        ax.set_ylabel(f"mean distance / GT depth (capped at {CAP})")
        style(ax)
    axes[1].legend(fontsize=8, frameon=False)
    fig.suptitle(
        "E52c probe: cross-view distance, 12 views, 6000 px. Point to point scores GT "
        "as high as Déjà View (sample spacing);\npoint to plane separates them 2-3x. "
        "Gate vs any differs little on Déjà View.",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(FIGURES / "e52c_probe.jpg", dpi=150)


def main() -> None:
    """Probe the 3 dev scenes, write the raw numbers and the figure."""
    cfg = yaml.safe_load(Path("configs/e52a_adapters.yaml").read_text(encoding="utf-8"))
    raw = yaml.safe_load(Path(cfg["eval"]).read_text(encoding="utf-8"))
    _, net = setup(cfg)
    rows = {name: probe_scene(net, raw, name) for name in SCENES}
    for name, row in rows.items():
        print(name, ", ".join(f"{k} {v:.4f}" for k, v in row.items()), flush=True)
    settings = {"views": VIEWS, "pixels": PIXELS, "cap": CAP, "strides": list(STRIDES)}
    RESULTS.write_text(
        yaml.safe_dump({"git": git_hash(), "settings": settings, "scenes": rows},
                       sort_keys=False),
        encoding="utf-8",
    )  # fmt: skip
    figure(rows)


if __name__ == "__main__":
    main()
