"""E69c rung 0: the formulation through the real scorer, without training.

For every scene of the named splits, in Deja View's frame and scored by `cloud_eval`:
- `refs`: `oracle-ovoxel-s<k>`, every child of the candidate voxels that holds a ground-truth
  point, at the mean of those points (what zero loss produces; candidates as in the
  generator, `reach` 1), and `dv-ovoxel-s<k>`, the children that hold a Deja View point at
  those points' mean (Deja View in this representation), for k in `SPLITS`; the child
  precision, recall and vertex error of `dv-ovoxel`, and the share of the GT-occupied
  children that the candidates contain, go to `runs/e69c/refs.json`;
- `density`: Deja View's pixels kept by the plain voxel density filter (E86a: every pixel in
  one instance) at q in `QUANTILES`, as clouds `e69c-density-q<q>`, so that `cloud_eval`
  scores them unchanged;
- `fine <split>`: the same map at the `FINE_ADDED` shares of added children, an extra the
  plan's grid does not have (its first step from 0 to 0.1 already ruins the Chamfer);
- `damage <split>`: the oracle damaged in a known way, scored by the surface half of
  `cloud_eval.cloud_scores` (the same two nearest-neighbour distances, `surface_scores`):
  a share r of its occupied children dropped, a share f of the empty candidate children
  added (vertex at the child centre), Gaussian noise of sigma child sizes on every vertex,
  over `DROPPED` x `ADDED` x `SIGMAS`, with common random numbers so that the map is
  monotone. Each variant records its measured child precision, recall and vertex error, the
  coordinates on which later rungs place a trained model.

uv run python -m experiments.scripts.e69c_oracle refs configs/e69c.yaml dev val
uv run python -m experiments.scripts.e69c_oracle density configs/e69c.yaml dev val
uv run python -m experiments.scripts.e69c_oracle damage configs/e69c.yaml 2 dev val
uv run python -m experiments.scripts.e69c_oracle fine configs/e69c.yaml 2 dev val
"""

import dataclasses
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree

from experiments.scripts.e69c_gate import QUANTILES, RAW, RUNS, SPLITS
from src.analysis.cloud_eval import apply_alignment
from src.analysis.filters import instance_voxel_density, keep_top, ranks
from src.analysis.scene_eval import load_inputs
from src.evaluation.labeled import align_output, surface_scores
from src.loaders.hypersim.config import load_config
from src.refinement.generator import (
    Children,
    GeneratorConfig,
    candidate_mask,
    child_corner,
    child_counts,
    child_scores,
    child_targets,
    load_rung,
    load_scene,
)
from src.refinement.multiview import points_from_rays
from src.refinement.volume import voxelize

DROPPED = (0.0, 0.1, 0.2, 0.3, 0.5)
ADDED = (0.0, 0.1, 0.2, 0.3, 0.5)
FINE_ADDED = (0.01, 0.02, 0.05)  # between the plan's 0 and 0.1, where the map turns
SIGMAS = (0.0, 0.1, 0.2, 0.3)
CPU = torch.device("cpu")


@dataclasses.dataclass(frozen=True)
class Frame:
    """A scene's grid in Deja View's frame: points = (lo + cells * cell) * scale."""

    lo: torch.Tensor
    cell: float
    scale: float


def children_cloud(
    voxel: torch.Tensor,
    kept: torch.Tensor,
    vertex: torch.Tensor,
    split: int,
    frame: Frame,
) -> np.ndarray:
    """Return the kept children's vertices (n, 3) float32 in Deja View's frame.

    voxel (M, 3) x y z; kept (M, K) bool; vertex (M, K, 3) in child units, may leave the child.
    """
    cells = voxel.float()[:, None] + child_corner(split) + vertex / split
    cloud = (frame.lo + cells[kept] * frame.cell) * frame.scale
    assert cloud.ndim == 2
    assert cloud.isfinite().all()
    return cloud.numpy().astype(np.float32)


def surface_chamfer(
    cloud: np.ndarray, truth: np.ndarray, truth_tree: cKDTree
) -> dict[str, float]:
    """Surface scores (cm) of an aligned cloud: `cloud_eval.cloud_scores`' two distances."""
    tree = cKDTree(cloud, balanced_tree=False, compact_nodes=False)
    comp = tree.query(truth, workers=-1)[0]
    acc = truth_tree.query(cloud, workers=-1)[0]
    return surface_scores(acc, comp)


def write_cloud(cfg: GeneratorConfig, name: str, label: str, cloud: np.ndarray) -> None:
    """Write a cloud as `cloud_eval`'s `cloud` variant `<label>` of the scene."""
    path = cfg.root / name / f"{label}_{cfg.views}_cloud.npz"
    np.savez_compressed(path, points=cloud)


def reference(
    cfg: GeneratorConfig, name: str, splits: tuple[int, ...]
) -> dict[int, dict]:
    """Write one scene's oracle and Deja View clouds per split; return its child scores."""
    pts, gt, has, scale = load_scene(cfg, name, "none")
    frame = Frame(pts.lo, pts.cell, scale)
    voxel = torch.nonzero(candidate_mask(voxelize(pts, semantics=False), cfg.reach))
    voxel = voxel.flip(1)
    gt_cells = (gt[has] - pts.lo) / pts.cell
    dv_cells = (pts.points - pts.lo) / pts.cell
    out = {}
    for split in splits:
        gt_occ, gt_vertex = child_targets(voxel, gt_cells, split, pts.dims)
        dv_occ, dv_vertex = child_targets(voxel, dv_cells, split, pts.dims)
        write_cloud(
            cfg,
            name,
            f"e69c-oracle-ovoxel-s{split}",
            children_cloud(voxel, gt_occ, gt_vertex, split, frame),
        )
        write_cloud(
            cfg,
            name,
            f"e69c-dv-ovoxel-s{split}",
            children_cloud(voxel, dv_occ, dv_vertex, split, frame),
        )
        corner = voxel.float()[:, None] + child_corner(split)
        centre = torch.where(
            dv_occ[..., None], dv_vertex, 0.5
        )  # a miss sits at the centre
        dv = Children(
            torch.where(dv_occ, 1.0, -1.0),
            corner + centre / split,
            corner,
            gt_occ,
            gt_vertex,
            torch.ones(len(voxel), dtype=torch.bool),
        )
        everything = torch.floor(gt_cells * split).long().unique(dim=0)
        empty = float((~gt_occ).sum())
        out[split] = child_scores(child_counts(dv, split)) | {
            "candidate_children": float(gt_occ.numel()),
            "occupied_children": float(gt_occ.sum()),
            "gt_children": float(len(everything)),
            "dv_added": float((dv_occ & ~gt_occ).sum() / max(empty, 1.0)),
        }
        print(f"{name} | s{split} | {out[split]}", flush=True)
    return out


def density(cfg: GeneratorConfig, name: str) -> None:
    """Write the plain voxel density filter's clouds of one scene at every quantile."""
    inputs = load_inputs(RAW, name, CPU)
    pixels = points_from_rays(inputs.scene.depth.double(), inputs.scene.rays.double())
    pixels = pixels.numpy()
    depth, k = inputs.scene.depth.numpy(), inputs.scene.k.numpy()
    count = instance_voxel_density(pixels, depth, k, np.full(depth.shape, -1))
    for q in QUANTILES:
        keep = keep_top(ranks(count), q).reshape(-1)
        write_cloud(
            cfg,
            name,
            f"e69c-density-q{q}",
            pixels.reshape(-1, 3)[keep].astype(np.float32),
        )
    print(f"{name}: density clouds at {QUANTILES}", flush=True)


def damage(
    cfg: GeneratorConfig, name: str, split: int, out: Path, added: tuple[float, ...]
) -> None:
    """Score the damaged oracle of one scene over the grid `added`; write `<name>.json`."""
    target = out / f"{name}.json"
    if target.exists():
        return
    pts, gt, has, scale = load_scene(cfg, name, "none")
    frame = Frame(pts.lo, pts.cell, scale)
    voxel = torch.nonzero(candidate_mask(voxelize(pts, semantics=False), cfg.reach))
    voxel = voxel.flip(1)
    occupied, vertex = child_targets(
        voxel, (gt[has] - pts.lo) / pts.cell, split, pts.dims
    )
    corner = voxel.float()[:, None] + child_corner(split)
    inputs = load_inputs(RAW, name, CPU)
    pixels = points_from_rays(inputs.scene.depth.double(), inputs.scene.rays.double())
    align = align_output(inputs.truth, pixels.numpy())[2]
    truth = inputs.truth.points
    truth_tree = cKDTree(truth, balanced_tree=False, compact_nodes=False)
    rng = np.random.default_rng(0)
    drawn = [torch.from_numpy(rng.random(occupied.shape)) for _ in range(2)]
    noise = torch.from_numpy(rng.normal(size=(*occupied.shape, 3))).float()
    rows = []
    for r, f, sigma in itertools.product(DROPPED, added, SIGMAS):
        t0 = time.perf_counter()
        kept = (occupied & (drawn[0] >= r)) | (~occupied & (drawn[1] < f))
        placed = torch.where(occupied[..., None], vertex, 0.5) + sigma * noise
        children = Children(
            torch.where(kept, 1.0, -1.0),
            corner + placed / split,
            corner,
            occupied,
            vertex,
            torch.ones(len(voxel), dtype=torch.bool),
        )
        cloud = children_cloud(voxel, kept, placed, split, frame).astype(np.float64)
        rows.append(
            {"r": r, "f": f, "sigma": sigma}
            | surface_chamfer(apply_alignment(cloud, align), truth, truth_tree)
            | child_scores(child_counts(children, split))
            | {"points": float(len(cloud))}
        )
        print(
            f"{name} | r {r} f {f} sigma {sigma} | chamfer {rows[-1]['chamfer_cm']:.3f} "
            f"| {time.perf_counter() - t0:.1f} s",
            flush=True,
        )
    target.write_text(json.dumps(rows))


def main() -> None:
    """Run `refs`, `density` or `damage` on the named splits; see the module docstring."""
    command, config, *rest = sys.argv[1:]
    cfg = load_rung(Path(config), "overfit1").cfg
    split = int(rest.pop(0)) if command in ("damage", "fine") else 0
    scenes = load_config(Path("configs/hypersim.yaml")).scenes(rest)
    names = [f"{s}_{c}" for s, c in scenes.items()]
    RUNS.mkdir(parents=True, exist_ok=True)
    if command == "refs":
        refs = {n: reference(cfg, n, SPLITS) for n in names}
        old = (
            json.loads((RUNS / "refs.json").read_text())
            if (RUNS / "refs.json").exists()
            else {}
        )
        (RUNS / "refs.json").write_text(json.dumps(old | refs, indent=1))
    elif command == "density":
        for n in names:
            density(cfg, n)
    else:
        assert command in ("damage", "fine"), command
        folder = RUNS / f"damage_s{split}{'_fine' if command == 'fine' else ''}"
        folder.mkdir(exist_ok=True)
        for n in names:
            damage(cfg, n, split, folder, FINE_ADDED if command == "fine" else ADDED)


if __name__ == "__main__":
    main()
