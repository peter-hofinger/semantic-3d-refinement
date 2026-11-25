"""E69g: E69e's rungs 1-3 with E69f's locked box and two-level candidates (issue #69).

`data` (rung 1): a 4-cell slab of the bedroom, seen along y in centimetres: Déjà View's
points, GT99v (seen / unseen), the coarse candidates (2^3 voxels within `coarse_reach`
coarse cells of Déjà View's), the fine candidates a perfect coarse stage passes on and their
occupied children's target vertices. Prints the shapes, ranges and counts it rests on.

`score` (rung 3): a trained model's whole-scene output (`generate`, its own gate, no GT) in
world metres, scored as E69f's rung 0 scores the baselines: voxelized at the child's size
against GT99v in the box (in-box) and, with Déjà View's out-of-box points added, against
all of it (pass-through); plus the oracle of the same candidates. Draws the slab of the
output and of Déjà View coloured by distance to GT99v on one centimetre scale.

uv run python -m experiments.scripts.e69g data|score [--model runs/.../<name>.pt]
"""

import argparse
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from experiments.scripts.e69e_rung0 import CM, score, to_world
from experiments.scripts.e69f_rung0 import Loaded, load, oracle
from src.device import best_device
from src.evaluation.labeled import nearest_distances
from src.refinement.generator import (
    GeneratorConfig,
    build,
    child_corner,
    child_targets,
    coarse_candidates,
    generate,
    load_rung,
    load_scene,
)

BEDROOM = "ai_006_007_cam_00"
CONFIG = Path("configs/e69f.yaml")
SLAB = 4  # cells along y
FIGURES = Path("experiments/figures")
VMAX = 10.0  # cm, top of the distance colour scale


def slab(cells: np.ndarray, y0: int) -> np.ndarray:
    """Return which points (n, 3), in cells, lie in the slab y0 <= y < y0 + SLAB."""
    return (cells[:, 1] >= y0) & (cells[:, 1] < y0 + SLAB)


def cm_of(s: Loaded, cells: np.ndarray) -> np.ndarray:
    """Return x and z (n, 2) of points in cells as centimetres from the grid's corner."""
    m = s.cell * s.scale * s.similarity[0]
    return CM * m * cells[:, [0, 2]]


def masks(s: Loaded, reach: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the occupied voxels, the coarse candidates (fine resolution) and the fine."""
    occupied = s.grid[0] > 0
    truth = torch.zeros_like(occupied)
    index = np.floor(s.truth_cells).astype(np.int64)
    keep = ((index >= 0) & (index < np.array(s.dims))).all(1)
    x, y, z = torch.from_numpy(index[keep]).unbind(1)
    truth[z, y, x] = True
    fine = coarse_candidates(occupied, reach, truth)
    wide = coarse_candidates(occupied, reach, torch.ones_like(truth))
    return occupied.numpy(), wide.numpy(), fine.numpy()


def data(s: Loaded, reach: int, split: int) -> list[str]:
    """Draw the rung-1 slab and return the checks it rests on."""
    occupied, wide, fine = masks(s, reach)
    voxel = torch.nonzero(torch.from_numpy(fine)).flip(1)
    occ, vertex = child_targets(
        voxel, torch.from_numpy(s.truth_cells).float(), split, s.dims
    )
    children = (voxel[:, None] + child_corner(split) + vertex / split)[occ].numpy()
    dv = (s.dv_norm - s.lo) / s.cell
    y0 = int(np.median(dv[:, 1])) - SLAB // 2
    panels = [
        ("Déjà View points", dv[slab(dv, y0)], None),
        (
            "GT99v: seen grey, unseen red",
            s.truth_cells[slab(s.truth_cells, y0)],
            "seen",
        ),
        (
            "coarse candidates (2^3 voxels)",
            np.argwhere(wide).astype(float)[:, ::-1] + 0.5,
            "vox",
        ),
        (
            "fine candidates + target vertices",
            np.argwhere(fine).astype(float)[:, ::-1] + 0.5,
            "kids",
        ),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(24, 6.5), sharex=True, sharey=True)
    for ax, (title, cloud, kind) in zip(axes, panels, strict=True):
        pts = cloud[slab(cloud, y0)] if kind in ("vox", "kids") else cloud
        xz = cm_of(s, pts)
        if kind == "seen":
            seen = s.seen[slab(s.truth_cells, y0)]
            ax.scatter(*xz[seen].T, s=0.2, c="0.6")
            ax.scatter(*xz[~seen].T, s=0.4, c="tab:red")
        else:
            ax.scatter(*xz.T, s=0.3 if kind else 0.2, c="tab:blue" if kind else "k")
        if kind == "kids":
            k = children[slab(children, y0)]
            ax.scatter(*cm_of(s, k).T, s=0.2, c="tab:orange")
        ax.set_title(title)
        ax.set_aspect("equal")
        ax.set_xlabel("x (cm, grid frame)")
    axes[0].set_ylabel("z (cm, grid frame)")
    fig.suptitle(
        f"E69g rung 1, bedroom: slab y {y0}-{y0 + SLAB} of {s.dims[1]} cells "
        f"(cell {CM * s.cell * s.scale * s.similarity[0]:.2f} cm); coarse reach {reach}"
    )
    fig.tight_layout()
    fig.savefig(FIGURES / "e69g_data.jpg", dpi=110)
    unseen = ~s.seen & s.inside
    covered_fine = fine[tuple(np.floor(s.truth_cells[unseen]).astype(int)[:, ::-1].T)]
    return [
        (
            f"grid {tuple(s.grid.shape)}, finite {bool(torch.isfinite(s.grid).all())}, "
            f"channel ranges {[(round(float(c.min()), 2), round(float(c.max()), 2)) for c in s.grid]}"
        ),
        (
            f"dims (x y z) {s.dims}; occupied {occupied.sum():,}; coarse candidates "
            f"{wide.sum() // 8:,} (x8 fine); fine candidates {fine.sum():,}"
        ),
        (
            f"children {occ.numel():,}, occupied {int(occ.sum()):,} "
            f"({100 * float(occ.float().mean()):.2f} %); vertices in [0, 1]: "
            f"{bool(((vertex >= 0) & (vertex <= 1)).all())}"
        ),
        (
            f"GT99v in box {s.inside.mean() * 100:.2f} %; in-box unseen inside a fine "
            f"candidate {100 * covered_fine.mean():.1f} %"
        ),
    ]


def distance_panel(ax: plt.Axes, s: Loaded, world: np.ndarray, title: str) -> float:
    """Scatter a slab of a world cloud coloured by distance to GT99v (cm); mean."""
    d = CM * nearest_distances(world, s.truth)
    cells = (
        (world - s.similarity[2]) @ s.similarity[1] / s.similarity[0] / s.scale - s.lo
    ) / s.cell
    y0 = int(np.median(((s.dv_norm - s.lo) / s.cell)[:, 1])) - SLAB // 2
    keep = slab(cells, y0)
    ax.scatter(
        *cm_of(s, cells[keep]).T, s=0.3, c=d[keep], vmin=0, vmax=VMAX, cmap="magma_r"
    )
    ax.set_title(f"{title}: mean {d.mean():.2f} cm to GT99v")
    ax.set_aspect("equal")
    ax.set_xlabel("x (cm, grid frame)")
    return float(d.mean())


def score_model(s: Loaded, path: Path) -> list[str]:
    """Score a trained model's output on one scene as rung 0 scores the baselines."""
    saved = yaml.safe_load(path.with_suffix(".yaml").read_text())["config"]
    cfg = GeneratorConfig(**(saved | {"root": Path(saved["root"])}))
    device = best_device()
    pts, _, _, scale = load_scene(cfg, s.name, cfg.semantics)
    model = build(cfg, pts.dims).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    world = to_world(
        generate(model, pts, scale, device, cfg).astype(np.float64), s.similarity
    )
    out_dv = s.dv[~s.dv_in]
    boxed = score(world, s.truth[s.inside], s.seen[s.inside], side=s.child_m)
    full = score(np.concatenate([world, out_dv]), s.truth, s.seen, side=s.child_m)
    dv_box = score(s.dv[s.dv_in], s.truth[s.inside], s.seen[s.inside], side=s.child_m)
    dv_full = score(s.dv, s.truth, s.seen, side=s.child_m)
    fine = masks(s, cfg.coarse_reach)[2]
    ceiling = score(
        oracle(s, fine)[0], s.truth[s.inside], s.seen[s.inside], voxel=False
    )
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5), sharex=True, sharey=True)
    distance_panel(axes[0], s, s.dv, "Déjà View")
    distance_panel(axes[1], s, world, path.stem)
    axes[0].set_ylabel("z (cm, grid frame)")
    fig.colorbar(
        axes[1].collections[0],
        ax=axes,
        label=f"distance to GT99v (cm, capped {VMAX:.0f})",
    )
    fig.suptitle(
        f"{s.name}: the same slab, Déjà View against {path.stem}; dark = far from GT99v"
    )
    fig.savefig(FIGURES / f"e69g_{path.stem}_{s.name[:10]}.jpg", dpi=110)

    def row(name: str, r: dict[str, float]) -> str:
        return (
            f"{name:24s} chamfer {r['chamfer']:7.3f}  acc {r['accuracy']:7.3f}  comp {r['completeness']:7.3f}"
            f"  seen {r['comp_seen']:7.3f}  unseen {r['comp_unseen']:7.3f}  points {r['points']:9,.0f}"
        )

    return [
        f"== {s.name} {path.stem} (cm, output voxelized at the child, {CM * s.child_m:.2f} cm)",
        "in-box:",
        row("Déjà View (baseline)", dv_box),
        row(path.stem, boxed),
        row("oracle of the candidates", ceiling),
        "pass-through:",
        row("Déjà View (baseline)", dv_full),
        row(path.stem + " + DV outside", full),
    ]


def main() -> None:
    """Run one part on the bedroom (or `--scenes`) and write its text."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["data", "score"])
    parser.add_argument("--model", type=Path)
    parser.add_argument("--scenes", nargs="+", default=[BEDROOM])
    args = parser.parse_args()
    cfg = load_rung(CONFIG, "overfit1").cfg
    lines: list[str] = []
    for name in args.scenes:
        s = load(cfg, name)
        if args.command == "data":
            lines += data(s, cfg.coarse_reach, cfg.split)
        else:
            assert args.model is not None, "score needs --model"
            lines += score_model(s, args.model)
    tag = args.command if args.command == "data" else f"score_{args.model.stem}"
    Path("experiments/results", f"e69g_{tag}.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
