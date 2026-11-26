"""E69i: why the O-Voxel arm completes nothing unseen on the bedroom (issue #69).

Three diagnostics of E69g's rung 3b, none of which trains on more than one crop:

`autopsy --model` (E2's children head): the logits of the children of every coarse candidate
that holds GT99v (the cells the loss is masked to, teacher-forced as in training), split by
coarse cell into those holding seen GT and those holding only unseen GT: positive rate, ROC-AUC
(Mann-Whitney), average precision (PR-AUC) and logit quantiles, once from one forward pass of
the whole grid (as `generate` runs) and once tiled into 32-cell crops scoring their 16-cell
interiors (as training runs): GroupNorm normalizes over whatever it is given, so the two
differ. Also the occupied share of the children of a GT-holding coarse cell against a flat
plane's (a surface fills about 1 / 8 of an 8^3 block of children, not a quarter to a half).

`crop --model`: the fixed crop whose interior holds the most in-box unseen GT voxels 8-13 cells
(Euclidean) from every voxel Déjà View occupies, inside the H6 fine candidates; how many voxels
Déjà View occupies in it and in the U-Net's receptive field of those targets (theoretical, and
effective: input voxels with a gradient above 1 % of the largest, on E2); then 500 steps of
E69g's one-crop overfit on it and, for reference, on E69g's Déjà View-rich crop, with child
precision and recall split into seen and unseen children. `direct_crop` draws its window from
`crop_window`, which is swapped for a fixed window here.

`visits`: crop draws only, no model: how often each voxel lies in a scored interior over 4 800
draws, for occupied unseen and seen GT voxels and for empty candidate voxels, under candidate-
and point-centred crops (H6) and candidate-centred crops on the latent's R12 candidates.

uv run python -m experiments.scripts.e69i autopsy|crop|visits [--model runs/.../<name>.pt]
"""

import argparse
import dataclasses
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from scipy.ndimage import distance_transform_edt, uniform_filter

from experiments.scripts.e69f_rung0 import Loaded, load
from experiments.scripts.e69g import BEDROOM, CONFIG
from src.device import best_device
from src.refinement import generator as gen
from src.refinement.generator import (
    GeneratorConfig,
    build,
    child_targets,
    coarse_candidates,
    decode_children,
    dilate,
    direct_terms,
    for_arm,
    load_rung,
    load_scenes,
    to_device,
)
from src.refinement.volume import voxelize

RESULTS = Path("experiments/results")
FAR = (8.0, 13.0)  # cells from Déjà View: E69f's C3, where the missing surface lies
STEPS = 4800
OVERFIT = 500


def occupancy(s: Loaded, mask: np.ndarray | None = None) -> torch.Tensor:
    """Return the voxels (Z, Y, X) holding a GT99v point (of `mask`, if given)."""
    cells = s.truth_cells if mask is None else s.truth_cells[mask]
    index = np.floor(cells).astype(np.int64)
    index = index[((index >= 0) & (index < np.array(s.dims))).all(1)]
    out = torch.zeros(s.dims[::-1], dtype=torch.bool)
    out[tuple(torch.from_numpy(index[:, ::-1].copy()).T)] = True
    return out


def pooled(mask: torch.Tensor) -> torch.Tensor:
    """Return the 2^3 max-pool of a (Z, Y, X) mask, as bool."""
    return torch.nn.functional.max_pool3d(mask.float()[None, None], 2)[0, 0] > 0


def up(mask: torch.Tensor) -> torch.Tensor:
    """Repeat a coarse (Z/2, Y/2, X/2) mask onto its 2^3 fine voxels."""
    return mask.repeat_interleave(2, 0).repeat_interleave(2, 1).repeat_interleave(2, 2)


def roc_auc(score: np.ndarray, label: np.ndarray) -> float:
    """Return the probability that a random positive outranks a random negative (ties half)."""
    order = np.argsort(score, kind="stable")
    ranks = np.empty(len(score))
    ranks[order] = np.arange(1, len(score) + 1)
    _, inverse, counts = np.unique(score, return_inverse=True, return_counts=True)
    first = np.cumsum(counts) - counts
    ranks = first[inverse] + (counts[inverse] + 1) / 2
    n1, n0 = label.sum(), (~label).sum()
    return float((ranks[label].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def average_precision(score: np.ndarray, label: np.ndarray) -> float:
    """Return the area under the precision-recall curve as the mean precision at each positive."""
    hit = label[np.argsort(-score, kind="stable")]
    precision = np.cumsum(hit) / np.arange(1, len(hit) + 1)
    return float(precision[hit].mean())


def stratum(name: str, logit: np.ndarray, label: np.ndarray) -> str:
    """Return one row: children, positive rate, AUCs, recall at 0 and -1, logit quantiles."""
    q = (10, 50, 90)
    pos, neg = np.percentile(logit[label], q), np.percentile(logit[~label], q)
    return (
        f"{name:34s} children {len(label):10,}  positive {100 * label.mean():5.2f} %  "
        f"ROC-AUC {roc_auc(logit, label):.3f}  PR-AUC {average_precision(logit, label):.3f}"
        f"  recall@0 {(logit[label] >= 0).mean():.3f}  recall@-1 {(logit[label] >= -1).mean():.3f}"
        f"  occupied p10/50/90 {pos.round(2).tolist()}  empty {neg.round(2).tolist()}"
    )


def load_model(path: Path) -> tuple[GeneratorConfig, torch.nn.Module, torch.device]:
    """Return a trained model in eval mode, its config and device."""
    saved = yaml.safe_load(path.with_suffix(".yaml").read_text())["config"]
    cfg = GeneratorConfig(**(saved | {"root": Path(saved["root"])}))
    device = best_device()
    s_pts = gen.load_scene(cfg, BEDROOM, cfg.semantics)[0]
    model = build(cfg, s_pts.dims).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    return cfg, model.eval(), device


def child_logits(
    model: torch.nn.Module, grid: torch.Tensor, candidates: torch.Tensor
) -> np.ndarray:
    """Return the logits (M, K) of the candidates' children from one forward pass of `grid`."""
    with torch.no_grad():
        fine = model.features(grid)
        out = decode_children(model.head, fine, candidates, model.corner)
    return out.logit.cpu().numpy()


def tiled_logits(
    model: torch.nn.Module,
    grid: torch.Tensor,
    candidates: torch.Tensor,
    cfg: GeneratorConfig,
) -> np.ndarray:
    """Return the logits, each from the 32-cell crop whose interior holds its voxel."""
    row = torch.full(candidates.shape, -1, dtype=torch.long)
    row[candidates] = torch.arange(int(candidates.sum()))
    k = cfg.split**3
    out = np.full((int(candidates.sum()), k), np.nan, dtype=np.float32)
    dims = np.array(candidates.shape)  # z y x
    size, margin = cfg.crop, cfg.margin
    starts = [
        sorted(
            {min(int(v), int(d) - size) for v in range(0, int(d), size - 2 * margin)}
        )
        for d in dims
    ]
    with torch.no_grad():
        for z in starts[0]:
            for y in starts[1]:
                for x in starts[2]:
                    lo = np.array([z, y, x])
                    own = [
                        slice(
                            0 if a == 0 else a + margin,
                            int(d) if a + size >= d else a + size - margin,
                        )
                        for a, d in zip(lo, dims, strict=True)
                    ]
                    window = tuple(slice(a, a + size) for a in lo)
                    local = torch.zeros(size, size, size, dtype=torch.bool)
                    inner = tuple(
                        slice(o.start - a, o.stop - a)
                        for o, a in zip(own, lo, strict=True)
                    )
                    local[inner] = candidates[tuple(own)]
                    if not local.any():
                        continue
                    fine = model.features(grid[(slice(None), *window)])
                    logit = decode_children(
                        model.head, fine, local.to(grid.device), model.corner
                    ).logit
                    out[row[tuple(own)][candidates[tuple(own)]].numpy()] = (
                        logit.cpu().numpy()
                    )
    assert np.isfinite(out).all(), "a candidate no crop's interior holds"
    return out


def plane_share(split: int, rng: np.random.Generator, n: int = 2000) -> float:
    """Return the mean occupied share of the (2 split)^3 children of a coarse cell a random plane cuts."""
    side = 2 * split
    grid = (np.indices((side,) * 3).reshape(3, -1).T + 0.5) / side
    shares = []
    while len(shares) < n:
        normal = rng.normal(size=3)
        normal /= np.linalg.norm(normal)
        offset = rng.uniform(-1, 1) * np.abs(normal).sum()
        # a child is occupied if the plane passes through it: |n.(c - p)| <= half its extent
        reach = np.abs(normal).sum() / (2 * side)
        hit = np.abs((grid - 0.5) @ normal - offset / 2) <= reach
        if hit.any():
            shares.append(hit.mean())
    return float(np.mean(shares))


def autopsy(s: Loaded, path: Path) -> list[str]:
    """Return the rows of E2's children head, split seen / unseen, whole-grid and crop-tiled."""
    cfg, model, device = load_model(path)
    pts = gen.load_scene(cfg, BEDROOM, cfg.semantics)[0]
    grid = voxelize(pts, semantics=model.semantics).to(device)
    occupied = grid[0].cpu() > 0
    truth, seen = occupancy(s), occupancy(s, s.seen)
    held, held_seen = pooled(truth), pooled(seen)
    wide = dilate(pooled(occupied), cfg.coarse_reach)
    on = wide & held
    assert torch.equal(up(on), coarse_candidates(occupied, cfg.coarse_reach, truth))
    candidates = up(on)
    voxel = torch.nonzero(candidates).flip(1)
    target, _ = child_targets(
        voxel, torch.from_numpy(s.truth_cells).float(), cfg.split, s.dims
    )
    label = target.numpy()
    seen_child, _ = child_targets(
        voxel, torch.from_numpy(s.truth_cells[s.seen]).float(), cfg.split, s.dims
    )
    parent = voxel.flip(1) // 2
    in_seen = held_seen[tuple(parent.T)].numpy()
    t0 = time.perf_counter()
    whole = child_logits(model, grid, candidates.to(device))
    t1 = time.perf_counter()
    tiled = tiled_logits(model, grid, candidates, cfg)
    t2 = time.perf_counter()
    coarse_index = (
        (parent[:, 0] * on.shape[1] + parent[:, 1]) * on.shape[2] + parent[:, 2]
    ).numpy()
    _, cell_of = np.unique(coarse_index, return_inverse=True)
    cell_share = np.bincount(cell_of, label.sum(1)) / (8 * label.shape[1])
    lines = [
        (
            f"== {s.name} {path.stem}: children of the {int(on.sum()):,} coarse candidates "
            f"holding GT99v (of {int(held.sum()):,} GT-holding coarse cells, "
            f"{100 * float(on.sum() / held.sum()):.1f} % within coarse reach "
            f"{cfg.coarse_reach}); forward {t1 - t0:.1f} s whole, {t2 - t1:.1f} s tiled"
        ),
        (
            f"positive rate {100 * label.mean():.2f} % of {label.size:,} children; per "
            f"coarse cell: mean {100 * cell_share.mean():.2f} %, median "
            f"{100 * np.median(cell_share):.2f} %, p90 {100 * np.percentile(cell_share, 90):.2f} %;"
            f" a random flat plane through a coarse cell: {100 * plane_share(cfg.split, np.random.default_rng(0)):.2f} %"
        ),
        (
            f"coarse cells holding seen GT {int((on & held_seen).sum()):,}, only unseen "
            f"{int((on & ~held_seen).sum()):,}; occupied children holding a seen point "
            f"{int(seen_child.sum()):,} of {int(target.sum()):,}"
        ),
    ]
    for name, logit in (
        ("whole grid (as generate)", whole),
        ("32-cell crops (as training)", tiled),
    ):
        lines.append(f"-- logits from {name}")
        lines.append(
            stratum("all GT-holding coarse cells", logit.ravel(), label.ravel())
        )
        for tag, cells in (
            ("seen coarse cells", in_seen),
            ("unseen-only coarse cells", ~in_seen),
        ):
            lines.append(stratum(tag, logit[cells].ravel(), label[cells].ravel()))
        unseen_child = label & ~seen_child.numpy()
        lines.append(
            f"{'occupied children by own points':34s} seen: recall@0 "
            f"{(logit[seen_child.numpy()] >= 0).mean():.3f}, median logit "
            f"{np.median(logit[seen_child.numpy()]):.2f}; unseen-only: recall@0 "
            f"{(logit[unseen_child] >= 0).mean():.3f}, median logit {np.median(logit[unseen_child]):.2f}"
        )
    diff = tiled - whole
    lines.append(
        f"tiled - whole logit: mean {diff.mean():.3f}, sd {diff.std():.3f}, "
        f"|diff| p50/p90 {np.percentile(np.abs(diff), [50, 90]).round(3).tolist()}"
    )
    return lines


def far_crop(
    s: Loaded, cfg: GeneratorConfig
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the crop corner (x y z) richest in far unseen GT, that mask, DV distances."""
    occupied = s.grid[0] > 0
    truth = occupancy(s)
    unseen_only = truth & ~occupancy(s, s.seen)
    fine = coarse_candidates(occupied, cfg.coarse_reach, truth)
    dist = distance_transform_edt(~occupied.numpy())
    far = unseen_only.numpy() & fine.numpy() & (dist >= FAR[0]) & (dist <= FAR[1])
    inner = cfg.crop - 2 * cfg.margin
    density = uniform_filter(far.astype(np.float32), size=inner, mode="constant")
    centre = np.array(np.unravel_index(np.argmax(density), far.shape))[::-1]  # x y z
    lo = np.clip(centre - cfg.crop // 2, 0, np.array(s.dims) - cfg.crop)
    return lo - lo % 2, far, dist


def receptive_radius(levels: int) -> int:
    """Theoretical receptive radius (fine cells) of `UNet3d` with `levels` widths.

    An upper bound: encoder level i, two 3^3 convolutions at stride 2^i (2^(i+1) cells)
    after a 2^3 average pool (2^(i-1)); decoder level i, trilinear upsampling from level
    i+1 (one coarse cell, 2^(i+1)) and two convolutions (2^(i+1)). Three levels: 29.
    GroupNorm's statistics span the whole input on top, so every voxel has some say.
    """
    down = sum(2 ** (i + 1) for i in range(levels))
    down += sum(2 ** (i - 1) for i in range(1, levels))
    rise = sum(2 ** (i + 2) for i in range(levels - 1))
    return down + rise


def window_of(lo: np.ndarray, size: int) -> tuple[slice, ...]:
    """Return the (C, Z, Y, X) index of a cubic crop with corner lo (x y z)."""
    return (slice(None), *(slice(int(lo[a]), int(lo[a]) + size) for a in (2, 1, 0)))


def audit(s: Loaded, cfg: GeneratorConfig, path: Path) -> tuple[list[str], np.ndarray]:
    """Count the far crop's Déjà View voxels, in the crop and in the receptive field."""
    lo, far, dist = far_crop(s, cfg)
    size, margin = cfg.crop, cfg.margin
    window = window_of(lo, size)
    dims = np.array(s.dims)
    inner = tuple(  # z y x; as `direct_crop`, the rim counts at the grid's border
        slice(
            0 if lo[a] == 0 else margin,
            size if lo[a] + size >= dims[a] else size - margin,
        )
        for a in (2, 1, 0)
    )
    occupied = (s.grid[0] > 0).numpy()[window[1:]]
    targets = np.zeros_like(occupied)
    targets[inner] = far[window[1:]][inner]
    radius = receptive_radius(len(cfg.widths))
    near = distance_transform_edt(~targets, sampling=1) <= radius
    e2cfg, model, device = load_model(path)
    grid = voxelize(gen.load_scene(e2cfg, BEDROOM, e2cfg.semantics)[0], semantics=False)
    sub = grid[window].to(device).requires_grad_()
    logit = decode_children(
        model.head,
        model.features(sub),
        torch.from_numpy(targets).to(device),
        model.corner,
    ).logit
    logit.sum().backward()
    grad = sub.grad.abs().sum(0).cpu().numpy()
    effective = grad >= 0.01 * grad.max()
    planar = np.argwhere(targets).astype(float)
    spread = np.sqrt(np.linalg.eigvalsh(np.cov(planar.T)))
    lines = [
        (
            f"== far crop of {s.name}: corner (x y z) {lo.tolist()}, {size}^3 cells, interior "
            f"{size - 2 * margin}^3"
        ),
        (
            f"far unseen GT voxels ({FAR[0]:.0f}-{FAR[1]:.0f} cells from Déjà View, in H6 fine "
            f"candidates) in the interior: {int(targets.sum()):,}; their distance to Déjà View "
            f"median {np.median(dist[window[1:]][targets]):.1f} cells; spread along their "
            f"principal axes {spread.round(1).tolist()} cells (a wall: one small)"
        ),
        (
            f"Déjà View voxels in the crop {int(occupied.sum()):,} of {occupied.size:,} "
            f"({100 * occupied.mean():.2f} %), in the interior {int(occupied[inner].sum()):,}"
        ),
        (
            f"theoretical receptive radius {radius} cells (widths {cfg.widths}): Déjà View "
            f"voxels within it of a target, inside the crop {int((occupied & near).sum()):,}"
        ),
        (
            f"effective field on {path.stem} (|d logits / d input| >= 1 % of the largest): "
            f"{int(effective.sum()):,} voxels, of them Déjà View's {int((occupied & effective).sum()):,};"
            f" share of the gradient on Déjà View's voxels {grad[occupied].sum() / grad.sum():.3f}"
        ),
    ]
    return lines, lo


def overfit(
    cfg: GeneratorConfig,
    lo: np.ndarray,
    s: Loaded,
    device: torch.device,
    steps: int,
) -> list[str]:
    """Return E69g's one-crop overfit on the crop at `lo`; recall split into seen / unseen.

    A child counts as seen if a seen GT99v point lies in it, unseen if only unseen ones do.
    """
    scene = load_scenes(cfg, cfg.train)[0]
    torch.manual_seed(0)
    model = build(cfg, scene.pts.dims).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    on = to_device(scene, device)
    grid = gen.scene_grid(model, on[0])
    size = np.minimum(cfg.crop, scene.pts.dims)
    drawn = gen.crop_window
    gen.crop_window = lambda *_: (grid[window_of(lo, cfg.crop)], lo, size)
    seen = torch.from_numpy(s.truth_cells[s.seen] - lo).float()
    lines, t0 = [], time.perf_counter()
    try:
        for step in range(steps + 1):
            model.train()
            children, _ = gen.direct_crop(
                model, grid, on, cfg, np.random.default_rng(3)
            )
            terms = direct_terms(children, cfg.split)
            if step % (steps // 5) == 0:
                voxel = (children.corner[:, 0] - model.corner[0]).round().long()
                held, _ = child_targets(
                    voxel.cpu(), seen, cfg.split, tuple(int(v) for v in size)
                )
                live = children.scored.cpu()
                kept = (children.logit > 0).cpu()[live]
                occ = children.occupied.cpu()[live]
                by_seen, by_unseen = occ & held[live], occ & ~held[live]
                lines.append(
                    f"step {step:4d} {time.perf_counter() - t0:4.0f} s  "
                    + "  ".join(
                        f"{k} {float(v.detach()):.3f}" for k, v in terms.items()
                    )
                    + f"  positive {float(occ.float().mean()):.3f}  precision "
                    f"{float((kept & occ).sum() / kept.sum().clamp(min=1)):.3f}  recall "
                    f"{float((kept & occ).sum() / occ.sum()):.3f}  seen "
                    f"{float((kept & by_seen).sum() / by_seen.sum().clamp(min=1)):.3f} "
                    f"({int(by_seen.sum())})  unseen "
                    f"{float((kept & by_unseen).sum() / by_unseen.sum().clamp(min=1)):.3f} "
                    f"({int(by_unseen.sum())})"
                )
            loss = sum(terms.values())
            opt.zero_grad()
            loss.backward()
            opt.step()
    finally:
        gen.crop_window = drawn
    return lines


def crop(s: Loaded, cfg: GeneratorConfig, path: Path, steps: int) -> list[str]:
    """Audit the far crop, then overfit it and E69g's Déjà View-rich reference crop."""
    lines, lo = audit(s, cfg, path)
    device = best_device()
    pts = gen.load_scene(cfg, BEDROOM, cfg.semantics)[0]
    reference = gen.crop_window(pts, s.grid, cfg, np.random.default_rng(3))[1]
    window = window_of(reference, cfg.crop)
    lines.append(
        f"reference crop (E69g's, rng 3): corner {reference.tolist()}, Déjà View voxels "
        f"{int((s.grid[0] > 0).numpy()[window[1:]].sum()):,}"
    )
    for name, corner in (("far unseen crop", lo), ("reference crop", reference)):
        lines.append(f"-- {steps} steps on the {name}, {cfg.widths}, rate {cfg.lr:g}")
        lines += overfit(cfg, corner, s, device, steps)
    return lines


def visit_counts(
    s: Loaded, cfg: GeneratorConfig, positive: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Return the scored-interior visits (Z, Y, X) of each voxel over `STEPS` drawn crops.

    As `direct_crop`: the rim is cut only where the crop has neighbours, and a crop whose
    interior holds no trainable positive (`positive`) is drawn again.
    """
    pts = gen.load_scene(cfg, BEDROOM, "none")[0]
    count = np.zeros(s.dims[::-1], dtype=np.int32)
    zone = {}
    plain = gen.dilate

    def cached(mask: torch.Tensor, reach: int) -> torch.Tensor:
        if reach not in zone:
            zone[reach] = plain(mask, reach)
        return zone[reach]

    gen.dilate = cached
    dims = np.array(s.dims)
    try:
        drawn = 0
        while drawn < STEPS:
            _, lo, size = gen.crop_window(pts, s.grid, cfg, rng)
            start = np.where(lo > 0, lo + cfg.margin, 0)
            stop = np.where(lo + size < dims, lo + size - cfg.margin, dims)
            region = tuple(slice(int(start[a]), int(stop[a])) for a in (2, 1, 0))
            if positive[region].any():
                count[region] += 1
                drawn += 1
    finally:
        gen.dilate = plain
    return count


def summary(name: str, visits: np.ndarray) -> str:
    """Return the mean, median, 10th percentile and share never visited of some voxels' visits."""
    return (
        f"{name:42s} voxels {len(visits):9,}  visits mean {visits.mean():6.1f}  median "
        f"{np.median(visits):5.0f}  p10 {np.percentile(visits, 10):4.0f}  never "
        f"{100 * (visits == 0).mean():5.1f} %"
    )


def visits(s: Loaded, cfg: GeneratorConfig) -> list[str]:
    """Return the visits per voxel class under E1c's, E2's and the latent's crop draws."""
    occupied = s.grid[0] > 0
    truth, seen = occupancy(s), occupancy(s, s.seen)
    fine = coarse_candidates(occupied, cfg.coarse_reach, truth).numpy()
    wide = up(dilate(pooled(occupied), cfg.coarse_reach)).numpy()
    r12 = dilate(occupied, 12).numpy()
    truth, seen = truth.numpy(), seen.numpy()
    dist = distance_transform_edt(~occupied.numpy())
    unseen = truth & ~seen & fine
    classes = {
        "occupied unseen-only GT (H6 fine cand.)": unseen,
        "  of them 8-13 cells from Déjà View": unseen
        & (dist >= FAR[0])
        & (dist <= FAR[1]),
        "occupied seen GT (H6 fine cand.)": seen & fine,
        "empty H6 coarse-candidate voxels": wide & ~truth,
    }
    lines = [f"== {s.name}: scored-interior visits over {STEPS:,} crops (seed 0)"]
    runs = (
        (
            "H6, candidate-centred (E1c)",
            dataclasses.replace(cfg, crop_centre="candidate"),
            truth & fine,
        ),
        ("H6, point-centred (E1, E2, E3)", cfg, truth & fine),
    )
    for name, variant, positive in runs:
        count = visit_counts(s, variant, positive, np.random.default_rng(0))
        lines.append(f"-- {name}")
        lines += [summary(k, count[m]) for k, m in classes.items()]
    latent = for_arm(load_rung(CONFIG, "overfit1a"), "latent-gt99")
    latent = dataclasses.replace(latent, crop_centre="candidate")
    count = visit_counts(s, latent, truth & r12, np.random.default_rng(0))
    lines.append("-- R12, candidate-centred (rung 3a's latent)")
    lines.append(summary("R12 candidate voxels", count[r12]))
    lines.append(summary("occupied GT in R12 candidates", count[truth & r12]))
    return lines


def untouched(s: Loaded, path: Path) -> list[str]:
    """Count a latent table's codes still bit-identical to their seed-0 start.

    Adam leaves a code that never had a gradient exactly where `build` put it, so the
    count is exact; split into R12 candidates, occupied GT voxels among them and the rest.
    """
    cfg, model, _ = load_model(path)
    torch.manual_seed(int(path.stem.rsplit("-s", 1)[1]))
    start = build(cfg, s.dims).table.detach()
    same = (model.table.detach().cpu() == start).all(0).numpy()
    r12 = dilate(s.grid[0] > 0, cfg.reach).numpy()
    truth = occupancy(s).numpy()
    return [
        f"== {path.stem}: codes never updated (bit-identical to seed {path.stem[-1]})",
        (
            f"R12 candidate voxels {100 * same[r12].mean():.2f} % of {int(r12.sum()):,}; "
            f"occupied GT among them {100 * same[r12 & truth].mean():.2f} %; "
            f"outside the candidates {100 * same[~r12].mean():.2f} %"
        ),
    ]


def main() -> None:
    """Run one diagnostic on the bedroom and write its text."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["autopsy", "crop", "visits", "untouched"])
    parser.add_argument("--model", type=Path)
    parser.add_argument(
        "--steps", type=int, default=OVERFIT, help="crop: overfit steps"
    )
    args = parser.parse_args()
    tail = f"_{args.steps}" if args.steps != OVERFIT else ""
    cfg = for_arm(load_rung(CONFIG, "overfit1"), "ovoxel-gt99")
    s = load(cfg, BEDROOM)
    if args.command == "autopsy":
        lines = autopsy(s, args.model)
    elif args.command == "untouched":
        lines = untouched(s, args.model)
    elif args.command == "crop":
        lines = crop(s, cfg, args.model, args.steps)
    else:
        lines = visits(s, cfg)
    (RESULTS / f"e69i_{args.command}{tail}.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
