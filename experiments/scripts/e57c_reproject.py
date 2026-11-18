"""E57c: confidence-kept points of other views, reprojected, as white seeds in the red areas, then the straight-run rule.

Categories as in `e57c_two_fills` (white kept by confidence, green oracle beyond 8 cm, blue oracle-kept not brought back, red dropped by both, gold brought
back by the straight-run rule). Seeds: for each target view v, every white pixel of the other views with an instance id >= 0 is projected (aligned 3D point,
the GT camera pose, pixel = floor of the projection, one pixel per point, no radius, no depth test); when it lands on a RED pixel of v with the same instance
id, that pixel is a seed. The straight-run rule then counts seeds as white: blue pixels between a white or a seed pixel and a green pixel on a row, column or
diagonal of one view, in one instance, come back. A: the seeds are used by the rule only; B: the seeds (view v's own red points there) are kept too. Prints the seeds, the blue pixels brought back, the share of the
squiggle's strip brought back and, per variant, the kept share, the Chamfer and the gap pixels of the pulled-back render's zone that the oracle-alone cloud covers.

uv run python -m experiments.scripts.e57c_reproject > experiments/results/e57c_reproject.txt
"""

from pathlib import Path

import numpy as np
import torch
import yaml

from experiments.scripts.e57c_hybrid import far_from
from experiments.scripts.e57c_oracle_curve import evaluate, greedy, rank_scores
from experiments.scripts.e57c_seams import DELTA, LEVELS, drop_order
from experiments.scripts.e57c_squiggles import ZONE, render_ids
from experiments.scripts.e57c_two_fills import (
    BLUE,
    GOLD,
    GREEN,
    RED,
    WHITE,
    runs_on_lines,
)
from src.analysis.scene_eval import SceneInputs, load_inputs
from src.analysis.scene_figures import (
    PRESENTATION_SCENE,
    PRESENTATION_VIEWS,
    render_cameras,
)


def project(
    points: np.ndarray, cam_to_world: np.ndarray, k: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return the pixel coordinates (u, v) of the world points (N, 3) in a view, and whether they are in front of it."""
    cam = (points - cam_to_world[:3, 3]) @ cam_to_world[:3, :3]
    z = cam[:, 2]
    safe = np.where(z > 1e-6, z, 1.0)
    return np.stack(
        [k[0, 0] * cam[:, 0] / safe + k[0, 2], k[1, 1] * cam[:, 1] / safe + k[1, 2]],
        axis=1,
    ), z > 1e-6


def seeds_of(
    category: np.ndarray, ids: np.ndarray, aligned: np.ndarray, inputs: SceneInputs
) -> np.ndarray:
    """Return the seed pixels (V, H, W): red pixels hit by a reprojected white point of the other views with the same instance id."""
    n, h, w = category.shape
    white = (category == WHITE) & (ids >= 0)
    seeds = np.zeros(category.shape, dtype=bool)
    for v in range(n):
        source = white.copy()
        source[v] = False
        uv, front = project(
            aligned[source], inputs.views.cam_to_world[v], inputs.views.intrinsics[v]
        )
        x, y = np.floor(uv[:, 0]).astype(int), np.floor(uv[:, 1]).astype(int)
        ok = front & (x >= 0) & (x < w) & (y >= 0) & (y < h)
        x, y, ident = x[ok], y[ok], ids[source][ok]
        hit = (category[v][y, x] == RED) & (ids[v][y, x] == ident)
        seeds[v][y[hit], x[hit]] = True
    return seeds


def connect(
    category: np.ndarray, ids: np.ndarray, extra_white: np.ndarray | None = None
) -> np.ndarray:
    """Return the blue pixels (V, H, W) on a straight run closed by white (or a seed) and green; `extra_white` pixels count as white."""
    shown = category if extra_white is None else np.where(extra_white, WHITE, category)
    return np.stack([runs_on_lines(shown[v], ids[v]) for v in range(len(ids))])


def main() -> None:
    """Apply the rule and print the checks."""
    raw = yaml.safe_load(Path("experiments/configs/e57c_oracle.yaml").read_text())
    inputs = load_inputs(raw, PRESENTATION_SCENE, torch.device("cpu"))
    truth, pixels = inputs.truth, inputs.truth.pixels
    assert pixels is not None
    points, align, pred, a, _, ranking = rank_scores(inputs)
    v_conf, v_oracle = drop_order(
        ranking["confidence"].astype(np.float64), greedy(truth, pred, a, LEVELS)
    )
    base, oracle = (1 - v_conf) >= 0.5, v_oracle <= 0.5
    gate = np.where(
        base,
        WHITE,
        np.where(oracle, np.where(far_from(pred, base, DELTA), GREEN, BLUE), RED),
    )
    category = np.full(pixels.shape, WHITE)
    category[pixels] = gate
    ids = inputs.labels.instance_ids
    aligned = points @ align[:3, :3].T + align[:3, 3]
    seeds = seeds_of(category, ids, aligned, inputs)
    print(
        f"seeds per view: {seeds.sum(axis=(1, 2)).tolist()} (red pixels per view {(category == RED).sum(axis=(1, 2)).tolist()})"
    )
    back0, back1 = connect(category, ids), connect(category, ids, seeds)
    print(
        f"blue pixels brought back: without seeds {back0.sum()}, with seeds {back1.sum()} (blue in total {(category == BLUE).sum()})"
    )
    after = np.where(back1, GOLD, category)
    kept0 = np.isin(np.where(back0, GOLD, category), [WHITE, GREEN, GOLD])
    kept_a = np.isin(after, [WHITE, GREEN, GOLD])
    kept_b = kept_a | seeds
    oracle_full = np.ones(category.shape, dtype=bool)
    oracle_full[pixels] = oracle
    union = np.ones(category.shape, dtype=bool)
    union[pixels] = base | oracle

    # the squiggle: gap pixels of the pulled-back render in the zone that the oracle-alone cloud covers
    camera, _ = render_cameras(inputs, PRESENTATION_VIEWS)["view 18, pulled back"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    zone = np.zeros(camera.size, dtype=bool)
    zone[ZONE] = True
    ids_oracle = render_ids(inputs, aligned, oracle_full, camera, device)
    ids_blue0 = render_ids(inputs, aligned, category == BLUE, camera, device)
    ids_kept0 = render_ids(inputs, aligned, kept0, camera, device)
    strip = np.unique(ids_blue0[(ids_kept0 < 0) & (ids_blue0 >= 0) & zone])
    print(
        f"the strip of the baseline: {len(strip)} source pixels; brought back: baseline {np.isin(after.reshape(-1)[strip], [GOLD]).mean():.3f} (with seeds), {(np.where(back0, GOLD, category).reshape(-1)[strip] == GOLD).mean():.3f} (without)"
    )
    cuts = {
        "fusion without seeds": kept0,
        "A: seeds as white for the rule": kept_a,
        "B: seeds kept too": kept_b,
        "union, no gate": union,
    }
    for name, kept in cuts.items():
        gap = (
            (render_ids(inputs, aligned, kept, camera, device) < 0)
            & (ids_oracle >= 0)
            & zone
        ).sum()
        print(
            f"{name}: kept {kept.mean():.3f}, Chamfer {evaluate(truth, points, kept)[2]:.3f} cm, zone gap pixels covered by the oracle-alone cloud {gap}"
        )


if __name__ == "__main__":
    main()
