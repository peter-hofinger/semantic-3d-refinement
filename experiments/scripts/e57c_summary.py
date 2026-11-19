"""E57c: all approaches of the oracle fusion scored on a split, with the stage times of each (per scene, CPU).

Approaches: Déjà View whole; confidence only (q 0.5); instance voxel density (q 0.6); greedy oracle (q 0.5, 10% rounds); one-shot marginal oracle (q 0.6); the union of the confidence cut and the oracle; the fusion
(confidence q_c 0.5, oracle pixels beyond 8 cm, blue pixels on straight runs between white and green) without seeds, with and without instance information; A, the same with reprojected white as seeds for the rule
(`e57c_reproject`), with the instance ids and with the whole scene as one instance; the product and the mean of the two gradients (`e57c_gradients`) at the q with the lowest mean over the split.
Metrics as `evaluate` (one Umeyama per cut, cm). Times: each stage once per scene, without the loading, the SAM 3 linking and the Chamfer scoring; `greedy_fast` and the partition cut are the optimized variants
(the reference's masks, asserted for the greedy path). On the bedroom: the squiggle's gap pixels in the zone.

uv run python -m experiments.scripts.e57c_summary --splits dev > experiments/results/e57c_summary_dev.txt
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from experiments.scripts.e57c_fast import greedy_fast
from experiments.scripts.e57c_gradients import FAR, QS, distance_to_kept
from experiments.scripts.e57c_hybrid import far_from
from experiments.scripts.e57c_oracle_curve import evaluate, greedy, removal_cost
from experiments.scripts.e57c_reproject import connect, seeds_of
from experiments.scripts.e57c_seams import DELTA, LEVELS, drop_order
from experiments.scripts.e57c_squiggles import ZONE, render_ids
from experiments.scripts.e57c_two_fills import BLUE, GOLD, GREEN, RED, WHITE
from src.analysis.filters import instance_voxel_density, keep_top, ranks
from src.analysis.scene_eval import SceneInputs, load_inputs
from src.analysis.scene_figures import (
    PRESENTATION_SCENE,
    PRESENTATION_VIEWS,
    render_cameras,
)
from src.evaluation.labeled import align_output, nearest_distances
from src.loaders.hypersim.config import load_config
from src.refinement.multiview import points_from_rays


def timed(times: dict[str, float], name: str, function):  # noqa: ANN001, ANN201
    """Run `function()`, store its seconds under `name`, return its result."""
    t0 = time.perf_counter()
    result = function()
    times[name] = time.perf_counter() - t0
    return result


def scene_rows(
    cfg: dict, scene: str, *, state: dict | None = None, score: bool = True
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """Return (kept share, accuracy, completeness, Chamfer) per approach and the stage times of one scene.

    `state` receives the scene's arrays (for the figures); `score=False` skips the Chamfers.
    """
    inputs = load_inputs(cfg, scene, torch.device("cpu"))
    truth, pixels = inputs.truth, inputs.truth.pixels
    assert pixels is not None
    ids = inputs.labels.instance_ids
    single = np.zeros_like(ids)
    t: dict[str, float] = {}
    points = timed(
        t,
        "lift",
        lambda: points_from_rays(
            inputs.scene.depth.double(), inputs.scene.rays.double()
        ).numpy(),
    )
    conf = inputs.geometry.confidence[pixels].astype(np.float64)
    timed(t, "confidence cut, sort", lambda: keep_top(ranks(conf), 0.5))
    timed(
        t,
        "confidence cut, partition",
        lambda: conf >= np.partition(conf, len(conf) // 2)[len(conf) // 2],
    )
    density = timed(
        t,
        "density",
        lambda: instance_voxel_density(
            points, inputs.scene.depth.numpy(), inputs.scene.k.numpy(), ids
        ),
    )
    pred, _, align = timed(t, "align (Umeyama)", lambda: align_output(truth, points))
    a = timed(t, "accuracy a_i", lambda: 100 * nearest_distances(truth.points, pred))
    path_ref = timed(t, "greedy, reference", lambda: greedy(truth, pred, a, LEVELS))
    path = timed(t, "greedy, incremental", lambda: greedy_fast(truth, pred, a, LEVELS))
    assert all((path[q] == path_ref[q]).all() for q in LEVELS), (
        "greedy_fast differs from the reference"
    )
    m = timed(t, "marginal cost m_i (one round)", lambda: removal_cost(truth, pred))
    v_conf, v_oracle = drop_order(conf, path)
    c = 1 - v_conf
    base, oracle = c >= 0.5, v_oracle <= 0.5
    marginal = keep_top(ranks(m - a), 0.6)
    gate = timed(
        t,
        "gate 8 cm",
        lambda: np.where(
            base,
            WHITE,
            np.where(oracle, np.where(far_from(pred, base, DELTA), GREEN, BLUE), RED),
        ),
    )
    category = np.full(pixels.shape, WHITE)
    category[pixels] = gate
    aligned = points @ align[:3, :3].T + align[:3, 3]
    back = {
        "ids": timed(t, "straight runs, ids", lambda: connect(category, ids)),
        "single": timed(
            t, "straight runs, one instance", lambda: connect(category, single)
        ),
    }
    seeds = {
        "ids": timed(t, "seeds, ids", lambda: seeds_of(category, ids, aligned, inputs)),
        "single": timed(
            t,
            "seeds, one instance",
            lambda: seeds_of(category, single, aligned, inputs),
        ),
    }
    back_seeds = {
        "ids": timed(
            t,
            "straight runs with seeds, ids",
            lambda: connect(category, ids, seeds["ids"]),
        ),
        "single": timed(
            t,
            "straight runs with seeds, one instance",
            lambda: connect(category, single, seeds["single"]),
        ),
    }
    d = timed(t, "distance to the kept, 10 cm", lambda: distance_to_kept(pred, base))
    g_c = np.minimum(c / 0.5, 1.0)
    g_o = np.where(oracle, np.clip(d / FAR, 0, 1), 0.0)
    decided = base | (oracle & (d > FAR))
    pool = ~decided
    gradient = {"product": g_c * g_o, "mean": (g_c + g_o) / 2}
    timed(t, "gradient ranking", lambda: ranks(gradient["product"][pool]))

    def full(kept: np.ndarray) -> np.ndarray:
        mask = np.ones(pixels.shape, dtype=bool)  # a pixel without GT is not scored
        mask[pixels] = kept
        return mask

    def full_category(after: np.ndarray) -> np.ndarray:
        return np.isin(after, [WHITE, GREEN, GOLD])

    masks = {
        "Déjà View": np.ones(pixels.shape, dtype=bool),
        "confidence only q 0.5": full(base),
        "density q 0.6": full(keep_top(ranks(density[pixels]), 0.6)),
        "greedy oracle q 0.5": full(oracle),
        "marginal oracle q 0.6 (one round)": full(marginal),
        "union, no gate": full(base | oracle),
        "fusion, no seeds, ids": full_category(np.where(back["ids"], GOLD, category)),
        "fusion, no seeds, one instance": full_category(
            np.where(back["single"], GOLD, category)
        ),
        "A: seeds, ids": full_category(np.where(back_seeds["ids"], GOLD, category)),
        "A: seeds, one instance": full_category(
            np.where(back_seeds["single"], GOLD, category)
        ),
    }
    for name, p in gradient.items():
        rank = ranks(p[pool])
        for q in QS:
            kept = decided.copy()
            kept[np.flatnonzero(pool)[keep_top(rank, q)]] = True
            masks[f"{name} q {q}"] = full(kept)
    if state is not None:
        state.update(
            inputs=inputs,
            points=points,
            aligned=aligned,
            pred=pred,
            a=a,
            m=m,
            c=c,
            v_oracle=v_oracle,
            base=base,
            oracle=oracle,
            category=category,
            seeds=seeds,
            back=back,
            back_seeds=back_seeds,
            path=path,
            masks=masks,
            pool=pool,
            decided=decided,
            gradient=gradient,
        )
    if not score:
        return {}, t
    rows = {
        k: np.array([k_mask.mean(), *evaluate(truth, points, k_mask)])
        for k, k_mask in masks.items()
    }
    if scene == PRESENTATION_SCENE:
        bedroom(inputs, aligned, masks, rows, oracle, pixels)
    return rows, t


def bedroom(
    inputs: SceneInputs,
    aligned: np.ndarray,
    masks: dict[str, np.ndarray],
    rows: dict[str, np.ndarray],
    oracle: np.ndarray,
    pixels: np.ndarray,
) -> None:
    """Print the bedroom's Chamfer and the gap pixels of the squiggle's zone (pulled-back view 18) that the oracle-alone cloud covers."""
    camera, _ = render_cameras(inputs, PRESENTATION_VIEWS)["view 18, pulled back"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    zone = np.zeros(camera.size, dtype=bool)
    zone[ZONE] = True
    oracle_full = np.ones(pixels.shape, dtype=bool)
    oracle_full[pixels] = oracle
    ids_oracle = render_ids(inputs, aligned, oracle_full, camera, device)
    for name in (
        "fusion, no seeds, ids",
        "fusion, no seeds, one instance",
        "A: seeds, ids",
        "A: seeds, one instance",
        "union, no gate",
    ):
        gap = (
            (render_ids(inputs, aligned, masks[name], camera, device) < 0)
            & (ids_oracle >= 0)
            & zone
        ).sum()
        print(
            f"bedroom, {name}: Chamfer {rows[name][3]:.3f} cm, kept {rows[name][0]:.3f}, zone gap pixels covered by the oracle-alone cloud {gap}"
        )


def main() -> None:
    """Print each scene's and the split's rows and times."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", nargs="*", default=["dev"])
    args = parser.parse_args()
    cfg = yaml.safe_load(Path("experiments/configs/e57c_oracle.yaml").read_text())
    split = load_config(Path(cfg["dataset"]))
    for name in args.splits:
        scenes = [f"{s}_{c}" for s, c in split.scenes([name]).items()]
        runs = []
        for scene in scenes:
            t0 = time.perf_counter()
            runs.append(scene_rows(cfg, scene))
            print(f"{scene}: {time.perf_counter() - t0:.0f} s", flush=True)
        rows = [r for r, _ in runs]
        mean = {k: np.mean([r[k] for r in rows], axis=0) for k in rows[0]}
        best = {
            m: min(
                (k for k in mean if k.startswith(f"{m} q ")), key=lambda k: mean[k][3]
            )
            for m in ("product", "mean")
        }
        shown = (
            [k for k in mean if not k.startswith(("product q", "mean q"))]
            + list(best.values())
            + ["product q 0.3", "mean q 0.4"]
        )
        print(
            f"{name}: mean over {len(rows)} scenes; kept share, accuracy / completeness / Chamfer (cm); per scene Chamfer"
        )
        for k in dict.fromkeys(shown):
            share, acc, comp, chamfer = mean[k]
            tag = " (best on this split)" if k in best.values() else ""
            print(
                f"  {k + tag:<46} {share:.3f}  {acc:.3f} / {comp:.3f} / {chamfer:.3f}   "
                + " / ".join(f"{r[k][3]:.2f}" for r in rows)
            )
        times = [t for _, t in runs]
        print("stage times (s per scene, mean over the scenes; per scene in brackets)")
        for k in times[0]:
            print(
                f"  {k:<42} {np.mean([t[k] for t in times]):6.2f}   ("
                + ", ".join(f"{t[k]:.2f}" for t in times)
                + ")"
            )


if __name__ == "__main__":
    main()
