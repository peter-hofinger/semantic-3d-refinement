"""E20a: Deja View places the surface, MoGe-3 shapes it; fitted per view or per segment.

Per view, MoGe-3's inverse depth m is mapped onto Deja View's inverse depth d by an affine
fit, d ≈ a m + b, as monocular depth is aligned to a reference (MiDaS, Ranftl et al. 2020),
and the fitted a m + b replaces d. Fitting one (a, b) per segment instead of per view lets
the segments decide where MoGe-3's relative depth may be rescaled independently; MonoInstance
(arXiv 2503.18363) does this per instance against a neural surface. The variants differ only
in the segments: `view` (one per view, the control), `sam3` (the per-view SAM 3 masks, their
unlabeled pixels one more segment) and `gt` (Hypersim's instances, the upper bound).

Each fit is least squares on the pixels valid in both, refitted once without the pixels whose
residual exceeds 10 % of d. A segment with fewer than `MIN_PIXELS` such pixels, a degenerate
fit or a <= 0 takes its view's fit; pixels MoGe-3 marks invalid, or whose fitted inverse
depth is not positive, keep Deja View's depth. Cameras, rays and confidence stay Deja
View's, so `scene_eval` scores the result as a backbone: `e20a-<variant>_<views>.npz`.

  uv run python experiments/scripts/e20a_fuse.py --splits dev val
"""

import argparse
import dataclasses
from pathlib import Path

import numpy as np

from src.geometry import Geometry
from src.loaders.hypersim.config import load_config
from src.loaders.scannetpp.gt import GroundTruth
from src.segmentation.base import Segmentation

ROOT = Path("data/processed/hypersim")
VIEWS = "views_n20_uniform_504"
MIN_PIXELS = 200
TRIM = 0.1


def fit(
    m: np.ndarray, d: np.ndarray, labels: np.ndarray, count: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Least-squares (a, b) of d ≈ a m + b per label in [0, count); n pixels used per label."""
    n = np.bincount(labels, minlength=count).astype(np.float64)
    sm, sd = np.bincount(labels, m, count), np.bincount(labels, d, count)
    smm, smd = np.bincount(labels, m * m, count), np.bincount(labels, m * d, count)
    den = n * smm - sm * sm
    with np.errstate(divide="ignore", invalid="ignore"):
        a = (n * smd - sm * sd) / den
        b = (sd - a * sm) / n
    return a, b, n


def robust_fit(
    m: np.ndarray, d: np.ndarray, labels: np.ndarray, count: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """`fit`, then refit without the pixels whose residual exceeds TRIM of d."""
    a, b, _ = fit(m, d, labels, count)
    keep = np.abs(a[labels] * m + b[labels] - d) <= TRIM * d
    keep &= np.isfinite(a[labels])
    return fit(m[keep], d[keep], labels[keep], count)


def fuse_view(
    depth: np.ndarray, moge: np.ndarray, valid: np.ndarray, segments: np.ndarray
) -> tuple[np.ndarray, dict]:
    """Fused depth (H, W) of one view and its counts; segments: (H, W) ids, -1 none."""
    assert depth.shape == moge.shape == valid.shape == segments.shape, (
        depth.shape,
        segments.shape,
    )
    assert np.all(np.isfinite(depth))
    assert np.all(depth > 0)
    d, m = 1 / depth[valid], 1 / moge[valid]
    a0, b0, _ = robust_fit(m, d, np.zeros(len(d), dtype=np.int64), 1)
    assert np.isfinite(a0[0]), a0
    assert a0[0] > 0, a0
    _, labels = np.unique(segments, return_inverse=True)
    labels = labels.reshape(segments.shape)
    count = int(labels.max()) + 1
    a, b, n = robust_fit(m, d, labels[valid], count)
    ok = (n >= MIN_PIXELS) & np.isfinite(a) & np.isfinite(b) & (a > 0)
    a, b = np.where(ok, a, a0[0]), np.where(ok, b, b0[0])
    inv = np.full(depth.shape, np.nan)
    inv[valid] = a[labels[valid]] * m + b[labels[valid]]
    use = valid & (inv > 0)
    out = np.where(use, 1 / np.where(use, inv, 1), depth).astype(np.float32)
    assert out.shape == depth.shape, out.shape
    assert np.all(np.isfinite(out))
    assert np.all(out > 0)
    pixels = labels.size
    return out, {
        "replaced": use.sum() / pixels,
        "segments": count,
        "own_fit": float(np.isin(labels[valid], np.flatnonzero(ok)).sum() / pixels),
    }


def segments_for(variant: str, folder: Path) -> np.ndarray:
    """(V, H, W) segment ids of a variant; -1 where none."""
    if variant == "sam3":
        return Segmentation.load(
            folder / f"sam3image-nyu40-t0.3_{VIEWS}.npz"
        ).instance_ids
    if variant == "gt":
        return GroundTruth.load(folder / f"gt_{VIEWS}.npz").instance_ids
    assert variant == "view", variant
    return np.zeros(
        Geometry.load(folder / f"dejaview_k16_{VIEWS}.npz").depth.shape, dtype=np.int64
    )


def main() -> None:
    """Write every variant's fused geometry for the scenes of the splits."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", nargs="+", default=["dev"])
    parser.add_argument("--variants", nargs="+", default=["view", "sam3", "gt"])
    args = parser.parse_args()
    scenes = load_config(Path("configs/hypersim.yaml")).scenes(args.splits)
    for scene, cam in scenes.items():
        folder = ROOT / f"{scene}_{cam}"
        geometry = Geometry.load(folder / f"dejaview_k16_{VIEWS}.npz")
        moge = np.load(folder / f"moge3depth_{VIEWS}.npz")
        for variant in args.variants:
            seg = segments_for(variant, folder)
            views, stats = zip(
                *(
                    fuse_view(
                        geometry.depth[v], moge["depth"][v], moge["valid"][v], seg[v]
                    )
                    for v in range(len(geometry.depth))
                ),
                strict=True,
            )
            fused = dataclasses.replace(geometry, depth=np.stack(views))
            fused.save(folder / f"e20a-{variant}_{VIEWS}.npz")
            mean = {k: np.mean([s[k] for s in stats]) for k in stats[0]}
            print(
                f"{scene}_{cam} {variant}: replaced {mean['replaced']:.1%}, own fit {mean['own_fit']:.1%}, "
                f"{mean['segments']:.0f} segments per view"
            )


if __name__ == "__main__":
    main()
