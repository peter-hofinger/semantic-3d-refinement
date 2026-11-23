"""DINOv2 patch features of the views, cached next to Deja View's (issue #69, E69c).

Pixal3D (Li et al. 2026) conditions its 3D generator on DINOv2 (Oquab et al. 2023)
features lifted into a pixel-aligned volume. Here each pixel's feature goes to the voxel
of its Deja View point (its depth puts it on the surface) and is averaged over every
point of the voxel (`volume.voxelize_features`). The cache holds what a view gives, the
patch features f (V, h, w, 768) of DINOv2-B/14 (Apache-2.0) in float16, written as
`<name>_<views>.npz` beside `dejaview_<views>.npz`. Refinement reads it and never runs
DINOv2.

The 768 channels are reduced to `channels` by a PCA, x = W (f - mu), with W the top
eigenvectors of the covariance of the patch features of the TRAIN split only: the dev
and val scenes are projected by a basis they did not shape. The covariance is
accumulated scene by scene in float64, so the fit is exact and needs no subsample.

`uv run python -m src.backbone.dinov2 configs/e69c.yaml` writes the missing caches of
the train, val and dev scenes and, once, the PCA `<name>_pca<channels>.npz` in the root.
"""

from __future__ import annotations

import argparse
import dataclasses
import time
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import Tensor, nn
from transformers import AutoModel

from src.device import best_device
from src.loaders.hypersim.config import HypersimConfig, load_config
from src.loaders.views import Views

PATCH = 14
MEAN = (0.485, 0.456, 0.406)  # ImageNet statistics, as DINOv2's preprocessing
STD = (0.229, 0.224, 0.225)
BATCH = 5  # views per forward pass


@dataclasses.dataclass(frozen=True)
class Pca:
    """x = W (f - mu): `mean` mu (D,) and `components` W (C, D), orthonormal rows."""

    mean: np.ndarray
    components: np.ndarray

    def project(self, features: np.ndarray) -> np.ndarray:
        """Return the features (..., D) reduced to (..., C), float32."""
        assert features.shape[-1] == self.mean.shape[0], features.shape
        out = (features.astype(np.float64) - self.mean) @ self.components.T
        assert np.isfinite(out).all()
        return out.astype(np.float32)

    def save(self, path: Path) -> None:
        """Write mu and W."""
        np.savez(path, mean=self.mean, components=self.components)

    @classmethod
    def load(cls, path: Path) -> Pca:
        """Read a basis written by `save`."""
        with np.load(path) as f:
            return cls(f["mean"], f["components"])


def fit_pca(scenes: Iterable[np.ndarray], channels: int) -> Pca:
    """Return the PCA of the features (..., D) of all scenes, one scene at a time.

    The mean and covariance are sums over every patch of every scene; the components
    are the eigenvectors of the `channels` largest eigenvalues, signed so that the
    entry of largest magnitude is positive (an eigenvector's sign is arbitrary).
    """
    total, outer, n = 0.0, 0.0, 0
    for scene in scenes:
        x = scene.reshape(-1, scene.shape[-1]).astype(np.float64)
        total, outer, n = total + x.sum(0), outer + x.T @ x, n + len(x)
    assert n > channels, "fewer patches than components"
    mean = total / n
    cov = outer / n - np.outer(mean, mean)
    values, vectors = np.linalg.eigh(cov)
    top = vectors[:, np.argsort(values)[::-1][:channels]].T
    sign = np.sign(top[np.arange(channels), np.abs(top).argmax(1)])
    return Pca(mean, top * sign[:, None])


def patch_features(
    model: nn.Module, images: np.ndarray, device: torch.device
) -> np.ndarray:
    """DINOv2 patch features (V, h, w, D) float16 of the images (V, H, W, 3) uint8.

    H and W are multiples of the patch size; the class token is dropped.
    """
    n, height, width, _ = images.shape
    assert images.dtype == np.uint8 and height % PATCH == 0 and width % PATCH == 0
    mean = torch.tensor(MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(STD, device=device).view(1, 3, 1, 1)
    out = []
    model.eval()
    with torch.no_grad():
        for i in range(0, n, BATCH):
            x = torch.from_numpy(images[i : i + BATCH]).to(device).permute(0, 3, 1, 2)
            tokens = model(pixel_values=(x.float() / 255 - mean) / std)
            patches = tokens.last_hidden_state[:, 1:]
            out.append(
                patches.reshape(len(x), height // PATCH, width // PATCH, -1)
                .half()
                .cpu()
                .numpy()
            )
    features = np.concatenate(out)
    assert np.isfinite(features).all(), "DINOv2 gave a non-finite feature"
    return features


def cache_path(root: Path, scene: str, name: str, views: str) -> Path:
    """Where a scene's patch features are cached."""
    return root / scene / f"{name}_{views}.npz"


def pca_path(root: Path, name: str, channels: int) -> Path:
    """Where the PCA basis is saved."""
    return root / f"{name}_pca{channels}.npz"


def pca_scenes(split: HypersimConfig) -> list[str]:
    """Return the scene folders the PCA is fitted on: the train split, no other."""
    return [f"{s}_{c}" for s, c in split.scenes(["train"]).items()]


def load_patches(root: Path, scene: str, name: str, views: str, pca: Pca) -> Tensor:
    """Return a scene's cached features reduced by the PCA: (V, C, h, w) float16."""
    with np.load(cache_path(root, scene, name, views)) as f:
        reduced = pca.project(f["features"])
    return torch.from_numpy(reduced).permute(0, 3, 1, 2).half().contiguous()


def main() -> None:
    """Write the missing feature caches of the config's scenes, then the PCA once."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    raw = yaml.safe_load(args.config.read_text())
    spec = raw["dinov2"]
    root, views = Path(raw["root"]), raw["views"]
    split = load_config(Path(raw["split_file"]))
    names = [
        f"{s}_{c}"
        for sp in ("train", "val", "dev")
        for s, c in split.scenes([sp]).items()
    ]
    device = best_device()
    model = AutoModel.from_pretrained(spec["model"]).to(device)
    for scene in names:
        path = cache_path(root, scene, spec["name"], views)
        if path.exists():
            continue
        t0 = time.perf_counter()
        images = Views.load(root / scene / f"{views}.npz").images
        features = patch_features(model, images, device)
        np.savez(path, features=features)
        print(  # noqa: T201 - a command reports its progress
            f"{scene}: {features.shape} in {time.perf_counter() - t0:.1f} s", flush=True
        )
    target = pca_path(root, spec["name"], spec["channels"])
    if not target.exists():
        scenes = (
            np.load(cache_path(root, s, spec["name"], views))["features"]
            for s in pca_scenes(split)
        )
        fit_pca(scenes, spec["channels"]).save(target)
        print(f"wrote {target}", flush=True)  # noqa: T201 - a command reports its output


if __name__ == "__main__":
    main()
