"""DINOv2 features: the PCA, its train-only fit and their voxelization (E69c)."""

import dataclasses
from pathlib import Path

import numpy as np
import pytest
import torch
from transformers import Dinov2Config, Dinov2Model

from src.backbone.dinov2 import PATCH, Pca, fit_pca, patch_features, pca_scenes
from src.loaders.hypersim.config import load_config
from src.refinement.volume import (
    ScenePoints,
    grid_channels,
    voxelize,
    voxelize_features,
)


def pixels(
    points: list[list[float]], dims: tuple[int, int, int] = (4, 2, 2)
) -> ScenePoints:
    """Pixels at the given points of a grid of unit cells."""
    n = len(points)
    return ScenePoints(
        torch.tensor(points),
        torch.zeros(n, 3),
        torch.zeros(n, 3),
        torch.zeros(n),
        torch.zeros(n, dtype=torch.long),
        torch.zeros(3),
        1.0,
        dims,
    )


def test_the_pca_recovers_the_plane_that_the_data_lie_in() -> None:
    rng = np.random.default_rng(0)
    basis, _ = np.linalg.qr(rng.normal(size=(6, 2)))
    mean = rng.normal(size=6)
    x = rng.normal(size=(500, 2)) * [3.0, 1.0] @ basis.T + mean
    pca = fit_pca([x], 2)
    assert pca.mean == pytest.approx(x.mean(0))
    assert pca.components @ pca.components.T == pytest.approx(np.eye(2), abs=1e-9)
    # the rows span the data's plane: a basis vector survives the round trip
    back = (pca.components.T @ pca.components) @ basis
    assert back == pytest.approx(basis, abs=1e-9)
    assert pca.project(x).std(0)[0] > pca.project(x).std(0)[1]  # largest variance first
    assert pca.project(pca.mean[None]) == pytest.approx(np.zeros((1, 2)), abs=1e-6)


def test_the_fit_over_scenes_equals_the_fit_over_their_union() -> None:
    rng = np.random.default_rng(1)
    a, b = rng.normal(size=(40, 3, 5, 6)), rng.normal(size=(30, 2, 5, 6)) + 1.0
    split = fit_pca([a, b], 3)
    union = fit_pca([np.concatenate([a.reshape(-1, 6), b.reshape(-1, 6)])], 3)
    assert split.mean == pytest.approx(union.mean)
    assert split.components == pytest.approx(union.components, abs=1e-9)


def test_the_basis_depends_on_the_scenes_it_is_fitted_on_and_on_nothing_else() -> None:
    rng = np.random.default_rng(2)
    train, other = (
        rng.normal(size=(200, 4)) * [4, 1, 1, 1],
        rng.normal(size=(200, 4)) * [1, 1, 1, 4],
    )
    fitted = fit_pca([train], 1)
    assert fit_pca([train], 1).components == pytest.approx(fitted.components)
    assert (
        abs(fitted.components[0, 0]) > 0.9
    )  # the train data's axis, not the other scenes'
    assert abs(fit_pca([other], 1).components[0, 3]) > 0.9


def test_the_pca_is_fitted_on_the_train_scenes_and_on_no_other_split() -> None:
    split = load_config(Path("configs/hypersim.yaml"))
    fitted = set(pca_scenes(split))
    assert len(fitted) == len(split.scenes(["train"]))
    for held_out in ("dev", "val", "test", "stress"):
        names = {f"{s}_{c}" for s, c in split.scenes([held_out]).items()}
        assert names and not fitted & names, held_out


def test_a_saved_basis_loads_back_unchanged(tmp_path: Path) -> None:
    pca = fit_pca([np.random.default_rng(3).normal(size=(50, 5))], 2)
    pca.save(tmp_path / "pca.npz")
    again = Pca.load(tmp_path / "pca.npz")
    assert again.mean == pytest.approx(pca.mean)
    assert again.components == pytest.approx(pca.components)


def test_patch_features_have_one_vector_per_patch_without_the_class_token() -> None:
    config = Dinov2Config(
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        patch_size=PATCH,
        image_size=56,
    )
    model = Dinov2Model(config)
    images = np.random.default_rng(0).integers(0, 255, (3, 28, 42, 3), dtype=np.uint8)
    out = patch_features(model, images, torch.device("cpu"))
    assert out.shape == (3, 2, 3, 16)
    assert out.dtype == np.float16
    again = patch_features(model, images[:1], torch.device("cpu"))
    assert again == pytest.approx(
        out[:1], abs=2e-3
    )  # a view's features do not depend on the batch


def test_two_views_in_one_voxel_give_the_mean_of_their_pixels_features() -> None:
    # view 0: two pixels in voxel A (x = 0), two in B (x = 1); view 1: four in A
    points = [[0.5, 0.5, 0.5]] * 2 + [[1.5, 0.5, 0.5]] * 2 + [[0.5, 0.5, 0.5]] * 4
    patches = torch.tensor([1.0, 2.0]).view(2, 1, 1, 1).expand(2, 2, 1, 1)
    patches = patches * torch.tensor([1.0, -1.0]).view(1, 2, 1, 1)
    grid = voxelize_features(pixels(points), patches, (2, 2))
    assert grid.shape == (2, 2, 2, 4)
    # A holds 2 pixels of value 1 and 4 of value 2; B holds 2 of value 1
    assert grid[0, 0, 0, 0].item() == pytest.approx((2 * 1 + 4 * 2) / 6)
    assert grid[1, 0, 0, 0].item() == pytest.approx(-(2 * 1 + 4 * 2) / 6)
    assert grid[0, 0, 0, 1].item() == pytest.approx(1.0)
    assert grid[:, :, :, 2:].abs().sum() == 0
    assert grid[:, 1].abs().sum() == 0


def test_a_pixels_feature_is_the_patch_grid_upsampled_bilinearly() -> None:
    patches = torch.tensor([0.0, 1.0]).view(1, 1, 1, 2)  # two patches along x
    points = [[x + 0.5, 0.5, 0.5] for x in range(4)]  # four pixels, one per cell
    grid = voxelize_features(pixels(points), patches, (1, 4))
    # pixel centres sit -.25, .25, .75, 1.25 patches from the first patch's centre
    assert grid[0, 0, 0].tolist() == pytest.approx([0.0, 0.25, 0.75, 1.0])


def test_the_voxelized_features_are_appended_after_the_other_channels() -> None:
    points = [[0.5, 0.5, 0.5]] * 2 + [[1.5, 0.5, 0.5]] * 2
    plain = pixels(points)
    patches = torch.tensor([1.0, 2.0]).view(2, 1, 1, 1)
    with_features = dataclasses.replace(plain, patches=patches, image=(1, 2))
    base = voxelize(plain, semantics=False)
    grid = voxelize(with_features, semantics=False)
    assert grid.shape[0] == grid_channels(semantics=False, features=1) == 9
    assert grid[:-1] == pytest.approx(base)
    assert grid[-1] == pytest.approx(voxelize_features(plain, patches, (1, 2))[0])
