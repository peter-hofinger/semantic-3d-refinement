"""Generative volumetric refiner (issue #69): a point generator on the fused voxel grid.

The per-view refiners and the volumetric refiner (`volume.py`, issue #41) and its
filter can only move or drop Deja View's pixels. Here the fused cloud is voxelized as in
`volume.py` (log count, mean confidence and colour, spread, optional class histogram), a
3D U-Net (as DFusion, Niu et al. 2022, puts one on fused depth maps) turns it into
features, and a generator head turns the features into a new cloud, free of the pixels:
it can lie where no pixel did and is not in their order. Five heads are compared
(`KINDS`): a lattice of `split`^3 points in every voxel within `reach` cells of an
occupied one (x = c + l + offset tanh(delta), an occupancy logit each); the same behind
a coarse occupancy gate (SG-NN, Dai et al. 2020); an implicit field queried twice
(ConvONet, Peng et al. 2020; Grad-PU, He et al. 2023); progressive splitting of a voxel
into 2 parents and 8 children (SnowflakeNet, Xiang et al. 2021); and every input point
seeding `copies` points (LiDiff, Nunes et al. 2024), whose untrained head returns the
input bit for bit. The output is every point whose logit is positive.

The lattice is not decoration: points that start on one spot receive the same gradient
and never separate. Every head starts at zero, so an untrained generator emits its
starting points unmoved.

Two losses (`loss`), in cells of the grid, on a random crop whose rim (`margin` cells)
only gives context. "chamfer": accuracy over the generated points of the crop's interior
and completeness over the ground truth's interior points against the generated points,
both over the points that can reach a surface only (a point can reach it if a
ground-truth point lies within `offset` cells of its starting position on every axis),
plus a cross-entropy of each point's logit against that reachability, weighted by the
ratio of unreachable to reachable points (X69d). "render": the points of the crop drawn
into the views with a soft depth test, supervised by the ground truth's depth maps (the
loss of feed-forward splatting, VolSplat, Wang et al. 2025), no point matched to
another. The ground truth is mapped into Deja View's frame by the evaluation's Umeyama
fit (`volume.gt_in_frame`), so the model is trained for the frame it is scored in.

E69c (kinds `ovoxel` and `latent`, loss `direct`): TRELLIS.2's O-Voxel (Xiang et al.
2025, arXiv 2512.14692) puts one vertex in every active voxel, and its upsampler decides
which of a voxel's 8 children are active. Here every candidate voxel has `split`^3
children, each with a logit and a vertex `child_min + child_size * sigmoid(v)`, so a
vertex never leaves its child. Every child has a fixed target (`child_targets`):
occupied if a ground-truth point lies in it, and then its vertex is the mean of those
points. The loss is the cross-entropy of the logits plus the L1 of the vertices (child
units, occupied children only), zero exactly at `oracle-ovoxel`; no output is matched to
another. Kind `latent` is the auto-decoder of DeepSDF (Park et al. 2019): a free 32-d
latent per voxel of one scene's grid replaces the U-Net, read by the same head, to show
that the head can carry the targets before an encoder is asked to produce its input.
`dino` adds DINOv2 features to the grid (`volume.voxelize_features`, Pixal3D, Li et al.
2026).

Protocol of the ladder in `docs/experiments.md`: one model per arm (`arms` of the
config) under under 1 M parameters, Adam at a constant learning rate, no weight decay,
no augmentation (every grid is built from all views); TensorBoard events under the run
folder (the train loss every step, a loss on a random validation crop after every
step, the fixed validation error every `eval_every`); the step with the least fixed
error on `select` scenes is kept. `generate-cloud` trains the arms of one rung of the
config and writes `<tag>-<rung>-<arm>-s<seed>_<views>_cloud.npz` (points in Deja View's
frame) for the prediction scenes; `cloud_eval` scores them. `--profile` times every
stage of a step instead.
"""

from __future__ import annotations

import argparse
import dataclasses
import functools
import math
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch import Tensor, nn
from torch.nn import functional as F  # noqa: N812
from torch.utils.tensorboard import SummaryWriter

from src.analysis.scene_eval import git_hash
from src.backbone.dinov2 import Pca, load_patches, pca_path
from src.device import best_device
from src.geometry import Geometry
from src.loaders.ground_truth import GroundTruth
from src.loaders.hypersim.config import load_config
from src.loaders.views import Views
from src.refinement.volume import (
    SEMANTICS,
    ScenePoints,
    UNet3d,
    compact,
    expand,
    grid_channels,
    gt_alignment,
    gt_in_frame,
    scene_points,
    voxelize,
)
from src.segmentation.base import Segmentation

CROP_TRIES = 20  # crops drawn before a scene is given up as having nothing to score
NEAREST_ELEMENTS = 2**26  # entries of a distance block (256 MiB): bounds its memory
REACH_SUBDIV = 4  # sub-voxels per cell of the grid that reachability reads
IMPLICIT_CHUNK = 200_000  # points per pass of the implicit generator's head
SPLIT_CHUNK = 50_000  # voxels per pass of the splitting generator's heads
MAX_PARAMETERS = 1_000_000  # rung 0 of the ladder: the simplest model
LOSSES = ("chamfer", "render", "direct")
LATENT = 32  # channels of the free latent per voxel (kind `latent`)


@dataclasses.dataclass(frozen=True)
class GeneratorConfig:
    """One rung of the config (`configs/e69a.yaml`) with one arm's settings applied."""

    root: Path
    views: str
    backbone: str
    segmentation: str
    tag: str  # prefix of the run's names and cloud files
    train: list[str]  # scene folders the model fits
    val: list[str]  # scene folders that only report, every step
    predict: list[str]  # scene folders a cloud is written for
    select: str  # "train" or "val": whose fixed error picks the kept step
    resolution: int  # cells along the scene box's longest side
    widths: list[int]  # U-Net channels per level
    split: int  # lattice points per voxel and axis
    reach: int  # voxels this many cells from an occupied one also emit points
    offset: float  # largest displacement of a point from its starting place, cells
    crop: int  # cells per side of a training crop
    margin: int  # cells of a crop's rim that give context only
    gen_cap: int  # generated points per crop in the accuracy and the cross-entropy
    gt_cap: int  # ground-truth points per crop that act as targets of the accuracy
    comp_cap: int  # ground-truth points per crop whose completeness is scored
    fixed_crops: int  # crops per scene of the fixed error
    fixed_scenes: int  # scenes of each list the fixed error is measured on, at most
    steps: int
    eval_every: int  # steps between two fixed errors
    amp: bool  # U-Net in bfloat16 on CUDA
    lr: float  # constant: no schedule, no weight decay
    seeds: list[int]  # one model per seed; names end in -s<seed>
    kind: str = "lattice"  # which generator: see `KINDS`
    semantics: str = "none"  # classes the grid sees: none, sam3 or gt
    copies: int = 2  # points per seed of a seeded generator
    loss: str = (
        "chamfer"  # "chamfer": reachable points to the GT points; "render"; "direct"
    )
    stride: int = 4  # the render loss draws the views at 1/stride of their size
    tau: float = 0.25  # softness of the render loss's depth test, cells
    render_views: int = 20  # views the render loss draws per step, at random
    clip: float = 0.0  # largest gradient norm, 0: none
    cosine: bool = False  # decay the rate to zero over the steps (rung 6 of the ladder)
    max_parameters: int = MAX_PARAMETERS  # raised only by a pre-registered escalation
    fixed_vertex: bool = (
        False  # `ovoxel`: every vertex at its child's centre, step 1 alone
    )
    dino: int = 0  # DINOv2 feature channels in the grid (the PCA's), 0: none
    dino_cache: str = "dinov2b"  # name of the DINOv2 cache and its PCA
    target: str = "gt20"  # "gt20": the input views' GT pixels; "gt99": E69d's GT99v
    box_percentile: float = 1.0  # the grid spans the percentile..100-percentile box ...
    box_pad: float = 0.05  # ... padded by this share, at the 1-99 % box's cell (E69f)
    coarse_reach: int = (
        0  # > 0: two-level candidates (`coarse_candidates`), not `reach`
    )


@dataclasses.dataclass(frozen=True)
class Rung:
    """A rung of the ladder: its settings, the arms it runs and the train-set sizes."""

    cfg: GeneratorConfig
    arms: dict[str, dict]  # name -> the settings that differ between the arms
    run: list[str]  # arms of the rung; `<arm>-sam3` and `<arm>-gt` see classes
    sizes: list[int]  # train scenes per model, spread evenly over the list; 0: all


def load_rung(path: Path, rung: str) -> Rung:
    """Read the config's shared settings with the rung's applied; scenes from splits."""
    raw = yaml.safe_load(path.read_text())
    arms, rungs = raw.pop("arms"), raw.pop("rungs")
    raw.pop("anchors", None)
    raw.pop("dinov2", None)  # the cache's own settings, read by `backbone/dinov2.py`
    assert rung in rungs, f"{rung} is not a rung of {path}: {list(rungs)}"
    raw |= rungs[rung]
    run, sizes = raw.pop("run"), raw.pop("sizes", [0])
    split = load_config(Path(raw.pop("split_file")))
    for key in ("train", "val", "predict"):
        names = split.scenes(raw.pop(f"{key}_splits", []))
        raw[key] = raw.pop(f"{key}_scenes", None) or [
            f"{s}_{c}" for s, c in names.items()
        ]
    raw["root"] = Path(raw["root"])
    cfg = GeneratorConfig(**raw)
    assert 2 * cfg.margin < cfg.crop and cfg.crop % 8 == 0, cfg
    assert cfg.select in ("train", "val") and cfg.loss in LOSSES, cfg
    return Rung(cfg, arms, run, sizes)


def split_arm(name: str) -> tuple[str, str]:
    """Return the arm and class source of a name: `seeded-sam3` is `seeded`, `sam3`."""
    base, _, sem = name.rpartition("-")
    return (base, sem) if sem in SEMANTICS[1:] else (name, "none")


def for_arm(
    rung: Rung, name: str, lrs: dict[str, float] | None = None
) -> GeneratorConfig:
    """Return the rung's config with an arm's settings; `<arm>-sam3` sees classes.

    lrs: learning rate per arm (without the class suffix) from the rate probe, if any.
    """
    base, sem = split_arm(name)
    assert base in rung.arms, f"{name}: no arm {base} in {list(rung.arms)}"
    cfg = dataclasses.replace(rung.cfg, semantics=sem, **rung.arms[base])
    if lrs and base in lrs:
        cfg = dataclasses.replace(cfg, lr=lrs[base])
    assert cfg.kind in KINDS and cfg.loss in LOSSES, cfg
    return cfg


def lattice(split: int) -> Tensor:
    """Offsets (split^3, 3) of a voxel's points from its centre, cells, x y z order."""
    axis = (torch.arange(split) + 0.5) / split - 0.5
    return torch.stack(torch.meshgrid(axis, axis, axis, indexing="ij"), -1).view(-1, 3)


def zero_mlp(inputs: int, outputs: int, *, hidden: int) -> nn.Sequential:
    """Two hidden layers with GELU; the last layer starts at zero."""
    net = nn.Sequential(
        nn.Linear(inputs, hidden),
        nn.GELU(),
        nn.Linear(hidden, hidden),
        nn.GELU(),
        nn.Linear(hidden, outputs),
    )
    nn.init.zeros_(net[-1].weight)
    nn.init.zeros_(net[-1].bias)
    return net


def read_field(feats: Tensor, size: Tensor, points: Tensor) -> Tensor:
    """Features (n, C) of a grid (C, Z, Y, X) at points (n, 3) in cells, trilinear.

    A module-level function: the runtime type checker keeps the closures it decorates,
    and a closure over the features kept one grid's graph alive per step.
    """
    where = (2 * points / size - 1).view(1, -1, 1, 1, 3)
    sampled = F.grid_sample(
        feats[None], where, align_corners=False, padding_mode="border"
    )
    return sampled.view(feats.shape[0], -1).T


@dataclasses.dataclass(frozen=True)
class Outputs:
    """What a generator returns for the candidate voxels (M of them)."""

    pos: Tensor  # (M, K, 3) points in cells, x y z
    logit: Tensor  # (M, K) keep the point if positive
    # the first split of a splitting generator: (M, K1, 3) points and (M, K1) logits
    coarse: tuple[Tensor, Tensor] | None = None
    gate: Tensor | None = None  # (M,) logit of each voxel's half-resolution parent


class Generator(UNet3d):
    """U-Net features, then per voxel `split`^3 points: offsets and occupancy logits."""

    distance = False
    seeded = False

    def __init__(
        self,
        grid_channels: int,
        widths: list[int],
        split: int,
        offset: float,
        *,
        semantics: bool,
    ) -> None:
        """Start the head at zero: every point on its lattice position, logit 0."""
        super().__init__(grid_channels, widths)
        self.offset = offset
        self.semantics = semantics
        base = lattice(split)
        self.head = nn.Conv3d(widths[0], len(base) * 4, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        self.register_buffer("base", base)

    def forward(
        self, grid: Tensor, candidates: Tensor, *, amp: bool = False
    ) -> Outputs:
        """Points (M, K, 3) in cells, x y z, and logits (M, K) of the candidate voxels.

        grid: (C, Z, Y, X); candidates: (Z, Y, X) bool. Cells are the grid's own: voxel
        (z, y, x) has its centre at (x + .5, y + .5, z + .5).
        """
        out = self.head(self.features(grid, amp=amp)[None])[0][:, candidates]
        out = out.view(len(self.base), 4, -1).permute(2, 0, 1)  # (M, K, 4)
        zyx = torch.nonzero(candidates)
        centre = zyx.flip(1).float() + 0.5
        pos = centre[:, None] + self.base + self.offset * torch.tanh(out[..., :3])
        return Outputs(pos, out[..., 3])


class GatedGenerator(Generator):
    """The lattice generator plus a coarse occupancy gate (a sparse hierarchy).

    SG-NN (Dai et al. 2020) completes a scene coarse to fine: each level decides which
    voxels of the next hold a surface and prunes the rest. Here the U-Net's half-size
    decoder level predicts, for each coarse voxel, whether any point of the voxels in it
    can reach a surface (the gate), supervised on that; the fine head is the lattice
    head of `Generator`, unchanged. While training the logits of the points are the fine
    head's; evaluating, a point is kept only if its parent's gate is open as well (the
    smaller of the two logits is positive). Unlike SG-NN the fine level does not see the
    coarse prediction, only the shared U-Net features.
    """

    def __init__(
        self,
        grid_channels: int,
        widths: list[int],
        split: int,
        offset: float,
        *,
        semantics: bool,
    ) -> None:
        """Start the gate at zero as well: every parent open to the same degree."""
        super().__init__(grid_channels, widths, split, offset, semantics=semantics)
        self.gate = nn.Conv3d(widths[1], 1, 1)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)

    def forward(
        self, grid: Tensor, candidates: Tensor, *, amp: bool = False
    ) -> Outputs:
        """Points and logits of the lattice head, and each voxel's parent gate logit."""
        fine, half = self.pyramid(grid, amp=amp)[:2]
        out = self.head(fine[None])[0][:, candidates]
        out = out.view(len(self.base), 4, -1).permute(2, 0, 1)  # (M, K, 4)
        voxel = torch.nonzero(candidates)  # z y x
        pos = (
            voxel.flip(1).float()[:, None]
            + 0.5
            + self.base
            + self.offset * torch.tanh(out[..., :3])
        )
        parent = self.gate(half[None])[0, 0][tuple((voxel // 2).unbind(1))]
        logit = out[..., 3]
        if not self.training:
            logit = torch.minimum(logit, parent[:, None])
        return Outputs(pos, logit, None, parent)


class ImplicitGenerator(UNet3d):
    """A field: a per-point head on trilinearly sampled U-Net features, queried twice.

    As the conditional implicit networks of ConvONet (Peng et al. 2020) and POCO (Boulch
    and Marlet 2022) read a feature grid at any position, and Grad-PU (He et al. 2023)
    moves points onto a learned surface: every candidate starts at its lattice position
    (`split`^3 per voxel), a shared MLP reads the features at that position and moves
    it, and a second query at the moved position moves it again and regresses the
    distance to the nearest ground-truth point. The two moves are summed before the
    tanh, so the total stays within `offset`; the distance, in cells, gives the score
    offset - d, and the points kept are those predicted to be within the offset of a
    surface: the reachability test, regressed instead of classified.
    """

    distance = True
    seeded = False

    def __init__(
        self,
        grid_channels: int,
        widths: list[int],
        split: int,
        offset: float,
        *,
        semantics: bool,
    ) -> None:
        """Start both moves at zero: every point on its lattice position."""
        super().__init__(grid_channels, widths)
        self.offset = offset
        self.semantics = semantics
        self.register_buffer("base", lattice(split))
        hidden = 2 * widths[0]
        mlp = functools.partial(zero_mlp, hidden=hidden)
        self.first = mlp(widths[0] + 3, 3)
        self.second = mlp(widths[0] + 6, 4)

    def move(
        self, feats: Tensor, size: Tensor, home: Tensor, code: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Return the points (n, 3) and scores (n,) of the lattice positions `home`."""
        a1 = self.first(torch.cat([read_field(feats, size, home), code], 1))
        moved = home + self.offset * torch.tanh(a1)
        seen = read_field(feats, size, moved)
        out = self.second(torch.cat([seen, (moved - home) / self.offset, code], 1))
        return (
            home + self.offset * torch.tanh(a1 + out[:, :3]),
            self.offset - F.softplus(out[:, 3]),
        )

    def forward(
        self, grid: Tensor, candidates: Tensor, *, amp: bool = False
    ) -> Outputs:
        """Points (M, K, 3) in cells and scores offset - d (M, K); see Generator."""
        feats = self.features(grid, amp=amp)
        size = torch.tensor(
            feats.shape[:0:-1], device=feats.device, dtype=torch.float32
        )
        centre = torch.nonzero(candidates).flip(1).float() + 0.5
        home = (centre[:, None] + self.base).reshape(-1, 3)
        code = self.base.repeat(len(centre), 1)

        out_pos, out_score = [], []
        for i in range(0, len(home), IMPLICIT_CHUNK):
            chunk = (home[i : i + IMPLICIT_CHUNK], code[i : i + IMPLICIT_CHUNK])
            pos, score = self.move(feats, size, *chunk)
            out_pos.append(pos)
            out_score.append(score)
        k = len(self.base)
        return Outputs(
            torch.cat(out_pos).view(-1, k, 3), torch.cat(out_score).view(-1, k)
        )


class SplitGenerator(UNet3d):
    """Progressive splitting: a voxel splits into 2 parent points, each into 4 children.

    As SnowflakeNet (Xiang et al. 2021) grows a cloud by splitting every point into
    children, one learned layer after another, with a loss at every level (here the
    parents are supervised too, `coarse` in `Outputs`). The first split reads the
    voxel's U-Net feature and a code for the parent's place in the voxel (the two
    halves along x), and gives each parent its move, a score and a feature; the second
    reads the parent's feature and move, and a code for the child's place in its half,
    and gives the child its own move on top of its parent's: a child is its lattice
    position plus offset * tanh(parent move + child move), so the reach stays the offset
    and the children of one parent move together unless the second split says otherwise.
    The
    lattice is the lattice of `Generator`, in the same order, which needs `split` 2.
    Unlike SnowflakeNet's skip-transformer there is no attention between the points.
    """

    distance = False
    seeded = False

    def __init__(
        self,
        grid_channels: int,
        widths: list[int],
        split: int,
        offset: float,
        *,
        semantics: bool,
    ) -> None:
        """Start both splits at zero: every point on its lattice position, logit 0."""
        super().__init__(grid_channels, widths)
        assert split == 2, "the splits are x, then y and z: 2 x 4 children"  # noqa: PLR2004
        self.offset = offset
        self.semantics = semantics
        base = lattice(split)
        self.register_buffer("base", base)
        self.register_buffer("coarse_base", base.view(2, 4, 3).mean(1))
        self.hidden = 2 * widths[0]
        self.first = zero_mlp(widths[0] + 3, 4 + self.hidden, hidden=self.hidden)
        self.second = zero_mlp(self.hidden + 3 + 3, 4, hidden=self.hidden)

    def split(
        self, f: Tensor, c: Tensor, parent_code: Tensor, child_code: Tensor
    ) -> tuple[Tensor, ...]:
        """Children (m, 8, 3) and logits (m, 8), parents (m, 2, 3) and logits (m, 2)."""
        m = len(f)
        out1 = self.first(
            torch.cat([f[:, None].expand(m, 2, -1), parent_code.expand(m, 2, 3)], 2)
        )  # (m, 2, 4 + hidden)
        move1, hidden = out1[..., :3], out1[..., 4:]
        inputs = torch.cat(
            [
                hidden[:, :, None].expand(m, 2, 4, -1),
                move1[:, :, None].expand(m, 2, 4, 3),
                child_code.expand(m, 2, 4, 3),
            ],
            3,
        )
        out2 = self.second(inputs)  # (m, 2, 4, 4)
        reach = self.offset * torch.tanh(move1[:, :, None] + out2[..., :3])
        lattice_pos = c[:, None, None] + self.base.view(2, 4, 3)
        return (
            (lattice_pos + reach).reshape(m, 8, 3),
            out2[..., 3].reshape(m, 8),
            c[:, None] + parent_code + self.offset * torch.tanh(move1),
            out1[..., 3],
        )

    def forward(
        self, grid: Tensor, candidates: Tensor, *, amp: bool = False
    ) -> Outputs:
        """Children (M, 8, 3), logits (M, 8), and the parents' (M, 2, 3) and (M, 2)."""
        feats = self.features(grid, amp=amp)[:, candidates].T  # (M, C)
        centre = torch.nonzero(candidates).flip(1).float() + 0.5
        parent_code = self.coarse_base  # (2, 3)
        child_code = self.base.view(2, 4, 3) - parent_code[:, None]  # (2, 4, 3)

        pos, logit, near, near_logit = [], [], [], []
        for i in range(0, len(feats), SPLIT_CHUNK):
            chunk = (feats[i : i + SPLIT_CHUNK], centre[i : i + SPLIT_CHUNK])
            for kept, new in zip(
                (pos, logit, near, near_logit),
                self.split(*chunk, parent_code, child_code),
                strict=True,
            ):
                kept.append(new)
        return Outputs(
            torch.cat(pos), torch.cat(logit), (torch.cat(near), torch.cat(near_logit))
        )


@dataclasses.dataclass(frozen=True)
class Seeds:
    """The points a seeded generator starts from, in the cells of the grid it reads."""

    pos: Tensor  # (M, 3) x y z
    color: Tensor  # (M, 3) in [0, 1]
    log_conf: Tensor  # (M,)


class SeededGenerator(UNet3d):
    """Every input point seeds `copies` points; a head moves each and scores it.

    The lattice generators start from nothing and have to rebuild, from voxel features,
    the position a Deja View pixel already has to a fraction of a cell (X69b-X69m: none
    beat the points they were given, even on the scenes they trained on). Here the
    points of the fused cloud are the seeds, as LiDiff (Nunes et al. 2024) copies the
    scan's points and offsets them to diffuse new ones: seed s gives copy k at
    s + offset tanh(delta_k) with a logit. A shared MLP reads the U-Net's feature at the
    seed (trilinear), the seed's place in its voxel, its colour and confidence, and the
    code of the copy. The head starts at zero with the logits on for copy 0 and off for
    the others, so an untrained generator returns its input points bit for bit; training
    moves them and can switch the other copies on where the input has holes. The points
    are not tied to the pixels afterwards: the output is an unordered cloud scored
    without correspondence.
    """

    distance = False
    seeded = True
    logit_on = 2.0

    def __init__(
        self,
        grid_channels: int,
        widths: list[int],
        copies: int,
        offset: float,
        *,
        semantics: bool,
    ) -> None:
        """Start the head at zero, copy 0 on, the others off."""
        super().__init__(grid_channels, widths)
        self.offset = offset
        self.semantics = semantics
        self.copies = copies
        self.register_buffer("code", torch.eye(copies))
        self.head = zero_mlp(widths[0] + 7 + copies, 4, hidden=2 * widths[0])

    def forward(self, grid: Tensor, seeds: Seeds, *, amp: bool = False) -> Outputs:
        """Points (M, K, 3) in cells and logits (M, K) of the seeds; see the class."""
        feats = self.features(grid, amp=amp)
        size = torch.tensor(feats.shape[:0:-1], device=feats.device).float()
        local = seeds.pos - seeds.pos.floor() - 0.5
        own = torch.cat(
            [
                read_field(feats, size, seeds.pos),
                local,
                seeds.color,
                seeds.log_conf[:, None],
            ],
            1,
        )
        pos, logit = [], []
        for k in range(self.copies):
            out = self.head(torch.cat([own, self.code[k].expand(len(own), -1)], 1))
            pos.append(seeds.pos + self.offset * torch.tanh(out[:, :3]))
            logit.append(out[:, 3] + (self.logit_on if k == 0 else -self.logit_on))
        return Outputs(torch.stack(pos, 1), torch.stack(logit, 1))


def child_corner(split: int) -> Tensor:
    """Offsets (split^3, 3) of each child's minimum corner from its voxel's, cells."""
    return lattice(split) + 0.5 - 0.5 / split


def decode_children(
    head: nn.Conv3d,
    feats: Tensor,
    candidates: Tensor,
    corner: Tensor,
    *,
    free_vertex: bool = True,
) -> Outputs:
    """Vertices (M, K, 3) and logits (M, K) of the candidate voxels' children.

    feats (C, Z, Y, X); candidates (Z, Y, X) bool; corner (K, 3): `child_corner`. The
    head gives 4 numbers per child: v and the logit; the vertex is child_min +
    sigmoid(v) / split cells, so it never leaves its child. A head at zero puts every
    vertex at its child's centre; without `free_vertex` it stays there (step 1 alone).
    """
    k = len(corner)
    split = round(k ** (1 / 3))
    out = head(feats[None])[0][:, candidates].view(k, 4, -1).permute(2, 0, 1)
    voxel = torch.nonzero(candidates).flip(1).float()  # x y z
    placed = (
        torch.sigmoid(out[..., :3])
        if free_vertex
        else torch.full_like(out[..., :3], 0.5)
    )
    pos = voxel[:, None] + corner + placed / split
    return Outputs(pos, out[..., 3])


class OVoxelGenerator(UNet3d):
    """U-Net features, then per candidate voxel `split`^3 children, each with a vertex.

    The two steps of TRELLIS.2's O-Voxel (Xiang et al. 2025): which children are active,
    and where the dual vertex of each lies in its child. Both heads are one 1 x 1
    convolution on the U-Net's full-size features, started at zero (logit 0, vertex at
    the child's centre). The output is the vertex of every child with a positive logit.
    """

    distance = False
    seeded = False

    def __init__(  # noqa: PLR0913 - the sizes and two switches of the head
        self,
        grid_channels: int,
        widths: list[int],
        split: int,
        *,
        semantics: bool,
        free_vertex: bool = True,
        coarse: bool = False,
    ) -> None:
        """Start the heads at zero; without `free_vertex` every vertex stays centred.

        With `coarse` a 1 x 1 gate on the U-Net's half-size level gives each coarse
        (2^3) voxel an occupancy logit (`two_level`).
        """
        super().__init__(grid_channels, widths)
        self.semantics = semantics
        self.free_vertex = free_vertex
        self.head = nn.Conv3d(widths[0], split**3 * 4, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        self.gate = nn.Conv3d(widths[1], 1, 1) if coarse else None
        if self.gate is not None:
            nn.init.zeros_(self.gate.weight)
            nn.init.zeros_(self.gate.bias)
        self.register_buffer("base", lattice(split))
        self.register_buffer("corner", child_corner(split))

    def forward(
        self, grid: Tensor, candidates: Tensor, *, amp: bool = False
    ) -> Outputs:
        """Return vertices (M, K, 3) in cells, x y z, and logits (M, K)."""
        return decode_children(
            self.head,
            self.features(grid, amp=amp),
            candidates,
            self.corner,
            free_vertex=self.free_vertex,
        )


def two_level(
    model: OVoxelGenerator,
    grid: Tensor,
    reach: int,
    truth: Tensor | None,
    *,
    amp: bool = False,
) -> tuple[Outputs, Tensor, Tensor, Tensor]:
    """Children of the two-level candidates; also the candidates, coarse logits, mask.

    grid (C, Z, Y, X), sides even; truth (Z, Y, X) bool, the voxels holding GT, or None.
    Coarse candidates: the 2^3 max-pool of the occupied voxels dilated by `reach` coarse
    cells (`coarse_candidates`). The fine candidates are the 8 voxels of each coarse
    candidate that the GT occupies while training (teacher forcing, as SG-NN and
    TRELLIS train their fine level) and of each with a logit >= 0 otherwise, so
    evaluation never sees the GT. Returns the children, the fine candidates (Z, Y, X),
    the coarse logits (Z/2, Y/2, X/2) and the coarse candidates.
    """
    assert model.gate is not None and all(d % 2 == 0 for d in grid.shape[1:])
    fine, half = model.pyramid(grid, amp=amp)[:2]
    logit = model.gate(half[None])[0, 0]
    occupied = grid[0] > 0
    wide = dilate(F.max_pool3d(occupied.float()[None, None], 2)[0, 0] > 0, reach)
    if truth is not None and model.training:
        keep = F.max_pool3d(truth.float()[None, None], 2)[0, 0] > 0
    else:
        keep = logit >= 0  # the untrained gate (logit 0) opens every candidate
    on = wide & keep
    candidates = on.repeat_interleave(2, 0).repeat_interleave(2, 1)
    candidates = candidates.repeat_interleave(2, 2)
    out = decode_children(
        model.head, fine, candidates, model.corner, free_vertex=model.free_vertex
    )
    return out, candidates, logit, wide


class LatentGenerator(nn.Module):
    """The auto-decoder of rung 3a: a free latent per voxel of one scene, the same head.

    As DeepSDF (Park et al. 2019) fits one code per shape, here `table` (LATENT, Z, Y,
    X) holds one code per voxel of one scene's grid and replaces the U-Net, so the head
    is the only network. It has no input, only the memory of one scene; it shows whether
    the head and the targets can reach the oracle before an encoder is asked to produce
    the latent. The grid it reads is the data grid with the table appended
    (`scene_grid`): the data channels give the candidates, the last LATENT channels the
    codes.
    """

    distance = False
    seeded = False

    def __init__(self, dims: tuple[int, int, int], split: int) -> None:
        """Make a table over a grid of `dims` (x y z) cells; the head starts at zero."""
        super().__init__()
        self.semantics = False
        nx, ny, nz = dims
        self.table = nn.Parameter(0.1 * torch.randn(LATENT, nz, ny, nx))
        self.head = nn.Conv3d(LATENT, split**3 * 4, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        self.register_buffer("base", lattice(split))
        self.register_buffer("corner", child_corner(split))

    def forward(
        self, grid: Tensor, candidates: Tensor, *, amp: bool = False
    ) -> Outputs:
        """Vertices and logits as `OVoxelGenerator`; the codes are the last channels."""
        assert not amp, "the latent head runs in float32"
        return decode_children(self.head, grid[-LATENT:], candidates, self.corner)


KINDS = {
    "ovoxel": OVoxelGenerator,
    "latent": LatentGenerator,
    "lattice": Generator,
    "gated": GatedGenerator,
    "implicit": ImplicitGenerator,
    "split": SplitGenerator,
    "seeded": SeededGenerator,
}  # the generator of a config


def build(cfg: GeneratorConfig, dims: tuple[int, int, int] | None = None) -> nn.Module:
    """Return the untrained generator of the config's `kind`, under 1 M parameters.

    dims (x y z cells): the grid of the one scene a `latent` generator is a table for.
    The table is memory, not a model: the limit counts the head only.
    """
    sem = cfg.semantics != "none"
    channels = grid_channels(semantics=sem, features=cfg.dino)
    if cfg.kind == "latent":
        assert dims is not None, "a latent generator needs its scene's grid"
        model = LatentGenerator(dims, cfg.split)
    elif cfg.kind == "ovoxel":
        model = OVoxelGenerator(
            channels,
            cfg.widths,
            cfg.split,
            semantics=sem,
            free_vertex=not cfg.fixed_vertex,
            coarse=cfg.coarse_reach > 0,
        )
    else:
        assert cfg.coarse_reach == 0, "two-level candidates: `ovoxel` only"
        model = KINDS[cfg.kind](
            channels,
            cfg.widths,
            cfg.copies if cfg.kind == "seeded" else cfg.split,
            cfg.offset,
            semantics=sem,
        )
    weights = sum(p.numel() for n, p in model.named_parameters() if n != "table")
    assert weights < cfg.max_parameters, cfg.widths
    return model


def box_of(cfg: GeneratorConfig) -> tuple[float, float]:
    """Return the grid's extent, (percentile, pad), for `scene_points`."""
    return cfg.box_percentile, cfg.box_pad


def dilate(mask: Tensor, reach: int) -> Tensor:
    """Voxels (Z, Y, X) within `reach` cells (Chebyshev) of a True one.

    The Chebyshev ball is a cube, so the 3D max-pool separates into one 1D pool per
    axis: 3 (2r + 1) reads per voxel instead of (2r + 1)^3 (reach 12: 75, not 15 625).
    """
    out = mask.float()[None, None]
    for kernel, pad in (
        ((2 * reach + 1, 1, 1), (reach, 0, 0)),
        ((1, 2 * reach + 1, 1), (0, reach, 0)),
        ((1, 1, 2 * reach + 1), (0, 0, reach)),
    ):
        out = F.max_pool3d(out, kernel, 1, pad)
    return out[0, 0] > 0


def coarse_candidates(occupied: Tensor, reach: int, truth: Tensor) -> Tensor:
    """Two-level candidates (Z, Y, X): fine voxels of coarse voxels holding `truth`.

    occupied, truth (Z, Y, X) bool, sides even. The coarse level is the 2^3 max-pool of
    `occupied` dilated by `reach` coarse cells; of those, the ones `truth` occupies pass
    their 8 fine voxels on (what a perfect coarse stage keeps, as TRELLIS's sparse
    structure and SG-NN's coarse-to-fine levels train their fine level). E69f: coarse
    reach 6 (12 fine cells) covers a median 96.8 % of the unseen GT, 3.7 k candidates
    per 32-cell crop where a fine reach of 12 needs 30.4 k.
    """
    assert occupied.shape == truth.shape and all(d % 2 == 0 for d in truth.shape)
    pool = F.max_pool3d(torch.stack([occupied, truth]).float()[:, None], 2)[:, 0] > 0
    keep = dilate(pool[0], reach) & pool[1]
    return keep.repeat_interleave(2, 0).repeat_interleave(2, 1).repeat_interleave(2, 2)


def candidate_mask(grid: Tensor, reach: int) -> Tensor:
    """Voxels (Z, Y, X) within `reach` cells (Chebyshev) of one that holds a point."""
    return dilate(grid[0] > 0, reach)


class EmptyCrop(Exception):  # noqa: N818 - a signal to draw again, not an error
    """A crop with nothing to score: no ground truth, candidate or reachable point."""


def nearest(queries: Tensor, targets: Tensor) -> Tensor:
    """Index in targets (m, 3) of each query's (n, 3) nearest point; no gradient.

    The distance matrix is built in blocks of at most `NEAREST_ELEMENTS` entries.
    """
    step = max(1, NEAREST_ELEMENTS // len(targets))
    with torch.no_grad():
        return torch.cat(
            [
                torch.cdist(queries[i : i + step], targets).argmin(1)
                for i in range(0, len(queries), step)
            ]
        )


def cloud_loss(
    inner: Tensor,
    generated: Tensor,
    truth: tuple[Tensor, Tensor],
    reachable: tuple[Tensor, Tensor],
) -> dict[str, Tensor]:
    """Accuracy and completeness (cells) of one crop, over the reachable points.

    inner (n, 3): the generated points that are scored; generated (g, 3): all of the
    crop; truth = (targets (m, 3), complete (q, 3)): the ground truth the accuracy looks
    up, and the points whose nearest generated point is the completeness; reachable =
    (n,) and (g,) bool masks of the points that can reach a surface. Both terms count
    the reachable points only, which are the ones the output keeps. Distances are
    differentiable through the picked neighbours.
    """
    targets, complete = truth
    inner_ok, all_ok = reachable
    assert inner.shape[1:] == (3,) and len(targets) and len(generated)
    assert inner_ok.shape == (len(inner),) and all_ok.shape == (len(generated),)
    if not (inner_ok.any() and all_ok.any()):
        raise EmptyCrop
    keep = generated[all_ok]
    acc = (inner - targets[nearest(inner, targets)]).norm(dim=1)[inner_ok]
    comp = (complete - keep[nearest(complete, keep)]).norm(dim=1)
    return {"acc": acc.mean(), "comp": comp.mean()}


def occupancy_loss(logit: Tensor, reachable: Tensor) -> Tensor:
    """Cross-entropy of the logits (n,) against reachability (n,) bool, class-balanced.

    The positives are weighted by the ratio of negatives to positives, so the rarer
    class counts as much (X69d).
    """
    assert logit.shape == reachable.shape and reachable.any()
    positive = reachable.sum()
    balance = (len(reachable) - positive).clamp(min=1) / positive
    return F.binary_cross_entropy_with_logits(
        logit, reachable.float(), pos_weight=balance
    )


def distance_loss(score: Tensor, distance: Tensor, offset: float) -> Tensor:
    """L1 between the predicted distance, offset - score, and the true one (n,), cells.

    The true distance is capped at twice the offset: what matters is whether it is
    within the offset, and a point far off the surface has no use for the exact value.
    """
    assert score.shape == distance.shape
    return (offset - score - distance.clamp(max=2 * offset)).abs().mean()


def gate_loss(gate: Tensor, reachable: Tensor, voxel: Tensor) -> Tensor:
    """Cross-entropy of the parents' gate logits (M,) against whether a child can reach.

    reachable (M,) bool: the voxel holds a reachable point; voxel (M, 3) x y z. A parent
    (voxel // 2) is open if any of its voxels is, and counts once, however many of its
    voxels are candidates.
    """
    parent = voxel // 2
    span = int(parent.max()) + 1
    key = (parent[:, 0] * span + parent[:, 1]) * span + parent[:, 2]
    _, inverse = torch.unique(key, return_inverse=True)
    parents = int(inverse.max()) + 1
    opened = torch.zeros(parents, device=gate.device)
    opened.scatter_reduce_(0, inverse, reachable.float(), "amax")
    logit = torch.zeros(parents, device=gate.device)
    logit.scatter_reduce_(0, inverse, gate, "mean", include_self=False)
    return occupancy_loss(logit, opened > 0)


def sub_voxel(points: Tensor, origin: Tensor, shape: tuple[int, ...]) -> Tensor:
    """Index (n, 3), z y x, of the sub-voxel of each point (n, 3) in cells, clamped."""
    i = ((points - origin) * REACH_SUBDIV).floor().long().flip(1)
    return torch.stack([i[:, a].clamp(0, shape[a] - 1) for a in range(3)], 1)


def reach_distance(
    home: Tensor, targets: Tensor, limit: float, origin: Tensor, size: Tensor
) -> Tensor:
    """Distance (n,), cells, from each point to the nearest target, per axis (max norm).

    home (n, 3) and targets (m, 3) in cells, x y z, inside the box of `size` cells at
    `origin`. The targets are binned into a grid of `REACH_SUBDIV` sub-voxels per cell
    and dilated one sub-voxel at a time (a 3 x 3 x 3 max pool), so a distance is exact
    to a quarter cell and costs one lookup per point and step, where a nearest
    neighbour search for every candidate took 1.5 s per crop. A point with no target
    within `limit` gets limit + one sub-voxel.
    """
    s = REACH_SUBDIV
    shape = tuple(int(n) * s for n in size.flip(0))  # z y x

    near = torch.zeros(shape, device=home.device)
    z, y, x = sub_voxel(targets, origin, shape).unbind(1)
    near[z, y, x] = 1.0
    z, y, x = sub_voxel(home, origin, shape).unbind(1)
    steps = round(limit * s)
    distance = torch.full((len(home),), (steps + 1) / s, device=home.device)
    for r in range(steps + 1):
        found = (near[z, y, x] > 0) & (distance > steps / s + 1e-6)
        distance[found] = r / s
        if r < steps:
            near = F.max_pool3d(near[None, None], 3, 1, 1)[0, 0]
    return distance


def reachable_points(
    home: Tensor, targets: Tensor, offset: float, origin: Tensor, size: Tensor
) -> Tensor:
    """Mask (n,) of the points with a target within `offset` of home on every axis.

    A point moves by at most `offset` cells per axis, so one without a ground-truth
    point that near cannot reach a surface (see `reach_distance`).
    """
    return reach_distance(home, targets, offset, origin, size) <= offset


@dataclasses.dataclass(frozen=True)
class RenderViews:
    """Cameras and ground-truth depth of the views, in the grid's frame, at 1/stride."""

    centre: Tensor  # (V, 3) camera centres
    axes: Tensor  # (V, 3, 3) columns: each camera's x, y, z axes in the frame
    focal: Tensor  # (V, 2) fx, fy, pixels of the reduced image
    principal: Tensor  # (V, 2) cx, cy, pixels of the reduced image
    depth: Tensor  # (V, h, w) z of the ground truth at each block's centre, 0: none
    spread: Tensor  # (V, h, w) largest minus smallest z of the block's pixels
    point: Tensor  # (V, h, w, 3) that ground-truth point

    def to(self, device: torch.device) -> RenderViews:
        """Return the same views on another device."""
        return RenderViews(*(t.to(device) for t in dataclasses.astuple(self)))


def render_views(
    dv: tuple[np.ndarray, np.ndarray],
    views: Views,
    gt: GroundTruth,
    scale: float,
    stride: int,
) -> RenderViews:
    """Cameras and ground truth in Deja View's frame over `scale` (the grid's frame).

    dv: Deja View's rays (N, 6) and depth (N,), as `scene_points` returns them.

    The ground-truth frame is Deja View's by the evaluation's similarity (s, R, t):
    gt = s R dv + t; its z-depth maps are cut into stride x stride blocks, a block
    counts where all its pixels have depth.
    """
    _, _, (s, rot, t) = gt_alignment(*dv, views, gt)
    z = gt.depth.astype(np.float64)
    n, h, w = z.shape
    assert h % stride == 0 and w % stride == 0, (h, w, stride)
    blocks = z.reshape(n, h // stride, stride, w // stride, stride)
    unit = s * scale  # ground-truth metres per normalized unit
    mean = np.where((blocks > 0).all((2, 4)), blocks.mean((2, 4)), 0.0) / unit
    spread = (blocks.max((2, 4)) - blocks.min((2, 4))) / unit
    k, c2w = views.intrinsics.astype(np.float64), views.cam_to_world.astype(np.float64)
    centre = (c2w[:, :3, 3] - t) @ rot / unit
    axes = np.einsum("ji,vjk->vik", rot, c2w[:, :3, :3])
    focal, principal = k[:, [0, 1], [0, 1]] / stride, k[:, :2, 2] / stride
    col, row = np.meshgrid(np.arange(w // stride) + 0.5, np.arange(h // stride) + 0.5)
    ray = np.stack(
        [
            (col[None] - principal[:, None, None, 0]) / focal[:, None, None, 0],
            (row[None] - principal[:, None, None, 1]) / focal[:, None, None, 1],
            np.ones((n, *col.shape)),
        ],
        -1,
    )
    point = centre[:, None, None] + np.einsum(
        "vik,vhwk->vhwi", axes, ray * mean[..., None]
    )
    return RenderViews(
        *(
            torch.from_numpy(a).float()
            for a in (centre, axes, focal, principal, mean, spread, point)
        )
    )


def load_render(cfg: GeneratorConfig, scene: str) -> RenderViews:
    """Return the render views of one scene, in the grid's frame (`load_scene`)."""
    folder = cfg.root / scene
    views = Views.load(folder / f"{cfg.views}.npz")
    geometry = Geometry.load(folder / f"{cfg.backbone}_{cfg.views}.npz")
    gt = GroundTruth.load(folder / f"gt_{cfg.views}.npz")
    none = np.full(geometry.depth.shape, -1)
    _, rays, depth, scale = scene_points(
        geometry, views.images, none, {}, cfg.resolution, box_of(cfg)
    )
    return render_views((rays, depth), views, gt, scale, cfg.stride)


def render_depth(
    points: Tensor, opacity: Tensor, rv: RenderViews, view: int, tau: float
) -> tuple[Tensor, Tensor]:
    """Depth (h, w) and alpha (h, w) of points (n, 3), opacities (n,), seen from a view.

    Each point is spread bilinearly over the four pixels around its projection. A
    pixel's depth is the mean of the z of the points on it, weighted by opacity times
    exp(-(z - nearest z) / tau): a soft depth test, so points behind the nearest
    surface count little, and differentiable in z and in the opacities. Its alpha is
    1 - exp(-total weight): near 1 where points are drawn, 0 where none are.
    """
    _, h, w = rv.depth.shape
    cam = (points - rv.centre[view]) @ rv.axes[view]
    z = cam[:, 2]
    base = cam[:, :2] / z.clamp(min=1e-6)[:, None] * rv.focal[view] + rv.principal[view]
    base = base - 0.5
    corner = base.floor()
    frac = base - corner
    x0, y0 = corner[:, 0].long(), corner[:, 1].long()
    index, weight, depth, alpha = [], [], [], []
    for dx in (0, 1):
        for dy in (0, 1):
            x, y = x0 + dx, y0 + dy
            ok = (z > 0) & (x >= 0) & (x < w) & (y >= 0) & (y < h)
            wx = frac[:, 0] if dx else 1 - frac[:, 0]
            wy = frac[:, 1] if dy else 1 - frac[:, 1]
            index.append((y * w + x)[ok])
            weight.append((opacity * wx * wy)[ok])
            depth.append(z[ok])
    index, weight, depth = torch.cat(index), torch.cat(weight), torch.cat(depth)
    nearest_z = torch.full((h * w,), float("inf"), device=z.device)
    nearest_z.scatter_reduce_(0, index, depth.detach(), "amin")
    soft = weight * torch.exp(-(depth - nearest_z[index]) / tau)
    total = torch.zeros(h * w, device=z.device).index_add(0, index, soft)
    summed = torch.zeros(h * w, device=z.device).index_add(0, index, soft * depth)
    alpha = 1 - torch.exp(-total)
    return (summed / total.clamp(min=1e-8)).view(h, w), alpha.view(h, w)


def pick_views(
    render: tuple[RenderViews, list[int]],
    box: tuple[Tensor, Tensor],
    size: tuple[float, int],
    rng: np.random.Generator,
) -> list[int]:
    """Return at most n of the views, at random, with a supervised pixel in the box.

    size = (cell, n); a pixel is supervised as in `render_terms`.
    """
    rv, views = render
    (lo, hi), (cell, n) = box, size
    seen = [
        v
        for v in views
        if (
            (rv.depth[v] > 0)
            & (rv.spread[v] < cell)
            & ((rv.point[v] >= lo) & (rv.point[v] < hi)).all(-1)
        ).any()
    ]
    if not seen:
        raise EmptyCrop
    return [int(v) for v in rng.choice(seen, min(n, len(seen)), replace=False)]


def render_terms(
    rv: RenderViews,
    views: list[int],
    drawn: tuple[Tensor, Tensor],
    box: tuple[Tensor, Tensor],
    cell: tuple[float, float],
) -> dict[str, Tensor]:
    """Depth and coverage losses, in cells, against the ground truth inside `box`.

    The supervised pixels of each view are those whose block has a ground-truth depth
    that varies by less than a cell and whose ground-truth point lies in the box
    (lo, hi). depth: |rendered z - ground-truth z| averaged with the pixels' alpha as
    weight (how far off what is drawn is); cover: the mean of 1 - alpha (a pixel with
    ground truth that nothing is drawn on). No point is matched to another. drawn:
    the points (n, 3) and their opacities (n,); cell: its size and the softness of
    the depth test in cells.
    """
    lo, hi = box
    (size, soft), (points, opacity) = cell, drawn
    off = mass = hole = count = points.new_zeros(())
    for v in views:
        depth, alpha = render_depth(points, opacity, rv, v, soft * size)
        inside = ((rv.point[v] >= lo) & (rv.point[v] < hi)).all(-1)
        sup = (rv.depth[v] > 0) & (rv.spread[v] < size) & inside
        off = off + (alpha[sup] * (depth[sup] - rv.depth[v][sup]).abs() / size).sum()
        mass = mass + alpha[sup].sum()
        hole = hole + (1 - alpha[sup]).sum()
        count = count + sup.sum()
    if count == 0:
        raise EmptyCrop
    return {"depth": off / mass.clamp(min=1e-6), "cover": hole / count}


@dataclasses.dataclass(frozen=True)
class TrainScene:
    """A scene on the CPU: compact pixels, GT in the grid's frame, view geometry."""

    pts: ScenePoints
    gt: Tensor  # (N, 3) float32 ground truth of each pixel, normalized frame
    has: Tensor  # (N,) bool, the pixel has ground truth
    render: RenderViews | None = None  # only for `loss: render`


def load_scene(
    cfg: GeneratorConfig, scene: str, semantics: str
) -> tuple[ScenePoints, Tensor, Tensor, float]:
    """One scene's pixels, its GT (`target`) in the grid's frame, where it is, scale."""
    folder = cfg.root / scene
    views = Views.load(folder / f"{cfg.views}.npz")
    geometry = Geometry.load(folder / f"{cfg.backbone}_{cfg.views}.npz")
    gt = GroundTruth.load(folder / f"gt_{cfg.views}.npz")
    if semantics == "gt":
        ids, classes = gt.instance_ids, gt.classes
    elif semantics == "sam3":
        masks = Segmentation.load(folder / f"{cfg.segmentation}_{cfg.views}.npz")
        ids, classes = masks.instance_ids, masks.classes
    else:
        ids, classes = np.full(geometry.depth.shape, -1), {}
    pts, rays, depth, scale = scene_points(
        geometry, views.images, ids, classes, cfg.resolution, box_of(cfg)
    )
    if cfg.dino:
        pca = Pca.load(pca_path(cfg.root, cfg.dino_cache, cfg.dino))
        patches = load_patches(cfg.root, scene, cfg.dino_cache, cfg.views, pca)
        assert patches.shape[0] * np.prod(views.images.shape[1:3]) == len(pts.points)
        pts = dataclasses.replace(
            pts, patches=patches, image=tuple(views.images.shape[1:3])
        )
    if cfg.target == "gt99":
        _, _, (s, rotation, t) = gt_alignment(rays, depth, views, gt)
        world = np.load(folder / f"gt99v1cm_{cfg.views}.npz")["points"]
        in_dv, has = (world - t) @ rotation / s, np.ones(len(world), dtype=bool)
    else:
        assert cfg.target == "gt20", cfg.target
        in_dv, has = gt_in_frame(rays, depth, views, gt)
    return (
        pts,
        torch.from_numpy(in_dv / scale).float(),
        torch.from_numpy(has),
        scale,
    )


def spread(items: list, n: int) -> list:
    """Return n of the items, evenly spaced over the list (all of them for n = 0)."""
    assert 0 <= n <= len(items), (n, len(items))
    if n in (0, len(items)):
        return items
    return [items[i] for i in np.linspace(0, len(items) - 1, n).round().astype(int)]


def load_scenes(cfg: GeneratorConfig, names: list[str]) -> list[TrainScene]:
    """Every scene, compact; one without pinhole rays is skipped."""
    data = []
    for name in names:
        try:
            pts, gt, has, _ = load_scene(cfg, name, cfg.semantics)
        except AssertionError as err:  # fit_pinhole: Deja View's rays are no pinhole
            print(f"{name}: skipped, pinhole fit failed ({err})", flush=True)  # noqa: T201
            continue
        render = load_render(cfg, name) if cfg.loss == "render" else None
        data.append(TrainScene(compact(pts), gt, has, render))
    assert data, f"no scene of {names} could be loaded"
    return data


def subsample(
    n: int, cap: int, rng: np.random.Generator, device: torch.device
) -> Tensor:
    """At most `cap` distinct indices of range(n), all of them if n <= cap."""
    if n <= cap:
        return torch.arange(n, device=device)
    return torch.from_numpy(rng.choice(n, cap, replace=False)).to(device)


@dataclasses.dataclass(frozen=True)
class Crop:
    """One training crop on the device: its box, candidate voxels and ground truth."""

    origin: Tensor  # (3,) corner, cells, x y z
    extent: Tensor  # (3,) size, cells
    voxel: Tensor  # (M, 3) index of each candidate voxel in the crop, x y z
    interior: Tensor  # (M,) bool: the voxel is further than the margin from the rim
    targets: Tensor  # (m, 3) ground truth the accuracy looks up
    complete: Tensor  # (q, 3) ground truth whose nearest point is the completeness


@dataclasses.dataclass(frozen=True)
class Level:
    """One level of a generator's points: K per candidate, scores, starting places."""

    pos: Tensor  # (M, K, 3) in the crop's cells
    score: Tensor  # (M, K)
    home: Tensor  # (M, K, 3) where each point starts, in the crop's cells
    distance: bool  # the score is offset - distance, not a logit


def level_losses(
    crop: Crop, level: Level, cfg: GeneratorConfig, rng: np.random.Generator
) -> tuple[dict[str, Tensor], Tensor]:
    """Return the accuracy, completeness and occupancy losses of a crop's level.

    Also the reachable mask (M * K,) of the level's points.
    """
    k = level.pos.shape[1]
    gen = (level.pos + crop.origin).reshape(-1, 3)  # global cells, x y z
    scored = torch.nonzero(crop.interior.repeat_interleave(k))[:, 0]
    if not len(scored):
        raise EmptyCrop
    scored = scored[subsample(len(scored), cfg.gen_cap, rng, gen.device)]
    home = (level.home + crop.origin).reshape(-1, 3)
    distance = reach_distance(
        home, crop.targets, 2 * cfg.offset, crop.origin, crop.extent
    )
    reachable = distance <= cfg.offset
    terms = cloud_loss(
        gen[scored], gen, (crop.targets, crop.complete), (reachable[scored], reachable)
    )
    score = level.score.reshape(-1)[scored]
    if level.distance:
        terms["occ"] = distance_loss(score, distance[scored], cfg.offset)
    else:
        terms["occ"] = occupancy_loss(score, reachable[scored])
    return terms, reachable


OnDevice = tuple[ScenePoints, Tensor, tuple[RenderViews, list[int]] | None]


def voxel_key(voxel: Tensor, dims: tuple[int, int, int]) -> Tensor:
    """Flat z-major key (n,) of voxels (n, 3) x y z in a grid of `dims` cells."""
    nx, ny, _ = dims
    return (voxel[:, 2] * ny + voxel[:, 1]) * nx + voxel[:, 0]


def child_targets(
    voxel: Tensor, points: Tensor, split: int, dims: tuple[int, int, int]
) -> tuple[Tensor, Tensor]:
    """Return the voxels' fixed child targets: occupied (M, K) bool, vertex (M, K, 3).

    voxel (M, 3) long, x y z, unique and in z-major order (as `torch.nonzero(mask)`
    gives them) in a grid of `dims` cells; points (n, 3) in cells of the same frame. A
    child is occupied if a point lies in it, and its vertex target is the mean of those
    points in child units (0 to 1 from its minimum corner; 0 where empty). The children
    are in `lattice` order. A point outside every listed voxel is ignored.
    """
    k = split**3
    assert voxel.shape[1:] == (3,) and points.shape[1:] == (3,)
    fine = torch.floor(points * split).long()
    inside = (fine >= 0).all(1) & (
        fine < torch.tensor(dims, device=fine.device) * split
    ).all(1)
    fine, frac = fine[inside], points[inside] * split - fine[inside]
    keys, wanted = voxel_key(voxel, dims), voxel_key(fine // split, dims)
    assert (keys[1:] > keys[:-1]).all(), "voxels must be unique and z-major"
    row = torch.searchsorted(keys, wanted).clamp(max=len(keys) - 1)
    held = keys[row] == wanted
    sub = fine[held] % split
    flat = row[held] * k + (sub[:, 0] * split + sub[:, 1]) * split + sub[:, 2]
    count = torch.zeros(len(keys) * k, device=fine.device)
    count.index_add_(0, flat, torch.ones(len(flat), device=fine.device))
    total = torch.zeros(len(keys) * k, 3, device=fine.device)
    total.index_add_(0, flat, frac[held])
    vertex = total / count.clamp(min=1)[:, None]
    assert vertex.min() >= 0 and vertex.max() < 1
    return (count > 0).view(len(keys), k), vertex.view(len(keys), k, 3)


@dataclasses.dataclass(frozen=True)
class Children:
    """The children of a crop's or scene's candidate voxels: output and target."""

    logit: Tensor  # (M, K) keep the child if positive
    pos: Tensor  # (M, K, 3) vertices, cells, x y z
    corner: Tensor  # (M, K, 3) each child's minimum corner, cells
    occupied: Tensor  # (M, K) bool target
    vertex: Tensor  # (M, K, 3) target vertex, child units; 0 where empty
    scored: Tensor  # (M,) bool: the voxel counts (not in the crop's rim)
    # two-level candidates: the scored coarse candidates' logits and targets (bool)
    coarse: tuple[Tensor, Tensor] | None = None

    def placed(self, split: int) -> Tensor:
        """Return the vertices in child units (M, K, 3): 0 to 1 inside their child."""
        return (self.pos - self.corner) * split


def direct_terms(
    children: Children, split: int, *, vertex: bool = True
) -> dict[str, Tensor]:
    """Return loss `direct`: cross-entropy of the logits, L1 of occupied vertices.

    Both over the scored voxels only; the L1 in child units, per coordinate. Zero
    exactly where every logit is right and every occupied child's vertex is its target.
    Without `vertex` (the vertices are fixed at the centres) the cross-entropy alone.
    """
    live = children.scored
    occupied = children.occupied[live]
    if not occupied.any():
        raise EmptyCrop
    terms = {
        "occ": F.binary_cross_entropy_with_logits(
            children.logit[live], occupied.float()
        )
    }
    if vertex:
        placed = children.placed(split)[live][occupied]
        terms["vertex"] = (placed - children.vertex[live][occupied]).abs().mean()
    if children.coarse is not None and children.coarse[1].any():
        terms["coarse"] = occupancy_loss(*children.coarse)
    return terms


def child_counts(children: Children, split: int) -> dict[str, float]:
    """Return counts over the scored voxels: kept, occupied, hits, vertex error.

    Counts, not rates, so that crops and scenes can be pooled before dividing.
    """
    live = children.scored
    kept, occupied = children.logit[live] > 0, children.occupied[live]
    error = (
        children.placed(split)[live][occupied] - children.vertex[live][occupied]
    ).abs()
    return {
        "kept": float(kept.sum()),
        "occupied": float(occupied.sum()),
        "hit": float((kept & occupied).sum()),
        "vertex_sum": float(error.sum()),
        "vertex_n": float(error.numel()),
    }


def child_scores(counts: dict[str, float]) -> dict[str, float]:
    """Return child precision, recall and the mean vertex error (child units)."""
    return {
        "precision": counts["hit"] / max(counts["kept"], 1.0),
        "recall": counts["hit"] / max(counts["occupied"], 1.0),
        "vertex_err": counts["vertex_sum"] / max(counts["vertex_n"], 1.0),
    }


def output_chamfer(
    children: Children, truth: tuple[Tensor, Tensor]
) -> tuple[Tensor, Tensor]:
    """Accuracy and completeness (cells) of the kept vertices against the crop's GT.

    truth = (targets (m, 3), complete (q, 3)) as in `cloud_loss`: accuracy over the kept
    vertices of the scored voxels, completeness of `complete` against all kept vertices
    of the crop. Infinite where nothing is kept (the untrained generator keeps nothing).
    """
    targets, complete = truth
    kept = children.logit > 0
    every = children.pos[kept]
    inner = children.pos[kept & children.scored[:, None]]
    if not len(inner):
        return torch.tensor(float("inf")), torch.tensor(float("inf"))
    acc = (inner - targets[nearest(inner, targets)]).norm(dim=1).mean()
    comp = (complete - every[nearest(complete, every)]).norm(dim=1).mean()
    return acc, comp


def crop_window(
    pts: ScenePoints, grid: Tensor, cfg: GeneratorConfig, rng: np.random.Generator
) -> tuple[Tensor, np.ndarray, np.ndarray]:
    """Return a random crop centred on a random point, with its corner and size."""
    centre = pts.points[int(rng.integers(len(pts.points)))]
    local = ((centre - pts.lo) / pts.cell).floor().long().cpu().numpy()
    size = np.minimum(cfg.crop, pts.dims)  # x y z
    lo = np.clip(local - size // 2, 0, np.array(pts.dims) - size)
    if cfg.coarse_reach:  # coarse voxels of the crop are the scene's (sizes are x8)
        lo -= lo % 2
    zyx = (2, 1, 0)
    window = (slice(None), *(slice(int(lo[a]), int(lo[a] + size[a])) for a in zyx))
    return grid[window], lo, size


def direct_crop(
    model: nn.Module,
    grid: Tensor,
    scene: OnDevice,
    cfg: GeneratorConfig,
    rng: np.random.Generator,
) -> tuple[Children, tuple[Tensor, Tensor]]:
    """Return a random crop's children with their targets, and its GT for the Chamfer.

    The GT points are in the crop's cells: those in the crop (at most `gt_cap`, the
    accuracy's targets) and those in its interior (at most `comp_cap`, the
    completeness'). The rim only gives context, so it is cut only where the crop has
    neighbours: at the grid's own border every voxel counts, or the border's voxels
    would never be scored (and a latent per voxel would stay at its initial value).
    """
    pts, gt, _ = scene
    sub, lo, size = crop_window(pts, grid, cfg, rng)
    device = sub.device
    extent = torch.from_numpy(size).float().to(device)
    cells = (gt - pts.lo) / pts.cell - torch.from_numpy(lo).float().to(device)
    in_crop = ((cells >= 0) & (cells < extent)).all(1)
    coarse = None
    if cfg.coarse_reach:
        truth = torch.zeros(sub.shape[1:], dtype=torch.bool, device=device)
        x, y, z = cells[in_crop].floor().long().unbind(1)
        truth[z, y, x] = True
        out, candidates, logit, wide = two_level(
            model, sub, cfg.coarse_reach, truth, amp=cfg.amp
        )
        coarse = (logit, F.max_pool3d(truth.float()[None, None], 2)[0, 0] > 0, wide)
    else:
        candidates = candidate_mask(sub, cfg.reach)
        if candidates.any():
            out = model(sub, candidates, amp=cfg.amp)
    if not candidates.any():
        raise EmptyCrop
    edge = (
        torch.from_numpy(lo > 0).to(device),
        torch.from_numpy(lo + size < pts.dims).to(device),
    )
    start = torch.where(edge[0], float(cfg.margin), 0.0)
    stop = torch.where(edge[1], extent - cfg.margin, extent)
    inner = ((cells >= start) & (cells < stop)).all(1)
    if not (in_crop.any() and inner.any()):
        raise EmptyCrop
    voxel = torch.nonzero(candidates).flip(1)
    occupied, vertex = child_targets(
        voxel, cells[in_crop], cfg.split, tuple(int(v) for v in size)
    )
    scored = ((voxel >= start) & (voxel < stop)).all(1)
    if coarse is not None:
        logit, held, wide = coarse
        parent = torch.nonzero(wide).flip(1)  # x y z, coarse cells
        live = ((2 * parent >= start) & (2 * parent + 2 <= stop)).all(1)
        index = tuple(parent[live].flip(1).unbind(1))
        coarse = (logit[index], held[index])
    children = Children(
        out.logit,
        out.pos,
        voxel.float()[:, None] + model.corner,
        occupied,
        vertex,
        scored,
        coarse,
    )
    truth = (cells[in_crop], cells[inner])
    return children, tuple(
        t[subsample(len(t), cap, rng, device)]
        for t, cap in zip(truth, (cfg.gt_cap, cfg.comp_cap), strict=True)
    )


def draw_direct(
    model: nn.Module,
    grid: Tensor,
    scene: OnDevice,
    cfg: GeneratorConfig,
    rng: np.random.Generator,
) -> tuple[Children, tuple[Tensor, Tensor]]:
    """Return `direct_crop` of the first of `CROP_TRIES` crops that has anything."""
    for _ in range(CROP_TRIES):
        try:
            return direct_crop(model, grid, scene, cfg, rng)
        except EmptyCrop:
            continue
    raise EmptyCrop


def crop_losses(
    model: nn.Module,
    grid: Tensor,
    scene: OnDevice,
    cfg: GeneratorConfig,
    rng: np.random.Generator,
) -> dict[str, Tensor]:
    """Losses of one random crop, centred on a random point, of a scene on the device.

    grid: the scene's voxel grid (`voxelize`); scene = (pts, gt, render): the pixels it
    is built from, their ground truth (n, 3), NaN-free, only for pixels that have one,
    and the render views and the views of the grid (only for `loss: render`).
    A generator with a coarse level is supervised there too, under the keys
    `<term>_coarse`. With `loss: render` the terms are the depth and coverage of the
    points drawn into the given views (`render_terms`), not the distances to the
    ground truth's points.
    """
    if cfg.loss == "direct":
        children = direct_crop(model, grid, scene, cfg, rng)[0]
        return direct_terms(children, cfg.split, vertex=not cfg.fixed_vertex)
    pts, gt, render = scene
    sub, lo, size = crop_window(pts, grid, cfg, rng)
    if model.seeded:
        origin = torch.from_numpy(lo).float().to(grid.device)
        near = (pts.points - pts.lo) / pts.cell - origin
        held = (
            (near >= 0) & (near < torch.from_numpy(size).float().to(grid.device))
        ).all(1)
        seeds = Seeds(near[held], pts.color[held], pts.log_conf[held])
        out = model(sub, seeds, amp=cfg.amp)
        voxel = seeds.pos.floor().long()
        start = seeds.pos[:, None].expand_as(out.pos)
    else:
        candidates = candidate_mask(sub, cfg.reach)
        out = model(sub, candidates, amp=cfg.amp)
        voxel = torch.nonzero(candidates).flip(1)
        start = (voxel.float() + 0.5)[:, None] + model.base
    device = out.pos.device
    origin = torch.from_numpy(lo).float().to(device)
    extent = torch.from_numpy(size).float().to(device)
    voxel = voxel.to(device)
    if cfg.loss == "render":
        assert render is not None
        points = pts.lo + (out.pos.reshape(-1, 3) + origin) * pts.cell
        box = (
            pts.lo + (origin + cfg.margin) * pts.cell,
            pts.lo + (origin + extent - cfg.margin) * pts.cell,
        )
        views = pick_views(render, box, (pts.cell, cfg.render_views), rng)
        return render_terms(
            render[0],
            views,
            (points, torch.sigmoid(out.logit.reshape(-1))),
            box,
            (pts.cell, cfg.tau),
        )
    cells = (gt - pts.lo) / pts.cell
    in_crop = ((cells >= origin) & (cells < origin + extent)).all(1)
    in_inner = (
        (cells >= origin + cfg.margin) & (cells < origin + extent - cfg.margin)
    ).all(1)
    targets, complete = cells[in_crop], cells[in_inner]
    if not (len(targets) and len(complete)):
        raise EmptyCrop
    crop = Crop(
        origin,
        extent,
        voxel,
        ((voxel >= cfg.margin) & (voxel < extent - cfg.margin)).all(1),
        targets[subsample(len(targets), cfg.gt_cap, rng, device)],
        complete[subsample(len(complete), cfg.comp_cap, rng, device)],
    )
    level = Level(out.pos, out.logit, start, model.distance)
    terms, reachable = level_losses(crop, level, cfg, rng)
    if out.coarse is not None:
        coarse_start = (voxel.float() + 0.5)[:, None] + model.coarse_base
        coarse = Level(*out.coarse, coarse_start, model.distance)
        terms |= {
            f"{k}_coarse": v for k, v in level_losses(crop, coarse, cfg, rng)[0].items()
        }
    if out.gate is not None:
        holds = reachable.view(len(voxel), -1).any(1)
        terms["gate"] = gate_loss(out.gate, holds, voxel.flip(1))
    return terms


def to_device(scene: TrainScene, device: torch.device) -> OnDevice:
    """Return the scene's pixels (all views), their GT and render views on a device."""
    pts = expand(scene.pts, torch.arange(len(scene.pts.points)), device)
    gt = scene.gt.to(device)[scene.has.to(device)]
    if scene.render is None:
        return pts, gt, None
    rv = scene.render.to(device)
    return pts, gt, (rv, list(range(len(rv.centre))))


def draw_crop(
    model: nn.Module,
    grid: Tensor,
    scene: OnDevice,
    cfg: GeneratorConfig,
    rng: np.random.Generator,
) -> dict[str, Tensor]:
    """Return `crop_losses` of the first of `CROP_TRIES` crops that has anything."""
    for _ in range(CROP_TRIES):
        try:
            return crop_losses(model, grid, scene, cfg, rng)
        except EmptyCrop:
            continue
    raise EmptyCrop


def scene_grid(model: nn.Module, pts: ScenePoints) -> Tensor:
    """Return the grid a model reads: the voxelized pixels, a latent table appended."""
    grid = voxelize(pts, semantics=model.semantics)
    if isinstance(model, LatentGenerator):
        assert grid.shape[1:] == model.table.shape[1:], "a latent table is one scene's"
        grid = torch.cat([grid, model.table])
    return grid


def step_loss(
    model: nn.Module,
    scene: TrainScene,
    cfg: GeneratorConfig,
    rng: np.random.Generator,
    device: torch.device,
) -> dict[str, Tensor]:
    """Return the terms of one random crop; the grid is built from all views."""
    on = to_device(scene, device)
    return draw_crop(model, scene_grid(model, on[0]), on, cfg, rng)


def direct_fixed_error(
    model: nn.Module,
    scenes: list[TrainScene],
    cfg: GeneratorConfig,
    device: torch.device,
) -> dict[str, float]:
    """Return the fixed error of a `direct` model: output Chamfer, child scores.

    Accuracy and completeness (cells) of the vertices with a positive logit, as the
    model's output would be scored, averaged over the crops (infinite where a crop keeps
    nothing); child precision and recall and the vertex error (child units), pooled.
    """
    model.eval()
    rng = np.random.default_rng(0)
    acc = comp = 0.0
    counts: dict[str, float] = {}
    crops = len(scenes) * cfg.fixed_crops
    with torch.no_grad():
        for scene in scenes:
            on = to_device(dataclasses.replace(scene, render=None), device)
            grid = scene_grid(model, on[0])
            for _ in range(cfg.fixed_crops):
                children, truth = draw_direct(model, grid, on, cfg, rng)
                a, c = output_chamfer(children, truth)
                acc, comp = acc + float(a) / crops, comp + float(c) / crops
                for key, value in child_counts(children, cfg.split).items():
                    counts[key] = counts.get(key, 0.0) + value
    return {
        "acc_cells": acc,
        "comp_cells": comp,
        "chamfer_cells": (acc + comp) / 2,
    } | child_scores(counts)


def fixed_error(
    model: nn.Module,
    scenes: list[TrainScene],
    cfg: GeneratorConfig,
    device: torch.device,
) -> dict[str, float]:
    """Mean accuracy and completeness (cells) on `fixed_crops` fixed crops per scene.

    The distances to the ground truth, whatever loss the model trains on (the points of
    a `direct` model are the kept vertices, with child scores beside): the same crops at
    every call, so the numbers of two steps and of two runs compare.
    """
    if cfg.loss == "direct":
        return direct_fixed_error(model, scenes, cfg, device)
    cfg = dataclasses.replace(cfg, loss="chamfer")
    model.eval()
    rng = np.random.default_rng(0)
    acc = comp = 0.0
    with torch.no_grad():
        for scene in scenes:
            on = to_device(dataclasses.replace(scene, render=None), device)
            grid = scene_grid(model, on[0])
            for _ in range(cfg.fixed_crops):
                terms = draw_crop(model, grid, on, cfg, rng)
                acc += float(terms["acc"]) / (len(scenes) * cfg.fixed_crops)
                comp += float(terms["comp"]) / (len(scenes) * cfg.fixed_crops)
    return {"acc_cells": acc, "comp_cells": comp, "chamfer_cells": (acc + comp) / 2}


def rate_factor(step: int, steps: int, *, cosine: bool) -> float:
    """Share of the learning rate at a step: 1, or a half cosine from 1 to 0 (SGDR)."""
    assert 0 <= step <= steps, (step, steps)
    return 0.5 * (1 + math.cos(math.pi * step / steps)) if cosine else 1.0


def train(
    cfg: GeneratorConfig,
    scenes: tuple[list[TrainScene], list[TrainScene]],
    device: torch.device,
    *,
    seed: int,
    log_dir: Path,
) -> tuple[nn.Module, list[dict]]:
    """Fit one generator on the first list of scenes; keep the step of least error.

    Every step logs the train loss and the loss of a random crop of a random scene of
    the second list (the validation scenes, in eval mode) to TensorBoard; every
    `eval_every` steps the fixed error on both lists, and the state with the least
    fixed error on the `select` list is kept (step 0, the untrained generator, is
    among the candidates, so the result is never worse than where training started).
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    fit, val = scenes
    watched = {"train": fit, "val": val}
    model = build(cfg, fit[0].pts.dims).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    decay = torch.optim.lr_scheduler.LambdaLR(
        opt, functools.partial(rate_factor, steps=cfg.steps, cosine=cfg.cosine)
    )
    writer = SummaryWriter(str(log_dir))
    log: list[dict] = []
    best, best_state = float("inf"), None
    t0 = time.perf_counter()
    for step in range(cfg.steps + 1):
        if step % cfg.eval_every == 0:
            err = {
                k: fixed_error(
                    model, spread(v, min(cfg.fixed_scenes, len(v))), cfg, device
                )
                for k, v in watched.items()
            }
            log.append({"step": step, "seconds": time.perf_counter() - t0, **err})
            for k, e in err.items():
                for term, value in e.items():
                    writer.add_scalar(f"fixed/{k}_{term}", value, step)
            print(f"step {step}: {log[-1]}", flush=True)  # noqa: T201
            if (key := err[cfg.select]["chamfer_cells"]) < best:
                best = key
                best_state = {
                    k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                }
                log[-1]["best"] = True
        if step == cfg.steps:
            break
        t_step = time.perf_counter()
        model.train()
        terms = step_loss(model, fit[rng.integers(len(fit))], cfg, rng, device)
        loss = sum(terms.values())
        assert torch.isfinite(loss), loss
        opt.zero_grad()
        loss.backward()
        if cfg.clip:
            nn.utils.clip_grad_norm_(model.parameters(), cfg.clip)
        opt.step()
        decay.step()
        model.eval()
        with torch.no_grad():
            held = step_loss(model, val[rng.integers(len(val))], cfg, rng, device)
        for name, t in ({"loss": loss} | terms).items():
            writer.add_scalar(f"train/{name}", float(t.detach()), step)
        writer.add_scalar("val/loss", float(sum(held.values())), step)
        for name, t in held.items():
            writer.add_scalar(f"val/{name}", float(t), step)
        writer.add_scalar("time/step_s", time.perf_counter() - t_step, step)
    writer.close()
    assert best_state is not None
    model.load_state_dict(best_state)
    return model, log


def profile(
    cfg: GeneratorConfig, scene: TrainScene, device: torch.device, steps: int = 12
) -> dict[str, float]:
    """Median seconds per stage of a training step, the first two steps discarded.

    Stages (rung 2 of the ladder: data and backprop apart): the scene to the device,
    the grid, the forward pass with the loss, the backward pass with the optimizer,
    the validation crop that follows each step; also the peak GPU memory in GiB.
    """
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    model = build(cfg, scene.pts.dims).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    stages: dict[str, list[float]] = {
        k: [] for k in ("to_device", "grid", "forward_loss", "backward", "val")
    }

    def timed(name: str, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:  # noqa: ANN401 - any stage
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = fn(*args, **kwargs)
        if device.type == "cuda":
            torch.cuda.synchronize()
        stages[name].append(time.perf_counter() - t0)
        return out

    def update(loss: Tensor) -> None:
        opt.zero_grad()
        loss.backward()
        opt.step()

    for _ in range(steps):
        model.train()
        on = timed("to_device", to_device, scene, device)
        grid = timed("grid", scene_grid, model, on[0])
        terms = timed("forward_loss", crop_losses, model, grid, on, cfg, rng)
        timed("backward", update, sum(terms.values()))
        model.eval()
        with torch.no_grad():
            timed("val", step_loss, model, scene, cfg, rng, device)
    out = {k: float(np.median(v[2:])) for k, v in stages.items()}
    out["step_s"] = sum(out.values())
    out["parameters"] = float(sum(p.numel() for p in model.parameters()))
    out["peak_gib"] = (
        torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else 0.0
    )
    return out


def scene_candidates(
    model: nn.Module, grid: Tensor, cfg: GeneratorConfig
) -> tuple[Outputs, Tensor]:
    """Return a whole scene's output and candidates (Z, Y, X); never reads the GT."""
    if cfg.coarse_reach:
        out, candidates, _, _ = two_level(
            model, grid, cfg.coarse_reach, None, amp=cfg.amp
        )
        return out, candidates
    candidates = candidate_mask(grid, cfg.reach)
    return model(grid, candidates, amp=cfg.amp), candidates


def scored_points(
    model: nn.Module, pts: ScenePoints, device: torch.device, cfg: GeneratorConfig
) -> tuple[Tensor, Tensor, Tensor]:
    """Return every candidate point (n, 3), its logit (n,) and lattice position (n, 3).

    The points are in the normalized frame of the grid; the lattice positions in its
    cells, exact (they lie on quarter cells, where a round trip through the normalized
    frame would move them across the borders of the reachability grid). All on the CPU.
    """
    model.eval()
    with torch.no_grad():
        on = expand(compact(pts), torch.arange(len(pts.points)), device)
        grid = scene_grid(model, on)
        if model.seeded:
            seeds = Seeds((on.points - on.lo) / on.cell, on.color, on.log_conf)
            out = model(grid, seeds, amp=cfg.amp)
            home = seeds.pos[:, None].expand_as(out.pos).reshape(-1, 3)
        else:
            out, candidates = scene_candidates(model, grid, cfg)
            centre = torch.nonzero(candidates).flip(1).float() + 0.5
            home = (centre[:, None] + model.base).reshape(-1, 3)
    points = (on.lo + out.pos.reshape(-1, 3) * on.cell).cpu()
    return points, out.logit.reshape(-1).cpu(), home.cpu()


def scene_children(
    model: nn.Module,
    pts: ScenePoints,
    gt: Tensor,
    cfg: GeneratorConfig,
    device: torch.device,
) -> Children:
    """Return a scene's candidate children with targets; gt (n, 3): its GT points."""
    model.eval()
    with torch.no_grad():
        on = expand(compact(pts), torch.arange(len(pts.points)), device)
        grid = scene_grid(model, on)
        out, candidates = scene_candidates(model, grid, cfg)
        voxel = torch.nonzero(candidates).flip(1)
        occupied, vertex = child_targets(
            voxel, (gt.to(device) - on.lo) / on.cell, cfg.split, on.dims
        )
    scored = torch.ones(len(voxel), dtype=torch.bool, device=device)
    corner = voxel.float()[:, None] + model.corner
    return Children(out.logit, out.pos, corner, occupied, vertex, scored)


def generate(
    model: nn.Module,
    pts: ScenePoints,
    scale: float,
    device: torch.device,
    cfg: GeneratorConfig,
) -> np.ndarray:
    """Return the cloud (M, 3) in Deja View's frame: points with a positive logit."""
    points, logit, _ = scored_points(model, pts, device, cfg)
    out = (points[logit > 0] * scale).numpy().astype(np.float32)
    assert out.ndim == 2 and out.shape[1] == 3 and np.isfinite(out).all()  # noqa: PLR2004
    return out


def run_model(
    cfg: GeneratorConfig,
    name: str,
    scenes: tuple[list[TrainScene], list[TrainScene]],
    device: torch.device,
    out: Path,
) -> None:
    """Train the model `name`, or load it and its curve from `out`; write its clouds."""
    saved = out / f"{name}.pt"
    record: dict = {"config": dataclasses.asdict(cfg) | {"root": str(cfg.root)}}
    if saved.exists():
        model = build(cfg, scenes[0][0].pts.dims).to(device)
        model.load_state_dict(torch.load(saved, map_location=device))
        old = out / f"{name}.yaml"
        if old.exists():
            kept = yaml.safe_load(old.read_text())
            record |= {k: kept[k] for k in ("seconds", "curve") if k in kept}
    else:
        t0 = time.perf_counter()
        seed = int(name.rsplit("-s", 1)[1])
        model, log = train(cfg, scenes, device, seed=seed, log_dir=out / "tb" / name)
        record |= {"seconds": time.perf_counter() - t0, "curve": log}
        torch.save(model.state_dict(), saved)
    record["generate_s"] = {}
    record["children"] = {}
    for scene in cfg.predict:
        pts, gt, has, scale = load_scene(cfg, scene, cfg.semantics)
        t0 = time.perf_counter()
        cloud = generate(model, pts, scale, device, cfg)
        record["generate_s"][scene] = time.perf_counter() - t0
        if cfg.loss == "direct":
            children = scene_children(model, pts, gt[has], cfg, device)
            record["children"][scene] = child_scores(child_counts(children, cfg.split))
        np.savez_compressed(
            cfg.root / scene / f"{name}_{cfg.views}_cloud.npz", points=cloud
        )
    record["git"] = git_hash()
    (out / f"{name}.yaml").write_text(yaml.safe_dump(record, sort_keys=False))
    print(f"{name}: wrote {len(cfg.predict)} clouds", flush=True)  # noqa: T201


def main() -> None:
    """Train the arms of one rung per size and seed, write each model's clouds.

    `--out` resumes a run: a model already saved there is loaded instead of trained.
    `--profile` times the stages of a step of every arm on the first train scene
    instead of training. `--set` overrides config fields of every arm (an escalation).
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--rung", required=True, help="a key of the config's `rungs`")
    parser.add_argument("--arms", nargs="+", help="arms to run; default: the rung's")
    parser.add_argument("--sizes", nargs="+", type=int, help="train scenes per model")
    parser.add_argument("--seeds", nargs="+", type=int, help="default: the config's")
    parser.add_argument("--out", type=Path, help="default: runs/<tag>/<rung>")
    parser.add_argument("--lr-file", type=Path, help="learning rate per arm (yaml)")
    parser.add_argument(
        "--set",
        nargs="+",
        default=[],
        metavar="KEY=VALUE",
        help="config fields to override for every arm, values in yaml (steps=4800)",
    )
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    overrides = dict(item.split("=", 1) for item in args.set)
    overrides = {k: yaml.safe_load(v) for k, v in overrides.items()}
    rung = load_rung(args.config, args.rung)
    device = best_device()
    out = args.out or Path("runs") / rung.cfg.tag / args.rung
    lrs = yaml.safe_load(args.lr_file.read_text()) if args.lr_file else None
    out.mkdir(parents=True, exist_ok=True)
    loaded: dict[tuple, list[TrainScene]] = {}

    def scenes_of(cfg: GeneratorConfig, names: list[str]) -> list[TrainScene]:
        key = (cfg.semantics, cfg.dino, cfg.loss == "render", tuple(names))
        if key not in loaded:
            loaded[key] = load_scenes(cfg, names)
        return loaded[key]

    result: dict[str, dict] = {}
    for arm in args.arms or rung.run:
        cfg = dataclasses.replace(for_arm(rung, arm, lrs), **overrides)
        if args.profile:
            result[arm] = profile(cfg, scenes_of(cfg, cfg.train[:1])[0], device)
            print(f"{arm}: {result[arm]}", flush=True)  # noqa: T201
            continue
        for size in args.sizes or rung.sizes:
            scenes = scenes_of(cfg, spread(cfg.train, size)), scenes_of(cfg, cfg.val)
            for seed in args.seeds or cfg.seeds:
                tail = (f"-n{size}" if size else "") + f"-s{seed}"
                name = f"{cfg.tag}-{args.rung}-{arm}{tail}"
                run_model(cfg, name, scenes, device, out)
    if args.profile:
        (out / "profile.yaml").write_text(yaml.safe_dump(result, sort_keys=False))
    print(f"wrote {out}")  # noqa: T201


if __name__ == "__main__":
    main()
