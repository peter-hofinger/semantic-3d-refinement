"""Dense bundle adjustment of a backbone's output (issue #82): training-free, learned.

Déjà View's views agree with each other (E56a, E52e), but its cameras and depths are
fitted per forward pass, not jointly against pixel matches. Here they are: dense
matches between overlapping views come from RAFT optical flow (Teed and Deng 2020),
and one Gauss-Newton bundle adjustment moves every view's pose and a coarse inverse
depth map to agree with them, as DROID-SLAM's dense BA layer (Teed and Deng 2021).

Unknowns: per view a pose update (omega, t), applied as R <- Exp(omega) R, c <- c + t
to the camera-to-world rotation and centre, and the inverse depth rho of the grid
pixels p (stride s) of every view. A match takes p in view i to q in view j:

    X = c_i + R_i K_i^-1 p / rho,    r = pi(K_j R_j^T (X - c_j)) - q,

    E = 1/2 sum_matches w h(r) + 1/2 lambda sum_p ((rho - rho_0) / rho_0)^2,

h the Cauchy loss (iteratively reweighted; consistent but wrong RAFT matches, tens
of pixels off, steered the cameras under a Huber loss), rho_0 the backbone's inverse depth (the
prior fixes the scale that matches alone leave free), view 0's pose fixed (the
gauge). Each step solves the normal equations with the depths eliminated (Schur
complement), so the reduced system has only 6V unknowns. The refined depth is the
backbone's full-resolution depth times the grid's ratio rho_0 / rho, upsampled.

Weights w: training-free, exp(-(e / sigma)^2) of RAFT's forward-backward error e;
learned, a small MLP on per-match features (that error, both confidences, the colour
difference, the starting residual and depth disagreement and, with semantics,
whether both ends lie in the same linked SAM 3 object), trained through the unrolled
solver on the Hypersim train scenes against the ground truth's points.
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
import yaml
from torch import Tensor, nn

from src.analysis.scene_eval import git_hash
from src.geometry import Geometry
from src.loaders.ground_truth import GroundTruth
from src.loaders.hypersim.config import load_config
from src.loaders.views import Views
from src.refinement.multiview import Scene, pinhole_rays, scene_from_geometry
from src.segmentation.base import Segmentation
from src.segmentation.linked import working_associator

log = logging.getLogger(__name__)
SEMANTICS = ("none", "sam3")
FEATURES = 7  # per-match features of the learned weights, without the semantic flag
MIN_DEPTH = 1e-6  # a match whose point is not in front of view j is dropped
MU_START, MU_MIN, MU_MAX = 1e-2, 1e-6, 1e6  # Levenberg-Marquardt damping
SMALL_ANGLE = 1e-6  # below it, log uses its limit theta / (2 sin theta) -> 1/2
RHO_RANGE = (0.5, 2.0)  # inverse depth stays within this factor of the backbone's


@dataclass
class BundleConfig:
    """Settings of the solver, the matches and the learned weights."""

    root: Path
    views: str
    backbone: str
    segmentation: str
    train: list[str]
    held_out: int  # last train scenes: the error the kept state is chosen on
    predict: list[str]
    stride: int  # grid of depth unknowns, in pixels
    window: int  # each view is matched to the views this many steps along the path
    iterations: int  # Gauss-Newton steps
    robust: float  # Cauchy scale delta, pixels
    sigma: float  # forward-backward error scale of the training-free weights, pixels
    prior: float  # lambda, the inverse-depth prior
    damping: float  # Levenberg-Marquardt term on the poses
    pose_prior: (
        float  # alpha: pulls each pose to the backbone's (centre in median depths)
    )
    models: dict[str, str]  # learned models: name -> semantics
    hidden: int
    steps: int
    lr: float
    seeds: list[int]
    free: dict[str, float] = dataclasses.field(default_factory=dict)  # name -> prior

    @classmethod
    def load(cls, path: Path) -> BundleConfig:
        """Read the config; scenes come from Hypersim splits."""
        raw = yaml.safe_load(path.read_text())
        split = load_config(Path(raw.pop("split")))
        for key in ("train", "predict"):  # `<key>_scenes` names scenes (smoke runs)
            if f"{key}_scenes" in raw:
                raw[key] = raw.pop(f"{key}_scenes")
            else:
                scenes = split.scenes(raw.pop(f"{key}_splits"))
                raw[key] = [f"{s}_{c}" for s, c in scenes.items()]
        raw["root"] = Path(raw["root"])
        assert all(s in SEMANTICS for s in raw["models"].values()), raw["models"]
        return cls(**raw)


def skew(v: Tensor) -> Tensor:
    """Cross-product matrices (..., 3, 3) of vectors (..., 3)."""
    z = torch.zeros_like(v[..., 0])
    x, y, w = v.unbind(-1)
    return torch.stack(
        [
            torch.stack([z, -w, y], -1),
            torch.stack([w, z, -x], -1),
            torch.stack([-y, x, z], -1),
        ],
        -2,
    )


def so3_log(rotation: Tensor) -> Tensor:
    """Axis-angle vectors (..., 3) of rotation matrices, angles below pi."""
    cos = ((torch.diagonal(rotation, dim1=-2, dim2=-1).sum(-1) - 1) / 2).clamp(-1, 1)
    theta = torch.arccos(cos)
    vee = torch.stack(
        [
            rotation[..., 2, 1] - rotation[..., 1, 2],
            rotation[..., 0, 2] - rotation[..., 2, 0],
            rotation[..., 1, 0] - rotation[..., 0, 1],
        ],
        -1,
    )
    factor = torch.where(
        theta > SMALL_ANGLE, theta / (2 * torch.sin(theta).clamp_min(1e-12)), 0.5
    )
    return factor[..., None] * vee


def pose_pull(scene: Scene, state: State, cfg: BundleConfig) -> tuple[Tensor, Tensor]:
    """Diagonal (6V) and gradient (6V) of the pose prior at the state.

    1/2 alpha (|log(R R_0^T)|^2 + |c - c_0|^2 / m^2), m the backbone's median depth;
    the Gauss-Newton diagonal with first-order rotation increments.
    """
    m = scene.depth.double().median()
    rot = so3_log(state.rotation @ scene.rotation.double().transpose(1, 2))
    cen = state.center - scene.center.double()
    per_view = torch.cat([torch.ones(3), torch.full((3,), 1.0 / float(m) ** 2)])
    diag = cfg.pose_prior * per_view.to(cen).repeat(len(cen))
    grad = diag * torch.cat([rot, cen], -1).reshape(-1)
    return diag, grad


def so3_exp(omega: Tensor) -> Tensor:
    """Rotation matrices (..., 3, 3) of axis-angle vectors (Rodrigues)."""
    theta = omega.norm(dim=-1, keepdim=True).clamp_min(1e-12)[..., None]
    k = skew(omega) / theta
    eye = torch.eye(3, dtype=omega.dtype, device=omega.device).expand_as(k)
    return eye + torch.sin(theta) * k + (1 - torch.cos(theta)) * k @ k


@dataclass
class Matches:
    """Directed view pairs (P, 2) and, per grid pixel of the first view, its match."""

    pairs: Tensor  # (P, 2) long: i, j
    target: Tensor  # (P, h, w, 2) matched pixel q in view j (pixel centres at integers)
    fb: Tensor  # (P, h, w) forward-backward error, pixels; inf where q leaves the image
    same: Tensor  # (P, h, w) bool, both ends in one linked object (False without masks)

    def to(self, device: torch.device) -> Matches:
        """Move every tensor to the device."""
        return Matches(
            *(getattr(self, f).to(device) for f in self.__dataclass_fields__)
        )


def grid(h: int, w: int, stride: int, device: torch.device) -> Tensor:
    """Pixel indices (gh, gw, 2) as x, y of the grid's pixels, centred in each cell."""
    ys = torch.arange(stride // 2, h, stride, device=device, dtype=torch.float64)
    xs = torch.arange(stride // 2, w, stride, device=device, dtype=torch.float64)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([xx, yy], -1)


def sample(maps: Tensor, xy: Tensor) -> Tensor:
    """Bilinear sample of maps (N, C, H, W) at pixel indices xy (N, ..., 2)."""
    _, _, h, w = maps.shape
    norm = torch.stack([2 * xy[..., 0] / (w - 1) - 1, 2 * xy[..., 1] / (h - 1) - 1], -1)
    flat = norm.reshape(len(maps), 1, -1, 2).to(maps.dtype)
    out = F.grid_sample(maps, flat, align_corners=True, padding_mode="border")
    return out.reshape(len(maps), maps.shape[1], *xy.shape[1:-1])


def window_pairs(views: int, window: int) -> Tensor:
    """Directed pairs of each view with the views up to `window` steps along the path.

    The views are sampled in order along the camera path, so neighbours overlap; pairs
    chosen by predicted overlap instead took far views RAFT cannot match (E82a smoke).
    """
    pairs = [
        (i, j) for i in range(views) for j in range(views) if 0 < abs(i - j) <= window
    ]
    return torch.tensor(pairs, dtype=torch.long)


def raft_matches(
    images: Tensor, pairs: Tensor, stride: int, model: nn.Module, batch: int = 8
) -> tuple[Tensor, Tensor]:
    """RAFT matches of each pair's grid pixels: targets (P, gh, gw, 2), fb error."""
    _, h, w, _ = images.shape
    x = images.permute(0, 3, 1, 2).float() / 127.5 - 1
    ph, pw = -h % 8, -w % 8
    x = F.pad(x, (0, pw, 0, ph), mode="replicate")
    p = grid(h, w, stride, images.device)
    targets, errors = [], []
    for s in range(0, len(pairs), batch):
        i, j = pairs[s : s + batch].T
        with torch.no_grad():
            fwd = model(x[i], x[j])[-1][..., :h, :w].double()
            bwd = model(x[j], x[i])[-1][..., :h, :w].double()
        n = len(i)
        q = p.expand(n, *p.shape) + sample(fwd, p.expand(n, *p.shape)).permute(
            0, 2, 3, 1
        )
        back = q + sample(bwd, q).permute(0, 2, 3, 1)
        fb = (back - p).norm(dim=-1)
        inside = (
            (q[..., 0] >= 0)
            & (q[..., 0] <= w - 1)
            & (q[..., 1] >= 0)
            & (q[..., 1] <= h - 1)
        )
        targets.append(q)
        errors.append(torch.where(inside, fb, torch.full_like(fb, torch.inf)))
    return torch.cat(targets), torch.cat(errors)


def same_object(ids: Tensor, pairs: Tensor, target: Tensor, stride: int) -> Tensor:
    """Whether both ends of each match carry one linked id (nearest pixel, >= 0)."""
    _, h, w = ids.shape
    p = grid(h, w, stride, ids.device).round().long()
    a = ids[pairs[:, 0]][:, p[..., 1], p[..., 0]]
    q = target.round().long()
    qx, qy = q[..., 0].clamp(0, w - 1), q[..., 1].clamp(0, h - 1)
    b = ids[pairs[:, 1][:, None, None], qy, qx]
    return (a == b) & (a >= 0)


@dataclass
class State:
    """The solver's variables: cameras and the grid's inverse depth (V, gh, gw)."""

    rotation: Tensor
    center: Tensor
    rho: Tensor


def residuals(
    scene: Scene, state: State, matches: Matches, p: Tensor
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Residuals (N, 2) of all matches and their Jacobians at zero update.

    Returns r and the Jacobians w.r.t. omega_i, t_i, omega_j, t_j (N, 2, 3 each) and
    rho (N, 2), flattened over pairs and grid pixels.
    """
    i, j = matches.pairs[:, 0], matches.pairs[:, 1]
    k_inv = torch.linalg.inv(scene.k.double())
    pix = F.pad(p, (0, 1), value=1.0)  # (gh, gw, 3)
    a = torch.einsum(
        "pab,pbc,hwc->phwa", state.rotation[i], k_inv[i], pix
    )  # R_i K_i^-1 p
    rho = state.rho[i]
    x = state.center[i][:, None, None] + a / rho[..., None]
    y = x - state.center[j][:, None, None]
    rt = state.rotation[j].transpose(1, 2)
    d = torch.einsum("pab,phwb->phwa", rt, y)
    kj = scene.k.double()[j]
    fx, fy = kj[:, 0, 0, None, None], kj[:, 1, 1, None, None]
    cx, cy = kj[:, 0, 2, None, None], kj[:, 1, 2, None, None]
    dz = d[..., 2]
    u = torch.stack([fx * d[..., 0] / dz + cx, fy * d[..., 1] / dz + cy], -1)
    r = torch.where(
        (dz > MIN_DEPTH)[..., None], u - matches.target, torch.nan
    )  # behind view j: dropped
    zero = torch.zeros_like(dz)
    proj = torch.stack(
        [
            torch.stack([fx / dz + zero, zero, -fx * d[..., 0] / dz**2], -1),
            torch.stack([zero, fy / dz + zero, -fy * d[..., 1] / dz**2], -1),
        ],
        -2,
    )  # (P, gh, gw, 2, 3)
    pr = proj @ rt[:, None, None]  # d u / d X
    j_ti = pr
    j_wi = pr @ -skew(a / rho[..., None])
    j_tj = -pr
    j_wj = pr @ skew(y)
    j_rho = (pr @ (-a / rho[..., None] ** 2)[..., None])[..., 0]
    flat = [t.reshape(-1, *t.shape[3:]) for t in (r, j_wi, j_ti, j_wj, j_tj, j_rho)]
    return tuple(flat)  # type: ignore[return-value]


def solve_step(  # noqa: PLR0913  state, data, weights and settings stay separate inputs
    scene: Scene,
    state: State,
    matches: Matches,
    weight: Tensor,
    cfg: BundleConfig,
    *,
    mu: float,
) -> State:
    """One Levenberg-Marquardt step (diagonal scaled by 1 + mu), depths eliminated.

    View 0 stays fixed.
    """
    rho0 = start(scene, cfg.stride).rho
    v, gh, gw = state.rho.shape
    p = grid(*scene.depth.shape[1:], cfg.stride, scene.depth.device)
    r, j_wi, j_ti, j_wj, j_tj, j_rho = residuals(scene, state, matches, p)
    norm = r.detach().norm(dim=-1).clamp_min(1e-9)
    w = weight.reshape(-1) / (1 + (norm / cfg.robust) ** 2)  # Cauchy IRLS
    w = torch.where(torch.isfinite(r).all(-1), w, torch.zeros_like(w))
    r = torch.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0)
    jp = torch.cat([j_wi, j_ti, j_wj, j_tj], -1).detach()  # (N, 2, 12)
    jd = torch.nan_to_num(j_rho.detach())
    jp = torch.nan_to_num(jp)
    n_pix = gh * gw
    i, j = matches.pairs[:, 0], matches.pairs[:, 1]
    pose_index = torch.stack([i, j], 1).repeat_interleave(n_pix, 0)  # (N, 2)
    depth_index = (i[:, None] * n_pix + torch.arange(n_pix, device=i.device)).reshape(
        -1
    )
    cols = (pose_index[:, :, None] * 6 + torch.arange(6, device=i.device)).reshape(
        -1, 12
    )
    wr = w[:, None] * r
    hpp = torch.zeros(6 * v, 6 * v, dtype=r.dtype, device=r.device)
    blocks = torch.einsum("nka,n,nkb->nab", jp, w, jp)
    hpp.index_put_(
        (cols[:, :, None].expand(-1, 12, 12), cols[:, None, :].expand(-1, 12, 12)),
        blocks,
        accumulate=True,
    )
    gp = torch.zeros(6 * v, dtype=r.dtype, device=r.device).index_add_(
        0, cols.reshape(-1), torch.einsum("nka,nk->na", jp, wr).reshape(-1)
    )
    hpd = torch.zeros(v * n_pix, 6 * v, dtype=r.dtype, device=r.device)
    hpd.index_put_(
        (depth_index[:, None].expand(-1, 12), cols),
        torch.einsum("nka,n,nk->na", jp, w, jd),
        accumulate=True,
    )
    rho_flat, rho0_flat = state.rho.reshape(-1), rho0.reshape(-1)
    hdd = torch.zeros_like(rho_flat).index_add_(
        0, depth_index, (w[:, None] * jd**2).sum(-1)
    )
    gd = torch.zeros_like(rho_flat).index_add_(0, depth_index, (jd * wr).sum(-1))
    hdd = (hdd + cfg.prior / rho0_flat**2) * (1 + mu)
    gd = gd + cfg.prior * (rho_flat - rho0_flat) / rho0_flat**2
    keep = torch.arange(6, 6 * v, device=r.device)  # view 0 fixed
    eye = torch.eye(6 * v, dtype=r.dtype, device=r.device)
    pull, g_pose = pose_pull(scene, state, cfg)
    hpp = hpp + torch.diag(pull)
    gp = gp + g_pose
    hpp = hpp + mu * torch.diag(torch.diagonal(hpp)) + cfg.damping * eye
    s = hpp - hpd.T @ (hpd / hdd[:, None])
    rhs = gp - hpd.T @ (gd / hdd)
    s = s[keep][:, keep]
    dp = torch.zeros(6 * v, dtype=r.dtype, device=r.device)
    dp = dp.index_put((keep,), -torch.linalg.solve(s, rhs[keep]))
    dd = -(gd + hpd @ dp) / hdd
    dp = dp.reshape(v, 6)
    rotation = so3_exp(dp[:, :3]) @ state.rotation
    rho = (rho_flat + dd).reshape(v, gh, gw)
    rho = torch.minimum(torch.maximum(rho, rho0 * RHO_RANGE[0]), rho0 * RHO_RANGE[1])
    return State(rotation, state.center + dp[:, 3:], rho)


def start(scene: Scene, stride: int) -> State:
    """Return the backbone's cameras and its inverse depth at the grid pixels."""
    v, h, w = scene.depth.shape
    p = grid(h, w, stride, scene.depth.device)
    depth = sample(scene.depth[:, None].double(), p.expand(v, *p.shape))[:, 0]
    return State(scene.rotation.double(), scene.center.double(), 1 / depth)


def energy(
    scene: Scene, state: State, matches: Matches, weight: Tensor, cfg: BundleConfig
) -> float:
    """E of the module docstring: weighted Cauchy cost of the residuals and the priors.

    A match whose point falls behind view j costs a residual of the image's size: a
    zero cost would let the solver drop matches by turning cameras away.
    """
    p = grid(*scene.depth.shape[1:], cfg.stride, scene.depth.device)
    n = residuals(scene, state, matches, p)[0].norm(dim=-1)
    n = torch.nan_to_num(n, nan=float(max(scene.depth.shape[1:])))  # behind view j
    w = weight.reshape(-1).detach()
    cost = 0.5 * cfg.robust**2 * torch.log1p((n / cfg.robust) ** 2)
    rho0 = start(scene, cfg.stride).rho
    prior = 0.5 * cfg.prior * ((state.rho - rho0) / rho0).pow(2).sum()
    pull, g_pose = pose_pull(scene, state, cfg)
    deviation = g_pose / pull.clamp_min(1e-30)
    poses = 0.5 * (pull * deviation**2).sum()
    return float((w * torch.nan_to_num(cost)).sum() + prior + poses)


def adjust(scene: Scene, matches: Matches, weight: Tensor, cfg: BundleConfig) -> State:
    """Run `cfg.iterations` Levenberg-Marquardt steps from the backbone's state.

    A step is kept only if it lowers the energy; mu shrinks tenfold after a kept step
    and grows tenfold after a rejected one.
    """
    state = start(scene, cfg.stride)
    mu = MU_START
    with torch.no_grad():
        current = energy(scene, state, matches, weight, cfg)
    for _ in range(cfg.iterations):
        candidate = solve_step(scene, state, matches, weight, cfg, mu=mu)
        with torch.no_grad():
            value = energy(scene, candidate, matches, weight, cfg)
        if value < current:
            state, current, mu = candidate, value, max(mu / 10, MU_MIN)
        else:
            mu = min(mu * 10, MU_MAX)
    return state


def free_weights(matches: Matches, sigma: float) -> Tensor:
    """Training-free weights exp(-(e / sigma)^2) of the forward-backward error."""
    return torch.exp(-((matches.fb / sigma) ** 2))


def features(scene: Scene, matches: Matches, images: Tensor, stride: int) -> Tensor:
    """Per-match features (P, gh, gw, FEATURES) of the learned weights."""
    _, h, w = scene.depth.shape
    p = grid(h, w, stride, scene.depth.device)
    i, j = matches.pairs[:, 0], matches.pairs[:, 1]
    n = len(i)
    pp = p.expand(n, *p.shape)
    conf = scene.conf[:, None].double().clamp_min(1e-6).log()
    rgb = images.permute(0, 3, 1, 2).double() / 255
    ci, cj = sample(conf[i], pp)[:, 0], sample(conf[j], matches.target)[:, 0]
    colour = (sample(rgb[i], pp) - sample(rgb[j], matches.target)).abs().sum(1)
    state = start(scene, stride)
    r = residuals(scene, state, matches, p)[0].reshape(n, *p.shape[:2], 2)
    depth_j = sample(scene.depth[j][:, None].double(), matches.target)[:, 0]
    k_inv = torch.linalg.inv(scene.k.double())
    a = torch.einsum(
        "pab,pbc,hwc->phwa", state.rotation[i], k_inv[i], F.pad(p, (0, 1), value=1.0)
    )
    x = state.center[i][:, None, None] + a / state.rho[i][..., None]
    z = torch.einsum(
        "pa,phwa->phw", state.rotation[j][:, :, 2], x - state.center[j][:, None, None]
    )
    disagree = (z.clamp_min(1e-6) / depth_j.clamp_min(1e-6)).log()
    fb = matches.fb.clamp(max=50.0)
    out = torch.stack(
        [
            fb,
            ci,
            cj,
            colour,
            r.norm(dim=-1).clamp(max=50.0),
            disagree,
            torch.isfinite(matches.fb).double(),
        ],
        -1,
    )
    return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


class MatchWeights(nn.Module):
    """A small MLP from per-match features (and the same-object flag) to a weight."""

    def __init__(self, hidden: int, *, semantics: bool) -> None:
        """Two hidden layers; the last bias starts the weights near 0.5."""
        super().__init__()
        self.semantics = semantics
        self.net = nn.Sequential(
            nn.Linear(FEATURES + semantics, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, feats: Tensor, same: Tensor) -> Tensor:
        """Weights (P, gh, gw)."""
        x = (
            torch.cat([feats, same[..., None].to(feats.dtype)], -1)
            if self.semantics
            else feats
        )
        return torch.sigmoid(self.net(x.float())[..., 0]).double()


def refined_geometry(
    scene: Scene, geometry: Geometry, state: State, stride: int
) -> Geometry:
    """Full-resolution geometry: the backbone's depth times the upsampled grid ratio."""
    v, h, w = scene.depth.shape
    rho0 = start(scene, stride).rho
    ratio = (rho0 / state.rho)[:, None]
    up = F.interpolate(ratio, size=(h, w), mode="bilinear", align_corners=False)[:, 0]
    depth = scene.depth.double() * up
    rays = pinhole_rays(scene.k.double(), state.rotation, state.center, h, w)
    c2w = torch.eye(4, dtype=torch.float64, device=depth.device).repeat(v, 1, 1)
    c2w[:, :3, :3], c2w[:, :3, 3] = state.rotation, state.center
    k = scene.k.double().clone()
    k[:, :2, 2] += 0.5  # Geometry's intrinsics put pixel corners at integers (COLMAP)
    out = Geometry(
        depth=depth.float().cpu().numpy(),
        confidence=geometry.confidence,
        intrinsics=k.float().cpu().numpy(),
        cam_to_world=c2w.float().cpu().numpy(),
        ray_origins=rays[..., :3].float().cpu().numpy(),
        ray_directions=rays[..., 3:].float().cpu().numpy(),
    )
    assert np.isfinite(out.depth).all() and (out.depth > 0).all()
    return out


@dataclass
class SceneData:
    """Everything the solver and the loss need for one scene; kept on the CPU."""

    name: str
    scene: Scene  # training scenes keep no ray maps (the solver uses K, R, c only)
    geometry: Geometry | None  # only the scenes written out
    matches: Matches
    feats: Tensor
    gt_points: (
        Tensor | None
    )  # (V, gh, gw, 3) ground-truth points at the grid pixels, NaN where none

    def to(self, device: torch.device) -> SceneData:
        """Return the tensors on the device."""
        gt = None if self.gt_points is None else self.gt_points.to(device)
        return SceneData(
            self.name,
            self.scene.to(device),
            self.geometry,
            self.matches.to(device),
            self.feats.to(device),
            gt,
        )


def matches_path(cfg: BundleConfig, name: str) -> Path:
    """Cache of a scene's RAFT matches, next to its views."""
    return cfg.root / name / f"raft-s{cfg.stride}-w{cfg.window}_{cfg.views}.npz"


def compute_matches(cfg: BundleConfig, names: list[str], device: torch.device) -> None:
    """Cache RAFT matches and same-object flags of every scene; existing caches stay."""
    from torchvision.models.optical_flow import Raft_Large_Weights, raft_large  # noqa: PLC0415  only this stage needs it

    model = raft_large(weights=Raft_Large_Weights.DEFAULT).to(device).eval()
    for name in names:
        out = matches_path(cfg, name)
        if out.exists():
            continue
        t0 = time.perf_counter()
        folder = cfg.root / name
        views = Views.load(folder / f"{cfg.views}.npz")
        geometry = Geometry.load(folder / f"{cfg.backbone}_{cfg.views}.npz")
        scene = scene_from_geometry(geometry).to(device)
        pairs = window_pairs(len(scene.center), cfg.window).to(device)
        images = torch.from_numpy(views.images).to(device)
        target, fb = raft_matches(images, pairs, cfg.stride, model)
        masks = Segmentation.load(folder / f"{cfg.segmentation}_{cfg.views}.npz")
        linked = working_associator().associate(masks, geometry)
        same = same_object(
            torch.from_numpy(linked.instance_ids).to(device), pairs, target, cfg.stride
        )
        np.savez_compressed(
            out,
            pairs=pairs.cpu().numpy(),
            target=target.cpu().numpy().astype(np.float32),
            fb=fb.cpu().numpy().astype(np.float32),
            same=same.cpu().numpy(),
        )
        seconds = time.perf_counter() - t0
        print(f"{name}: {len(pairs)} pairs, {seconds:.1f} s", flush=True)  # noqa: T201


def load_scene(
    cfg: BundleConfig, name: str, device: torch.device, *, truth: bool
) -> SceneData:
    """Load a scene's backbone output, cached matches and, to train, its GT points."""
    folder = cfg.root / name
    views = Views.load(folder / f"{cfg.views}.npz")
    geometry = Geometry.load(folder / f"{cfg.backbone}_{cfg.views}.npz")
    scene = scene_from_geometry(geometry).to(device)
    with np.load(matches_path(cfg, name)) as f:
        matches = Matches(
            torch.from_numpy(f["pairs"]).long(),
            torch.from_numpy(f["target"]).double(),
            torch.from_numpy(f["fb"]).double(),
            torch.from_numpy(f["same"]),
        ).to(device)
    images = torch.from_numpy(views.images).to(device)
    feats = features(scene, matches, images, cfg.stride).cpu()
    gt_points = None
    if truth:
        gt = GroundTruth.load(folder / f"gt_{cfg.views}.npz")
        _, h, w = gt.depth.shape
        p = grid(h, w, cfg.stride, device)
        idx = p.round().long()
        depth = (
            torch.from_numpy(gt.depth).to(device).double()[:, idx[..., 1], idx[..., 0]]
        )
        k = torch.from_numpy(views.intrinsics).to(device).double()
        c2w = torch.from_numpy(views.cam_to_world).to(device).double()
        pix = F.pad(
            idx.double() + 0.5, (0, 1), value=1.0
        )  # GT intrinsics: corners at integers
        cam = torch.einsum("vab,hwb->vhwa", torch.linalg.inv(k), pix) * depth[..., None]
        pts = (
            torch.einsum("vab,vhwb->vhwa", c2w[:, :3, :3], cam)
            + c2w[:, None, None, :3, 3]
        )
        gt_points = torch.where(
            (depth > 0)[..., None], pts, torch.full_like(pts, torch.nan)
        ).cpu()
        scene = dataclasses.replace(scene, rays=torch.empty(0, device=device))
        geometry = None
    return SceneData(
        name,
        scene.to(torch.device("cpu")),
        geometry,
        matches.to(torch.device("cpu")),
        feats,
        gt_points,
    )


def grid_points(scene: Scene, state: State, stride: int) -> Tensor:
    """World points (V, gh, gw, 3) of the solver's state."""
    _, h, w = scene.depth.shape
    p = grid(h, w, stride, scene.depth.device)
    k_inv = torch.linalg.inv(scene.k.double())
    a = torch.einsum(
        "vab,vbc,hwc->vhwa", state.rotation, k_inv, F.pad(p, (0, 1), value=1.0)
    )
    return state.center[:, None, None] + a / state.rho[..., None]


def point_error(points: Tensor, truth: Tensor) -> Tensor:
    """Mean relative 3D error after a Sim(3) onto the truth, the Sim(3) held fixed."""
    ok = torch.isfinite(truth).all(-1) & torch.isfinite(points).all(-1)
    x, y = points[ok], truth[ok]
    with torch.no_grad():
        mx, my = x.mean(0), y.mean(0)
        u, s, vt = torch.linalg.svd((y - my).T @ (x - mx) / len(x))
        d = torch.ones(3, dtype=x.dtype, device=x.device)
        d[2] = torch.sign(torch.linalg.det(u @ vt))
        rot = u @ torch.diag(d) @ vt
        scale = (s * d).sum() / (x - mx).pow(2).sum(-1).mean()
    aligned = scale * (x - mx) @ rot.T + my
    depth = (y - my).norm(dim=-1).median()
    return ((aligned - y).norm(dim=-1) / depth).mean()


def held_error(model: MatchWeights, d: SceneData, cfg: BundleConfig) -> float:
    """Point error of one held-out scene after the solver with the model's weights."""
    state = adjust(d.scene, d.matches, model(d.feats, d.matches.same), cfg)
    return point_error(grid_points(d.scene, state, cfg.stride), d.gt_points).item()


def train_model(
    cfg: BundleConfig,
    data: list[SceneData],
    semantics: str,
    seed: int,
    device: torch.device,
) -> tuple[MatchWeights, list[dict]]:
    """Train the weights through the unrolled solver; keep the least held-out error."""
    torch.manual_seed(seed)
    model = MatchWeights(cfg.hidden, semantics=semantics != "none").to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    fit, held = data[: len(data) - cfg.held_out], data[len(data) - cfg.held_out :]
    rng = np.random.default_rng(seed)
    best, best_state, history = np.inf, None, []
    t0 = time.perf_counter()
    for step in range(cfg.steps + 1):
        if step % max(1, cfg.steps // 20) == 0:
            with torch.no_grad():
                err = float(
                    np.mean([held_error(model, d.to(device), cfg) for d in held])
                )
            history.append(
                {"step": step, "held_out": err, "seconds": time.perf_counter() - t0}
            )
            print(f"step {step}: held-out {err:.5f}", flush=True)  # noqa: T201
            if err < best:
                best, best_state = (
                    err,
                    {k: v.clone() for k, v in model.state_dict().items()},
                )
        if step == cfg.steps:
            break
        d = fit[rng.integers(len(fit))].to(device)
        loss = point_error(
            grid_points(
                d.scene,
                adjust(d.scene, d.matches, model(d.feats, d.matches.same), cfg),
                cfg.stride,
            ),
            d.gt_points,
        )
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    assert best_state is not None
    model.load_state_dict(best_state)
    return model, history


def write(cfg: BundleConfig, d: SceneData, weight: Tensor, name: str) -> None:
    """Adjust one scene on weight's device; save it as backbone `<name>_<views>.npz`."""
    assert d.geometry is not None
    d = d.to(weight.device)
    with torch.no_grad():
        state = adjust(d.scene, d.matches, weight, cfg)
    geometry = refined_geometry(d.scene, d.geometry, state, cfg.stride)
    geometry.save(cfg.root / d.name / f"{name}_{cfg.views}.npz")


def main() -> None:
    """Run a stage: `matches` (RAFT, cached), `free` (untrained) or `train`."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("matches", "free", "train"))
    parser.add_argument("--config", type=Path, default=Path("configs/e82a_bundle.yaml"))
    parser.add_argument("--out", type=Path, help="run folder of `train`")
    parser.add_argument("--only", nargs="*", help="model names with -s<seed>")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    cfg = BundleConfig.load(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.stage == "matches":
        compute_matches(cfg, cfg.train + cfg.predict, device)
        return
    predict = [load_scene(cfg, n, device, truth=False) for n in cfg.predict]
    if args.stage == "free":
        for name, prior in cfg.free.items():
            c = dataclasses.replace(cfg, prior=prior)
            for d in predict:
                write(c, d, free_weights(d.matches.to(device), cfg.sigma), name)
            print(f"{name}: wrote {len(predict)} scenes", flush=True)  # noqa: T201
        return
    out = args.out or Path("runs") / f"bundle-{time.strftime('%Y%m%d-%H%M%S')}"
    out.mkdir(parents=True, exist_ok=True)
    data = [load_scene(cfg, n, device, truth=True) for n in cfg.train]
    record = {"config": yaml.safe_load(args.config.read_text()), "git": git_hash()}
    for base, semantics in cfg.models.items():
        for seed in cfg.seeds:
            name = f"{base}-s{seed}"
            if args.only and name not in args.only:
                continue
            saved = out / f"{name}.pt"
            model = MatchWeights(cfg.hidden, semantics=semantics != "none").to(device)
            if saved.exists():
                model.load_state_dict(torch.load(saved, map_location=device))
            else:
                model, history = train_model(cfg, data, semantics, seed, device)
                torch.save(model.state_dict(), saved)
                record[name] = history
                (out / "run.yaml").write_text(yaml.safe_dump(record, sort_keys=False))
            for d in predict:
                with torch.no_grad():
                    weight = model(d.feats.to(device), d.matches.same.to(device))
                write(cfg, d, weight, name)
            print(f"{name}: wrote {len(predict)} scenes", flush=True)  # noqa: T201


if __name__ == "__main__":
    main()
