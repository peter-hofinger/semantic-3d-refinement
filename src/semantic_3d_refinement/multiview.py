"""Joint refinement of every view of a scene: the monocular energy of each view plus cross-view agreement E_mv.

Cameras stay fixed: Déjà View's, each with the pinhole fitted to its rays (`views.fit_pinhole`). All views share one
normalization, d_i = s / D_i with s the scene's median depth, so a residual means the same in every view. Each pixel
p of view i is lifted, x = o_i + D_i(p) r_i(p), and projected into every other view j:

  E_mv = 1/N Σ_{i≠j} Σ_{p ∈ M_ij} ( log z_j(x) + log d_j(π_j(x)) - log s )²

z_j(x) is x's depth in camera j and d_j(π_j(x)) is j's inverse depth, bilinear at the projection, so the residual is
the log ratio of the two depths: zero where both views put the surface in the same place. N is the number of pixels
over all views, as for the per-view terms. The matches M_ij are fixed on the input, as BiNI's cut edges are, so E
stays smooth: x lands in j at least a pixel inside the border, the two depths agree within `mv_tol`, and so do j's
four interpolated pixels. This drops what j does not see (occluded or out of view) and interpolation across j's
occlusion edges.

  E = 1/V Σ_i E_i + λ_mv E_mv,   E_i the monocular energy of view i (`refine.energy_maps`), δ_i its residual.

This is the E_mv of the method in `Claude.md` without segments: agreement is required wherever the input already
roughly agrees, not only within a segment.
"""

import argparse
import dataclasses
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import yaml
from scipy.spatial import cKDTree
from torch import Tensor

from semantic_3d_refinement.evaluate import accuracy, scan_depth, score
from semantic_3d_refinement.normals import predicted_normals
from semantic_3d_refinement.refine import (
    NormalTerm,
    RefineConfig,
    energy_maps,
    normal_term,
    points_from_rays,
    term_weights,
    variant_label,
)
from semantic_3d_refinement.views import check_inputs, fit_pinhole, git_hash

VOXEL = 0.005
PLANE_K = 24
SAMPLES = 200_000


@dataclass(frozen=True)
class Scene:
    """All views of one backbone export with pinhole rays: depth, conf (V, H, W), rays (V, H, W, 6), cameras."""

    depth: Tensor
    rays: Tensor
    conf: Tensor
    k: Tensor
    rotation: Tensor
    center: Tensor

    def to(self, device: torch.device) -> "Scene":
        """Move every tensor to the device."""
        return Scene(*(getattr(self, f).to(device) for f in self.__dataclass_fields__))


def load_scene(path: Path) -> Scene:
    """Every view of a Déjà View export, its rays replaced by the fitted pinhole (`rays: pinhole`)."""
    data = np.load(path)
    rays, ks = [], []
    for v in range(len(data["depth"])):
        ray_map = np.concatenate([data["ray_origins"][v], data["ray_directions"][v]], axis=-1).astype(np.float64)
        r, k = fit_pinhole(ray_map, data["cam_to_world"][v, :3, :3].astype(np.float64))
        rays.append(r)
        ks.append(k)
    rays = np.stack(rays)
    scene = Scene(
        *(torch.from_numpy(np.ascontiguousarray(a)).float() for a in (data["depth"], rays, data["confidence"])),
        torch.from_numpy(np.stack(ks)).float(),
        torch.from_numpy(data["cam_to_world"][:, :3, :3]).float(),
        torch.from_numpy(rays[:, 0, 0, :3]).float(),
    )
    for v in range(len(scene.depth)):
        check_inputs(scene.depth[v], scene.rays[v], scene.conf[v])
    return scene


@dataclass(frozen=True)
class Matches:
    """Fixed matches (view i, pixel p) -> view j: p's flat index in (V, H, W), j, and the ray in j's camera frame.

    x_j = offset + D_i(p) * direction is the lifted point in camera j's coordinates.
    """

    src: Tensor
    dst: Tensor
    offset: Tensor
    direction: Tensor


def project(x: Tensor, k: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Pixel coordinates u, v (pixel centers at integers) and depth z of camera-frame points under per-point K."""
    z = x[:, 2]
    return k[:, 0, 0] * x[:, 0] / z + k[:, 0, 2], k[:, 1, 1] * x[:, 1] / z + k[:, 1, 2], z


def bilinear(d: Tensor, view: Tensor, u: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
    """Sample d (V, H, W) at fractional pixels of the given views; also return the four corners' max/min ratio.

    u, v are clamped to [0, W-1] x [0, H-1], so a line-search trial step that leaves the image stays defined.
    """
    _, h, w = d.shape
    u, v = u.clamp(0, w - 1 - 1e-4), v.clamp(0, h - 1 - 1e-4)
    x0, y0 = u.detach().floor().long(), v.detach().floor().long()
    fu, fv = u - x0, v - y0
    base = (view * h + y0) * w + x0
    flat = d.reshape(-1)
    c = torch.stack([flat[base], flat[base + 1], flat[base + w], flat[base + w + 1]])
    value = (1 - fv) * ((1 - fu) * c[0] + fu * c[1]) + fv * ((1 - fu) * c[2] + fu * c[3])
    return value, c.amax(0) / c.amin(0)


def residuals(d: Tensor, scale: Tensor, scene: Scene, m: Matches) -> Tensor:
    """Return log z_j(x) + log d_j(π_j(x)) - log s per match, for normalized inverse depth d (V, H, W)."""
    depth = scale / d.reshape(-1)[m.src]
    u, v, z = project(m.offset + depth[:, None] * m.direction, scene.k[m.dst])
    dj, _ = bilinear(d, m.dst, u, v)
    return z.clamp_min(1e-6).log() + dj.clamp_min(1e-6).log() - scale.log()


@torch.no_grad()
def find_matches(scene: Scene, d0: Tensor, scale: Tensor, tol: float) -> Matches:
    """Match every ordered pair of views wherever the input agrees within tol (log depth)."""
    n, h, w = d0.shape
    src, dst, offsets, directions = [], [], [], []
    flat = torch.arange(h * w, device=d0.device)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            world_to_j = scene.rotation[j].T
            offset = (scene.rays[i, ..., :3].reshape(-1, 3) - scene.center[j]) @ world_to_j.T
            direction = scene.rays[i, ..., 3:].reshape(-1, 3) @ world_to_j.T
            u, v, z = project(offset + (scale / d0[i].reshape(-1))[:, None] * direction, scene.k[j].expand(h * w, 3, 3))
            dst_j = torch.full_like(flat, j)
            dj, spread = bilinear(d0, dst_j, u, v)
            log_ratio = (z.clamp_min(1e-6) * dj / scale).log().abs()
            keep = (z > 0) & (u >= 1) & (u <= w - 2) & (v >= 1) & (v <= h - 2)
            keep &= (log_ratio < tol) & (spread.log() < tol)
            src.append(i * h * w + flat[keep])
            dst.append(dst_j[keep])
            offsets.append(offset[keep])
            directions.append(direction[keep])
    m = Matches(torch.cat(src), torch.cat(dst), torch.cat(offsets), torch.cat(directions))
    assert (m.src // (h * w) != m.dst).all() and m.offset.isfinite().all() and m.direction.isfinite().all()
    return m


@dataclass(frozen=True)
class MultiviewConfig:
    """Weight of E_mv and the tolerance (log depth) under which the input's matches count."""

    lambda_mv: float = 1.0
    mv_tol: float = 0.05

    def __post_init__(self) -> None:
        """Check ranges."""
        assert self.lambda_mv >= 0 and 0 < self.mv_tol < 1


MV_KEYS = {f.name for f in dataclasses.fields(MultiviewConfig)}


@dataclass(frozen=True)
class Problem:
    """What stays fixed during the solve: the scene, its scale s, d0 = s / D0, weights, normal terms and matches."""

    scene: Scene
    scale: Tensor
    d0: Tensor
    w: Tensor
    normals: list[NormalTerm | None]
    matches: Matches


def scene_energy(delta: Tensor, cfg: RefineConfig, mv: MultiviewConfig, p: Problem) -> tuple[Tensor, dict[str, float]]:
    """E = 1/V Σ_i E_i + λ_mv E_mv; also each term's unweighted value, the per-view ones averaged over views."""
    n = len(p.d0)
    terms: dict[str, Tensor] = {}
    for i in range(n):
        for k, v in energy_maps(delta[i], p.d0[i], p.w[i], cfg, p.normals[i]).items():
            terms[k] = terms.get(k, 0) + v.mean() / n
    terms["E_mv"] = residuals(p.d0 + delta, p.scale, p.scene, p.matches).square().sum() / p.d0.numel()
    weights = term_weights(cfg) | {"E_mv": mv.lambda_mv}
    e = sum(weights[k] * terms[k] for k in terms)
    assert isinstance(e, Tensor) and e.isfinite(), e
    return e, {"E": e.item()} | {k: v.item() for k, v in terms.items()}


def refine_scene(
    scene: Scene, cfg: RefineConfig, mv: MultiviewConfig, normals: Tensor | None = None
) -> tuple[Tensor, list[dict[str, float]], int]:
    """Optimize the residuals of all views jointly by L-BFGS; return depth (V, H, W), energy log and match count.

    Normals (V, H, W, 3), unit, in the frame of the rays, enable E_n per view and cut E_bd as in `refine.refine`.
    """
    assert normals is not None or cfg.lambda_n == 0, "E_n needs normals"
    scale = scene.depth.median()
    d0 = scale / scene.depth
    w = scene.conf / scene.conf.mean(dim=(1, 2), keepdim=True)
    terms = [None if normals is None else normal_term(normals[i], scene.rays[i], d0[i], cfg.edge_tol)
             for i in range(len(d0))]  # fmt: skip
    problem = Problem(scene, scale, d0, w, terms, find_matches(scene, d0, scale, mv.mv_tol))
    delta = torch.zeros_like(d0, requires_grad=True)

    log = [scene_energy(delta, cfg, mv, problem)[1]]
    if cfg.iters > 0:
        opt = torch.optim.LBFGS(
            [delta],
            lr=1.0,
            max_iter=cfg.iters,
            history_size=cfg.history_size,
            tolerance_grad=1e-9,
            tolerance_change=1e-12,
            line_search_fn="strong_wolfe",
        )

        def closure() -> Tensor:
            opt.zero_grad()
            e, t = scene_energy(delta, cfg, mv, problem)
            e.backward()
            log.append(t)
            return e

        opt.step(closure)
        with torch.no_grad():
            log.append(scene_energy(delta, cfg, mv, problem)[1])
    depth_hat = scale / (d0 + delta.detach())
    assert depth_hat.shape == scene.depth.shape and depth_hat.isfinite().all() and (depth_hat > 0).all()
    assert log[-1]["E"] <= log[0]["E"] + 1e-6, (log[0], log[-1])
    assert math.isfinite(log[-1]["E_mv"])
    return depth_hat, log, len(problem.matches.src)


def similarity(a: np.ndarray, b: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Scale, rotation and translation minimizing Σ ‖s R a + t - b‖² over corresponding points (Umeyama 1991)."""
    assert a.shape == b.shape and a.shape[1] == 3 and len(a) >= 3, (a.shape, b.shape)  # noqa: PLR2004
    ma, mb = a.mean(0), b.mean(0)
    u, sv, vt = np.linalg.svd((b - mb).T @ (a - ma) / len(a))
    sign = np.diag([1.0, 1.0, np.sign(np.linalg.det(u @ vt))])
    rotation = u @ sign @ vt
    scale = float((sv * np.diag(sign)).sum() / ((a - ma) ** 2).sum(1).mean())
    return scale, rotation, mb - scale * rotation @ ma


def voxel_downsample(points: np.ndarray, size: float) -> np.ndarray:
    """One point per occupied voxel of the given size, the mean of its points."""
    keys = np.floor(points / size).astype(np.int64)
    _, inverse, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    out = np.zeros((len(counts), 3))
    np.add.at(out, inverse.ravel(), points)
    return out / counts[:, None]


def local_planes(tree: cKDTree, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """PCA over each query's k nearest points: the unit normal and the RMS distance to the fitted plane."""
    _, idx = tree.query(queries, k=k, workers=-1)
    nb = tree.data[idx]
    nb = nb - nb.mean(1, keepdims=True)
    values, vectors = np.linalg.eigh(np.einsum("nki,nkj->nij", nb, nb) / k)
    return vectors[..., 0], np.sqrt(values[:, 0].clip(0))


@dataclass(frozen=True)
class Truth:
    """A scene's scan and ETH3D's cameras, the scan's depth in each view, and the similarity from Déjà View's frame."""

    tree: cKDTree
    grid: cKDTree
    k: np.ndarray
    cam_to_world: np.ndarray
    depth: np.ndarray
    align: tuple[float, np.ndarray, np.ndarray]


def load_truth(paths: dict, scene: Scene) -> Truth:
    """Scan (and its 5 mm grid), cameras and per-view scan depth; Déjà View's centers aligned to ETH3D's."""
    scan, cams = np.load(paths["scan"]), np.load(paths["gt_views"])
    scan_gpu = torch.from_numpy(scan).to(scene.depth.device)
    h, w = scene.depth.shape[1:]
    depth = np.stack([
        scan_depth(scan_gpu, torch.from_numpy(k).to(scan_gpu), torch.from_numpy(c).to(scan_gpu), h, w).cpu().numpy()
        for k, c in zip(cams["intrinsics"], cams["cam_to_world"], strict=True)
    ])  # fmt: skip
    centers = cams["cam_to_world"][:, :3, 3].astype(np.float64)
    align = similarity(scene.center.cpu().numpy().astype(np.float64), centers)
    grid = cKDTree(voxel_downsample(scan.astype(np.float64), VOXEL))
    k, c2w = cams["intrinsics"].astype(np.float64), cams["cam_to_world"].astype(np.float64)
    return Truth(cKDTree(scan), grid, k, c2w, depth.astype(np.float64), align)


def fused_metrics(depth: np.ndarray, rays: np.ndarray, truth: Truth, tolerances_cm: list[int | float]) -> dict:
    """Accuracy of all views' points (where the scan saw the pixel), and the fidelity of the surface they form.

    The cloud is mapped by the camera similarity, the same for every variant. Fidelity on a 5 mm voxel grid, so point
    density does not change it: at 200k sampled grid points, the RMS distance of the 24 nearest grid points to their
    plane (thickness: layering and noise, no ground truth needed), and the angle between that plane and the scan's.
    """
    s, rotation, t = truth.align
    points = s * points_from_rays(torch.from_numpy(depth), torch.from_numpy(rays)).numpy() @ rotation.T + t
    points = points[truth.depth > 0]
    out = {f"fused_{k}": v for k, v in accuracy(truth.tree.query(points, workers=-1)[0], tolerances_cm).items()}
    grid = voxel_downsample(points, VOXEL)
    queries = grid[np.random.default_rng(0).choice(len(grid), min(SAMPLES, len(grid)), replace=False)]
    normals, thickness = local_planes(cKDTree(grid), queries, PLANE_K)
    scan_normals, _ = local_planes(truth.grid, truth.grid.data[truth.grid.query(queries, workers=-1)[1]], PLANE_K)
    angle = np.degrees(np.arccos(np.abs((normals * scan_normals).sum(-1)).clip(0, 1)))
    return out | {"thickness_mm": float(1000 * np.median(thickness)), "normal_angle_deg": float(np.median(angle))}


def scan_fidelity(truth: Truth) -> dict[str, float]:
    """Return the scan's own thickness on its grid, the floor of `thickness_mm`."""
    queries = truth.grid.data[np.random.default_rng(0).choice(truth.grid.n, min(SAMPLES, truth.grid.n), replace=False)]
    return {"thickness_mm": float(1000 * np.median(local_planes(truth.grid, queries, PLANE_K)[1]))}


def scene_label(cfg: RefineConfig, mv: MultiviewConfig) -> str:
    """Name the active terms, e.g. "E_d + E_mv"; E_d alone is the input."""
    label = variant_label(cfg) + (" + E_mv" if mv.lambda_mv > 0 else "")
    return label + (" (= input)" if label == "E_d" else "")


def run_variant(
    scene: Scene, normals: Tensor, cfg: RefineConfig, mv: MultiviewConfig, problem: tuple[Truth, Matches, dict]
) -> tuple[dict, np.ndarray]:
    """Refine all views under one configuration and score it: fused cloud, per-view protocol, disagreement."""
    truth, matches, raw = problem
    t0 = time.perf_counter()
    depth_hat, log, n = refine_scene(scene, cfg, mv, normals)
    if depth_hat.is_cuda:
        torch.cuda.synchronize()
    seconds = time.perf_counter() - t0
    scale = scene.depth.median()
    disagreement = 100 * residuals(scale / depth_hat, scale, scene, matches).abs().median().item()
    depth = depth_hat.cpu().numpy().astype(np.float64)
    record = {
        "label": scene_label(cfg, mv),
        "seconds": seconds,
        "evaluations": len(log) - 1,
        "matches": n,
        "energy_initial": log[0],
        "energy_final": log[-1],
        "disagreement_pct": disagreement,
        **fused_metrics(depth, scene.rays.cpu().numpy().astype(np.float64), truth, raw["tolerances_cm"]),
        "per_view": {},
    }
    for v in raw["views"]:
        dist, _ = score(depth[v], truth.depth[v], truth.k[v], truth.cam_to_world[v], truth.tree)
        record["per_view"][v] = accuracy(dist[np.isfinite(dist)], raw["tolerances_cm"])
    return record, depth_hat.cpu().numpy()


def main() -> None:
    """Refine every scene jointly per variant: `uv run python -m semantic_3d_refinement.multiview --cfg <cfg>`."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=Path, required=True)
    cfg_path: Path = parser.parse_args().cfg
    raw = yaml.safe_load(cfg_path.read_text())
    base = RefineConfig(**raw["refine"])
    device = torch.device(base.device if base.device != "cuda" or torch.cuda.is_available() else "cpu")
    base = dataclasses.replace(base, device=device.type)

    out = Path(raw["output_dir"]) / f"{cfg_path.stem}-{time.strftime('%Y%m%d-%H%M%S')}"
    out.mkdir(parents=True)
    scenes, arrays = {}, {}
    for name, paths in raw["scenes"].items():
        t0 = time.perf_counter()
        scene = load_scene(Path(paths["input"])).to(device)
        normals = torch.stack([
            predicted_normals({**raw, **paths, "view": v}, {"depth": scene.depth[v], "rays": scene.rays[v]})[0]
            for v in range(len(scene.depth))
        ])  # fmt: skip
        truth = load_truth(paths, scene)
        scale = scene.depth.median()
        matches = find_matches(scene, scale / scene.depth, scale, raw["multiview"]["mv_tol"])
        s, rotation, t = truth.align
        centers = s * scene.center.cpu().numpy() @ rotation.T + t
        center_residual = np.linalg.norm(centers - truth.cam_to_world[:, :3, 3], axis=1)
        scenes[name] = {
            "setup_seconds": time.perf_counter() - t0,
            "align": {"scale": s, "rotation": rotation.tolist(), "translation": t.tolist()},
            "center_residual_cm": float(100 * np.median(center_residual)),
            "scan": scan_fidelity(truth),
            "variants": [],
        }
        refine_scene(scene, dataclasses.replace(base, iters=1), MultiviewConfig(**raw["multiview"]), normals)
        for i, o in enumerate(raw["variants"]):
            mv = MultiviewConfig(**{**raw["multiview"], **{k: v for k, v in o.items() if k in MV_KEYS}})
            cfg = dataclasses.replace(base, **{k: v for k, v in o.items() if k not in MV_KEYS})
            record, depth = run_variant(scene, normals, cfg, mv, (truth, matches, raw))
            scenes[name]["variants"].append(record)
            arrays[f"{name}_{i}_depth"] = depth
            keys = ("fused_median_cm", "thickness_mm", "normal_angle_deg", "disagreement_pct", "seconds")
            print(name, record["label"], {k: round(record[k], 3) for k in keys})  # noqa: T201

    summary = {"config": raw, "resolved": dataclasses.asdict(base), "git": git_hash(), "scenes": scenes}
    (out / "run.yaml").write_text(yaml.safe_dump(summary, sort_keys=False, allow_unicode=True))
    np.savez_compressed(out / "results.npz", **arrays)
    print(f"wrote {out}")  # noqa: T201


if __name__ == "__main__":
    main()
