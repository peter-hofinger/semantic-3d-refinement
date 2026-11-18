"""E52c, solver check: can the cross-view distance be driven towards the GT at all.

Rung 3 with the term alone (LOG, E52c) stopped at c / c_DV 0.89-0.90 while the GT error
grew past Déjà View's. This check removes the network. The start is Déjà View's cached
depth of the bedroom's 12 probe views (`e52c_probe.py`), lifted along the pinhole fit of
its rays (`scene_from_geometry`, the repo's convention: the raw ray map's 14 px steps
show up in normals); the free variables are one of

- patch: a log-scale per 14 px patch, bilinearly upsampled, d_p' = d_p exp(u(p)); a
  depth field as fine as Déjà View's tokens. Per pixel would leave most pixels without a
  gradient, as a step draws 6000 of ~2.3 M.
- view: a log-scale and an offset per view, d_p' = d_p exp(a_v) + b_v.

Each Adam step lifts x_p = o_v + d_p' r_p, re-fits the exact Sim(3) onto the GT on 20000
fixed pixels (`umeyama_torch`, as training's `align=exact`), draws 6000 pixels, and
minimises one loss on the aligned points: the cross-view distance `cross_view_distance`
(normals from the current maps at stride 2, detached; partners in the same GT instance,
`gate`, or any, `any`), or, as the solver's own known answer, the GT error (mean relative
3D error capped 0.5, over Déjà View's on the same draw, `gt`). Every 25 steps it records,
on one fixed draw (seed 1, as the probe), c / c_DV for both partner rules and the GT
ratio over all GT-valid pixels; GT's own c / c_DV is the floor. It also records the
Sim(3)'s scale over Déjà View's and, at the end, quantiles of the depth over the start:
c is divided by the GT depth, so a fit that shrinks the aligned cloud lowers c.

If the term's minimum lies near the GT, minimising it moves the points towards the GT
(GT ratio < 1) and towards GT's c; fitting the GT should then also lower c.

uv run python experiments/scripts/e52c_solver.py
writes `experiments/results/e52c_solver.yaml` and `experiments/figures/e52c_solver.jpg`;
`--plot` redraws the figure from the yaml.
"""

import argparse
import time
from pathlib import Path

import matplotlib as mpl
import numpy as np
import torch
import yaml
from torch import Tensor

mpl.use("Agg")
import matplotlib.pyplot as plt
from e52a_figures import FIGURES, style
from e52a_segment_adapters import load_scene

from src.analysis.scene_eval import git_hash, input_paths
from src.backbone.adapters import PATCH_SIZE, cross_view_distance, map_normals
from src.evaluation.labeled import umeyama_torch
from src.geometry import Geometry
from src.refinement.multiview import points_from_rays, scene_from_geometry

SCENE = "ai_006_007_cam_00"  # bedroom
VIEWS, PIXELS, FIT, CAP, STRIDE, MAX_ERROR = 12, 6000, 20000, 0.05, 2, 0.5
STEPS, LR, EVERY = 1000, 1e-3, 25
QUANTILES = (0.001, 0.05, 0.5, 0.95, 0.999)  # of the final depth over the start depth
PARAMS = ("patch", "view")
LOSSES = {"gate": ("c, same instance", "#eb6834"), "any": ("c, any partner", "#3a7bd5"),
          "gt": ("GT error (known answer)", "#2a9d55")}  # fmt: skip
RESULTS = Path("experiments/results/e52c_solver.yaml")


class Problem:
    """The bedroom's 12 views on one device: start depth, rays, GT and draws."""

    def __init__(self, raw: dict, device: torch.device) -> None:
        """Load Déjà View's cache and the GT; fix the Sim(3) pixels and the eval draw."""
        sel = np.linspace(0, 19, VIEWS).round().astype(int)
        truth = load_scene(raw, SCENE)
        pinhole = scene_from_geometry(
            Geometry.load(input_paths(raw, SCENE)["geometry"])
        )
        self.depth = pinhole.depth[sel].to(device)
        self.rays = pinhole.rays[sel].to(device)
        assert self.depth.shape == truth.depth[sel].shape, self.depth.shape
        valid = torch.from_numpy(truth.depth[sel] > 0).to(device)
        self.valid = valid
        self.gt = torch.from_numpy(truth.points[sel]).to(device)
        self.gt_depth = torch.from_numpy(truth.depth[sel]).to(device)[valid]
        self.view = torch.nonzero(valid)[:, 0]
        self.inst = torch.from_numpy(truth.instances[sel]).to(device)[valid].long()
        n = int(valid.sum())
        self.fit = torch.from_numpy(
            np.random.default_rng(0).choice(n, FIT, replace=False)
        )
        self.fit = self.fit.to(device)
        self.eval = torch.from_numpy(
            np.random.default_rng(1).choice(n, PIXELS, replace=False)
        ).to(device)
        self.n = n
        self.dv, self.dv_scale = self.align(points_from_rays(self.depth, self.rays))
        self.dv_error = self.error(self.dv)
        self.c_dv = {
            g: self.distance(self.dv, self.eval, gate=g) for g in (True, False)
        }
        assert all(c > 0 for c in self.c_dv.values()), self.c_dv

    def align(self, x: Tensor) -> tuple[Tensor, float]:
        """Apply the exact Sim(3) of the fit pixels onto the GT; return maps, scale."""
        s, r, t = umeyama_torch(
            x[self.valid][self.fit].double(), self.gt[self.valid][self.fit].double()
        )
        out = (x.double() @ (s * r.T) + t).float()
        assert torch.isfinite(out).all()
        return out, s.item()

    def error(self, aligned: Tensor) -> Tensor:
        """Relative 3D error per GT-valid pixel, capped at MAX_ERROR."""
        err = (aligned[self.valid] - self.gt[self.valid]).norm(dim=-1) / self.gt_depth
        return err.clamp(max=MAX_ERROR)

    def distance(self, aligned: Tensor, at: Tensor, *, gate: bool) -> Tensor:
        """Cross-view distance of the drawn pixels, normals from the maps, detached."""
        normal = map_normals(aligned.detach(), STRIDE)[self.valid][at]
        return cross_view_distance(
            aligned[self.valid][at], normal, self.view[at], self.inst[at],
            self.gt_depth[at], gate=gate, cap=CAP,
        )  # fmt: skip


def depth_of(problem: Problem, param: str, theta: Tensor) -> Tensor:
    """(V, H, W) depth from the start depth and the free variables."""
    d = problem.depth
    if param == "patch":
        scale = torch.nn.functional.interpolate(
            theta[:, None], size=d.shape[1:], mode="bilinear"
        )[:, 0]
        return d * scale.exp()
    return d * theta[:, 0, None, None].exp() + theta[:, 1, None, None]


def solve(problem: Problem, param: str, loss: str, steps: int) -> dict:
    """Adam on one parametrisation and loss; ratios on the fixed draw every EVERY steps."""
    assert param in PARAMS, param
    assert loss in LOSSES, loss
    v, h, w = problem.depth.shape
    shape = (
        (v, -(-h // PATCH_SIZE), -(-w // PATCH_SIZE)) if param == "patch" else (v, 2)
    )
    theta = torch.zeros(shape, device=problem.depth.device, requires_grad=True)
    opt = torch.optim.Adam([theta], lr=LR)
    rng = np.random.default_rng(2)
    record: dict = {
        "step": [],
        "gt_ratio": [],
        "c_gate": [],
        "c_any": [],
        "sim3_scale": [],
    }
    start = time.perf_counter()
    for it in range(steps + 1):
        x, scale = problem.align(
            points_from_rays(depth_of(problem, param, theta), problem.rays)
        )
        if it % EVERY == 0 or it == steps:
            with torch.no_grad():
                record["step"].append(it)
                gt_ratio = problem.error(x).mean() / problem.dv_error.mean()
                record["gt_ratio"].append(gt_ratio.item())
                record["sim3_scale"].append(scale / problem.dv_scale)
                for g, key in ((True, "c_gate"), (False, "c_any")):
                    c = problem.distance(x, problem.eval, gate=g) / problem.c_dv[g]
                    record[key].append(c.item())
        if it == steps:
            break
        at = torch.from_numpy(rng.choice(problem.n, PIXELS, replace=False))
        at = at.to(x.device)
        if loss == "gt":
            value = problem.error(x)[at].mean() / problem.dv_error[at].mean()
        else:
            value = problem.distance(x, at, gate=loss == "gate")
        opt.zero_grad(set_to_none=True)
        value.backward()
        opt.step()
    assert torch.isfinite(theta).all()
    with torch.no_grad():
        ratio = (depth_of(problem, param, theta) / problem.depth)[problem.valid]
        q = torch.tensor(QUANTILES, device=ratio.device)
        record["depth_ratio_quantiles"] = torch.quantile(ratio, q).tolist()
    record["seconds"] = time.perf_counter() - start
    return record


def figure(runs: dict, floor: dict) -> None:
    """GT ratio and c / c_DV per step, one panel per parametrisation."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    for ax, param in zip(axes, PARAMS, strict=True):
        for loss, (label, colour) in LOSSES.items():
            r = runs[f"{param}_{loss}"]
            c = r["c_any"] if loss == "any" else r["c_gate"]
            ax.plot(
                r["step"], r["gt_ratio"], color=colour, label=f"GT ratio, min {label}"
            )
            ax.plot(r["step"], c, color=colour, ls="--", label=f"c / c_DV, min {label}")
            ax.plot(r["step"], r["sim3_scale"], color=colour, ls=":",
                    label=f"Sim(3) scale / DV's, min {label}")  # fmt: skip
        ax.axhline(floor["gate"], color="0.4", lw=0.8, ls=":")
        ax.axhline(1.0, color="0.7", lw=0.6)
        ax.set_yscale("log")
        ax.set_xlabel("Adam step")
        ax.set_title(
            "log-scale per 14 px patch"
            if param == "patch"
            else "scale + offset per view",
            fontsize=9,
        )
        style(ax)
    axes[0].set_ylabel("ratio to Déjà View (bedroom, 12 views, log)")
    axes[1].legend(
        fontsize=7, frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0)
    )
    fig.suptitle(
        "Minimising the cross-view distance c on Déjà View's cached depth, no network. "
        "Solid: GT error, dashed: c, dotted: the Sim(3)'s scale, all over Déjà View's;\n"
        "grey dotted line: GT's own c. Minimising c drives it below GT's by shrinking the "
        "fit (c is divided by GT depth) while the GT error grows 17x.",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(FIGURES / "e52c_solver.jpg", dpi=150)


def main() -> None:
    """Solve every parametrisation and loss, write the raw numbers and the figure."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--plot", action="store_true", help="redraw from the yaml")
    args = parser.parse_args()
    if args.plot:
        done = yaml.safe_load(RESULTS.read_text(encoding="utf-8"))
        figure(done["runs"], done["floor"])
        return
    cfg = yaml.safe_load(Path("configs/e52a_adapters.yaml").read_text(encoding="utf-8"))
    raw = yaml.safe_load(Path(cfg["eval"]).read_text(encoding="utf-8"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    problem = Problem(raw, device)
    floor = {
        key: (
            problem.distance(problem.gt, problem.eval, gate=g) / problem.c_dv[g]
        ).item()
        for g, key in ((True, "gate"), (False, "any"))
    }
    print("GT's c / c_DV", floor, flush=True)
    runs = {}
    for param in PARAMS:
        for loss in LOSSES:
            r = solve(problem, param, loss, args.steps)
            runs[f"{param}_{loss}"] = r
            print(f"{param} {loss}: GT ratio {r['gt_ratio'][-1]:.3f}, c gate "
                  f"{r['c_gate'][-1]:.3f}, c any {r['c_any'][-1]:.3f}, "
                  f"{r['seconds']:.0f} s", flush=True)  # fmt: skip
    settings = {"scene": SCENE, "views": VIEWS, "pixels": PIXELS, "fit": FIT, "cap": CAP,
                "stride": STRIDE, "steps": args.steps, "lr": LR, "device": str(device)}  # fmt: skip
    RESULTS.write_text(
        yaml.safe_dump({"git": git_hash(), "settings": settings, "floor": floor,
                        "runs": runs}, sort_keys=False),
        encoding="utf-8",
    )  # fmt: skip
    figure(runs, floor)


if __name__ == "__main__":
    main()
