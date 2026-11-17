"""E52d, gradient check: whether the adapters' training gradient lowers the loss.

E52d found B_k moving steadily in a direction that raises the loss it was computed from.
Training's error of one pass (`e52a_segment_adapters.step_loss`) is L(θ) = e(x(θ), S),
with x the model's points and S = (s, R, t) the least-squares Umeyama Sim(3) of a pixel
subset, detached. The loss actually measured re-fits S at every θ: L*(θ) = e(x(θ),
S(x(θ))). Its gradient adds ∂e/∂S ∂S/∂θ, which vanishes only if S minimizes e (the
envelope theorem); it is fit by least squares while e is the clipped L1 of the relative
error, so the detached gradient g may not be L*'s.

Per draw (scene, views, pixel subset) and adapter state, in fp32: g from the detached S
and g* from a differentiable Umeyama (SVD, float64), their cosine, and central finite
differences of the losses along -g/|g| and -g*/|g*| at step lengths `--steps` (norm of the
whole parameter vector), divided by the slope the gradient predicts (1 = exact):

- autograd: L with S frozen along -g, checks backprop through the checkpointed loop;
- detached: L* along -g, does the training gradient lower the measured loss?
- exact: L* along -g*, checks g*.

States: zero (B_k = 0, A from the kept state), E52d arm A's kept state (step 1000), and
that state with B_k doubled (its norm at ~step 2000 in arm A, a stand-in for the diverging
phase; same direction, which E52d did not check). The mean gradient over a state's draws
is compared too, as training follows that average.

uv run python experiments/scripts/e52d_gradient_check.py --cfg configs/e52a_adapters.yaml --state runs/e52d_A/pool-grid.pt --out runs/e52d_gradcheck
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from e52a_segment_adapters import (
    Scene,
    adapter_for,
    codes_of,
    forward,
    load_scene,
    scenes_of,
    segments_of,
    setup,
)
from torch import Tensor

from src.analysis.scene_eval import git_hash
from src.backbone.adapters import SegmentAdapter
from src.evaluation.labeled import umeyama, umeyama_torch


class Draw:
    """One scene, view subset and pixel subset; the model's points and both losses."""

    def __init__(
        self,
        net: torch.nn.Module,
        adapter: SegmentAdapter,
        scene: Scene,
        sel: np.ndarray,
        cfg: dict,
        seed: int,
    ) -> None:
        """Fix the inputs; the pixel subset as training draws it from `seed`."""
        self.net, self.adapter, self.cfg = net, adapter, cfg
        self.device = next(net.parameters()).device
        self.images = (
            torch.from_numpy(scene.images[sel]).permute(0, 3, 1, 2).float() / 255
        )[None].to(self.device)
        self.codes = codes_of(segments_of(scene, "grid", sel, cfg), "grid")[0]
        valid = torch.from_numpy(scene.depth[sel] > 0).to(self.device)
        self.valid = valid
        self.y = torch.from_numpy(scene.points[sel]).to(self.device)[valid]
        self.depth = torch.from_numpy(scene.depth[sel]).to(self.device)[valid]
        n = int(valid.sum())
        rng = np.random.default_rng(seed)
        self.pick = torch.from_numpy(
            rng.choice(n, size=min(cfg["umeyama_pixels"], n), replace=False)
        ).to(self.device)

    def points(self) -> Tensor:
        """Return the model's world points on the GT pixels, with grad to the adapter."""
        out = forward(self.net, self.images, self.adapter, self.codes)
        return out["world_points"][0].float()[self.valid]

    def loss(self, x: Tensor, sim: tuple[Tensor, Tensor, Tensor]) -> Tensor:
        """Clipped L1 of the relative 3D error under the Sim(3) (s, R, t)."""
        s, r, t = (v.to(x.dtype) for v in sim)
        err = (x @ (s * r).T + t - self.y).norm(dim=-1) / self.depth
        assert torch.isfinite(err).all()
        return err.clamp(max=self.cfg["max_error"]).mean()

    def fit(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Least-squares Sim(3) of the pixel subset, differentiable, float64."""
        return umeyama_torch(x[self.pick].double(), self.y[self.pick].double())


def flat(params: list[Tensor], grads: tuple[Tensor, ...]) -> Tensor:
    """One vector of the gradients (zeros where none)."""
    return torch.cat(
        [
            (torch.zeros_like(p) if g is None else g).flatten()
            for p, g in zip(params, grads, strict=True)
        ]
    )


def shifted(params: list[Tensor], theta: list[Tensor], step: Tensor) -> None:
    """Set the parameters to theta + step (one flat vector)."""
    i = 0
    with torch.no_grad():
        for p, p0 in zip(params, theta, strict=True):
            p.copy_(p0 + step[i : i + p.numel()].view_as(p))
            i += p.numel()


def check_draw(
    draw: Draw, params: list[Tensor], lengths: list[float]
) -> tuple[dict, Tensor, Tensor]:
    """Gradients, their cosine and the finite-difference ratios of one draw."""
    theta = [p.detach().clone() for p in params]
    x = draw.points()
    sim_det = tuple(v.detach() for v in draw.fit(x.detach()))
    ref = umeyama(x[draw.pick].detach().double().cpu().numpy(),
                  draw.y[draw.pick].double().cpu().numpy())  # fmt: skip
    assert np.isclose(ref[0], sim_det[0].item(), rtol=1e-9), (ref[0], sim_det[0])
    assert np.allclose(ref[1], sim_det[1].cpu().numpy(), atol=1e-9)
    loss_det = draw.loss(x, sim_det)
    loss_star = draw.loss(x, draw.fit(x))
    assert torch.isclose(loss_det, loss_star.to(loss_det.dtype), rtol=1e-6)
    g = flat(
        params,
        torch.autograd.grad(loss_det, params, retain_graph=True, allow_unused=True),
    )
    g_star = flat(params, torch.autograd.grad(loss_star, params, allow_unused=True))
    row = {
        "loss": loss_det.item(),
        "grad_norm": g.norm().item(),
        "exact_grad_norm": g_star.norm().item(),
        "cosine": torch.nn.functional.cosine_similarity(g, g_star, dim=0).item(),
    }
    for name, direction, measured in (
        ("autograd", g, "frozen"), ("detached", g, "refit"), ("exact", g_star, "refit"),
    ):  # fmt: skip
        unit = -direction / direction.norm()
        predicted = direction @ unit  # the slope the gradient promises: -|direction|
        for h in lengths:
            values = []
            for sign in (1, -1):
                shifted(params, theta, sign * h * unit)
                with torch.no_grad():
                    xs = draw.points()
                    sim = sim_det if measured == "frozen" else draw.fit(xs)
                    values.append(draw.loss(xs, sim).item())
            slope = (values[0] - values[1]) / (2 * h)
            row[f"{name}_ratio_{h:g}"] = slope / predicted.item()
    shifted(params, theta, torch.zeros_like(g))
    return row, g, g_star


def main() -> None:
    """Load the kept adapter, check every state on every draw, write the records."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True, help="arm A's kept .pt")
    parser.add_argument("--splits", nargs="+", default=["dev"])
    parser.add_argument("--draws", type=int, default=3, help="per scene and state")
    parser.add_argument("--views", type=int, default=4)
    parser.add_argument("--steps", type=float, nargs="+", default=[1e-3, 1e-2])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(args.cfg.read_text(encoding="utf-8")) | {"autocast": False}
    raw = yaml.safe_load(Path(cfg["eval"]).read_text(encoding="utf-8"))
    args.out.mkdir(parents=True, exist_ok=False)
    _, net = setup(cfg)
    device = next(net.parameters()).device
    adapter = adapter_for(cfg, "pool", device)
    kept = torch.load(args.state, map_location=device)["state"]
    states = {"zero": {**kept, "up": torch.zeros_like(kept["up"])}, "kept": kept,
              "kept_up_x2": {**kept, "up": 2 * kept["up"]}}  # fmt: skip
    params = [adapter.up, adapter.down.weight]
    scenes = [load_scene(raw, n) for n in scenes_of(cfg, raw, args.splits)]
    rng = np.random.default_rng(cfg["seed"])
    rec: dict = {"rows": [], "mean_gradient": {}}
    t0 = time.perf_counter()
    for state, sd in states.items():
        adapter.load_state_dict(sd)
        sums = [
            torch.zeros(sum(p.numel() for p in params), device=device) for _ in "ab"
        ]
        for scene in scenes:
            for _ in range(args.draws):
                k = min(args.views, len(scene.images))
                sel = np.sort(rng.choice(len(scene.images), size=k, replace=False))
                draw = Draw(net, adapter, scene, sel, cfg, int(rng.integers(2**31)))
                row, g, g_star = check_draw(draw, params, args.steps)
                sums[0] += g
                sums[1] += g_star
                row |= {"state": state, "scene": scene.name, "views": sel.tolist()}
                rec["rows"].append(row)
                print(f"{state} {scene.name} {sel.tolist()}: "
                      + ", ".join(f"{k} {v:.4g}" for k, v in row.items()
                                  if isinstance(v, float)) + f" ({time.perf_counter() - t0:.0f} s)",
                      flush=True)  # fmt: skip
        cos = torch.nn.functional.cosine_similarity(sums[0], sums[1], dim=0).item()
        rec["mean_gradient"][state] = {"cosine": cos}
        print(f"{state}: cosine of the mean gradients {cos:.4f}", flush=True)
    summary = {"args": {k: str(v) for k, v in vars(args).items()}, "git": git_hash()}
    (args.out / "e52d_gradcheck.yaml").write_text(
        yaml.safe_dump(summary | {"config": cfg, "records": rec}, sort_keys=False,
                       allow_unicode=True), encoding="utf-8")  # fmt: skip


if __name__ == "__main__":
    main()
