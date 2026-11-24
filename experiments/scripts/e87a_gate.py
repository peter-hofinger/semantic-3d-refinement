"""E87a gate analysis: where the per-token gate on the segment pooling opens.

The pooling step of rung 5's kept states is x_i += g_i δ_i, δ_i = B_k A (m_s(i) - x_i),
g_i = sigmoid(w_k · x_i + b_k) (`src/backbone/adapters.py`). Its dev score (E87a rung 5)
says the gate changes nothing on average; this reads the gate itself, on the 3 dev
scenes at their 20 views, every loop step k: g_i, and the pull the token takes,
|g_i δ_i| / |x_i| (the gated update relative to the token). Tokens are the patch tokens
that carry a GT instance (the same set for both arms; grid tiles cover every patch), each
with its GT class (NYU40 name of its instance) and whether its patch holds a GT depth
edge (`src/analysis/edges.py`, τ = 5%).

`--collect` (GPU, 3080): each run's kept adapter `runs/e87a_main_<arm>_s<s>/*.pt`, built
from that run's own config, through `adapters.forward` under bf16 autocast as in
training; `token_gate` is wrapped to record g and δ per step.
-> `runs/e87a_gate/<arm>_s<s>_<scene>.npz` (g, pull: (K, tokens); class, edge, view per
token).
Default (CPU): those files -> `figures/e87a_gate.jpg`, `results/e87a_gate.yaml`.

uv run python experiments/scripts/e87a_gate.py --collect
uv run python experiments/scripts/e87a_gate.py
"""

import argparse
from pathlib import Path

import matplotlib as mpl
import numpy as np
import torch
import yaml

mpl.use("Agg")
import matplotlib.pyplot as plt
from e52a_figures import FIGURES, style
from e52a_segment_adapters import Scene, adapter_for, codes_of, segments_of, setup

import src.backbone.adapters as ad
from src.analysis.edges import depth_edges
from src.analysis.scene_eval import input_paths
from src.loaders.ground_truth import GroundTruth
from src.loaders.views import Views
from src.segmentation.base import NO_INSTANCE

RUNS = Path("runs")
OUT = RUNS / "e87a_gate"
RESULTS = Path("experiments/results/e87a_gate.yaml")
ARMS = {"gt": "#eb6834", "grid": "#3a7bd5"}
SEEDS = (0, 1, 2)
SCENES = {
    "ai_010_007_cam_00": "living room",
    "ai_006_007_cam_00": "bedroom",
    "ai_044_001_cam_00": "lobby",
}
MAP_SCENE, MAP_VIEW = "ai_006_007_cam_00", 18  # bedroom, issue #7
TOP_CLASSES = 12  # classes shown, by token count over the dev scenes


class Recorder:
    """Wraps `token_gate`: keeps g and |δ| / |x| per patch token for every call (step k)."""

    def __init__(
        self, adapter: ad.SegmentAdapter, joint: torch.Tensor, rows: np.ndarray
    ) -> None:
        """Joint codes (1, V T, S); rows: flat indices of the analysed patch tokens."""
        self.adapter, self.joint, self.rows = adapter, joint, torch.from_numpy(rows)
        self.g: list[np.ndarray] = []
        self.pull: list[np.ndarray] = []
        self.inner = ad.token_gate

    def __call__(
        self, x: torch.Tensor, w: torch.Tensor, b: torch.Tensor
    ) -> torch.Tensor:
        """Return the gate as in training; record it with the update it scales."""
        g = self.inner(x, w, b)
        k = len(self.g)
        assert self.adapter.up is not None
        assert self.adapter.down is not None
        delta = (
            self.adapter.down(ad.segment_offsets(x, self.joint)) @ self.adapter.up[k]
        )
        rows = self.rows.to(x.device)
        gi = g[0, rows, 0].float()
        size = delta[0, rows].float().norm(dim=-1) / x[0, rows].float().norm(dim=-1)
        assert torch.isfinite(size).all()
        self.g.append(gi.cpu().numpy())
        self.pull.append((gi * size).cpu().numpy())
        return g


def token_table(gt: GroundTruth, patch_ids: np.ndarray) -> dict[str, np.ndarray]:
    """Class, edge flag and view of every patch with a GT instance, and its flat index."""
    v, h, w = gt.depth.shape
    p = ad.PATCH_SIZE
    edges = np.stack([depth_edges(d, d > 0) for d in gt.depth])
    edge = edges.reshape(v, h // p, p, w // p, p).any(axis=(2, 4)).reshape(v, -1)
    view, patch = np.nonzero(patch_ids != NO_INSTANCE)
    tokens = ad.SPECIAL_TOKENS + patch_ids.shape[1]
    names = np.array(
        [gt.classes.get(int(i), "unknown") for i in patch_ids[view, patch]]
    )
    return {
        "row": view * tokens + ad.SPECIAL_TOKENS + patch,
        "view": view,
        "patch": patch,
        "class": names,
        "edge": edge[view, patch],
    }


def collect() -> None:
    """Record the gate of every kept state on every dev scene (GPU)."""
    OUT.mkdir(parents=True, exist_ok=True)
    net = None
    for arm in ARMS:
        for s in SEEDS:
            run = RUNS / f"e87a_main_{arm}_s{s}"
            rec = yaml.safe_load((run / "e52a.yaml").read_text(encoding="utf-8"))
            cfg = rec["config"]
            raw = raw_of(cfg)
            if net is None:
                _, net = setup(cfg)
            device = next(net.parameters()).device
            label = f"lora_pool_gate-{arm}"
            adapter = adapter_for(cfg, "lora_pool_gate", device)
            state = torch.load(run / f"{label}.pt", map_location=device)["state"]
            adapter.load_state_dict(state)
            for name in SCENES:
                paths = input_paths(raw, name)
                views, gt = Views.load(paths["views"]), GroundTruth.load(paths["gt"])
                patch_ids = ad.patch_segments(gt.instance_ids)
                table = token_table(gt, patch_ids)
                scene = Scene(name, views.images, gt.depth, np.empty(0), patch_ids)
                sel = np.arange(len(views.images))
                codes, _ = codes_of(segments_of(scene, arm, sel, cfg), arm)
                joint = codes.to(device).reshape(1, -1, codes.shape[-1])
                rec_ = Recorder(adapter, joint, table["row"])
                ad.token_gate = rec_
                images = (
                    torch.from_numpy(views.images).permute(0, 3, 1, 2).float() / 255
                )
                try:
                    with (
                        torch.no_grad(),
                        torch.autocast(device.type, dtype=torch.bfloat16),
                    ):
                        ad.forward(net, images[None].to(device), adapter, codes)
                finally:
                    ad.token_gate = rec_.inner
                assert len(rec_.g) == net.inference_steps, len(rec_.g)
                np.savez_compressed(
                    OUT / f"{arm}_s{s}_{name}.npz",
                    g=np.stack(rec_.g).astype(np.float32),
                    pull=np.stack(rec_.pull).astype(np.float32),
                    **{k: v for k, v in table.items() if k != "row"},
                )
                print(f"{arm} s{s} {name}: mean g {np.mean(rec_.g):.3f}", flush=True)


def raw_of(cfg: dict) -> dict:
    """Load the evaluation config a run's config names."""
    return yaml.safe_load(Path(cfg["eval"]).read_text(encoding="utf-8"))


def load(arm: str, s: int) -> dict[str, np.ndarray]:
    """One run's tokens over the three dev scenes, concatenated; scene per token."""
    parts = [np.load(OUT / f"{arm}_s{s}_{n}.npz") for n in SCENES]
    out = {k: np.concatenate([p[k] for p in parts], axis=-1) for k in ("g", "pull")}
    for k in ("class", "edge", "view", "patch"):
        out[k] = np.concatenate([p[k] for p in parts])
    out["scene"] = np.concatenate(
        [np.full(len(p["edge"]), n) for n, p in zip(SCENES, parts, strict=True)]
    )
    return out


def numbers() -> dict:
    """Gate and pull per step, by class and at edges, per run; seed means per arm."""
    data = {(a, s): load(a, s) for a in ARMS for s in SEEDS}
    first = data["gt", 0]
    for d in data.values():
        assert (d["class"] == first["class"]).all(), "token sets differ between runs"
    names, counts = np.unique(first["class"], return_counts=True)
    top = [str(n) for n in names[np.argsort(-counts)][:TOP_CLASSES]]
    out: dict = {"tokens": len(first["edge"]), "edge_share": float(first["edge"].mean()),
                 "classes": {c: int(counts[names == c][0]) for c in top}, "runs": {}}  # fmt: skip
    for (arm, s), d in data.items():
        g, pull = d["g"], d["pull"]
        out["runs"][f"{arm}_s{s}"] = {
            "g_mean": float(g.mean()),
            "g_per_step": [float(x) for x in g.mean(1)],
            "g_p10_p90_per_step": [[float(x) for x in np.percentile(r, [10, 90])] for r in g],
            "g_std_tokens": float(g.mean(0).std()),
            "share_g_below_0.1": float((g.mean(0) < 0.1).mean()),
            "share_g_above_0.9": float((g.mean(0) > 0.9).mean()),
            "pull_mean": float(pull.mean()),
            "pull_per_step": [float(x) for x in pull.mean(1)],
            "edge": {"g": float(g[:, d["edge"]].mean()), "pull": float(pull[:, d["edge"]].mean())},
            "interior": {"g": float(g[:, ~d["edge"]].mean()),
                         "pull": float(pull[:, ~d["edge"]].mean())},
            "by_class_g": {c: float(g[:, d["class"] == c].mean()) for c in top},
            "by_class_pull": {c: float(pull[:, d["class"] == c].mean()) for c in top},
            "by_scene_g": {SCENES[n]: float(g[:, d["scene"] == n].mean()) for n in SCENES},
        }  # fmt: skip
    return out


def figure(res: dict) -> None:
    """Gate per loop step; by class; edge vs interior; bedroom view 18 maps."""
    fig = plt.figure(figsize=(18, 9.5))
    grid = fig.add_gridspec(2, 4, height_ratios=[1, 1.05])
    ax = fig.add_subplot(grid[0, 0])
    for arm, colour in ARMS.items():
        for s in SEEDS:
            r = res["runs"][f"{arm}_s{s}"]
            k = np.arange(len(r["g_per_step"]))
            ax.plot(k, r["g_per_step"], color=colour, ls=("-", "--", ":")[s],
                    label=f"{arm}, seed {s}")  # fmt: skip
            if s == 0:
                band = np.array(r["g_p10_p90_per_step"])
                ax.fill_between(k, band[:, 0], band[:, 1], color=colour, alpha=0.15)
    ax.axhline(0.5, color="0.4", lw=0.8, ls=":", label="start (g = 1/2)")
    ax.set_ylim(0, 1)
    ax.set_xlabel("loop step k")
    ax.set_ylabel("gate g (mean over tokens; band: 10-90% seed 0)")
    ax.set_title("gate per loop step", fontsize=10)
    ax.legend(fontsize=6.5, frameon=False)
    ax = fig.add_subplot(grid[0, 1])
    for arm, colour in ARMS.items():
        for s in SEEDS:
            r = res["runs"][f"{arm}_s{s}"]
            ax.plot(r["pull_per_step"], color=colour, ls=("-", "--", ":")[s])
    ax.set_xlabel("loop step k")
    ax.set_ylabel("pull |g δ| / |x| (mean over tokens)")
    ax.set_title("gated update relative to the token", fontsize=10)
    ax = fig.add_subplot(grid[0, 2:])
    classes = list(res["classes"])
    width = 0.4
    for j, (arm, colour) in enumerate(ARMS.items()):
        vals = np.array(
            [
                [res["runs"][f"{arm}_s{s}"]["by_class_g"][c] for c in classes]
                for s in SEEDS
            ]
        )
        x = np.arange(len(classes)) + (j - 0.5) * width
        ax.bar(
            x,
            vals.mean(0),
            width * 0.9,
            color=colour,
            alpha=0.8,
            label=f"{arm}, seed mean",
        )
        for v in vals:
            ax.plot(x, v, "o", color="k", ms=2.5)
    ax.axhline(0.5, color="0.4", lw=0.8, ls=":")
    ax.set_xticks(range(len(classes)),
                  [f"{c}\n({res['classes'][c]})" for c in classes], fontsize=7)  # fmt: skip
    ax.set_ylabel("gate g (mean over steps and tokens)")
    ax.set_ylim(0, 1)
    ax.set_title(
        "gate by GT class (tokens on the 3 dev scenes; dots: seeds)", fontsize=10
    )
    ax.legend(fontsize=7, frameon=False)
    ax = fig.add_subplot(grid[1, 0])
    for j, (arm, colour) in enumerate(ARMS.items()):
        for i, where in enumerate(("interior", "edge")):
            vals = [res["runs"][f"{arm}_s{s}"][where]["g"] for s in SEEDS]
            x = i + (j - 0.5) * width
            ax.bar(x, np.mean(vals), width * 0.9, color=colour, alpha=0.8,
                   label=arm if i == 0 else None)  # fmt: skip
            ax.plot([x] * len(vals), vals, "o", color="k", ms=2.5)
    ax.axhline(0.5, color="0.4", lw=0.8, ls=":")
    ax.set_xticks([0, 1], ["patch without\nGT depth edge", "patch with\nGT depth edge"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("gate g (mean over steps and tokens)")
    ax.set_title(
        f"gate at depth edges ({100 * res['edge_share']:.0f}% of tokens)", fontsize=10
    )
    ax.legend(fontsize=7, frameon=False)
    maps = [(arm, 0) for arm in ARMS]
    cfg = yaml.safe_load(
        (RUNS / "e87a_main_gt_s0" / "e52a.yaml").read_text(encoding="utf-8")
    )
    images = Views.load(input_paths(raw_of(cfg["config"]), MAP_SCENE)["views"]).images
    img = images[MAP_VIEW]
    rows, cols = img.shape[0] // ad.PATCH_SIZE, img.shape[1] // ad.PATCH_SIZE
    ax = fig.add_subplot(grid[1, 1])
    ax.imshow(img)
    ax.set_title(f"bedroom, view {MAP_VIEW}", fontsize=10)
    ax.axis("off")
    for j, (arm, s) in enumerate(maps):
        d = np.load(OUT / f"{arm}_s{s}_{MAP_SCENE}.npz")
        m = np.full(rows * cols, np.nan)
        on = d["view"] == MAP_VIEW
        m[d["patch"][on]] = d["g"][:, on].mean(0)
        ax = fig.add_subplot(grid[1, 2 + j])
        im = ax.imshow(m.reshape(rows, cols), vmin=0, vmax=1, cmap="viridis")
        ax.set_title(
            f"gate g, {arm} seed {s} (mean over steps; blank: no GT instance)",
            fontsize=9,
        )
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.035)
    for a in fig.axes:
        if a.get_xlabel() or a.get_ylabel():
            style(a)
    g = {
        arm: np.mean([res["runs"][f"{arm}_s{s}"]["g_mean"] for s in SEEDS])
        for arm in ARMS
    }
    fig.suptitle(
        f"E87a, rung 5's kept states on the 3 dev scenes (20 views): mean gate g = {g['gt']:.2f} "
        f"(GT instances), {g['grid']:.2f} (grid tiles); start 0.5.",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(FIGURES / "e87a_gate.jpg", dpi=130)


def main() -> None:
    """Collect on the GPU, or write the numbers and the figure."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--collect", action="store_true")
    if parser.parse_args().collect:
        collect()
        return
    res = numbers()
    RESULTS.write_text(
        yaml.safe_dump(res, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    for run, r in res["runs"].items():
        print(f"{run}: g {r['g_mean']:.3f} (edge {r['edge']['g']:.3f}, interior "
              f"{r['interior']['g']:.3f}), <0.1 {r['share_g_below_0.1']:.3f}, "
              f">0.9 {r['share_g_above_0.9']:.3f}, pull {r['pull_mean']:.4f}")  # fmt: skip
    figure(res)


if __name__ == "__main__":
    main()
