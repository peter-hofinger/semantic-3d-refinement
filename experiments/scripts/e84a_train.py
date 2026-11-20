"""E84a (issue #84): train the keep head on the expected Chamfer of the kept cloud.

One head (`src.refinement.keep_chamfer`, a copy of Déjà View's confidence branch) is the
only trainable part, one loss, whole scenes per step (see the module's docstring for how
the whole-scene sums are backpropagated). Rung 3 of docs/experiments.md is the bedroom
alone, rung 4 the dev scenes: the config's `scenes`. A step is every scene once, their
losses averaged. Every `eval_every` steps the exact Chamfer of the thresholded cloud is
logged next to the loss, per scene, so a falling loss with a rising exact completeness
shows. Writes `runs/<cfg>-<time>/{run.yaml, head.pt, probabilities_<scene>[_step<k>].npy, tb/}`,
the snapshots at the config's `snapshots` steps (before that step's update).

`uv run python experiments/scripts/e84a_train.py --cfg configs/e84a_overfit_bedroom.yaml`
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.tensorboard import SummaryWriter

from src.analysis.scene_eval import git_hash
from src.backbone.dejaview import DejaViewBackbone
from src.refinement.keep_chamfer import (
    KeepHead,
    KeepScene,
    exact_chamfer,
    loss_backward,
    probabilities,
)


def evaluate(
    head: KeepHead, scene: KeepScene, chunk: int, threshold: float, seed: int
) -> dict:
    """Exact Chamfer of the cloud of s >= threshold and of one Bernoulli(s) draw, and s's spread."""
    with torch.no_grad():
        s = probabilities(head, scene.features, chunk)[scene.valid]
        draw = torch.bernoulli(s, generator=torch.Generator(s.device).manual_seed(seed))

    def score(kept: torch.Tensor) -> dict[str, float]:
        return exact_chamfer(kept.cpu().numpy(), scene.probes, scene.pred, scene.truth)

    return {
        "threshold": score(s >= threshold),
        "bernoulli": score(draw.bool()),
        "s": {
            "mean": s.mean().item(),
            "above_half": (s >= 0.5).float().mean().item(),
            "undecided": ((s > 0.05) & (s < 0.95)).float().mean().item(),
        },
    }


def temperature(cfg: dict | None, step: int) -> float:
    """Logit temperature at a step: geometric from cfg's start to end over cfg's steps."""
    if cfg is None:
        return 1.0
    fraction = min(step / cfg["steps"], 1.0)
    return cfg["start"] * (cfg["end"] / cfg["start"]) ** fraction


def sync() -> float:
    """Wall time after the GPU's queue is empty."""
    torch.cuda.synchronize()
    return time.perf_counter()


def main() -> None:
    """Train the head on the config's scenes and write the run folder."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cfg", type=Path, required=True)
    args = parser.parse_args()
    raw = yaml.safe_load(args.cfg.read_text())
    torch.manual_seed(raw["seed"])
    out = Path("runs") / f"{args.cfg.stem}-{time.strftime('%Y%m%d-%H%M%S')}"
    out.mkdir(parents=True)

    t0 = sync()
    scenes = [KeepScene.load(name, raw, "cuda") for name in raw["scenes"]]
    load_s = sync() - t0
    if raw.get("init") == "deja_view":
        steps = int(raw["backbone"].removeprefix("dejaview_k"))
        head = KeepHead.from_deja_view(DejaViewBackbone(steps=steps).output_block())
    else:
        head = KeepHead()
    head = head.cuda()
    optimizer = torch.optim.Adam(head.parameters(), lr=raw["lr"])
    history, step_s = [], []
    with SummaryWriter(out / "tb") as writer:
        for step in range(raw["steps"] + 1):
            head.temperature = temperature(raw.get("temperature"), step)
            t0 = sync()
            optimizer.zero_grad()
            record = [
                loss_backward(
                    head,
                    s,
                    raw["eps"],
                    raw["chunk"],
                    1 / len(scenes),
                    tail=raw.get("tail", False),
                )
                for s in scenes
            ]
            step_s.append(sync() - t0)
            if step in raw["snapshots"]:
                for s in scenes:
                    with torch.no_grad():
                        p = probabilities(head, s.features, raw["chunk"])
                    np.save(
                        out / f"probabilities_{s.name}_step{step}.npy",
                        p.cpu().numpy().astype(np.float16),
                    )
            if step < raw["steps"]:
                optimizer.step()
            mean = {k: float(np.mean([r[k] for r in record])) for k in record[0]}
            for key, value in mean.items():
                writer.add_scalar(f"train/{key}", value, step)
            if step % raw["eval_every"] == 0:
                exact = [
                    evaluate(head, s, raw["chunk"], raw["threshold"], step)
                    for s in scenes
                ]
                for s, e in zip(scenes, exact, strict=True):
                    for cut, scores in e.items():
                        for key, value in scores.items():
                            writer.add_scalar(f"{cut}/{s.name}/{key}", value, step)
                history.append(
                    {
                        "step": step,
                        **mean,
                        "temperature": head.temperature,
                        "exact": dict(zip(raw["scenes"], exact, strict=True)),
                    }
                )
                print(
                    f"step {step}: L {mean['loss']:.3f} A {mean['accuracy_cm']:.3f} "
                    f"C {mean['completeness_cm']:.3f} | mean s {mean['mean_s']:.3f} "
                    f"(T {head.temperature:.2f}) | "
                    + " | ".join(
                        f"{cut} "
                        + " ".join(
                            f"{e[cut]['chamfer_cm']:.3f} (A {e[cut]['accuracy_cm']:.3f} "
                            f"C {e[cut]['completeness_cm']:.3f} kept {e[cut]['kept']:.2f})"
                            for e in exact
                        )
                        for cut in ("threshold", "bernoulli")
                    )
                    + f" | {step_s[-1]:.2f} s/step"
                )
    torch.save(head.state_dict(), out / "head.pt")
    for s in scenes:
        with torch.no_grad():
            p = probabilities(head, s.features, raw["chunk"])
        np.save(out / f"probabilities_{s.name}.npy", p.cpu().numpy().astype(np.float16))
    (out / "run.yaml").write_text(
        yaml.safe_dump(
            {
                "config": raw,
                "git": git_hash(),
                "seconds": {
                    "load_scenes": load_s,
                    "step_median": float(np.median(step_s)),
                    "peak_gpu_gb": torch.cuda.max_memory_allocated() / 1e9,
                },
                "history": history,
            },
            sort_keys=False,
        )
    )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
