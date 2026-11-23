"""E84 on Niessner's ladder (issue #84): a keep head trained on single views, train and val logged every step.

A sample is one view: Déjà View's 66 cached channels (RGB and depth are inside them) in, a
keep probability s per pixel out. Its loss is the expected Chamfer (cm) of that view's kept
cloud against that view's GT points (`expected_chamfer`: R = 8 probes, the Poisson tail).
Rung 3 trains on one view, rung 4 on one view of each dev scene, rung 5 on growing sets of
views and validates on val scenes. A step is one batch of train views (forward and
backward) plus one random batch of val views (forward only), both logged to TensorBoard.
The theoretical optimum of a sample is its loss with free per-pixel logits (no network,
`free_optimum`), the uncut cloud (s = 1) its trivial baseline; both are logged as
constants. The network is a shallow dilated ResNet (`KeepResNet`) on inputs standardized
by a fixed affine map over the training views; constant Adam, no schedule, no gain.

`uv run python experiments/scripts/e84_ladder.py --cfg configs/e84_rung3.yaml`
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.tensorboard import SummaryWriter

from src.analysis.scene_eval import git_hash
from src.loaders.views import Views
from src.refinement.keep_chamfer import (
    KeepResNet,
    KeepScene,
    exact_chamfer,
    expected_chamfer,
    standardize,
)


def gain_at(step: int, cfg: dict) -> float:
    """Return the logit gain of `objective: anneal` at a step: geometric from gain[0] to gain[1]."""
    low, high = cfg["gain"]
    return low * (high / low) ** (step / cfg["steps"])


def sample_loss(
    logit: torch.Tensor, sample: KeepScene, cfg: dict, gain: float = 1.0
) -> torch.Tensor:
    """Return the expected Chamfer (cm) of the sample's s = sigmoid(logit) (1, H, W).

    With `objective: cut` the forward s is the deployed cut (1 where logit >= 0, else 0)
    and the gradient is the sigmoid's (straight-through): the loss is then the Chamfer
    of the cloud a threshold delivers, not of random keeps. With `objective: anneal` s =
    sigmoid(gain * logit): the random-keep loss at gain 1, the cut as the gain grows; the
    cut (logit >= 0) does not depend on the gain.
    """
    s = torch.sigmoid(gain * logit)[sample.valid]
    if cfg.get("objective") == "cut":
        s = (s >= 0.5).float() + (s - s.detach())
    return expected_chamfer(s, *sample.tensors, cfg["eps"], tail=cfg["tail"])[0]


def batch_loss(
    net: KeepResNet, batch: list[KeepScene], cfg: dict, gain: float = 1.0
) -> torch.Tensor:
    """Return the mean sample loss of a batch, one forward pass over its views."""
    batch = [s.to("cuda") for s in batch]
    x = net.logits(torch.cat([s.features for s in batch]).float())
    losses = [sample_loss(x[i : i + 1], s, cfg, gain) for i, s in enumerate(batch)]
    return torch.stack(losses).mean()


def mimic_batch(
    net: KeepResNet, batch: list[KeepScene], targets: dict[str, torch.Tensor]
) -> torch.Tensor:
    """Return the mean cross-entropy of the batch's s against its targets (E84r's test of the net)."""
    batch = [s.to("cuda") for s in batch]
    x = net.logits(torch.cat([s.features for s in batch]).float())
    return torch.stack(
        [
            torch.nn.functional.binary_cross_entropy_with_logits(
                x[i][s.valid[0]], targets[s.name]
            )
            for i, s in enumerate(batch)
        ]
    ).mean()


def free_fit(
    sample: KeepScene, cfg: dict, block: int = 1
) -> tuple[float, torch.Tensor]:
    """Return the loss and s (valid pixels) of free logits constant on block x block cells.

    Adam from s = 0.5. block = 1 is the sample's optimum with no network; a larger block is
    the optimum of any function that cannot tell pixels within a cell apart.
    """
    sample = sample.to("cuda")
    h, w = sample.valid.shape[1:]
    z = torch.zeros(1, 1, -(-h // block), -(-w // block), device="cuda")
    z.requires_grad_()
    optimizer = torch.optim.Adam([z], lr=cfg["free_lr"])

    def expand() -> torch.Tensor:
        return torch.nn.functional.interpolate(z, scale_factor=block)[:, 0, :h, :w]

    best, x_best = float("inf"), expand().detach()
    for _ in range(cfg["free_steps"]):
        optimizer.zero_grad()
        x = expand()
        loss = sample_loss(x, sample, cfg)
        if loss.item() < best:  # the cut objective is not monotone under Adam
            best, x_best = loss.item(), x.detach()
        loss.backward()
        optimizer.step()
    return best, torch.sigmoid(x_best)[sample.valid]


def uncut(sample: KeepScene, cfg: dict) -> float:
    """Return the loss of keeping every pixel (s = sigmoid(20))."""
    sample = sample.to("cuda")
    ones = torch.full(sample.valid.shape, 20.0, device=sample.valid.device)
    return sample_loss(ones, sample, cfg).item()


def view_count(scene: str, cfg: dict) -> range:
    """Return the indices of a scene's views (a scene can have fewer than 20)."""
    path = Path(cfg["root"]) / scene / f"{cfg['views']}.npz"
    return range(len(Views.load(path)))


def load(
    spec: dict[str, list[int] | str], cfg: dict, n: int | None = None
) -> list[KeepScene]:
    """Return the views of a {scene: views or "all"} spec as samples on `store`.

    With `n`, a fixed random subset of n of the spec's (scene, view) pairs; the subsets
    of growing n are nested (the first n of one seeded permutation).
    """
    pairs = [
        (scene, v)
        for scene, views in spec.items()
        for v in (view_count(scene, cfg) if views == "all" else views)
    ]
    if n is not None:
        order = np.random.default_rng(cfg["subset_seed"]).permutation(len(pairs))
        pairs = [pairs[i] for i in order[:n]]
    out = []
    for scene in dict.fromkeys(s for s, _ in pairs):
        which = [v for s, v in pairs if s == scene]
        out += KeepScene.views(scene, cfg, which, cfg["store"])
    return out


def draw(samples: list[KeepScene], n: int, rng: np.random.Generator) -> list[KeepScene]:
    """Return min(n, len) samples drawn without replacement."""
    return [
        samples[i]
        for i in rng.choice(len(samples), min(n, len(samples)), replace=False)
    ]


def cut_chamfer(net: KeepResNet, sample: KeepScene) -> dict[str, float]:
    """Return the exact Chamfer (cm) of the cut s >= 0.5 of a sample."""
    sample = sample.to("cuda")
    with torch.no_grad():
        x = net.logits(sample.features.float())[sample.valid]
    kept = (x >= 0).cpu().numpy()
    return exact_chamfer(kept, sample.probes, sample.pred, sample.truth)


def main() -> None:
    """Train, log every step, write the run folder."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cfg", type=Path, required=True)
    cfg = yaml.safe_load(parser.parse_args().cfg.read_text())
    torch.manual_seed(cfg["seed"])
    rng = np.random.default_rng(cfg["seed"])
    out = Path("runs") / f"{cfg['name']}-{time.strftime('%Y%m%d-%H%M%S')}"
    out.mkdir(parents=True)
    train = load(cfg["train"], cfg, cfg.get("n_train"))
    val = load(cfg["val"], cfg) if cfg.get("val") else []
    standardize(train + val, reference=train)
    eval_train, eval_val = (s[:: max(1, len(s) // cfg["n_eval"])] for s in (train, val))
    net = KeepResNet(**cfg["resnet"]).cuda()
    optimizer = torch.optim.Adam(net.parameters(), lr=cfg["lr"])
    anneal = cfg.get("objective") == "anneal"
    ref_cfg = cfg | {"objective": "cut"} if anneal else cfg  # the references are cuts
    reference = {
        s.name: {"uncut": uncut(s, ref_cfg)}
        | {f"optimum_block{k}": free_fit(s, ref_cfg, k)[0] for k in cfg["blocks"]}
        for s in eval_train
    }
    val_uncut = float(np.mean([uncut(s, cfg) for s in eval_val])) if val else None
    targets = (
        {s.name: free_fit(s, cfg)[1] for s in train}
        if cfg.get("objective") == "mimic"
        else {}
    )
    print(reference, flush=True)
    groups = [("train", eval_train)] + ([("val", eval_val)] if val else [])
    seconds, history = [], []
    with SummaryWriter(out / "tb") as writer:
        for step in range(cfg["steps"] + 1):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            gain = gain_at(step, cfg) if anneal else 1.0
            batch = draw(train, cfg["batch"], rng)
            optimizer.zero_grad()
            if cfg.get("objective") == "mimic":
                loss = mimic_batch(net, batch, targets)
            else:
                loss = batch_loss(net, batch, cfg, gain)
            loss.backward()
            optimizer.step()
            if val:
                with torch.no_grad():
                    val_loss = batch_loss(net, draw(val, cfg["batch"], rng), cfg, gain)
                writer.add_scalar("loss/val", val_loss.item(), step)
            torch.cuda.synchronize()
            seconds.append(time.perf_counter() - t0)
            writer.add_scalar("loss/train", loss.item(), step)
            writer.add_scalar("gain", gain, step)
            if step % cfg["eval_every"] == 0:
                with torch.no_grad():
                    losses = {
                        k: [batch_loss(net, [s], cfg, gain).item() for s in group]
                        for k, group in groups
                    }
                row = {f"{k}_loss": float(np.mean(v)) for k, v in losses.items()}
                if step % cfg["cut_every"] == 0:
                    row |= {
                        f"{k}_cut": float(
                            np.mean([cut_chamfer(net, s)["chamfer_cm"] for s in group])
                        )
                        for k, group in groups
                    }
                history.append({"step": step, **row, "train_each": losses["train"]})
                print(
                    f"step {step}: batch {cfg.get('objective', 'chamfer')} train "
                    f"{loss.item():.3f} | eval "
                    + " ".join(f"{k} {v:.3f}" for k, v in row.items())
                    + f" | {1000 * np.median(seconds[-cfg['eval_every'] :]):.0f} ms/step",
                    flush=True,
                )
    torch.save(net.state_dict(), out / "net.pt")
    (out / "run.yaml").write_text(
        yaml.safe_dump(
            {
                "config": cfg,
                "git": git_hash(),
                "reference": reference,
                "val_uncut_mean": val_uncut,
                "step_ms_median": 1000 * float(np.median(seconds)),
                "history": history,
            },
            sort_keys=False,
        )
    )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
