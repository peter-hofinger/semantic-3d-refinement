"""Scene folders per split, read from a dataset's split file.

Two formats: a Hypersim config (`configs/hypersim.yaml`, split -> {scene: camera},
checked against Hypersim's official partition; the folder is `<scene>_<camera>`) and a
plain list per split (`sage-prepare`'s `splits.yaml`, split -> [room folder]).
"""

from __future__ import annotations

from pathlib import Path

import yaml

from src.loaders.hypersim.config import load_config


def scene_folders(path: Path, splits: list[str]) -> dict[str, list[str]]:
    """Scene folder names of each requested split."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if "splits" in raw:
        config = load_config(path)
        return {s: [f"{n}_{c}" for n, c in config.scenes([s]).items()] for s in splits}
    unknown = set(splits) - set(raw)
    if unknown:
        msg = f"unknown splits {sorted(unknown)}, have {list(raw)}"
        raise ValueError(msg)
    return {s: [str(n) for n in raw[s] or []] for s in splits}
