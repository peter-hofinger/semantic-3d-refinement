"""Hypersim settings, read from a YAML file (see configs/hypersim.yaml)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from src.loaders.scannetpp.config import PrepareConfig


@dataclass
class HypersimConfig:
    """Paths, scenes (scene name -> camera trajectory) and settings for Hypersim."""

    data_root: Path
    metadata_root: Path
    output_root: Path
    scenes: dict[str, str]
    prepare: PrepareConfig = field(default_factory=PrepareConfig)


def load_config(path: Path) -> HypersimConfig:
    """Read a Hypersim config file."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return HypersimConfig(
        data_root=Path(raw["data_root"]),
        metadata_root=Path(raw["metadata_root"]),
        output_root=Path(raw["output_root"]),
        scenes={str(k): str(v) for k, v in raw["scenes"].items()},
        prepare=PrepareConfig(**raw.get("prepare", {})),
    )
