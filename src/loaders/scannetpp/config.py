"""ScanNet++ settings, read from a YAML file (see configs/scannetpp.yaml)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class DownloadConfig:
    """What to fetch from the ScanNet++ server."""

    root_url: str
    assets: list[str]
    metadata: bool = True


@dataclass
class PrepareConfig:
    """How to turn a raw scene into backbone input."""

    num_frames: int = 20
    sampling: str = "uniform"
    long_side: int = 504
    patch_size: int = 14


@dataclass
class ScannetppConfig:
    """Paths, scenes and settings for ScanNet++."""

    data_root: Path
    output_root: Path
    scenes: list[str]
    download: DownloadConfig
    prepare: PrepareConfig = field(default_factory=PrepareConfig)


def load_config(path: Path) -> ScannetppConfig:
    """Read a ScanNet++ config file."""
    raw = yaml.safe_load(path.read_text())
    return ScannetppConfig(
        data_root=Path(raw["data_root"]),
        output_root=Path(raw["output_root"]),
        scenes=list(raw["scenes"]),
        download=DownloadConfig(**raw["download"]),
        prepare=PrepareConfig(**raw.get("prepare", {})),
    )
