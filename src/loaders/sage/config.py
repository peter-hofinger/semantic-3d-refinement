"""SAGE-10k settings, read from a YAML file (see configs/sage.yaml)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class RenderConfig:
    """Views per room and how they are rendered."""

    num_frames: int
    path_frames: int
    path_anchors: int
    width: int
    height: int
    fov_x_deg: float
    samples: int
    device: str


@dataclass(frozen=True)
class SageConfig:
    """Paths, which rooms, rendering and the NYU40 class words."""

    repo: str
    revision: str
    raw_root: Path
    kit_root: Path
    output_root: Path
    rooms: int
    held_out: int
    seed: int
    render: RenderConfig
    props: tuple[str, ...]
    words: dict[str, str | None]


def load_config(path: Path) -> SageConfig:
    """Read a SAGE-10k config file."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    paths = {k: Path(raw[k]) for k in ("raw_root", "kit_root", "output_root")}
    cfg = SageConfig(
        **{k: raw[k] for k in ("repo", "revision", "rooms", "held_out", "seed")},
        **paths,
        render=RenderConfig(**raw["render"]),
        props=tuple(raw["props"]),
        words=dict(raw["words"]),
    )
    if not 0 <= cfg.held_out < cfg.rooms:
        msg = f"held_out must be in [0, rooms), got {cfg.held_out} of {cfg.rooms}"
        raise ValueError(msg)
    return cfg
