"""Hypersim settings and scene split, read from a YAML file (see configs/hypersim.yaml).

The split (issue #3) keeps scenes that choices are made on apart from scenes that
results are reported on. Every scene is one camera trajectory of a scene in
Hypersim's official v1 split (`metadata_images_split_scene_v1.csv`): dev, val and
stress come from its val partition, test from its test partition, so no scene we
report on can be one Hypersim recommends for training. dev, val and test hold only
centred pinhole cameras, because Déjà View always predicts a centred principal
point (E30); the tilt-shift scenes stay in `stress`.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

from src.loaders.hypersim.scene import camera_from_cam_from_uv
from src.loaders.scannetpp.config import PrepareConfig

OFFICIAL_PARTITION = {"dev": "val", "val": "val", "test": "test", "stress": "val"}
CENTRED_SPLITS = ("dev", "val", "test")
CENTRE_TOL = 0.5  # pixels between the principal point and the image centre


@dataclass
class HypersimConfig:
    """Paths, split name -> {scene name -> camera trajectory} and settings."""

    data_root: Path
    metadata_root: Path
    output_root: Path
    splits: dict[str, dict[str, str]]
    prepare: PrepareConfig = field(default_factory=PrepareConfig)

    def scenes(self, names: list[str]) -> dict[str, str]:
        """Scenes of the named splits, scene name -> camera trajectory."""
        unknown = set(names) - set(self.splits)
        if unknown:
            msg = f"unknown splits {sorted(unknown)}, have {list(self.splits)}"
            raise ValueError(msg)
        return {s: c for name in names for s, c in self.splits[name].items()}


def load_config(path: Path) -> HypersimConfig:
    """Read a Hypersim config file; a scene may sit in at most one split."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    splits = {
        str(name): {str(s): str(c) for s, c in (scenes or {}).items()}
        for name, scenes in raw["splits"].items()
    }
    if set(splits) != set(OFFICIAL_PARTITION):
        msg = f"splits must be {list(OFFICIAL_PARTITION)}, got {list(splits)}"
        raise ValueError(msg)
    seen = [s for scenes in splits.values() for s in scenes]
    shared = sorted({s for s in seen if seen.count(s) > 1})
    if shared:
        msg = f"scenes in more than one split: {shared}"
        raise ValueError(msg)
    return HypersimConfig(
        data_root=Path(raw["data_root"]),
        metadata_root=Path(raw["metadata_root"]),
        output_root=Path(raw["output_root"]),
        splits=splits,
        prepare=PrepareConfig(**raw.get("prepare", {})),
    )


def principal_point_offsets(metadata_root: Path) -> dict[str, float]:
    """Pixels between principal point and image centre per scene; inf = rejected."""
    offsets = {}
    path = metadata_root / "metadata_camera_parameters.csv"
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            w = int(float(row["settings_output_img_width"]))
            h = int(float(row["settings_output_img_height"]))
            m = np.array(
                [
                    [float(row[f"M_cam_from_uv_{i}{j}"]) for j in range(3)]
                    for i in range(3)
                ]
            )
            try:
                k, _ = camera_from_cam_from_uv(m, w, h)
            except ValueError:
                offsets[row["scene_name"]] = float("inf")
                continue
            offsets[row["scene_name"]] = float(
                np.hypot(k[0, 2] - w / 2, k[1, 2] - h / 2)
            )
    return offsets


def check_splits(cfg: HypersimConfig) -> None:
    """Raise unless each trajectory is public, in its partition, centred if required."""
    path = cfg.metadata_root / "metadata_images_split_scene_v1.csv"
    partition, public = {}, set()
    with path.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["included_in_public_release"] == "True":  # others have no partition
                partition[r["scene_name"]] = r["split_partition_name"]
                public.add((r["scene_name"], r["camera_name"]))
    offsets = principal_point_offsets(cfg.metadata_root)
    errors = []
    for name, scenes in cfg.splits.items():
        for scene, camera in scenes.items():
            if (scene, camera) not in public:
                errors.append(f"{name}: {scene} {camera} is not in the public release")
            if partition.get(scene) != OFFICIAL_PARTITION[name]:
                errors.append(
                    f"{name}: {scene} is in the official {partition.get(scene)} "
                    f"partition, not {OFFICIAL_PARTITION[name]}"
                )
            if name in CENTRED_SPLITS and offsets[scene] > CENTRE_TOL:
                off = offsets[scene]
                errors.append(
                    f"{name}: {scene} principal point {off:.1f} px off-centre"
                )
    if errors:
        raise ValueError("\n".join(errors))
