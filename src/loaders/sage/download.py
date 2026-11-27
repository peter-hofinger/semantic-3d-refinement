"""Fetch the dataset kit and `rooms` SAGE-10k rooms at the pinned revision.

Rooms are spread evenly over SAGE's sorted list (offset by the seed), so they mix the
generation batches; the chosen names are written to `<raw_root>/rooms.txt`, which
`sage-prepare` reads. A room already unzipped is skipped, so a run can be resumed.
"""

from __future__ import annotations

import argparse
import logging
import zipfile
from pathlib import Path

import numpy as np
from huggingface_hub import HfApi, hf_hub_download, snapshot_download

from src.loaders.sage.config import SageConfig, load_config

log = logging.getLogger(__name__)


def select_rooms(names: list[str], rooms: int, seed: int) -> list[str]:
    """`rooms` names evenly spaced over the sorted list, shifted by the seed."""
    names = sorted(names)
    if rooms > len(names):
        msg = f"asked for {rooms} rooms, SAGE has {len(names)}"
        raise ValueError(msg)
    picks = (np.arange(rooms) * len(names) // rooms + seed) % len(names)
    return [names[i] for i in picks]


def fetch_room(cfg: SageConfig, name: str) -> Path:
    """Download and unzip one room into `<raw_root>/<name>`; the zip is removed."""
    folder = cfg.raw_root / name
    if any(folder.glob("layout_*.json")):
        return folder
    path = Path(
        hf_hub_download(
            cfg.repo,
            f"scenes/{name}.zip",
            repo_type="dataset",
            revision=cfg.revision,
            local_dir=cfg.raw_root / "_zips",
        )
    )
    with zipfile.ZipFile(path) as z:
        z.extractall(folder)
    path.unlink()
    return folder


def main() -> None:
    """Download the kit and the chosen rooms; a room that fails is logged."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/sage.yaml"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    cfg = load_config(args.config)
    snapshot_download(
        cfg.repo,
        repo_type="dataset",
        revision=cfg.revision,
        allow_patterns=["kits/*"],
        local_dir=cfg.kit_root,
    )
    tree = HfApi().list_repo_tree(
        cfg.repo, path_in_repo="scenes", repo_type="dataset", revision=cfg.revision
    )
    names = [Path(t.path).stem for t in tree if t.path.endswith(".zip")]
    chosen = select_rooms(names, cfg.rooms, cfg.seed)
    cfg.raw_root.mkdir(parents=True, exist_ok=True)
    (cfg.raw_root / "rooms.txt").write_text("\n".join(chosen) + "\n")
    for i, name in enumerate(chosen):
        try:
            fetch_room(cfg, name)
        except (OSError, zipfile.BadZipFile) as err:
            log.error("%s: not downloaded: %s", name, err)  # noqa: TRY400
            continue
        log.info("%d/%d %s", i + 1, len(chosen), name)


if __name__ == "__main__":
    main()
