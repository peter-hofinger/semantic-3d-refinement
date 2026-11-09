"""Download the three archives per ETH3D scene that the loader reads.

Per scene: the undistorted DSLR images with their PINHOLE calibration, the distorted
JPEGs (only for the THIN_PRISM_FISHEYE calibration the depth maps belong to) and the
ground-truth depth maps; ~0.4-3 GB per scene, ~16 GB for all 13. The archives stay
packed: `eth3d-prepare` reads the files it needs straight out of them, because the
unpacked depth maps alone would take ~44 GB (97.5 MB per image).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import requests

from src.loaders.eth3d.scene import ARCHIVE_KINDS, Eth3dConfig, load_config

log = logging.getLogger(__name__)

CHUNK = 1 << 22  # bytes per write


def download(url: str, path: Path, session: requests.Session) -> None:
    """Fetch url to path unless a file of the server's size is there already."""
    head = session.head(url, timeout=60)
    head.raise_for_status()
    size = int(head.headers["Content-Length"])
    if path.is_file() and path.stat().st_size == size:
        log.info("%s: present (%.2f GB)", path.name, size / 1e9)
        return
    part = path.with_suffix(".part")
    with session.get(url, stream=True, timeout=600) as r:
        r.raise_for_status()
        with part.open("wb") as f:
            for chunk in r.iter_content(CHUNK):
                f.write(chunk)
    if part.stat().st_size != size:
        msg = f"{url}: got {part.stat().st_size} bytes, expected {size}"
        raise OSError(msg)
    part.replace(path)
    log.info("%s: downloaded (%.2f GB)", path.name, size / 1e9)


def download_scenes(cfg: Eth3dConfig, scenes: list[str]) -> None:
    """Fetch every archive kind of the given scenes into the archive root."""
    cfg.archive_root.mkdir(parents=True, exist_ok=True)
    with requests.Session() as session:
        for scene in scenes:
            for kind in ARCHIVE_KINDS:
                path = cfg.archive(scene, kind)
                download(f"{cfg.base_url}/{path.name}", path, session)


def main() -> None:
    """Download the archives of the chosen scenes (default: all of the config)."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/eth3d.yaml"))
    parser.add_argument("--scenes", nargs="+", help="default: every scene")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    cfg = load_config(args.config)
    scenes = args.scenes or list(cfg.scenes)
    unknown = set(scenes) - set(cfg.scenes)
    if unknown:
        msg = f"not in the config: {sorted(unknown)}"
        raise ValueError(msg)
    download_scenes(cfg, scenes)


if __name__ == "__main__":
    main()
