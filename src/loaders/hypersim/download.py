"""Download only the files the loader needs from Hypersim's per-scene ZIP archives.

Each scene is one ZIP of 1-20 GB on Apple's server. The archive is read over HTTP
range requests (idea from ml-hypersim contrib/99991 by Thomas Germer, MIT), so only
the central directory and the selected files are transferred: about 1.4 MB per frame.
"""

from __future__ import annotations

import argparse
import io
import logging
import zipfile
from pathlib import Path

import requests

from src.loaders.hypersim.config import check_splits, load_config
from src.loaders.hypersim.scene import HypersimScene

log = logging.getLogger(__name__)

BASE_URL = (
    "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes"
)
GEOMETRY_KINDS = ("depth_meters", "semantic", "semantic_instance", "render_entity_id")
MIN_READ = 1 << 20  # bytes per range request at least, fewer round trips
METADATA_URL = "https://raw.githubusercontent.com/apple/ml-hypersim/main"
METADATA_FILES = (
    "evermotion_dataset/analysis/metadata_images_split_scene_v1.csv",
    "evermotion_dataset/analysis/metadata_camera_trajectories.csv",
    "contrib/mikeroberts3000/metadata_camera_parameters.csv",
)


class RemoteFile(io.RawIOBase):
    """Seekable read-only file backed by HTTP range requests."""

    def __init__(self, url: str, session: requests.Session) -> None:
        """Ask the server for the file size."""
        self.url = url
        self.session = session
        head = session.head(url, allow_redirects=True, timeout=60)
        head.raise_for_status()
        self.size = int(head.headers["content-length"])
        self.offset = 0

    def seekable(self) -> bool:
        """Support seek (zipfile needs it)."""
        return True

    def readable(self) -> bool:
        """Support read."""
        return True

    def tell(self) -> int:
        """Return the current offset."""
        return self.offset

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        """Move the read offset."""
        base = {io.SEEK_SET: 0, io.SEEK_CUR: self.offset, io.SEEK_END: self.size}
        self.offset = min(max(0, base[whence] + offset), self.size)
        return self.offset

    def readinto(self, buffer: memoryview) -> int:  # type: ignore[override]
        """Fill `buffer` from the current offset with one range request."""
        n = min(len(buffer), self.size - self.offset)
        if n == 0:
            return 0
        end = self.offset + n - 1
        r = self.session.get(
            self.url, headers={"Range": f"bytes={self.offset}-{end}"}, timeout=120
        )
        r.raise_for_status()
        data = r.content[:n]
        buffer[: len(data)] = data
        self.offset += len(data)
        return len(data)


def wanted_files(scene: HypersimScene) -> set[str]:
    """Archive paths needed for one trajectory: metadata, cameras, all public frames."""
    s, cam = scene.scene, scene.camera
    paths = {f"{s}/_detail/metadata_scene.csv"}
    paths |= {
        f"{s}/_detail/{cam}/camera_keyframe_{k}.hdf5"
        for k in ("frame_indices", "orientations", "positions")
    }
    for i in scene.public_frame_ids():
        paths.add(f"{s}/images/scene_{cam}_final_preview/frame.{i:04d}.tonemap.jpg")
        paths |= {
            f"{s}/images/scene_{cam}_geometry_hdf5/frame.{i:04d}.{k}.hdf5"
            for k in GEOMETRY_KINDS
        }
    return paths


def download_scene(
    scene: HypersimScene, data_root: Path, session: requests.Session
) -> None:
    """Fetch the wanted files of one trajectory into data_root, skip existing ones."""
    wanted = wanted_files(scene)
    missing = {p for p in wanted if not (data_root / p).is_file()}
    log.info(
        "%s: %d files wanted, %d missing", scene.scene_id, len(wanted), len(missing)
    )
    if not missing:
        return
    remote = RemoteFile(f"{BASE_URL}/{scene.scene}.zip", session)
    with zipfile.ZipFile(io.BufferedReader(remote, buffer_size=MIN_READ)) as archive:
        names = set(archive.namelist())
        absent = missing - names
        if absent:
            msg = f"{scene.scene_id}: not in the archive: {sorted(absent)[:5]}"
            raise FileNotFoundError(msg)
        for n, path in enumerate(sorted(missing), 1):
            archive.extract(path, data_root)
            if n % 50 == 0 or n == len(missing):
                log.info("  %d / %d", n, len(missing))


def download_metadata(metadata_root: Path, session: requests.Session) -> None:
    """Fetch the split and camera-parameter CSVs from the ml-hypersim repository."""
    metadata_root.mkdir(parents=True, exist_ok=True)
    for path in METADATA_FILES:
        out = metadata_root / Path(path).name
        if out.is_file():
            continue
        r = session.get(f"{METADATA_URL}/{path}", timeout=120)
        r.raise_for_status()
        out.write_bytes(r.content)
        log.info("metadata: %s", out)


def main() -> None:
    """Download the metadata and the trajectories of the chosen splits."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/hypersim.yaml"))
    parser.add_argument(
        "--splits", nargs="+", default=["dev"], help="dev val test stress"
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    cfg = load_config(args.config)
    with requests.Session() as session:
        download_metadata(cfg.metadata_root, session)
        check_splits(cfg)
        for scene, camera in cfg.scenes(args.splits).items():
            hs = HypersimScene(cfg.data_root, cfg.metadata_root, scene, camera)
            download_scene(hs, cfg.data_root, session)


if __name__ == "__main__":
    main()
