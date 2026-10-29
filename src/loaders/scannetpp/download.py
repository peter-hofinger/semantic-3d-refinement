"""Download the ScanNet++ scenes and assets listed in a config.

The token is read from SCANNETPP_TOKEN (put it in .env, see .env.example).
"""

from __future__ import annotations

import argparse
import logging
import os
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING

from dotenv import load_dotenv

from src.loaders.scannetpp.assets import (
    ASSETS,
    METADATA_FILES,
    local_path,
    remote_path,
)
from src.loaders.scannetpp.config import load_config

if TYPE_CHECKING:
    from src.loaders.scannetpp.config import ScannetppConfig

log = logging.getLogger(__name__)

RETRY_STATUS = {429, 500, 502, 503, 504}
HTTP_UNAUTHORIZED = 401


def read_token() -> str:
    """Read the download token from the environment or .env."""
    load_dotenv()
    token = os.environ.get("SCANNETPP_TOKEN", "")
    if not token:
        msg = (
            "SCANNETPP_TOKEN is not set. Copy .env.example to .env and add your token."
        )
        raise SystemExit(msg)
    return token


def fetch(url: str, target: Path, retries: int = 5) -> None:
    """Download `url` to `target`, retrying on rate limits and server errors."""
    target.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(retries):
        try:
            urllib.request.urlretrieve(url, target)  # noqa: S310 https URL from config
        except urllib.error.HTTPError as err:
            if err.code not in RETRY_STATUS or attempt == retries - 1:
                raise
            time.sleep(0.5 * 2**attempt)
        else:
            return


class Downloader:
    """Fetches files from the ScanNet++ server into the data root."""

    def __init__(self, cfg: ScannetppConfig, token: str) -> None:
        """Store the config and token."""
        self.cfg = cfg
        self.token = token

    def url(self, remote: str) -> str:
        """Build the download URL for a server path."""
        return self.cfg.download.root_url.replace("TOKEN", self.token).replace(
            "FILEPATH", remote
        )

    def metadata(self) -> None:
        """Download split lists and class lists."""
        for rel in METADATA_FILES:
            target = self.cfg.data_root / rel
            if not target.exists():
                fetch(self.url(rel), target)
        log.info("metadata: done")

    def asset(self, scene_id: str, asset: str) -> None:
        """Download one asset of one scene, unzipping it if needed."""
        target = self.cfg.data_root / local_path(scene_id, asset)
        if target.exists():
            log.info("%s %s: already there", scene_id, asset)
            return
        url = self.url(remote_path(scene_id, asset))
        if not ASSETS[asset][1]:
            fetch(url, target)
        else:
            archive = target.with_suffix(".zip")
            fetch(url, archive)
            with zipfile.ZipFile(archive) as zf:
                zf.extractall(archive.parent)
            archive.unlink()
        log.info("%s %s: done", scene_id, asset)


def main() -> None:
    """Download everything listed in the config."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/scannetpp.yaml"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    cfg = load_config(args.config)
    unknown = [a for a in cfg.download.assets if a not in ASSETS]
    if unknown:
        msg = f"unknown assets in config: {unknown}"
        raise SystemExit(msg)

    downloader = Downloader(cfg, read_token())
    try:
        if cfg.download.metadata:
            downloader.metadata()
        for scene_id in cfg.scenes:
            for asset in cfg.download.assets:
                downloader.asset(scene_id, asset)
    except urllib.error.HTTPError as err:
        if err.code == HTTP_UNAUTHORIZED:
            msg = "Download refused (401): the token is invalid or expired."
            raise SystemExit(msg) from None
        raise


if __name__ == "__main__":
    main()
