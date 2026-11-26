"""Render SAGE-10k rooms into backbone input and exact ground truth.

Writes, per room, `views_n<N>_uniform_<W>.npz` (Views), `gt_<views stem>.npz`
(GroundTruth, NYU40 classes) and `labels_<views stem>.json` (SAGE's type and
description per instance) into <output_root>/<room>/. The views are `num_frames`
poses evenly spaced along the kit's planned path of `path_frames` poses, seeded per
room. A room
already written is skipped, so a run can be resumed; `--shard i/n` splits the rooms
over n processes (one GPU each). `sage-splits` writes the split file once the
backbone has run.
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import os
import time
import zlib
from pathlib import Path

import numpy as np

from src.loaders.frames import select_frames
from src.loaders.ground_truth import NO_MESH, STUFF_BASE, GroundTruth
from src.loaders.sage.config import SageConfig, load_config
from src.loaders.sage.scene import (
    Part,
    assemble,
    camera_path,
    intrinsics,
    kit,
    label_parts,
    ray_cast,
)
from src.loaders.views import Views

log = logging.getLogger(__name__)


def output_path(cfg: SageConfig, name: str) -> Path:
    """Where the views of a room are written; the name encodes the settings."""
    r = cfg.render
    return cfg.output_root / name / f"views_n{r.num_frames}_uniform_{r.width}.npz"


def gt_path(cfg: SageConfig, name: str) -> Path:
    """Return where a room's ground truth goes; written last, it marks the room done."""
    path = output_path(cfg, name)
    return path.with_name(f"gt_{path.stem}.npz")


def done(cfg: SageConfig, name: str) -> bool:
    """Whether a room is already prepared."""
    return gt_path(cfg, name).is_file()


def prepare_room(cfg: SageConfig, folder: Path) -> tuple[Views, GroundTruth, dict, str]:
    """Render one room: views, ground truth, SAGE labels and device, stage times."""
    from src.loaders.sage.render import render_views  # noqa: PLC0415  needs bpy

    clock = [("start", time.perf_counter())]
    (layout_path,) = folder.glob("layout_*.json")
    layout = json.loads(layout_path.read_text())
    if len(layout["rooms"]) != 1:
        msg = f"{len(layout['rooms'])} rooms in one layout"
        raise ValueError(msg)
    plan, meshes = assemble(layout, folder, cfg.kit_root)
    labels = label_parts(layout, list(meshes), cfg.words, cfg.props)
    parts = [
        Part(
            key,
            labels.instance[key],
            np.asarray(m["mesh"].vertices),
            np.asarray(m["mesh"].faces),
        )
        for key, m in meshes.items()
    ]
    clock.append(("assemble", time.perf_counter()))
    r = cfg.render
    seed = zlib.crc32(folder.name.encode()) ^ cfg.seed
    path = np.stack(camera_path(plan, meshes, r, seed, cfg.kit_root))
    picked = select_frames(path[:, :3, 3], r.num_frames, "uniform")
    cam_to_world = path[picked]
    clock.append(("plan", time.perf_counter()))
    k = intrinsics(r.width, r.height, r.fov_x_deg)
    ids, depth = ray_cast(parts, k, cam_to_world, r.height, r.width)
    clock.append(("ray cast", time.perf_counter()))
    room = layout["rooms"][0]["dimensions"]
    images, device = render_views(meshes, room, cam_to_world, r, kit(cfg.kit_root)[1])
    clock.append(("render", time.perf_counter()))
    stages = ", ".join(
        f"{k} {t - t0:.0f} s" for (_, t0), (k, t) in itertools.pairwise(clock)
    )
    views = Views(
        scene_id=folder.name,
        names=[f"path.{i:04d}" for i in picked],
        images=images,
        intrinsics=np.tile(k, (len(picked), 1, 1)).astype(np.float32),
        cam_to_world=cam_to_world.astype(np.float32),
    )
    seen = set(np.unique(ids[ids >= 0]).tolist())
    gt = GroundTruth(ids, depth, {i: c for i, c in labels.nyu40.items() if i in seen})
    sage = {i: s for i, s in labels.sage.items() if i in seen}
    return views, gt, sage, f"{device} ({stages})"


def main() -> None:
    """Prepare every downloaded room of `rooms.txt`; a room that fails is logged."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/sage.yaml"))
    parser.add_argument("--limit", type=int, help="only the first rooms (smoke runs)")
    parser.add_argument("--shard", default="0/1", help="i/n: every n-th room from i")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    cfg = load_config(args.config)
    names = (cfg.raw_root / "rooms.txt").read_text().split()
    i, n = map(int, args.shard.split("/"))
    for name in names[: args.limit][i::n]:
        if done(cfg, name):
            continue
        start = time.perf_counter()
        try:
            views, gt, sage, device = prepare_room(cfg, cfg.raw_root / name)
        except (ValueError, OSError) as err:
            log.error("%s: not prepared: %s", name, err)  # noqa: TRY400
            continue
        path = output_path(cfg, name)
        views.save(path)
        path.with_name(f"labels_{path.stem}.json").write_text(
            json.dumps(sage, indent=1)
        )
        gt.save(gt_path(cfg, name))  # last: its presence marks the room done
        ids = gt.instance_ids
        log.info(
            "%s: %.0f s on %s; labeled %.0f%% (stuff %.0f%%), no geometry %.0f%%,"
            " %d instances",
            name,
            time.perf_counter() - start,
            device,
            100 * (ids >= 0).mean(),
            100 * (ids >= STUFF_BASE).mean(),
            100 * (ids == NO_MESH).mean(),
            len(np.unique(ids[(ids >= 0) & (ids < STUFF_BASE)])),
        )
    logging.shutdown()
    # bpy's teardown can spin for minutes after the last room (3080: 9+ min at 70% CPU
    # after two rooms, not reproduced by a bare render); every file is closed by now
    os._exit(0)


if __name__ == "__main__":
    main()
