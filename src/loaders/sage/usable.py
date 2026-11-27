"""Write SAGE's split file from the rooms Deja View's cameras can be scored on.

`scene_eval` fits a pinhole to every view's predicted ray map and stops when the median
residual reaches MAX_PINHOLE_RESIDUAL (`multiview.fit_pinhole`). SAGE's planner frames
objects from above (median pitch about 50 deg down; Hypersim 1 deg, at most 35), and on
some of these views Deja View's rays break (issue #99 pre-flight: 21-144 px on 5 of the
20 views of one room, below 5 px on all views of the other). Such a room is skipped,
as E82a skipped Hypersim's two non-pinhole rooms. The last `held_out` usable rooms of
`rooms.txt` are val, the rest train; `<output_root>/skipped.yaml` lists each skipped
room with its worst view's residual. Raise `rooms` to get more usable rooms.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import yaml

from src.geometry import Geometry
from src.loaders.sage.config import load_config
from src.loaders.sage.prepare import done, output_path
from src.refinement.multiview import MAX_PINHOLE_RESIDUAL, pinhole_residuals

log = logging.getLogger(__name__)


def assign_splits(names: list[str], usable: set[str], held_out: int) -> dict:
    """Train and val among the usable rooms, in list order; val: the last `held_out`."""
    kept = [n for n in names if n in usable]
    cut = max(len(kept) - held_out, 0)
    return {"train": kept[:cut], "val": kept[cut:]}


def main() -> None:
    """Check every prepared room's backbone cache and write splits.yaml."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/sage.yaml"))
    parser.add_argument("--backbone", default="dejaview_k16")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    cfg = load_config(args.config)
    names = (cfg.raw_root / "rooms.txt").read_text().split()
    usable, skipped, missing = set(), {}, []
    for name in names:
        views = output_path(cfg, name)
        cache = views.with_name(f"{args.backbone}_{views.name}")
        if not (done(cfg, name) and cache.is_file()):
            missing.append(name)
            continue
        worst = float(pinhole_residuals(Geometry.load(cache)).max())
        if worst < MAX_PINHOLE_RESIDUAL:
            usable.add(name)
        else:
            skipped[name] = round(worst, 1)
    splits = assign_splits(names, usable, cfg.held_out)
    (cfg.output_root / "splits.yaml").write_text(yaml.safe_dump(splits))
    (cfg.output_root / "skipped.yaml").write_text(yaml.safe_dump(skipped))
    log.info(
        "%d usable (train %d, val %d), %d skipped (rays no pinhole), %d not prepared",
        len(usable),
        len(splits["train"]),
        len(splits["val"]),
        len(skipped),
        len(missing),
    )


if __name__ == "__main__":
    main()
