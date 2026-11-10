"""E39: the torch association against the numpy one it replaced, on real scenes.

    uv run python experiments/scripts/e39_association_check.py <scene folder>...

Links the eval's cached SAM 3 masks with the cached Déjà View geometry. The numpy
reprojection is the oracle kept in `tests/test_segmentation.py`. Per scene: directed
pairs, largest share difference, whether the linked ids are bit-identical, and the
time of each (median of 3, warm).
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, "tests")
from test_segmentation import overlaps_reference

from src.geometry import Geometry
from src.segmentation.associate import Associator
from src.segmentation.base import Segmentation
from src.segmentation.linked import working_associator

EVAL = yaml.safe_load(Path("configs/eval_hypersim.yaml").read_text())


class NumpyAssociator(Associator):
    """The working association with the numpy reprojection."""

    def overlaps(
        self, seg: Segmentation, geometry: Geometry
    ) -> dict[tuple[int, int], float]:
        """Return the oracle's overlaps."""
        return overlaps_reference(self, seg, geometry)


def median_time(fn: object, *args: object) -> tuple[float, object]:
    """Return the median of 3 timed calls after a warm-up, and the last result."""
    fn(*args)
    times = []
    for _ in range(3):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = fn(*args)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return statistics.median(times), out


def main() -> None:
    """Print one row per scene."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scenes", type=Path, nargs="+")
    args = parser.parse_args()
    stem = EVAL["views"]
    print(f"{torch.cuda.get_device_name()}")
    print(
        "scene | pairs numpy | pairs torch | max share diff | linked ids identical | numpy (s) | torch (s)"
    )
    print("---|---|---|---|---|---|---")
    for folder in args.scenes:
        seg = Segmentation.load(folder / f"{EVAL['segmentation']}_{stem}.npz")
        geometry = Geometry.load(folder / f"{EVAL['backbone']}_{stem}.npz")
        new = working_associator()
        old = NumpyAssociator(**vars(new))
        t_old, linked_old = median_time(old.associate, seg, geometry)
        t_new, linked_new = median_time(new.associate, seg, geometry)
        o_old, o_new = old.overlaps(seg, geometry), new.overlaps(seg, geometry)
        diff = max(abs(o_old[k] - o_new.get(k, np.inf)) for k in o_old)
        same = np.array_equal(linked_old.instance_ids, linked_new.instance_ids) and (
            linked_old.classes == linked_new.classes
        )
        print(
            f"{folder.name} | {len(o_old)} | {len(o_new)} | {diff:.2e} | {same} | "
            f"{t_old:.2f} | {t_new:.2f}"
        )


if __name__ == "__main__":
    main()
