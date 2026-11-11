"""E45a figures: Déjà View's error per class on KITTI-360, over what its geometry predicts.

Reads what `e25a_class_split.py --edges plane` wrote for KITTI-360 and for Hypersim
(`e25a.yaml`, `e25a_maps.npz`) and the town scene's cached views. Panels as E25a's.

1. `e45a_class_excess.jpg`: per class (>= 3 scenes), the excess Σr / Σr̂ of each scene as a
   dot and their geometric mean as a bar, on a log axis: KITTI-360 with GT classes,
   KITTI-360 with SAM 3's eroded classes, Hypersim dev + val with GT classes under the
   same edge rule. Below: KITTI-360's permutation null of S with the measured S, and
   SAM 3's per-class mean log excess against GT's on the same pixels.
2. `e45a_town.jpg`: town scene 0010_0000000984_0000001116, views 0, 6, 12, 18: image, GT
   class, relative error r on one % scale, and r / r̂ on one log scale. A debug visual.

uv run python experiments/scripts/e45a_figures.py runs/e45a_kitti360 runs/e45a_hypersim_plane
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import yaml
from e25a_figures import FIGURES, class_panel, scene_figure, summary_panels

R_MAX = 0.3  # relative error, shared colour scale of the r maps (E25a's bedroom: 0.08)
TOWN, VIEWS = "0010_0000000984_0000001116", (0, 6, 12, 18)


def main() -> None:
    """Write both figures; see the module docstring."""
    parser = argparse.ArgumentParser()
    parser.add_argument("kitti360", type=Path)
    parser.add_argument("hypersim", type=Path)
    args = parser.parse_args()
    kitti360, hypersim = (
        yaml.safe_load((p / "e25a.yaml").read_text(encoding="utf-8"))
        for p in (args.kitti360, args.hypersim)
    )
    assert kitti360["edges"] == hypersim["edges"] == "plane", "one edge rule for both"
    k = [kitti360["scenes"][n] for n in kitti360["groups"]["all"]]
    h = [hypersim["scenes"][n] for n in hypersim["groups"]["all"]]
    fig = plt.figure(figsize=(15, 13))
    grid = fig.add_gridspec(2, 3, height_ratios=(3, 1.1), hspace=0.3, wspace=0.55)
    class_panel(
        fig.add_subplot(grid[0, 0]),
        k,
        "gt",
        f"KITTI-360 ({len(k)} scenes), GT classes",
    )
    class_panel(
        fig.add_subplot(grid[0, 1]), k, "sam3", "KITTI-360, SAM 3 classes (eroded 3 px)"
    )
    class_panel(
        fig.add_subplot(grid[0, 2]),
        h,
        "gt",
        f"Hypersim dev + val ({len(h)} scenes), GT classes",
    )
    summary_panels(
        fig.add_subplot(grid[1, 0]),
        fig.add_subplot(grid[1, 1]),
        k,
        kitti360["summary"],
        "KITTI-360",
    )
    fig.suptitle(
        "E45a: Déjà View's relative error per class over the error its pixels' geometry predicts "
        "(plane-rule edge distance x depth quartile x incidence angle); dots = scenes, bars = geometric mean, grey band = within 1.5x",
        fontsize=11,
    )
    fig.savefig(
        FIGURES / "e45a_class_excess.jpg",
        dpi=100,
        bbox_inches="tight",
        pil_kwargs={"quality": 88},
    )
    raw = yaml.safe_load(Path(kitti360["config"]).read_text(encoding="utf-8"))
    scene_figure(
        args.kitti360,
        raw,
        TOWN,
        VIEWS,
        1.75,
        R_MAX,
        f"E45a, KITTI-360 {TOWN}: Déjà View's relative error, and the same over what each pixel's "
        "edge distance, depth and incidence angle predict (red = worse than its geometry)",
        FIGURES / "e45a_town.jpg",
    )


if __name__ == "__main__":
    main()
