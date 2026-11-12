"""E45b report: per-scene tables and figures of the edge analysis on Tanks and Temples.

Reads what `e25b_boundary_oracle.py --cfg configs/eval_tanksandtemples.yaml --edges plane
--align interior --maps truck caterpillar meetingroom` wrote (`e25b.yaml`,
`e25b_maps.npz`). Three scenes carry no statistics, so every number is printed per scene.

1. Per scene: the share of the GT pixels in the edge band and in the GT's border band,
   Déjà View's error there and elsewhere, and its band pixels and band error by t.
2. Per scene and variant: mean error, error in both bands, share of Déjà View's band
   and border error removed, accuracy / completeness / Chamfer.
3. `e45b_<scene>.jpg`: views VIEWS of each scene: image, Déjà View's t in the edge
   band, and the per-pixel error of Déjà View and of each snap on one cm scale
   (`e25b_figures.scene_figure`).

uv run python experiments/scripts/e45b_figures.py runs/e45b_tanksandtemples > experiments/results/e45b_tanksandtemples_scenes.txt
"""

import argparse
from pathlib import Path

import yaml
from e25b_boundary_oracle import SOURCES, T_LABELS
from e25b_figures import FIGURES, scene_figure

VIEWS = (0, 5, 10, 15)


def main() -> None:
    """Print the per-scene tables and write one figure per scene."""
    parser = argparse.ArgumentParser()
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    data = yaml.safe_load((args.out / "e25b.yaml").read_text(encoding="utf-8"))
    raw = yaml.safe_load(Path(data["config"]).read_text(encoding="utf-8"))
    scenes = data["scenes"]
    print(f"run {args.out}, git {data['git']}, edges {data['edges']}, {data['align']}")
    print(
        "\nDéjà View: scene | edge band px share | its error share | error cm |"
        " border band px share | its error share | error cm | error cm elsewhere |"
        " error cm overall | scale"
    )
    for name, rec in scenes.items():
        dv = rec["variants"]["Déjà View"]
        print(
            f"{name} | {rec['band_px_share']:.3f} | {dv['band_error_share']:.3f} |"
            f" {dv['band_error_cm']:.1f} | {rec['border_px_share']:.3f} |"
            f" {dv['border_error_share']:.3f} | {dv['border_error_cm']:.1f} |"
            f" {dv['interior_error_cm']:.1f} | {dv['error_cm']:.1f} | {dv['scale']:.3f}"
        )
    print(
        "\nDéjà View's edge-band pixels with a GT pixel across the edge, by t: scene |"
        " share of the band | " + " | ".join(f"{lab} px / error" for lab in T_LABELS)
    )
    for name, rec in scenes.items():
        dv = rec["variants"]["Déjà View"]
        cells = [
            f"{dv['t_px'][lab]:.3f} / {dv['t_error'][lab]:.3f}" for lab in T_LABELS
        ]
        print(f"{name} | {dv['partner_share']:.3f} | " + " | ".join(cells))
    print(
        "\nsnaps: scene | variant | snapped px | error cm | edge band error cm |"
        " removed | border band error cm | removed | accuracy cm | completeness cm |"
        " Chamfer cm"
    )
    for name, rec in scenes.items():
        dv = rec["variants"]["Déjà View"]
        for variant, v in rec["variants"].items():
            removed = [
                "-" if variant == "Déjà View" else f"{1 - v[key] / dv[key]:.3f}"
                for key in ("band_error_cm", "border_error_cm")
            ]
            print(
                f"{name} | {variant} | {v['snapped_px']:.3f} | {v['error_cm']:.1f} |"
                f" {v['band_error_cm']:.1f} | {removed[0]} | {v['border_error_cm']:.1f}"
                f" | {removed[1]} | {v['accuracy_cm']:.2f} | {v['completeness_cm']:.2f}"
                f" | {v['chamfer_cm']:.2f}"
            )
    variants = (
        "Déjà View",
        *(s for s in SOURCES if s in next(iter(scenes.values()))["variants"]),
    )
    for name in scenes:
        scene_figure(
            args.out,
            raw,
            name,
            VIEWS,
            variants,
            f"E45b, Tanks and Temples {name}: where Déjà View's edge pixels sit (t, GT edge band only) and the error before / after snapping\n"
            "each segment's boundary band onto a plane fitted inside it; look at the outlines against the background and at thin parts",
            FIGURES / f"e45b_{name}.jpg",
        )


if __name__ == "__main__":
    main()
