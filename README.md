# semantic-3d-refinement
Semantic-guided geometry refinement and post-optimization for feed-forward 3D reconstruction models (VGGT, DejaView, DA3)

## Setup

```bash
uv sync
cp .env.example .env   # then add your ScanNet++ token
```

## Data (ScanNet++)

Scenes, assets and preprocessing settings live in `configs/scannetpp.yaml`.

```bash
uv run scannetpp-download   # fetch scenes into data/scannetpp (official layout)
uv run scannetpp-prepare    # subsample + resize -> data/processed/scannetpp/<scene>/views_*.npz
```

Each `.npz` holds `images` (N, H, W, 3), `intrinsics` (N, 3, 3) and `cam_to_world` (N, 4, 4,
OpenCV camera, same world frame as the laser-scan mesh). Load it with
`src.loaders.Views.load(path)`.

## Pipeline

```
Views --backbone--> Geometry --segmenter--> Segmentation --refiner--> Geometry --> PointCloud
```

| Type | Contents |
|---|---|
| `Views` | images, intrinsics, cam_to_world |
| `Geometry` | depth (N, H, W), confidence, intrinsics, cam_to_world |
| `Segmentation` | instance id map (N, H, W), same id = same object in every view, -1 = none; class per id |
| `PointCloud` | points, colors, instance id per point, class per id |

Each stage is a small interface (`Backbone.predict`, `Segmenter.segment`,
`Refiner.refine`) with a placeholder implementation, so the pipeline runs end to
end before the real models exist:

```bash
uv run run-pipeline data/processed/scannetpp/7b6477cb95/views_n20_uniform_504.npz
```

## Layout

```
configs/            YAML configs
data/               downloaded and preprocessed files (git-ignored)
src/                the code, imported as `src`
  loaders/          dataset loaders and the shared Views format
    scannetpp/      download, scene loading, preprocessing
  geometry.py       Geometry (depth + cameras), lifting to 3D
  backbone/         Backbone interface; Déjà View goes here
  segmentation/     Segmentation, Segmenter interface; SAM 3 goes here
  refinement/       Refiner interface; the optimizer goes here
  output.py         PointCloud, PLY export
  pipeline.py       chains the stages
  evaluation/       metrics against ground truth
tests/
```
