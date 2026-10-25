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
`semantic_3d_refinement.data.Views.load(path)`.

## Layout

```
configs/                     YAML configs
src/semantic_3d_refinement/
  data/                      datasets and the shared Views format
    scannetpp/               download, scene loading, preprocessing
  backbone/                  Déjà View inference
  segmentation/              masks and cross-view linking
  refinement/                semantic-guided refinement
  evaluation/                metrics against ground truth
tests/
```
