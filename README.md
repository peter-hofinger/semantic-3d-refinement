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

## Data (Hypersim)

Scenes (scene -> camera trajectory) and settings live in `configs/hypersim.yaml`; the
three default scenes are from the official v1 validation split.

```bash
uv run hypersim-download   # metadata CSVs + only the needed files (~45 MB per trajectory)
uv run hypersim-prepare    # -> data/processed/hypersim/<scene>_<cam>/views_*.npz + gt_views_*.npz
```

The download reads Apple's per-scene ZIPs with HTTP range requests, so it fetches only
the tone-mapped images, depth, NYU40 class, instance and render-entity maps of the
trajectory's public frames. `prepare` writes `Views` (same format as ScanNet++, OpenCV
cameras in meters) and `GroundTruth` (instance ids, z-depth, classes) straight from
Hypersim's per-pixel maps, so `eval-instances` works unchanged. Notes:
- Most Hypersim cameras are tilt-shift: `M_cam_from_uv` is split into a pinhole K
  (off-centre principal point) and a camera rotation; the few with skew are rejected.
- Walls, floors and ceilings have no instance ids in Hypersim; each such class becomes
  one instance (`STUFF_BASE + NYU40 id`). Pixels without a label stay unlabeled.
- Class names are NYU40 (many small objects are `otherprop`).

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

## Déjà View

Runs the released `nvidia/dvlt` checkpoint (16 loop steps, weights download on
first use) and saves depth, confidence, per-pixel rays and cameras next to the
views as `dejaview_k16_<views>.npz` (load with `src.geometry.Geometry.load`):

```bash
uv run dejaview-predict data/processed/scannetpp/*/views_n20_uniform_504.npz
```

## Segmentation (SAM 3 + association)

`--segmenter sam3` (`src/segmentation/linked.py`) runs SAM 3 image mode on every
view (text prompts, default ScanNet++'s top-100 classes from
`data/scannetpp/metadata/semantic_benchmark/top100.txt`, score threshold 0.3) and
links the per-view masks across views by reprojecting them with the backbone's depth
and cameras (`src/segmentation/associate.py`). Settings and how they were chosen:
`experiments/LOG.md` (E9-E17). Needs access to the gated `facebook/sam3` weights
(`uv run hf auth login`); ~1-2 min per 20-view scene on an RTX 5090, ~2 GB GPU memory.

```bash
uv run run-pipeline data/processed/scannetpp/7b6477cb95/views_n20_uniform_504.npz     --geometry data/processed/scannetpp/7b6477cb95/dejaview_k16_views_n20_uniform_504.npz     --segmenter sam3 --out outputs/7b6477cb95/cloud.ply
```

`--geometry` reuses a saved `dejaview-predict` output (`--backbone dejaview` runs
Déjà View instead). Next to the cloud it saves `cloud_segmentation.npz`, which loads
with `src.segmentation.Segmentation.load(path)`:

- `instance_ids` (N, H, W) int32 at the views' resolution, -1 = no mask; the same id
  is the same object in every view
- `classes`: dict id -> class name (the prompt that found it)

A refiner gets the same object as `Refiner.refine(views, geometry, segmentation)`.

## Refinement (joint, all views)

`--refiner joint` (`src/refinement/multiview.py`) refines the inverse depth of all views
at once, per linked SAM 3 instance s of view i as d = a_{i,s} d⁰ + b_{i,s} + δ, cameras
fixed, by Gauss-Newton with matrix-free conjugate gradients:

E = E_d + E_mv + E_bd + E_n + ‖δ‖², every weight 1 for now (`configs/refine.yaml`).

- **E_d**: Huber anchor to Déjà View's depth, weighted by its confidence times the share
  of other views that agree on the pixel's instance, so it lets go where views disagree.
- **E_mv**: overlapping views must agree on depth, only between pixels of one instance.
- **E_bd**: thin-plate smoothing of inverse depth, only within one instance.
- **E_n**: BiNI's normal term on DSINE normals (`src/refinement/normals.py`), cut on the
  input's occlusion contours. DSINE is non-commercial and stays outside git
  (`data/third_party/dsine_hub`); `uv run dsine-predict <views npz>` caches its normals.

```bash
D=data/processed/scannetpp/5ee7c22ba0
uv run run-pipeline $D/views_n20_uniform_504.npz --geometry $D/dejaview_k16_views_n20_uniform_504.npz \
    --segmenter sam3 --refiner joint --out outputs/5ee7c22ba0/cloud.ply
```

`cloud.ply` holds per point xyz, rgb, `instance` (same id = same object in every view,
-1 = none) and `semantic` (index into the class names listed as `comment class <i>
<name>` in the header). Next to it: `cloud_backbone.npz` (Déjà View), `cloud_geometry.npz`
(refined) and `cloud_segmentation.npz`.

## Evaluation (Hypersim)

One protocol for every variant (`src/analysis/scene_eval.py`, `configs/eval_hypersim.yaml`):
- **Scenes:** Hypersim validation scenes, 20 views each. The test split is never used, and no
  weight is tuned: every energy term that is on has weight 1.
- **Inputs:** cached Déjà View geometry and linked SAM 3 masks, with one prompt per NYU40
  class.
- **Black box:** only the output, one labeled cloud in the first view's frame, is scored.
  It is registered onto the ground truth from the two clouds alone: a start from the first
  camera and the ratio of median distances to it, then ICP that also fits the scale
  (as π³ and VGGT are evaluated). Each variant is registered on its own.
- **Scores** (`src/evaluation/labeled.py`), against Hypersim's per-pixel ground truth on a
  1 cm grid:
  - accuracy, completeness and F-score at 2 cm (Tanks and Temples);
  - mIoU and instance AP50 (ScanNet's protocol), on labels transferred to the ground-truth
    points.

```bash
uv run hypersim-download && uv run hypersim-prepare
uv run python -m src.analysis.scene_eval --cfg configs/eval_hypersim.yaml   # inputs: dejaview-predict, sam3-segment
uv run python -m src.analysis.scene_figures --run runs/eval_hypersim-<time>
```

Kitchen, conference room and restaurant, 20 views each, every energy term at weight 1
(`figures/eval_hypersim-20261103-111445`; 2.4 min for all 8 variants):

| F-score at 2 cm (fused cloud) | kitchen | conference | restaurant |
|---|---|---|---|
| Déjà View | 0.633 | 0.381 | 0.519 |
| + agreement (E_d + E_mv) | 0.621 | 0.382 | 0.486 |
| + cut E_bd, a, b, same-segment E_mv, SAM 3 masks (`configs/refine_semantic.yaml`) | 0.644 | 0.377 | 0.517 |
| same, shuffled masks (control) | 0.600 | 0.368 | 0.516 |
| same, ground-truth masks | 0.641 | 0.375 | 0.505 |

- **Agreement:** the views become consistent (disagreement 0.5–1 % → 0.05 %); precision rises
  (+0.03 to +0.06), recall falls (−0.14 to −0.16), and the F-score does not improve.
- **Smoothing:** cut at SAM 3's mask boundaries, it no longer melts objects together (kitchen
  F-score 0.518 uncut, 0.629 cut).
- **Masks:** shuffled masks score like real ones on two of three scenes, and ground-truth
  masks no better than SAM 3's. Reruns move F-scores by up to 0.006 and, on the conference
  room, label scores by up to 0.1 (see the report).

## Layout

```
configs/            YAML configs
data/               downloaded and preprocessed files (git-ignored)
src/                the code, imported as `src`
  loaders/          dataset loaders and the shared Views format
    scannetpp/      download, scene loading, preprocessing
    hypersim/       download, scene loading, preprocessing + ground truth
  geometry.py       Geometry (depth + cameras), lifting to 3D
  backbone/         Backbone interface; Déjà View goes here
  segmentation/     Segmentation, Segmenter interface, SAM 3 + association
  refinement/       Refiner interface; joint refinement by Gauss-Newton (multiview.py),
                    the normal term on DSINE normals (normals.py)
  output.py         PointCloud, PLY export
  pipeline.py       chains the stages
  evaluation/       metrics against ground truth: instances (instances.py), labeled
                    cloud (labeled.py: F-score, mIoU, instance AP50)
  analysis/         the Hypersim protocol (scene_eval.py) and its figures
experiments/        SAM 3 study (LOG.md, scripts, results, figures)
figures/            refinement figures, one folder per run with a README
tests/
```
