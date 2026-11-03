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

Scenes and settings live in `configs/hypersim.yaml`, split into four sets of one camera
trajectory per scene, none filtered by hand: **dev** (3 val scenes; choices are made
here), **val** (the other 11 val scenes with a centred camera; checks that a choice holds),
**test** (all 26 test scenes with a centred camera; reported numbers only) and **stress**
(3 tilt-shift val scenes, see below). The loaders check each set against Hypersim's
official v1 split, so no scene used for choices or reports is from its training split.

```bash
uv run hypersim-download --splits dev   # metadata CSVs + only the needed files (~45 MB per trajectory)
uv run hypersim-prepare --splits dev    # -> data/processed/hypersim/<scene>_<cam>/views_*.npz + gt_views_*.npz
```

The download reads Apple's per-scene ZIPs with HTTP range requests, so it fetches only
the tone-mapped images, depth, NYU40 class, instance and render-entity maps of the
trajectory's public frames. `prepare` writes `Views` (same format as ScanNet++, OpenCV
cameras in meters) and `GroundTruth` (instance ids, z-depth, classes) straight from
Hypersim's per-pixel maps, so `eval-instances` works unchanged. Notes:
- Most Hypersim cameras are tilt-shift: `M_cam_from_uv` is split into a pinhole K
  (off-centre principal point) and a camera rotation; the few with skew are rejected.
- Déjà View always predicts a centred principal point, so use plain-camera scenes with it
  (14 of the 46 validation scenes). On tilt-shift scenes its point maps degrade with the
  offset: Rel. L2 0.010-0.026 on the three plain scenes vs 0.031-0.116 on the three
  tilt-shift ones (E30 in `experiments/LOG.md`).
- Walls, floors and ceilings have no instance ids in Hypersim; each such class becomes
  one instance (`STUFF_BASE + NYU40 id`). Pixels without a label stay unlabeled.
- Class names are NYU40 (many small objects are `otherprop`).
- Ground-truth depth of a mirror is the mirror surface; Déjà View partly predicts the
  reflected room there.
- 0.69% of Déjà View's training data is Hypersim and its scene split is not published,
  so results on Hypersim may be slightly optimistic.

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
at once, per pixel as d = d⁰ + δ, cameras fixed, by Gauss-Newton with matrix-free
conjugate gradients:

E = E_d + E_mv + λ_n E_n + ‖δ‖², every weight 1 for now and λ_n = 1e-4 for E_n's units
(`configs/refine.yaml`).

- **E_d**: Huber anchor to Déjà View's depth, weighted by its confidence times the share
  of other views that agree on the pixel's instance, so it lets go where views disagree.
- **E_mv**: overlapping views must agree on depth, only between pixels of one instance and
  not where one view's surface hides the other's: a gap counts as occluded above Otsu's
  threshold on the scene's gaps (about 4 % on Hypersim), so there is no tolerance to set.
- **E_n**: BiNI's normal term on DSINE normals (`src/refinement/normals.py`), the only
  shape term: only between pixels of one instance and cut on the input's occlusion
  contours. DSINE is non-commercial and stays outside git
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
- **Scenes:** the `dev` split of `configs/hypersim.yaml`, three validation scenes with
  centred cameras, 20 views each. The test split is never used, and no weight is tuned: every
  energy term has weight 1.
- **Inputs:** cached Déjà View geometry, DSINE normals and linked SAM 3 masks, with one prompt
  per NYU40 class.
- **Output:** a labeled point per input pixel, in the first view's frame. Output pixel i and
  ground-truth pixel i see the same surface, so each variant is aligned onto the ground
  truth by one least-squares similarity on these pixels (Umeyama, as VGGT and π³), and
  every score is taken on them. No score has a threshold, cap or grid of its own. Any
  method returning a point and a label per input pixel can be scored this way; one with
  another representation (mesh, Gaussians) renders depth and labels into its own
  predicted cameras first, as point-map methods (VGGT, π³, PanSt3R) are scored.
- **Scores** (`src/evaluation/labeled.py`), against Hypersim's per-pixel ground truth (its
  meshes are not public):
  - accuracy and completeness (mean and median, uncapped), the 7-Scenes / NRGBD protocol
    of CUT3R, π³ and VGGT, for the scene and per class;
  - mIoU, scene-level PQ / SQ / RQ (Panoptic Lifting, PanSt3R) and AP / AP50 / AP25
    (3D instance AP as SegVGGT, IGGT and PanSt3R report it), on the output's label
    of each ground-truth pixel. An object is one segment over all views, so an object
    split across views counts as two.
  - Walls, floor and ceiling are background: one segment per class. mIoU and PQ are
    averaged over all classes, PQ also over objects (PQ_th) and background (PQ_st), as
    PanSt3R; AP is over object classes, as the instance benchmarks. The surface scores
    cover every pixel, as in the geometry protocols above.

```bash
uv run hypersim-download --splits dev && uv run hypersim-prepare --splits dev
uv run python -m src.analysis.scene_eval --cfg configs/eval_hypersim.yaml   # inputs: dejaview-predict, dsine-predict, sam3-segment
uv run python -m src.analysis.scene_figures --run runs/eval_hypersim-<time>
```

Living room, bedroom and hotel lobby, 20 views each, every energy weight 1, none tuned
(`figures/eval_hypersim-20261103-180820`; 1.5 min):

| cm, ↓ | living room | bedroom | lobby |
|---|---|---|---|
| accuracy, Déjà View → Refined | 2.19 → 1.79 | 2.52 → 2.20 | 9.31 → 8.63 |
| completeness | 1.01 → 1.41 | 1.17 → 1.72 | 4.80 → 7.96 |

| labels (both variants) | living room | bedroom | lobby |
|---|---|---|---|
| mIoU | 0.393 | 0.434 | 0.528 |
| PQ, all / objects / background | 0.325 / 0.167 / 0.956 | 0.297 / 0.187 / 0.916 | 0.375 / 0.204 / 0.889 |
| AP / AP50 / AP25 | 0.285 / 0.370 / 0.443 | 0.297 / 0.382 / 0.480 | 0.193 / 0.278 / 0.419 |

- **Accuracy improves on every scene**, and the views fuse: disagreement between
  overlapping views falls 6–8×.
- **Completeness drops**, by more than accuracy gains. The refined surfaces are thinner
  than Déjà View's scatter, which costs most in the large lobby. Setting the weights is the
  next step.
- **Labels** are the linked SAM 3 masks and do not move with refinement. mIoU counts classes
  SAM 3 finds but the scene lacks (4–5 per scene) as 0.
- **Noise:** GPU sums are not bitwise reproducible; the F-score used to vary by about 0.006
  between runs, and the spread of these scores is not measured yet.
- The tilt-shift scenes of the `stress` split are in `figures/eval_hypersim-20261103-150543`.

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
                    output (labeled.py: accuracy, completeness, mIoU, PQ, AP)
  analysis/         the Hypersim protocol (scene_eval.py), its figures and point renders
experiments/        SAM 3 study (LOG.md, scripts, results, figures)
figures/            refinement figures, one folder per run with a README
tests/
```
