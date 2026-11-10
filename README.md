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
- One stored camera orientation is 10 × a rotation (ai_031_004, keyframe 0): the loader
  divides a uniform scale out and rejects any other non-rotation (E34 in `experiments/LOG.md`).
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
(`uv run hf auth login`). With the 37 `configs/prompts/nyu40.txt` prompts of the evaluation, a
20-view scene takes ~17 s for SAM 3 and ~0.6 s for the association on an RTX 3080, ~2.9 GB
GPU memory (`experiments/LOG.md`, E39).

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

One protocol for every variant (`src/analysis/scene_eval.py`, `configs/eval_hypersim.yaml`).
Geometry comes first, segmentation second.

- **Data:** the splits of `configs/hypersim.yaml`, 20 evenly spaced views per scene, 504 px.
  No weight is tuned: every energy term has weight 1.
- **Output:** per input pixel, a 3D point in view 0's frame (V, H, W, 3) and an instance id
  (V, H, W). The same id means the same object in every view. Each id has a class and a
  confidence.
- **Alignment:** one Umeyama Sim(3) on the pixel pairs, since output pixel i matches
  ground-truth pixel i.
- **Classes:** the dataset's full label set (Hypersim: all 40 NYU40 classes). Wall, floor and
  ceiling have no instances in Hypersim, so all pixels of such a class in a scene form one
  segment, in both ground truth and prediction. Unlabeled pixels are void for label scores,
  but still count for geometry.
- **Labels:** scored per pixel, so geometric drift doesn't affect them.
- **Thresholds:** none beyond the metrics' own definitions.

| Tier | Metrics |
|---|---|
| Headline | mean accuracy, mean completeness, mean Chamfer (all uncapped), mIoU, PQ^sc, class-aware AP |
| Supplementary | medians, SQ, RQ, PQ / SQ / RQ for objects (_th) and background (_st), AP50, AP25, class-agnostic AP, refinement runtime |
| Debug (team only) | all scores per class, alignment scale |

- **Aggregation:** every split except stress gets a split aggregate. Dev, val and stress also
  get per-scene scores. mIoU pools each class over the split's scenes, then averages over
  classes. All other scores are averaged over the scenes. Within a scene, PQ averages over
  classes appearing in the ground truth or prediction, and AP over object classes in the
  ground truth.
- **Comparability:** per-pixel points and labels and PQ^sc follow PanSt3R and Pano3D, and the
  evenly spaced views follow Pano3D. Our numbers are not directly comparable to
  competitors, because scenes, classes and scored views differ. Comparing requires running
  their methods on this protocol.

```bash
uv run hypersim-download --splits dev stress && uv run hypersim-prepare --splits dev stress
uv run python -m src.analysis.scene_eval --cfg configs/eval_hypersim.yaml --splits dev stress   # inputs: dejaview-predict, dsine-predict, sam3-segment
uv run python -m src.analysis.scene_figures --run runs/eval_hypersim-<time>
```

Every energy weight 1, none tuned (`figures/eval_hypersim-20261105-104445`; 1.2 min for both
splits on an RTX 3080). Its `presentation/` folder holds the shared bedroom at the views of
issue #7 (18, 0, 12, 6), `debug/` everything else. Déjà View → Refined:

| cm, ↓ | dev aggregate | living room | bedroom | lobby |
|---|---|---|---|---|
| accuracy | 4.68 → 4.21 | 2.19 → 1.79 | 2.52 → 2.20 | 9.31 → 8.63 |
| completeness | 2.32 → 3.69 | 1.01 → 1.41 | 1.17 → 1.72 | 4.80 → 7.96 |
| Chamfer | 3.50 → 3.95 | 1.60 → 1.60 | 1.85 → 1.96 | 7.05 → 8.30 |

| labels (both variants) | dev aggregate | living room | bedroom | lobby |
|---|---|---|---|---|
| mIoU | 0.360 | 0.345 | 0.413 | 0.457 |
| PQ^sc | 0.292 | 0.286 | 0.269 | 0.321 |
| AP | 0.216 | 0.228 | 0.276 | 0.145 |

- **Refinement does not improve Chamfer on any dev scene:** it is 3.50 → 3.95 cm on dev,
  unchanged in the living room and worse elsewhere. Accuracy improves on every scene and
  disagreement between overlapping views falls 6-8×, but the refined surfaces are thinner
  than Déjà View's scatter, so completeness loses more than accuracy gains, most in the
  large lobby. Setting the weights is the next step.
- **Stress** (tilt-shift cameras, no aggregate) shows the same pattern: Chamfer 2.17 → 2.39 in
  the kitchen, 15.29 → 15.91 in the conference room, 3.99 → 4.36 in the restaurant.
- **Labels** are the linked SAM 3 masks and do not move with refinement. mIoU counts classes
  SAM 3 finds but the scene lacks (4-5 per dev scene) as 0. The dev mIoU is pooled over 29
  classes (0.360); the mean of its scenes would be 0.405.
- **Noise:** GPU sums are not bitwise reproducible, and the spread of these scores is not
  measured yet.
- **Val** (11 scenes no choice was made on, `--splits val`, RTX 5090, 2.2 min) confirms it:
  accuracy 6.07 → 5.75 cm (better on 9 of 11), completeness 4.15 → 5.67 cm (worse on all
  11), Chamfer 5.11 → 5.71 cm (worse on 10 of 11). Per scene: E34 in `experiments/LOG.md`.
- `test` runs with the same command once its inputs are cached; it has not been run.

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
                    output (labeled.py: accuracy, completeness, Chamfer, mIoU, PQ, AP)
  analysis/         the Hypersim protocol (scene_eval.py), its figures and point renders
experiments/        SAM 3 study (LOG.md, scripts, results, figures)
figures/            refinement figures, one folder per run with a README
tests/
```
