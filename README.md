# semantic-3d-refinement

Can multi-view panoptic segments (SAM 3 masks linked across views) refine a frozen
feed-forward 3D reconstruction (Déjà View) into surfaces more accurate than its raw
output? **Not in our experiments, and our controls say the semantics are not what helps.** This
repository holds the pipeline, an evaluation protocol on Hypersim, and the experiments
behind that answer, negative results included.

![Déjà View and the refined output on the shared bedroom, views 18, 0, 12, 6: image, ground-truth objects, linked SAM 3 objects, each output's distance to the nearest ground-truth point (cm), and refined minus Déjà View. Blue is closer, red further. Refinement helps in patches (bed frame, far wall in view 6) and hurts in others (walls in views 18 and 12).](figures/eval_hypersim-20261110-093416/presentation/ai_006_007_cam_00_views.png)

## Results

Chamfer distance in cm, lower is better, after one Sim(3) alignment per scene (protocol:
[Evaluation](#evaluation-hypersim)). Hypersim: 20 views at 504 px; **dev** is 3 scenes
where choices were made, **val** 11 scenes that checked them; `test` has not been run.
Noise floor 0.036 cm (E40).

| Question | Baseline | Result | Where |
|---|---|---|---|
| Does joint refinement (Gauss-Newton on inverse depth) beat the raw prediction? | Déjà View: dev 3.50, val 5.10 | No: dev 3.93, val 5.64. Accuracy improves (4.68 → 4.20 on dev), completeness loses more (2.32 → 3.66). On ETH3D it is worse on 13 of 13 scenes | [Evaluation](#evaluation-hypersim), E26a |
| Does it beat simply dropping Déjà View's least dense points, at equal completeness? | Voxel-density filter | No: dev accuracy 3.09 cm for the filter, 4.20 for the refiner. Val was not run | E45 |
| Do the segments explain what small gain there is? | The same refiner with no segments | No: ids permuted at random beat SAM 3 beyond noise, and ground-truth ids gain less than SAM 3's. Chamfer against no segments, dev: SAM 3 -0.069, ground truth -0.021, permuted -0.249 (val: -0.119, -0.043, -0.346) | E8a, E26a |
| Does a learned refiner do better? | Déjà View | Per-view U-Net: val Chamfer 5.06 against 5.10, the size of the noise floor, masks or not. Fused voxel grid with a 3D U-Net: no model beats Déjà View on val | E11a, E41b |

In the current energy the segments only switch off part of the cross-view term (E8a-E8c);
`experiments/LOG.md` holds every setting we kept and why.

## Reproduce

Needs an RTX 3080-class GPU and access to the gated `facebook/sam3` weights
(`uv run hf auth login`). Not re-run from a clean clone since the last change; the
numbers above come from the runs named in `experiments/LOG.md`.

```bash
uv sync --all-groups
uv run hypersim-download --splits dev && uv run hypersim-prepare --splits dev
uv run python -m src.analysis.scene_eval --cfg configs/eval_hypersim.yaml --splits dev
```

`scene_eval` reads cached Déjà View, SAM 3 and DSINE outputs; the commands that write them
are listed under [Evaluation](#evaluation-hypersim).

## Team and credits

Pouya Shekarchizadeh Esfahani, Peter Hofinger, Mircea-Alexandru Răuțoiu. Project at TUM,
Niessner Lab, supervised by Umut Kocasarı and Robin Borth. Everyone worked on every part.

Built on, none of it copied in (third-party code lives in `data/third_party/`, git-ignored,
at a pinned commit):

- [Déjà View](https://github.com/nv-tlabs/dvlt) (NVIDIA, `dvlt` at f59bb18), the frozen
  backbone; SAM 3 (Meta, gated weights) for masks.
- BiNI's normal term (Cao et al., ECCV 2022), re-implemented from the paper in
  `src/refinement/normals.py`; no BiNI code (GPL-3) is used. Normals from DSINE (Bae and
  Davison, CVPR 2024; non-commercial licence), run outside git.
- Hypersim (Roberts et al., ICCV 2021) and ETH3D for evaluation; the protocol follows
  PanSt3R and Pano3D. PanSt3R, EPS3D and IGGT are run from their released code as
  competitors in E26a. Alignment is Umeyama's (1991) similarity transform.

## Setup

```bash
uv sync --all-groups
cp .env.example .env   # only for ScanNet++: add your ScanNet++ token
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
`Refiner.refine`). Each also has a trivial implementation (flat depth, no objects,
identity) so the pipeline's plumbing runs without a GPU or model weights:

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

E = E_d + E_mv + λ_n E_n + ‖δ‖², every weight 1 and λ_n = 1e-4 for E_n's units
(`configs/refine.yaml`). E_d's weight is 1 by definition, since E's scale does not move its
minimizer.

- **E_d**: Huber anchor to Déjà View's depth, weighted by its confidence times the share
  of views (its own included) that agree on the pixel's instance, so it weakens where views
  disagree.
- **E_mv**: overlapping views must agree on depth, only between pixels of one instance and
  not where one view's surface hides the other's: a gap counts as occluded above Otsu's
  threshold on the scene's gaps (about 4 % on Hypersim), so there is no tolerance to set.
- **E_n**: BiNI's normal term on DSINE normals (`src/refinement/normals.py`), the only
  shape term: only between pixels of one instance and cut on the input's occlusion
  contours. DSINE is non-commercial and stays outside git
  (`data/third_party/dsine_hub`); `uv run dsine-predict <views npz>` caches its normals.

**Hand-set constants (E40, issue #28).** Each was varied alone on the three dev scenes,
against a noise floor of 0.033 / 0.037 / 0.036 cm (accuracy / completeness / Chamfer). Only
λ_d and the confidence exponent were removed. The rest stay, for these reasons:

| Constant | Value | Why it stays |
|---|---|---|
| λ_mv, λ_δ | 1, 1 | 0.3 and 3 trade accuracy for completeness beyond noise on dev; λ_δ = 0 is singular |
| E_mv neighbours | 4 | 2 and 8 trade the same way; all 19 views exceed 10 GB |
| E_d Huber threshold | 0.05 | removing it ties on dev, but failed the val check (see below) |
| E_mv border margin | 1 px | keeps the four bilinear corners in the image; 0 px failed the same val check |
| E_n occlusion cut | 0.02 | Otsu's per-scene threshold ties on dev, but on val reaches 0.067, beyond what dev tested |
| E_n cut window, closing budget | 7 px | spans Déjà View's 2-4 px edge ramps; 3-11 px tie, no closing loses 0.05 cm |
| E_d weight | c × u | dropping c ties on dev (E37); combined with the three above, accuracy got worse on all 11 val scenes, and on dev restoring c recovered most of it |
| λ_n, DSINE | 1e-4 | issue #21 |
| Gauss-Newton × CG, CG tol | 5 × 10, 1e-3 | issue #10: where Gauss-Newton levels off, 4.2 → 1.9 s per scene |

On dev, removing Huber, the border margin and the fixed cut each tie. Combined, they cost 0.5 cm
in one val scene (ai_009_009, where Déjà View fails), which put the val means beyond noise, so
all three stay. Three dev scenes cannot separate settings that only matter on failures. Details
are in E40 in `experiments/LOG.md`.

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
  Every energy weight is 1 except λ_n = 1e-4, which sets E_n's units; E40 and E21a varied
  them on dev and kept these values.
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

**Competitors' geometry on ETH3D (E26a, issue #26).** ETH3D's 13 high-res training
scenes (7 indoor, 6 outdoor; in none of the methods' listed training sets), 20 views at
504 px, each output aligned by its own Sim(3); mean / median over the scenes, cm, ↓
(`experiments/results/e26a_eth3d.txt`, `experiments/figures/e26a_eth3d.png`). Every method
at its released checkpoint and its own settings, nothing tuned. PE3R is not in the table:
it needs more than the 3080's 10 GB, and we dropped it rather than wait on the cluster.

| | accuracy | completeness | Chamfer |
|---|---|---|---|
| Déjà View | 12.22 / 8.99 | **7.76** / **5.54** | **9.99** / **7.26** |
| Refined, no segments | **12.15** / **8.71** | 9.50 / 6.58 | 10.82 / 7.65 |
| Refined, SAM 3 (ours) | 12.22 / 8.76 | 9.29 / 6.37 | 10.75 / 7.56 |
| Refined, permuted SAM 3 | 12.32 / 8.80 | 8.89 / 6.18 | 10.60 / 7.49 |
| Déjà View, filtered (instance voxel density, q 0.5) | **6.11** / 4.41 | 7.86 / 5.73 | **6.99** / 5.38 |
| flag-refine-filter (E27a, issue #27) | 6.13 / **4.37** | **7.85** / 5.66 | **6.99** / **5.38** |
| PanSt3R v2 | 88.88 / 35.05 | 154.16 / 21.36 | 121.52 / 28.76 |
| EPS3D | 38.24 / 28.70 | 33.57 / 17.21 | 35.90 / 24.36 |
| IGGT | 26.56 / 13.33 | 27.09 / 8.37 | 26.83 / 10.85 |

Our refinement is worse than its input on Chamfer on all 13 scenes. The segments beat no
segments on 11 of 13 by 0.07 cm (mean), but randomly permuted ids beat the real ones on 12 of
13, so the semantics are not what helps here. Déjà View leads every competitor on the mean;
IGGT has the best indoor median (Chamfer 3.81 against 6.20) and fails outdoors; PanSt3R's
frozen MUSt3R_512_cvpr breaks on 5 scenes (over 2 m). The filter, not the refinement,
carries flag-refine-filter here: Chamfer 9.99 -> 6.99 cm against Déjà View on all 13 scenes,
but a tie with the filter alone on the same pixels (median -0.02 cm, 9 of 13 scenes).

```bash
uv run hypersim-download --splits dev stress && uv run hypersim-prepare --splits dev stress
uv run python -m src.analysis.scene_eval --cfg configs/eval_hypersim.yaml --splits dev stress   # inputs: dejaview-predict, dsine-predict, sam3-segment
uv run python -m src.analysis.scene_figures --run runs/eval_hypersim-<time>
```

Every energy weight 1 but λ_n, Gauss-Newton 5 × 10 (`figures/eval_hypersim-20261110-093416`;
about 1 min for both splits on an RTX 3080, refinement 2 s per scene). Its `presentation/` folder holds the shared bedroom at the views of
issue #7 (18, 0, 12, 6), `debug/` everything else. Déjà View → Refined:

| cm, ↓ | dev aggregate | living room | bedroom | lobby |
|---|---|---|---|---|
| accuracy | 4.68 → 4.20 | 2.19 → 1.79 | 2.52 → 2.21 | 9.31 → 8.60 |
| completeness | 2.32 → 3.66 | 1.01 → 1.40 | 1.17 → 1.71 | 4.80 → 7.87 |
| Chamfer | 3.50 → 3.93 | 1.60 → 1.59 | 1.85 → 1.96 | 7.05 → 8.23 |

| labels (both variants) | dev aggregate | living room | bedroom | lobby |
|---|---|---|---|---|
| mIoU | 0.360 | 0.345 | 0.413 | 0.457 |
| PQ^sc | 0.292 | 0.286 | 0.269 | 0.321 |
| AP | 0.216 | 0.228 | 0.276 | 0.145 |

- **Refinement does not improve Chamfer on dev:** it is 3.50 → 3.93 cm, 0.01 cm better in
  the living room and worse elsewhere. Accuracy improves on every scene and
  disagreement between overlapping views falls 6-7×, but the refined surfaces are thinner
  than Déjà View's scatter, so completeness loses more than accuracy gains, most in the
  large lobby. No weight improves both: λ_mv, λ_δ and the neighbour count each trade
  accuracy against completeness on dev (E40).
- **Against simply dropping points** (E45, issue #27): Déjà View's points cut by a per-pixel score at the
  scene's own quantiles until they are exactly as complete as the refiner's output (3.66 cm on dev), and scored by
  the same protocol. Dev accuracy at that completeness: voxel density 3.09, confidence 3.52, PCL's statistical
  outlier removal 3.80, the refiner 4.20 cm; depth gradient (4.73), MoGe's depth edge (4.59), random (4.68) and
  multi-view consistency (4.33) do worse than the refiner. Voxel density beats it on all 3 dev and 3 stress scenes.
  The refinement buys less accuracy per unit of completeness than dropping points does; val was not run.
  Refining only the points a filter keeps does not change that: below voxel density's own curve, above
  confidence's only where completeness is already 5 cm or worse (dev, 3 scenes).
  Averaging each voxel's points into one (voxel fusion) does not help either; the best of seven voxel
  variants counts only the points of the pixel's own SAM 3 instance (3.00 cm, dev; chosen on dev, gain over
  voxel density beyond noise only in the lobby).
- **Stress** (tilt-shift cameras, no aggregate) shows the same pattern: Chamfer 2.17 → 2.37 in
  the kitchen, 15.29 → 15.93 in the conference room, 3.99 → 4.34 in the restaurant.
- **Labels** are the linked SAM 3 masks and do not move with refinement. mIoU counts classes
  SAM 3 finds but the scene lacks (4-5 per dev scene) as 0. The dev mIoU is pooled over 29
  classes (0.360); the mean of its scenes would be 0.405.
- **Noise:** GPU sums are not bitwise reproducible; repeated runs on one GPU agree to
  0.00002 cm on the dev aggregate, two GPUs to about 0.035 cm (E40).
- **Val** (11 scenes no value was chosen on; E40 used it to reject three changes, `--splits
  val`, RTX 3080) confirms it: accuracy 6.06 → 5.70 cm (better on 10 of 11), completeness
  4.14 → 5.58 cm (worse on all 11), Chamfer 5.10 → 5.64 cm (worse on 10 of 11). Per scene:
  E41 in `experiments/LOG.md`.
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
  backbone/         Backbone interface, Déjà View (dejaview.py), SegVGGT (segvggt.py)
  segmentation/     Segmentation, Segmenter interface, SAM 3 + association
  refinement/       Refiner interface; joint refinement by Gauss-Newton (multiview.py),
                    the normal term on DSINE normals (normals.py)
  output.py         PointCloud, PLY export
  pipeline.py       chains the stages
  evaluation/       metrics against ground truth: instances (instances.py), labeled
                    output (labeled.py: accuracy, completeness, Chamfer, mIoU, PQ, AP)
  analysis/         the Hypersim protocol (scene_eval.py), its figures and point renders
experiments/        the experiment log (LOG.md), one script per experiment, results, figures
figures/            refinement figures, one folder per run with a README
tests/
```
