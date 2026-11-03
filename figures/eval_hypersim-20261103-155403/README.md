# Evaluation protocol: Déjà View against the refined cloud on Hypersim's dev scenes

Run `runs/eval_hypersim-20261103-155403` (`configs/eval_hypersim.yaml`, `configs/refine.yaml`;
git `32d6e93`). Reproduce, about 2.5 min for the cached inputs and 4 min for the rest on an RTX 3080
(SAM 3 needs access to the gated `facebook/sam3` on Hugging Face, or a local copy with
`HF_HUB_OFFLINE=1`):

```bash
uv run hypersim-download --splits dev && uv run hypersim-prepare --splits dev
uv run dejaview-predict data/processed/hypersim/*/views_n20_uniform_504.npz
uv run sam3-segment --mode image --prompts configs/prompts/nyu40.txt --threshold 0.3 data/processed/hypersim/*/views_n20_uniform_504.npz
uv run dsine-predict data/processed/hypersim/*/views_n20_uniform_504.npz
uv run python -m src.analysis.scene_eval --cfg configs/eval_hypersim.yaml       # 1.7 min
uv run python -m src.analysis.scene_figures --run runs/eval_hypersim-<time>     # 2 min
```

## Setup

- **Scenes:** the `dev` split of `configs/hypersim.yaml`, three Hypersim validation scenes
  with centred pinhole cameras, 20 views each, uniform over the trajectory, 504 × 378:
  `ai_010_007` (living room), `ai_006_007` (bedroom) and `ai_044_001` (hotel lobby). Déjà View
  always predicts a centred principal point, so the tilt-shift scenes of the earlier report
  (`figures/eval_hypersim-20261103-150543`) are the `stress` split now and are not used for
  choices (E30 in `experiments/LOG.md`). The test split is never used.
- **Pipeline:** Déjà View (K = 16) → SAM 3 image mode, one prompt per NYU40 class, threshold
  0.3, linked across views → DSINE normals on Déjà View's intrinsics → joint refinement.
- **Energy** (`src/refinement/multiview.py`), Gauss-Newton, 10 steps of 20 CG iterations,
  per pixel d = d⁰ + δ: E = E_d + E_mv + λ_n E_n + ‖δ‖², **every weight 1, none tuned**, λ_n =
  1e-4 for E_n's units (BiNI's residuals are log-depth steps times the focal length). E_d is
  Huber, weighted by confidence × consistency u; E_n and E_mv stay within one linked
  instance; E_n is cut on the input's occlusion contours (`edge_tol`, closed contours from a
  7 px window); E_mv drops matches above Otsu's threshold on the scene's depth gaps.
- **Black box:** only the output cloud is scored, registered onto the ground truth from the
  clouds alone (first-camera start, ICP with scale; `labeled.register`).
- **Scores** (`src/evaluation/labeled.py`), on a 1 cm grid: accuracy and completeness (mean,
  capped at 20 cm), precision, recall and F-score at 5 cm, for the scene and per class; mIoU
  and AP50 on labels transferred to the ground-truth points within 5 cm. Disagreement
  between overlapping views is a diagnostic, not a score.

## Ablations

The parts removed this week were removed on the restaurant, a stress scene. Each decision is
repeated here on the dev scenes, one change at a time. A removed part is run on the last
commit that still has it: E_bd on `d357e8e`, a, b and `SCALE` on `1d86004`. The rest run on
the final code. Rule: remove a part if both the scores and the renders improve or hold.
F-score at 5 cm:

| variant | living room | bedroom | lobby | renders | decision |
|---|---|---|---|---|---|
| final energy | 0.950 | 0.911 | 0.648 | | |
| with E_bd (`d357e8e`) | 0.940 | 0.902 | 0.644 | | |
| … without E_bd | 0.941 | 0.901 | 0.642 | as with it | **E_bd removed** |
| with a, b (`1d86004`) | 0.944 | 0.905 | 0.609 | | |
| … without a, b, δ penalty kept | **0.951** | **0.910** | **0.642** | as with them | **a, b and `min_pixels` removed** |
| … without `SCALE`, λ_n = 1e-4 | 0.945 | 0.900 | 0.615 | as with it | **`SCALE` removed** (the same energy) |
| no occlusion cut (`edge_tol` ∞) | 0.936 | 0.889 | 0.626 | smears across depth jumps | kept |
| per-edge cut, no window or closing | 0.953 | 0.912 | 0.653 | more stray points over the bed | kept |
| one residual per edge (n_p only) | 0.952 | 0.908 | 0.646 | as the final energy | kept (no gain either way) |

- **Every decision of the restaurant holds on dev.** Without a, b the lobby gains most
  (0.609 → 0.642, AP50 0.153 → 0.247).
- **Per-edge cutting scores within noise of the final cut, but its renders are worse**
  (bedroom, figure below), so the closed contours stay.
- **Noise.** The solver is not bitwise reproducible on the GPU. The same energy ran three
  times (final energy in the ablation run and in this run, and "without a, b" on `1d86004`):
  0.950/0.950/0.951, 0.911/0.910/0.910 and 0.648/0.644/0.642. Differences below about 0.006
  are noise.

![bedroom ablations](ai_006_007_cam_00_ablations.png)
![living room ablations](ai_010_007_cam_00_ablations.png)
![lobby ablations](ai_044_001_cam_00_ablations.png)

*Each variant from the same camera, with the image colors (top) and gray shading (bottom).
Removing the occlusion cut smears surfaces across depth jumps (the bed, the lobby's
balcony). Cutting edge by edge leaves more stray points over the bed.*

![bedroom removed parts](ai_006_007_cam_00_removed.png)
![living room removed parts](ai_010_007_cam_00_removed.png)
![lobby removed parts](ai_044_001_cam_00_removed.png)

*The removed parts, each pair on its own commit: E_bd (first two), then a, b and `SCALE`
(last three). Each pair looks the same.*

## Results

| scene | variant | accuracy (cm) | completeness (cm) | precision | recall | F-score | mIoU | AP50 | disagreement (%) | GN steps | s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| living room | Déjà View | 2.24 | 1.03 | 0.899 | 0.986 | 0.941 | 0.544 | 0.375 | 0.30 | – | – |
| living room | Refined | 1.81 | 1.36 | 0.935 | 0.965 | **0.950** | 0.502 | 0.324 | 0.05 | 10 | 4.5 |
| bedroom | Déjà View | 2.71 | 1.34 | 0.868 | 0.964 | 0.913 | 0.505 | 0.375 | 0.56 | – | – |
| bedroom | Refined | 2.46 | 1.75 | 0.883 | 0.938 | 0.910 | 0.493 | 0.372 | 0.09 | 10 | 3.3 |
| lobby | Déjà View | 6.13 | 3.82 | 0.582 | 0.811 | 0.677 | 0.713 | 0.542 | 0.51 | – | – |
| lobby | Refined | 5.89 | 5.30 | 0.610 | 0.682 | 0.644 | 0.546 | 0.247 | 0.06 | 10 | 4.0 |

F-score at 5 cm per class, Déjà View → Refined (classes with at least 100 ground-truth voxels
in a scene):

| class | living room | bedroom | lobby |
|---|---|---|---|
| wall | 0.93 → 0.96 | 0.94 → 0.94 | 0.61 → 0.56 |
| floor | 0.99 → 0.99 | 0.98 → 0.98 | 0.96 → 0.95 |
| table | 0.82 → 0.81 | 0.90 → 0.88 | 0.78 → 0.69 |
| lamp | 0.81 → 0.78 | 0.81 → 0.80 | – |
| sofa | – | – | 0.67 → 0.64 |
| bed | 0.89 → 0.90 | 0.94 → 0.93 | – |
| door | – | 0.96 → 1.00 | 0.40 → 0.25 |

Energy at the start and after 10 steps (means over all pixels):

| scene | E_d | E_mv | λ_n E_n | ‖δ‖² |
|---|---|---|---|---|
| living room | 0 → 6.4e-06 | 1.5e-04 → 4.5e-05 | 1.6e-03 → 1.5e-04 | 0 → 1.7e-04 |
| bedroom | 0 → 1.6e-05 | 2.6e-04 → 1.2e-04 | 4.6e-03 → 2.0e-04 | 0 → 3.7e-04 |
| lobby | 0 → 7.6e-06 | 2.2e-04 → 1.9e-05 | 1.5e-03 → 5.3e-05 | 0 → 7.7e-05 |

## Figures

![summary](summary.png)

*Every score of every variant; dashed: Déjà View per scene.*

![bedroom render](ai_006_007_cam_00_render.png)

*Bedroom: each cloud (all 20 views' pixels) rendered from the same two cameras, both fixed
by the ground truth: a dollhouse view from above without the ceiling, and the view that sees
deepest into the room, pulled back (`src/analysis/render.py`).*

![living room render](ai_010_007_cam_00_render.png)
![lobby render](ai_044_001_cam_00_render.png)

![bedroom section](ai_006_007_cam_00_section.png)
![living room section](ai_010_007_cam_00_section.png)
![lobby section](ai_044_001_cam_00_section.png)

*A 2 cm slice 75 cm above the floor, gray ground truth, one color per view; bottom: 1 m of a
wall.*

![bedroom energy](ai_006_007_cam_00_energy.png)

*Bedroom: each term's weighted cost per pixel on Déjà View's depth (top) and the refined
depth (bottom), log scale over four decades below each term's 99.5th percentile. Second
column: the consistency u that gates E_d.*

![living room energy](ai_010_007_cam_00_energy.png)
![lobby energy](ai_044_001_cam_00_energy.png)

![bedroom normals](ai_006_007_cam_00_normals.png)
![living room normals](ai_010_007_cam_00_normals.png)
![lobby normals](ai_044_001_cam_00_normals.png)

![bedroom classes](ai_006_007_cam_00_classes.png)
![living room classes](ai_010_007_cam_00_classes.png)
![lobby classes](ai_044_001_cam_00_classes.png)

*F-score at 5 cm and IoU per class with at least 100 ground-truth voxels.*

![bedroom views](ai_006_007_cam_00_views.png)
![bedroom from above](ai_006_007_cam_00_topdown.png)
![living room views](ai_010_007_cam_00_views.png)
![living room from above](ai_010_007_cam_00_topdown.png)
![lobby views](ai_044_001_cam_00_views.png)
![lobby from above](ai_044_001_cam_00_topdown.png)

## Findings

Three scenes, one run each. A claim holds on all three unless a scene is named.

1. **Accuracy improves everywhere** (2.24 → 1.81, 2.71 → 2.46, 6.13 → 5.89 cm), and so does
   precision. On the living room, the F-score passes Déjà View's for the first time (0.941 →
   0.950).
2. **The views fuse.** Disagreement between overlapping views falls 6–8× (0.30 → 0.05 %,
   0.56 → 0.09 %, 0.51 → 0.06 %).
3. **Completeness and recall drop.** The refined surfaces are thinner. Déjà View's scatter
   around a surface covers more ground-truth points within 5 cm than one surface in place does.
   The depth moves little: a median of 0.15–0.22 %, and more than 10 % at only 0.1–0.3 % of the
   pixels. The registration is the same for both clouds (0.04–0.13°, 1–2 cm, scale within
   0.1 %).
4. **The lobby loses most, and its labels with it** (recall 0.81 → 0.68, mIoU 0.71 → 0.55,
   AP50 0.54 → 0.25; IoU of the desk 0.66 → 0.25, of the doors 0.67 → 0.25). It is the largest scene (registration
   scale 9.3, against 4.7 and 2.4), so a relative change of 0.2 % is centimetres there. Labels
   are transferred to the ground truth within 5 cm, so they fall with the recall.
5. **On the dev scenes the method is closer to Déjà View than on the stress scenes** (F-score
   +0.009, −0.003, −0.033 against −0.021, −0.030, −0.047). The weights are still all 1.
   Setting them on dev, and checking on the `val` split, is the next step.
