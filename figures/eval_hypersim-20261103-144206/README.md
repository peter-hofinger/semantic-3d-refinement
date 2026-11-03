# Evaluation protocol: Déjà View against the refined cloud on three Hypersim scenes

Run `runs/eval_hypersim-20261103-144206` (`configs/eval_hypersim.yaml`, `configs/refine.yaml`;
git `d357e8e` plus the uncommitted changes of the commit that adds this report). Reproduce:

```bash
uv run dsine-predict data/processed/hypersim/*/views_n20_uniform_504.npz
uv run python -m src.analysis.scene_eval --cfg configs/eval_hypersim.yaml       # 1.5 min
uv run python -m src.analysis.scene_figures --run runs/eval_hypersim-<time>     # 2.5 min
```

## Setup

- **Scenes:** Hypersim validation split, 20 views each, uniform over the trajectory, 504 × 378:
  `ai_015_001` (kitchen), `ai_047_009` (conference room), `ai_052_001` (restaurant). The test
  split is never used.
- **Pipeline:** Déjà View (K = 16) → SAM 3 image mode, one prompt per NYU40 class, threshold
  0.3, linked across views → DSINE normals on Déjà View's intrinsics → joint refinement.
- **Energy** (`src/refinement/multiview.py`), Gauss-Newton, 10 steps of 20 CG iterations:
  E = E_d + E_mv + E_n + ‖δ‖², **every weight 1, none tuned**. E_d is Huber, weighted by
  confidence × consistency u; E_n and E_mv stay within one linked instance; a, b per (view,
  instance). E_n is the only shape term: the proposal's E_bd is dropped (finding 1). E_n
  carries a fixed factor of 1e-4 for its units (BiNI's residuals are log-depth steps times the
  focal length) and is cut on the input's occlusion contours. E_mv drops matches whose depth
  gap lies above Otsu's threshold on the scene's gaps (0.038–0.048 here).
- **Black box:** only the output cloud is scored, registered onto the ground truth from the
  clouds alone (first-camera start, ICP with scale; `labeled.register`).
- **Scores** (`src/evaluation/labeled.py`), on a 1 cm grid: accuracy and completeness (mean,
  capped at 20 cm), precision, recall and F-score at 5 cm, for the scene and per class; mIoU
  and AP50 on labels transferred to the ground-truth points within 5 cm. Disagreement
  between overlapping views is a diagnostic, not a score.

## Results

| scene | variant | accuracy (cm) | completeness (cm) | precision | recall | F-score | mIoU | AP50 | disagreement (%) | GN steps | s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| kitchen | Déjà View | 3.13 | 1.39 | 0.851 | 0.977 | 0.910 | 0.577 | 0.535 | 0.48 | – | – |
| kitchen | Refined | 3.12 | 1.92 | 0.837 | 0.937 | 0.884 | 0.536 | 0.493 | 0.04 | 10 | 4.6 |
| conference | Déjà View | 7.94 | 3.25 | 0.495 | 0.852 | 0.626 | 0.505 | 0.304 | 0.96 | – | – |
| conference | Refined | 8.57 | 4.99 | 0.462 | 0.703 | 0.558 | 0.375 | 0.204 | 0.15 | 10 | 4.6 |
| restaurant | Déjà View | 4.80 | 2.43 | 0.687 | 0.895 | 0.777 | 0.547 | 0.210 | 0.60 | – | – |
| restaurant | Refined | 4.91 | 3.67 | 0.664 | 0.775 | 0.715 | 0.485 | 0.162 | 0.09 | 10 | 4.5 |

F-score at 5 cm per class, Déjà View → Refined:

| class | kitchen | conference | restaurant |
|---|---|---|---|
| wall | 0.94 → 0.94 | 0.70 → 0.57 | 0.74 → 0.61 |
| floor | 0.99 → 0.98 | 0.88 → 0.87 | 0.95 → 0.94 |
| chair | 0.89 → 0.81 | 0.46 → 0.50 | 0.75 → 0.66 |
| table | 0.92 → 0.89 | 0.00 → 0.00 (finding 5) | 0.81 → 0.78 |
| lamp | 0.87 → 0.79 | 0.00 → 0.00 (finding 5) | 0.76 → 0.69 |

Energy at the start and after 10 steps (means over all pixels; λ = 1):

| scene | E_d | E_mv | E_n | ‖δ‖² |
|---|---|---|---|---|
| kitchen | 0 → 1.5e-05 | 2.5e-04 → 5.5e-05 | 4.2e-03 → 2.7e-04 | 0 → 3.6e-04 |
| conference | 0 → 4.2e-05 | 7.0e-04 → 1.5e-04 | 3.1e-03 → 2.0e-04 | 0 → 2.8e-04 |
| restaurant | 0 → 2.3e-05 | 3.3e-04 → 1.4e-04 | 8.1e-03 → 6.3e-04 | 0 → 5.7e-04 |

## Figures

![summary](summary.png)

*Every score of every variant; dashed: Déjà View per scene.*

![restaurant render](ai_052_001_cam_00_render.png)

*Restaurant: each cloud (all 20 views' pixels) rendered from the same two cameras, both fixed
by the ground truth: a dollhouse view from above without the ceiling, and the view that sees
deepest into the room, pulled back. Each point covers its source pixel's patch of surface
(`src/analysis/render.py`); shaded is gray under a light at the camera, which shows geometry
the image colors hide.*

![kitchen render](ai_015_001_cam_00_render.png)
![conference render](ai_047_009_cam_00_render.png)

![restaurant instance cut](ai_052_001_cam_00_instance_cut.png)

*Restaurant, refined, same camera: without E_bd and with E_n across instance boundaries
(left, a scratch run of the previous commit with λ_bd = 0), and this commit's E_n cut at
instance boundaries (right). The cut leaves more stray points between the tables and under the
ceiling (finding 2).*

![kitchen section](ai_015_001_cam_00_section.png)

*Kitchen: a 2 cm slice 75 cm above the floor, gray ground truth, one color per view; bottom:
1 m of the right-hand wall.*

![conference section](ai_047_009_cam_00_section.png)
![restaurant section](ai_052_001_cam_00_section.png)

![restaurant energy](ai_052_001_cam_00_energy.png)

*Restaurant, view 5: each term's weighted cost per pixel on Déjà View's depth (top) and the
refined depth (bottom), log scale over four decades below each term's 99.5th percentile.
Second column: the consistency u that gates E_d (dark: all other views agree on the
instance).*

![kitchen energy](ai_015_001_cam_00_energy.png)
![conference energy](ai_047_009_cam_00_energy.png)

![restaurant normals](ai_052_001_cam_00_normals.png)

*Restaurant, view 5: surface normals of each variant (central differences, color =
direction).*

![kitchen normals](ai_015_001_cam_00_normals.png)
![conference normals](ai_047_009_cam_00_normals.png)

![kitchen classes](ai_015_001_cam_00_classes.png)
![conference classes](ai_047_009_cam_00_classes.png)
![restaurant classes](ai_052_001_cam_00_classes.png)

*F-score at 5 cm and IoU per class with at least 100 ground-truth voxels.*

![kitchen views](ai_015_001_cam_00_views.png)

*Kitchen, 4 of 20 views: each pixel's distance to the nearest ground-truth point, in cm; last
column: Refined minus Déjà View, blue means closer.*

![kitchen from above](ai_015_001_cam_00_topdown.png)
![conference views](ai_047_009_cam_00_views.png)
![conference from above](ai_047_009_cam_00_topdown.png)
![restaurant views](ai_052_001_cam_00_views.png)
![restaurant from above](ai_052_001_cam_00_topdown.png)

## Findings

Three scenes, one run each; claims hold on all three unless a scene is named.

1. **E_bd is redundant next to E_n.** In a scratch run of the previous commit with λ_bd = 0
   and nothing else changed, the depth differed from the full energy's by a median of
   0.006–0.016 % (the refinement itself moves it by 0.30–0.63 %), the renders were
   indistinguishable, and the F-score at 5 cm moved by at most 0.003 (0.880 → 0.878,
   0.558 → 0.557, 0.704 → 0.707). At the end of the solve E_bd was 3–7× smaller than E_n.
2. **Cutting E_n at instance boundaries scores a little better and looks a little worse.**
   This commit keeps a wall's pixels from shaping the table in front of it, as E_bd's stencils
   did. Against the full energy of the previous commit, F rises 0.880 → 0.884, 0.558 → 0.558,
   0.704 → 0.715 and completeness improves (1.97 → 1.92, 5.37 → 4.99, 4.13 → 3.67 cm), but the
   renders show more stray points between objects: pixels on a mask's bleeding edge or in a
   tiny mask fragment lose the neighbors that pulled them onto a surface.
3. **The views fuse.** Disagreement between overlapping views falls 6–12× (0.48 → 0.04 %,
   0.96 → 0.15 %, 0.60 → 0.09 %).
4. **The refined cloud still scores lower than Déjà View's.** The F-score at 5 cm falls on
   every scene (0.910 → 0.884, 0.626 → 0.558, 0.777 → 0.715), mostly through completeness
   (1.39 → 1.92, 3.25 → 4.99, 2.43 → 3.67 cm); walls lose most in the conference room and the
   restaurant (0.70 → 0.57, 0.74 → 0.61), conference chairs gain (0.46 → 0.50).
5. **Conference room, table and lamps: the ground truth is behind the glass.** Hypersim labels
   the conference table *desk*; its *table* and *lamps* are furniture in the next room, 10–28 m
   from the cameras, seen through the glass wall, which Déjà View puts on the glass at about
   half their distance. Both score 0 in either variant.
