# Evaluation protocol: Déjà View against the refined cloud on three Hypersim scenes

Run `runs/eval_hypersim-20261103-131714` (`configs/eval_hypersim.yaml`, `configs/refine.yaml`;
git `d7ee0ec` plus the uncommitted changes of the commit that adds this report). Reproduce:

```bash
uv run dsine-predict data/processed/hypersim/*/views_n20_uniform_504.npz
uv run python -m src.analysis.scene_eval --cfg configs/eval_hypersim.yaml       # 1.5 min
uv run python -m src.analysis.scene_figures --run runs/eval_hypersim-<time>     # 2 min
```

## Setup

- **Scenes:** Hypersim validation split, 20 views each, uniform over the trajectory, 504 × 378:
  `ai_015_001` (kitchen), `ai_047_009` (conference room), `ai_052_001` (restaurant). The test
  split is never used.
- **Pipeline:** Déjà View (K = 16) → SAM 3 image mode, one prompt per NYU40 class, threshold
  0.3, linked across views → DSINE normals on Déjà View's intrinsics → joint refinement.
- **Energy** (`src/refinement/multiview.py`), Gauss-Newton, 10 steps of 20 CG iterations:
  E = E_d + E_mv + E_bd + E_n + ‖δ‖², **every weight 1, none tuned**. E_d is Huber,
  weighted by confidence × consistency u; E_bd and E_mv stay within one linked instance;
  a, b per (view, instance). E_n carries a fixed factor of 1e-4 for its units: BiNI's
  residuals are log-depth steps times the focal length, so E_n would otherwise start four to
  five orders of magnitude above the other terms. E_mv drops matches whose depth gap lies
  above Otsu's threshold on the scene's gaps (0.038–0.048 here; earlier a fixed 0.05).
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
| kitchen | Refined | 3.09 | 1.97 | 0.837 | 0.930 | 0.881 | 0.514 | 0.493 | 0.05 | 10 | 5.4 |
| conference | Déjà View | 7.94 | 3.25 | 0.495 | 0.852 | 0.626 | 0.505 | 0.304 | 0.96 | – | – |
| conference | Refined | 8.63 | 5.42 | 0.468 | 0.691 | 0.558 | 0.384 | 0.207 | 0.15 | 10 | 5.4 |
| restaurant | Déjà View | 4.80 | 2.43 | 0.687 | 0.895 | 0.777 | 0.547 | 0.210 | 0.60 | – | – |
| restaurant | Refined | 5.14 | 4.08 | 0.650 | 0.754 | 0.698 | 0.480 | 0.128 | 0.09 | 10 | 5.3 |

F-score at 5 cm per class, Déjà View → Refined (– : not in the scene):

| class | kitchen | conference | restaurant |
|---|---|---|---|
| wall | 0.94 → 0.93 | 0.70 → 0.53 | 0.74 → 0.62 |
| floor | 0.99 → 0.98 | 0.88 → 0.87 | 0.95 → 0.93 |
| chair | 0.89 → 0.79 | 0.46 → 0.55 | 0.75 → 0.66 |
| table | 0.92 → 0.88 | 0.00 → 0.00 (finding 5) | 0.81 → 0.77 |
| desk | – | 0.91 → 0.92 | – |
| lamp | 0.87 → 0.76 | 0.00 → 0.00 (finding 5) | 0.76 → 0.64 |

Energy at the start and after 10 steps (means over all pixels; λ = 1):

| scene | E_d | E_mv | E_bd | E_n | ‖δ‖² |
|---|---|---|---|---|---|
| kitchen | 0 → 1.8e-05 | 2.5e-04 → 4.9e-05 | 1.0e-03 → 4.3e-05 | 5.5e-03 → 2.9e-04 | 0 → 5.4e-04 |
| conference | 0 → 5.0e-05 | 7.0e-04 → 1.8e-04 | 7.2e-04 → 2.6e-05 | 4.3e-03 → 2.6e-04 | 0 → 4.2e-04 |
| restaurant | 0 → 3.0e-05 | 3.3e-04 → 1.7e-04 | 3.7e-03 → 2.0e-04 | 1.1e-02 → 7.8e-04 | 0 → 1.2e-03 |

## Figures

![summary](summary.png)

*Every score of every variant; dashed: Déjà View per scene.*

![restaurant render](ai_052_001_cam_00_render.png)

*Restaurant: each cloud (all 20 views' pixels) rendered from the same two cameras, both fixed
by the ground truth: a dollhouse view from above without the ceiling, and the view that sees
deepest into the room, pulled back. Each point covers its source pixel's patch of surface
(`src/analysis/render.py`); shaded is gray under a light at the camera, which shows geometry
the image colors hide. Refined walls, floor and ceiling are smooth where Déjà View's are
bumpy, and fewer stray points float in the room; ghosts that several views agree on stay
(the kitchen's arch in front of the wall).*

![kitchen render](ai_015_001_cam_00_render.png)
![conference render](ai_047_009_cam_00_render.png)

![kitchen section](ai_015_001_cam_00_section.png)

*Kitchen: a 2 cm slice 75 cm above the floor, gray ground truth, one color per view; bottom:
1 m of the right-hand wall. Déjà View's views scatter over a few centimetres around the wall;
refined, they lie on one thin line, a few centimetres from the ground truth.*

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

Three scenes, one run; claims hold on all three unless a scene is named.

1. **With E_n in its units, no term dominates.** At the start E_n (4–11e-3) and E_bd
   (0.7–3.7e-3) are the largest terms, E_mv is 2.5–7e-4. E_n, E_bd and E_mv fall 2–28× during
   the solve; ‖δ‖² ends largest. All 10 Gauss-Newton steps are accepted on every scene.
2. **The views fuse.** Disagreement between overlapping views falls about tenfold (0.48 → 0.05 %,
   0.96 → 0.15 %, 0.60 → 0.09 %). In the cross sections the views' walls collapse onto one
   thin line, as on the week 2 slide.
3. **The fused cloud still scores lower than Déjà View's.** The F-score at 5 cm falls on every
   scene (0.910 → 0.881, 0.626 → 0.558, 0.777 → 0.698), through completeness (1.39 → 1.97,
   3.25 → 5.42, 2.43 → 4.08 cm) more than accuracy (kitchen: 3.13 → 3.09). The fused wall in the
   kitchen close-up sits a few centimetres from the ground truth, and the small objects in the
   middle of the kitchen slice look sparser. Most classes lose; the conference chairs gain
   (0.46 → 0.55) and the desk holds (0.91 → 0.92).
4. **Labels follow geometry.** mIoU falls by 0.06–0.12 and AP50 by 0.04–0.10, as labels
   transfer only within 5 cm of the ground truth.
5. **Conference room, table and lamps: the ground truth is behind the glass.** Hypersim labels
   the conference table *desk* (0.91 → 0.92). Its *table* and *lamps* are furniture in the next
   room, 10–28 m from the cameras, seen through the glass wall. Déjà View puts their pixels on
   the glass, at about half their true distance (median 0.51× for the table, 0.54× for the
   lamps; mostly 3.5–7 m), so no predicted point comes within 20 cm and both score 0 in either
   variant. A depth network sees glass as a surface; the ground truth sees what lies behind it.
