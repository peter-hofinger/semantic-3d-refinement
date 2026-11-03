# Evaluation protocol: Déjà View against the refined cloud on three Hypersim scenes

Run `runs/eval_hypersim-20261103-150543` (`configs/eval_hypersim.yaml`, `configs/refine.yaml`;
git `1d86004` plus the uncommitted changes of the commit that adds this report). Reproduce:

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

Each against the baseline (the previous commit, `1d86004`), one change at a time, rule: remove
a part if both the scores and the renders improve or hold. F-score at 5 cm:

| variant | kitchen | conference | restaurant | renders (figures below) | decision |
|---|---|---|---|---|---|
| baseline | 0.885 | 0.558 | 0.719 | | |
| no a, b, no δ penalty | 0.917 | 0.625 | 0.777 | – | invalid: the solve stops after 0–2 steps |
| no a, b, δ penalty kept | **0.891** | 0.556 | **0.731** | as the baseline | **a, b and `min_pixels` removed** |
| a, b for every instance (`min_pixels` 1) | 0.884 | 0.558 | 0.722 | as the baseline | moot once a, b go |
| no occlusion cut (`edge_tol` ∞) | 0.829 | 0.549 | 0.656 | ramps and smears across depth jumps | kept |
| no `SCALE`, λ_n = 1e-4 | 0.894 | 0.558 | 0.714 | as the baseline | **`SCALE` removed** (the same energy) |
| per-edge cut, no window or closing | 0.896 | 0.561 | 0.736 | more stray points, the kitchen's arch in pieces | kept |
| one residual per edge (n_p only) | 0.886 | 0.557 | 0.714 | as the baseline | kept (no gain either way) |

- **The δ penalty cannot go.** 8–10 % of the pixels have E_d's weight 0 (no other view
  agrees on their instance), and 1,300–2,700 of them per scene have no normal edge or match
  either: without ‖δ‖² nothing holds them, the normal equations are singular there, and every
  step is rejected by the positivity guard. The "no δ penalty" scores are Déjà View's.
- **Without a, b, a correction reaches only a few pixels past the matches.** On the two-view
  test scene, unmatched pixels within 2 px of a match move from 2.9 % to 2.2 % error, farther
  ones keep the prediction. On Hypersim this costs nothing measurable.
- **Noise.** The solver is not bitwise reproducible on the GPU (atomic sums: 30 % of pixels
  differ in the last bits, a few by up to 16 %), and the registration of the conference room
  amplifies that: the final configuration scored 0.556 in the ablation run and 0.596 in this
  one, from identical code (registration scale 3.852 vs 3.846). Differences below about 0.04 in
  the conference room and about 0.005 in the other two are noise; the decisions rest on the
  kitchen, the restaurant and the renders.

## Results

| scene | variant | accuracy (cm) | completeness (cm) | precision | recall | F-score | mIoU | AP50 | disagreement (%) | GN steps | s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| kitchen | Déjà View | 3.13 | 1.39 | 0.851 | 0.977 | 0.910 | 0.577 | 0.535 | 0.48 | – | – |
| kitchen | Refined | 3.05 | 1.97 | 0.849 | 0.933 | 0.889 | 0.571 | 0.535 | 0.06 | 10 | 3.5 |
| conference | Déjà View | 7.94 | 3.25 | 0.495 | 0.852 | 0.626 | 0.505 | 0.304 | 0.96 | – | – |
| conference | Refined | 8.08 | 4.25 | 0.494 | 0.751 | 0.596 | 0.457 | 0.300 | 0.19 | 10 | 3.8 |
| restaurant | Déjà View | 4.80 | 2.43 | 0.687 | 0.895 | 0.777 | 0.547 | 0.210 | 0.60 | – | – |
| restaurant | Refined | 4.80 | 3.52 | 0.677 | 0.794 | 0.730 | 0.500 | 0.155 | 0.10 | 10 | 3.9 |

F-score at 5 cm per class, Déjà View → Refined:

| class | kitchen | conference | restaurant |
|---|---|---|---|
| wall | 0.94 → 0.92 | 0.70 → 0.70 | 0.74 → 0.66 |
| floor | 0.99 → 0.99 | 0.88 → 0.87 | 0.95 → 0.94 |
| chair | 0.89 → 0.88 | 0.46 → 0.45 | 0.75 → 0.69 |
| table | 0.92 → 0.93 | 0.00 → 0.00 (finding 4) | 0.81 → 0.78 |
| lamp | 0.87 → 0.83 | 0.00 → 0.00 (finding 4) | 0.76 → 0.70 |

Energy at the start and after 10 steps (means over all pixels):

| scene | E_d | E_mv | λ_n E_n | ‖δ‖² |
|---|---|---|---|---|
| kitchen | 0 → 1.2e-05 | 2.5e-04 → 5.4e-05 | 4.2e-03 → 2.7e-04 | 0 → 3.7e-04 |
| conference | 0 → 3.4e-05 | 7.0e-04 → 1.7e-04 | 3.1e-03 → 2.0e-04 | 0 → 3.0e-04 |
| restaurant | 0 → 2.1e-05 | 3.3e-04 → 1.2e-04 | 8.1e-03 → 6.9e-04 | 0 → 6.2e-04 |

## Figures

![restaurant ablations](ai_052_001_cam_00_ablations.png)
![kitchen ablations](ai_015_001_cam_00_ablations.png)

*Every ablation of the table above (bar the invalid one and `SCALE`, the same energy) from
the same camera, image colors and gray shading. Removing the occlusion cut smears surfaces
across depth jumps; cutting edge by edge leaves more stray points and breaks the kitchen's
arch into pieces; the others look as the baseline.*

![summary](summary.png)

*Every score of every variant; dashed: Déjà View per scene.*

![restaurant render](ai_052_001_cam_00_render.png)

*Restaurant: each cloud (all 20 views' pixels) rendered from the same two cameras, both fixed
by the ground truth: a dollhouse view from above without the ceiling, and the view that sees
deepest into the room, pulled back (`src/analysis/render.py`).*

![kitchen render](ai_015_001_cam_00_render.png)
![conference render](ai_047_009_cam_00_render.png)

![kitchen section](ai_015_001_cam_00_section.png)

*Kitchen: a 2 cm slice 75 cm above the floor, gray ground truth, one color per view; bottom:
1 m of the right-hand wall.*

![conference section](ai_047_009_cam_00_section.png)
![restaurant section](ai_052_001_cam_00_section.png)

![restaurant energy](ai_052_001_cam_00_energy.png)

*Restaurant, view 5: each term's weighted cost per pixel on Déjà View's depth (top) and the
refined depth (bottom), log scale over four decades below each term's 99.5th percentile.
Second column: the consistency u that gates E_d.*

![kitchen energy](ai_015_001_cam_00_energy.png)
![conference energy](ai_047_009_cam_00_energy.png)

![restaurant normals](ai_052_001_cam_00_normals.png)
![kitchen normals](ai_015_001_cam_00_normals.png)
![conference normals](ai_047_009_cam_00_normals.png)

![kitchen classes](ai_015_001_cam_00_classes.png)
![conference classes](ai_047_009_cam_00_classes.png)
![restaurant classes](ai_052_001_cam_00_classes.png)

*F-score at 5 cm and IoU per class with at least 100 ground-truth voxels.*

![kitchen views](ai_015_001_cam_00_views.png)
![kitchen from above](ai_015_001_cam_00_topdown.png)
![conference views](ai_047_009_cam_00_views.png)
![conference from above](ai_047_009_cam_00_topdown.png)
![restaurant views](ai_052_001_cam_00_views.png)
![restaurant from above](ai_052_001_cam_00_topdown.png)

## Findings

Three scenes, one run each; claims hold on all three unless a scene is named.

1. **Simpler and no worse.** Without a, b, `min_pixels` and `SCALE`, the kitchen and the
   restaurant score 0.889 and 0.730 (baseline 0.885 and 0.719); kitchen mIoU and AP50 rise
   to 0.571 and 0.535 (from 0.536 and 0.493).
2. **The views fuse.** Disagreement between overlapping views falls 5–8× (0.48 → 0.06 %,
   0.96 → 0.19 %, 0.60 → 0.10 %).
3. **The refined cloud still scores lower than Déjà View's**, through completeness (1.39 →
   1.97, 3.25 → 4.25, 2.43 → 3.52 cm); accuracy holds or improves in the kitchen and the
   restaurant (3.13 → 3.05, 4.80 → 4.80 cm).
4. **Conference room, table and lamps: the ground truth is behind the glass.** Hypersim labels
   the conference table *desk*; its *table* and *lamps* are furniture in the next room, 10–28 m
   from the cameras, seen through the glass wall, which Déjà View puts on the glass at about
   half their distance. Both score 0 in either variant.
