# bini_dsine_kitchen-20261026-141933

git `24a5f76963becc2042b188e3e4e397a197231b03`

| metric | value |
|---|---|
| views | 20 |
| normals_seconds | 4.444 |
| refine_seconds | 18.78 |
| scale_to_m | 1.048 |
| camera_ate_m | 0.007364 |
| mean_point_shift_mm | 3.809 |
| mv_median_input | 0.01046 |
| mv_median_refined | 0.01022 |
| mv_inlier_1pct_input | 0.4685 |
| mv_inlier_1pct_refined | 0.4844 |


## Findings (looked at every figure)

- DSINE (hugoycj/DSINE-hub, given Déjà View's intrinsics) gives clean normals: no 14 px patch grid, crisp
  object edges. Its normals sit a median 11.7° from Déjà View's (office); a per-view rotation explains only
  2-6° of that, so the disagreement is local.
- BiNI's default depth weight (λ = 1e-4) lets integration drift: cross-view median 17% (office), 0.4 m
  mean point shift. The surface only stays near the input from λ ≈ 100 up (bini_sweep).
- At λ = 1000 (picked on these two scenes, so optimistic) cross-view agreement improves slightly:
  median 1.046% → 1.022%, within 1% 46.9% → 48.4%, mean point shift 3.8 mm.
- But the detail doesn't transfer. At that λ the BiNI normals look like Déjà View's: the patch grid stays,
  DSINE's smooth surfaces don't come through, and occlusion edges get staircase artifacts where the two
  disagree on where the edge is (detail_crops). A uniform depth prior can't take high frequencies from the
  normals and low frequencies from the depth at once: weak anchors drift, strong anchors ignore the normals.
- Next: anchor only the low frequencies (prior on blurred log depth), so the normals set the detail.

## normal_sources

Normals per view: Déjà View's depth, DSINE's prediction (rotated into the same frame), and the depth after BiNI integrates DSINE's normals against Déjà View's depth.

![normal_sources](normal_sources.png)

## detail_crops

The same three, zoomed where DSINE's normals carry the most detail.

![detail_crops](detail_crops.png)

## bini_sweep

Cross-view disagreement over all views versus the depth-prior weight λ and the discontinuity sharpness k, one at a time (circle: this run). Dashed: Déjà View input.

![bini_sweep](bini_sweep.png)

## normals

Surface normals from the point maps (world frame as RGB): noise and edges show here first.

![normals](normals.png)

## multiview_maps

Cross-view disagreement per pixel: median over the other views of |x - x_j| / D_j under the frozen predicted cameras. A GT-free proxy; occlusions put a floor under it.

![multiview_maps](multiview_maps.png)

## multiview_per_view

Median cross-view disagreement per view, before and after.

![multiview_per_view](multiview_per_view.png)
