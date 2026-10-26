# bini_blur_office-20261026-151852

git `45ec80bfe588c09b0441fa6f925dc71e8475eb51`

| metric | value |
|---|---|
| views | 1 |
| normals_seconds | 0.1308 |
| refine_seconds | 9.976 |
| scale_to_m | 1.76 |
| camera_ate_m | 0.008577 |
| mean_point_shift_mm | 19.37 |
| angle_to_normals_deg_input | 9.746 |
| angle_to_normals_deg_refined | 5.042 |


## Findings (looked at every figure)

- Single view (office view 6), ~10 s per setting. Every setting pulls the normals toward
  DSINE's (see the angles in compare.png), and walls take DSINE's orientation.
- The 14 px patch grid survives every setting, blurred prior included. BiNI starts from Déjà View's depth, whose
  patch seams are small depth steps; its discontinuity weights read them as edges and keep the surface broken
  there. No prior setting can remove that: the weights would have to start from DSINE's normals instead.
- Staircase artifacts at occlusion edges (monitors, table legs) appear in every setting and get slightly worse
  with sigma > 0, where Déjà View and DSINE disagree on where the edge is.
- Blurring the prior buys no visible detail over plain BiNI with a lower lambda (sigma = 0, lambda = 100).

## compare

One view under several BiNI prior settings (sigma = blur of the depth prior in px, λ = its weight); bottom row zoomed where DSINE has the most detail. In brackets: median angle to DSINE's normals.

![compare](compare.png)

## normal_sources

Normals per view: Déjà View's depth, DSINE's prediction (rotated into the same frame), and the depth after BiNI integrates DSINE's normals against Déjà View's depth.

![normal_sources](normal_sources.png)

## detail_crops

The same three, zoomed where DSINE's normals carry the most detail.

![detail_crops](detail_crops.png)
