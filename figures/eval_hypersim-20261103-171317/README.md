# Evaluation protocol with pixel-aligned, threshold-free scores

Run `runs/eval_hypersim-20261103-171317` (`configs/eval_hypersim.yaml`, `configs/refine.yaml`;
the energy of `figures/eval_hypersim-20261103-155403`, scored with the new
`src/evaluation/labeled.py`). Same inputs, scenes and variants as that run; only the scoring
changed:

- **Alignment:** one similarity per output by Umeyama on its pixels with ground-truth depth,
  instead of a first-camera start and ICP on the clouds.
- **Surface:** accuracy, completeness (mean and median) and Chamfer on every pixel, without the
  1 cm grid, the 20 cm cap and the F-score at 5 cm.
- **Labels:** the output's label at each ground-truth pixel, instead of the nearest predicted
  point within 5 cm; mIoU, scene-level PQ and ScanNet's AP / AP50 / AP25, without the
  100-voxel minimum. Walls, floor and ceiling are one segment per class, scored per class and
  left out of the label aggregates; the surface scores cover every pixel.

```bash
uv run python -m src.analysis.scene_eval --cfg configs/eval_hypersim.yaml       # 1.5 min
uv run python -m src.analysis.scene_figures --run runs/eval_hypersim-<time>
```

## Results

| scene | variant | accuracy (cm) | median | completeness (cm) | median | Chamfer (cm) |
|---|---|---|---|---|---|---|
| living room | Déjà View | 2.19 | 1.20 | 1.01 | 0.75 | 1.60 |
| living room | Refined | 1.79 | 1.11 | 1.41 | 0.93 | 1.60 |
| bedroom | Déjà View | 2.52 | 1.78 | 1.17 | 0.81 | 1.85 |
| bedroom | Refined | 2.20 | 1.64 | 1.72 | 1.22 | 1.96 |
| lobby | Déjà View | 9.31 | 5.33 | 4.80 | 2.36 | 7.05 |
| lobby | Refined | 8.63 | 4.84 | 7.96 | 3.66 | 8.30 |

| scene | mIoU | PQ | SQ | RQ | AP | AP50 | AP25 |
|---|---|---|---|---|---|---|---|
| living room | 0.252 | 0.167 | 0.219 | 0.189 | 0.285 | 0.370 | 0.443 |
| bedroom | 0.353 | 0.187 | 0.337 | 0.220 | 0.297 | 0.382 | 0.480 |
| lobby | 0.419 | 0.204 | 0.458 | 0.247 | 0.193 | 0.278 | 0.419 |

- Label scores are the same for both variants: the labels are the linked SAM 3 masks, scored
  per pixel, and refinement does not change them.
- Background classes alone: IoU and PQ 0.85-0.98 (wall, floor, ceiling in every scene).
- mIoU over only the object classes present in the ground truth would be 0.378 / 0.489 /
  0.699; the 4-5 classes per scene that SAM 3 finds but the scene lacks count as 0.
- Uncapped, the lobby's accuracy is 9.31 cm against 6.13 cm with the 20 cm cap: its outliers
  are far.
