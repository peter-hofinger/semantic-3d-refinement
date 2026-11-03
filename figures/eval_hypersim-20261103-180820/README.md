# Evaluation protocol with pixel-aligned, threshold-free scores

Run `runs/eval_hypersim-20261103-180820` (`configs/eval_hypersim.yaml`, `configs/refine.yaml`;
the energy of `figures/eval_hypersim-20261103-155403`, scored with the new
`src/evaluation/labeled.py`). Same inputs, scenes and variants as that run; only the scoring
changed:

- **Alignment:** one similarity per output by Umeyama on its pixels with ground-truth depth,
  instead of a first-camera start and ICP on the clouds.
- **Surface:** accuracy and completeness (mean and median) on every pixel, without the
  1 cm grid, the 20 cm cap and the F-score at 5 cm.
- **Labels:** the output's label at each ground-truth pixel, instead of the nearest predicted
  point within 5 cm; mIoU, scene-level PQ and 3D instance AP / AP50 / AP25,
  without the 100-voxel minimum. Walls, floor and ceiling are one segment per class; mIoU and
  PQ are averaged over all classes and over object classes alone (`_th`, PQ_st for the
  background), AP over object classes. The surface scores cover every pixel.

```bash
uv run python -m src.analysis.scene_eval --cfg configs/eval_hypersim.yaml       # 1.5 min
uv run python -m src.analysis.scene_figures --run runs/eval_hypersim-<time>
```

## Results

| scene | variant | accuracy (cm) | median | completeness (cm) | median |
|---|---|---|---|---|---|
| living room | Déjà View | 2.19 | 1.20 | 1.01 | 0.75 |
| living room | Refined | 1.79 | 1.11 | 1.41 | 0.93 |
| bedroom | Déjà View | 2.52 | 1.78 | 1.17 | 0.81 |
| bedroom | Refined | 2.20 | 1.64 | 1.72 | 1.22 |
| lobby | Déjà View | 9.31 | 5.33 | 4.80 | 2.36 |
| lobby | Refined | 8.63 | 4.84 | 7.96 | 3.66 |

| scene | mIoU | mIoU_th | PQ | SQ | RQ | PQ_th | SQ_th | RQ_th | PQ_st | AP | AP50 | AP25 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| living room | 0.393 | 0.252 | 0.325 | 0.367 | 0.351 | 0.167 | 0.219 | 0.189 | 0.956 | 0.285 | 0.370 | 0.443 |
| bedroom | 0.434 | 0.353 | 0.297 | 0.424 | 0.337 | 0.187 | 0.337 | 0.220 | 0.916 | 0.297 | 0.382 | 0.480 |
| lobby | 0.528 | 0.419 | 0.375 | 0.566 | 0.436 | 0.204 | 0.458 | 0.247 | 0.889 | 0.193 | 0.278 | 0.419 |

- Pixel grids line up: shifting Déjà View's output by one pixel (±1 in u or v) raises the
  median per-pixel error after alignment in the living room (2.91 → 3.04-5.20 cm) and the
  bedroom (3.43 → 3.52-3.98 cm). In the lobby the error is dominated by the fit (16.93 cm),
  and two shifts lower it slightly (16.87, 15.63 cm). Half a pixel down in v lowers it in
  every scene (2.59, 3.37, 16.47 cm), within Déjà View's known ray-map deviation from a
  pinhole (0.6-0.9 px, `Claude.md`): an error of the method, scored as such.
- Label scores are the same for both variants: the labels are the linked SAM 3 masks, scored
  per pixel, and refinement does not change them.
- Background classes alone: IoU and PQ 0.85-0.98 (wall, floor, ceiling in every scene),
  so the all-class means sit 0.08-0.17 above the object-only ones.
- mIoU over only the object classes present in the ground truth would be 0.378 / 0.489 /
  0.699; the 4-5 classes per scene that SAM 3 finds but the scene lacks count as 0.
- Uncapped, the lobby's accuracy is 9.31 cm against 6.13 cm with the 20 cm cap: its outliers
  are far.
