# Evaluation protocol on dev and stress: splits, aggregates and tiers

Run `runs/eval_hypersim-20261110-093416` (`configs/eval_hypersim.yaml`, `configs/refine.yaml`):
the energy after E40 and E41 (Gauss-Newton 5 × 10, the consistency u counting the pixel's own
view; `experiments/LOG.md`). It replaces `eval_hypersim-20261105-104445` (10 × 20, u over the
other views only). Protocol: `README.md`.

```bash
uv run python -m src.analysis.scene_eval --cfg configs/eval_hypersim.yaml --splits dev stress   # ~1 min
uv run python -m src.analysis.scene_figures --run runs/eval_hypersim-<time>                     # 3.9 min
```

## Figures

- `presentation/`: only the bedroom `ai_006_007` and, wherever a figure shows views, exactly the
  views of issue #7, in its order: 18, 0, 12, 6 (frames 94, 0, 62, 31). The section and
  from-above figures and the renders' dollhouse show the whole scene, so they have no view to
  choose; the clouds behind them fuse all 20 views. Each interior render is its view's camera
  pulled back by up to 0.5 m and raised 0.2 m. The masks are SAM 3 with NYU40 prompts, as in
  the evaluation; E32's numbers for these views used the top100 prompts.
- `debug/`: the other five scenes at views 0 and 10, and the score summaries of each split.

## Results

Déjà View → Refined, cm, lower is better. The dev aggregate is the mean over its three scenes.

| split | scene | accuracy | median | completeness | median | Chamfer | median |
|---|---|---|---|---|---|---|---|
| dev | aggregate | 4.68 → 4.20 | 2.77 → 2.53 | 2.32 → 3.66 | 1.30 → 1.92 | 3.50 → 3.93 | 2.04 → 2.23 |
| dev | living room | 2.19 → 1.79 | 1.20 → 1.10 | 1.01 → 1.40 | 0.75 → 0.92 | 1.60 → 1.59 | 0.97 → 1.01 |
| dev | bedroom | 2.52 → 2.21 | 1.78 → 1.65 | 1.17 → 1.71 | 0.81 → 1.22 | 1.85 → 1.96 | 1.29 → 1.44 |
| dev | lobby | 9.31 → 8.60 | 5.33 → 4.83 | 4.80 → 7.87 | 2.36 → 3.63 | 7.05 → 8.23 | 3.85 → 4.23 |
| stress | kitchen | 2.98 → 2.77 | 1.75 → 1.66 | 1.35 → 1.98 | 0.77 → 1.25 | 2.17 → 2.37 | 1.26 → 1.45 |
| stress | conference room | 16.63 → 16.28 | 8.02 → 7.57 | 13.95 → 15.58 | 2.72 → 4.49 | 15.29 → 15.93 | 5.37 → 6.03 |
| stress | restaurant | 5.48 → 5.06 | 2.78 → 2.56 | 2.50 → 3.62 | 1.26 → 1.82 | 3.99 → 4.34 | 2.02 → 2.19 |

Labels, the same for both variants:

| split | scene | mIoU | PQ | SQ | RQ | PQ_th | PQ_st | AP | AP50 | AP25 | class-agnostic AP |
|---|---|---|---|---|---|---|---|---|---|---|---|
| dev | aggregate | 0.360 | 0.292 | 0.397 | 0.330 | 0.159 | 0.920 | 0.216 | 0.286 | 0.371 | 0.177 |
| dev | living room | 0.345 | 0.286 | 0.322 | 0.310 | 0.142 | 0.956 | 0.228 | 0.296 | 0.354 | 0.296 |
| dev | bedroom | 0.413 | 0.269 | 0.385 | 0.306 | 0.167 | 0.916 | 0.276 | 0.355 | 0.446 | 0.175 |
| dev | lobby | 0.457 | 0.321 | 0.485 | 0.373 | 0.167 | 0.889 | 0.145 | 0.208 | 0.314 | 0.060 |
| stress | kitchen | 0.348 | 0.221 | 0.358 | 0.261 | 0.167 | 0.582 | 0.388 | 0.559 | 0.586 | 0.072 |
| stress | conference room | 0.344 | 0.239 | 0.399 | 0.333 | 0.110 | 0.669 | 0.254 | 0.360 | 0.469 | 0.104 |
| stress | restaurant | 0.324 | 0.199 | 0.332 | 0.283 | 0.081 | 0.670 | 0.128 | 0.235 | 0.283 | 0.088 |

## Findings

- Refinement does not improve median Chamfer on any of the six scenes, and mean Chamfer only in
  the living room (1.60 → 1.59 cm). Accuracy improves on all six, completeness gets worse on all
  six and by more. Against the run this one replaces, every score moved by at most 0.09 cm
  (lobby completeness 7.96 → 7.87), all of it from the 5 × 10 budget: u with the own view
  changed no score by more than 0.007 cm (E41).
- Disagreement between overlapping views falls 5-8× (6-7× on dev), so the views fuse, but
  that does not lower Chamfer.
- The `views` figures show where it goes both ways: in the lobby refinement moves large
  regions closer and others further away (`debug/ai_044_001_cam_00_views.png`, last column). In the bedroom it lowers the error on the
  bed frame in views 18 and 12 but raises it by a few cm on the walls and door there, and lowers
  it on the far wall in view 6 (`presentation/ai_006_007_cam_00_views.png`, last column).
- Class-aware AP exceeds class-agnostic AP in five of six scenes (the living room is the
  exception). Class-aware AP weighs every class equally, class-agnostic AP every instance;
  whether that explains the gap has not been checked.
- The dev mIoU, pooled over 29 classes, is 0.360; the mean of its scenes would be 0.405. The
  difference has not been broken down by class.
- Refinement takes 1.9-2.5 s per scene on an RTX 3080 (3.5-4.5 s at 10 × 20).
- Scoring Déjà View's points takes about 4 s per scene, down from about 15 s (167 s to 5 s in
  the conference room, from cKDTree's median splits), and association about 2.5 s, down from
  about 14 s. On all six scenes the linked labels are identical and every earlier score
  agrees within 1e-9.
