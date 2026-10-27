# Experiment log: SAM 3 for cross-view instance masks (branch `sam3`)

Every experiment gets an entry: goal, code version, command, settings, results, figures, conclusion.
Raw script output is in `results/`, figures in `figures/`, one-off scripts in `scripts/`.
Data, caches and model outputs stay local under `data/processed/scannetpp/<scene>/` (git-ignored).

**Common setup** unless stated otherwise:
- Scenes (ScanNet++ `nvs_sem_val`, held out from Déjà View training): `5ee7c22ba0` kitchen, `7b6477cb95` office.
- Views: 20 DSLR frames per scene, uniform over capture order, resized to 504x336
  (`uv run scannetpp-prepare`, `configs/scannetpp.yaml`), GT poses. Camera step between consecutive views:
  kitchen median 0.30 m (max 1.57), office median 0.92 m (max 1.95).
- SAM 3: `facebook/sam3` via `transformers 5.20.2`, bf16, score threshold 0.5, mask threshold 0.5.
- Prompts (16, hand-picked frequent ScanNet++ classes, `PROMPTS` in `src/segmentation/sam3.py`):
  wall, floor, ceiling, door, window, table, chair, monitor, keyboard, cabinet, shelf, whiteboard, box, lamp,
  sink, refrigerator.
- Hardware: laptop RTX 3050 Laptop GPU (4 GB), Windows 11, torch 2.14.1+cu126.
- Scorer: `uv run eval-instances <views.npz> <segmentation.npz>` (`src/evaluation/instances.py`), see E11 for the
  current version. Per view, a predicted mask matches a GT instance at IoU >= 0.5. GT instances smaller than
  0.2% of the image are not scored. "objects" = without wall/floor/ceiling.

Metrics:
- recall / precision: matched (view, instance) pairs over GT / predicted pairs.
- pixel recall: GT area of matched instances over scored GT area.
- leakage: pixels of each predicted mask outside the GT instance it mostly covers; cross-class leakage: the
  part on a GT object of another class (smoothing within such a mask would cross a real boundary).
- consistent: share of GT objects matched in >= 2 views that keep a single predicted id.
- id switches: sum over GT objects of (distinct predicted ids - 1). merges: predicted ids matched to > 1 GT object.

| id | experiment | result in one line |
|---|---|---|
| E1 | GT instance masks + depth by mesh raycasting | GT for 2D evaluation; aligns visually with the images |
| E2 | SAM 3 image mode, one view | fits in 4 GB (2.1 GB peak), 3 s/view, good masks, floor missed |
| E3 | SAM 3 video mode GPU memory | 1 prompt per session fits; >= 4 prompts OOM; Windows stalls instead of OOM |
| E4 | SAM 3 video mode on the 20 sparse views | only ~half of objects keep one id; look-alike objects merged |
| E5 | Decision: no dense-sequence video mode | would require video input |
| E6 | Association sanity check (GT masks, GT geometry) | matcher recovers ids: consistent 1.00 / 0.93, 0 merges |
| E7 | SAM 3 image mode + association, GT geometry | consistent 0.79 / 0.77, merges 1 / 5 (upper bound) |
| E8 | Déjà View on the laptop | fits (3.25 GB), per-view depth error < 1% after scale |
| E9 | SAM 3 image mode + association, Déjà View geometry | ~ same as E7: consistent 0.75 / 0.77, merges 1 / 4 |
| E10 | Recall breakdown per class | pixel recall 0.84 / 0.78; misses: unprompted classes, carpet floor, small things |
| E11 | Scorer v2/v3: pixel recall, leakage, unlabeled handling | office cross-class leakage 4.5%, kitchen 0.5% |
| E12 | Prompt vocabulary arm B: ScanNet++ top100 (+ floor synonyms) | pixel recall 0.92 / 0.79 (from 0.84 / 0.78), consistency drops to 0.60 / 0.51 (near-synonym classes); carpet/rug never fire |
| E13 | Association across classes (strong 3D overlap) | consistent top100 0.60 / 0.51 -> 0.83 / 0.76 at 0.7; kitchen links all right, office: 1 chair->monitor error, cabinet confusions |
| E14 | Plan step (a): link only if >= f of a mask's in-image pixels are visible | chair -> monitor link gone; f = 0.25: office merges 5 -> 4, wrong cross-class links 8 -> 3, kitchen consistent 0.86 -> 0.79 (stacked boxes); f = 0.5 too strict |
| E15 | Plan step (b): threshold 0.3, masks <= 0.5 kept only if linked to another view | pixel recall +0.03 to +0.05 at same precision / leakage / consistency; office floor 3/18 -> 9/18 views; kept low masks 54-100% real, dropped 2-10% |
| E16 | Plan step (c), arm C: VLM vocabulary (Qwen3-VL-2B, 4-bit, one list per scene) | about equal to top100 (pixel recall 0.94 / 0.83 vs 0.95 / 0.83), precision -0.06 / -0.08, office +1 merge; finds pedestal fan 1 -> 8/8, coat rack 5 -> 8/9; loses thin frames/sills |
| E17 | Grid (a) x (b) x vocabulary, 8 setups x 2 scenes | (b) helps everywhere (pixel recall +0.02 / +0.04, precision +0.01) but adds an office merge unless (a) is on; (a) removes 1-2 office merges, costs kitchen consistency -0.03; VLM precision -0.06 / -0.08. Best: top100 + a + b (= working setting) |
| E18 | Boundaries of SAM 3 masks vs depth edges and Déjà View bleeding (GT: 5 cm mesh) | SAM 3 boundaries match GT object boundaries (F 0.66 / 0.64 at 1 px, 0.80 / 0.79 at 3 px; control 0.17-0.25); bleeding exists on ScanNet++ (error 3-4% at depth edges vs 0.5-0.7% far). At the same density Déjà View's own depth edges find depth edges better than SAM 3; SAM 3 adds which object a pixel belongs to: bleeding pixels in the right mask 0.72 kitchen, only 0.39 office (0.23 wrong object) |
| E19 | Arm D: top100 + VLM prompts in one SAM 3 run | recall about as top100 (pixel recall 0.94 / 0.84 vs 0.95 / 0.83), precision -0.11 / -0.15, office merges 4 -> 5 (VLM's fan merge); bleeding pixels in the right object office 0.39 -> 0.48 (VLM alone 0.46), kitchen 0.72 -> 0.74, all from pedestal fan + coat rack. Not adopted as default; the VLM's value is objects outside a fixed list |
| E20 | Proxy test: can segments fix bleeding? (local plane fit within a segment near Déjà View's depth jumps) | without segments (or shuffled) the fix is harmful (share > 5% 0.21 -> 0.39 / 0.23 -> 0.52); SAM 3 makes it safe but nearly neutral (mean error -2% / -2%); GT segments -10% / -19%. SAM 3 helps on right-object pixels, hurts on wrong-object ones, leaves no-mask pixels (24% office) unchanged |
| E21 | Class-agnostic SAM 2 automatic masks fill the pixels without a SAM 3 mask (option B) | no gain: automatic masks cover only 33% / 22% of the no-mask pixels, the E20 proxy fix changes 7-11% of them, results = next-view control (realistic mean error 0.046 / 0.068 as SAM 3 alone). Gaps are mostly slivers next to a mask (63-82% within 3 px) plus unprompted objects (pedestal fan) |

## Reference table (current scorer, E11)

`uv run python experiments/scripts/score_table.py` -> `results/score_table.txt`

| scene | method | subset | recall | pixel recall | precision | leakage | cross-class leakage | consistent | GT in >=2 views | id switches | merges |
|---|---|---|---|---|---|---|---|---|---|---|---|
| kitchen | video mode (E4) | all | 0.62 | 0.87 | 0.66 | 0.012 | 0.007 | 0.50 | 26 | 31 | 13 |
| kitchen | video mode (E4) | objects | 0.53 | 0.72 | 0.53 | 0.015 | 0.011 | 0.61 | 18 | 10 | 3 |
| kitchen | image, no association (E6) | all | 0.64 | 0.84 | 0.83 | 0.009 | 0.005 | 0.00 | 28 | 166 | 0 |
| kitchen | image, no association (E6) | objects | 0.59 | 0.71 | 0.75 | 0.014 | 0.011 | 0.00 | 22 | 99 | 0 |
| kitchen | image + assoc GT geometry (E7) | all | 0.64 | 0.84 | 0.83 | 0.009 | 0.005 | 0.79 | 28 | 7 | 1 |
| kitchen | image + assoc GT geometry (E7) | objects | 0.59 | 0.71 | 0.75 | 0.014 | 0.011 | 0.77 | 22 | 6 | 1 |
| kitchen | image + assoc Deja View (E9) | all | 0.64 | 0.84 | 0.83 | 0.009 | 0.005 | 0.75 | 28 | 8 | 1 |
| kitchen | image + assoc Deja View (E9) | objects | 0.59 | 0.71 | 0.75 | 0.014 | 0.011 | 0.73 | 22 | 7 | 1 |
| kitchen | image top100 + assoc Deja View (E12) | all | 0.84 | 0.92 | 0.49 | 0.010 | 0.007 | 0.60 | 35 | 17 | 1 |
| kitchen | image top100 + assoc Deja View (E12) | objects | 0.88 | 0.96 | 0.41 | 0.016 | 0.014 | 0.55 | 29 | 16 | 1 |
| kitchen | image + assoc Deja View, cross-class 0.7 (E13) | all | 0.64 | 0.84 | 0.83 | 0.009 | 0.005 | 0.86 | 28 | 4 | 1 |
| kitchen | image + assoc Deja View, cross-class 0.7 (E13) | objects | 0.59 | 0.71 | 0.75 | 0.014 | 0.011 | 0.86 | 22 | 3 | 1 |
| kitchen | image + assoc Deja View, cross-class 0.5 (E13) | all | 0.64 | 0.84 | 0.83 | 0.009 | 0.005 | 0.86 | 28 | 4 | 1 |
| kitchen | image + assoc Deja View, cross-class 0.5 (E13) | objects | 0.59 | 0.71 | 0.75 | 0.014 | 0.011 | 0.86 | 22 | 3 | 1 |
| kitchen | image-top100 + assoc Deja View, cross-class 0.7 (E13) | all | 0.84 | 0.92 | 0.49 | 0.010 | 0.007 | 0.83 | 35 | 6 | 1 |
| kitchen | image-top100 + assoc Deja View, cross-class 0.7 (E13) | objects | 0.88 | 0.96 | 0.41 | 0.016 | 0.014 | 0.83 | 29 | 5 | 1 |
| kitchen | image-top100 + assoc Deja View, cross-class 0.5 (E13) | all | 0.84 | 0.92 | 0.49 | 0.010 | 0.007 | 0.86 | 35 | 5 | 1 |
| kitchen | image-top100 + assoc Deja View, cross-class 0.5 (E13) | objects | 0.88 | 0.96 | 0.41 | 0.016 | 0.014 | 0.86 | 29 | 4 | 1 |
| kitchen | image + assoc Deja View, cross-class 0.7, visible >= 0.5 (E14) | all | 0.64 | 0.84 | 0.83 | 0.009 | 0.005 | 0.64 | 28 | 11 | 1 |
| kitchen | image + assoc Deja View, cross-class 0.7, visible >= 0.5 (E14) | objects | 0.59 | 0.71 | 0.75 | 0.014 | 0.011 | 0.59 | 22 | 10 | 1 |
| kitchen | image + assoc Deja View, cross-class 0.7, visible >= 0.25 (E14) | all | 0.64 | 0.84 | 0.83 | 0.009 | 0.005 | 0.79 | 28 | 7 | 1 |
| kitchen | image + assoc Deja View, cross-class 0.7, visible >= 0.25 (E14) | objects | 0.59 | 0.71 | 0.75 | 0.014 | 0.011 | 0.77 | 22 | 6 | 1 |
| kitchen | image-top100 + assoc Deja View, cross-class 0.7, visible >= 0.5 (E14) | all | 0.84 | 0.92 | 0.49 | 0.010 | 0.007 | 0.63 | 35 | 15 | 1 |
| kitchen | image-top100 + assoc Deja View, cross-class 0.7, visible >= 0.5 (E14) | objects | 0.88 | 0.96 | 0.41 | 0.016 | 0.014 | 0.59 | 29 | 14 | 1 |
| kitchen | image-top100 + assoc Deja View, cross-class 0.7, visible >= 0.25 (E14) | all | 0.84 | 0.92 | 0.49 | 0.010 | 0.007 | 0.80 | 35 | 8 | 1 |
| kitchen | image-top100 + assoc Deja View, cross-class 0.7, visible >= 0.25 (E14) | objects | 0.88 | 0.96 | 0.41 | 0.016 | 0.014 | 0.79 | 29 | 7 | 1 |
| kitchen | image t0.3 + assoc Deja View, xc 0.7, vf 0.25, score > 0.5 (E15) | all | 0.64 | 0.84 | 0.83 | 0.009 | 0.005 | 0.79 | 28 | 7 | 1 |
| kitchen | image t0.3 + assoc Deja View, xc 0.7, vf 0.25, score > 0.5 (E15) | objects | 0.59 | 0.71 | 0.75 | 0.014 | 0.011 | 0.77 | 22 | 6 | 1 |
| kitchen | image t0.3 + assoc Deja View, xc 0.7, vf 0.25, no filter (E15) | all | 0.75 | 0.89 | 0.68 | 0.009 | 0.006 | 0.74 | 31 | 9 | 1 |
| kitchen | image t0.3 + assoc Deja View, xc 0.7, vf 0.25, no filter (E15) | objects | 0.70 | 0.80 | 0.58 | 0.015 | 0.012 | 0.70 | 23 | 8 | 1 |
| kitchen | image t0.3 + assoc Deja View, xc 0.7, vf 0.25, multi-view filter (E15) | all | 0.74 | 0.89 | 0.81 | 0.009 | 0.006 | 0.81 | 31 | 7 | 1 |
| kitchen | image t0.3 + assoc Deja View, xc 0.7, vf 0.25, multi-view filter (E15) | objects | 0.69 | 0.80 | 0.73 | 0.014 | 0.011 | 0.78 | 23 | 6 | 1 |
| kitchen | image-top100 t0.3 + assoc Deja View, xc 0.7, vf 0.25, score > 0.5 (E15) | all | 0.84 | 0.92 | 0.49 | 0.010 | 0.007 | 0.80 | 35 | 8 | 1 |
| kitchen | image-top100 t0.3 + assoc Deja View, xc 0.7, vf 0.25, score > 0.5 (E15) | objects | 0.88 | 0.96 | 0.41 | 0.016 | 0.014 | 0.79 | 29 | 7 | 1 |
| kitchen | image-top100 t0.3 + assoc Deja View, xc 0.7, vf 0.25, no filter (E15) | all | 0.92 | 0.95 | 0.35 | 0.011 | 0.008 | 0.79 | 38 | 9 | 1 |
| kitchen | image-top100 t0.3 + assoc Deja View, xc 0.7, vf 0.25, no filter (E15) | objects | 0.94 | 0.97 | 0.28 | 0.018 | 0.015 | 0.77 | 30 | 8 | 1 |
| kitchen | image-top100 t0.3 + assoc Deja View, xc 0.7, vf 0.25, multi-view filter (E15) | all | 0.91 | 0.95 | 0.50 | 0.010 | 0.007 | 0.82 | 38 | 8 | 1 |
| kitchen | image-top100 t0.3 + assoc Deja View, xc 0.7, vf 0.25, multi-view filter (E15) | objects | 0.94 | 0.97 | 0.42 | 0.017 | 0.015 | 0.80 | 30 | 7 | 1 |
| kitchen | image-vlm t0.3 + assoc Deja View, xc 0.7, vf 0.25, score > 0.5 (E16) | all | 0.83 | 0.93 | 0.43 | 0.010 | 0.007 | 0.82 | 34 | 7 | 1 |
| kitchen | image-vlm t0.3 + assoc Deja View, xc 0.7, vf 0.25, score > 0.5 (E16) | objects | 0.85 | 0.94 | 0.35 | 0.016 | 0.013 | 0.82 | 28 | 6 | 1 |
| kitchen | image-vlm t0.3 + assoc Deja View, xc 0.7, vf 0.25, multi-view filter (E16) | all | 0.89 | 0.94 | 0.44 | 0.010 | 0.007 | 0.83 | 36 | 7 | 1 |
| kitchen | image-vlm t0.3 + assoc Deja View, xc 0.7, vf 0.25, multi-view filter (E16) | objects | 0.90 | 0.96 | 0.35 | 0.016 | 0.013 | 0.82 | 28 | 6 | 1 |
| kitchen | image-top100 t0.3 + assoc Deja View, xc 0.7, no vf, score > 0.5 (E17) | all | 0.84 | 0.92 | 0.49 | 0.010 | 0.007 | 0.83 | 35 | 6 | 1 |
| kitchen | image-top100 t0.3 + assoc Deja View, xc 0.7, no vf, score > 0.5 (E17) | objects | 0.88 | 0.96 | 0.41 | 0.016 | 0.014 | 0.83 | 29 | 5 | 1 |
| kitchen | image-top100 t0.3 + assoc Deja View, xc 0.7, no vf, multi-view filter (E17) | all | 0.91 | 0.95 | 0.50 | 0.010 | 0.007 | 0.84 | 38 | 6 | 1 |
| kitchen | image-top100 t0.3 + assoc Deja View, xc 0.7, no vf, multi-view filter (E17) | objects | 0.94 | 0.97 | 0.42 | 0.017 | 0.015 | 0.83 | 30 | 5 | 1 |
| kitchen | image-vlm t0.3 + assoc Deja View, xc 0.7, no vf, score > 0.5 (E17) | all | 0.83 | 0.93 | 0.43 | 0.010 | 0.007 | 0.85 | 34 | 5 | 1 |
| kitchen | image-vlm t0.3 + assoc Deja View, xc 0.7, no vf, score > 0.5 (E17) | objects | 0.85 | 0.94 | 0.35 | 0.016 | 0.013 | 0.86 | 28 | 4 | 1 |
| kitchen | image-vlm t0.3 + assoc Deja View, xc 0.7, no vf, multi-view filter (E17) | all | 0.89 | 0.94 | 0.44 | 0.010 | 0.007 | 0.86 | 36 | 5 | 1 |
| kitchen | image-vlm t0.3 + assoc Deja View, xc 0.7, no vf, multi-view filter (E17) | objects | 0.90 | 0.96 | 0.35 | 0.016 | 0.013 | 0.86 | 28 | 4 | 1 |
| kitchen | image-top100+vlm t0.3 + assoc Deja View, xc 0.7, vf 0.25, score > 0.5 (E19) | all | 0.85 | 0.93 | 0.38 | 0.010 | 0.007 | 0.80 | 35 | 8 | 1 |
| kitchen | image-top100+vlm t0.3 + assoc Deja View, xc 0.7, vf 0.25, score > 0.5 (E19) | objects | 0.88 | 0.96 | 0.30 | 0.017 | 0.014 | 0.79 | 29 | 7 | 1 |
| kitchen | image-top100+vlm t0.3 + assoc Deja View, xc 0.7, vf 0.25, multi-view filter (E19) | all | 0.91 | 0.94 | 0.39 | 0.011 | 0.007 | 0.84 | 38 | 7 | 1 |
| kitchen | image-top100+vlm t0.3 + assoc Deja View, xc 0.7, vf 0.25, multi-view filter (E19) | objects | 0.93 | 0.97 | 0.31 | 0.018 | 0.015 | 0.83 | 30 | 6 | 1 |
| office | video mode (E4) | all | 0.59 | 0.80 | 0.60 | 0.080 | 0.047 | 0.47 | 36 | 37 | 20 |
| office | video mode (E4) | objects | 0.57 | 0.80 | 0.54 | 0.083 | 0.047 | 0.47 | 30 | 31 | 16 |
| office | image, no association (E6) | all | 0.58 | 0.78 | 0.71 | 0.077 | 0.045 | 0.00 | 39 | 217 | 0 |
| office | image, no association (E6) | objects | 0.57 | 0.80 | 0.67 | 0.098 | 0.049 | 0.00 | 32 | 174 | 0 |
| office | image + assoc GT geometry (E7) | all | 0.58 | 0.78 | 0.71 | 0.077 | 0.045 | 0.77 | 39 | 13 | 5 |
| office | image + assoc GT geometry (E7) | objects | 0.57 | 0.80 | 0.67 | 0.098 | 0.049 | 0.75 | 32 | 11 | 4 |
| office | image + assoc Deja View (E9) | all | 0.58 | 0.78 | 0.71 | 0.077 | 0.045 | 0.77 | 39 | 13 | 4 |
| office | image + assoc Deja View (E9) | objects | 0.57 | 0.80 | 0.67 | 0.098 | 0.049 | 0.75 | 32 | 11 | 3 |
| office | image top100 + assoc Deja View (E12) | all | 0.66 | 0.79 | 0.46 | 0.082 | 0.047 | 0.51 | 49 | 36 | 5 |
| office | image top100 + assoc Deja View (E12) | objects | 0.68 | 0.83 | 0.42 | 0.104 | 0.052 | 0.45 | 42 | 34 | 4 |
| office | image + assoc Deja View, cross-class 0.7 (E13) | all | 0.58 | 0.78 | 0.71 | 0.077 | 0.045 | 0.77 | 39 | 10 | 5 |
| office | image + assoc Deja View, cross-class 0.7 (E13) | objects | 0.57 | 0.80 | 0.67 | 0.098 | 0.049 | 0.75 | 32 | 8 | 4 |
| office | image + assoc Deja View, cross-class 0.5 (E13) | all | 0.58 | 0.78 | 0.71 | 0.077 | 0.045 | 0.77 | 39 | 10 | 5 |
| office | image + assoc Deja View, cross-class 0.5 (E13) | objects | 0.57 | 0.80 | 0.67 | 0.098 | 0.049 | 0.75 | 32 | 8 | 4 |
| office | image-top100 + assoc Deja View, cross-class 0.7 (E13) | all | 0.66 | 0.79 | 0.46 | 0.082 | 0.047 | 0.76 | 49 | 16 | 5 |
| office | image-top100 + assoc Deja View, cross-class 0.7 (E13) | objects | 0.68 | 0.83 | 0.42 | 0.104 | 0.052 | 0.74 | 42 | 14 | 4 |
| office | image-top100 + assoc Deja View, cross-class 0.5 (E13) | all | 0.66 | 0.79 | 0.46 | 0.082 | 0.047 | 0.80 | 49 | 12 | 5 |
| office | image-top100 + assoc Deja View, cross-class 0.5 (E13) | objects | 0.68 | 0.83 | 0.42 | 0.104 | 0.052 | 0.79 | 42 | 10 | 4 |
| office | image + assoc Deja View, cross-class 0.7, visible >= 0.5 (E14) | all | 0.58 | 0.78 | 0.71 | 0.077 | 0.045 | 0.59 | 39 | 23 | 3 |
| office | image + assoc Deja View, cross-class 0.7, visible >= 0.5 (E14) | objects | 0.57 | 0.80 | 0.67 | 0.098 | 0.049 | 0.53 | 32 | 21 | 2 |
| office | image + assoc Deja View, cross-class 0.7, visible >= 0.25 (E14) | all | 0.58 | 0.78 | 0.71 | 0.077 | 0.045 | 0.77 | 39 | 11 | 4 |
| office | image + assoc Deja View, cross-class 0.7, visible >= 0.25 (E14) | objects | 0.57 | 0.80 | 0.67 | 0.098 | 0.049 | 0.75 | 32 | 9 | 3 |
| office | image-top100 + assoc Deja View, cross-class 0.7, visible >= 0.5 (E14) | all | 0.66 | 0.79 | 0.46 | 0.082 | 0.047 | 0.61 | 49 | 29 | 3 |
| office | image-top100 + assoc Deja View, cross-class 0.7, visible >= 0.5 (E14) | objects | 0.68 | 0.83 | 0.42 | 0.104 | 0.052 | 0.57 | 42 | 27 | 2 |
| office | image-top100 + assoc Deja View, cross-class 0.7, visible >= 0.25 (E14) | all | 0.66 | 0.79 | 0.46 | 0.082 | 0.047 | 0.76 | 49 | 16 | 4 |
| office | image-top100 + assoc Deja View, cross-class 0.7, visible >= 0.25 (E14) | objects | 0.68 | 0.83 | 0.42 | 0.104 | 0.052 | 0.74 | 42 | 14 | 3 |
| office | image t0.3 + assoc Deja View, xc 0.7, vf 0.25, score > 0.5 (E15) | all | 0.58 | 0.78 | 0.71 | 0.077 | 0.045 | 0.77 | 39 | 11 | 4 |
| office | image t0.3 + assoc Deja View, xc 0.7, vf 0.25, score > 0.5 (E15) | objects | 0.57 | 0.80 | 0.67 | 0.098 | 0.049 | 0.75 | 32 | 9 | 3 |
| office | image t0.3 + assoc Deja View, xc 0.7, vf 0.25, no filter (E15) | all | 0.64 | 0.82 | 0.57 | 0.076 | 0.046 | 0.78 | 41 | 12 | 4 |
| office | image t0.3 + assoc Deja View, xc 0.7, vf 0.25, no filter (E15) | objects | 0.61 | 0.81 | 0.51 | 0.098 | 0.050 | 0.76 | 33 | 10 | 3 |
| office | image t0.3 + assoc Deja View, xc 0.7, vf 0.25, multi-view filter (E15) | all | 0.63 | 0.81 | 0.73 | 0.076 | 0.045 | 0.78 | 41 | 10 | 4 |
| office | image t0.3 + assoc Deja View, xc 0.7, vf 0.25, multi-view filter (E15) | objects | 0.60 | 0.81 | 0.68 | 0.097 | 0.050 | 0.76 | 33 | 9 | 3 |
| office | image-top100 t0.3 + assoc Deja View, xc 0.7, vf 0.25, score > 0.5 (E15) | all | 0.66 | 0.79 | 0.46 | 0.082 | 0.047 | 0.76 | 49 | 16 | 4 |
| office | image-top100 t0.3 + assoc Deja View, xc 0.7, vf 0.25, score > 0.5 (E15) | objects | 0.68 | 0.83 | 0.42 | 0.104 | 0.052 | 0.74 | 42 | 14 | 3 |
| office | image-top100 t0.3 + assoc Deja View, xc 0.7, vf 0.25, no filter (E15) | all | 0.74 | 0.83 | 0.34 | 0.082 | 0.049 | 0.71 | 51 | 20 | 4 |
| office | image-top100 t0.3 + assoc Deja View, xc 0.7, vf 0.25, no filter (E15) | objects | 0.73 | 0.84 | 0.30 | 0.106 | 0.055 | 0.70 | 43 | 17 | 3 |
| office | image-top100 t0.3 + assoc Deja View, xc 0.7, vf 0.25, multi-view filter (E15) | all | 0.73 | 0.83 | 0.48 | 0.082 | 0.048 | 0.75 | 51 | 16 | 4 |
| office | image-top100 t0.3 + assoc Deja View, xc 0.7, vf 0.25, multi-view filter (E15) | objects | 0.72 | 0.84 | 0.43 | 0.105 | 0.054 | 0.72 | 43 | 15 | 3 |
| office | image-vlm t0.3 + assoc Deja View, xc 0.7, vf 0.25, score > 0.5 (E16) | all | 0.66 | 0.80 | 0.39 | 0.081 | 0.051 | 0.74 | 46 | 15 | 4 |
| office | image-vlm t0.3 + assoc Deja View, xc 0.7, vf 0.25, score > 0.5 (E16) | objects | 0.68 | 0.84 | 0.34 | 0.102 | 0.057 | 0.72 | 39 | 13 | 3 |
| office | image-vlm t0.3 + assoc Deja View, xc 0.7, vf 0.25, multi-view filter (E16) | all | 0.71 | 0.83 | 0.40 | 0.080 | 0.051 | 0.72 | 47 | 15 | 5 |
| office | image-vlm t0.3 + assoc Deja View, xc 0.7, vf 0.25, multi-view filter (E16) | objects | 0.70 | 0.85 | 0.35 | 0.103 | 0.058 | 0.69 | 39 | 14 | 4 |
| office | image-top100 t0.3 + assoc Deja View, xc 0.7, no vf, score > 0.5 (E17) | all | 0.66 | 0.79 | 0.46 | 0.082 | 0.047 | 0.76 | 49 | 16 | 5 |
| office | image-top100 t0.3 + assoc Deja View, xc 0.7, no vf, score > 0.5 (E17) | objects | 0.68 | 0.83 | 0.42 | 0.104 | 0.052 | 0.74 | 42 | 14 | 4 |
| office | image-top100 t0.3 + assoc Deja View, xc 0.7, no vf, multi-view filter (E17) | all | 0.73 | 0.83 | 0.48 | 0.081 | 0.048 | 0.76 | 51 | 15 | 6 |
| office | image-top100 t0.3 + assoc Deja View, xc 0.7, no vf, multi-view filter (E17) | objects | 0.72 | 0.84 | 0.43 | 0.105 | 0.054 | 0.72 | 43 | 15 | 4 |
| office | image-vlm t0.3 + assoc Deja View, xc 0.7, no vf, score > 0.5 (E17) | all | 0.66 | 0.80 | 0.39 | 0.081 | 0.051 | 0.74 | 46 | 14 | 5 |
| office | image-vlm t0.3 + assoc Deja View, xc 0.7, no vf, score > 0.5 (E17) | objects | 0.68 | 0.84 | 0.34 | 0.102 | 0.057 | 0.72 | 39 | 12 | 4 |
| office | image-vlm t0.3 + assoc Deja View, xc 0.7, no vf, multi-view filter (E17) | all | 0.71 | 0.84 | 0.40 | 0.080 | 0.051 | 0.74 | 47 | 13 | 7 |
| office | image-vlm t0.3 + assoc Deja View, xc 0.7, no vf, multi-view filter (E17) | objects | 0.70 | 0.85 | 0.35 | 0.103 | 0.058 | 0.69 | 39 | 13 | 5 |
| office | image-top100+vlm t0.3 + assoc Deja View, xc 0.7, vf 0.25, score > 0.5 (E19) | all | 0.69 | 0.80 | 0.32 | 0.086 | 0.051 | 0.73 | 49 | 17 | 4 |
| office | image-top100+vlm t0.3 + assoc Deja View, xc 0.7, vf 0.25, score > 0.5 (E19) | objects | 0.71 | 0.84 | 0.28 | 0.109 | 0.058 | 0.71 | 42 | 15 | 3 |
| office | image-top100+vlm t0.3 + assoc Deja View, xc 0.7, vf 0.25, multi-view filter (E19) | all | 0.75 | 0.84 | 0.33 | 0.085 | 0.052 | 0.71 | 51 | 18 | 5 |
| office | image-top100+vlm t0.3 + assoc Deja View, xc 0.7, vf 0.25, multi-view filter (E19) | objects | 0.74 | 0.86 | 0.28 | 0.109 | 0.059 | 0.67 | 43 | 17 | 4 |

Earlier entries (E4, E7, E9) first reported precision and recall with scorer v1; the values above supersede them
(differences: recall +0.01 from ignoring the "SPLIT" label, precision +0.02 to +0.05 from ignoring masks on
unlabeled GT). id consistency, switches and merges are unchanged.

## Plan after E13 (26.10.2026): stack (a) -> (b) -> (c), then a factorial grid

The three open options change different stages, so they can be combined, and each makes the next one safer:

| step | option | stage | target | depends on |
|---|---|---|---|---|
| E14 | (a) evidence-aware overlap (visible point count) | association: how a link is scored | office chair -> monitor link (E13) | - |
| E15 | (b) score threshold 0.5 -> 0.3, keep a < 0.5 mask only if linked to a mask in >= 1 other view | mask selection | office floor (scores 0.2-0.46, E12) | (a): the filter trusts links, so weak-evidence links must not pass it |
| E16 | (c) arm C, VLM vocabulary (one list per scene, run on every view) | prompts | unprompted objects | (b): filters one-view masks of hallucinated classes; (a), E13: more classes -> more cross-class links |

- **Stacking:** one change per run, each compared with the run before it. Base = E13 working setting
  (cross-class 0.7) with both prompt sets (16, top100); E16 compares the VLM vocabulary with top100. A step stays
  in the stack if its target improves without more merges or cross-class leakage.
- **E17, grid at the end:** the effect of a step depends on what is already stacked (e.g. (c) on top of a+b is not
  (c) alone). So after E16 every combination a on/off x b on/off x vocabulary top100/VLM (8 setups, full
  factorial, which also covers leave-one-out) is scored for the report. Cheap: SAM 3 runs once per vocabulary at
  threshold 0.3 with per-mask scores saved; 0.5 is then an offline filter, and association is ~13 s per run.
  Two scenes only: differences of 1-2 consistent objects are noise.
- **E17 settings, fixed 26.10 before running:** base for all 8 setups = SAM 3 cache at threshold 0.3 +
  association with Déjà View geometry and `--cross-class 0.7`; (a) on = `--min-visible-fraction 0.25`, off = no
  visibility check; (b) on = `--keep-linked-below 0.5`, off = `--min-score 0.5` (same masks as threshold 0.5, E15);
  vocabulary = top100 or the E16 VLM list. Same 20 views and scorer.
- Same rules as before: same 20 views and scorer; settings fixed before each run from the mechanism, never from
  these scenes' misses.

---

## E1: Ground-truth instance masks and depth (26.10.2026)
- **Goal:** ScanNet++ annotates instances only on the 3D laser-scan mesh; we need per-view 2D GT to score SAM 3.
- **Code:** `4071459`, `src/loaders/scannetpp/gt.py`.
- **Command:** `uv run scannetpp-render-gt data/processed/scannetpp/*/views_n20_uniform_504.npz`;
  figures `uv run python experiments/scripts/figures_gt.py`.
- **Method:** vertex -> segment (`segments.json`) -> object (`segments_anno.json`, `objectId`, label); triangle id =
  majority of its 3 vertices. Open3D raycast through every pixel center (+0.5, same convention as
  `Geometry.world_points`) with the GT camera; hit triangle -> instance id, hit distance -> z-depth.
  -1 = unlabeled mesh, -2 = ray misses the mesh (ignored in scoring).
- **Results** (`results/e01_gt_stats.txt`):

  | scene | labeled | unlabeled | no mesh | instances | depth range |
  |---|---|---|---|---|---|
  | kitchen | 93% | 4% | 4% | 42 | 0.19-2.44 m |
  | office | 92% | 2% | 6% | 92 | 0.27-4.76 m |

- **Figures:** `figures/e01_gt_overlay_<scene>.jpg` (4 views: image, instances, depth);
  `figures/<scene>_gt.jpg` (all 20 views with ids).
- **Observations:** object edges line up with the images (camera conventions correct). Walls are one instance per
  wall plane. Holes (no mesh): windows/glass, ceiling lights, under the copier. Thin objects are coarse (mesh
  decimated to 5 cm). Some kitchen images contain magenta privacy patches from ScanNet++ anonymization.

## E2: SAM 3 image mode on one view (26.10.2026)
- **Goal:** does SAM 3 fit on a 4 GB GPU, how fast is it, how good are the masks.
- **Code:** `experiments/scripts/sam3_image_smoke.py` (originally a scratch script).
- **Settings:** office view 6, 12 prompts (wall, floor, ceiling, door, window, table, chair, monitor, keyboard,
  cabinet, bookshelf, whiteboard; note "bookshelf", not "shelf" as in the later 16-prompt list). Image features
  computed once and reused for every prompt.
- **Results** (`results/e02_sam3_image_smoke.txt`): weights 1.7 GB on GPU, peak 2.1 GB, 3 s for 12 prompts.
  21 masks: whiteboard 0.97, keyboard 0.96, door 0.97, table 0.92, all 5 monitors as separate instances
  (0.85-0.93), chairs incl. occluded ones (0.55-0.88), 3 wall planes. Floor 0 (carpet clearly visible),
  ceiling 0 (thin strip only), window/bookshelf 0 (not in view). "door" also returned 3 lower-score masks on cabinet
  doors; white cabinets split into 2 "cabinet" masks + part labeled "wall".
- **Figure:** `figures/e02_sam3_image_smoke_7b6477cb95_v6.jpg`.
- **Conclusion:** fits comfortably; per-view mask quality is good; overlapping prompts need resolution
  (highest score wins).

## E3: SAM 3 video mode GPU memory (26.10.2026)
- **Goal:** find a video-mode setup that fits in 4 GB, after the first full run stalled.
- **What happened first:** a run with all 16 prompts in one session and tracker state on the GPU (default
  `inference_state_device`) filled the 4 GB; Windows (WDDM) then silently moved memory into shared system
  memory instead of raising out-of-memory. After 15 min the first scene had not finished, GPU utilization near
  zero. Stopped manually.
- **Code:** `experiments/scripts/sam3_video_memory.py`; capped at 3.5 GB
  (`torch.cuda.set_per_process_memory_fraction`) so it fails fast; tracker state and frames on CPU.
- **Command:** `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True uv run python
  experiments/scripts/sam3_video_memory.py 7b6477cb95 <prompts...>`.
- **Results** (office, 20 views; `results/e03_video_memory_*.txt`):

  | prompts in one session | peak GPU | time for 20 frames |
  |---|---|---|
  | 1 (chair) | 2.35 GB | 36 s |
  | 2 (wall, chair) | 2.69 GB | 64 s |
  | 4 (wall, table, chair, monitor) | OOM on frame 0 | - |
  | 16 | OOM on frame 0 | - |

  (Rerun 26.10 with `expandable_segments`. The original 1-prompt run without it peaked at 2.38 GB; the 16-prompt
  run is not saved in `results/`, it failed the same way on frame 0.)
- **Conclusion:** detection runs all prompts of a session at once; grouping prompts saves no time. Video mode
  therefore runs one session per prompt (`src/segmentation/sam3.py`, `Sam3VideoSegmenter`), tracker state on
  CPU, with a `--max-gpu-gb` cap. Outputs of the first frames are released with a delay (hotstart buffer).

## E4: SAM 3 video mode on the 20 sparse views (26.10.2026)
- **Goal:** task step 1: does the tracker keep instance ids across Déjà View's sparse views?
- **Code:** `43be2a6` (`Sam3VideoSegmenter`; instance id = prompt index * 1000 + SAM 3 object id).
- **Command:** `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1 uv run sam3-segment
  data/processed/scannetpp/*/views_n20_uniform_504.npz` -> `sam3video_views_n20_uniform_504.npz`;
  `uv run eval-instances <views> <seg>`.
- **Runtime:** kitchen 402 s, office 468 s, peak GPU 2.72 / 2.43 GB. 47 / 68 tracked instances.
- **Results:** see reference table (rows "video mode"). About half of the GT objects keep one id
  (consistent 0.50 kitchen, 0.47 office), 31 / 37 id switches, 13 / 20 merges.
- **Breakdown** (`results/e04_ids_video_<scene>.txt`, `uv run python experiments/scripts/breakdowns.py ids
  <scene> sam3video_views_n20_uniform_504`):
  - splits: office monitor GT 52 gets 6 ids over 13 views; kitchen wall GT 31 gets 9 ids over 11 views.
  - merges: office id 7008 covers all 4 monitors; id 5000 three tables; id 13000 three ceiling lamps; id 11000
    both whiteboards (on opposite walls). Kitchen wall ids hop between walls.
  - single, distinct objects track well (floor 1 id, door).
  - some class confusion: "chair" masks matched to a monitor; "wall" masks on heater / storage cabinet.
- **Figures:** `figures/<scene>_sam3video_views_n20_uniform_504.jpg` vs `figures/<scene>_gt.jpg`.
- **Conclusion:** as the organizers predicted, sparse video mode does not give reliable cross-view ids. With
  0.3-0.9 m camera steps the tracker falls back to appearance, which cannot tell identical monitors, chairs or
  lamps apart. Merges are the worse error for refinement (geometry of different objects would be tied together).
  The missing cue is 3D position.

## E5: Decision: no dense-sequence video mode (26.10.2026)
- Task step 2 was: run video mode on all DSLR frames (255 / 406) and read masks off at the 20 selected views.
- Dropped: it would require a video as input and give up Déjà View's unordered multi-view setting. (It would
  also take ~3 h per scene on the laptop.) Note: the DSLR frames are not frames of the iPhone video either.
- Next: SAM 3 image mode per view + association in 3D (task step 3).

## E6: Association sanity check with GT masks and GT geometry (26.10.2026)
- **Goal:** test the matcher in isolation: with perfect masks and perfect depth it should recover the ids.
- **Code:** `d0f54a8`, `src/segmentation/associate.py` (`Associator`); check in
  `experiments/scripts/sanity_checks.py` -> `results/sanity_checks.txt`.
- **Method (association):** lift every mask's pixels with depth, project into every other view; keep points
  whose depth agrees with the target view's depth within 5% (occlusion test) and that fall inside the image;
  overlap a->b = fraction of a's visible points landing in mask b. Edge score = min of both directions,
  classes must be equal, threshold 0.5. Greedy union-find, strongest edge first, never joining two masks of
  the same view. ~12 s per scene on CPU.
- **Input:** GT masks with fresh ids in every view (402 / 790 per-view masks).
- **Results:** kitchen 62 objects, consistent 1.00, 0 switches, 0 merges; office 235 objects, consistent 0.93,
  4 switches, 0 merges. (Object count > GT count because small pieces don't link; they are below the scoring
  size.) The "image, no association" rows of the reference table show the starting point for SAM 3 masks
  (consistent 0 by construction).

## E7: SAM 3 image mode + association with GT geometry (26.10.2026)
- **Goal:** upper bound for the association approach (perfect depth and cameras).
- **Code:** `d0f54a8`.
- **Commands:** `HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True uv run sam3-segment --mode image
  data/processed/scannetpp/*/views_n20_uniform_504.npz` -> `sam3image_views_n20_uniform_504.npz`
  (per-view ids = view * 1000 + k); `uv run python -m src.segmentation.associate <views> <seg> --geometry gt`.
- **Runtime:** image mode 59 s (kitchen) / 53 s (office), peak 2.08 GB; 255 / 391 masks. Association ~11 s.
- **Results:** reference table rows "image + assoc GT geometry": consistent 0.79 / 0.77 (video: 0.50 / 0.47),
  switches 7 / 13 (31 / 37), merges 1 / 5 (13 / 20), precision 0.83 / 0.71 (0.66 / 0.60).
- **Figures:** `figures/<scene>_sam3image_views_n20_uniform_504_assoc-gt.jpg`.

## E8: Déjà View on the laptop (26.10.2026)
- **Goal:** get Déjà View depth and cameras for association without GT; check they are sane.
- **Code:** Pouya's `src/backbone/dejaview.py` (`nvidia/dvlt`, K=16, bf16), run with a 3.5 GB cap:
  `uv run python -c "import torch; torch.cuda.set_per_process_memory_fraction(3.5 / 4.29); from
  src.backbone.dejaview import main; main()" data/processed/scannetpp/*/views_n20_uniform_504.npz`
  -> `dejaview_k16_views_n20_uniform_504.npz`.
- **Results** (`results/sanity_checks.txt`): fits, peak 3.25 GB. Depth 20x336x504, finite, confidence 1.05-93.
  Per-view scale-aligned median absolute relative depth error vs GT: kitchen 0.8% (worst view 1.4%), office
  0.7% (worst 1.5%). Cameras are in Déjà View's own frame (association does not need GT alignment).

## E9: SAM 3 image mode + association with Déjà View geometry (26.10.2026)
- **Goal:** the real method, no GT used.
- **Code:** `d0f54a8`. **Command:** `uv run python -m src.segmentation.associate <views>
  <scene>/sam3image_views_n20_uniform_504.npz --geometry <scene>/dejaview_k16_views_n20_uniform_504.npz`.
- **Results:** reference table rows "image + assoc Deja View": consistent 0.75 / 0.77, switches 8 / 13,
  merges 1 / 4: almost identical to the GT-geometry upper bound (E7).
  Breakdown: `results/e09_ids_assoc_dejaview_<scene>.txt`.
- **Figures:** `figures/<scene>_sam3image_views_n20_uniform_504_assoc-dejaview_k16_views_n20_uniform_504.jpg`.
  Office: the two whiteboards now get separate ids; the 5 monitors keep their own ids.
- **Conclusion:** Déjà View geometry is good enough for association. Remaining errors come from the per-view
  SAM 3 masks (e.g. one "table" mask over the whole connected desk block, where GT has several tables).

## E10: Recall breakdown per class (26.10.2026)
- **Goal:** why is instance recall only ~0.6?
- **Command:** `uv run python experiments/scripts/breakdowns.py recall <scene>
  sam3image_views_n20_uniform_504_assoc-dejaview_k16_views_n20_uniform_504`
  -> `results/e10_recall_assoc_dejaview_<scene>.txt`.
- **Results:** instance recall 0.64 / 0.58, but pixel-weighted recall 0.84 (kitchen) / 0.78 (office): misses are
  mostly small objects.
  1. Not prompted: copy machine (5.4% of scored kitchen pixels, found in 2/12 views), bottle crate, microwave,
     calendar, trash bin, fan, blinds, coat rack, heater.
  2. Prompted but missed: office floor 3/18 (9.8% of pixels, carpet), walls, office tables 21/41 (12.1%),
     storage cabinets 14/24 (10.5%), windows 4/23. Part of this is granularity (one SAM 3 mask over several GT tables or walls
     fails IoU >= 0.5).
  3. Thin / tiny: door frames, windowsills, sockets, cables: irrelevant for geometry.
- **Note:** these are the test scenes; prompts must not be chosen from these misses.

## E11: Scorer v2/v3: pixel recall, leakage, unlabeled handling (26.10.2026)
- **Goal:** measure what matters for refinement: covered surface and masks crossing object boundaries.
- **Code:** v2 `59311d2` (pixel recall, leakage, cross-class leakage; "SPLIT" helper label treated as
  unlabeled). v3 (this commit): predicted masks lying mostly (> 50%) on unlabeled GT are ignored for precision
  (nothing to judge them by, like COCO ignore regions). Found while re-running the sanity checks: with v2, GT
  scored against itself gave precision 0.965 / 0.980 because GT "SPLIT" masks counted as false positives.
- **Sanity** (`results/sanity_checks.txt`): GT vs GT -> recall, pixel recall, precision 1.000, leakage 0.000;
  GT relabeled per view -> consistent 0.000 (265 / 384 switches); GT with id pairs 2k, 2k+1 merged ->
  leakage 0.096 / 0.090, cross-class 0.083 / 0.065.
- **Results:** reference table. Leakage is a per-view property, so association does not change it.
  Kitchen leakage ~1% (cross-class 0.5%), office ~8% (cross-class 4.5%).
- **Office leakage breakdown** (`results/e11_leakage_assoc_dejaview_7b6477cb95.txt`): by predicted class
  cabinet 38%, table 24%, door 16%, window 5% of leaked pixels. Same-class: one "cabinet" mask over two adjacent
  storage cabinets (19%), one "table" mask over several tables (19%). Cross-class: cabinet -> bookshelf (12%),
  door -> wall (11%), cabinet -> wall (6%).
- **Conclusion:** judge prompt and threshold changes by pixel recall and cross-class leakage, not only recall.

## E12: Prompt vocabulary arm B: ScanNet++ top100 classes (26.10.2026)
- **Goal:** plan step 2, arm B. Does a generic vocabulary (the 100 classes of the ScanNet++ semantic benchmark,
  `data/scannetpp/metadata/semantic_benchmark/top100.txt`) find more surface than the 16 hand-picked prompts
  (arm A = E9), without more leakage and without breaking association? Not chosen from these scenes' misses.
- **Code:** this commit. `sam3-segment --prompts FILE` (repeatable; one prompt per line, `prompt = class` maps a
  synonym to a class, `load_prompts` in `src/segmentation/sam3.py`); the prompt files' names go into the cache
  name, `sam3image-<stem>+<stem>_<views>.npz` (no `--prompts` = the 16 built-in prompts, old name).
- **Commands:**
  `HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True uv run sam3-segment --mode image --prompts
  data/scannetpp/metadata/semantic_benchmark/top100.txt data/processed/scannetpp/*/views_n20_uniform_504.npz`;
  then `associate ... --geometry <scene>/dejaview_k16_views_n20_uniform_504.npz` and `eval-instances` as in E9;
  breakdowns `uv run python experiments/scripts/breakdowns.py {ids,recall,precision,leakage} <scene>
  sam3image-top100_views_n20_uniform_504_assoc-dejaview_k16_views_n20_uniform_504 [prompt files]`.
- **Runtime** (`results/e12_sam3image_top100.txt`): 261 s (kitchen) / 271 s (office), ~4.5x arm A; peak GPU
  unchanged at 2.10 / 2.08 GB (prompts run one at a time on cached image features). 583 / 732 masks
  (arm A 255 / 391); association 15 s.
- **Results** (reference table, rows "image top100"), arm A -> arm B, all classes:

  | scene | pixel recall | recall | precision | cross-class leakage | consistent | GT in >=2 views | id switches | merges |
  |---|---|---|---|---|---|---|---|---|
  | kitchen | 0.84 -> 0.92 | 0.64 -> 0.84 | 0.83 -> 0.49 | 0.005 -> 0.007 | 0.75 -> 0.60 | 28 -> 35 | 8 -> 17 | 1 -> 1 |
  | office | 0.78 -> 0.79 | 0.58 -> 0.66 | 0.71 -> 0.46 | 0.045 -> 0.047 | 0.77 -> 0.51 | 39 -> 49 | 13 -> 36 | 4 -> 5 |

- **Breakdowns** (`results/e12_{recall,ids,precision,leakage}_top100_assoc_dejaview_<scene>.txt`):
  - Recall: kitchen gains copy machine 2/12 -> 12/12 views (found as "printer", 5.4% of pixels), microwave
    2 -> 11, calendar 0 -> 16, bottle crate 26 -> 33, ceiling lamp 1 -> 4. Office: small objects gain (trash
    can, chairs, ceiling lamps), but the big misses stay: floor 3/18 (9.8% of pixels), tables 21/41, storage
    cabinets 13/24, walls 42/59. Hence office pixel recall +0.01 only.
  - Consistency: the drop comes from **near-synonym classes in top100**. The same object is called "monitor" in
    one view and "tv" in another (all 5 office monitors), "chair" / "office chair", "cabinet" / "storage cabinet",
    "window" / "window frame"; association only links masks of the same class. Split objects whose ids carry
    different classes: office 19 of 24 (arm A 3 of 9), kitchen 9 of 14 (arm A 3 of 7). Merges unchanged.
    The consistency denominator also grows (more objects found in >= 2 views).
  - Precision: mostly parts of one annotated GT object, which the GT cannot score: 71 "bottle" masks inside the
    kitchen bottle crate, cloth / marker on the fridge, sockets on office walls, pipe parts of the coat rack.
    Duplicates ("tv" next to "monitor" on the same screen) add more. Not harmful for refinement in itself:
    leakage is unchanged.
  - Leakage: kitchen 0.009 -> 0.010, office 0.077 -> 0.082; cross-class within 0.002 (same big-mask errors as
    arm A: cabinet / table / door masks).
- **Variant B2, top100 + floor synonyms** (`configs/prompts/floor_synonyms.txt`: carpet, rug -> floor; caveat:
  this idea came after seeing the office carpet miss in E10, so it is weaker evidence). Result: identical to B1
  (same 583 / 732 masks, same scores; `results/e12_sam3image_top100+floor_synonyms.txt`). Raw scores without a
  threshold (`experiments/scripts/sam3_prompt_scores.py` -> `results/e12_floor_prompt_scores.txt`): carpet <= 0.22,
  rug <= 0.15 in every view, while "floor" itself scores 0.2-0.46 on most office views (>= 0.5 in 4 of 20) and
  0.5-0.79 in 11 kitchen views. The office floor is a threshold problem, not a vocabulary problem.
- **Figures:** `figures/<scene>_sam3image-top100_views_n20_uniform_504_assoc-dejaview_k16_views_n20_uniform_504.jpg`.
- **Conclusion:** a larger generic vocabulary raises recall where arm A had unprompted objects (kitchen pixel
  recall +0.08, objects-only 0.71 -> 0.96) at the same leakage, but as is it hurts cross-view ids, because
  synonymous class names block association. Before comparing with arm C (VLM vocabulary) the class rule needs a
  fix, as its own experiment: e.g. group near-synonym classes for the class check, or relax
  the same-class rule for strong 3D overlap (plan item 5). The office floor needs plan item 3 (lower threshold with
  multi-view filter), not synonyms.

## E13: Association across classes at strong 3D overlap (26.10.2026)
- **Goal:** E12 showed that near-synonym class names (monitor / tv, chair / office chair, cabinet / storage
  cabinet) block association, which only links masks of the same class. Instead of a hand-written synonym table
  (would be built from these test scenes' failures, and would not carry over to a VLM vocabulary), let geometry
  decide: link masks of different classes too, but at a stricter overlap. Settings fixed before the runs:
  off (E9 / E12), 0.7, 0.5 (= class ignored); same-class links stay at 0.5.
- **Code:** this commit. `Associator(cross_class_overlap=...)`, CLI `--cross-class X`, output
  `<seg>_assoc-xc<X>-<geometry>.npz`. An object now takes the class covering most of its pixels (identical to before
  when all its masks share one class). Regression check: with the option off, all four E9 / E12 caches are
  reproduced bit for bit.
- **Commands:** `uv run python -m src.segmentation.associate <views> <scene>/<seg>.npz --geometry
  <scene>/dejaview_k16_views_n20_uniform_504.npz --cross-class {0.7,0.5}` for `<seg>` in `sam3image_*` (16
  prompts) and `sam3image-top100_*`; then `eval-instances` (`results/e13_cross_class_runs.txt`). ~13 s per run.
- **Results** (reference table, rows E13), consistent / id switches, all classes:

  | scene | prompts | off | 0.7 | 0.5 |
  |---|---|---|---|---|
  | kitchen | 16 | 0.75 / 8 | 0.86 / 4 | 0.86 / 4 |
  | kitchen | top100 | 0.60 / 17 | 0.83 / 6 | 0.86 / 5 |
  | office | 16 | 0.77 / 13 | 0.77 / 10 | 0.77 / 10 |
  | office | top100 | 0.51 / 36 | 0.76 / 16 | 0.80 / 12 |

  Merges (scorer) unchanged everywhere except office 16 prompts 4 -> 5. Recall, pixel recall, precision and
  leakage cannot change (association does not change masks). With cross-class linking top100 is as consistent as
  the 16 prompts while keeping its recall gain (kitchen pixel recall 0.92 vs 0.84).
- **Are the new links right?** The scorer's merge count only sees masks matched at IoU >= 0.5, so two extra checks
  on all masks (each mask -> the GT object holding > 50% of its labeled pixels):
  - `breakdowns.py purity` (`results/e13_purity_*`): objects whose masks lie on different GT objects:
    kitchen 3 -> 3 (16 prompts), 4 -> 4 (top100); office 7 -> 8 (16), 9 -> 9 (top100), same at 0.7 and 0.5.
  - `experiments/scripts/cross_class_links.py <scene> <seg stem> <X>` (`results/e13_links_*`): every cross-class
    link that ended up in an object:

    | scene | prompts | 0.7: right / wrong / unlabeled | 0.5: right / wrong / unlabeled |
    |---|---|---|---|
    | kitchen | 16 | 15 / 0 / 0 | 17 / 0 / 0 |
    | kitchen | top100 | 64 / 0 / 0 | 83 / 0 / 0 |
    | office | 16 | 16 / 8 / 2 | 22 / 8 / 3 |
    | office | top100 | 176 / 15 / 4 | 202 / 18 / 5 |

  - The wrong office links are of two kinds. (1) The two adjacent storage cabinets (GT 34, 35): "wall" / "door" /
    "cabinet" masks lying on one cabinet link to masks on the other; both GT objects have the same class and arm A
    already merged them (same-class), so this extends an existing error rather than crossing a class boundary.
    0.5 adds 3 more of these (overlap 0.52-0.63). (2) **One real cross-object error:** a "chair" mask in view 5
    (on GT office chair 58) links to the "monitor" / "tv" masks of GT monitor 52 in 4 views, at overlap 0.88-0.98
    in both directions, at every threshold. Not investigated yet; suspicion: few of the chair's points pass the
    visibility test in the other views (chair close in front of the monitor, 5% depth tolerance), so the overlap
    fraction rests on little evidence (`min_visible` = 50 px).
- **Figures:** `figures/<scene>_<seg>_assoc-xc{0.7,0.5}-dejaview_k16_views_n20_uniform_504.jpg`.
- **Conclusion:** linking by 3D overlap regardless of class removes most near-synonym splits (office top100
  consistent 0.51 -> 0.76 / 0.80) with no wrong links in the kitchen and few in the office. 0.5 vs 0.7: +0.03-0.04
  consistent for 3 more same-class cabinet confusions; the one cross-object error is threshold-independent. Working
  setting: **0.7** (more conservative, merges are the worse error for refinement). This also matches the organizers'
  point that semantics should act through grouping and identity rather than category rules. Next for association:
  make the overlap evidence-aware (visible point count) to catch the chair -> monitor case.

## E14: Plan step (a): visibility-aware overlap (26.10.2026)
- **Goal:** remove the one real cross-object link of E13 (office "chair" mask 5006 in view 5 -> monitor GT 52 in 4
  views, overlap 0.88-0.98) by making the overlap evidence-aware, without hurting correct links. First step of the
  stack in "Plan after E13".
- **Diagnosis first** (`experiments/scripts/link_evidence.py 7b6477cb95 sam3image_views_n20_uniform_504 5006` ->
  `results/e14_link_evidence_chair5006_7b6477cb95.txt`). The E13 suspicion (few visible points) was only half right:
  the links rest on 300-450 visible points, far above `min_visible` = 50. What is small is the **share** of the
  monitor masks that can be checked: of monitor 9015's 1914 pixels inside view 5's image, only 310 (16%) pass the
  depth test, the rest is hidden behind the chair; the visible sliver lands in the chair mask, which leaks onto the
  monitor in view 5. Same for the other monitors: visible / inside 0.07, 0.11, 0.17, 0.22. Correct chair -> chair
  links of the same mask: 0.47-0.99.
- **Change** (this commit): `Associator(min_visible_fraction=f)`, CLI `--min-visible-fraction f`, output
  `<seg>_assoc-xc0.7-vf<f>-<geometry>.npz`. A direction a -> b counts only if at least f of a's pixels that project
  inside b's image (with known depth) are visible there; otherwise it is treated like too few visible pixels (no
  overlap, so no link). Applies to all links, same-class too. Off (f = 0) reproduces all four E9 / E12 / E13 caches
  bit for bit. Settings f = 0.5 (most of the mask seen) and 0.25. **Caveat:** both were fixed after the diagnosis
  above, so 0.25 is set knowing the bad ratios (<= 0.22); weaker evidence than a blind choice.
- **Commands:** `uv run python -m src.segmentation.associate <views> <scene>/<seg>.npz --geometry
  <scene>/dejaview_k16_views_n20_uniform_504.npz --cross-class 0.7 --min-visible-fraction {0.5,0.25}` for `<seg>` in
  `sam3image_*`, `sam3image-top100_*`; `eval-instances` (`results/e14_visible_fraction_runs.txt`); checks
  `cross_class_links.py <scene> <stem> 0.7 vf<f>` (`results/e14_links_*`) and `breakdowns.py purity`
  (`results/e14_purity_*`). ~13 s per run.
- **Results** (reference table, rows E14), all classes, consistent / id switches / merges:

  | scene | prompts | E13 (xc 0.7) | f = 0.5 | f = 0.25 |
  |---|---|---|---|---|
  | kitchen | 16 | 0.86 / 4 / 1 | 0.64 / 11 / 1 | 0.79 / 7 / 1 |
  | kitchen | top100 | 0.83 / 6 / 1 | 0.63 / 15 / 1 | 0.80 / 8 / 1 |
  | office | 16 | 0.77 / 10 / 5 | 0.59 / 23 / 3 | 0.77 / 11 / 4 |
  | office | top100 | 0.76 / 16 / 5 | 0.61 / 29 / 3 | 0.76 / 16 / 4 |

  Wrong cross-class links (office, `cross_class_links.py`): 16 prompts 8 -> 3, top100 15 -> 9, same at both f; the
  three left with 16 prompts are the known storage-cabinet confusion (GT 34 / 35). Kitchen: still no wrong link.
  Impure objects (`purity`): office 8 -> 7 (16) and 9 -> 8 (top100), kitchen unchanged. Masks, so recall, precision
  and leakage, are unchanged by construction.
- **What f = 0.25 costs** (`breakdowns.py ids`, kitchen 16 prompts): new splits of stacked boxes (GT 9, 34, 35,
  which hide each other) and of the copy machine (2 views); all same-class links.
- **Figures:** `figures/<scene>_<seg>_assoc-xc0.7-vf{0.5,0.25}-dejaview_k16_views_n20_uniform_504.jpg`.
- **Conclusion:** the chair -> monitor error came from a mask that is mostly hidden in the other view, not from a
  low point count; requiring a visible share removes it. f = 0.5 is far too strict (consistency -0.15 to -0.22).
  f = 0.25 removes one merge and most wrong cross-class links in the office at unchanged office consistency, but
  costs ~2 objects of consistency in the kitchen (within the noise rule of the plan, but one-sided). Merges are the
  worse error for refinement, so **f = 0.25 stays in the stack (working setting: xc 0.7 + vf 0.25)**; E17 will
  measure it with and without (b) and (c). A possible refinement, not tested (idea from these scenes' failures):
  apply the visibility rule only to cross-class links, where the evidence bar should be higher anyway.

## E15: Plan step (b): lower score threshold with a multi-view filter (26.10.2026)
- **Goal:** the office floor scores 0.2-0.46 in most views (E12) and is lost at threshold 0.5. Lower the SAM 3 score
  threshold to 0.3, but keep a mask scoring <= 0.5 only if association links it to a mask in at least one other view
  (false detections rarely reappear at the same 3D place). Second step of the stack, on top of the E14 working
  setting (cross-class 0.7, visible share 0.25). Settings fixed beforehand: 0.3 (from the plan) and 0.5.
- **Code** (this commit): `Segmentation.scores` (per-mask detection score, optional in the `.npz`, old caches load
  unchanged); `sam3-segment --threshold X` saves scores, output `sam3image[-<prompts>]-t<X>_<views>.npz`.
  `Associator(min_score=..., keep_linked_below=...)`, CLI `--min-score S` (drop masks scoring <= S before linking)
  and `--keep-linked-below S` (masks scoring <= S stay only if their object spans >= 2 views), output
  `..._assoc-xc0.7-vf0.25-{ms,kl}<S>-<geometry>.npz`. Scores compare like SAM 3's own threshold (pass if > value):
  with >= instead, 1-5 masks per run scoring exactly 0.500 (bf16) slipped through.
- **Checks:** the 0.3 run filtered offline at 0.5 gives the same per-view masks as the old 0.5 run (same partition
  and classes, all 4 caches), and its association (`--min-score 0.5`) reproduces the E14 scores exactly. So one SAM 3
  run per vocabulary covers both thresholds (needed for the E17 grid). New options off: E14 caches bit for bit.
- **Commands:** `HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True uv run sam3-segment --mode image
  --threshold 0.3 [--prompts .../top100.txt] <both views>` (`results/e15_sam3image_t0.3.txt`: 52-60 s / 259-262 s per
  scene, peak 2.08-2.13 GB, unchanged); `associate ... --cross-class 0.7 --min-visible-fraction 0.25
  [--min-score 0.5 | --keep-linked-below 0.5]` + `eval-instances` (`results/e15_threshold_runs.txt`); breakdowns
  `breakdowns.py recall|purity` (`results/e15_recall_*`, `results/e15_purity_*`),
  `experiments/scripts/low_score_masks.py` (`results/e15_low_score_*`).
- **Masks:** 255 -> 376 (kitchen), 391 -> 552 (office) with 16 prompts; 583 -> 925, 732 -> 1099 with top100.
- **Results** (reference table, rows E15), all classes, score > 0.5 (= E14) -> multi-view filter:

  | scene | prompts | pixel recall | recall | precision | cross-class leakage | consistent | id switches | merges |
  |---|---|---|---|---|---|---|---|---|
  | kitchen | 16 | 0.84 -> 0.89 | 0.64 -> 0.74 | 0.83 -> 0.81 | 0.005 -> 0.006 | 0.79 -> 0.81 | 7 -> 7 | 1 -> 1 |
  | kitchen | top100 | 0.92 -> 0.95 | 0.84 -> 0.91 | 0.49 -> 0.50 | 0.007 -> 0.007 | 0.80 -> 0.82 | 8 -> 8 | 1 -> 1 |
  | office | 16 | 0.78 -> 0.81 | 0.58 -> 0.63 | 0.71 -> 0.73 | 0.045 -> 0.045 | 0.77 -> 0.78 | 11 -> 10 | 4 -> 4 |
  | office | top100 | 0.79 -> 0.83 | 0.66 -> 0.73 | 0.46 -> 0.48 | 0.047 -> 0.048 | 0.76 -> 0.75 | 16 -> 16 | 4 -> 4 |

  Control without the filter (all masks > 0.3 kept): about the same recall, but precision falls to 0.68 / 0.35
  (kitchen) and 0.57 / 0.34 (office), and top100 consistency drops (office 0.71). The filter is what makes 0.3 usable.
- **Target:** office floor matched in 3/18 -> 9/18 views (both prompt sets), kitchen floor 9/13 -> 12/13; office
  walls 42 -> 49 of 59.
- **Is the filter right?** (`low_score_masks.py`; masks scoring 0.3-0.5, match = IoU >= 0.5 with a GT instance in
  their view): kept masks match GT in 67% / 54% (kitchen 16 / top100) and 100% / 75% (office); dropped masks in 10% /
  2% and 5% / 4%. Dropped: mostly parts and look-alikes in one view (bottle, container, socket, marker, door, chair).
  Kept but unmatched: mostly parts inside GT objects (paper, book, cloth), as in E12.
- **Purity:** impure objects +1 to +2 per run (kitchen 3 -> 4, 4 -> 5; office 7 -> 8, 8 -> 10), with more objects
  spanning >= 2 views (e.g. office 16: 44 -> 49), so the rate stays ~0.16; new office case: a door object picks up a
  doorframe mask. Merges (scorer) unchanged.
- **Figures:** `figures/<scene>_sam3image[-top100]-t0.3_views_n20_uniform_504_assoc-xc0.7-vf0.25[-ms0.5|-kl0.5]-dejaview_k16_views_n20_uniform_504.jpg`.
- **Conclusion:** the multi-view filter separates real from false low-score detections well (kept 54-100% real,
  dropped 2-10%), so the threshold can go down to 0.3: pixel recall +0.03 to +0.05 in every run, the office floor
  found in 3x more views, at unchanged precision, leakage, consistency and merges. **Stays in the stack. Working
  setting: SAM 3 threshold 0.3, `--cross-class 0.7 --min-visible-fraction 0.25 --keep-linked-below 0.5`.** The floor
  is still missed in half of the office views (scores < 0.3 or no link). Next: E16 = (c), VLM vocabulary.

## E16: Plan step (c), arm C: VLM vocabulary (26.10.2026)
- **Goal:** prompts from what is in the scene instead of a fixed list (organizer idea): a VLM names the objects in
  the views, the list becomes the SAM 3 prompts. Target: unprompted objects (E9: copy machine, bottle crate,
  microwave, fan, coat rack, ...). Third step of the stack, on top of the E15 working setting; compared with top100
  (the generic list it would replace) at the same settings.
- **Model:** `Qwen/Qwen3-VL-2B-Instruct` (natively in `transformers 5.20.2`, not gated, 4.3 GB in bf16), weights in
  4-bit NF4 via `bitsandbytes` 0.50.2 (new dependency, Linux/Windows only) with bf16 compute, to fit the 4 GB GPU.
  Chosen from the natively supported 1-4B models (`Qwen3.5-0.8B` fits without 4-bit but is weaker; 4B models fit
  only on the 3080). Local model only (ScanNet++ images are not sent to an API).
- **Code** (this commit): `src/segmentation/vocabulary.py`, CLI `uv run vlm-vocabulary <views>` -> prompt file
  `vlm-qwen3vl2binstruct_<views>.txt` (format of `sam3-segment --prompts`) + raw answers `.json` next to the views.
- **Protocol** (fixed before the run): one question per view, same generic text for every view and scene
  (`QUESTION`: every distinct kind of object and surface, structural ones included, short singular nouns, one per
  line); greedy decoding, at most 256 new tokens; names lowercased, bullets/numbering/parentheses removed, at most 3
  words; scene vocabulary = union over the 20 views in first-seen order (one list per scene, so classes agree
  across views). Every name becomes its own class (no synonym mapping).
- **Repetition loops:** the 2B model with greedy decoding ends 18/20 (kitchen) and 16/20 (office) answers by
  repeating one name until the token limit ("monitor\nmonitor\n..."). Duplicates are dropped, so this costs time,
  not names, but the first run kept the cut-off last line ("c"). Fix (general, not scene-specific): drop the last
  line of an answer that hit the limit. Prompt files rebuilt from the saved answers without a VLM rerun:
  `experiments/scripts/e16_reparse_vocabulary.py` (re-tokenizes to find truncated answers).
- **Vocabularies** (`results/e16_vocabulary_<scene>.txt`, raw answers `.json`): kitchen 46 names, office 60. They
  contain most E9 unprompted objects (copier, crate, microwave, calendar, trash can; fan, coat rack, curtain), vague
  names (surface, plastic, text, card, cog, laser) and near-synonyms (cable/cord/wire;
  socket/power outlet/wall outlet/electrical outlet; whiteboard/blackboard; office: "desk" but no "table").
- **Commands:** `HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True uv run vlm-vocabulary <both views>`;
  `uv run sam3-segment --mode image --threshold 0.3 --prompts <scene>/vlm-qwen3vl2binstruct_views_n20_uniform_504.txt
  <scene views>` per scene (`results/e16_sam3image_vlm_t0.3.txt`); `associate ... --cross-class 0.7
  --min-visible-fraction 0.25 {--min-score 0.5 | --keep-linked-below 0.5}` + `eval-instances`
  (`results/e16_vlm_runs.txt`); `breakdowns.py recall|purity|ids|precision` (`results/e16_*`).
- **Runtime:** VLM about 4 min per scene (office, 12 s per view, from file times; most of it the repetition loops;
  the kitchen run also loaded the model, its time and the process peak memory were not logged; `nvidia-smi` showed
  1.8 GB in use during the run). SAM 3: 131 s / 169 s, peak 2.11-2.13 GB (top100: 259-262 s, 16 prompts: 52-60 s).
  Masks: 1064 / 1508 (top100: 925 / 1099).
- **Results** (reference table, rows E16), all classes, multi-view filter (working setting):

  | scene | vocabulary | prompts | pixel recall | recall | precision | cross-class leakage | consistent | id switches | merges |
  |---|---|---|---|---|---|---|---|---|---|
  | kitchen | 16 (E15) | 16 | 0.89 | 0.74 | 0.81 | 0.006 | 0.81 | 7 | 1 |
  | kitchen | top100 (E15) | 100 | 0.95 | 0.91 | 0.50 | 0.007 | 0.82 | 8 | 1 |
  | kitchen | VLM | 46 | 0.94 | 0.89 | 0.44 | 0.007 | 0.83 | 7 | 1 |
  | office | 16 (E15) | 16 | 0.81 | 0.63 | 0.73 | 0.045 | 0.78 | 10 | 4 |
  | office | top100 (E15) | 100 | 0.83 | 0.73 | 0.48 | 0.048 | 0.75 | 16 | 4 |
  | office | VLM | 60 | 0.83 | 0.71 | 0.40 | 0.051 | 0.72 | 15 | 5 |

- **Target (recall per GT class, VLM vs top100):** office pedestal fan 1/8 -> 8/8, coat rack 5/9 -> 8/9, storage
  cabinet 13 -> 14/24; lost: doorframe 3/8 -> 0/8 (kitchen), 4/9 -> 0/9 (office), office windowsill 5/9 -> 0/9,
  kitchen calendar 18 -> 16/18. The kitchen targets (copy machine, bottle crate, microwave) were already found by
  top100 (via other prompts), so the VLM adds nothing there. vs the 16 prompts the gain is large (pixel recall
  +0.05 / +0.02), but top100 gives the same.
- **Precision:** unmatched masks mostly bottle (parts inside GT bottle crates), vague classes (surface, plastic),
  thin cables/cords and the near-synonym groups (outlets, cable/cord, computer next to monitor). Parts inside one GT
  object are not harmful for refinement; cross-class leakage is about unchanged (+0.000 / +0.003).
- **Merges / purity:** office merges 4 -> 5; the new one is a fan object covering the pedestal fan and an adjacent
  GT "object". Impure objects 5 / 11 (top100: 5 / 10); the kitchen floor is now an object of class "surface"
  (grouping right, class name vague).
- **Figures:** `figures/<scene>_sam3image-vlm-qwen3vl2binstruct_views_n20_uniform_504-t0.3_views_n20_uniform_504_assoc-xc0.7-vf0.25-{ms0.5,kl0.5}-dejaview_k16_views_n20_uniform_504.jpg`.
- **Conclusion:** on these two scenes the VLM vocabulary is about as good as top100 for recall, a bit worse for
  precision, at half the SAM 3 time but plus the VLM call. It finds objects top100 does not name (pedestal fan,
  coat rack) and loses thin frames and sills, which do not matter for geometry. Its advantage is that it does not
  need a dataset-specific class list; top100 comes from ScanNet++ annotations, so on these scenes it is an
  optimistic baseline. Not a clear win for the stack: one more office merge, slightly lower precision. The E17
  grid (a x b x vocabulary) decides. Possible follow-ups (not run): arm D (top100 + VLM union), and a repetition
  penalty or early stop to cut the VLM time.

## E17: Grid (a) visible share x (b) multi-view filter x vocabulary (26.10.2026)
- **Goal:** E14-E16 measured each step on top of the previous ones; the effect of a step can depend on what is
  already stacked. Full factorial for the report: (a) on/off x (b) on/off x vocabulary top100/VLM, both scenes
  (8 setups x 2 scenes = 16 runs). Settings fixed before running (see "Plan after E13", commit `0613474`).
- **Settings:** every run from the SAM 3 threshold-0.3 cache of its vocabulary, association with Déjà View geometry
  and `--cross-class 0.7`. (a) on = `--min-visible-fraction 0.25`, off = no visibility check. (b) on =
  `--keep-linked-below 0.5`, off = `--min-score 0.5` (= threshold 0.5, E15). Vocabulary: top100 (plain `top100.txt`, E12 B1;
  the floor synonyms of B2 were not used, they never reach 0.3 anyway) or the E16 VLM list. Same 20 views and scorer.
- **Code / command:** `uv run python experiments/scripts/e17_grid.py > experiments/results/e17_grid.txt`
  (associates and scores all 16 runs in one go, ~13 s per run, CPU only; saves each result under the CLI's name, new
  ones `..._assoc-xc0.7-{ms,kl}0.5-dejaview_...`); figures of the 8 new runs with `eval-instances --figures
  experiments/figures` (`results/e17_eval_instances.txt`, same numbers as the script); merges per setup
  `breakdowns.py ids` (`results/e17_merges_7b6477cb95.txt`).
- **Reproducibility check:** the 8 setups with (a) on were cached before (E15 / E16); the script reassociates them
  and asserts identical ids and classes: all 8 identical. `score_table.py` rerun: old rows unchanged, 16 rows added.
- **Results** (`results/e17_grid.txt`, all classes; objects-only rows there and in the reference table):

  | scene | vocabulary | (a) | (b) | pixel recall | recall | precision | cross-class leakage | consistent | id switches | merges |
  |---|---|---|---|---|---|---|---|---|---|---|
  | kitchen | top100 | off | off | 0.92 | 0.84 | 0.49 | 0.007 | 0.83 | 6 | 1 |
  | kitchen | top100 | off | on | 0.95 | 0.91 | 0.50 | 0.007 | 0.84 | 6 | 1 |
  | kitchen | top100 | on | off | 0.92 | 0.84 | 0.49 | 0.007 | 0.80 | 8 | 1 |
  | kitchen | top100 | on | on | 0.95 | 0.91 | 0.50 | 0.007 | 0.82 | 8 | 1 |
  | kitchen | VLM | off | off | 0.93 | 0.83 | 0.43 | 0.007 | 0.85 | 5 | 1 |
  | kitchen | VLM | off | on | 0.94 | 0.89 | 0.44 | 0.007 | 0.86 | 5 | 1 |
  | kitchen | VLM | on | off | 0.93 | 0.83 | 0.43 | 0.007 | 0.82 | 7 | 1 |
  | kitchen | VLM | on | on | 0.94 | 0.89 | 0.44 | 0.007 | 0.83 | 7 | 1 |
  | office | top100 | off | off | 0.79 | 0.66 | 0.46 | 0.047 | 0.76 | 16 | 5 |
  | office | top100 | off | on | 0.83 | 0.73 | 0.48 | 0.048 | 0.76 | 15 | 6 |
  | office | top100 | on | off | 0.79 | 0.66 | 0.46 | 0.047 | 0.76 | 16 | 4 |
  | office | top100 | on | on | 0.83 | 0.73 | 0.48 | 0.048 | 0.75 | 16 | 4 |
  | office | VLM | off | off | 0.80 | 0.66 | 0.39 | 0.051 | 0.74 | 14 | 5 |
  | office | VLM | off | on | 0.84 | 0.71 | 0.40 | 0.051 | 0.74 | 13 | 7 |
  | office | VLM | on | off | 0.80 | 0.66 | 0.39 | 0.051 | 0.74 | 15 | 4 |
  | office | VLM | on | on | 0.83 | 0.71 | 0.40 | 0.051 | 0.72 | 15 | 5 |

- **Main effects** (mean change when the factor is switched on, over the 4 settings of the other two; all classes,
  kitchen / office):

  | factor | pixel recall | recall | precision | cross-class leakage | consistent | id switches | merges |
  |---|---|---|---|---|---|---|---|
  | (a) visible share 0.25 | +0.00 / -0.00 | +0.00 / -0.00 | +0.00 / -0.00 | +0.000 / +0.000 | -0.03 / -0.01 | +2.0 / +1.0 | 0 / -1.5 |
  | (b) threshold 0.3 + multi-view filter | +0.02 / +0.04 | +0.06 / +0.06 | +0.01 / +0.01 | +0.000 / +0.000 | +0.01 / -0.00 | 0 / -0.5 | 0 / +1.0 |
  | (c) VLM instead of top100 | +0.00 / +0.01 | -0.02 / -0.01 | -0.06 / -0.08 | -0.000 / +0.003 | +0.02 / -0.02 | -1.0 / -1.5 | 0 / +0.5 |

- **Interaction (a) x (b)** (office; `results/e17_merges_7b6477cb95.txt`): without (a), (b) adds a merge (top100
  5 -> 6, VLM 5 -> 7): a low-score "wall" mask links to a heater and its wall (GT 33 + 39), kept by the filter
  because the link exists. With (a) this link and the E13 chair -> monitor merge (GT 52 + 58, present in every
  (a)-off setup) are gone; office merges 4 (top100) / 5 (VLM) with both on. So (a) helps more when (b) is on
  (merges -2 vs -1), as the plan expected: the filter trusts links, so weak-evidence links must not pass it. The
  VLM's extra office merge (pedestal fan + an adjacent GT "object") appears only with (b) on and is not removed by (a).
  Kitchen: 1 merge in every setup; no other interaction is larger than ~1 object.
- **Figure:** `figures/e17_main_effects.jpg` (main effects per metric and scene); per-run contact sheets
  `figures/<scene>_sam3image-{top100,vlm-...}-t0.3_views_n20_uniform_504_assoc-xc0.7-{ms,kl}0.5-dejaview_k16_views_n20_uniform_504.jpg`
  (new (a)-off runs; the (a)-on ones are from E15 / E16).
- **Conclusion:**
  - **(b) is a clear win**: in every one of the 8 settings pixel recall +0.01 to +0.04 and recall +0.05 to +0.07,
    precision +0.01 to +0.02, leakage unchanged. Its only cost (one office merge) appears when (a) is off.
  - **(a) is a trade-off, kept**: it removes the harmful office merges (chair -> monitor, heater -> wall) and costs
    ~1 kitchen object of consistency (stacked boxes, E14) in every setting. With (b) on it is needed. Caveat (E14):
    its value 0.25 was set after seeing the office ratios.
  - **(c) VLM is not adopted**: precision -0.06 / -0.08 in every setting, recall about equal, consistency mixed
    (+0.02 / -0.02). top100 stays, with the caveat that it is a ScanNet++-specific list (optimistic on ScanNet++);
    the VLM list is the dataset-independent fallback.
  - **Best setup on both scenes: top100 + (a) + (b) = the current working setting** (office: fewest merges, 4, at the
    highest pixel recall; kitchen: -0.02 consistency vs (a) off). Two scenes only: the (a) consistency cost and the
    merge differences are 1-2 objects each.

## E18: Boundaries of SAM 3 masks vs depth edges, and bleeding on ScanNet++ (27.10.2026)
- **Goal:** Peter's ETH3D analysis (branch `monocular-refinement`) shows bleeding at depth edges is Déjà View's largest
  error (median 4-6% at edges vs < 1% far away). Segments can only fix it if (1) their boundaries lie on the true depth
  edges and (2) they put the pixels on each side of an edge into the right object. The scorer so far measured IoU and
  leakage, not boundary position. Also: does bleeding show up on our ScanNet++ scenes too?
- **Settings:** working segmentation (E17 best: top100 t0.3 + `--cross-class 0.7 --min-visible-fraction 0.25
  --keep-linked-below 0.5`; association does not move boundaries), Déjà View K=16 depth, GT from E1 (5 cm mesh). Only
  pixels where the GT ray hits the mesh. Edge maps (both pixels of a changing 4-neighbour pair are marked):
  GT object boundaries (GT id changes, unlabeled is its own id, "SPLIT" = unlabeled), GT depth edges (relative depth
  jump > 5%), "confirmed" GT depth edges (Canny 50/150 image edge within 2 px; removes some mesh artifacts without
  using SAM 3 or Déjà View), Déjà View depth edges (> 5%, and at a lower threshold, 1.2% in both scenes, chosen so
  they are as dense as the SAM 3 boundaries: same edge budget), SAM 3 boundaries (id changes, also to "no mask").
  Coverage at tolerance d = share of target edge pixels with an edge pixel within d px. Control: SAM 3 boundaries of
  the next view (same density, unrelated position). Déjà View error: |s * depth - GT| / GT with per-view median scale
  s. Bleeding pixel: error > 5% within 3 px of a GT depth edge. "Which object": a pixel is "right" if the SAM 3 mask
  containing it has the pixel's own GT object as its dominant GT object (labeled GT pixels only).
- **Code / command:** `uv run python experiments/scripts/e18_boundaries.py > experiments/results/e18_boundaries.txt`
  (CPU, ~1 min for both scenes).
- **GT resolution caveat:** 5 cm of mesh is 7.5 px (kitchen) / 5.8 px (office) at the median GT depth (10-90%: 3-14 px).
  GT outlines and depth edges are only accurate to a few px; the 1-3 px numbers below are limited by the GT. The
  example figure shows mesh artifacts counted as depth edges and "bleeding": inside monitor screens, on the copier
  lid, on the office chair's mesh back and the black bottle crates (dark / glossy surfaces). The Canny filter keeps
  most of them (textured or reflective), so it changes the numbers only a little.
- **Results** (`results/e18_boundaries.txt`, kitchen / office):
  1. **SAM 3 boundaries vs GT object boundaries:** precision / recall 0.61 / 0.72 and 0.65 / 0.63 at 1 px, 0.75 / 0.84
     and 0.82 / 0.76 at 3 px (F 0.66 / 0.64 at 1 px, 0.80 / 0.79 at 3 px); control F 0.10 / 0.14 at 1 px, 0.17 / 0.25
     at 3 px. Near depth edges, SAM 3 boundaries lie a median 1.0 px from the GT depth edge (as do Déjà View's depth
     edges) and 2.8 / 2.0 px from Déjà View's depth edge: they follow the image, not the predicted depth.
  2. **Signed offset of matched masks:** within 3 px of the GT outline the masks miss 2.4x more object pixels than
     they add outside (ratio 0.40 / 0.42): SAM 3 masks are slightly thinner than the mesh objects, not fatter (part of
     it is probably the coarse mesh thickening objects).
  3. **Bleeding exists on ScanNet++ too:** Déjà View median error 4.2% / 3.3% within 1 px of a GT depth edge (share
     > 5%: 0.45 / 0.41), 0.6% / 0.5% more than 15 px away; 60% / 67% of all pixels with error > 5% lie within 3 px of a
     GT depth edge (43% / 54% for confirmed edges). Same picture as Peter's ETH3D numbers, but here part of it is GT
     error (above).
  4. **Locating depth edges** (confirmed GT depth edges covered at 3 px): GT object boundaries 0.70 / 0.94, SAM 3
     0.67 / 0.78, Déjà View depth edges (5%) 0.50 / 0.63, **Déjà View depth edges at SAM 3's density 0.78 / 0.84**,
     SAM 3 or Déjà View (5%) 0.75 / 0.86, control 0.17 / 0.25. Bleeding pixels covered at 3 px: SAM 3 0.60 / 0.72, dense
     Déjà View edges 0.76 / 0.85. At the same edge budget **Déjà View's own depth discontinuities find depth edges
     better than SAM 3 boundaries**: a blurred edge is still a jump of > 1.2%. Only 55% (kitchen) / 87% (office) of the
     GT depth edges lie on a GT object boundary at all (kitchen: shelves with contents, crates, parts of the copier).
  5. **Which object SAM 3 assigns a pixel to** (confirmed edges; with all edges within 0.04):

     | pixels | kitchen right / other / no mask | office right / other / no mask |
     |---|---|---|
     | bleeding | 0.72 / 0.08 / 0.19 | 0.39 / 0.23 / 0.38 |
     | all <= 3 px from an edge | 0.79 / 0.05 / 0.16 | 0.56 / 0.14 / 0.30 |
     | > 15 px from an edge | 0.96 / 0.01 / 0.03 | 0.83 / 0.05 / 0.12 |

     By GT class: foreground objects are mostly right (kitchen copy machine 0.91, microwave 0.93, trash bin 0.87;
     office chair 0.69, monitor 0.64, coat rack 0.71); the background side of edges is mostly wrong or uncovered
     (office wall 0.14 right / 0.50 other, table 0.19 / 0.36, floor 0.05 / 0.46). Pedestal fan (office, 9% of the
     bleeding pixels): 0.92 no mask; it is not in top100 (the E16 VLM list found it). At background-side "other"
     pixels both SAM 3 (mask) and Déjà View (depth) side with the foreground against the mesh, so some of these may be
     mesh errors; this cannot be separated without finer GT.
  6. **Eroding masks does not fix the wrong assignments:** office bleeding pixels, erosion 0 / 1 / 2 / 3 px: right
     0.39 / 0.27 / 0.19 / 0.14, other 0.23 / 0.14 / 0.10 / 0.07 (right:other 1.7 -> 2.0); kitchen right 0.72 -> 0.39,
     other 0.08 -> 0.03. Erosion turns right assignments into "no mask" about as fast as wrong ones: the wrong ones come
     largely from whole masks on the wrong object (desk-block "table" mask, merged wall planes, E11), not a thin rim.
- **Figures:** `figures/e18_boundaries.jpg` (coverage of confirmed GT depth edges and of bleeding pixels vs tolerance,
  Déjà View error vs distance to an edge, which object SAM 3 assigns pixels to); `figures/e18_example.jpg` (view with
  the most bleeding pixels per scene: GT depth edges vs SAM 3 boundaries; Déjà View depth edges and bleeding pixels).
- **Conclusion:**
  - SAM 3 boundaries are accurate where they exist (~1 px from the mesh outline, F 0.8 at 3 px, following the image
    rather than the blurred depth), and bleeding occurs on ScanNet++ too. But **for locating depth edges, segments add
    little over Déjà View's own depth**: at the same density its depth discontinuities cover more depth edges and more
    bleeding pixels. The union of both is the best edge map we have.
  - **What segments add is which object a pixel belongs to**, which a refiner needs to pull a bled pixel onto the right
    surface. This works on the foreground side (0.64-0.93 right for chairs, monitors, appliances) and in the kitchen
    overall (0.72), but not in the office (0.39 right, 0.23 wrong object), mainly on the background side (wall / table /
    floor) and for unprompted objects (fan).
  - For the refiner: use segments as soft evidence of which surface a pixel belongs to, trusted more on the object
    side of an edge than on the background side; do not use them as the only edge map; erosion is not a cheap fix.
    Whether this is good enough only the refinement experiment can decide (no / SAM 3 / GT segments).
  - The 1-3 px numbers are limited by the 5 cm mesh. A sharper check needs laser-depth GT (Peter's ETH3D scenes, or the
    ScanNet++ laser point cloud, not downloaded).

## E19: Arm D: top100 + VLM prompts in one SAM 3 run (27.10.2026)
- **Goal:** E17 rejected the VLM list for its precision, but E18 found that unprompted objects matter for bleeding:
  the office pedestal fan (not in top100, found by the VLM list) holds 9% of the office bleeding pixels, 0.92 of them
  without a mask. Does the union of both lists keep top100's scores and add the VLM's objects? Plan item 2, arm D.
- **Settings:** SAM 3 image mode, threshold 0.3, prompts = `top100.txt` followed by the scene's E16 VLM list
  (`--prompts` twice; duplicates dropped, so names in both keep the top100 entry; every name its own class, no
  synonym mapping): 125 prompts kitchen / 133 office. One run, not a merge of the two caches: SAM 3 resolves
  overlapping masks by score across all prompts. Association as the working setting (Déjà View geometry,
  `--cross-class 0.7 --min-visible-fraction 0.25`, (b) on = `--keep-linked-below 0.5`, off = `--min-score 0.5`).
  Same 20 views and scorer. The E18 bleeding analysis was also run on the VLM-only and the union segmentation.
- **Code / commands** (this commit):
  - `HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True uv run sam3-segment --mode image --threshold
    0.3 --prompts data/scannetpp/metadata/semantic_benchmark/top100.txt --prompts
    <scene>/vlm-qwen3vl2binstruct_views_n20_uniform_504.txt <scene>/views_n20_uniform_504.npz` per scene
    (`results/e19_sam3image_top100+vlm_t0.3.txt`).
  - `uv run python experiments/scripts/e19_union.py > experiments/results/e19_union.txt`: associates and scores
    top100, VLM and union with (b) off / on; the 8 top100 / VLM runs are reassociated and asserted identical to
    their E15 / E16 / E17 caches (all 8 identical).
  - `experiments/scripts/e18_boundaries.py` now takes an optional segmentation stem and figure prefix (default
    unchanged; rerun without arguments reproduces `results/e18_boundaries.txt` exactly):
    `PYTHONIOENCODING=utf-8 uv run python experiments/scripts/e18_boundaries.py <stem> e19_{vlm,union}` ->
    `results/e19_boundaries_{vlm,union}.txt`, `figures/e19_{vlm,union}_{boundaries,example}.jpg`.
  - `breakdowns.py precision|ids <scene> <union stem> <both prompt files>` -> `results/e19_{precision,ids}_union_<scene>.txt`.
  - `uv run python experiments/scripts/e19_figure.py` -> `figures/e19_union.jpg`; `score_table.py` rerun: old rows
    unchanged, 8 rows added.
- **Runtime:** SAM 3 346 s kitchen / 372 s office (top100 259-262 s, VLM 131-169 s), peak GPU 2.11-2.13 GB.
  Masks 1277 / 1827 (top100 925 / 1099, VLM 1064 / 1508).
- **Results** (`results/e19_union.txt`, all classes, working setting (b) on):

  | scene | vocabulary | pixel recall | recall | precision | cross-class leakage | consistent | id switches | merges |
  |---|---|---|---|---|---|---|---|---|
  | kitchen | top100 | 0.95 | 0.91 | 0.50 | 0.007 | 0.82 | 8 | 1 |
  | kitchen | VLM | 0.94 | 0.89 | 0.44 | 0.007 | 0.83 | 7 | 1 |
  | kitchen | top100 + VLM | 0.94 | 0.91 | 0.39 | 0.007 | 0.84 | 7 | 1 |
  | office | top100 | 0.83 | 0.73 | 0.48 | 0.048 | 0.75 | 16 | 4 |
  | office | VLM | 0.83 | 0.71 | 0.40 | 0.051 | 0.72 | 15 | 5 |
  | office | top100 + VLM | 0.84 | 0.75 | 0.33 | 0.052 | 0.71 | 18 | 5 |

  Which object SAM 3 assigns bleeding pixels to (E18 metric, confirmed GT depth edges; right / other object / no mask):

  | scene | top100 (E18) | VLM | top100 + VLM |
  |---|---|---|---|
  | kitchen | 0.72 / 0.08 / 0.19 | 0.73 / 0.09 / 0.19 | 0.74 / 0.09 / 0.17 |
  | office | 0.39 / 0.23 / 0.38 | 0.46 / 0.24 / 0.29 | 0.48 / 0.26 / 0.25 |

- **Where the bleeding gain comes from** (office, by GT class, right object top100 -> VLM -> union): pedestal fan
  0.02 -> 0.88 -> 0.88, coat rack 0.71 -> 0.88 -> 0.88; monitor, office chair, bookshelf, ceiling lamp unchanged;
  the background side does not improve (wall 0.14 / 0.13 / 0.14 with "other" 0.50 -> 0.56 -> 0.58, table 0.19-0.20,
  floor 0.02-0.06).
- **Precision** (`results/e19_precision_union_*.txt`): the extra unmatched masks are near-synonym duplicates and
  parts (computer / tv on monitors, cable / cord on tables, socket / electrical outlet / wall outlet on walls,
  door on storage cabinets, shelf inside bookshelves), mostly inside one GT object, so not harmful for refinement
  (cross-class leakage +0.004 office, 0 kitchen).
- **Merges** (`results/e19_ids_union_*.txt`): office = the 4 top100 merges (desk block over 3 tables and over 2
  tables, two storage cabinets, two wall planes; now under the class "desk") + the VLM's fan + adjacent GT "object".
  Kitchen: the same single box merge.
- **Figure:** `figures/e19_union.jpg` (scores per vocabulary and scene; which object bleeding pixels get).
- **Conclusion:**
  - The union adds little over the VLM list alone (pixel recall +0.00 / +0.01, bleeding pixels in the right object
    +0.01 / +0.02) and costs the most precision (-0.05 / -0.07 vs VLM); for instance-level scores top100 stays best
    (E17).
  - For the refinement-relevant metric the VLM objects matter: in the office the share of bleeding pixels in the
    right object rises 0.39 -> 0.48, entirely from two objects top100 does not name (pedestal fan, coat rack). The
    share in a wrong object rises a little too (0.23 -> 0.26).
  - So whether the VLM prompts are worth it depends on what the refiner uses: if segments only cut smoothing (more
    masks = fewer wrong ties), the union / VLM list is better; if it ties pixels across views within a segment,
    the extra fan merge and near-synonym fragments are a (small) risk. Keep top100 as the default for now, and
    run the refiner experiment with both top100 and the union to decide. Two scenes; the gain is two objects.

## E20: Proxy test: can segments fix Déjà View's bleeding? (27.10.2026)
- **Goal:** first number on whether segments improve geometry, before a refiner with segments exists. E18 showed
  bleeding at depth edges and that SAM 3 puts bleeding pixels in the right object 0.72 (kitchen) / 0.39 (office).
  Here the Déjà View depth near edges is replaced by a fit to nearby pixels of the same segment, with no / SAM 3 /
  GT / shuffled segments. **This is a crude stand-in, not the planned refiner (no multi-view term, no energy,
  no Huber); say so when presenting.**
- **Settings (fixed before running, not tuned on these scenes):**
  - Fix: for each region pixel, fit a plane in inverse depth (1/z is linear in pixel coordinates on a plane) or the
    mean inverse depth to the same-segment pixels outside the region in a 21 x 21 px window; >= 20 such pixels needed,
    else the pixel keeps its Déjà View depth (as do pixels with no segment). Sensitivity: windows 11 / 21 / 41 px.
  - Regions: **realistic** = <= 3 px from Déjà View's own depth jumps at E18's dense threshold (1.2%), no GT used;
    12% / 17% of the pixels, contains 66% / 81% of the bleeding pixels, and many correct ones. **Oracle** = E18's
    bleeding pixels (error > 5% within 3 px of a GT depth edge, all edges); the fit then excludes both regions.
  - Segments: none (one segment for the image = local smoothing), SAM 3 top100 and top100 + VLM union (working
    setting, E17 / E19), GT instances (SPLIT = none), shuffled (top100 masks of the next view).
  - Error as E18: |s * depth - GT| / GT, per-view median scale s; evaluated where the GT ray hits the mesh.
- **Code / command:** `uv run python experiments/scripts/e20_proxy_fix.py > experiments/results/e20_proxy_fix.txt`
  (CPU, ~1 min; rerun gives identical output).
- **Caveat:** GT depth and GT instances come from the same 5 cm mesh, so GT segments line up with the GT depth
  edges by construction; the GT row is optimistic. The oracle region is selected by error > 5%, so any change
  lowers its error (regression to the mean): compare with the none / shuffled rows, not with Déjà View.
- **Results** (`results/e20_proxy_fix.txt`, plane fit, 21 px; kitchen / office):

  | segments | realistic: share > 5% | realistic: mean error | oracle: changed px, mean error after (Déjà View before) |
  |---|---|---|---|
  | Déjà View (input) | 0.208 / 0.231 | 0.047 / 0.070 | - |
  | none | 0.392 / 0.523 | 0.069 / 0.113 | 0.155 (0.173) / 0.226 (0.255) |
  | SAM 3 top100 | 0.219 / 0.243 | 0.046 / 0.068 | 0.139 (0.162) / 0.182 (0.228) |
  | SAM 3 union | 0.219 / 0.247 | 0.046 / 0.068 | 0.138 (0.160) / 0.195 (0.237) |
  | GT | 0.208 / 0.211 | 0.043 / 0.056 | 0.109 (0.153) / 0.077 (0.218) |
  | shuffled | 0.364 / 0.445 | 0.066 / 0.105 | 0.158 (0.175) / 0.224 (0.244) |

  1. **Without segments the fix is harmful:** local smoothing near Déjà View's depth jumps nearly doubles the share of
     pixels with error > 5% and raises the mean error +46% / +62% (it mixes both sides of the edge). Shuffled segments
     are almost as bad. Agrees with Peter's note that segment-free smoothing never beats the input.
  2. **SAM 3 segments make it safe but nearly neutral:** mean error -2% / -2%, share > 5% +0.01. GT segments: -10% /
     -19%, share > 5% equal / -0.02. SAM 3 gets ~25% (kitchen) / ~10% (office) of the GT gain. top100 = union.
  3. **Why SAM 3 is neutral** (realistic region, by E18's which-object label of the top100 mask): right object (86% /
     63% of the labeled region pixels): mean error 0.041 -> 0.039 / 0.043 -> 0.036; other object (3% / 13%):
     0.142 -> 0.160 / 0.115 -> 0.138; no mask (12% / 24%): unchanged (0.074 / 0.110, the worst pixels). The gains on the
     object side are cancelled by wrong-object masks, and the largest errors sit where there is no mask.
  4. **Oracle region** (same pixels changed by none and SAM 3): SAM 3 beats none (0.139 vs 0.144 / 0.182 vs 0.198) and
     shuffled; GT is far better in the office (0.083 on the pixels all three change). On right-object bleeding pixels
     SAM 3 lowers the office error 0.185 -> 0.125 (share > 5% 1.00 -> 0.75).
  5. **Mean vs plane fit:** the mean fit is worse for every segmentation in the realistic region (share > 5% 0.28-0.34
     even with GT): edges lie on slanted surfaces, a plane is needed. Window: 11 px changes few pixels (all rows ~
     input), 41 px hurts SAM 3 (0.27) and none (0.57-0.67) more, GT stays best.
- **Figures:** `figures/e20_proxy_fix.jpg` (share > 5% and mean error per segmentation, both regions and scenes);
  `figures/e20_example.jpg` (error maps of the office view with most bleeding: Déjà View, none, SAM 3, GT).
- **Conclusion:**
  - Segments are what makes edge-aware smoothing possible at all: without them (or with wrong ones) it damages the
    depth. This supports using segments to cut smoothing.
  - With SAM 3 masks this simple fix does not yet reduce bleeding measurably (-2%), while GT segments do (-10 / -19%):
    the headroom from better segments is real, but limited by two things E18 already showed: wrong-object masks on the
    background side (office 13% of the region pixels, where the fix makes it worse) and no mask (office 24%, the
    worst pixels).
  - Next levers: (i) trust a mask only on the object side of an edge / down-weight background-side masks;
    (ii) class-agnostic masks for the uncovered pixels (option B); (iii) the real refiner (multi-view term, Huber,
    confidence) may do better than this local proxy; run it with the same no / SAM 3 / GT / shuffled rows.

## E21: Class-agnostic SAM 2 masks for the pixels without a SAM 3 mask (27.10.2026)
- **Goal:** E20: the pixels without a SAM 3 mask are the worst ones near edges (office 24% of the edge region) and no
  segment-based fix can touch them. Plan item 4 / option B: class-agnostic automatic masks for these regions. Measure
  how much they cover, whether they put pixels in the right object, and whether the E20 proxy fix then improves.
- **Settings (fixed before running, not tuned):** `facebook/sam2.1-hiera-small` via the `transformers`
  mask-generation pipeline (SAM 3 has no automatic mode in `transformers`), SAM 2's own automatic-mask defaults:
  32 x 32 points, predicted IoU >= 0.8, stability >= 0.95, no crops (the pipeline's default IoU 0.88 is SAM 1's).
  float32 (bf16 breaks the pipeline's NMS: box and score dtypes differ); 1.5 GB peak, ~6 s/view on the laptop.
  Overlapping automatic masks painted large to small (finer wins); own id per mask and view, no class, no
  association. Fill = working top100 segmentation unchanged, automatic ids only where it has no mask. Control: fill
  with the automatic masks of the next view. Regions, which-object label and proxy fix exactly as E20 (plane, 21 px).
- **Code / command:** `uv run python experiments/scripts/e21_automatic_masks.py generate` (GPU, ~2 min/scene, cache
  `sam2auto-hiera-small_views_n20_uniform_504.npz` per scene), then
  `uv run python experiments/scripts/e21_automatic_masks.py > experiments/results/e21_automatic_masks.txt` (CPU,
  ~2 min; rerun identical). `e20_proxy_fix.py` now has a `main()` so E21 can import it (E20 output unchanged).
- **Results** (`results/e21_automatic_masks.txt`, kitchen / office):
  1. **Coverage is low:** median 12 / 19 automatic masks per view, covering 70% / 42% of the pixels (SAM 3: 95% /
     85%) and only **33% / 22% of the pixels SAM 3 leaves without a mask**.
  2. **Realistic region** (no GT): share with a mask, which object (E18 label, labeled pixels), mean error after the
     proxy fix:

     | segments | with a mask | right | other | none | mean error after fix |
     |---|---|---|---|---|---|
     | Déjà View (input) | | | | | 0.047 / 0.070 |
     | SAM 3 top100 | 0.86 / 0.76 | 0.86 / 0.64 | 0.03 / 0.13 | 0.11 / 0.23 | 0.046 / 0.068 |
     | automatic only | 0.42 / 0.31 | 0.40 / 0.28 | 0.02 / 0.04 | 0.58 / 0.68 | 0.047 / 0.069 |
     | top100 + automatic | 0.89 / 0.80 | 0.88 / 0.67 | 0.03 / 0.14 | 0.09 / 0.19 | 0.046 / 0.068 |
     | top100 + next view's automatic | 0.94 / 0.85 | 0.89 / 0.68 | 0.06 / 0.18 | 0.05 / 0.14 | 0.047 / 0.069 |
     | GT | | | | | 0.043 / 0.056 |

  3. **On the no-mask pixels themselves** (14% / 24% of the realistic region): the fill changes only 7% / 9% of them
     (the fit needs >= 20 same-segment pixels outside the region within 21 px; fill pieces next to edges are small),
     mean error 0.064 -> 0.065 / 0.112 -> 0.112; the next-view control is about the same. Oracle region: changes 1% /
     11%, error unchanged. Of the labeled no-mask pixels, 81% / 82% still have no mask after the fill, 13-15% get the
     right object, 4-5% another one.
  4. **What the gaps are** (no-mask pixels of the realistic region): 57% / 37% within 1 px of a SAM 3 mask, 82% / 63%
     within 3 px, 98% / 90% within 10 px: mostly **slivers along mask outlines** (E18: SAM 3 masks are slightly thinner
     than the objects), which an automatic mask does not fill either. The rest are uncovered objects; GT class of the
     office bleeding gaps: pedestal fan 0.20 (not in top100; E19's VLM list finds it), table 0.10, office chair 0.09,
     wall 0.09, floor 0.08, monitor 0.08; kitchen: bottle crate 0.15, copy machine 0.12, ceiling lamp 0.12.
- **Figures:** `figures/e21_automatic_masks.jpg` (which object and proxy-fix error per segmentation, both regions and
  scenes); `figures/e21_example.jpg` (office view 14: SAM 3, automatic, filled; dark = no mask, magenta = bleeding).
- **Conclusion:**
  - Class-agnostic SAM 2 masks with default settings do not help: they cover a fifth to a third of the gaps and the
    proxy fix does not improve (= control). Not worth a format change in `Segmentation` for now.
  - The gaps are mostly 1-3 px slivers along SAM 3 outlines, i.e. exactly the pixels between two surfaces. Filling
    them needs a decision which side a pixel belongs to, which a mask cannot give; this points back to the refiner
    (assign edge pixels by depth agreement with either side, multi-view evidence) rather than to more masks.
  - The rest are whole objects without a mask (pedestal fan, crates, lamps): a vocabulary problem (VLM / union, E19).
  - Caveat: one model size and its default settings; denser points or crops would cover more, but that would be
    tuning on these test scenes.
