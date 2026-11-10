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
| E22 | Side decision: edge pixels take the nearby segment plane closest to their Déjà View depth (instead of only their own mask) | harmful for every segmentation, GT included (realistic mean error SAM 3 0.068 -> 0.093 office, GT 0.056 -> 0.091; kitchen 0.046 -> 0.062): Déjà View's own depth at an edge does not tell the side. Even the GT-chosen best candidate is worse than own mask on the realistic region; on bleeding pixels it would help (office 0.238 -> 0.220 SAM 3). Gap pixels need another cue (multi-view / image), the own mask stays the better side cue |
| E23 | Déjà View paper's point-map metric on our scenes (one Sim(3) per scene) | Rel. L2 0.024 / 0.026, inlier ratio (< 3%) 82-84% / 86-89% vs paper 0.015 / 93.3%: same order, somewhat worse (5 cm mesh GT, plain least-squares alignment, our 20 views; normaliser unclear). Error is uneven over views (worst 0.061 kitchen v12, 0.070 office v5); per-view alignment helps the kitchen (0.017, 92%) more than the office -> kitchen has cross-view drift |
| E24 | 3D point-cloud overlap from Replicate Any Scene (RAS) instead of our reprojection overlap, same linking | not adopted: consistency collapses 0.82 -> 0.27 kitchen / 0.75 -> 0.18 office (id switches 8 -> 100 / 16 -> 112) for fewer merges (1 -> 0 / 4 -> 1); still 0.47 / 0.34 with GT geometry, so the rule, not Déjà View, is the cause: single-view masks see different parts of an object and the cloud rule has no visibility test. RAS's either-direction rule: worse on both (consistent 0.34 / 0.45, merges 4 / 5, wrong links 14 / 27). Reprojection + vf0.25 stays |
| E25 | Progressive VLM vocabulary (RAS scheme: one view at a time, list so far shown, only new names), Qwen3-VL-2B | the 2B model copies the list instead of adding to it (16 of 20 views add nothing): 12 / 22 names vs 46 / 60 (union). Short list -> precision 0.45 -> 0.69 / 0.40 -> 0.60, but recall 0.89 -> 0.82 / 0.72 -> 0.64 (pixel recall -0.02); synonyms stay (socket / power outlet, chair / office chair, all from view 1); loops still 8 / 3 of 20. Scheme untested with a model that follows the instruction -> step 2 |
| E26 | Bigger VLM: Qwen3.8-27B (4-bit, thinking off, greedy) with both schemes (per-view union, progressive) | greedy decoding makes the 27B model loop into word association ("...velocity, acceleration, force...", "...HTTP, FTP, SSH...") in 4-8 of 20 answers -> 389 / 506 names (union), 249 / 371 (progressive); precision collapses to 0.15-0.26 (top100 0.51 / 0.48), pixel recall 0.85-0.95 / 0.88-0.90 (top100 0.95 / 0.83), consistency and merges ~unchanged. Outside the loops it follows "only new names" (13-16 of 20 views add <= 1 name, correct NONE answers), unlike the 2B model. Lists as produced are unusable; decoding (repetition control) is the next variable |
| E27 | Qwen3.8-27B as E26 + repetition penalty 1.1 (still greedy), both schemes | loops not fixed: still 3-7 of 20 answers run into word association (truncated 3 / 7 union, 5 / 4 progressive vs 5 / 8 and 5 / 4); the penalty moves the drift to other words (verbs, pronouns, utensils, physics terms) and removes some loops while creating others (kitchen view 1 now loops). Lists 265 / 476 (union), 323 / 236 (progressive); precision 0.23-0.28 (E26 0.15-0.26, top100 0.51 / 0.48), pixel recall 0.88-0.95, consistency / merges ~E26; kitchen union cross-class leakage 0.033 (junk prompt "workstation"). Repetition penalty is not the fix; next: cap the answer (deterministic) or Qwen's sampling |
| E28 | Qwen3.8-27B as E26 with answers capped at 64 new tokens (greedy), both schemes | best 27B variant so far: lists 90 / 79 (union) and 96 / 91 (progressive) instead of 389 / 506 and 249 / 371; union answers are exactly the first 64 tokens of E26's (20/20). Union cap64: pixel recall 0.94 / 0.89 (top100 0.95 / 0.83), recall 0.91 / 0.75, consistent 0.84 / 0.76, merges 1 / 5, cross-class leakage 0.008 / 0.048 (all ~top100 or better), but precision 0.29 / 0.33 (top100 0.51 / 0.48) from abstract names at loop starts ("edge, corner, surface, front, back"). Progressive: same picture, its few loops dump ~25 junk names each |
| E29 | Hypersim loader: views + GT from Hypersim's per-pixel maps, three val scenes (kitchen, conference room, restaurant) | correct to float16 precision (lifted points vs Hypersim's position map: median 1.2 mm); most Hypersim cameras are tilt-shift (254 + 31 of 482 scenes), handled exactly as K + camera rotation, skewed ones rejected; walls / floors / ceilings have no instances -> one per class; conference room 23% unlabeled (behind glass). Per-view GT re-associated with GT geometry: consistent 1.00 / 1.00 / 0.98, 0 merges (working setting 0.94 / 0.91 / 0.94) |
| E30 | First Hypersim run of the segmenter: six val scenes (3 tilt-shift, 3 plain cameras), top100 vs Qwen3.8-27B union cap64, Déjà View vs GT geometry | Déjà View always predicts a centred principal point: on plain cameras depth 0.45-0.87%, Rel. L2 0.010-0.026 and association consistency ~ GT geometry (0.82 / 0.70 / 0.81 vs 0.82 / 0.76 / 0.81); on tilt-shift scenes consistency drops 0.12-0.31 (conference room Rel. L2 0.116). VLM list not better than top100 outside ScanNet++: precision lower everywhere (0.10-0.43 vs 0.23-0.49), part names split chairs (conference consistent 0.55 -> 0.36), the name "lobby" merges wall and floor (cross-class leakage 0.188). Protocol: plain-camera scenes; top100 stays default |
| E31 | Rank the 20 bedroom views (Hypersim ai_006_007) by Déjà View's bleeding at GT depth edges, to choose the shared visual views | bleeding pixels 0.7-6.4% of the image; top views 18, 19, 0, 12; median error near edges 1.8-3.0% vs 0.3-0.9% far (as E18 / ETH3D). Bleeding sits on thin structures in front of distant surfaces (ornate bed frame, vanity mirror frame, chair and stool legs); planes stay below 1%. Visual views: frames 94, 0, 62 + 31 (overview) |
| E32 | Visual of the segmenter on the four shared bedroom views (matched colours instead of id numbers, error map) | right object 0.83-0.92 of labeled pixels, other object 0.01-0.03, no mask 0.06-0.16; shared objects keep one colour across views. Harmful errors sit at the bed frame: the bed mask fills the gaps between the bars and takes the wall / floor behind them, exactly where Déjà View bleeds (E31). Largest gaps: mirror and floor mat never masked |
| E35 | Learned single-view depth refiner (issue #11), with vs without SAM 3 masks, trained on 40 Hypersim train scenes | dev Chamfer 3.50 -> 3.20 cm with masks (3.29 without), better on all 3 scenes in accuracy and completeness; old protocol read it as no gain (F-score 0.844 -> 0.845); one seed, val not run |
| E39 | Faster panoptic labeling (issue #29): prompts encoded once, detector batched 4 prompts at a time, overlaps on the GPU, association in torch | bedroom on an RTX 3080: SAM 3 29.7 -> 16.6 s, association 1.94 -> 0.57 s, pipeline 37.1 -> 22.6 s; dev mIoU / PQ^sc / AP 0.360 / 0.292 / 0.216 unchanged; 99.85-99.98% of pixels keep their mask (bf16 rounding of the batched detector), linked ids from the torch association bit-identical |
| E40 | Remove or justify every hand-set constant in the energy (issue #28): one-at-a-time dev sweep, val check | λ_d and conf_power removed (no-ops); no Huber, no border margin and E_n cut at Otsu's threshold each tie on dev, but combined they lost 0.5 cm in val's ai_009_009 (val means beyond noise), so all three stay with that reason; λ_mv, λ_δ, neighbors and the cut window stay (alternatives trade accuracy for completeness); refiner unchanged within run-to-run spread |
| E41 | After E40's review: Gauss-Newton 5 x 10 (chosen in #10, never applied), u counting the pixel's own view, λ_δ = 0 tried | own-view u changes no score by more than 0.007 cm; λ_δ = 0 looks better at 5 x 10 only because 4 of 6 solves stop early, converged it is worse (dev completeness +0.46 cm), so λ_δ stays; dev Chamfer 3.953 -> 3.928, val 5.723 -> 5.643 (ai_009_009 -0.81 cm; without it -0.006) |
| E42 | E_n as the angle between DSINE's normals and the closer-neighbor normals of the depth (Peter's proposal), 3 weights | λ_a 4e-4 passes dev (accuracy +0.010, completeness -0.229, Chamfer -0.110 cm vs BiNI) but fails val on accuracy (+0.090, worse on 10 of 11) while completeness is better on 11 of 11 and Chamfer -0.187; the λ_mv / λ_δ trade again, so BiNI stays; adopting it is a choice of metric, left open |
| E43 | The normal term as tangent residuals (n_DSINE . t / \|t\| for the closer-neighbor tangents), E42's simpler form | λ_t 4e-4 passes dev (+0.025 / -0.249 / -0.112 cm vs BiNI) and fails val on accuracy (+0.102, worse on 10 of 11; completeness -0.506, Chamfer -0.202): within a few thousandths of E42's angle term on dev, the same trade; BiNI stays |
| E44 | E_mv reads the other view at its nearest pixel, as u does (no interpolation, no 1 px border, no four-corner check) | dev ties (+0.011 / -0.016 / -0.002 cm), 1.9 -> 1.4 s per scene; val worse on all three (+0.048 / +0.105 / +0.077, median scene +0.015 / +0.051 / +0.026), so reverted: E_mv stays bilinear |

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

## E22: Side decision for edge pixels: which nearby segment's surface? (27.10.2026)
- **Goal:** E20 fitted each edge pixel only to its own mask; E21 showed the pixels without a mask are mostly 1-3 px
  slivers along SAM 3 outlines, i.e. pixels between two surfaces. Test the obvious single-view side decision: every
  segment near the pixel is a candidate, the pixel takes the candidate surface closest to its own Déjà View depth.
  Question: does SAM 3 then move towards GT's gain? Same crude local proxy as E20 (not the refiner).
- **Settings (fixed before running, not tuned):** candidates = all segments with >= 20 interior pixels (outside the
  region) in the 21 px window, also for pixels with no mask; plane in inverse depth per candidate exactly as E20;
  pick the plane closest to the pixel's Déjà View inverse depth (absolute difference); pixels without a candidate keep
  Déjà View. Rows per segmentation (none, SAM 3 top100, union, GT, shuffled): own mask (= E20, reference), side
  decision, oracle pick (candidate with the lowest GT error = best possible choice among the candidates; uses GT).
  Regions, error and which-object label as E20. With "none" there is one candidate, all rules = E20 none (checked).
- **Deviation after the first run (stated openly):** the first run picked the oracle candidate by absolute
  inverse-depth difference to GT, not by the evaluated error; it came out worse than own mask, which looked like a bug.
  Fixed to the evaluated error, and a second side rule "side decision, log" (closest by |log(plane / Déjà View)|,
  scale-free) added. Both changed almost nothing (oracle 0.0799 -> 0.0798, log vs abs <= 0.002), so that was not the
  cause; the pre-registered rule is reported as the main row. Checked on office view 3: on pixels own mask changes,
  the oracle pick is never worse than own mask (0 / 18471 px) -> no bug; the difference comes from pixels own mask
  leaves alone (no mask or < 20 own-segment pixels nearby).
- **Code / command:** `uv run python experiments/scripts/e22_side_decision.py > experiments/results/e22_side_decision.txt`
  (CPU, ~2 min; imports `e20_proxy_fix.py`).
- **Results** (`results/e22_side_decision.txt`, kitchen / office, mean error / share > 5%):

  | segments | rule | realistic region | oracle region (bleeding px) |
  |---|---|---|---|
  | Déjà View (input) | | 0.047 / 0.208, 0.070 / 0.231 | 0.174 / 1.00, 0.259 / 1.00 |
  | none | (all rules) | 0.069 / 0.392, 0.113 / 0.523 | 0.157 / 0.875, 0.232 / 0.867 |
  | SAM 3 top100 | own mask (E20) | 0.046 / 0.219, 0.068 / 0.243 | 0.159 / 0.897, 0.238 / 0.882 |
  | SAM 3 top100 | side decision | 0.062 / 0.281, 0.093 / 0.301 | 0.184 / 0.906, 0.268 / 0.842 |
  | SAM 3 top100 | oracle pick | 0.054 / 0.249, 0.080 / 0.269 | 0.152 / 0.809, 0.220 / 0.734 |
  | GT | own mask (E20) | 0.043 / 0.208, 0.056 / 0.211 | 0.143 / 0.845, 0.192 / 0.727 |
  | GT | side decision | 0.061 / 0.280, 0.091 / 0.292 | 0.181 / 0.904, 0.274 / 0.834 |
  | GT | oracle pick | 0.053 / 0.250, 0.074 / 0.253 | 0.151 / 0.811, 0.209 / 0.675 |
  | shuffled | side decision | 0.066 / 0.365, 0.104 / 0.434 | 0.160 / 0.885, 0.247 / 0.874 |

  union = top100 within 0.001-0.005 everywhere.
  1. **The side decision by Déjà View depth is harmful for every segmentation, GT included:** realistic mean error
     +31% / +34% vs the input (SAM 3), and GT segments do not rescue it (0.061 / 0.091). On bleeding pixels it is worse
     than doing nothing with segments (none). SAM 3 vs GT makes almost no difference under this rule -> the rule, not
     the segments, is the bottleneck.
  2. **Why:** it does not just snap back to Déjà View (85-91% of the region changes, median depth change 0.7-0.9%,
     2-5% on bleeding pixels), but it picks the wrong surface. On right-object pixels it leaves the own mask for 18-19%
     of the pixels: chosen surface belongs to the pixel's GT object 0.79 / 0.66 vs own mask 0.97 / 0.83 (realistic);
     on bleeding pixels 0.36 vs 0.62 (office). A bleeding pixel's Déjà View depth lies between the two surfaces,
     so it is no evidence which side it belongs to; the extrapolated planes add noise.
  3. **Gap pixels (no top100 mask, 14% / 24% of the realistic region):** side decision makes them worse (office
     0.112 -> 0.141, share > 5% 0.32 -> 0.43). On bleeding gap pixels the right candidate usually exists (oracle pick
     0.318 -> 0.272 SAM 3, 0.248 GT) but Déjà View cannot find it (0.314).
  4. **Oracle pick shows the limit of the candidate set:** in the realistic region even the GT-chosen candidate is
     worse than own mask (office 0.080 vs 0.068 SAM 3): where the own segment has too few pixels nearby, Déjà View is
     better than every nearby plane, so "always replace" is wrong and a keep option is needed. On bleeding pixels a
     good choice would help: SAM 3 oracle pick 0.220 vs own mask 0.238 vs none 0.232 (office).
- **Figures:** `figures/e22_side_decision.jpg` (mean error and share > 5% per segmentation and rule, both regions and
  scenes); `figures/e22_example.jpg` (office view 14: Déjà View, SAM 3 own mask, SAM 3 / GT side decision).
- **Conclusion:**
  - Answer to the question: no, SAM 3 does not move towards GT's gain with this rule, because the rule itself fails
    (also with GT). The single-view Déjà View depth cannot decide the side of an edge pixel.
  - The own mask stays the better side cue for pixels that have one (E20 own-mask rule is the best rule here); for
    gap pixels, and for pixels whose mask is wrong, the side must come from other evidence: other views (the pixel's
    3D point seen away from the edge in another view; the refiner's E_mv), image affinity, or confidence, plus a
    "keep Déjà View" option. This is the refiner's job, not a single-view proxy's.
  - Caveat: crude local proxy; one rule, fixed window; GT segments and GT depth share the 5 cm mesh.

## E23: The Déjà View paper's point-map metric on our two scenes (27.10.2026)
- **Goal:** put our Déjà View output next to the paper's own ScanNet++ numbers (arXiv 2605.30215: Rel. L2 0.015,
  inlier ratio 93.3% at < 3% error, one Sim(3) for the whole scene; view count not stated). Our E8 number (per-view
  median depth scale, 0.7-0.8% median) is more lenient; this is the stricter, comparable one.
- **Settings:** Déjà View K=16 cache, 20 views at 504 px; GT world points = GT depth (5 cm mesh) lifted with the GT
  cameras; every pixel where the GT ray hits the mesh (96% / 94%). Umeyama Sim(3) least squares on all points, no
  outlier rejection. The normaliser of "relative" is not given in our notes, so two are reported: per-ray (distance of
  the GT point from its own camera; origin-free) and first camera (|GT| in view 0's camera frame). Also one Sim(3) per
  view: the gap to the scene-level number is cross-view drift.
- **Code / command:** `uv run python experiments/scripts/e23_pointmap_metric.py > experiments/results/e23_pointmap_metric.txt`
  (CPU, ~1 min).
- **Results** (`results/e23_pointmap_metric.txt`):

  | scene | alignment | Rel. L2 per-ray / first camera | median (per-ray) | inlier ratio < 3% per-ray / first camera |
  |---|---|---|---|---|
  | paper (ScanNet++) | scene Sim(3) | 0.015 | | 93.3% |
  | kitchen | scene Sim(3) | 0.024 / 0.023 | 0.017 | 81.5% / 84.4% |
  | kitchen | per-view Sim(3) | 0.017 / 0.016 | 0.010 | 92.0% / 92.5% |
  | office | scene Sim(3) | 0.026 / 0.025 | 0.012 | 88.8% / 86.3% |
  | office | per-view Sim(3) | 0.025 / 0.022 | 0.009 | 85.1% / 88.5% |

  1. **Same order as the paper, somewhat worse:** Rel. L2 0.024-0.026 vs 0.015, inliers 82-89% vs 93%. Likely
     contributors, not separated here: GT from the 5 cm mesh (not the laser scan; E18: 6-7 px edge blur, mesh
     artefacts), least-squares alignment pulled by outliers, our view selection (uniform over capture order, large
     camera steps in the office), unclear normaliser.
  2. **Errors are uneven over views:** kitchen view 12 0.061, view 16 0.046; office view 5 0.070, view 14 0.045
     (scene alignment); most views 0.011-0.025.
  3. **Kitchen has cross-view drift:** one Sim(3) per view cuts its error 0.024 -> 0.017 and lifts inliers 82% ->
     92%, i.e. the views do not agree on one similarity frame (Peter's ETH3D finding: per-view offsets 3%). Office:
     per-view alignment barely helps (0.026 -> 0.025; inliers even drop 89% -> 85% under the per-ray normaliser,
     since least squares minimises absolute, not relative, error); its error is within views (view 5, 14-16).
- **Figure:** `figures/e23_pointmap_metric.jpg` (mean relative error per view, scene vs per-view Sim(3), paper line).
- **Conclusion:** our Déjà View run is in the paper's range, a bit worse under our (stricter, mesh-based) GT; no sign
  of a broken setup. There is headroom both in single views (a few bad views) and across views (kitchen drift), which
  a multi-view term (E_mv) targets. Caveat: one run, two scenes, metric details guessed.

## E24: 3D point-cloud overlap (Replicate Any Scene) as an association variant (02.11.2026)
- **Goal:** test the overlap rule of Replicate Any Scene (RAS, arXiv 2604.10789, Sec. 3.2, Eqs. 1-2) in our
  association. RAS lifts SAM 3 masks with VGGT depth and compares point clouds in 3D: a -> b = share of a's points
  whose nearest point of b is within tau_a = 3 x mean nearest-neighbour distance inside a's own cloud. No projection,
  so no visibility handling; a monitor hidden behind a chair is simply not near the chair cloud -> could fix the
  E14 chair -> monitor case without our visibility rule (vf0.25).
- **Settings (fixed before running):** same per-view masks (`sam3image-top100-t0.3`), same linking as the working
  setting (cross-class 0.7, `--keep-linked-below 0.5`, greedy, one mask per view, min overlap 0.5), Déjà View K=16
  geometry; only the overlap measure changes. Rows: reprojection + vf0.25 (working setting), reprojection without
  vf, cloud with our two-sided rule (both directions >= threshold), cloud with RAS's own rule (either direction,
  reference only), and reprojection + vf0.25 / cloud with rendered GT geometry (upper bound). Differences to RAS:
  RAS aggregates all masks of a SAM 3 *track* into one cloud (dense video); we have one mask per view, so each cloud
  is a single view's mask. Masks with < 50 lifted points get no links (as `min_visible`). a -> b is estimated on a
  fixed random sample of 1000 of a's points (seed 0); box prefilter (boxes within tau_a).
- **Code / command:** `Associator(overlap_rule="cloud", direction_rule="min"|"max")` in
  `src/segmentation/associate.py` (CLI `--overlap cloud`, `--either-direction`; cache tags `pc`, `or`);
  `uv run python experiments/scripts/e24_cloud_overlap.py > experiments/results/e24_cloud_overlap.txt`
  (CPU, ~12 min; cloud rule ~2 min kitchen / ~3.5 min office per run vs 13 s reprojection). scipy added as a
  direct dependency (was already installed indirectly). Reprojection rows reproduce the E17 caches exactly.
  New output column "wrong links": objects joining per-view masks that lie mostly (> 50%) on different GT objects.
- **Results** (`results/e24_cloud_overlap.txt`, all classes):

  | scene | overlap | geometry | pixel recall | precision | consistent | id switches | merges | wrong links |
  |---|---|---|---|---|---|---|---|---|
  | kitchen | reproject + vf0.25 (working) | DV | 0.95 | 0.50 | 0.82 | 8 | 1 | 5 |
  | kitchen | reproject | DV | 0.95 | 0.50 | 0.84 | 6 | 1 | 5 |
  | kitchen | cloud, both directions | DV | 0.94 | 0.50 | 0.27 | 100 | 0 | 3 |
  | kitchen | cloud, either direction (RAS) | DV | 0.95 | 0.48 | 0.34 | 50 | 4 | 14 |
  | kitchen | reproject + vf0.25 | GT | 0.95 | 0.50 | 0.95 | 2 | 1 | 5 |
  | kitchen | cloud, both directions | GT | 0.94 | 0.50 | 0.47 | 57 | 0 | 3 |
  | office | reproject + vf0.25 (working) | DV | 0.83 | 0.48 | 0.75 | 16 | 4 | 10 |
  | office | reproject | DV | 0.83 | 0.48 | 0.76 | 15 | 6 | 12 |
  | office | cloud, both directions | DV | 0.82 | 0.47 | 0.18 | 112 | 1 | 6 |
  | office | cloud, either direction (RAS) | DV | 0.83 | 0.45 | 0.45 | 42 | 5 | 27 |
  | office | reproject + vf0.25 | GT | 0.83 | 0.48 | 0.80 | 12 | 5 | 12 |
  | office | cloud, both directions | GT | 0.82 | 0.47 | 0.34 | 69 | 1 | 7 |

  1. **Cloud rule with both directions links far too little:** objects 305 -> 412 kitchen, 414 -> 547 office;
     consistency 0.82 -> 0.27 / 0.75 -> 0.18. It does remove almost all merges (1 -> 0, 4 -> 1, only the two
     adjacent storage cabinets remain) and the chair -> monitor link (also gone with vf0.25).
  2. **Not Déjà View's fault:** with GT geometry the cloud rule still reaches only 0.47 / 0.34 (reprojection 0.95 /
     0.80). Cause: two single-view masks of one object see different parts of it (other side, occluded part, image
     border); points of a that b's view never saw have no partner in b's cloud, so a -> b stays low. Reprojection
     only counts points visible in the other view, which is exactly what the cloud rule lacks. RAS avoids this by
     aggregating many frames of a track into one fuller cloud; with sparse single views it breaks.
  3. **tau is tight for Déjà View:** median tau = 2.1% / 1.8% of the mask depth (10-90%: ~1-4 cm in scene units),
     about the size of Déjà View's cross-view disagreement (Peter, ETH3D: 0.4-0.8%; per-view offsets ~3%; E23 kitchen
     drift). This explains the further drop GT -> DV (0.47 -> 0.27, 0.34 -> 0.18).
  4. **RAS's either-direction rule** recovers some consistency (0.34 / 0.45) but merges and wrong links go up
     (4 / 5 merges, 14 / 27 wrong links vs 1 / 4 and 5 / 10): partial masks (door + doorframe, socket + wall,
     whiteboard + eraser, adjacent cabinets/tables) lie inside a bigger cloud. Confirms not adopting that rule.
  5. Side result: with the working setting, GT geometry is now clearly better than Déjà View in the kitchen (0.95 vs
     0.82 consistent; E9 at t0.5 / 16 prompts: 0.79 vs 0.75). More, smaller masks (top100, t0.3) seem more
     sensitive to geometry errors; the office gap is small (0.80 vs 0.75).
- **Figure:** `figures/e24_cloud_overlap.jpg` (pixel recall, consistent, merges, wrong links per setup).
- **Conclusion:** the RAS overlap rule does not transfer to sparse single-view masks; reprojection + vf0.25 stays.
  What RAS and we share is the design (ids resolved in 3D with lifted masks + union-find), not the overlap measure.
  Not tried (would be tuning on the test scenes or a new method): larger tau, or cloud overlap restricted to the
  points visible in the other view (= reprojection again). A track-level variant (merge objects after our
  association by comparing their aggregated multi-view clouds, as RAS does) is a possible later step against
  remaining splits.

## Plan: progressive VLM vocabulary (02.11.2026)
Motivation: E16's VLM vocabulary (one question per view, union of the answers) gives near-synonym duplicates
(E12/E16) and repetition loops (16-18 of 20 answers). Replicate Any Scene (arXiv 2604.10789, Sec. 3.1, Table 2) builds
the list progressively: one frame at a time, the VLM sees the current list and names only new objects, "treating
'sofa' and 'couch' as identical"; in their ablation same recall as per-frame union (91.4 vs 91.3), no redundant names.
Their prompt is not published; ours is written and fixed before any run.
- **Step 1 (E25): same model, new scheme.** Qwen3-VL-2B-Instruct (4-bit), same 20 views in the same order, greedy,
  256 new tokens (as E16). View 1 gets E16's question unchanged; every later view gets E16's question + the current
  list + "name only objects and surfaces NOT already in the list, treat synonyms as the same object (e.g. sofa and
  couch), answer NONE if nothing new". Exact duplicates of listed names are dropped when parsing. Output: a prompt
  file `vlmprog-<model>_<views>.txt`, so SAM 3, association and scorer are unchanged. One change only: the scheme.
- **Run:** student cluster (RTX 5090). VLM for both scenes, then SAM 3 image mode at threshold 0.3 with three
  vocabularies on the same machine (top100, E16 union list, progressive list), since bf16 results differ slightly
  across GPUs. Association locally with the working setting (`--cross-class 0.7 --min-visible-fraction 0.25
  --keep-linked-below 0.5`, Déjà View K=16 geometry), scorer as E17.
- **Report:** list size, near-synonym pairs (listed by hand from the lists), truncated answers, VLM time; pixel
  recall, recall, precision, cross-class leakage, consistent, merges, wrong links (E24 check); top100 / union /
  progressive side by side. Expectation: fewer synonyms -> precision back up towards top100 (E17: VLM -0.06/-0.08),
  fewer loops. Risks: a 2B model may ignore "only new" (then ~ union + dedupe), order dependence.
- **Step 2 (E26), only after E25:** the bigger VLM (TODO Qwen-3.8-27B: check checkpoint name, image input,
  transformers support; 4-bit ~15 GB on the 5090) with whichever scheme wins E25.
- Caveat: the two ScanNet++ scenes are test scenes, top100 is ScanNet++-specific; the real generalisation test is
  Hypersim (no loader yet).

## E25: Progressive VLM vocabulary, Qwen3-VL-2B (02.11.2026)
- **Goal:** step 1 of the plan above: does Replicate Any Scene's progressive list (one view at a time, the VLM sees
  the list so far and names only new objects, synonyms counting as the same) remove E16's near-synonyms and loops
  without losing recall? Same model as E16, only the scheme changes.
- **Settings:** Qwen3-VL-2B-Instruct 4-bit, greedy, 256 new tokens, same 20 views in the same order. View 1 gets
  E16's question; later views get E16's question + "These objects and surfaces are already listed: <list>. Name only
  objects and surfaces in this image that are NOT already listed. Treat synonyms as the same object (for example,
  sofa and couch). If nothing new is visible, answer NONE." (`NEW_ONLY` in `src/segmentation/vocabulary.py`, written
  before the run). Exact duplicates are dropped. SAM 3 image mode, threshold 0.3, with top100 / E16 union list /
  progressive list, all three on the cluster in one job (RTX 5090, job 658); association locally with the working
  setting (xc0.7, vf0.25, kl0.5, Déjà View K=16). Laptop rows (E17 caches) for cross-machine noise.
- **Code / command:** `uv run vlm-vocabulary --progressive <views...>` -> `vlmprog-qwen3vl2binstruct_<views>.txt`
  (+ `.json` with raw answers, new names and time per view). Cluster job `experiments/scripts/e25_vlm_progressive.sbatch`
  (log `results/e25_cluster_job658.txt`; VLM both scenes 87 s incl. load, SAM 3 5-62 s per run, peak 2.8 GB), SAM 3
  caches copied to `<scene>/e25_cluster/`. Scoring: `uv run python experiments/scripts/e25_vlm_progressive.py >
  experiments/results/e25_vlm_progressive.txt` (CPU, ~2 min; rerun identical; laptop rows reproduce the E17 caches).
- **Results** (`results/e25_vlm_progressive.txt`, all classes, cluster runs):

  | scene | vocabulary | names | recall | pixel recall | precision | consistent | GT in >=2 views | merges | wrong links | mixed-class objects |
  |---|---|---|---|---|---|---|---|---|---|---|
  | kitchen | top100 | 100 | 0.92 | 0.95 | 0.51 | 0.82 | 38 | 1 | 5 | 20 |
  | kitchen | VLM union (E16) | 46 | 0.89 | 0.94 | 0.45 | 0.83 | 36 | 1 | 5 | 18 |
  | kitchen | VLM progressive | 12 | 0.82 | 0.92 | 0.69 | 0.85 | 33 | 1 | 4 | 13 |
  | office | top100 | 100 | 0.73 | 0.83 | 0.48 | 0.76 | 51 | 5 | 10 | 22 |
  | office | VLM union (E16) | 60 | 0.72 | 0.84 | 0.40 | 0.73 | 48 | 5 | 11 | 21 |
  | office | VLM progressive | 22 | 0.64 | 0.82 | 0.60 | 0.78 | 41 | 4 | 10 | 12 |

  (mixed-class objects: linked objects whose per-view masks carry more than one class name.)
  1. **The 2B model copies the list instead of extending it.** New names per view: kitchen 8, 0, 2, 0, 0, 0, 0, 1,
     0 ... (16 of 20 views add nothing), office 15, 0, 0, 4, 0, 0, 0, 2, 0 ... (16 of 20). The answers mostly
     repeat listed names ("cabinet | window | paper | surface | door"), i.e. the instruction "only new" is not
     followed; the list is essentially view 1's answer plus a few names.
  2. **Short list -> precision up, recall down:** precision 0.45 -> 0.69 / 0.40 -> 0.60 (above top100's
     0.51 / 0.48), recall 0.89 -> 0.82 / 0.72 -> 0.64; pixel recall only -0.02 (large surfaces and frequent objects
     are in view 1's list). Fewer GT objects are found in >= 2 views (36 -> 33, 48 -> 41), so the slightly higher
     consistency (0.85 / 0.78) is over fewer, easier objects. Merges unchanged.
  3. **Synonyms are not collapsed:** socket / power outlet, chair / office chair (office) and the junk names
     "surface", "paper" (kitchen) all come from view 1, which has no list yet. Fewer mixed-class objects (18 -> 13,
     21 -> 12) come from the shorter list, not from merged synonyms.
  4. **Loops remain** in the long answers (truncated 8 / 3 of 20 vs 18 / 16): fewer, because most answers are short
     copies. VLM time 60 s / 11 s per scene on the 5090 (kitchen view 1 loop 43 s incl. warm-up).
  5. Cross-machine noise (cluster vs laptop, same lists): recall / precision within 0.01, consistent within 0.01-0.04,
     id switches +-2, merges +-1, as on 28.10.
- **Figure:** `figures/e25_vlm_progressive.jpg` (names, pixel recall, precision, consistent, merges per vocabulary).
- **Conclusion:** with Qwen3-VL-2B the progressive scheme is not what RAS describes: the model does not follow "name
  only new objects", so the result is a shorter list (precision/recall trade-off), not a cleaner one. This tests the
  model's instruction following as much as the scheme. Step 2 (bigger VLM) should therefore run **both** schemes
  (union and progressive), not only "the winner" as planned: E25 has no clear winner, and the scheme's idea is only
  tested once the model follows it. Not changed (would be tuning on the test scenes): prompt wording, a synonym rule
  for view 1.

## E26: Bigger VLM (Qwen3.8-27B), both schemes (02.11.2026)
- **Goal:** step 2 of the plan, changed after E25: run the bigger VLM with **both** schemes (per-view union as E16,
  progressive as E25). Questions: does it follow "only new names"? Fewer synonyms? Recall vs top100? Loops?
- **Model check (before running):** `Qwen/Qwen3.8-27B` exists on Hugging Face (Apache-2.0, not gated, 55.6 GB bf16
  safetensors), natively multimodal (`qwen3_5`, `Qwen3_5ForConditionalGeneration`; loads with
  `AutoModelForImageTextToText` in transformers 5.20.2, as `vocabulary.py` already does). Hybrid attention (3 of 4
  layers linear / gated delta rule). It **thinks by default**; the only code change is `enable_thinking=False` in
  `apply_chat_template` (checked: for Qwen3-VL-2B the prompt is identical with and without the flag, so E16/E25 stay
  reproducible; for the 27B it removes the system line "Reasoning effort is set to xhigh" and inserts an empty
  `<think></think>`; image tokens identical).
- **Settings:** as E16/E25, only the model changes: 4-bit nf4 (bitsandbytes), greedy, 256 new tokens, same 20 views
  in the same order, same questions (`QUESTION`, `NEW_ONLY`). Qwen's model card recommends sampling with presence
  penalty 1.5 for non-thinking mode; deliberately not used (one change per run). SAM 3 image mode t0.3 with the two new
  lists on the cluster; top100 / 2B lists reuse job 658's caches (same cluster). Association locally, working setting
  (xc0.7, vf0.25, kl0.5, Déjà View K=16).
- **Code / command:** `uv run vlm-vocabulary --checkpoint Qwen/Qwen3.8-27B [--progressive] <views...>` ->
  `vlm[prog]-qwen3827b_<views>.txt` + `.json`. Smoke test `experiments/scripts/e26_smoke.sbatch` (job 663: load 19 s,
  19.1 GB weights, peak 19.5 GB, 19-23 tokens/s without the optional flash-linear-attention / causal-conv1d kernels).
  Full job `experiments/scripts/e26_vlm_big.sbatch` (job 664, 11 min: VLM ~2 min per scheme for both scenes incl.
  load, SAM 3 70-140 s per run, peak 19.8 GB), raw log `results/e26_cluster_job664.txt`; SAM 3 caches in
  `<scene>/e26_cluster/`. Scoring `uv run python experiments/scripts/e26_vlm_big.py > experiments/results/e26_vlm_big.txt`
  (CPU ~4 min, rerun identical; the E25 rows reproduce E25's association caches exactly, asserted).
- **Results** (`results/e26_vlm_big.txt`, all classes, all cluster runs):

  | scene | vocabulary | names | recall | pixel recall | precision | consistent | GT in >=2 views | merges | wrong links | mixed-class objects |
  |---|---|---|---|---|---|---|---|---|---|---|
  | kitchen | top100 | 100 | 0.92 | 0.95 | 0.51 | 0.82 | 38 | 1 | 5 | 20 |
  | kitchen | 2B union (E16) | 46 | 0.89 | 0.94 | 0.45 | 0.83 | 36 | 1 | 5 | 18 |
  | kitchen | 2B progressive (E25) | 12 | 0.82 | 0.92 | 0.69 | 0.85 | 33 | 1 | 4 | 13 |
  | kitchen | 27B union | 389 | 0.82 | 0.85 | 0.15 | 0.80 | 35 | 3 | 8 | 40 |
  | kitchen | 27B progressive | 249 | 0.91 | 0.95 | 0.24 | 0.82 | 38 | 1 | 8 | 45 |
  | office | top100 | 100 | 0.73 | 0.83 | 0.48 | 0.76 | 51 | 5 | 10 | 22 |
  | office | 2B union (E16) | 60 | 0.72 | 0.84 | 0.40 | 0.73 | 48 | 5 | 11 | 21 |
  | office | 2B progressive (E25) | 22 | 0.64 | 0.82 | 0.60 | 0.78 | 41 | 4 | 10 | 12 |
  | office | 27B union | 506 | 0.74 | 0.88 | 0.24 | 0.73 | 49 | 6 | 11 | 51 |
  | office | 27B progressive | 371 | 0.77 | 0.90 | 0.26 | 0.71 | 51 | 5 | 14 | 48 |

  1. **Greedy decoding makes the 27B model loop into free word association.** Answers start with a sensible list,
     then drift: "...shape, size, dimension, scale, ... velocity, acceleration, force, energy...", "...HDMI, USB,
     Ethernet, ... HTTP, HTTPS, FTP, SMTP, SSH...", "...primary, secondary, tertiary, quaternary, quinary...",
     "...curtain fondness, curtain affection, curtain love, curtain passion...". Truncated at 256 tokens: 5 / 8 of 20
     (union), 5 / 4 (progressive); some loops end before the limit, so dropping truncated answers does not clean the
     lists (union from non-truncated answers only: 50 / 100 names, still with "primary ... pentadecimal"). Lists:
     389 / 506 names (union), 249 / 371 (progressive) vs 46 / 60 and 12 / 22 for the 2B model.
  2. **Outside the loops, it follows "only new names" (unlike the 2B model).** Progressive, new names per view:
     kitchen 15, 0, 0, 0, 0, 95, 7, 0, 0, 58, 0, ... (16 of 20 views add <= 1 name; the big numbers are the loops),
     office 14, 117, 0, 91, 1, 0, 0, 81, 66, 0, ... (13 of 20). Smoke test, one kitchen view, list "wall, floor,
     cabinet, table" -> "microwave, box, paper, doorknob, counter, mat" (only new names); several views answer "NONE".
     Clean first answers: kitchen "wall, floor, ceiling, microwave, table, cardboard box, door, cabinet, paper, cable,
     mat, drawer, handle, card, box"; office "door, wall, floor, whiteboard, chair, fan, bucket, box, table, cabinet,
     light switch, handle, paper, cardboard". The synonym question cannot be judged on these lists (the loops add
     near-duplicates such as socket / wall socket / electrical outlet, curtain / curtain panel / curtain drape).
  3. **Junk names cost precision, little else.** Precision 0.15 / 0.24 (union) and 0.24 / 0.26 (progressive) vs
     top100 0.51 / 0.48: SAM 3 finds masks for abstract prompts ("room", "surface", "area", "corner"), 3469 / 2895
     per-view masks (27B union) vs 920 / 1092 (top100). Pixel recall stays high or rises (office 0.88 / 0.90 vs 0.83
     top100: more prompts cover more GT pixels); kitchen union drops to 0.85 (not analysed further; likely junk masks
     winning the highest-score overlap rule, e.g. a "room" mask over two walls). Consistency within 0.05 of top100,
     merges ~same (kitchen union 3: "room" -> 2 walls, "wall" -> 2 walls), wrong links up (8 vs 5, 11-14 vs 10),
     mixed-class objects about doubled (40-51 vs 20-22).
  4. Runtime: VLM 47-79 s per scene on the 5090 (each loop ~8 s); SAM 3 time grows with the list (70-140 s per run).
- **Figure:** `figures/e26_vlm_big.jpg` (names, pixel recall, precision, consistent, merges for the five vocabularies).
- **Conclusion:** E26 as run tests greedy decoding more than the model. The 27B model does follow the progressive
  instruction where the 2B model did not (finding 2), so the progressive scheme becomes testable, but the lists are
  unusable because of degenerate loops under greedy decoding. The next variable is decoding (one change), e.g.
  `repetition_penalty` / `no_repeat_ngram_size` (deterministic) or Qwen's recommended non-thinking sampling
  (temperature 0.7, top-p 0.8, top-k 20, presence penalty 1.5; needs a fixed seed). Not decided yet (E27, with the user).

## Plan: E27 = decoding fix for the 27B (02.11.2026)
Decided with the user: deterministic repetition control (a), not Qwen's recommended sampling (b).
- **One change vs E26:** `repetition_penalty=1.1` in `model.generate` (Hugging Face `RepetitionPenaltyLogitsProcessor`:
  logits of every token already in prompt or answer divided by 1.1). Still greedy, 256 new tokens, Qwen3.8-27B 4-bit,
  thinking off, same questions, views, order, both schemes. Value fixed now (mild), not tuned on scores.
- **Expected side effects:** the penalty also hits the newline token every line uses and, in the progressive prompt,
  the listed names (that is the intent there). Too strong -> lists end early or the one-name-per-line format breaks.
- **Caveat:** most E26 loops drift into *new* words ("velocity, acceleration, force"), not repeated n-grams, so
  `no_repeat_ngram_size` would not help (not used); the penalty helps where tokens recur ("curtain X, curtain Y",
  "top surface, bottom surface"). It may not stop the drift; then that is the result, and the next options are capping
  names per answer / fewer new tokens (deterministic) or (b).
- **Smoke test first** (sanity check of the setting, not tuning): two answers that looped in E26 (office progressive
  view 2 DSC03492 with view 1's 14 names; kitchen union view 3 DSC08856), at 1.0 and 1.1. Check: loop gone, line
  format intact, list not cut to a few names. If 1.1 clearly breaks the format, ask before changing the value.
- **Full run:** cluster, VLM both schemes + SAM 3 t0.3 with the two new lists; association locally with the working
  setting; report as E26 (names, new names per view, truncated, loops by eye, synonyms by hand, recall / pixel recall /
  precision / consistent / merges / wrong links), with the E26 rows kept.

## E27: Qwen3.8-27B with repetition penalty 1.1 (02.11.2026)
- **Goal:** plan above: does a mild repetition penalty (still greedy) remove E26's loops, so that the 27B lists become
  usable and the progressive scheme can be judged?
- **Settings:** as E26, one change: `repetition_penalty=1.1` in `generate` (both schemes, same 20 views, order,
  questions, 256 new tokens, 4-bit, thinking off). SAM 3 image mode t0.3 with the two new lists on the cluster;
  association locally with the working setting (xc0.7, vf0.25, kl0.5, Déjà View K=16). Value fixed before running.
- **Code / command:** `VlmVocabulary(repetition_penalty=...)`, CLI `vlm-vocabulary --repetition-penalty 1.1` -> tag
  `vlm[prog]-qwen3827b-rp1.1_<views>` (default 1.0 keeps all earlier outputs and tags). Smoke test
  `experiments/scripts/e27_smoke.sbatch` (job 666), full job `experiments/scripts/e27_vlm_rp.sbatch` (job 667 on VC18,
  ~10 min; VLM 2 min per scheme for both scenes, SAM 3 66-130 s per run, peak 19.7 GB), raw log
  `results/e27_cluster_job667.txt`, SAM 3 caches in `<scene>/e27_cluster/`. Scoring
  `uv run python experiments/scripts/e27_vlm_rp.py > experiments/results/e27_vlm_rp.txt` (CPU ~4 min; rerun identical;
  the E25/E26 rows reproduce their association caches exactly, asserted).
- **Smoke test** (job 666): rp 1.0 reproduces E26's answers word for word. Kitchen union view 3 (DSC08856): 92 names
  ending in "...matte surface, textured surface..." -> 13 clean names at 1.1 ("wall, ceiling, floor, desk, cabinet,
  paper, notebook, box, light switch, door handle, cardboard box, sticky note, calendar"). Office progressive view 2
  (DSC03492): still loops at 1.1, into other words. Line format intact, lists not cut short -> setting kept.
- **Results** (`results/e27_vlm_rp.txt`, all classes, cluster runs; top100 and E26 rows for reference):

  | scene | vocabulary | names | recall | pixel recall | precision | cross-class leakage | consistent | merges | wrong links | mixed-class objects |
  |---|---|---|---|---|---|---|---|---|---|---|
  | kitchen | top100 | 100 | 0.92 | 0.95 | 0.51 | 0.007 | 0.82 | 1 | 5 | 20 |
  | kitchen | 27B union (E26) | 389 | 0.82 | 0.85 | 0.15 | 0.012 | 0.80 | 3 | 8 | 40 |
  | kitchen | 27B progressive (E26) | 249 | 0.91 | 0.95 | 0.24 | 0.008 | 0.82 | 1 | 8 | 45 |
  | kitchen | 27B union rp1.1 | 265 | 0.80 | 0.89 | 0.23 | 0.033 | 0.78 | 1 | 5 | 35 |
  | kitchen | 27B progressive rp1.1 | 323 | 0.90 | 0.95 | 0.28 | 0.008 | 0.81 | 1 | 6 | 31 |
  | office | top100 | 100 | 0.73 | 0.83 | 0.48 | 0.049 | 0.76 | 5 | 10 | 22 |
  | office | 27B union (E26) | 506 | 0.74 | 0.88 | 0.24 | 0.063 | 0.73 | 6 | 11 | 51 |
  | office | 27B progressive (E26) | 371 | 0.77 | 0.90 | 0.26 | 0.047 | 0.71 | 5 | 14 | 48 |
  | office | 27B union rp1.1 | 476 | 0.73 | 0.88 | 0.23 | 0.066 | 0.74 | 6 | 11 | 48 |
  | office | 27B progressive rp1.1 | 236 | 0.73 | 0.88 | 0.28 | 0.060 | 0.70 | 6 | 11 | 39 |

  1. **The loops stay.** Truncated answers 3 / 7 (union) and 5 / 4 (progressive) of 20 vs 5 / 8 and 5 / 4 in E26;
     each loop still adds ~50-110 names. The penalty changes *where* the drift goes, not whether it happens: "deny,
     allow, block, enable, disable, activate...", "who, whom, whose, this, that...", "fork, knife, spoon, chopsticks,
     ladle...", "velocity, acceleration, momentum, inertia, friction...", "stone, rock, pebble, gravel, sand, dust...".
     It also moves loops between views: kitchen view 3 is fixed (smoke test), but kitchen view 1, clean in E26, now
     loops (its 109 names open both rp1.1 lists). As stated in the plan's caveat: the drift is into new words, so a
     penalty on repeated tokens cannot stop it.
  2. **Outside the loops nothing changes much:** clean answers are 5-20 names, and the progressive scheme still follows
     "only new names" (15 / 15 of 20 views add <= 1 name, "NONE" answers).
  3. **Scores:** precision a bit up (0.15-0.26 -> 0.23-0.28, still about half of top100's 0.51 / 0.48), only because
     the lists are somewhat shorter; pixel recall 0.88-0.95, consistency and merges as E26 (office 6 merges, among them
     "room" -> two walls, "desk" -> three tables, "fan"/"wood" -> pedestal fan + object). Kitchen union rp1.1 has
     cross-class leakage 0.033 (all others 0.006-0.012): a "workstation" mask covering cabinet, wall, whiteboard and
     calendar (>= 0.021 of the 0.034, own check); one junk name can create the harmful error type.
- **Figure:** `figures/e27_vlm_rp.jpg` (names, pixel recall, precision, consistent, merges for the seven vocabularies).
- **Conclusion:** repetition penalty 1.1 does not make the 27B lists usable; it shifts the loops. Not tried: a higher
  penalty (would be tuning on the test scenes, and the plan's risk of broken line format grows). The loops are an
  open-ended list question meeting greedy decoding; options that limit the answer itself are more promising: a cap on
  names per answer / much lower `max_new_tokens` (e.g. 64, deterministic, also cuts the loops' cost), or Qwen's
  recommended non-thinking sampling (temperature 0.7, top-p 0.8, top-k 20, presence penalty 1.5, fixed seed). To decide
  with the user. top100 stays the default vocabulary.

## Plan: E28 = answer cap for the 27B (02.11.2026)
Decided with the user after E27: cap the answer length instead of a stronger penalty or sampling.
- **One change vs E26:** `max_new_tokens` 256 -> 64. Greedy, no repetition penalty (1.0, as E26), Qwen3.8-27B 4-bit,
  thinking off, same questions, views, order, both schemes. Value fixed before running.
- **Why 64:** clean (non-truncated) E26 answers are 10-42 tokens (measured with the Qwen3-VL-2B tokenizer as a proxy
  for the 27B's; the only longer non-truncated answers, 181 and 245, are loops that ended by themselves). So 64 should
  cut only loops, keeping their first ~20-25 names; the last, possibly cut-off line of a truncated answer is dropped
  (unchanged parsing).
- **Expectations / checks:** union = greedy with the same prompts, so each answer should be the first 64 tokens of its
  E26 answer (check; bf16 may break exact equality). Progressive differs (the list fed back is shorter). Loops are
  not removed, only shortened: their first ~20 names include drift words ("edge", "corner", "surface"), so junk
  prompts remain; report how many. VLM time should drop (each loop ~8 s -> ~2 s).
- **Run / report:** as E27 (cluster job, SAM 3 t0.3 with the two new lists, association locally with the working
  setting; rows top100, E26, E27, E28).

## E28: Qwen3.8-27B with answers capped at 64 tokens (02.11.2026)
- **Goal:** plan above: does capping the answer length (deterministic, greedy) make the 27B lists usable?
- **Settings:** as E26, one change: `max_new_tokens` 256 -> 64 (no repetition penalty). SAM 3 image mode t0.3 with the
  two new lists on the cluster; association locally with the working setting (xc0.7, vf0.25, kl0.5, Déjà View K=16).
- **Code / command:** `vlm-vocabulary --max-new-tokens 64` -> tag `vlm[prog]-qwen3827b-mt64_<views>` (default 256 keeps
  all earlier outputs). Job `experiments/scripts/e28_vlm_cap.sbatch` (job 676 on VC16, ~19 min, of which ~16 min loading
  the 27B weights from VC18's scratch; answering itself 11-30 s per scene; SAM 3 24-55 s per run; peak 19.5 GB), raw log
  `results/e28_cluster_job676.txt`, SAM 3 caches in `<scene>/e28_cluster/`. Scoring
  `uv run python experiments/scripts/e28_vlm_cap.py > experiments/results/e28_vlm_cap.txt` (CPU ~5 min; rerun
  identical; the E25-E27 rows reproduce their association caches exactly, asserted).
- **Check (as expected in the plan):** union = same prompts, greedy, so every capped answer is the start of its E26
  answer: 20 / 20 in both scenes (exact, also across bf16 runs). Truncated (cap hit): 5 / 9 (union), 3 / 3 (progressive).
- **Results** (`results/e28_vlm_cap.txt`, all classes, cluster runs; selected rows):

  | scene | vocabulary | names | recall | pixel recall | precision | cross-class leakage | consistent | merges | wrong links | mixed-class objects |
  |---|---|---|---|---|---|---|---|---|---|---|
  | kitchen | top100 | 100 | 0.92 | 0.95 | 0.51 | 0.007 | 0.82 | 1 | 5 | 20 |
  | kitchen | 2B union (E16) | 46 | 0.89 | 0.94 | 0.45 | 0.007 | 0.83 | 1 | 5 | 18 |
  | kitchen | 27B union (E26) | 389 | 0.82 | 0.85 | 0.15 | 0.012 | 0.80 | 3 | 8 | 40 |
  | kitchen | 27B union rp1.1 (E27) | 265 | 0.80 | 0.89 | 0.23 | 0.033 | 0.78 | 1 | 5 | 35 |
  | kitchen | 27B union cap64 | 90 | 0.91 | 0.94 | 0.29 | 0.008 | 0.84 | 1 | 5 | 37 |
  | kitchen | 27B progressive cap64 | 96 | 0.89 | 0.94 | 0.36 | 0.007 | 0.83 | 1 | 6 | 31 |
  | office | top100 | 100 | 0.73 | 0.83 | 0.48 | 0.049 | 0.76 | 5 | 10 | 22 |
  | office | 2B union (E16) | 60 | 0.72 | 0.84 | 0.40 | 0.051 | 0.73 | 5 | 11 | 21 |
  | office | 27B union (E26) | 506 | 0.74 | 0.88 | 0.24 | 0.063 | 0.73 | 6 | 11 | 51 |
  | office | 27B union rp1.1 (E27) | 476 | 0.73 | 0.88 | 0.23 | 0.066 | 0.74 | 6 | 11 | 48 |
  | office | 27B union cap64 | 79 | 0.75 | 0.89 | 0.33 | 0.048 | 0.76 | 5 | 11 | 37 |
  | office | 27B progressive cap64 | 91 | 0.75 | 0.89 | 0.30 | 0.048 | 0.73 | 5 | 11 | 34 |

  1. **The cap bounds the damage of each loop:** a loop now adds at most ~25 names (the start of the drift) instead
     of ~100, so the lists shrink to 79-96 names, the size of top100. Precision, merges and leakage improve over
     E26/E27; recall and pixel recall stay within 0.02 of E26's best rows.
  2. **Union cap64 is on par with top100 on everything except precision:** pixel recall 0.94 / 0.89 (office +0.06 over
     top100; likely from names top100 lacks, e.g. fan and coat rack as in E19; not checked per object), recall 0.91 / 0.75,
     consistency 0.84 / 0.76, merges 1 / 5 and cross-class leakage 0.008 / 0.048 equal to top100. Against the 2B union
     (E16): recall and pixel recall up (office pixel recall 0.84 -> 0.89), precision down (0.45 / 0.40 -> 0.29 / 0.33).
  3. **Precision stays low (0.29-0.36 vs 0.51 / 0.48)** because the loops' first names are abstract or part words:
     "edge, corner, surface, top, side, front, back, object", "frame, panel, base", progressive also "inner, outer, upper,
     lower, frontal, rear, lateral" and "marker cap, marker body, marker tip, marker ink...". SAM 3 finds masks for such
     prompts (1380-2058 per-view masks vs 920 / 1092 for top100). By the asymmetry noted after E17, low precision (extra
     part masks) mostly removes constraints for the refiner; the harmful errors (merges, cross-class leakage) are at
     top100's level here.
  4. **Union vs progressive: no clear winner.** Progressive follows "only new names" (16 / 16 of 20 views add nothing)
     but each of its 3 loops dumps ~25 junk names at once; precision 0.36 vs 0.29 kitchen, 0.30 vs 0.33 office;
     consistency slightly lower in the office (0.73 vs 0.76). Union is simpler and needs no list in the prompt.
- **Figure:** `figures/e28_vlm_cap.jpg` (names, pixel recall, precision, consistent, merges for nine vocabularies).
- **Conclusion:** capping the answer is the decoding fix that works: Qwen3.8-27B union cap64 matches top100 on pixel
  recall, consistency, merges and leakage (office pixel recall better) without being a ScanNet++ list, at the cost of
  precision from abstract part names. Candidate VLM vocabulary for the Hypersim test (where top100 has no advantage).
  Not done (would be tuning on the test scenes): picking the cap from these scores, a stop-word list for abstract names.

## E29: Hypersim loader (views + ground truth) on three validation scenes (02.11.2026)
- **Goal:** move the segmentation study to Hypersim (team decision 02.11; issue #8: Hypersim GT semantics). Check that
  the new loader gives correct cameras, depth and instance GT before any method runs on it.
- **Scenes** (official v1 split `val`, chosen from preview frames, before any method ran): `ai_015_001` cam_00 kitchen,
  `ai_047_009` cam_00 conference room (18 identical office chairs), `ai_052_001` cam_00 restaurant (crowded, repeated
  chairs and tables). 20 frames uniform over the trajectory's public frames, 1024x768 -> 504x378.
- **Code / command:** `src/loaders/hypersim/` (config, scene, download, prepare), `configs/hypersim.yaml`,
  `uv run hypersim-download` (HTTP range reads of Apple's per-scene ZIPs, 504 files / ~45 MB per trajectory),
  `uv run hypersim-prepare`. Checks: `uv run python experiments/scripts/e29_hypersim_loader.py >
  experiments/results/e29_hypersim_loader.txt` (CPU). Tests `tests/test_hypersim.py` (9).
- **Findings while building it:**
  1. **Most Hypersim cameras are tilt-shift**, not only the 22 scenes with off-diagonal terms in `M_cam_from_uv`'s first
     two rows: 254 of 482 scenes have a term in the third row (ray z depends on v), 31 more have skew; 197 are plain.
     All three chosen scenes are tilt-shift without skew. Exact handling: pixel -> ray is linear (A = F M S); QR gives
     A = Q U, U = K^-1 (zero skew, off-centre principal point, e.g. cy 216 instead of 384 in the conference room), Q is
     folded into cam_to_world. Skewed cameras are rejected.
  2. **Walls / floors / ceilings have no instance ids** (100% of their pixels -1); every other class has instances.
     Decision (user): one instance per class (`STUFF_BASE + NYU40 id`, shared by all views). Unlike ScanNet++, walls
     are not split per plane -> a SAM 3 mask of one wall plane will often miss IoU 0.5 against the whole-room wall.
  3. **Labels have gaps:** conference room 23% of pixels have geometry but no class and no instance (foyer and stairs
     behind the glass walls) -> unlabeled, ignored by the scorer. Kitchen / restaurant: 0.1% / 0.4%.
  4. NYU40 classes are coarse: `otherprop` holds most small objects (50 of 77 kitchen instances, 152 of 244 restaurant).
     3 tiny kitchen instances (0.01% of pixels) have no class -> `unknown`.
  5. `depth_meters` is distance to the camera centre (converted to z); all files float16.
- **Results** (`results/e29_hypersim_loader.txt`):
  - Position-map check (kitchen frame 40, full resolution, tilt-shift camera): our K, z-depth and pose lift every pixel
    to Hypersim's own world position within median 1.2 mm, p99 4.1 mm (relative 4e-4 = float16 rounding).
  - Labeled / stuff / unlabeled share: kitchen 0.999 / 0.554 / 0.001, conference 0.765 / 0.361 / 0.235, restaurant
    0.996 / 0.395 / 0.004; 77 / 34 / 244 instances in the 20 views (32 / 22 / 107 large enough to score).
  - Scorer GT vs GT: all 1.000, leakage 0 (all three).
  - GT masks relabeled per view + GT geometry, re-associated (cross-view poses check):

  | scene | association | consistent (all) | id switches | merges |
  |---|---|---|---|---|
  | kitchen | default | 1.000 | 0 | 0 |
  | kitchen | working (xc0.7 vf0.25 kl0.5) | 0.938 | 2 | 0 |
  | conference | default | 1.000 | 0 | 0 |
  | conference | working | 0.909 | 3 | 0 |
  | restaurant | default | 0.981 | 3 | 0 |
  | restaurant | working | 0.944 | 7 | 0 |

    (ScanNet++ with the default setting: kitchen 1.00, office 0.93.) The visibility rule (vf0.25) costs 2-4 objects
    here with perfect masks: hidden parts of a GT mask (chairs behind the table) make a direction not count.
  - Cross-view z agreement (< 1%, pairs with >= 10% of points inside the other view): median 0.67 / 0.81 / 0.67, min
    0.014 / 0.28 / 0.07. Not a clean pose test (occlusion and nearest-pixel sampling at grazing angles count as
    disagreement); the association rows above are the stronger evidence.
- **Figures:** `figures/e29_<scene>_gt.jpg` (GT contact sheets: chairs keep their ids over all views, glass gaps visible).
- **Conclusion:** loader correct to file precision, GT usable with `eval-instances` and the association unchanged.
  Caveats for the coming SAM 3 runs: whole-room wall / floor instances, coarse NYU40 classes (cross-class leakage
  between two `otherprop` objects counts as same class), principal points far from the centre (check Déjà View copes).
  Not checked: whether Déjà View or SAM 3 trained on Hypersim (MoGe-2 did).

## Plan: E30 = first Hypersim run of the segmenter (02.11.2026)
- **Goal:** does the segmenter (SAM 3 image mode + 3D association, working setting) carry over from ScanNet++ to
  Hypersim, and is the VLM vocabulary better than top100 outside ScanNet++ (top100 is a ScanNet++ list)?
- **Settings, fixed before running (none tuned on Hypersim):** E29's three scenes, 20 views at 504x378; Déjà View K=16;
  SAM 3 image mode t0.3; vocabularies top100 (plain list) and Qwen3.8-27B union cap 64 tokens (E28 setting, one list
  per scene from the same 20 views); association `--cross-class 0.7 --min-visible-fraction 0.25 --keep-linked-below 0.5`
  with Déjà View geometry, and with GT geometry as the upper bound. Scorer `eval-instances` unchanged.
- **Checks first:** Déjà View copes with the off-centre principal points (fitted K vs GT, per-view scale median depth
  error vs GT, as E8).
- **Training-data overlap:** Déjà View: 0.69% of its training set is Hypersim (user, from the paper); scene-level
  split not checked -> the three scenes are from the official v1 val split, but treat DV depth here as possibly
  optimistic. MoGe-2 also trained on Hypersim (not used here). SAM 3: not checked.
- **Expected caveats (E29):** walls / floors / ceilings are one instance per class -> SAM 3 wall-plane masks miss
  IoU 0.5; NYU40 `otherprop` makes cross-class leakage look smaller; conference room 23% unlabeled.
- **Run:** cluster job `experiments/scripts/e30_hypersim.sbatch`; association + scoring locally.
- **Change after job 760 (02.11), before any segmentation scoring:** Déjà View's fitted cameras always have a centred
  principal point; a free pinhole fit to its raw rays agrees (cx / cy within 2-3 px of the centre, residual 0.06 deg),
  so the network cannot represent Hypersim's tilt-shift cameras. Conference room (cy 106 vs 189): point-map Rel. L2
  0.116, 15% inliers. Only 14 of 46 val scenes have plain cameras. Decision (user): finish E30 on the three tilt-shift
  scenes and add three plain-camera val scenes chosen from previews before scoring: `ai_010_007` living room,
  `ai_006_007` bedroom, `ai_044_001` hotel lobby (`ai_003_010` was picked first but has 78% unlabeled pixels:
  walls / floor / ceiling without class). Same settings; job `experiments/scripts/e30_plain.sbatch`.

## E30: first Hypersim run of the segmenter, six val scenes (03.11.2026)
- **Goal / settings:** as "Plan: E30" above, fixed before running, nothing tuned on Hypersim. Six scenes from the v1
  `val` split, 20 views at 504x378: three tilt-shift (ts) scenes from E29 (kitchen `ai_015_001`, conference room
  `ai_047_009`, restaurant `ai_052_001`) and three plain-camera scenes (living room `ai_010_007`, bedroom `ai_006_007`,
  hotel lobby `ai_044_001`), all cam_00. Déjà View K=16; SAM 3 image mode t0.3 with top100 and with the Qwen3.8-27B
  union cap64 list (E28 setting, one list per scene); association xc0.7 vf0.25 kl0.5 with Déjà View and with GT
  geometry. Scorer `eval-instances` unchanged.
- **Runs:** cluster jobs 760 (ts scenes) and 776 (`experiments/scripts/e30_plain.sbatch`: plain scenes + the restaurant
  VLM SAM 3 rerun after the `ID_STRIDE` fix 60b3d68); peak GPU 19.4 GB (27B VLM). Caches in
  `data/processed/hypersim/<scene>/` (Déjà View, VLM lists) and `<scene>/e30_cluster/` (SAM 3). Analysis on CPU:
  `PYTHONIOENCODING=utf-8 uv run python experiments/scripts/e30_hypersim.py > experiments/results/e30_hypersim.txt`
  (~15 min, mostly association of the restaurant's 11k VLM masks); rerun gives an identical file.
- **Déjà View vs GT** (fitted K vs GT K, depth abs rel median with per-view scale, paper's Rel. L2 / inliers < 3% with
  one Sim(3) per scene):

  | scene | cy DV - GT (px) | depth abs rel | Rel. L2 | inliers < 3% |
  |---|---|---|---|---|
  | kitchen (ts) | +16.3 | 0.0098 | 0.031 | 80.7% |
  | conference (ts) | +82.3 | 0.0200 | 0.116 | 15.5% |
  | restaurant (ts) | +13.9 | 0.0139 | 0.048 | 74.5% |
  | living room | -0.3 | 0.0045 | 0.010 | 97.0% |
  | bedroom | -0.3 | 0.0071 | 0.022 | 91.7% |
  | lobby | -0.1 | 0.0087 | 0.026 | 78.5% |

  Déjà View always predicts a centred principal point (cy 188.6-188.9 of 189); its error grows with how far the true
  one is off-centre. On plain cameras it is as good as on ScanNet++ (E8: 0.7-0.8%, E23: Rel. L2 0.024-0.026) or better.
  0.69% of Déjà View's training data is Hypersim (scene split not checked) -> plain-scene numbers may be optimistic.
- **Segmentation, all classes** (Déjà View geometry; GT geometry in brackets where it differs):

  | scene | vocabulary | names | recall | pixel recall | precision | cross-class leakage | consistent | merges |
  |---|---|---|---|---|---|---|---|---|
  | kitchen (ts) | top100 | 100 | 0.76 | 0.69 | 0.33 | 0.009 | 0.68 (0.80) | 0 (1) |
  | kitchen (ts) | 27B | 60 | 0.80 | 0.67 | 0.17 | 0.010 | 0.71 (0.89) | 0 |
  | conference (ts) | top100 | 100 | 0.84 | 0.81 | 0.45 | 0.028 | 0.55 (0.86) | 1 (0) |
  | conference (ts) | 27B | 71 | 0.77 | 0.76 | 0.15 | 0.027 | 0.36 (0.68) | 1 (0) |
  | restaurant (ts) | top100 | 100 | 0.76 | 0.77 | 0.38 | 0.044 | 0.67 (0.85) | 4 (2) |
  | restaurant (ts) | 27B | 88 | 0.79 | 0.78 | 0.10 | 0.051 | 0.53 (0.69) | 2 |
  | living room | top100 | 100 | 0.91 | 0.88 | 0.47 | 0.006 | 0.82 | 0 |
  | living room | 27B | 33 | 0.94 | 0.88 | 0.43 | 0.007 | 0.83 | 0 |
  | bedroom | top100 | 100 | 0.85 | 0.85 | 0.49 | 0.021 | 0.70 (0.76) | 0 |
  | bedroom | 27B | 89 | 0.90 | 0.88 | 0.17 | 0.022 | 0.71 (0.79) | 0 |
  | lobby | top100 | 100 | 0.80 | 0.65 | 0.23 | 0.007 | 0.81 | 1 |
  | lobby | 27B | 77 | 0.58 | 0.58 | 0.16 | 0.188 | 0.78 | 1 (2) |

- **Findings:**
  1. **Association with Déjà View geometry matches GT geometry on plain cameras, not on tilt-shift ones.** Consistent
     with Déjà View vs GT geometry: plain 0.82 / 0.82, 0.70 / 0.76, 0.81 / 0.81 (both vocabularies alike); tilt-shift
     0.68 / 0.80, 0.55 / 0.86, 0.67 / 0.85. The gap follows the camera error above (largest in the conference room).
     On plain cameras the ScanNet++ result (E9: Déjà View ~ GT for association) carries over to Hypersim.
  2. **The 27B VLM vocabulary is not better than top100 on Hypersim**, although top100 is a ScanNet++ list. Instance
     recall is higher in 4 of 6 scenes (+0.03 to +0.05) and pixel recall within ±0.03 there, but precision is lower in
     every scene (0.10-0.43 vs 0.23-0.49) and consistency drops where the list has part names: conference room 0.55 ->
     0.36 ("backrest, armrest, seat, leg" split the 18 chairs into parts). The hypothesis "VLM generalizes better
     outside ScanNet++" is not supported on these six scenes.
  3. **One abstract name can break a scene:** the lobby list contains "lobby"; SAM 3 masks the whole room with it,
     and the association joins the wall and floor instances -> cross-class leakage 0.188 (top100 0.007), recall
     0.80 -> 0.58. Same failure as E28's abstract names ("surface", "interior", "exterior view", "reflection",
     "shadow" are in other lists here), now with a large cost.
  4. Merges stay rare (0-4 per scene); most restaurant merges are adjacent `otherprop` objects (cups / glasses with
     their saucers) under a "cup" / "glass" mask, which the coarse NYU40 GT may count as two objects.
  5. 27B answers hitting the 64-token cap: 9 / 17 / 15 / 1 / 12 / 20 of 20 (cap working as intended, the last
     partial line is dropped). Hypersim wall / floor / ceiling = one instance per class, so pixel recall "all" is
     dominated by whether SAM 3 covers a whole room's wall with masks of IoU >= 0.5 (lobby, kitchen 0.65-0.69).
- **Figure:** `figures/e30_hypersim.jpg` (pixel recall, precision, consistency, merges per scene, both vocabularies,
  Déjà View vs GT geometry).
- **Conclusion:** for the team protocol (issue #3): with Déjà View, use plain-camera Hypersim scenes (14 of 46 val
  scenes); tilt-shift scenes measure Déjà View's camera model, not the method. Segmenter default stays top100; the
  VLM vocabulary needs a filter for abstract / part names before it can replace it (not tuned here: these are now
  seen scenes).

## E31: which bedroom views show Déjà View's edge errors best (03.11.2026)
- **Goal:** choose the views of the shared visual scene (Hypersim `ai_006_007` bedroom, plain camera, E30) that show
  bleeding most clearly. A selection aid, not a method result.
- **Definitions as E18** (copied into the script, same thresholds): GT depth edge = relative GT depth jump > 5% to a
  4-neighbour; error = relative depth error after a per-view median scale; near-edge = within 3 px of a GT depth edge,
  far = more than 10 px; a near-edge pixel bleeds if its error is > 5%. Ranked by bleeding pixels as a share of the
  image (what a viewer sees). Déjà View K=16 cache from E30, 20 views at 504x378.
- **Command:** `PYTHONIOENCODING=utf-8 uv run python experiments/scripts/e31_bedroom_edges.py >
  experiments/results/e31_bedroom_edges.txt` (CPU, seconds).
- **Results:** bleeding pixels range from 6.4% of the image (view 18, frame 94) to 0.7% (view 8, frame 41, a near-empty
  wall and door view). Top four: views 18, 19, 0, 12 (6.4 / 5.8 / 5.3 / 4.7%); median error near edges 1.8-3.0% vs
  0.3-0.9% far from edges, the same contrast as on ScanNet++ (E18) and ETH3D (Peter). A quarter to a third of all
  near-edge pixels bleed in every view (0.19-0.34).
- **Where:** almost all bleeding sits on thin structures in front of distant surfaces: the ornate metal bed frame
  (views 18, 19, 12) and the vanity's mirror frame, chair and stool legs (view 0). Large planes (walls, door,
  wardrobe front) stay below 1%. View 0's far error (0.9%) is raised by the mirror: GT depth is the mirror surface,
  the error map suggests Déjà View partly predicts the reflected room (not measured).
- **Figure:** `figures/e31_bedroom_edges.jpg` (the four chosen views 18, 0, 12, 6 with their rank: image, error map on a shared 0-10% scale, GT depth edges
  with bleeding pixels in red).
- **Conclusion:** visual views for the bedroom = frames 94, 0, 62 (ranks 1, 3, 4; rank 2 = frame 99 is nearly the same
  viewpoint as 94) plus frame 31 (rank 9) as the room overview. The bed frame is the clearest test of whether a refiner
  sharpens thin structures without smoothing across them.

## E32: how the four shared bedroom views segment (03.11.2026)
- **Goal:** a visual of the segmenter on the shared views (E31: views 18, 0, 12, 6 of Hypersim `ai_006_007`), in a form
  that shows identity without id numbers (old TODO: better segmentation visuals than id overlays).
- **Setting:** E30's cache, SAM 3 image mode top100 t0.3 + 3D linking xc0.7 vf0.25 kl0.5 with Déjà View geometry.
  Colours: each linked object takes the colour of the GT object it covers most over all 20 views (one linked object
  per GT object, largest overlap wins); further linked objects on the same GT object get other colours. Same colour
  in the GT and SAM 3 columns = same object; a colour change of one object between views = id switch. Error map per
  labeled GT pixel: right object / other object / no mask.
- **Command:** `PYTHONIOENCODING=utf-8 uv run python experiments/scripts/e32_bedroom_segmentation.py >
  experiments/results/e32_bedroom_segmentation.txt` (CPU, seconds).
- **Results** (share of labeled GT pixels): right object 0.83-0.92, other object 0.01-0.03, no mask 0.06-0.16. Over the
  scene 263 linked objects touch labeled pixels; 52 carry their GT object's colour, 211 are extra (part masks,
  single-view masks, splits; precision 0.49 in E30).
- **Where it goes wrong** (pixel counts per view from a breakdown of the same data):
  1. Other object, mostly at the bed frame: the "bed" mask fills the gaps between the ornate bars, so wall, floor and
     pillow pixels seen through the frame get the bed's id (view 18: 2.3k px). These are the pixels where Déjà View
     bleeds most (E31), so here the mask would tie the background to the frame.
  2. Other object by GT granularity: Hypersim labels the clothes inside the wardrobe separately; SAM 3's "cabinet" mask
     covers them (2.0k / 1.2k px in views 18 / 12). Not a real error for geometry.
  3. No mask: the mirror (view 0: 17k px, the largest single gap) and the floor mat (rug) in views 12 / 6 (12.5k /
     18.2k px) are never masked, although top100 contains mirror and rug.
  4. Walls: GT has one wall instance for the whole room; SAM 3 masks wall planes separately (the magenta wall is an
     extra object). Counted right (it lies on the wall), only a granularity difference.
  Cross-view identity of the shared objects holds in these views: door, wardrobe, vanity, pink chair and stool keep
  one colour.
- **Figure:** `figures/e32_bedroom_segmentation.jpg` (per view: image, GT instances, SAM 3 + linking in matched colours,
  error map).
- **Conclusion:** on the bedroom the harmful error is concentrated in one place: masks of thin objects with holes
  (the bed frame) cover what is behind them, exactly where depth bleeds. A refiner that smooths within segments would
  pull the wall seen through the frame towards the frame; such masks need trimming (e.g. by Déjà View depth jumps)
  before they can cut or tie geometry.

## Plan: E33 = trim masks at Déjà View depth jumps (03.11.2026)
- **Goal:** stop masks of thin objects with holes from covering what is seen through them (E32: the "bed" mask takes
  the wall, floor and pillows between the frame's bars, exactly where Déjà View bleeds, E31). Also answers the
  bed-frame headroom question: how much of the bed's wrong-object area a geometric trim removes.
- **Change (the only one):** per view and per raw SAM 3 mask, split the mask into 4-connected regions that never
  cross a Déjà View depth jump (relative jump > 5% between 4-neighbours, the E18 / E31 edge definition, not tuned).
  A region is removed from the mask if it lies on the far side of most (> 50%) of the jumps on its border with the
  rest of the mask: it is background seen through a hole or past an outline. Removed pixels lose this mask; overlap
  resolution and the association then run unchanged. Both thresholds fixed here, before running.
- **Setting (as Peter's protocol, PR #17):** Hypersim `dev` split (living room `ai_010_007`, bedroom `ai_006_007`,
  lobby `ai_044_001`), 20 views at 504x378, Déjà View K=16, SAM 3 image mode with `configs/prompts/nyu40.txt` at
  t0.3, association xc0.7 vf0.25 kl0.5 on Déjà View geometry. Baseline: the same masks untrimmed. nyu40 instead of
  E30-E32's top100 so that the result plugs into the team's protocol (user, 03.11); E32's bed numbers are therefore
  not directly comparable.
- **Scores:** (1) `eval-instances` against the Hypersim GT instances, unchanged; plus E32's per-pixel breakdown
  (right object / other object / no mask) for the bedroom's bed. (2) `src.analysis.scene_eval` with the trimmed
  masks as the segmentation: the Déjà View variant isolates the labels (geometry unchanged), the refined variant
  shows whether the refiner's instance gating gains from cleaner masks. Same machine for both runs (bf16 noise).
  Geometry is reported for the refined variant too (accuracy and completeness, not only labels). Hypothesis, not a
  claim: E_mv and E_n act within one instance, so a mask that spans a depth jump lets them pull the far surface
  (wall, floor behind the bars) towards the near one. That would match the pattern in Peter's dev table (issue #5):
  accuracy better on all three scenes, completeness worse by more, (acc + comp) / 2 = 1.60 -> 1.60, 1.85 -> 1.96,
  7.06 -> 8.30 cm. If trimming does not move completeness, the drop comes from elsewhere.
- **Protocol in flux:** Peter's `metrics` branch (issue #5, not pushed on 04.11) replaces F@5cm with accuracy /
  completeness (mean, median), PQ / PQ_th / PQ_st and AP / AP50 / AP25. E33 is scored first with `scene_eval` as on
  main, and rescored with the new metrics once they merge; the masks and runs stay the same.
- **Caveat:** the idea comes from looking at the bedroom (E32), which is a dev scene; the thresholds are not tuned
  on it. A gain counts once it also holds on `val`; `test` is not used.
- **Run:** cluster job 824 (`~/mircea/jobs/e33_inputs.sbatch`) builds the nyu40 masks and DSINE normals for the
  three scenes; trimming and `eval-instances` on CPU; `scene_eval` on the cluster.
- **Progress 04.11:** job 824 done (SAM 3 nyu40 ~11 s and 2.2 GB peak GPU per scene; DSINE ~3 s per scene). GPU
  nodes have no git, which `scene_eval` needs for the run's hash: the login node's `/usr/bin/git` copied to
  `~/mircea/bin` (on the job's PATH), and the `data` symlink ignored (f874685), so runs are no longer marked dirty.
  Timing job 879 (bedroom only, untrimmed nyu40 = the baseline, a421cc4, RTX 5090): 2:30 min wall for the whole
  scene, 2.9 GB peak RAM, refiner 1.2 s. Déjà View -> Refined: accuracy 2.705 -> 2.453 cm, completeness 1.314 ->
  1.743 cm, F@5cm 0.915 -> 0.909, mIoU 0.488 -> 0.479, AP50 0.396 -> 0.411. Same direction as Peter's run on 03.11
  (`figures/eval_hypersim-20261103-155403`), not the same numbers: his masks came from another machine (bf16).

## E33: trim masks at Déjà View depth jumps -> does not work: removes far more true mask than leak (04.11.2026)
- **Goal, change, setting:** as in the plan above; thresholds (5% jump, far side of > 50% of cuts) not changed.
- **Commit:** 1f5f991 (`trim_mask` in `src/segmentation/trim.py`, `sam3-segment --trim dejaview_k16`).
- **Command:** cluster job 881 (`~/mircea/jobs/e33_run.sbatch`, VC15, RTX 5090): `sam3-segment --mode image
  --threshold 0.3 --prompts configs/prompts/nyu40.txt --trim dejaview_k16 <3 dev views>`, then
  `python -m src.analysis.scene_eval --cfg` with `configs/eval_hypersim.yaml` and `segmentation:` set to
  `sam3image-nyu40-t0.3` (baseline) or `sam3image-nyu40-t0.3-trim` (runs `e33_baseline-20261104-131001`,
  `e33_trim-20261104-131231`, both git 1f5f991 clean, same job, same GPU). 2D scores:
  `uv run python experiments/scripts/e33_mask_trim.py > experiments/results/e33_mask_trim.txt` (CPU, laptop).
- **Runtime:** trimming costs ~10 s per scene on top of SAM 3's 11 s (21 s vs 11 s; the first scene's 199 s is
  weight loading from a slow disk). Refiner 1.0-1.6 s per scene either way.
- **`scene_eval`** (Déjà View = labels only, geometry identical; Refined = joint refiner, every weight 1):

  | scene | variant | masks | accuracy (cm) | completeness (cm) | F@5cm | mIoU | AP50 |
  |---|---|---|---|---|---|---|---|
  | living room | Déjà View | untrimmed | 2.258 | 1.037 | 0.940 | 0.543 | 0.375 |
  | living room | Déjà View | trimmed | 2.258 | 1.037 | 0.940 | 0.500 | 0.375 |
  | living room | Refined | untrimmed | 1.816 | 1.358 | 0.953 | 0.504 | 0.324 |
  | living room | Refined | trimmed | 1.826 | 1.363 | 0.953 | 0.464 | 0.371 |
  | bedroom | Déjà View | untrimmed | 2.705 | 1.314 | 0.915 | 0.488 | 0.396 |
  | bedroom | Déjà View | trimmed | 2.705 | 1.314 | 0.915 | 0.448 | 0.348 |
  | bedroom | Refined | untrimmed | 2.455 | 1.743 | 0.909 | 0.477 | 0.411 |
  | bedroom | Refined | trimmed | 2.417 | 1.745 | 0.910 | 0.431 | 0.370 |
  | lobby | Déjà View | untrimmed | 6.058 | 3.814 | 0.683 | 0.722 | 0.545 |
  | lobby | Déjà View | trimmed | 6.058 | 3.814 | 0.683 | 0.512 | 0.268 |
  | lobby | Refined | untrimmed | 5.628 | 5.156 | 0.656 | 0.564 | 0.231 |
  | lobby | Refined | trimmed | 5.640 | 5.223 | 0.651 | 0.424 | 0.218 |

  Run-to-run noise of the same setting (bedroom, job 879 vs 881): accuracy 2.453 vs 2.455 cm, mIoU 0.479 vs 0.477.
- **2D (`e33_mask_trim.txt`), per labeled GT pixel over all 20 views, untrimmed -> trimmed:** right object
  0.935 -> 0.834 (living room), 0.849 -> 0.649 (bedroom), 0.905 -> 0.626 (lobby); other object 0.005 -> 0.003,
  0.019 -> 0.011, 0.008 -> 0.006; masked share of the image 0.952 -> 0.838, 0.877 -> 0.671, 0.877 -> 0.631.
  Recall (all) 0.87 -> 0.81, 0.85 -> 0.69, 0.76 -> 0.58; leakage 0.005 -> 0.004, 0.021 -> 0.017, 0.008 -> 0.009.
  Bed (GT 52): bed masks on other GT objects 9.2% -> 6.2% of their area (wall 21.6k -> 8.6k px, floor 3.6k ->
  1.7k), but the bed's own pixels with the right object 0.786 -> 0.593.
- **Figure:** `figures/e33_mask_trim.jpg` (in `experiments/`) (bedroom, E31 views 18/0/12/6; untrimmed, trimmed, pixels that lost
  their mask). View 18 shows the intended effect: the wall and floor seen between the bed frame's bars leave the
  bed mask. Views 0, 12, 6 show the failure: a wall mask spans two wall planes with an occluding edge between
  them (the alcove behind the floor lamp, view 0: 26% of the image loses its mask), the floor is cut where the
  bed and rug occlude it, and the open wardrobe loses its door. These pixels lose their mask altogether: the
  lower-scoring masks under them do not cover them either.
- **Conclusion:** the rule does what it was written to do, and that is the wrong thing. Depth order alone cannot
  tell "seen through a hole of this object" from "a farther part of the same object or class": one wall or floor
  mask legitimately spans occluding edges. Per pixel it removes about 25x more correct mask than wrong mask
  (bedroom: right -0.200, other -0.008), so every label score drops (mIoU -0.04 to -0.21 for Déjà View).
  Geometry does not move: completeness of the refined output changes by +0.005 / +0.002 / +0.067 cm, within or
  near noise, so leaking masks are not what drives the refiner's completeness drop (plan's hypothesis rejected
  on these three scenes; the trim also removed coverage, so this is not a clean isolation of leakage). No
  threshold was changed after seeing results. Not adopted. A version restricted to thing classes with holes is
  not pursued: on the bed alone the trim removes ~128k bed-mask px, of which ~22k were leak (43k -> 21k on
  other objects) and ~105k the bed's own (mattress behind the footboard, headboard between the bars), so
  sparing wall and floor would not change the ~5:1 ratio. The trim code is reverted right after this report.

## E34: the evaluation protocol on the 11 val scenes -> refiner loses F@5cm on 7 of 10, completeness on all (05.11.2026)
- **Goal:** check on scenes no choice was made on whether the dev picture of PR #17's joint refiner holds
  (dev: accuracy better on 3/3, completeness worse on 3/3, F@5cm +0.013 / -0.006 / -0.027; E33 baseline).
  Nothing changed in the method or the scorer; the val split is the one of `configs/hypersim.yaml`.
- **Commit:** main 5f3af22 (clean) for every job; config `configs/eval_hypersim_val.yaml` (= `eval_hypersim.yaml`
  with the 11 val scenes; the job read an identical copy in `~/mircea/configs/`, the file is added here).
- **Data prep (laptop):** `uv run hypersim-download --splits val` 15.7 min for the 10 scenes not yet on disk
  (14-48 MB raw each; ai_003_010 skipped as present), `uv run hypersim-prepare --splits val` 1 min (20 views
  504x378 each, ~18 MB views + GT per scene). Contact sheets (RGB, GT depth, GT instances) of all 11 looked at
  before scoring: ai_003_010 has no labels on wall, floor and ceiling (78% of pixels unlabeled; geometry still
  scored, per-class only on objects); ai_005_005 is a bathroom of mirrors and glass (GT depth goes through the
  glass); the 21-28% "no geometry" in ai_004_004 / ai_004_010 is sea, sky and open windows, as it should be.
- **Commands (cluster, RTX 5090):** job 896 (VC15, 6:39): `dejaview-predict` per scene (12 s each incl. model
  load, 43 s for the first), `sam3-segment --mode image --threshold 0.3 --prompts configs/prompts/nyu40.txt`
  (178 s for all 11), `dsine-predict` (52 s for all 11). Job 969 (VC17, 3:53 for all 11 scenes, refiner 1.2-2.1 s
  per scene): `python -m src.analysis.scene_eval --cfg configs/eval_hypersim_val.yaml`, run
  `eval_hypersim_val-20261105-094825`. Table and figure: `uv run python experiments/scripts/e34_val_eval.py
  runs/eval_hypersim_val-20261105-094825/run.yaml > experiments/results/e34_val_eval.txt` (laptop, CPU).
- **ai_031_004 is invalid, not a result:** both variants score the cap (20 cm / 20 cm / F 0.000) because the
  registration starts at scale 421 (all other scenes 2.8-7.1). Cause: in Hypersim's raw
  `cam_00/camera_keyframe_orientations.hdf5` the rotation of keyframe 0 is exactly 10 x an orthonormal matrix
  (singular values 10, 10, 10; the other 99 keyframes and every keyframe of the other 10 val scenes have
  |R R^T - I| < 1e-6, as do all 17 prepared scenes but this one). Keyframe 0 is our view 0, which the scorer
  uses for the GT cloud and as the registration start. The loader takes the rotation as stored and does not
  check it. Déjà View, SAM 3 and DSINE never see the cameras, so their caches are fine. Excluded below; not
  fixed yet (planned: divide by cbrt(det) in `src/loaders/hypersim/scene.py`, assert a uniformly scaled
  rotation, re-prepare, rescore).
- **Results** (Déjà View = backbone output; Refined = joint refiner, every weight 1; F at 5 cm):

  | scene | acc DV (cm) | acc Ref | comp DV (cm) | comp Ref | F DV | F Ref | mIoU DV | mIoU Ref |
  |---|---|---|---|---|---|---|---|---|
  | ai_003_010 | 2.359 | 2.105 | 1.247 | 1.484 | 0.942 | 0.944 | 0.482 | 0.398 |
  | ai_004_003 (bathroom) | 2.220 | 2.398 | 1.576 | 2.129 | 0.931 | 0.902 | 0.867 | 0.841 |
  | ai_004_004 | 7.006 | 7.716 | 1.544 | 2.043 | 0.767 | 0.718 | 0.607 | 0.544 |
  | ai_004_010 | 8.215 | 7.438 | 1.470 | 1.848 | 0.722 | 0.746 | 0.829 | 0.817 |
  | ai_005_005 | 5.916 | 6.119 | 2.346 | 3.470 | 0.748 | 0.707 | 0.557 | 0.525 |
  | ai_007_001 | 5.043 | 4.699 | 2.186 | 3.049 | 0.790 | 0.786 | 0.444 | 0.396 |
  | ai_009_007 | 5.962 | 6.132 | 3.861 | 4.488 | 0.685 | 0.660 | 0.598 | 0.589 |
  | ai_009_009 | 13.210 | 11.221 | 10.739 | 10.748 | 0.299 | 0.351 | 0.199 | 0.186 |
  | ai_044_003 (cam_01) | 4.441 | 4.163 | 2.751 | 4.132 | 0.791 | 0.741 | 0.354 | 0.332 |
  | ai_050_002 | 3.916 | 3.588 | 2.554 | 3.481 | 0.810 | 0.793 | 0.555 | 0.504 |
  | **mean of 10** | 5.829 | 5.558 | 3.027 | 3.687 | 0.749 | 0.735 | 0.549 | 0.513 |

  Refined vs Déjà View, scenes better / worse: accuracy 6 / 4, completeness 0 / 10, F@5cm 3 / 7, mIoU 0 / 10,
  AP50 3 / 3 (mean 0.437 -> 0.439). Run-to-run noise on dev was ~0.002 cm (E33); one run per setting here.
- **Figure:** `figures/e34_val_eval.jpg` (in `experiments/`): F@5cm, accuracy and completeness per scene, both
  variants on shared axes, ai_031_004 greyed. The completeness bars are higher for Refined in every scene; the
  accuracy gains are largest where Déjà View is worst (ai_009_009 -2.0 cm, ai_004_010 -0.8 cm).
- **Conclusion:** the dev picture holds and is worse on val. Completeness gets worse on every scene (+0.66 cm
  on average), accuracy better on 6 of 10 (-0.27 cm on average), and at 5 cm the net effect is a loss on 7 of 10
  scenes (mean F -0.014; dev mean -0.007). Labels get worse too (mIoU lower on all 10, -0.036 on average), as on
  all three dev scenes (E33: -0.039, -0.011, -0.158). The weights were never tuned (all 1, as on dev), so this is
  the untuned refiner, not the method's ceiling; it makes the completeness study (next, E35: why the refiner
  loses completeness) the first thing to answer before any weight search. ai_009_009 (a 28 m hall) is a Déjà
  View failure (F 0.30) that this refiner barely changes (0.35). One trajectory per scene, 10 scenes: per-scene
  differences of a few thousandths are within noise; the 10 / 10 completeness and mIoU counts are not.
- **Addendum, rescored with PR #19's protocol (05.11.2026):** the protocol changed after this entry (main f645d08:
  one Umeyama Sim(3) on the pixel pairs instead of scaled ICP, Chamfer instead of F@5cm, labels scored per pixel),
  so the same caches were scored again. Cluster job 994 (VC15, RTX 5090, 2:25 for dev and val): `scene_eval`
  at main f645d08 (clean) with `--splits dev val` and two local config copies, no code change:
  `eval_hypersim.yaml` with `per_scene: [dev, val]` (val keeps per-scene scores) and a dataset yaml whose val list
  leaves out ai_031_004 (its bad camera would skew the split aggregate). Run
  `eval_hypersim_v2_val10-20261105-123026`. Table and figure: `uv run python
  experiments/scripts/e34_val_eval_v2.py runs/eval_hypersim_v2_val10-20261105-123026/run.yaml >
  experiments/results/e34_val_eval_v2.txt`; figure `figures/e34_val_eval_v2.jpg` (in `experiments/`): Refined
  minus Déjà View per scene for accuracy, completeness and Chamfer on one cm scale, split means on top.

  | split | accuracy (cm) | completeness (cm) | Chamfer (cm) | scenes better: acc / comp / Chamfer |
  |---|---|---|---|---|
  | dev (3) | 4.651 -> 4.178 | 2.307 -> 3.657 | 3.479 -> 3.917 | 3 / 0 / 1 |
  | val (10) | 6.213 -> 5.884 | 4.100 -> 5.685 | 5.156 -> 5.785 | 8 / 0 / 1 |

  Label scores are identical for both variants, as the protocol intends (val mIoU 0.392, PQ^sc 0.265, AP 0.353;
  dev 0.352 / 0.294 / 0.217). The one dev scene with better Chamfer is the living room by 0.004 cm, the one val
  scene is ai_005_005 (4.504 -> 4.450); ai_004_010 is worse by 0.003 cm, within noise. Dev agrees with PR #19's
  RTX 3080 run to 0.03 cm (Chamfer 3.50 -> 3.95 there), the expected cross-GPU difference.
  **Conclusion:** the finding holds under the new protocol: completeness worse on every scene (val +1.59 cm), a
  smaller accuracy gain (val -0.33 cm), Chamfer worse on 9 of 10 val scenes. Per-scene numbers move a lot
  between the two alignments (bathroom ai_004_003 Déjà View completeness 1.58 -> 10.48 cm; ai_009_009 accuracy
  better under ICP, worse under Umeyama), so per-scene claims depend on the alignment; the split-level pattern
  does not.
- **Addendum, loader fix for ai_031_004 (08.11.2026):** `src/loaders/hypersim/scene.py` now passes every stored
  orientation through `unscaled_rotation`: divide by cbrt(det), then require |R^T R - I| <= 1e-5; a reflection,
  a non-uniform scale or a shear raises (tests: three scales recovered to 1e-12, the three bad forms rejected).
  Only keyframe 0 of ai_031_004 cam_00 has cbrt(det) != 1 (10.000000). Re-prepared views differ from the old ones
  only in view 0's `cam_to_world` (images, intrinsics and GT identical); the camera rebuilt from the fixed one
  times 10 matches the old file to 5e-7. Check: `uv run python experiments/scripts/e34_ai031004_camera.py >
  experiments/results/e34_ai031004_camera.txt` (laptop, 35 s) lifts each view's GT z-depth, projects it into every
  other view and compares depths where the point is visible (within 5%):

  | camera of view 0 | view 0 pairs that overlap | median rel. depth error, view 0 | other 342 pairs |
  |---|---|---|---|
  | stored (10 R) | 0 / 38 | - | 322 overlap, 0.083% |
  | unscaled | 38 / 38 (mean overlap 34.6%) | 0.104% | 322 overlap, 0.083% |

  Figure `figures/e34_ai031004_camera.jpg` (in `experiments/`): GT points from above; with the stored camera,
  view 0's points spread 10 times too far (out to 80 m in a courtyard about 10 m across), after the fix they lie on the other
  views' walls. **Conclusion:** the fixed GT is multi-view consistent, as well as any other pair in the scene
  (0.104% vs 0.083% median), so ai_031_004 can be scored. Déjà View, SAM 3 and DSINE caches need no rerun
  (they never read cameras); only `views_n20_uniform_504.npz` was replaced on the cluster (old copy kept).
- **Addendum, all 11 val scenes after the fix (08.11.2026):** cluster job 1373 (VC17, RTX 5090, 2:14), `scene_eval` at
  main f645d08 (clean) with `--splits val` and a config copy that differs from the addendum above only in reading
  `configs/hypersim.yaml` (all 11 val scenes); run `eval_hypersim_v2_val-20261108-183432`. Table: `uv run python
  experiments/scripts/e34_val11.py runs/eval_hypersim_v2_val-20261108-183432/run.yaml >
  experiments/results/e34_val11.txt`. ai_031_004 now registers at scale 2.78 (other scenes 2.75-7.24; 421 before).

  | | accuracy (cm) | completeness (cm) | Chamfer (cm) | scenes better: acc / comp / Chamfer |
  |---|---|---|---|---|
  | ai_031_004 (courtyard) | 4.586 -> 4.399 | 4.623 -> 5.525 | 4.604 -> 4.962 | |
  | val (11) | 6.065 -> 5.749 | 4.147 -> 5.671 | 5.106 -> 5.710 | 9 / 0 / 1 |
  | the other 10, this run | 6.213 -> 5.885 | 4.100 -> 5.685 | 5.156 -> 5.785 | |

  The other 10 scenes reproduce job 994 (same GPU type): Déjà View identical to 0.001 cm, Refined accuracy 5.885 vs
  5.884. Label scores are unchanged by the refiner, as the protocol intends (val mIoU 0.395, PQ^sc 0.256, AP 0.353).
  **Conclusion:** ai_031_004 behaves like the rest of val: accuracy better, completeness worse, Chamfer worse.
  With it, completeness is worse on 11 of 11 val scenes and Chamfer on 10 of 11; the split means move by less
  than 0.15 cm, so nothing in the conclusions above changes. Val is now complete; quote these 11-scene numbers.
- **Note (08.11.2026):** `configs/eval_hypersim_val.yaml` is deleted on merging main: it is in the config format
  from before PR #19, which `scene_eval` no longer reads. Val now runs with `configs/eval_hypersim.yaml --splits
  val`, whose `per_scene` now includes val (the copy job 1373 read differs only in a comment and in not keeping
  stress per scene, which a `--splits val` run never reads). The first E34 run
  stays reproducible from 5f3af22 with the file as it was in ca1b6b8.

## Plan: E35 = where Déjà View's error comes from, in and out of domain, and where the refiner loses completeness (09.11.2026)
- **Goal:** split the error of Déjà View, and of Déjà View + joint refiner, by its source instead of one number
  per scene, so that the next method aims at the largest part (feedback on 05.11 and the split agreed in issue
  #13). Takes in the completeness study announced as E35 in E34's conclusion: the same split answers why the
  refiner gains accuracy and loses completeness (val: completeness 4.15 -> 5.67 cm, worse on 11 of 11 scenes).
  Analysis only: no change to the method, the scorer or the backbones.
- **Scenes:**
  - *Out of domain:* the 13 scenes of ETH3D's high-res multi-view training set (7 indoor, 6 outdoor, 14-76 DSLR
    images each, depth rendered from laser scans). ETH3D is not in Déjà View's 29-dataset training mix (paper
    Table 6); the paper evaluates on it zero-shot, which gives a sanity check of our numbers. Geometry only:
    ETH3D has no labels.
  - *In domain:* Hypersim dev (3) and val (11), the scenes of E34; stress (3 tilt-shift, E30) as the known
    off-centre-camera failure. Test is not used.
- **ETH3D loader (new):** undistorted DSLR JPEGs (COLMAP PINHOLE cameras, `cameras.txt` / `images.txt`) to the
  `Views` + `GroundTruth` format of the Hypersim loader: 20 views per scene by `select_frames(..., "uniform")` in
  file-name order (all of them for pipes, 14, and meadow, 15), long side 504 (504x336), images resized as in
  Hypersim, GT sampled nearest. GT depth: ETH3D's depth maps are raw float32 dumps, infinity where no scan point,
  and match the *distorted* originals (THIN_PRISM_FISHEYE), not the undistorted images. Each undistorted pixel is
  therefore mapped through the distortion model of its original camera and its depth sampled nearest there
  (no mixing across edges); no hit -> `NO_MESH`, every hit an unlabeled pixel. Two facts are checked on the data
  before any score, not assumed: whether the stored value is z-depth or distance along the ray, and that it is
  in metres. Check as in E34's ai_031_004 addendum: lift each view's GT, project into every other view, median
  relative depth error where visible (a z / ray mix-up grows towards the image corners). Glass has no laser
  return, so ETH3D has no GT there; nothing to separate.
- **Error split** (pixel pairs, after the protocol's one Umeyama Sim(3) per scene; per-pixel error = 3D distance
  to its GT point; acc / comp / Chamfer reported as well):
  1. *Camera / global:* align each view on its own (one Sim(3) per view) and rescore. The error that disappears
     comes from views placed or scaled wrong relative to each other, not from their depth.
  2. *Cross-view inconsistency:* for pixel pairs that the GT says see the same surface (reprojected GT depth
     within 5%, as in E34), the relative disagreement of the predicted depths; share of pairs over 5% (double
     surfaces).
  3. *Edges:* error by distance to the nearest GT depth edge (relative jump > 5% between 4-neighbours, the
     E18 / E31 definition) in bins 0-1, 1-3, 3-10, > 10 px, against each bin's share of pixels. Bins instead of
     one band width, so no width is chosen.
  4. *Object placement vs shape* (Hypersim, GT instances with >= 200 px over the 20 views): after the scene
     Sim(3), fit each object's own isotropic scale and 3D shift to its GT points (closed form); error that stays
     is shape. On ETH3D only with SAM 3 instances (nyu40 prompts), reported apart because mask errors mix in.
  5. *Glass and mirrors* (Hypersim NYU40 `window` and `mirror`): reported apart and left out of 1-4.
- **Completeness (the former E35 study), DV -> refined, per GT pixel:** the change of its distance to the
  nearest predicted point, summed by NYU40 class, by edge-distance bin (as in 3), by the number of views that
  see the point (GT visibility: 1, 2-3, 4-7, 8+), by Déjà View confidence (quartiles per scene) and by how far
  its own pixel's depth moved (|delta| quartiles). Also: non-finite refined points, and in the 0-3 px edge bins
  the share of pixels whose refined depth lies closer to the GT depth across the edge than to their own (two
  surfaces pulled into one). Figure: GT points that lose coverage, DV and refined side by side on a shared cm
  scale (bedroom `ai_006_007`, views 18, 0, 12, 6). Also checks the registration scale on the bedroom and the
  lobby (E35's open item; living room was within 0.1%).
- **Hypotheses, written before running:** (H1) the completeness loss sits in the 0-3 px edge bins: E_n and E_mv
  act within one instance, so a far surface inside the same mask is pulled to the near one (E33 removed leaking
  masks and nothing moved, so this would be within correct masks). (H2) it sits where few views see a point:
  E_mv pulls views to a consensus that rarely-seen surfaces cannot hold. (H3) it sits at low confidence, where the
  Huber anchor is weak. Out of domain, (H4) the camera / inconsistency share grows on ETH3D relative to Hypersim.
  None of these is a claim until the numbers say so.
- **What the result decides (as in #13):** camera or inconsistency dominant -> test-time adaptation of Déjà View;
  edges dominant -> 3D or MoGe-3 refinement (Pouya, #20) against a confidence / gradient filter baseline (Peter);
  shape dominant -> a learned per-object shape prior; Déjà View fine everywhere on ETH3D -> stay on the
  off-centre camera case. The edge bins test the premise of Pouya's 3D refiner (2D refinement does not fix edges).
- **Fixed before running:** 20 views, long side 504, Déjà View K=16, the protocol of `configs/eval_hypersim.yaml`,
  every refiner weight 1, the 5% thresholds, the bins and the 200 px object size. Nothing is chosen on these
  numbers; if a threshold has to change, the entry says so.
- **Cost:** download ~12 GB (undistorted images 5.5 GB, depth maps ~5.9 GB summed over scenes, plus the distorted
  calibration, if it is not in the undistorted archive). Prepare on the laptop, upload the npz files (~20 MB per
  scene). Cluster: one job for Déjà View, SAM 3 nyu40 and DSINE on the 13 ETH3D scenes (~10 min, from E34's
  per-scene times), `scene_eval` on ETH3D and Hypersim dev/val/stress (~10 min), then the split on CPU. ETH3D's
  licence is not stated on its download page: data stays local, and its terms are checked before any ETH3D image
  goes into a committed figure.
- **Numbering:** this entry takes E35, as E34 announced; Pouya's learned-refiner entry (E34 on
  `pouya/learned-refiner`) moves to E36 when it merges.
- **Progress 09.11, ETH3D loader (`src/loaders/eth3d/`, `configs/eth3d.yaml`):** `uv run eth3d-download`
  fetches three archives per scene (undistorted images, distorted JPEGs for their calibration, depth maps;
  17 GB in all, ~75 min at ~3.5 MB/s), `uv run eth3d-prepare` reads the 20 picked frames straight out of
  them (~1 min per scene; unpacked, the depth maps alone would be ~44 GB). Three things the download page does
  not say, found on the data:
  - The depth maps belong to the *distorted* images (6048 x 4032; undistorted 6220 x 4141). The loader looks
    each pixel up through our reimplementation of COLMAP's THIN_PRISM_FISHEYE model; mapped this way the
    distorted pipes image matches the undistorted one to 1.1 grey levels on average (11.8 without the mapping).
  - They hold **z-depth in metres**, and are **dotted**: laser points splatted on ~1 in 10 pixels (pipes: 9-13%
    of full-resolution pixels). Sampling at the pixel centre gave depth on 11% of pipes' pixels; each 504 px
    pixel now takes the dot nearest its centre within its footprint (~12 x 12 source pixels): 78%. One laser
    sample per pixel, so no mixing across edges, as Hypersim's nearest sampling.
  - The undistorted principal point is ~1 px off centre at 504 px (pipes: +0.75, -1.0). Images are cropped
    symmetrically about it (~0.5% of the pixels), so ETH3D does not repeat E30's off-centre camera: off by
    <= 0.04 px in every view after the crop.
  Several scenes use more than one camera (courtyard: 4); every image keeps its own.
- **GT check** (`uv run python experiments/scripts/e35_eth3d_gt_check.py >
  experiments/results/e35_eth3d_gt_check.txt`, laptop CPU, ~10 min), all 13 scenes, picked views only:

  | check | result over the 13 scenes |
  |---|---|
  | GT depth on prepared pixels | 47-92% (outdoor 47-73%: sky, glass, outside the scan; indoor 78-92%) |
  | SfM points reprojected with our camera model vs COLMAP's observations | median 0.65-0.90 px, 90th percentile 1.9-2.5 px (6048 px images) |
  | GT depth / camera z, image radius 0-1000 / 1000-2000 / 2000-4000 px | 0.9992-1.0003 in every bin |
  | GT depth / distance to the camera centre, same bins | 0.96-0.99 / 0.90-0.93 / 0.79-0.83 |
  | multi-view: median rel. depth error of GT seen from another view | 0.049-0.120% (pairs overlapping: 134-326 of 380) |

  Hypersim's GT, checked the same way in E34, gave 0.083% (ai_031_004). Figure
  `figures/e35_eth3d_gt_check.jpg` (in `experiments/`): the middle view of each scene and its GT depth; edges
  and windows line up with the image, glass has no laser return (white), indoor depths reach 43-56 m in
  delivery_area, relief and relief_2; where a view shows them, they are the far ends of long corridors. ETH3D data: CC BY-NC-SA 4.0, Schöps et al., CVPR 2017.
  **Conclusion:** the camera model, the depth convention and the scale hold on every scene, and the prepared
  GT is as multi-view consistent as Hypersim's; ETH3D can be scored. Not yet run: Déjà View, SAM 3, DSINE and
  the refiner on these scenes (cluster job).
- **Scoring ETH3D (decided 09.11):** in a separate E35 script that calls the same refiner and scorer functions
  as `src.analysis.scene_eval` (same protocol: one Umeyama Sim(3) on the pixels, accuracy / completeness /
  Chamfer; no label scores, as ETH3D has no labels), not by extending `scene_eval`, so the evaluation
  module stays as it is. Hypersim dev / val / stress keep running through `scene_eval`.

## E35: where the error comes from -> edges in domain, cameras out of domain; the refiner's completeness loss follows low confidence, not edges (09.11.2026)
- **Goal, scenes, split and hypotheses:** as in the plan entry above, run as planned except where noted.
- **Commits:** 5ca14a9 (`configs/eval_eth3d.yaml`, `experiments/scripts/e35_eth3d_eval.py`), 4487a62 (pinhole
  bound, below), 47b531e and this commit (`experiments/scripts/e35_error_split.py`,
  `experiments/scripts/e35_figures.py`).
- **Commands** (cluster, RTX 5090; `runs/` stays local):
  - Backbones on ETH3D: `dejaview-predict`, `sam3-segment --mode image --threshold 0.3 --prompts
    configs/prompts/nyu40.txt`, `dsine-predict` on the 13 scenes (job 1488, 7:43: Déjà View 13-17 s per scene,
    SAM 3 3:09 for all, DSINE 54 s); SAM 3 and DSINE for the 3 stress scenes, same settings (job 1489).
  - `uv run python -m src.analysis.scene_eval --cfg configs/eval_hypersim.yaml --splits dev val stress` (job 1492,
    3:14, git 5ca14a9, run `eval_hypersim-20261109-145132`): dev and val equal E34's.
  - `uv run python experiments/scripts/e35_eth3d_eval.py --cfg configs/eval_eth3d.yaml` (job 1495, 1:27, git
    47b531e, run `eval_eth3d-20261109-152420`) -> `results/e35_eth3d_eval.txt`.
  - `uv run python experiments/scripts/e35_error_split.py runs/<run> --maps <scene>` on both runs (CPU, 1-3 min)
    -> `results/e35_error_split_{hypersim,eth3d}.txt`; the completeness it recomputes equals the run's on all
    30 scenes (to 0.001 cm).
  - `uv run python experiments/scripts/e35_figures.py runs/eval_hypersim-20261109-145132
    runs/eval_eth3d-20261109-152420` (laptop) -> `results/e35_figures.txt` and the three figures below.
- **Changed after the plan:**
  - *Pinhole bound.* Every view's ray map is replaced by its pinhole fit (`fit_pinhole`), which asserted a
    median ray-to-pinhole distance below 3 px (set on ScanNet++, 0.6-0.9 px). Déjà View's rays on ETH3D exceed it
    in 19 of 259 views over 7 scenes (per-scene median over views 0.55-2.81 px; relief_2 9 of 20 views; worst view
    5.72 px, playground), so the first ETH3D run stopped. Leaving those scenes out would remove the
    out-of-domain cases E35 is about; the bound moved to 10 px (4487a62), a value for a broken fit rather than
    one just above the ETH3D maximum. In-domain scores do not change (every Hypersim view was below 3 px).
  - *Objects on SAM 3 instances for Hypersim too.* The plan fitted GT instances on Hypersim and SAM 3 instances
    on ETH3D; the first numbers showed that the two do not compare, so both are reported on Hypersim.
- **ETH3D, protocol scores, Déjà View -> refined (cm, mean of scenes):**

  | scenes | accuracy | completeness | Chamfer |
  |---|---|---|---|
  | indoor (7) | 6.70 -> 6.46 | 4.43 -> 5.12 | 5.56 -> 5.79 |
  | outdoor (6) | 18.69 -> 18.67 | 11.67 -> 13.91 | 15.18 -> 16.29 |
  | all 13 | 12.23 -> 12.09 | 7.77 -> 9.18 | 10.00 -> 10.64 |

  Completeness and Chamfer worse on 13 of 13 scenes, accuracy better on 10 (worse on electro, playground,
  terrace): the pattern of E34 on Hypersim val. Déjà View fails on playground (accuracy 48.8 cm; the next worst,
  terrace, 21.9). Not compared with the Déjà View paper's ETH3D numbers (other metric and alignment).
- **Error split, Déjà View** (per-pixel error = distance to the pixel's own GT point after the protocol's Sim(3);
  scenes weighted equally; full tables per group and per scene in the results files):

  | | Hypersim dev (3) | val (11) | stress (3) | ETH3D indoor (7) | outdoor (6) |
  |---|---|---|---|---|---|
  | per-pixel error, cm | 14.58 | 17.60 | 26.21 | 13.86 | 38.78 |
  | 1. one Sim(3) per view instead, cm | 13.53 (-7%) | 16.95 (-4%) | 27.37 (+4%) | 10.39 (-25%) | 30.94 (-20%) |
  | 2. view pairs disagreeing by > 5% | 0.049 | 0.080 | 0.155 | 0.047 | 0.051 |
  | 3. pixels within 3 px of a GT depth edge | 0.108 | 0.144 | 0.223 | 0.026 | 0.038 |
  | 3. their share of the error | 0.450 | 0.475 | 0.417 | 0.068 | 0.070 |
  | 4. GT objects: error -> own scale + shift, cm | 15.05 -> 15.14 | 16.67 -> 16.08 | 23.36 -> 22.53 | - | - |
  | 4. SAM 3 objects: error -> own scale + shift, cm | 11.56 -> 9.92 | 14.04 -> 10.20 | 22.22 -> 15.88 | 12.35 -> 7.63 | 58.58 -> 13.83 |
  | 5. glass / mirror: pixel share, error cm | 0.004, 9.6 | 0.022, 27.9 | 0.043, 13.8 | - | - |

  The Sim(3) fits are least squares, so the mean distance can rise where the alignment was already right (stress
  per view, GT objects on dev). Every refined point is finite.
- **Completeness change, refined - Déjà View** (shares of the change; contributions sum to it):

  | | Hypersim dev | val | stress | ETH3D indoor | outdoor |
  |---|---|---|---|---|---|
  | change, cm | +1.35 | +1.52 | +1.17 | +0.69 | +2.24 |
  | within 3 px of a GT depth edge | 0.20 | 0.26 | 0.28 | 0.04 | 0.03 |
  | more than 10 px from one | 0.57 | 0.48 | 0.50 | 0.84 | 0.90 |
  | lowest-confidence quarter of pixels | 0.39 | 0.57 | 0.49 | 0.58 | 0.50 |
  | points seen by 1 / 2-3 / 4-7 / 8+ views | .02/.05/.12/.81 | .04/.13/.28/.56 | .06/.13/.19/.62 | .04/.32/.55/.10 | .06/.34/.44/.16 |
  | wall pixels (Hypersim) | 0.47 | 0.62 | 0.39 | - | - |
  | near-edge pixels nearer the far side, Déjà View -> refined | 0.222 -> 0.218 | 0.237 -> 0.235 | 0.237 -> 0.236 | 0.252 -> 0.248 | 0.299 -> 0.299 |

  Per scene, the lowest-confidence quarter takes more than its 25% of the loss in 29 of 30 scenes; the
  exception is the bedroom ai_006_007 (0.23). The refiner halves the median cross-view gap (Hypersim dev 0.58 ->
  0.26%, ETH3D 0.79 -> 0.53%) while coverage gets worse.
- **Figures:** `figures/e35_error_split.jpg` (the two tables as bars per group);
  `figures/e35_coverage_ai_006_007.jpg` (bedroom views 18, 0, 12, 6) and `figures/e35_coverage_office.jpg`
  (ETH3D office views 0, 5, 10, 15, fixed before looking): image, Déjà View confidence, each GT pixel's
  completeness distance before and after on one cm scale, and the difference. In both scenes the difference is
  red over whole walls, doors and ceilings, not in bands along depth edges; office's worst patch is its
  low-confidence ceiling (view 10).
- **Hypotheses:**
  - H1 (completeness loss in the 0-3 px edge bins): **rejected.** Near-edge pixels lose more per pixel on Hypersim
    (20-28% of the loss on 11-22% of the pixels), but 48-90% of the loss lies more than 10 px from any edge, and
    no more near-edge pixels end up nearer the far side of their edge than before.
  - H2 (where few views see a point): **rejected.** Points seen by 1-3 views hold 7-19% of the loss on Hypersim
    (17-20% of the pixels) and 36-40% on ETH3D (28-30%): at most a mild excess; most of the loss is on points
    seen by 4 or more views.
  - H3 (at low confidence): **supported** on both datasets, and per scene in 29 of 30.
  - H4 (camera / inconsistency grow out of domain): **half.** Per-view alignment removes 20-25% of Déjà View's
    error on ETH3D against 4-7% on Hypersim, so placing the views relative to each other gets worse out of
    domain; views do not disagree more (0.047-0.051 of pairs over 5% on ETH3D, 0.049-0.080 on Hypersim dev and
    val).
- **Conclusion:**
  - In domain, depth edges carry the largest share of Déjà View's error (42-48% within 3 px of an edge on
    Hypersim); this supports the premise of the 3D refiner (#20) on Hypersim. On ETH3D the dotted laser GT
    under-counts edges (2.6-3.8% of pixels near one), so the edge share there is not comparable.
  - Out of domain, the camera / global part grows from about 5% to 20-25% of the error, and Déjà View's rays move
    further from a pinhole (up to 5.7 px per view); test-time adaptation or pose refinement aims at this part.
  - The object split is no clean measure: on the same Hypersim scenes the fit removes 4% on GT instances and
    27% on SAM 3's, so it depends on how the segments are cut, not only on placement.
  - The joint refiner's completeness loss is not an edge effect and not holes: whole low-confidence surfaces
    (walls, ceilings) move off the GT while the views agree more with each other. Where confidence is low the
    confidence-weighted anchor E_d is weak and E_mv / E_n set the surface, and a self-consistent consensus off
    the GT looks like this. Not tested here: rerun with E_d's weight not scaled by confidence (one change), or
    with E_mv off on the lowest-confidence quarter, and see whether the loss goes.

## Plan: E37 = E_d's weight without Déjà View's confidence, against E35's completeness loss (09.11.2026)
- **Goal:** test E35's reading of H3. E_d's weight per pixel is w = (C / mean C) * u (`build_problem`), so where
  Déjà View's confidence C is low the Huber anchor is weak and E_mv / E_n set the surface. If that is why whole
  low-confidence walls and ceilings move off the GT, dropping C from w should take most of the completeness loss
  away, above all in the lowest-confidence quarter.
- **Change (one):** `RefineConfig.conf_power` p, w = (C / mean C)^p * u. p = 1 is the refiner as it is (bit for
  bit); p = 0 leaves the consistency u alone. Everything else as E35: every weight 1, the same Gauss-Newton / CG
  budget, the same inputs, alignment and scorer.
- **Scenes:** as E35, Hypersim dev (3), val (11), stress (3) through `scene_eval`, ETH3D's 13 through
  `e35_eth3d_eval.py`; dev is where a choice would be made, val and ETH3D check it. No test scene.
- **Variants, in one run per dataset so both refiners share a GPU (bf16 is not bit-reproducible across GPUs):**
  Déjà View; Refined (p = 1); Refined, p = 0.
- **Measured:** accuracy, completeness, Chamfer per group and per scene; E35's completeness split (by confidence
  quartile, edge bin, views, class) of each refined variant against Déjà View; coverage maps of the bedroom
  (views 18, 0, 12, 6) and ETH3D office (views 0, 5, 10, 15), all three variants on one cm scale.
- **Expected, written before running:** completeness loss on val and ETH3D at most half of p = 1's, the
  lowest-confidence quarter's share of the loss falling below its 25% of the pixels; accuracy gives back part of
  its gain (low-confidence pixels are where Déjà View is least accurate, so anchoring them harder keeps their
  error). If completeness does not move, H3's mechanism is not the anchor weight but something E_mv / E_n do
  regardless of it, and the next step is E35's other variant (E_mv off on the lowest quarter).
- **Fixed before running:** p in {1, 0} only; no value in between is tried on these numbers.
- **Cost:** E35's jobs took 3:14 (Hypersim, 2 variants) and 1:27 (ETH3D); a third variant adds about half, so one
  job of ~8 min on one RTX 5090.
- **Numbering:** E36 stays reserved for Pouya's learned-refiner entry, as E35 announced.

## E37: E_d without Déjà View's confidence -> does not work: removes 1-3% of the completeness loss (09.11.2026)
- **Goal, change, scenes and expectations:** as in the plan entry above, run as planned.
- **Commits:** 0f9f210 (`RefineConfig.conf_power`, tests), b90bc67 (plan, `experiments/configs/e37_{hypersim,eth3d}.yaml`,
  `e35_error_split.py --variants`), this commit (`experiments/scripts/e37_anchor.py`, `e37_figures.py`).
- **Commands** (cluster, one RTX 5090, VC18, git b90bc67 clean; `runs/` stays local):
  - job 1503, 9:18 in all: `scene_eval --cfg experiments/configs/e37_hypersim.yaml --splits dev val stress` (2:29,
    run `e37_hypersim-20261109-162959`), `e35_eth3d_eval.py --cfg experiments/configs/e37_eth3d.yaml` (2:04, run
    `e37_eth3d-20261109-163228`), then `e35_error_split.py <run> --variants 0 1` and `--variants 0 2` on both
    -> `results/e37_eval.txt` (log), `results/e37_error_split_{hypersim,eth3d}.txt` (p = 0 against Déjà View).
    Job 1504 redid the Hypersim splits with `--maps ai_006_007_cam_00` (1503 passed the name without `_cam_00`).
  - `uv run python experiments/scripts/e37_anchor.py` (laptop, the 7 scenes whose Déjà View output is cached
    there) -> `results/e37_anchor.txt`; `uv run python experiments/scripts/e37_figures.py runs/e37_hypersim-20261109-162959
    runs/e37_eth3d-20261109-163228` -> `results/e37_figures.txt` and the figures below.
- **Reproduction:** the p = 1 variant equals E35 (completeness within 0.001 cm on every scene).
- **Protocol scores, cm, mean of scenes (Déjà View / refined p = 1 / refined p = 0):**

  | scenes | accuracy | completeness | Chamfer |
  |---|---|---|---|
  | Hypersim dev (3) | 4.65 / 4.18 / 4.18 | 2.31 / 3.66 / 3.63 | 3.48 / 3.92 / 3.91 |
  | Hypersim val (11) | 6.07 / 5.75 / 5.75 | 4.15 / 5.67 / 5.63 | 5.11 / 5.71 / 5.69 |
  | ETH3D indoor (7) | 6.70 / 6.46 / 6.46 | 4.43 / 5.12 / 5.11 | 5.56 / 5.79 / 5.78 |
  | ETH3D outdoor (6) | 18.69 / 18.67 / 18.65 | 11.67 / 13.91 / 13.89 | 15.18 / 16.29 / 16.27 |

  p = 0 has the better completeness than p = 1 on 26 of 30 scenes (dev 2/3, val 10/11, stress 3/3, ETH3D 11/13),
  by at most 0.23 cm (ai_009_009, 19.23 -> 19.01; worse by up to 0.09, playground); the loss against Déjà View
  stays (val +1.52 -> +1.49 cm). Accuracy moves by at most 0.14 cm either way (ai_009_009 better, 21.35 -> 21.22).
- **Completeness change by Déjà View confidence quartile** (`results/e37_error_split_*.txt`; contributions in cm,
  they sum to the change): the lowest quarter carries val +0.87 -> +0.84, ETH3D outdoor +1.11 -> +1.11, ETH3D
  indoor +0.40 -> +0.37, dev +0.53 -> +0.51 (p = 1 -> p = 0). Its share of the loss does not fall.
- **Why so little** (`results/e37_anchor.txt`): inside Huber's threshold the per-pixel anchor's stiffness on δ is
  λ_d w / 2 from E_d plus λ_δ from ‖δ‖², which no confidence scales. With every weight 1 that is at most
  1 + c / 2: on the 7 cached scenes the confidence c (divided by its view's mean) averages 0.43 in its lowest
  quarter and 1.69 in its highest, so the stiffness bound is 1.22 there against 1.85, and p = 0 raises it to 1.5.
  Low-confidence pixels were never anchored much more weakly than the rest, and p = 0 stiffens them by about a fifth.
- **Figures:** `figures/e37_confidence.jpg` (the change per group and quartile, both refiners side by side);
  `figures/e37_coverage_ai_006_007.jpg` and `figures/e37_coverage_office.jpg` (Déjà View confidence, each GT
  pixel's completeness change for p = 1 and p = 0 on one ±5 cm scale, and p = 0 - p = 1 on ±1 cm). The p = 1 and
  p = 0 maps look the same; their difference is below 1 cm except in small patches (office view 10's ceiling
  beams, bedroom's lamp and bed). In office view 0 the high-confidence left wall turns red as much as the
  low-confidence ones: the loss spreads over whole surfaces whatever their confidence.
- **Conclusion:**
  - E35's reading of H3 is wrong in its mechanism: the loss does sit on low-confidence pixels, but not because
    the confidence weight leaves them unanchored; E_d is at most a third of the anchor next to λ_δ‖δ‖².
    Confidence most likely marks where Déjà View's views disagree, which is where E_mv moves the depth most (E35:
    the quarter of pixels whose depth moved most carries 34-46% of the loss on Hypersim, 37-54% on ETH3D). This
    is a reading, not a measurement.
  - Not worth keeping as a method change: p = 0 is a consistent but negligible gain. `conf_power` stays at 1 in
    `configs/refine.yaml`.
  - Not tested, next candidates: (a) whether completeness rewards spread: Déjà View's 20 overlapping layers of a
    wall scatter around the GT, so some layer lies near every GT point; E_mv pulls them into one surface (E35:
    median cross-view gap halved), and if that surface sits off the GT every GT point is further away. Scoring
    each view's cloud alone before and after refinement would tell. (b) E35's other variant, E_mv off on the
    lowest-confidence quarter. (a) is analysis only and comes first.

## Plan: E38 = does completeness reward Déjà View's spread, which the refiner merges away (09.11.2026)
- **Goal:** test E37's candidate (a). Each scene is 20 overlapping views, so Déjà View gives every surface up to
  20 layers that scatter around the GT; completeness takes the nearest point of any layer, and some layer lies
  near almost every GT point. E_mv pulls the layers into one surface (E35: median cross-view gap halved). If
  that surface sits off the GT, fused completeness gets worse even when no single view does. Analysis only: no
  change to the method, the scorer or any run.
- **Inputs:** E37's two runs (git b90bc67), variants 0 (Déjà View) and 1 (Refined, the refiner as it is, p = 1);
  Hypersim dev (3), val (11), stress (3), ETH3D (13). Every point aligned by the run's own scene Sim(3).
- **Measured, per scene** (GT pixels as the protocol's completeness: every pixel with GT depth, all views):
  - *Fused completeness* C_f: a GT point's distance to the nearest predicted point of any view (the protocol's).
  - *Own-view completeness* C_o: a GT point of view i to the nearest predicted point of view i alone.
  - *Fusion bonus* B = C_o - C_f >= 0, per GT point. Exactly ΔC_f = ΔC_o - ΔB (refined minus Déjà View,
    averaged over the same GT points), so the completeness loss splits into "each view moved off the GT" (ΔC_o)
    and "the views stopped covering for each other" (-ΔB).
  - *Spread:* per GT point, the number of views with a predicted point within 2 cm and within 5 cm; the share of
    GT points whose nearest predicted point comes from another view than their own (C_f < C_o).
  - ΔC_o and -ΔB by Déjà View confidence quartile (per scene, as E35), to tie back to H3.
- **Decision, fixed before running:** on Hypersim val and ETH3D, if -ΔB carries at least half of ΔC_f, the loss
  is mainly the merge of a spread that completeness rewarded (and the metric, not only the surface, is part of
  the story); if ΔC_o carries more than half, each view's surface moves off the GT and the refiner itself is
  wrong there. Thresholds 2 and 5 cm only; no other value is tried on these numbers.
- **Expected, written before running:** -ΔB at least half of the loss on val, since E35's coverage maps show
  the loss on large surfaces where views agree more after refinement; ΔC_o positive but smaller.
- **Figures:** per group, ΔC_f split into ΔC_o and -ΔB, Déjà View and Refined; histogram of views within 2 cm;
  bedroom ai_006_007 views 18, 0, 12, 6 and ETH3D office views 0, 5, 10, 15: ΔC_o and -ΔB per GT pixel on one
  cm scale.
- **Cost:** CPU only on the cluster, one k-d tree per view and scene; expected under 30 min with 16 cores.

## E38: most of the completeness loss is the views no longer covering for each other, not each view moving off the GT (09.11.2026)
- **Goal, inputs and decision rule:** as in the plan entry above, run as planned.
- **Commits:** the plan commit (plan, `experiments/scripts/e38_spread.py`), this commit (`e38_figures.py`, results,
  figures).
- **Commands** (cluster, CPU job 1519, 16 cores, git b90bc67 + `e38_spread.py`, 5 min for both runs):
  `e38_spread.py runs/e37_hypersim-20261109-162959 --maps ai_006_007_cam_00` and
  `e38_spread.py runs/e37_eth3d-20261109-163228 --maps office` -> `results/e38_spread_{hypersim,eth3d}.txt`;
  laptop: `uv run python experiments/scripts/e38_figures.py runs/e37_hypersim-20261109-162959
  runs/e37_eth3d-20261109-163228` -> `results/e38_figures.txt` and the figures below.
- **Check:** C_f equals the run's own completeness to 0.001 cm on all 30 scenes.
- **Completeness split, cm, mean of scenes, variant 0 (Déjà View) -> 1 (Refined, p = 1):**

  | scenes | C_f | C_o | B | ΔC_f = ΔC_o + (-ΔB) | -ΔB share |
  |---|---|---|---|---|---|
  | Hypersim dev (3) | 2.31 -> 3.66 | 5.15 -> 5.17 | 2.85 -> 1.52 | +1.35 = +0.02 + 1.33 | 0.99 |
  | Hypersim val (11) | 4.15 -> 5.67 | 7.03 -> 7.65 | 2.88 -> 1.98 | +1.52 = +0.62 + 0.90 | 0.59 |
  | Hypersim stress (3) | 5.96 -> 7.13 | 9.46 -> 9.83 | 3.50 -> 2.70 | +1.17 = +0.37 + 0.80 | 0.68 |
  | ETH3D indoor (7) | 4.43 -> 5.12 | 6.87 -> 7.04 | 2.44 -> 1.92 | +0.69 = +0.18 + 0.52 | 0.75 |
  | ETH3D outdoor (6) | 11.67 -> 13.91 | 16.43 -> 17.33 | 4.76 -> 3.42 | +2.23 = +0.90 + 1.33 | 0.60 |

  -ΔB > ΔC_o on 8 of 11 val scenes and 9 of 13 ETH3D scenes. The scenes where ΔC_o dominates are Déjà View's
  failures: ai_009_009 (28 m hall, ΔC_o +4.93 cm) and playground (accuracy 48.8 cm in E35, ΔC_o +4.65). Without
  them the -ΔB share is 0.80 on val (10 scenes) and 0.85 on ETH3D (12).
- **Spread** (views with a predicted point within 2 cm of a GT point): the mean barely moves (val 2.61 -> 2.78,
  ETH3D 1.16 -> 1.20), but its distribution splits. GT points no view comes within 2 cm of rise from 37% to 48%
  (val) and from 59% to 66% (ETH3D), while points covered by 8 or more views rise too. Where the merged surface
  lies on the GT every view is there; where it lies off, none is.
- **By Déjà View confidence quartile:** ΔC_o sits in the lowest quarter (val +0.50 of +0.62, ETH3D +0.43 of
  +0.51); -ΔB is spread over all four (val +0.37 / +0.24 / +0.17 / +0.13). E35's H3 signal is therefore the
  smaller part, the low-confidence pixels whose own surface does move off the GT.
- **Figures:** `figures/e38_split.jpg` (ΔC_o and -ΔB per group; the views-within-2-cm distribution, val and
  ETH3D indoor); `figures/e38_coverage_ai_006_007.jpg` and `figures/e38_coverage_office.jpg` (ΔC_o and -ΔB per GT
  pixel, ±5 cm). Bedroom view 6: the back wall's own cloud moves onto its GT (ΔC_o -0.69 cm, blue) while the same
  wall loses more from the other views (-ΔB +1.16, red). Office: ΔC_o is near zero apart from view 10's ceiling
  beams, and -ΔB is red on every wall.
- **Conclusion:**
  - The expectation holds: on val and ETH3D the merge, not each view moving off the GT, carries 59-75% of the
    completeness loss (80-85% without Déjà View's two failure scenes). Déjà View's 20 layers scatter around the
    surface and the metric takes the nearest layer, so some layer sits close to most GT points. E_mv pulls the
    layers onto one consensus surface that is no closer to the GT than one view alone was (C_o unchanged on dev,
    +0.6 cm on val), and the nearest-layer advantage goes.
  - So the completeness loss is mostly the protocol rewarding a scattered cloud. Its 20 per-pixel clouds score
    a thick shell better than one surface that sits a few cm off. The refiner is not making each view's surface
    better either: E_mv reaches agreement, not the GT.
  - Not tested, next candidates: (a) a protocol that scores one surface (fuse the views, e.g. one point per
    voxel, or own-view completeness next to C_f). That changes the evaluation, so ask the team first. (b) What
    would move the consensus onto the GT rather than to the views' mean: E_mv weighted by confidence on both
    sides of a pair, or the 3D refiner Pouya proposes. (c) E35's other variant (E_mv off on the
    lowest-confidence quarter) aims at ΔC_o, the smaller part, and now matters less.

## E36: learned single-view depth refiner, with and without SAM 3 masks -> Chamfer 3.50 -> 3.20 cm on dev with masks (05.11.2026, rescored 09.11.2026)
- **Numbering:** E35 on Pouya's branch; E36 here, the number E35's plan reserved for it (E35 is the failure
  analysis, written the same week).
- **Goal:** issue #11, a BetterDepth/PatchRefiner-style learned refiner as an alternative to the energy; does
  conditioning on the masks help it?
- **Setting:** `learned-refine` (`src/refinement/learned.py`, `configs/learned.yaml`): residual U-Net (7.85M
  parameters) on log depth; input RGB, Déjà View K=16 log depth and log confidence, with masks also the
  boundaries and the wall/floor/ceiling area of the per-view SAM 3 masks (`sam3image-nyu40-t0.3`). Per view the
  depth is scaled onto the GT (median ratio), the net fits the remaining log error, L1 + gradient L1 on 4 scales
  (weight 0.5). 4000 steps, batch 8, 256 px crops, AdamW 1e-3 one-cycle, seed 0. One run, nothing tuned.
- **Data:** new `train` split in `configs/hypersim.yaml`: 40 of the 153 centred scenes of Hypersim's official
  train partition, evenly spaced in name order; 36 fit, last 4 held out (loss only). 20 views at 504 px each.
  On the workstation SAM 3 runs with `HF_HUB_OFFLINE=1` (cached weights, no token).
- **Commands:** `uv run learned-refine`, then `scene_eval --cfg` with `configs/eval_learned.yaml`,
  `configs/eval_learned_rgb.yaml`, `configs/eval_hypersim.yaml` (workstation, RTX 3080; runs
  `learned-20261105-112054`, `eval_learned-20261105-113133`, `eval_learned_rgb-20261105-113319`,
  `eval_hypersim-20261105-113507`, dirty tree before this commit). Training 5 min per model.
- **Held out (4 train scenes):** mean abs log error 3.68% -> 3.43% with masks, 3.50% without.
- **Old protocol (05.11, scaled ICP, F-score at 5 cm):** Deja View 0.844, learned with masks 0.845, without
  0.839, read as no real gain.
- **Rescored with the PR #19 protocol (09.11):** `scene_eval --splits dev` on `main` merged into this branch,
  the Mac (CPU), runs `eval_hypersim-20261109-113140`, `eval_learned-20261109-114835`,
  `eval_learned_rgb-20261109-120739`. Deja View's dev Chamfer is the same as PR #19's RTX 3080 run (3.50 cm).

| variant | accuracy cm | completeness cm | Chamfer cm | living room | bedroom | lobby |
|---|---|---|---|---|---|---|
| Deja View | 4.68 | 2.32 | 3.50 | 1.60 | 1.85 | 7.05 |
| learned, masks | 4.34 | 2.05 | **3.20** | 1.56 | 1.74 | 6.30 |
| learned, no masks | 4.43 | 2.16 | 3.29 | 1.57 | 1.85 | 6.45 |
| Deja View + joint | 4.21 | 3.69 | 3.95 | 1.60 | 1.96 | 8.30 |
| learned, masks + joint | 3.82 | 3.20 | 3.51 | 1.57 | 1.84 | 7.14 |
| learned, no masks + joint | 3.95 | 3.34 | 3.64 | 1.58 | 1.95 | 7.40 |

  (scene columns: Chamfer cm.) Label scores are the same for every variant (mIoU 0.36), as the protocol scores
  the SAM 3 labels per pixel.
- **Conclusion:** with masks the refiner improves accuracy, completeness and Chamfer on all three dev scenes,
  the first variant that improves completeness at all; most of the mean comes from the lobby (7.05 -> 6.30).
  Without masks it gains less (3.29), and on the bedroom nothing (1.85). The joint refiner on top loses
  completeness again, as it does on raw Deja View. One seed, three scenes: the val split and more seeds come
  before any claim, also on whether the masks help.

## Plan: E39 = faster panoptic labeling (SAM 3 + association), same labels (09.11.2026)
- **Numbering:** written as E35; renumbered E36 because main's E35 (learned refiner, PR #24) merged meanwhile, then
  E39 before the PR because main took E35-E38 meanwhile (failure analysis, PR #30; the learned refiner moved to E36).
  Commit messages up to 622aaad, the mask tags (`-e35oldA`, `-e35oldB`, `-e35b4`, `-e35new`) and the run names
  (`e35_oldA-*`, `e35_new-*`) keep the old number; every cited commit hash is unchanged. The scripts and result
  files were renamed `e35_*` -> `e39_*` with their content unchanged but for the number.
- **Goal (issue #29):** quality of life, not research: SAM 3 + association were estimated at ~80% of the
  pipeline's runtime, mostly Python on the CPU. Make them faster without changing what they output.
- **Setting:** Hypersim `dev` split (living room `ai_010_007`, bedroom `ai_006_007`, lobby `ai_044_001`) on the
  cached inputs of `configs/eval_hypersim.yaml` (20 views at 504x378, Déjà View K=16, DSINE, `nyu40` prompts at
  t0.3, working association xc0.7 vf0.25 kl0.5); this machine, one RTX 3080 (10 GB), 16 CPU threads. Unchanged:
  the evaluation, prompts, thresholds, views, resolution, association rule.
- **Step 1, profile before:** one dev scene (the bedroom, the shared visual scene) through every pipeline stage
  with warm models: Déjà View forward, SAM 3 per-view masks, association, DSINE, joint refiner. Weight loading is
  timed separately; each stage runs 3 times, the median is reported. SAM 3 is split further (image encoder, text
  encoder, detector + mask decoder, post-processing, painting) to see where its time goes.
- **Decision rule:** if SAM 3 + association are not the bulk of the runtime (< 50% of the stages above), log
  that, skip step 2, and report the profile only.
- **Step 2, change (as the issue proposes, nothing else):** encode each prompt once per scene (`text_embeds`),
  run all prompts of a view as one batch, resolve mask overlaps (higher score wins) on the GPU; the
  association's reprojection of all 380 view pairs in torch on the GPU. CPU tests show each new path matches
  the old one on random inputs before the old code is deleted.
- **Step 3, profile after:** as step 1.
- **Labels:** SAM 3 is run on the three dev scenes three times: twice with the current code (old A, old B) and
  once with the new code. Each set of masks is written under its own name (the eval's cache stays untouched) and
  scored by `scene_eval` on `dev` through a config copy that changes only `segmentation:`.
- **Noise floor:** bf16 kernels may be bit-identical between two runs on one GPU; then old A vs old B says
  nothing. The floor is therefore the larger of |old A - old B| and |old A - the cached masks| (same code, made
  on another GPU, the cross-GPU difference E34 already treats as noise), per metric on the dev aggregate.
- **Success:** mIoU, PQ^sc and AP on dev (and Chamfer of the refined variant, which uses the labels) of the new
  code differ from old A by no more than the noise floor, or are better. Otherwise the change is not merged as
  is: find which part changes the masks first.
- **Amendment before any new-code run (09.11.2026):** the planned noise floor is 0. The old code is deterministic
  on this GPU: old A, old B and the eval's cached masks are bit-identical on all three dev scenes (same ids, same
  scores; `e39_compare_masks.py`), so the cache was made here with the same code and is no second reference.
  Batching prompts changes the shapes of SAM 3's bf16 matmuls, so the GPU may pick other kernels that round
  differently, which is the same kind of difference as running on another GPU. The floor is therefore the
  dev difference between the same code's masks from two GPUs, already on record: README run
  `eval_hypersim-20261105-104445` (this RTX 3080) vs E34's addendum (RTX 5090, job 994): mIoU 0.360 vs 0.352,
  PQ^sc 0.292 vs 0.294, AP 0.216 vs 0.217, i.e. |Δ| 0.008 / 0.002 / 0.001. Success: new vs old A within these,
  or better; bit-identical masks pass trivially. The comparison of the masks themselves (identical pixels,
  matched score differences) is reported as well.

## E39: faster panoptic labeling -> SAM 3 + association 31.6 s -> 17.1 s per scene, same dev scores (09.11.2026)
- **Goal, setting:** as in the plan above (issue #29); RTX 3080, torch 2.14.1+cu130, transformers 5.20.2, SAM 3
  weights from the local Hugging Face cache (`HF_HUB_OFFLINE=1`: the gated repo needs no token then).
- **Commits:** before = 066d620 (old code), after = 79538a1 (cc79b76: SAM 3; 79538a1: association). The same
  scripts measure both: `experiments/scripts/e39_profile.py`, `e39_masks.py`, `e39_compare_masks.py`,
  `e39_association_check.py`, run at each commit; outputs in `results/e39_*.txt`.
- **Commands:** `HF_HUB_OFFLINE=1 uv run python experiments/scripts/e39_profile.py
  data/processed/hypersim/ai_006_007_cam_00 [--breakdown]` at each commit; `e39_masks.py <tag> <3 dev folders>`
  writes `sam3image-nyu40-t0.3-<tag>` masks (old A, old B at cf45970 = 066d620's code; new at 79538a1);
  `python -m src.analysis.scene_eval --splits dev --cfg <copy>` with `configs/eval_hypersim.yaml` changed only in
  `segmentation: sam3image-nyu40-t0.3-e35oldA` (at 066d620, run `e35_oldA-20261109-205226`) or `-e35new` (at
  79538a1, run `e35_new-20261109-205640`); both clean.
- **Profile before** (bedroom, warm models, median of 3; runs within 0.1 s of each other; weight loading 1.0 / 1.1 /
  0.7 s for Déjà View / SAM 3 / DSINE, not in the total): SAM 3 + association were 85% of the stages, so the change
  went ahead. Under cProfile with blocking launches, 42 of SAM 3's 47 s were its 740 single-prompt forward passes
  (20 views x 37 prompts), 9.6 s of them re-encoding the prompt; image features 3.7 s; post-processing and the
  CPU painting together < 1 s. The issue's guess that CPU painting dominates did not hold: the GPU work was
  launched in batches of one. Association: 1.9 s, nearly all the 380 numpy reprojections.
- **Change:** each prompt encoded once per scene (one at a time: a batched encoder rounds differently in bf16,
  one at a time it is bit-identical to before), the detector run on 4 prompts per pass, overlap resolution and id
  compaction on the GPU; the association projects each view into all 19 others in one batch of torch ops. Checks
  on the way: with the detector at batch 1 the new SAM 3 path reproduces the old masks bit for bit (bedroom);
  batch 1 / 4 / 8 / 16 / 37 took 23.6 / 16.4 / 15.1 / 14.9 / 18.9 s at 2.1 / 2.9 / 3.8 / 5.7 / 7.6 GB peak (with a
  batched text encoder), so batch 4 is the largest under `sam3-segment`'s 3.5 GB default cap.
- **Results, time per stage** (bedroom ai_006_007, RTX 3080, warm, median of 3):

  | stage | before (s) | after (s) |
  |---|---|---|
  | Déjà View | 1.10 | 1.11 |
  | SAM 3 (per-view masks) | 29.70 | **16.57** |
  | association | 1.94 | **0.57** |
  | DSINE | 1.11 | 1.09 |
  | joint refiner | 3.24 | 3.24 |
  | total | 37.09 | 22.57 |
  | SAM 3 + association | 31.64 (85%) | 17.14 (76%) |

  SAM 3 alone on the three dev scenes: 29.1-29.7 s before, 16.5-16.9 s after, peak 2.1 -> 2.9 GB.
- **Results, labels** (dev aggregate, labels identical for both variants as the protocol intends):

  | | mIoU | PQ^sc | AP | Chamfer Refined (cm) |
  |---|---|---|---|---|
  | before (old A) | 0.360 | 0.292 | 0.216 | 3.953 |
  | after | 0.360 | 0.292 | 0.216 | 3.954 |
  | noise floor (amendment) | 0.008 | 0.002 | 0.001 | - |

  Per scene the only change at 3 decimals is the lobby's PQ^sc 0.321 -> 0.322 (Refined Chamfer 8.295 -> 8.301 cm,
  completeness 7.955 -> 7.965). Masks: 467 / 575 / 1776 -> 466 / 578 / 1786; 99.98 / 99.97 / 99.85% of pixels keep
  the mask they shared most pixels with. The association alone (old vs new on the cached masks): the same
  18029 / 8252 / 54878 directed pairs, linked ids bit-identical on all three scenes, largest share difference
  1.5e-4 (lobby, one float64 rounding flip at a pixel edge); 2.90 / 1.97 / 2.63 s -> 0.62 / 0.56 / 0.62 s.
- **Noise:** old A, old B and the eval's cached masks are bit-identical on all three scenes (results
  `e39_compare_masks.txt`), so the floor is the recorded cross-GPU difference set in the amendment.
- **Conclusion:** passes: SAM 3 + association take 46% less time (31.6 -> 17.1 s per 20-view scene, whole
  pipeline 37.1 -> 22.6 s), the dev label scores are unchanged to 3 decimals, and the remaining difference
  is bf16 rounding from batching the detector, smaller than what the same code shows between two GPUs. SAM 3 is
  still 73% of the pipeline: under blocking launches 16.9 of its 21.6 s are the 200 detector passes (20 views x
  10 batches) and 3.8 s the image encoder; more would need larger batches (memory) or fewer prompts, both out of
  this issue's scope. The trial and run masks (`-e35oldA`, `-e35oldB`, `-e35b4`, `-e35new`) stay in the git-ignored
  scene folders; the eval's own cache is unchanged.

## Plan: E40 = remove or justify every hand-set constant in the energy (10.11.2026)
- **Goal (issue #28, follow-up to point 2 of #13):** every row of the issue's table gets a decision: **remove**
  the setting, **derive** it from the data (as Otsu's threshold did for E_mv's occlusion tolerance), or **keep** it
  with a one-sentence reason. On a tie the simpler energy wins: remove over derive over keep. Out of scope and
  left as they are: E_n's λ_n and normal estimator (#21), the Gauss-Newton / CG budget (#10).
- **Setting:** Hypersim dev (living room `ai_010_007`, bedroom `ai_006_007`, lobby `ai_044_001`) on the cached
  inputs of `configs/eval_hypersim.yaml` (20 views at 504x378, Déjà View K=16, DSINE, linked SAM 3 masks, nyu40
  at t0.3), the default `configs/refine.yaml`, this machine's RTX 3080. Unchanged: the evaluation, the inputs, the
  solver budget (10 Gauss-Newton steps x 20 CG iterations, tol 1e-3).
- **Method:** each setting is varied alone from the current default (one change per variant), all variants in one
  `scene_eval` run so they share a GPU and inputs (`experiments/configs/e40_sweep.yaml`). Then the chosen
  decisions are combined and the combination is compared once more with the unchanged refiner on dev; if it falls
  outside the noise floor although every part passed alone, step back one decision at a time in reverse table
  order. Decided on dev only.

  | row | setting now | variants (each alone) | remove / derive candidate |
  |---|---|---|---|
  | E_d Huber threshold | 0.05 | quadratic (no Huber); k = the scene's Otsu occlusion threshold; 0.02, 0.1 to read the slope | remove: quadratic; derive: Otsu's tol, the scene's same-surface noise level in log depth |
  | E_d weight | c^p u, p = 1 | u removed (w = c) | confidence: **E37's result reused**, not rerun; u: kept on a tie (agreed 10.11, see below) |
  | E_mv neighbors | 4 | 2, 8, all 19 | remove: match every overlapping pair |
  | E_mv border margin | 1 px | 0 px (the bilinear support, u in [0, W-1]) | remove the extra pixel |
  | E_n occlusion cut | 0.02 | 0.01, 0.05; the scene's Otsu occlusion threshold | derive: the same Otsu threshold E_mv uses (both are log-depth departures) |
  | E_n mid-depth window | 7 px | 3, 5, 11 px | none found before running; keep or not by the sweep |
  | E_n contour closing budget | = window | 0 (no closing), 2 x window | remove: no closing, which deletes `close_cuts` |
  | λ_δ | 1 | 0.3, 3 | none: λ_δ = 0 is singular on stress (module docstring) |
  | λ_d, λ_mv | 1, 1 | λ_mv 0.3, 3 | λ_d: remove as a parameter, E's overall scale does not change its minimizer, so one weight is fixed to 1 by definition |

  A value from a grid (0.3, 3, 0.02, ...) is adopted only if it beats the default by more than the noise floor
  on dev; it then stays as "keep, chosen on the dev grid". The grid values themselves are not new constants.
- **E37 reused:** E_d without the confidence (p = 0) against p = 1 on dev (RTX 5090): accuracy 4.18 / 4.18,
  completeness 3.66 / 3.63, Chamfer 3.92 / 3.91 cm, completeness better on 2 of 3 dev scenes. A tie or better,
  so by this issue's rule the confidence leaves E_d and `conf_power` goes; it is not rerun alone here, only as
  part of the final combination.
- **Consistency u (decided with Peter before running):** u is one of the energy's semantic parts; removing it is
  adopted only if dev is better beyond noise. On a tie it stays: whether semantics help is its own ablation
  (Claude.md step 5), not a constant to tidy away.
- **Check:** the final configuration on val (11 scenes) once, with the unchanged refiner in the same run. Val's
  inputs are prepared on this machine first (`hypersim-download`, `hypersim-prepare`, `dejaview-predict`,
  `sam3-segment --prompts configs/prompts/nyu40.txt --threshold 0.3`, `dsine-predict`, E34's settings). No test
  scene.
- **Noise floor:** the unchanged refiner twice (two variants in one run, and a second process). If that gives 0
  (deterministic on one GPU), the floor is the dev difference of the unchanged refiner between two GPUs: this
  run's numbers against E34's addendum (RTX 5090, job 994: accuracy 4.178, completeness 3.657, Chamfer 3.917 cm),
  per metric on the dev aggregate. Fixed here, before any variant runs.
- **Success:** a change passes if accuracy, completeness and Chamfer on the dev aggregate each stay within the
  noise floor of the unchanged refiner, or get better. Per-scene numbers are reported, not used to decide.
- **Report per change:** accuracy, completeness and Chamfer on dev, per scene and aggregate; the bedroom
  (`ai_006_007`, views 18, 0, 12, 6) refined depth of the final configuration against the unchanged refiner.
  mIoU, PQ^sc and AP once at the end (labels do not move with refinement).
- **Cost:** ~25 variants x 3 dev scenes at ~4 s each, plus scoring, ~10 min; val prep ~30 min, val run ~5 min.
- **Amendment before any score (10.11.2026):** the first sweep run stopped with CUDA out of memory and wrote no
  scores. A probe (refiner alone, peak allocated memory) found the cause: matching all 19 views does not fit on the
  RTX 3080's 10 GB in the living room (neighbors 4 / 8: 16.1M / 26.1M matches, 4.8 / 7.2 GB; 19: out of memory;
  bedroom 19: 14.0M matches, 4.3 GB; lobby 19: 26.2M, 7.2 GB). "neighbors all" is dropped from the sweep, so the
  remove option of that row cannot be tested on this machine; 2 and 8 stay.

## E40: remove or justify every hand-set constant in the energy -> two removed, the rest kept: dev cannot tell three candidates apart, val can (10.11.2026)
- **Goal, setting, method, noise floor and success rule:** as in the plan above (issue #28); RTX 3080, torch
  2.14.1+cu130, Hypersim dev for every decision, val as the check, no test scene. Out of scope, unchanged: λ_n and
  the normal estimator (#21), the Gauss-Newton / CG budget (#10).
- **Commits:** d8675a5 (sweep knobs), ce36225 (plan), 0de7b92 (amendment), caea80c (`e40_table.py`), 954bfc1 (first
  final energy), a97064a (confidence kept), 679291c (final: back to the refiner before E40 but for λ_d and
  conf_power), this commit (`e40_check.py`, `e40_figures.py`, `e40_attribution.yaml`, results, figure).
- **Commands** (this machine; `runs/` stays local):
  - `uv run python -m src.analysis.scene_eval --cfg experiments/configs/e40_sweep.yaml --splits dev` at caea80c's
    code (run `e40_sweep-20261110-081852`; it records git 50d6117, the same tree before a lint fix of
    `e40_table.py` was folded in) -> `e40_table.py <run> --split dev --floor 0.033 0.037 0.036 --labels` ->
    `results/e40_sweep.txt` (every variant per scene and aggregate).
  - Val inputs, prepared here with E34's settings: `hypersim-download --splits val` and `hypersim-prepare` (~25
    min, network-bound), `dejaview-predict` (1:09 for all 11), `sam3-segment --prompts configs/prompts/nyu40.txt
    --threshold 0.3` (3:26), `dsine-predict` (0:42), `HF_HUB_OFFLINE=1`.
  - `scene_eval --cfg configs/eval_hypersim.yaml --splits dev val` at caea80c (refiner before E40, run
    `eval_hypersim-20261110-083124`, a git worktree), 954bfc1 (`-083337`), a97064a (`-083934`) and 679291c
    (`-084435`) -> `e40_check.py <caea80c run> <the three others>` -> `results/e40_check.txt`.
  - `scene_eval --cfg experiments/configs/e40_attribution.yaml --splits dev` at caea80c (`e40_attribution-20261110-083642`)
    -> `results/e40_attribution.txt`.
  - `uv run python experiments/scripts/e40_figures.py runs/e40_sweep-20261110-081852 runs/eval_hypersim-20261110-083337
    runs/eval_hypersim-20261110-083934 runs/eval_hypersim-20261110-084435` -> `results/e40_figures.txt`,
    `figures/e40_bedroom.jpg`.
- **Noise floor:** the unchanged refiner twice in one process, in a second process and in a third: dev aggregates
  equal to 0.00002 cm, per val scene to 0.0006 cm. The depth maps themselves are not bit-identical between
  repeats (3-14% of the pixels differ in the four scenes checked, a few by up to 10%: `index_add`'s atomic sums reorder on the GPU), but the
  scores agree to the third decimal, so, as planned, the floor is the cross-GPU difference against E34's
  addendum (RTX 5090): **0.033 / 0.037 / 0.036 cm** (accuracy / completeness / Chamfer, dev aggregate).
- **Sweep, dev aggregate, cm** (refiner before E40: 4.211 / 3.694 / 3.953; Δ = variant - that; per scene in
  `results/e40_sweep.txt`). "neighbors all" (19) did not fit on 10 GB (amendment above).

  | row | variant | Δ accuracy | Δ completeness | Δ Chamfer | within floor |
  |---|---|---|---|---|---|
  | E_d Huber | quadratic (none) | +0.000 | +0.000 | +0.000 | yes |
  | | Otsu's threshold | -0.000 | -0.000 | -0.000 | yes |
  | | 0.02 / 0.1 | -0.001 / -0.000 | +0.000 / -0.000 | -0.000 / -0.000 | yes |
  | E_d weight | u removed (w = c) | +0.001 | -0.011 | -0.005 | yes |
  | E_mv neighbors | 2 | +0.097 | -0.260 | -0.081 | no |
  | | 8 | -0.014 | +0.228 | +0.107 | no |
  | E_mv border | 0 px | +0.000 | -0.004 | -0.002 | yes |
  | E_n cut | 0.01 / 0.05 | -0.009 / +0.030 | -0.017 / +0.026 | -0.013 / +0.028 | yes / yes |
  | | Otsu's threshold | +0.005 | +0.005 | +0.005 | yes |
  | E_n window | 3 / 5 / 11 px | +0.006 / -0.002 / +0.003 | -0.005 / -0.001 / +0.005 | +0.001 / -0.002 / +0.004 | yes |
  | E_n closing budget | 0 (no closing) | +0.050 | +0.030 | +0.040 | no |
  | | 2 x window | -0.006 | -0.007 | -0.007 | yes |
  | λ_δ | 0.3 / 3 | -0.043 / +0.062 | +0.184 / -0.279 | +0.070 / -0.108 | no / no |
  | λ_mv | 0.3 / 3 | +0.058 / -0.021 | -0.323 / +0.245 | -0.132 / +0.112 | no / no |

  Labels do not move (mIoU / PQ^sc / AP 0.360 / 0.292 / 0.216 for every variant). Huber's threshold binds on only
  0.3 / 0.9 / 0.3% of the pixels (living room / bedroom / lobby, |δ| > 0.05 after refinement), which is why
  removing it changes nothing on dev.
- **Val check 1** (Huber removed, border 0 px, E_n cut at Otsu's threshold, confidence removed per E37, λ_d
  removed; 954bfc1): dev +0.018 / -0.027 / -0.005 (passes); val **+0.074** / -0.008 / +0.033: accuracy worse on
  11 of 11 val scenes, beyond the floor. ai_009_009 (Déjà View's failure hall) +0.469 cm; without it +0.034.
- **Attribution, on dev only** (the combination with one change undone at a time, `results/e40_attribution.txt`):
  the knob version of the combination reproduces 954bfc1's dev scores exactly; undoing the confidence removal
  recovers the most accuracy (-0.013 cm; Huber -0.010, but only once the confidence is gone; E_n cut -0.004;
  border 0). The confidence removal was the one change taken from E37 rather than tested alone here, so it was
  stepped back (a97064a).
- **Val check 2** (the same, confidence kept; a97064a): dev +0.005 / -0.000 / +0.002; val **+0.051 / +0.042 /
  +0.046**, all beyond the floor, while the median scene moves by +0.001 / +0.001 / +0.002 and the mean without
  ai_009_009 is +0.007 / -0.011 / -0.002. ai_009_009 alone: +0.486 / +0.575 / +0.530 cm, against a run-to-run
  spread of 0.0001 cm there, so it is an effect, not noise. Otsu's threshold is 0.024-0.037 on dev but
  0.023-0.067 on val (ai_009_009: 0.048), so the derived cut runs where dev never tested (dev's sweep: 0.05
  already +0.03 cm), and Huber's linear branch acts where corrections are large, as in a failed scene.
- **Second val run, stated plainly:** the plan said one val run. Val was run twice to check a configuration, plus
  once at 679291c as an identity check; no value was chosen among alternatives on val. Which of the three changes
  carries ai_009_009's loss is not known: finding out would mean choosing on val.
- **Decisions:**

  | row | decision | reason | dev Δ (acc / comp / Chamfer, cm) |
  |---|---|---|---|
  | λ_d | **remove** | E's scale does not move its minimizer (test: scaling every weight by 7 scales E by 7, same x) | 0 (identical) |
  | E_d weight: confidence exponent (`conf_power`) | **remove**, w = c u | the exponent stays 1, so the knob is dead code; dropping c itself tied on dev (E37) but carried the accuracy loss of val check 1 | 0 |
  | E_d weight: consistency u | **keep** | ties without it (-0.005 Chamfer); semantics stay unless better beyond noise (agreed before running) | +0.001 / -0.011 / -0.005 without |
  | E_d Huber 0.05 | **keep** | removing it ties on dev, but the combination with it lost 0.5 cm in val's failure scene, where corrections are large | 0.000 without |
  | E_mv neighbors 4 | **keep** | 2 and 8 trade accuracy against completeness beyond noise; all 19 exceed 10 GB | see table |
  | E_mv border 1 px | **keep** (`MATCH_BORDER`) | 0 px ties on dev, but failed val in combination; the margin keeps the four bilinear corners in the image when a step moves the projection | -0.002 Chamfer at 0 px |
  | E_n cut 0.02 | **keep** | Otsu's threshold ties on dev, but spans 0.023-0.067 on val, beyond what dev tested, and failed val in combination | +0.005 at Otsu |
  | E_n window 7 px | **keep** (`CUT_WINDOW`) | spans Déjà View's 2-4 px occlusion ramps on both sides (fc6c3c4); 3-11 px tie on dev | within ±0.006 |
  | E_n closing budget = window | **keep** | no closing loses accuracy beyond noise; twice the window ties | +0.050 without |
  | λ_δ 1 | **keep** | 0 is singular; 0.3 / 3 trade accuracy against completeness beyond noise | see table |
  | λ_mv 1 | **keep** | 0.3 / 3 trade as λ_δ does | see table |
  | λ_n, normal estimator | out of scope | issue #21 | - |
  | Gauss-Newton x CG, CG tol | out of scope | issue #10 | - |

- **Final refiner** (679291c): the one before E40 within the run-to-run spread (dev aggregate 4.211 / 3.694 /
  3.953 cm, val 5.755 / 5.690 / 5.723, every scene within 0.0006 cm), so the README's numbers stand. Labels
  rechecked once: dev mIoU / PQ^sc / AP 0.360 / 0.292 / 0.216, val 0.397 / 0.258 / 0.345 (val differs from E34's
  0.395 / 0.256 / 0.353, most likely because SAM 3 ran here with E39's batched code, not on the RTX 5090;
  not checked further).
- **Figure:** `figures/e40_bedroom.jpg` (in `experiments/`), bedroom views 18, 0, 12, 6: each pixel's distance to
  the nearest GT point before E40, and its change under each change that passed dev, both val-checked
  combinations and the final energy, on one ±2 cm scale. Every change is local to object edges (bed-frame bars,
  chair backs, the vanity); no surface moves as a whole. Combination 1 adds red along the thin bed bars of view 18
  (mean +0.006 cm over the bedroom's 20 views); the final column is blank (fewer than 0.01% of
  the pixels move by more than 0.5 cm).
- **Conclusion:**
  - Two settings went: λ_d (a gauge, by definition) and the confidence exponent (dead at 1). Every other
    constant stays, now with its reason where it is set and listed in the README.
  - Three candidates (no Huber, no border margin, the E_n cut derived from Otsu) are each indistinguishable from
    the current value on dev, yet together they lose 0.5 cm in one val scene. Three dev scenes cannot separate
    settings that only act on failures and edge cases; the "prefer the simpler energy on a tie" rule is only as
    good as the scenes the tie is measured on.
  - Every weight in the sweep (λ_mv, λ_δ, neighbors) trades accuracy against completeness, the trade E38 explained
    (the merge of the views' spread). None improves both, so no weight search over them can pass this issue's rule;
    a Chamfer-only rule would pick λ_mv 0.3 (-0.132 cm) or λ_δ 3 (-0.108 cm) on dev. That is a choice of metric,
    left open.
  - Not tested, next candidates: the three candidates one at a time on a split not yet used (train scenes, or
    test with the team's consent); whether the derived cut works if Otsu's threshold is computed per view pair
    instead of per scene.

## E41: after E40's review: Gauss-Newton 5 x 10, u counting the own view, λ_δ = 0 -> two adopted, λ_δ stays (10.11.2026)
- **Why:** Peter's review of E40. (1) The solver budget 5 x 10 had been chosen in issue #10 (week 2 slides:
  4.2 -> 1.9 s per dev scene, Chamfer 3.95 -> 3.93 cm) but `configs/refine.yaml` still ran 10 x 20. (2) u should
  count the pixel's own view: the vote of all views that see the point, which also keeps E_d's weight above 0
  wherever the confidence is, so (3) λ_δ might no longer be needed.
- **No plan entry first:** (2) and (3) were run as a trial while answering the review, with a switch in the working
  tree (runs `-dirty` at f16c915, E40's report, with the budget set per variant); the switch was then removed
  and the adopted changes committed (4722b2a, 150db44). Decided on dev, stress as a second look; val once at the end.
- **Commands** (RTX 3080): `scene_eval --splits dev stress` with two trial configs (5 x 10; variants: 10 x 20,
  5 x 10, λ_δ = 0, own view, own view + λ_δ = 0; then converged budgets 5 x 100 and 20 x 100), runs
  `u_own-20261110-090929`, `u_own_converged-20261110-091217` -> `e40_table.py` -> `results/e41_trials.txt`;
  at 150db44 (clean) `scene_eval --splits dev val stress` (`eval_hypersim-20261110-093140`) -> `e40_check.py` against
  E40's final refiner (`eval_hypersim-20261110-083124`) -> `results/e41_check.txt`; `scene_eval --splits dev stress`
  (`eval_hypersim-20261110-093416`) and `scene_figures` -> `figures/eval_hypersim-20261110-093416/` (replaces
  `eval_hypersim-20261105-104445`; `scene_figures` cannot draw val's 11 scenes, its palette has 4 colours).
- **Trial, dev aggregate, cm** (accuracy / completeness / Chamfer; per scene and stress in `results/e41_trials.txt`):

  | variant | dev | Gauss-Newton steps taken (of 5) |
  |---|---|---|
  | 10 x 20 (before) | 4.211 / 3.694 / 3.953 | 10 of 10 |
  | 5 x 10 | 4.198 / 3.659 / 3.929 | 5, all scenes |
  | 5 x 10, u with own view | 4.199 / 3.658 / 3.928 | 5, all scenes |
  | 5 x 10, λ_δ = 0 | 4.395 / 2.653 / 3.524 | 0-4 (conference room 0: no step accepted) |
  | 5 x 10, own view, λ_δ = 0 | 4.154 / 3.579 / 3.866 | 3-5 (lobby 3, bedroom 4, conference room 4) |
  | 5 x 100, own view, λ_δ = 0 | 4.181 / 4.114 / 4.148 | 4-5 |
  | 20 x 100, own view, λ_δ = 0 | 4.181 / 4.116 / 4.149 | 3-20 |
  | 20 x 100 (with λ_δ) | 4.216 / 3.701 / 3.958 | 9-20 |

  u with the own view moves no dev or stress score by more than 0.007 cm. With λ_δ = 0 the normal equations are no
  longer singular (Déjà View's confidence is at least 0.015 of its view's mean on all 17 prepared scenes, so
  E_d's weight is at least ~4e-4), but they are ill-conditioned: at 5 x 10 the line search finds no decrease after
  3-4 steps on 4 of 6 scenes. The 5 x 10 "gain" is that early stop (as week 2's "stopped early: 1 x 5" point);
  converged (5 x 100 and 20 x 100 agree to 0.002 cm), completeness is 0.46 cm and Chamfer 0.22 cm worse than
  with λ_δ. Removing λ_δ leaves E_d alone as the anchor, about a third of the stiffness (E37), so the views merge
  more (E38).
- **Adopted** (4722b2a, 150db44): 5 x 10 and u counting the own view, u = (agreeing others + 1) / (views seeing the
  point + 1). λ_δ stays, with this reason in the docstring.
- **Check, refiner before E41 -> after** (`results/e41_check.txt`):

  | split | accuracy | completeness | Chamfer | scenes worse (acc / comp / Chamfer) |
  |---|---|---|---|---|
  | dev (3) | 4.211 -> 4.199 | 3.694 -> 3.658 | 3.953 -> 3.928 | 1 / 0 / 0 |
  | val (11) | 5.755 -> 5.701 | 5.690 -> 5.585 | 5.723 -> 5.643 | 5 / 3 / 3 |

  Most of val's change is ai_009_009 (-0.79 / -0.84 / -0.81 cm), the scene that carried E40's val loss; without
  it +0.019 / -0.032 / -0.006, within the noise floor. Labels unchanged (dev 0.360 / 0.292 / 0.216, val
  0.397 / 0.258 / 0.345). Against Déjà View nothing changes in kind: dev Chamfer 3.50 -> 3.93, val 5.10 -> 5.64 (worse
  on 10 of 11 val scenes). The bedroom figures (`figures/eval_hypersim-20261110-093416/presentation/`) look as before.
- **Conclusion:** the agreed budget is now the default, at half the time and within noise of the old one (better on
  every dev metric). u with the own view is the cleaner definition at no cost. λ_δ is not replaceable by the floor
  u now has: the energy it leaves is ill-conditioned, and converged it merges the views more.

## Plan: E42 = E_n as the angle between DSINE's normals and the normals of the refined depth (10.11.2026)
- **Goal (Peter's proposal):** replace BiNI's normal term by the angle between DSINE's normal and the normal
  reconstructed from the current depth, the latter from each axis' closer neighbor (left or right, up or down,
  whichever 3D point is nearer; `normals.closer_neighbor_normals`). At a clean occlusion step the closer neighbor
  lies on the pixel's own surface, so the term needs no cut: if it ties, it removes E_n's occlusion cut (0.02), the
  cut's 7 px window and its closing budget.
- **Looked at first** (`scripts/e42_normals_figure.py` -> `figures/e42_normals_ai_006_007.jpg`,
  `results/e42_normals_figure.txt`; bedroom views 18, 0, 12, 6, Déjà View's input depth): the two normal maps
  agree in colour on every surface, so frame and orientation are right. The angle is 11.5° median over the 20
  views (view 0: 17.9°), 18% of the pixels above 30°. Walls show a fine noise texture (per-pixel differences of
  Déjà View's depth); every object outline is a band above 60°: Déjà View draws occlusion edges as 2-4 px ramps,
  and on a ramp both neighbors lie on the ramp, so the closer one does not escape it. Thin structures (bed frame,
  chair legs) are above 60° throughout.
- **Energy:** E_a = λ_a mean_p (1 - cos θ_p), written as the residual √(λ_a / 2) (n_rec(d)_p - n_DSINE,p) (3 per
  pixel; its square is λ_a (1 - cos θ) ≈ λ_a θ² / 2, smooth at θ = 0 where θ itself is not). The neighbor choice and
  the facing flip are fixed at each linearization. No instance gating, no contour cut: the closer neighbor is the
  edge handling. J is applied matrix-free by autograd (`torch.func.linearize` / `vjp`); the Jacobi diagonal of
  J^T J exactly, by probing with the 5 colours of (y + 2x) mod 5, which give the 5-point stencil of every pixel
  5 different colours.
- **Weight, fixed before running:** BiNI's residuals are log-depth steps times the focal length, about the slope
  error in radians, so a slope error θ costs about 2 λ_n θ² per pixel (two edges, two residuals each, weight 1/2);
  E_a charges λ_a θ² / 2. Equal stiffness: λ_a = 4 λ_n = 4e-4. Coarse grid around it: 4e-5, 4e-4, 4e-3. The run logs
  both terms' initial energy to check the match.
- **Setting:** as E41 (dev for decisions, stress to look, RTX 3080, 5 x 10, u with the own view). Variants in one
  run: Déjà View; Refined (BiNI E_n); E_n off (λ_n = 0); E_a at the three weights instead of E_n.
- **Success:** an E_a variant passes if accuracy, completeness and Chamfer on dev each stay within E40's noise
  floor (0.033 / 0.037 / 0.036 cm) of Refined, or get better; then it replaces BiNI (the simpler term on a tie:
  three constants go) and is checked once on val. If E_n off also ties, neither normal term earns its place and the
  entry says so.
- **Expected, written before running:** E_a smooths the walls' texture as E_n does, but pulls ramp pixels toward
  DSINE's facing normal, which either sharpens the ramp or drags the far surface forward; no guess at the sign.
  The run time grows by the autograd products (one jvp and one vjp per CG iteration, 5 per Gauss-Newton step for
  the diagonal).

## E42: E_n as an angle to DSINE's normals -> better Chamfer, worse accuracy on dev and val; BiNI stays (10.11.2026)
- **Goal, energy, weights, rule:** as in the plan above; run as planned.
- **Commits:** 0c9d35d (plan, `closer_neighbor_normals`, the look-first figure), 55e91bc (`angle_residuals`,
  `stencil_diagonal`, `RefineConfig.lambda_a`, tests), the two config commits, this commit (results, figure).
- **Commands** (RTX 3080, clean tree): `scene_eval --cfg experiments/configs/e42_angle.yaml --splits dev stress`
  (`e42_angle-20261110-094442`) -> `e40_table.py` -> `results/e42_dev.txt`, `results/e42_stress.txt`;
  `e42_figures.py runs/e42_angle-20261110-094442` -> `results/e42_figures.txt`, `figures/e42_bedroom.jpg`;
  `scene_eval --cfg experiments/configs/e42_val.yaml --splits val` (`e42_val-20261110-094828`) -> `results/e42_val.txt`.
- **Dev aggregate, cm** (Δ against BiNI's E_n, the refiner of E41; floor 0.033 / 0.037 / 0.036):

  | variant | accuracy | completeness | Chamfer | Δ | passes | s per scene |
  |---|---|---|---|---|---|---|
  | BiNI E_n (λ_n 1e-4) | 4.199 | 3.658 | 3.928 | - | - | 1.9 |
  | no normal term | 4.266 | 3.451 | 3.858 | +0.067 / -0.207 / -0.070 | no | 1.9 |
  | E_a, λ_a 4e-5 | 4.243 | 3.497 | 3.870 | +0.044 / -0.161 / -0.059 | no | 3.7 |
  | E_a, λ_a 4e-4 | 4.208 | 3.429 | 3.819 | +0.010 / -0.229 / -0.110 | yes | 3.4 |
  | E_a, λ_a 4e-3 | 4.271 | 2.863 | 3.567 | +0.073 / -0.795 / -0.361 | no | 3.2 |

  Per scene, λ_a 4e-4 is worse in accuracy in the living room (+0.017) and the bedroom (+0.061) and better in the
  lobby (-0.049); stress, all three scenes: accuracy +0.05 to +0.16, completeness -0.12 to -0.64, Chamfer -0.04 to
  -0.24. Every solve took all 5 Gauss-Newton steps (no early stop, unlike E41's λ_δ = 0). Labels unchanged.
- **The weight did not match as planned:** at x = 0, E_a at 4e-4 is 1.5-3 % of BiNI's E_n (e.g. bedroom 4.4e-5 vs
  4.6e-3), not equal. BiNI's start energy is dominated by large residuals where an edge ramp is not cut: its cost
  grows without bound with the step, while E_a's is bounded (|n_rec - n_DSINE| <= 2), so E_a acts like a robust,
  bounded normal prior. The small-angle argument of the plan holds only where both are small.
- **Val, once** (λ_a 4e-4 against BiNI, `results/e42_val.txt`): accuracy 5.701 -> 5.791 (+0.090, worse on 10 of
  11), completeness 5.585 -> 5.120 (-0.465, better on 11 of 11), Chamfer 5.643 -> 5.456 (-0.187, better on 9 of 11);
  without ai_009_009 +0.112 / -0.235 / -0.061. Fails the rule on accuracy.
- **Figure:** `figures/e42_bedroom.jpg` (in `experiments/`), bedroom views 18, 0, 12, 6: the angle between DSINE's
  normals and the closer-neighbor normals of Déjà View's, BiNI's and E_a's depth (0-60°), and each pixel's distance
  to the GT, E_a minus BiNI (±2 cm). Over the 20 views the angle's median is 11.5° (Déjà View), 6.0° (BiNI), 3.4°
  (E_a); that is the quantity E_a minimizes, so it says nothing about the geometry. Against BiNI, E_a moves points
  further from the GT on the walls of views 18 and 6 (+0.5 to +1 cm) and on the bed frame, closer on some furniture
  edges in view 12; on view 0's right wall, where Déjà View's surface and DSINE disagree by about 60° about its
  orientation, the change is a striped ±1 cm pattern (not investigated). Mean over the bedroom +0.047 cm.
- **Figure, normals after refinement (asked for after the report):** `figures/e42_normals_after_ai_006_007.jpg`
  (`scripts/e42_normals_after.py runs/e42_angle-20261110-094442`), bedroom views 18, 0, 12, 6: DSINE's normals and
  the closer-neighbor normals of Déjà View's, BiNI-refined and E_a-refined depth, one colouring. Déjà View's are
  noisy everywhere and multicoloured along every outline. BiNI smooths the noise but leaves low-frequency blotches
  on the walls (view 18's back wall, view 0's left wall) and keeps the multicoloured outlines: its edges are cut
  there, so the ramp pixels are not shaped. E_a's are the closest to DSINE: walls and floor nearly uniform, outlines
  one or two pixels wide, the bed frame's bars readable in views 18 and 12. What it leaves is isolated speckle
  (single pixels whose closer neighbor flips), densest on view 0's right wall and the ceiling corner above it,
  where the distance map shows the striped ±1 cm change. Smoother normals did not mean closer points: the same
  depth is +0.047 cm further from the GT in the bedroom than BiNI's.
- **Conclusion:**
  - The angle term is the same trade E40 found for λ_mv, λ_δ and the neighbor count: completeness and Chamfer
    better, accuracy worse; here on both dev (inside the floor) and val (beyond it). Under this issue's rule
    (all three metrics) BiNI stays; under a Chamfer-only rule E_a at 4e-4 would win on dev and val (-0.110 /
    -0.187 cm), and dropping the normal term entirely would already give most of the dev gain (-0.070). Which
    metric decides is the open question E40 raised.
  - If adopted, it would remove E_n's occlusion cut, the cut window and the closing budget, and cost 1.5 s more
    per scene (autograd products).
  - The trial term stays in the code, off (λ_a = 0), until that choice is made; if BiNI stays for good it goes.

## Plan: E43 = the normal term as tangent residuals (10.11.2026)
- **Goal:** a simpler form of E42's angle term, from the review of E42 (Peter): instead of reconstructing the
  depth's normal and comparing it with DSINE's, ask that each pixel's two chosen tangents lie in DSINE's tangent
  plane. Per pixel, r_x = n_DSINE . t_x / |t_x| and r_y = n_DSINE . t_y / |t_y|, t_x and t_y the tangents to the
  closer neighbor along each axis, as in E42. Both vanish exactly when the reconstructed normal is parallel to
  DSINE's. This drops the cross product, the normal's normalization and the facing flip, and it does not depend on
  DSINE's sign convention. Each residual is the sine of the angle between a tangent and DSINE's plane, so it is
  bounded by 1, a robust term as E42's was. `closer_neighbor_normals` stays for figures.
- **Scope:** a trial next to BiNI's E_n, which stays the default (issue #21 owns E_n's choice). E42's angle residual
  is replaced by this one in the code; E42 stays reproducible at 83571a1.
- **Energy and weight:** E_t = λ_t mean_p (r_x² + r_y²) / 2. For a normal tilted by a small θ and orthogonal
  tangents, r_x² + r_y² ≈ θ², so E_t ≈ λ_t θ² / 2, the small-angle stiffness of E42's λ_a (1 - cos θ): the same grid
  4e-5, 4e-4, 4e-3 compares directly with E42's. Jacobian by autograd, exact Jacobi diagonal by the 5-colour
  probing of E42 (the residuals read the same 5-pixel stencil).
- **Setting, rule:** as E42: dev decides, stress is looked at, RTX 3080, 5 x 10, u with the own view; variants
  Déjà View, Refined (BiNI), E_t at the three weights. Success as before against BiNI (all three metrics within
  0.033 / 0.037 / 0.036 cm or better); a weight that passes is checked once on val. E42's angle term at the same
  weights is the second reference (its dev numbers, same GPU and inputs).
- **Expected, written before running:** close to E42 at each weight, since both are bounded and equal for small
  angles; they differ for large ones (a tangent residual saturates at 90° of its own tangent, the angle term's at
  180°), i.e. at the ramps, where E42's speckle sat. No guess at which is better there.
- **Figures:** bedroom views 18, 0, 12, 6: the closer-neighbor normals of BiNI's, E42's and E_t's depth next to
  DSINE's; each pixel's distance to the GT, E_t minus BiNI.

## E43: the normal term as tangent residuals -> same as E42's angle term, simpler; the same trade, BiNI stays (10.11.2026)
- **Goal, energy, weights, rule:** as in the plan above; run as planned.
- **Commits:** 9802e84 (plan), 33d3b17 (tangent residuals, `closer_tangents`, tests), 1bf6af2 and 3f8c120 (configs),
  this commit (results, figure).
- **Commands** (RTX 3080, clean tree): `scene_eval --cfg experiments/configs/e43_tangent.yaml --splits dev stress`
  (`e43_tangent-20261110-100804`) -> `e40_table.py` -> `results/e43_{dev,stress}.txt`; `scene_eval --cfg
  experiments/configs/e43_val.yaml --splits val` (`e43_val-20261110-101022`) -> `results/e43_val.txt`;
  `e43_figures.py runs/e42_angle-20261110-094442 runs/e43_tangent-20261110-100804` -> `results/e43_figures.txt`,
  `figures/e43_bedroom.jpg`.
- **Dev aggregate, cm** (Δ against BiNI, the refiner of E41: 4.199 / 3.658 / 3.928; E42's angle term at the same
  weight for comparison):

  | weight | tangent E_t: Δ acc / comp / Chamfer | passes | angle E_a (E42) |
  |---|---|---|---|
  | 4e-5 | +0.047 / -0.173 / -0.063 | no | +0.044 / -0.161 / -0.059 |
  | 4e-4 | +0.025 / -0.249 / -0.112 | yes | +0.010 / -0.229 / -0.110 |
  | 4e-3 | +0.088 / -0.746 / -0.329 | no | +0.073 / -0.795 / -0.361 |

  Every solve took all 5 steps; 3.3-3.5 s per scene (BiNI 1.9). E_t's start energy at 4e-4 is 1.8-6.9e-5, as
  E42's (the small-angle match holds between the two bounded terms).
- **Val, once** (λ_t 4e-4 against BiNI): accuracy 5.701 -> 5.803 (+0.102, worse on 10 of 11), completeness 5.585 ->
  5.079 (-0.506, better on 11 of 11), Chamfer 5.643 -> 5.441 (-0.202, better on 9 of 11); without ai_009_009
  +0.117 / -0.260 / -0.072. Fails on accuracy, as E42 did (+0.090 / -0.465 / -0.187).
- **Figure:** `figures/e43_bedroom.jpg` (in `experiments/`), bedroom views 18, 0, 12, 6: DSINE's normals and the
  closer-neighbor normals of BiNI's, E42's and E43's depth; distance to the GT, E_t minus BiNI. The angle and
  tangent columns look alike (angle to DSINE over the 20 views: median 3.4° both, > 30° on 5.9 % vs 6.7 % of the
  pixels; BiNI 6.0°, 12.8 %). The one visible difference is view 0's right wall, where Déjà View and DSINE
  disagree about the wall's orientation: E43 leaves vertical streaks where E42 left speckle. Distance to the GT
  +0.050 cm against BiNI over the bedroom (E42: +0.047).
- **Conclusion:** the tangent form is the simpler implementation of the same term (no cross product, no
  normalization of the normal, no facing flip, no dependence on DSINE's sign) and scores within a few thousandths
  of the angle form. It shares its trade: completeness and Chamfer better than BiNI on dev and val, accuracy worse
  on val beyond noise. BiNI stays the default (issue #21 owns that choice); E_t stays in the code, off, as the
  candidate if the team decides by Chamfer. The angle term is not kept.

## Plan: E44 = E_mv reads the other view at its nearest pixel, as u does (10.11.2026)
- **Goal (Peter):** one rule for "where does a pixel land in another view". u already takes the nearest pixel of
  the projection, with no border and no window; E_mv's instance gate does too, but E_mv's depth comparison
  interpolates j's inverse depth bilinearly from the four pixels around the projection, and for that it needs
  the 1 px border (`MATCH_BORDER`, so the four stay in the image when a step moves the projection) and a check
  that the four agree within the occlusion threshold (else the interpolation would mix two surfaces).
- **Change:** each match stores j's nearest pixel at the input, fixed as the matches are; the residual is
  log z_j(x) + log d_j(q) - log s with q that pixel. A projection counts if its nearest pixel lies in the image,
  as for u. Removed: `bilinear`, `corners`, `MATCH_BORDER`, the four-corner check; E_mv's Jacobian has two entries
  per match instead of five; the instance gate reads q directly.
- **What it costs, before running:** j's depth is read up to half a pixel from where x projects, so on a surface
  slanted to j the residual carries a quantization error of up to half a pixel's depth step (a few tenths of a
  percent on Hypersim's walls), and E_mv pulls the views toward that slightly stair-stepped reading. q is fixed on
  the input, so the energy stays smooth.
- **Setting, rule:** as E41-E43 (dev decides, stress looked at, RTX 3080, BiNI E_n, 5 x 10, u with the own view).
  Reference: the refiner before E44, whose scores on this GPU are on record (E41/E43 runs: dev 4.199 / 3.658 /
  3.928, val 5.701 / 5.585 / 5.643; the same GPU repeats to 0.00005 cm). A tie or better on dev (E40's floor) is
  adopted, since it removes a constant and a check; then val once. `disagreement_pct`, a diagnostic, changes
  meaning with it (nearest pixel, not interpolated) and is not compared across the change.
- **Figure:** bedroom views 18, 0, 12, 6: each pixel's distance to the GT, after minus before.

## E44: E_mv at the nearest pixel -> ties dev, loses val; reverted (10.11.2026)
- **Goal, change, rule:** as in the plan above; run as planned.
- **Commits:** 9e46516 (plan), d7cbef4 (the change, tests), f5530f5 (results, figure, `e44_figures.py`;
  `e40_check.py` skips splits a run lacks), fe1f299 (revert of d7cbef4).
- **Commands** (RTX 3080, clean tree at d7cbef4): `scene_eval --cfg configs/eval_hypersim.yaml --splits dev stress`
  (`eval_hypersim-20261110-101625`), then `--splits val` (`eval_hypersim-20261110-101721`) -> `e40_check.py` against
  E41's run at 150db44 (`eval_hypersim-20261110-093140`, bilinear) -> `results/e44_check.txt`;
  `e44_figures.py runs/eval_hypersim-20261110-093140 runs/eval_hypersim-20261110-101625` ->
  `results/e44_figures.txt`, `figures/e44_bedroom.jpg`.
- **Results, nearest pixel minus bilinear, cm** (accuracy / completeness / Chamfer; floor 0.033 / 0.037 / 0.036):

  | split | aggregate | median scene | scenes worse | without ai_009_009 |
  |---|---|---|---|---|
  | dev (3) | +0.011 / -0.016 / -0.002 | -0.015 / -0.014 / -0.016 | 1 / 0 / 1 | (not in dev) |
  | val (11) | +0.048 / +0.105 / +0.077 | +0.015 / +0.051 / +0.026 | 7 / 8 / 7 | +0.012 / +0.044 / +0.028 |

  Stress moves by at most 0.07 cm per scene (conference room completeness +0.069). E_mv matches 1-3 % more
  pixels per scene (the former border and the pixels whose four corners disagreed); the solve is faster, 1.4
  instead of 1.9 s per dev scene (two Jacobian entries per match instead of five). Labels unchanged.
- **Figure:** `figures/e44_bedroom.jpg` (in `experiments/`), bedroom views 18, 0, 12, 6: distance to the GT with
  bilinear E_mv, and nearest minus bilinear on ±2 cm. The change is not at edges: whole wall and furniture
  patches move by ±0.5-1 cm in both directions, most in views 12 and 6; over the bedroom's 20 views the mean is
  -0.016 cm.
- **Conclusion:** dev cannot tell the two readings apart, val can, and it prefers bilinear: the half-pixel reading
  error and the extra matches (pixels at occlusion edges that the four-corner check used to drop are the likely
  ones, not measured separately) cost completeness. Reverted: E_mv reads j bilinearly again, with its 1 px border
  and four-corner check, whose reasons stand (`MATCH_BORDER`). u keeps its nearest pixel; it only reads a label,
  for which interpolation has no meaning.

## Plan: E45 = the refiner against filters of Déjà View's points at equal completeness (10.11.2026)
- **Goal (issue #27):** the refiner lowers accuracy error and raises completeness error (dev, E41: 4.65 -> 4.20 /
  2.31 -> 3.66 cm). Dropping points does the same. So the question is whether the refiner beats the obvious
  baseline: Déjà View's points, filtered until they are exactly as complete as the refiner's output. Is accuracy
  (and Chamfer) then better or worse than the refiner's, beyond noise?
- **Setting:** Hypersim dev (3) for the answer, stress (3) for a second look, val (11) once as a check. Cached
  inputs of `configs/eval_hypersim.yaml`, the default refiner on main (E41's: 5 x 10, u counting the own view,
  BiNI E_n), this machine's RTX 3080 for one `scene_eval --splits dev val stress` run. That run gives the refined
  depth and Déjà View's; the filters run on its stored Déjà View depth, on the CPU. No test scene.
- **Scoring (the protocol as is, nothing new):** a filtered output is Déjà View's per-pixel cloud with the removed
  pixels set to NaN. The protocol scores such an output as it stands: Umeyama on the pixels that remain, then
  accuracy over the remaining points and completeness over every GT point (`score_output`'s surface part, shared
  through one function, not copied). Labels do not move and are not scored again.
- **Filters:** each is one score per pixel, higher means keep. The cut keeps the top fraction q of the scene's
  pixels (all 20 views pooled, as VGGT's demo takes its confidence percentile over all frames). Ties are broken by
  one fixed random permutation (seed 0): within a tie the filter has no preference. Scores use Déjà View's own
  outputs (depth, confidence, rays, the linked SAM 3 ids, DSINE's normals), never the GT.

  | filter | score (keep high) | source |
  |---|---|---|
  | confidence | C | Déjà View's confidence, as DUSt3R / VGGT clouds are cut (VGGT demo: a percentile of C) |
  | depth gradient | -\|∇ log D\| (central differences) | flying pixels sit on steep depth; log makes it scale-free |
  | both | min of the two scores' ranks | kept at q only if it passes both cuts at the same level |
  | u | the refiner's consistency u | E_d's semantic weight: share of views agreeing on the instance |
  | depth edge | -log(max / min D in the 3 x 3 window) | MoGe's point-cloud export (`utils3d.depth_edge`, rtol cut) |
  | statistical outlier | -mean distance to the k = 20 nearest points of the scene | Rusu et al. 2008 (PCL); k = Open3D's documented default |
  | multi-view consistency | number of other views whose depth agrees | MVS fusion (COLMAP, Schönberger et al. 2016; MVSNet, Yao et al. 2018); agree = E_mv's own test (Otsu's threshold, bilinear, four corners) |
  | voxel density | points of the scene in the pixel's voxel | issue #27; voxel side = the scene's median pixel footprint D / f (no GT scale) |
  | grazing angle | \|cos\| between the view ray and DSINE's normal | Curless & Levoy 1996 weight range samples by this cosine |
  | random | a uniform random number (seed 0) | the floor any filter must beat |
  | refined + confidence | C on the refined depth | not a baseline: the refiner's own curve |

  Their constants are the cited ones (k = 20, the 3 x 3 window, seed 0); the cut itself is a quantile, and the
  grid of q is the curve's resolution, not a threshold: q = 1, 0.95, ..., 0.05 on dev and stress, 1, 0.9, ..., 0.1
  on val (only the equal-completeness point is read there).
- **Equal completeness:** per scene, q is solved so that the filter's completeness equals the refiner's on that
  scene: linear interpolation between the two grid points around it, then two secant steps, each scored exactly.
  The residual gap is reported. Accuracy and Chamfer there, per scene; the split's numbers are means over scenes,
  as the protocol aggregates.
- **Noise floor and rule, fixed before running:** E40's cross-GPU floor, 0.033 / 0.037 / 0.036 cm (accuracy /
  completeness / Chamfer, dev aggregate). At equal completeness, the refiner lies **above** a filter's curve
  (better) if its accuracy is lower by more than 0.033 cm and Chamfer by more than 0.036; **below** (worse) if
  higher by more than that; otherwise a tie. Filters on Déjà View's cached depth are deterministic; the floor is
  the refiner's own.
- **Expected, written before running:** E38 found that the refiner's completeness loss is mostly the views no
  longer covering for each other, not points moving off the GT. A filter removes the worst points without moving
  the rest, so the confidence and multi-view filters should reach the refiner's accuracy at its completeness or
  beat it, i.e. the refiner lies on or below their curves on dev. Random lies far below the refiner. No guess for
  the others.
- **Report:** accuracy-completeness curves per filter with the refiner as one point, per scene and dev aggregate,
  then stress; each filter's accuracy and Chamfer at the refiner's completeness (dev, stress, val); the bedroom
  (`ai_006_007`, views 18, 0, 12, 6): which pixels the best filter removes, each pixel's distance to the GT, and
  the clouds rendered from the same cameras (`scene_figures.render_figure`), refiner next to the best filter.
- **Cost:** one GPU run, ~5 min. Each scoring takes about 2 s (3.8M points); 11 filters x (6 scenes x 24 + 11 x 14
  scorings), about 1.8 h of CPU in one process, split over processes to fit about 30 min.
- **Amendment after the first dev / stress pass (10.11.2026):** the planned solve (interpolation, then two
  secant steps) left completeness gaps up to 2.65 cm (MVS consistency on the lobby; voxel density, u and
  confidence up to 0.82 cm), against a floor of 0.037 cm: where completeness rises steeply as q falls, two steps do
  not converge. Those matches were looked at and are not used. Replaced, before val ran, by bisection of the same
  bracketing grid step until the gap is below 0.0005 cm (half the last printed digit of every score) or the
  bracket is one pixel wide. Dev and stress keep their grid curves (`--reuse`, the same code scores them) and
  are solved again; val runs once, with the bisection.
