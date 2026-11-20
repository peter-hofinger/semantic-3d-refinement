# Experiment log: semantic-guided refinement of Déjà View, from SAM 3 masks to the joint refiner

Entries follow the Experiments section of `CLAUDE.md`. E1-E45 predate its issue-based names (Mircea's proposal, 10.11.2026).
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
| E33 | Trim SAM 3 masks at Déjà View's depth jumps, against mask leakage | not adopted: the trim removes far more correct mask than leak (right object 0.935 / 0.849 / 0.905 -> 0.834 / 0.649 / 0.626 on dev), mIoU -0.04 to -0.21, refined completeness unchanged, so leaking masks do not drive the refiner's completeness loss |
| E34 | The evaluation protocol on Hypersim's 11 val scenes (ai_031_004 after its camera fix), Déjà View vs joint refiner | val accuracy 6.07 -> 5.75, completeness 4.15 -> 5.67, Chamfer 5.11 -> 5.71 cm; completeness worse on 11 of 11, Chamfer on 10: the dev picture holds and is worse on val |
| E35 | Déjà View's error by source (camera, cross-view, edges, objects) in and out of domain (Hypersim, ETH3D), and where the refiner loses completeness | in domain 42-48% of the error lies within 3 px of a depth edge; out of domain one Sim(3) per view removes 20-25% (Hypersim 4-7%); the refiner's completeness loss (ETH3D 7.77 -> 9.18 cm, worse on 13 of 13) is not at edges and not few views, it follows low confidence: whole surfaces move off the GT while views agree more |
| E36 | Learned single-view depth refiner (issue #11), with vs without SAM 3 masks, trained on 40 Hypersim train scenes | dev Chamfer 3.50 -> 3.20 cm with masks (3.29 without), better on all 3 scenes in accuracy and completeness; old protocol read it as no gain (F-score 0.844 -> 0.845); one seed, val not run |
| E37 | E_d's weight without Déjà View's confidence (conf_power p = 0 against p = 1) | does not work: val completeness loss +1.52 -> +1.49 cm, ETH3D outdoor +2.23 -> +2.22, the lowest-confidence quarter keeps its share; the unweighted λ_δ‖δ‖² is at least 2/3 of the anchor, so confidence was never what made it weak; p stays 1 |
| E38 | Fused completeness split into each view's own completeness and the bonus from views covering for each other | the lost bonus is 59% of the loss on val, 75% / 60% on ETH3D indoor / outdoor (80-85% without the two Déjà View failures); each view alone barely moves (val +0.62 cm); GT points with no view within 2 cm 37% -> 48%: E_mv merges the scattered layers onto a consensus no closer to the GT |
| E39 | Faster panoptic labeling (issue #29): prompts encoded once, detector batched 4 prompts at a time, overlaps on the GPU, association in torch | bedroom on an RTX 3080: SAM 3 29.7 -> 16.6 s, association 1.94 -> 0.57 s, pipeline 37.1 -> 22.6 s; dev mIoU / PQ^sc / AP 0.360 / 0.292 / 0.216 unchanged; 99.85-99.98% of pixels keep their mask (bf16 rounding of the batched detector), linked ids from the torch association bit-identical |
| E40 | Remove or justify every hand-set constant in the energy (issue #28): one-at-a-time dev sweep, val check | λ_d and conf_power removed (no-ops); no Huber, no border margin and E_n cut at Otsu's threshold each tie on dev, but combined they lost 0.5 cm in val's ai_009_009 (val means beyond noise), so all three stay with that reason; λ_mv, λ_δ, neighbors and the cut window stay (alternatives trade accuracy for completeness); refiner unchanged within run-to-run spread |
| E41 | After E40's review: Gauss-Newton 5 x 10 (chosen in #10, never applied), u counting the pixel's own view, λ_δ = 0 tried | own-view u changes no score by more than 0.007 cm; λ_δ = 0 looks better at 5 x 10 only because 4 of 6 solves stop early, converged it is worse (dev completeness +0.46 cm), so λ_δ stays; dev Chamfer 3.953 -> 3.928, val 5.723 -> 5.643 (ai_009_009 -0.81 cm; without it -0.006) |
| E42 | E_n as the angle between DSINE's normals and the closer-neighbor normals of the depth (Peter's proposal), 3 weights | λ_a 4e-4 passes dev (accuracy +0.010, completeness -0.229, Chamfer -0.110 cm vs BiNI) but fails val on accuracy (+0.090, worse on 10 of 11) while completeness is better on 11 of 11 and Chamfer -0.187; the λ_mv / λ_δ trade again, so BiNI stays; adopting it is a choice of metric, left open |
| E43 | The normal term as tangent residuals (n_DSINE . t / \|t\| for the closer-neighbor tangents), E42's simpler form | λ_t 4e-4 passes dev (+0.025 / -0.249 / -0.112 cm vs BiNI) and fails val on accuracy (+0.102, worse on 10 of 11; completeness -0.506, Chamfer -0.202): within a few thousandths of E42's angle term on dev, the same trade; BiNI stays |
| E44 | E_mv reads the other view at its nearest pixel, as u does (no interpolation, no 1 px border, no four-corner check) | dev ties (+0.011 / -0.016 / -0.002 cm), 1.9 -> 1.4 s per scene; val worse on all three (+0.048 / +0.105 / +0.077, median scene +0.015 / +0.051 / +0.026), so reverted: E_mv stays bilinear |
| E45 | The refiner against ten filters of Déjà View's points (confidence, depth gradient, both, u, MoGe's depth edge, PCL's statistical outlier, MVS consistency, voxel density, grazing angle, random), each cut at the scene's own quantiles to the refiner's completeness (issue #27) | at equal completeness the refiner lies below four filter curves beyond noise: voxel density (dev accuracy 3.09 vs 4.20 cm, better on 3 of 3 dev and 3 of 3 stress scenes), confidence (3.52), confidence + gradient (3.68), statistical outlier (3.80); above depth gradient, MoGe's edge, u, grazing angle and random, and above MVS consistency by 0.13 cm; second arm, filter first and then refine (dev only): below voxel density's own curve down to q = 0.5, above confidence's and the outlier filter's but only at completeness >= 5 cm, and nothing beats voxel density alone; third arm, voxel fusion (dev only): averaging a voxel's points never helped; the best of its seven variants is a filter, instance voxel density (own-instance count per voxel), 3.000 cm against voxel density's 3.090, beyond the floor only in the lobby; val and the later arms' stress not run |
| E8a | Segment ablation (issue #8): the joint refiner given no, SAM 3's, Hypersim GT, per-view permuted and 4 px shifted segments, one run, dev + val + stress | the segments only filter E_mv's matches: completeness follows the share kept (per-scene Spearman 1.00 dev, 0.90 val); Chamfer vs no segments: SAM 3 -0.069 / -0.119 cm (dev / val), GT -0.021 / -0.043, permuted -0.249 / -0.346; GT gains least and scrambled ids most, so SAM 3 is not the bottleneck and the gain is not semantic; every refined variant stays worse than Déjà View on Chamfer |
| E8b | Controls for E8a (issue #8): E_mv's matches cut at random to SAM 3's per-scene count, SAM 3's ids gating E_mv alone, a random cut at the permuted ids' count | a random cut of SAM 3's size acts like no cut (Chamfer vs Off -0.016 / -0.015 cm, dev / val), so E8a's "the gain is the count" is wrong; SAM 3 cuts the matches with the largest residuals (12.7% / 15.8% of E_mv's energy in 8.6% / 9.8% of matches), 2x as often at mask edges, and whole regions linked inconsistently; the gate alone carries about a third of SAM 3's gain over the random cut; still the same accuracy-for-completeness trade, nothing near Déjà View's Chamfer |
| E8c | Residual test for E8b (issue #8): E_mv keeps the matches of smallest initial absolute residual at SAM 3's and the permuted ids' per-scene counts, no segments | residual @ SAM 3 = gate only on completeness and Chamfer (Chamfer +0.011 / -0.003 cm, dev / val; accuracy +0.042 / +0.025) while sharing only 15% / 18% of SAM 3's cut and removing 4x its share of E_mv's energy; at the permuted count it gets 98% / 84% of permuted's Chamfer gain; the scores do not care which non-random cut, so the segments are one of several equivalent ways to do less E_mv |
| E11a | E36's learned refiner on the 11 val scenes with 3 seeds, with and without masks (issue #11) | val Chamfer 5.102 -> 5.057 with masks, 5.062 without (noise floor 0.036), by trading accuracy (+0.11) for completeness (-0.19); dev 3.50 -> 3.27 / 3.35; masks help on dev only |
| E21a | BiNI's E_n weight λ_n swept (0, 1e-5 to 1e-3) and E_n off checked on val (issue #21) | no weight improves all three scores; off: dev Chamfer 3.928 -> 3.858 but accuracy +0.067, val 5.643 -> 5.420, accuracy +0.126; λ_n = 1e-4 stays |
| E21b | MoGe-3's normals in place of DSINE's in E_n (issues #21, #20) | dev accuracy / completeness / Chamfer -0.020 / -0.023 / -0.021 cm, inside the noise floor; still 3.91 against Deja View's 3.50 |
| E20a | MoGe-3's depth fitted onto Deja View's in inverse depth, per view, per SAM 3 segment or per GT instance (issue #20) | no variant beats Deja View (val Chamfer 5.102 vs 6.192 / 5.196 / 5.142); per segment gains on completeness, loses more on accuracy; MoGe-3 alone has 2.7-6.6x Deja View's median depth error per scene |
| E41a | All views fused into one 64-cell voxel grid, a 3D U-Net and a per-pixel depth residual, without / with SAM 3 / with GT classes, 35 train scenes (issue #41) | worse than Deja View (val Chamfer 5.47 / 5.32 / 5.54 vs 5.10): held-out error rises after step 300, it overfits; SAM 3 classes ahead of none |
| E41b | E41a on 136 train scenes, 128-cell grid, 20k steps, view-subset augmentation, best held-out step, 2 seeds (issue #41) | close to Deja View but not better on val (5.24 / 5.29 / 5.25 vs 5.10, seed means); dev SAM 3 3.41 vs 3.50; class effects within seed noise |
| E25a | Déjà View's error per class over what its pixels' edge distance, depth and incidence angle predict; Hypersim dev + val with GT classes, SAM 3's classes on Hypersim and ETH3D (issue #25) | no class fails in its own way except glass (window 2.23x its geometry); S passes the permutation test by 0.002 (p = 0.05), without window and mirror p = 0.45; split halves agree at median r = 0.21; largest per-class excess error +0.29 cm of 17.0; SAM 3's labels add spread (S 0.249 vs 0.183 on the same pixels) |
| E25b | Oracle bound for segments at Déjà View's depth edges: where edge pixels sit between the two surfaces, and each segment's border snapped onto a plane fitted inside it (GT instances, GT depth edges, SAM 3, Déjà View's depth edges); Hypersim dev + val (issue #25) | 64% of the edge error is pixels on the surface across the edge, 18% bleeding; GT instances remove 19% of it (11 undragged scenes), GT depth edges 12%, SAM 3 4%, Déjà View's edges -4%; instances beat depth edges on Chamfer in 2/14 scenes -> the headroom is edges, not semantics; Chamfer moves 0.13 cm for an 8.4 cm drop in edge error; snaps dragged the Sim(3) on 3 val scenes |
| E45a | E25a out of domain: Déjà View's error per class on 24 KITTI-360 scenes (2 drives), plane-rule edges, GT up to 80 m (cap chosen after the first run); Hypersim rerun under the plane rule (issue #45; unrelated to E45) | poles alone fail beyond their geometry (1.80x, 14 of 18 scenes), cars 1.33 and vegetation 1.15 do not; headroom 2.06 cm of a 296 cm mean error; S misses the permutation test by 0.002 (p = 0.054); SAM 3's eroded masks keep no pole class; scene-wide error 14% of depth (median) against 3.4% on Hypersim: depth compressed beyond 80 m (uncapped, the Sim(3) inflates the near scene 1.02-1.37x and the error is 34%), focal length 2-25% short; Hypersim under the plane rule repeats E25a (p 0.041 vs 0.050) |
| E45b | E25b out of domain: Déjà View's error at depth edges and three oracle snaps on Tanks and Temples (Truck, Caterpillar, Meeting room; laser GT rendered by our loader, camera fitted to the scans: f = 0.605 W, k1 = -0.04), plane-rule edges, Sim(3) fitted on interior pixels (chosen after the protocol's failed); Hypersim rerun under the plane rule and under that alignment (issue #45; unrelated to E45) | the protocol's Sim(3) fails: under 1% of the GT pixels, at the scan's border where Déjà View shows the far background, shrink Truck to 0.54 and Caterpillar to 0.25 of the GT (41% and 52% error; interior fit: 8.8% and 13.0% mean, 2.6% and 5.9% median); the edge error is the same kind as on Hypersim (pixels on the surface across the edge 53-78% of it, bleeding 6-11%); perfect depth edges remove 9%, 12% and 3% of the edge-band error, SAM 3 3%, 10% and -1%, Déjà View's own edges make it worse; outdoors half of Déjà View's error lies in the 3 px along the object's outline, where knowing the outline removes 39-44% and Chamfer falls 8.2 -> 5.6 and 18.3 -> 12.1 cm (SAM 3: 10-23%, 7.5 and 14.7 cm); Hypersim under the plane rule repeats E25b, and with the fixed alignment GT instances remove 16% on all 14 scenes |

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
  (CLAUDE.md step 5), not a constant to tidy away.
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

## E45: the refiner against filters of Déjà View's points at equal completeness -> below four filters, beyond noise (10.11.2026)
- **Goal, setting, filters, rule:** as in the plan entry and its amendment above (issue #27): dev decides, stress is a
  second look, RTX 3080, the refiner on main (E41's). Run as planned, except **val was not run**: it was stopped
  twice, once by GPU memory taken by another job and once at Peter's request because it took too long (CPU shared
  with the second arm's sweep). No val number is reported; a partial file of 3 val scenes stays local and unread.
- **Commits:** 26454fc (plan), bc7fb38 (`align_output`, `match_candidates`, no behaviour change), 87d133e (filters,
  tests, sweep), 536ec1f (amendment: bisection), 2823884 (per-scene checkpoints), 8c18a86 (`render_figure` draws a
  filtered cloud), this commit (`e45_figures.py`, results, figures).
- **Commands** (this machine; `runs/` stays local):
  - `scene_eval --cfg configs/eval_hypersim.yaml --splits dev val stress` at 26454fc, clean
    (`eval_hypersim-20261110-110027`): dev 4.199 / 3.658 / 3.928 cm, val 5.701 / 5.585 / 5.643, E41's numbers to the
    third decimal.
  - `e45_filters.py runs/eval_hypersim-20261110-110027 --splits dev stress --step 0.05` (grid curves, at 87d133e's
    code), then `--reuse` at 536ec1f (the bisection; the grid scores come from the same function) ->
    `results/e45_dev_stress.txt`, 14 min for the re-solve.
  - `e45_figures.py runs/eval_hypersim-20261110-110027` -> `results/e45_figures.txt` and the figures below.
- **Checks:** at q = 1 every filter reproduces the run's Déjà View scores to 0.000000 cm on all 6 scenes. Every
  equal-completeness match is within 0.0005 cm of the refiner's completeness except 12 of 60, which stop at one
  pixel of q with gaps up to 0.0144 cm (confidence, lobby), all below the 0.037 cm floor.
- **Results, at the refiner's completeness on every scene, mean over the split's scenes** (cm; Δ = refiner minus
  filter, > 0: the refiner is worse; floor 0.033 accuracy, 0.036 Chamfer; "scenes" = scenes on which the refiner's
  accuracy is better):

  | filter | dev q | dev accuracy | dev Chamfer | dev Δ acc / Chamfer | scenes | stress Δ acc / Chamfer | scenes | verdict (dev) |
  |---|---|---|---|---|---|---|---|---|
  | voxel density | 0.574 | 3.090 | 3.373 | +1.109 / +0.555 | 0 of 3 | +4.303 / +2.151 | 0 of 3 | refiner below |
  | confidence | 0.911 | 3.523 | 3.593 | +0.675 / +0.335 | 1 of 3 | +3.824 / +1.912 | 0 of 3 | refiner below |
  | confidence + gradient | 0.872 | 3.675 | 3.666 | +0.524 / +0.262 | 1 of 3 | +2.912 / +1.456 | 0 of 3 | refiner below |
  | statistical outlier (PCL) | 0.920 | 3.796 | 3.727 | +0.403 / +0.201 | 0 of 3 | +2.528 / +1.264 | 0 of 3 | refiner below |
  | **joint refiner** | 1 | **4.199** | **3.928** | - | - | - | - | - |
  | multi-view consistency (MVS) | 0.939 | 4.330 | 3.993 | -0.131 / -0.065 | 3 of 3 | +2.108 / +1.053 | 0 of 3 | refiner above |
  | depth edge (MoGe) | 0.813 | 4.590 | 4.124 | -0.391 / -0.195 | 3 of 3 | +0.938 / +0.469 | 0 of 3 | refiner above |
  | random | 0.166 | 4.680 | 4.169 | -0.481 / -0.241 | 3 of 3 | -0.312 / -0.156 | 3 of 3 | refiner above |
  | depth gradient | 0.807 | 4.729 | 4.194 | -0.531 / -0.266 | 3 of 3 | +0.686 / +0.343 | 1 of 3 | refiner above |
  | u | 0.773 | 4.954 | 4.306 | -0.755 / -0.378 | 3 of 3 | -0.218 / -0.109 | 1 of 3 | refiner above |
  | grazing angle | 0.827 | 5.543 | 4.600 | -1.344 / -0.672 | 3 of 3 | -1.641 / -0.821 | 3 of 3 | refiner above |

  Relative to Déjà View on dev (4.675 / 2.324 / 3.500): every row has completeness 3.658 (+57%); the refiner gains
  10.2% accuracy and loses 12.2% Chamfer, voxel density gains 33.9% accuracy and Chamfer 3.6%, confidence 24.6% and
  -2.7%. Chamfer moves by half of accuracy's Δ, as it must at equal completeness.
- **Per scene, the means hide a skew.** The dev means are dominated by the lobby (ai_044_001, accuracy ~9 cm against
  ~2 cm in the other two), the stress means by the conference room (ai_047_009, ~16 cm). In the living room
  (ai_010_007) the refiner's point lies below the confidence curve (1.79 against ~2.06 cm at completeness 1.40), so
  confidence beats it on 2 of 3 dev scenes, not all. Voxel density and statistical outlier removal beat it on every
  dev and stress scene. On the conference room, cutting about 5% of the points by confidence takes accuracy from
  16.6 to 10 cm at almost no completeness: Déjà View's failure scene has a few very far points, which a cut removes
  and the refiner keeps.
- **Refine, then cut** (the refiner's own confidence curve, `refined + confidence`): cutting the refined cloud loses
  completeness much faster than cutting Déjà View's (5% of its points take dev completeness 3.66 -> 4.47 cm), as E38
  predicts: the merged views no longer cover for each other. Read off the curves, it beats confidence on Déjà View at
  lower completeness on dev (accuracy 3.17 against 3.45 cm at completeness 6.1) and ties on stress, but never
  reaches voxel density's curve.
- **Figures** (in `experiments/figures/`):
  - `e45_curves_dev.jpg`, `e45_curves_stress.jpg`: per filter (rows) and scene (columns, then the mean at each q)
    the filter's curve, random removal, the refiner's own confidence curve and the refiner's point.
    Gradient, MoGe's edge, u and grazing angle bend upward (worse) after their first few percent. A reading, not
    measured: past the flying pixels they remove correct geometry (edges that are steep in the GT too, walls seen
    at grazing angles).
  - `e45_equal_completeness.jpg`: the table's Δ for dev and stress against the floor.
  - `e45_bedroom.jpg`: bedroom views 18, 0, 12, 6, voxel density cut to the refiner's completeness there (q = 0.47):
    the pixels it removes, distance to the GT of the kept points and of the refiner. It keeps 47% of the bedroom's
    pixels, but unevenly: 73% of view 0 and 33-40% of views 18, 12, 6, whose walls become sparse speckle. The kept
    points' mean distance to the GT is 1.77 cm, the refiner's 2.20 cm over all pixels.
  - `e45_ai_006_007_render.jpg`: GT, refiner and voxel density rendered from the dollhouse and the four views pulled
    back. The refiner keeps the flying pixels at occlusion edges as long streaks across the room (dollhouse, shaded);
    voxel density has none of them, its cloud is thinner and grainier but holds the room's layout.
- **Conclusion:**
  - Done as the issue asks: at equal completeness the refiner lies **below** the curves of voxel density, confidence,
    confidence + gradient and statistical outlier removal on dev, beyond noise (by 0.40-1.11 cm accuracy, 0.20-0.56
    cm Chamfer) and on stress by more; **above** depth gradient, MoGe's edge, u, grazing angle, random and, narrowly
    (0.13 cm), MVS consistency on dev. Dropping Déjà View's points by density or confidence buys more accuracy per
    unit of completeness than the refinement does.
  - The expectation held for confidence and random and failed for MVS consistency, which loses to the refiner by
    0.13 cm on dev. A likely reason, not measured: the count of agreeing views is an integer (19 at most), so most
    of its cut falls to the random tie-break within one band.
  - Why the protocol lets half the cloud go (E38): 20 overlapping per-pixel layers cover most GT points several
    times, so removing the least supported points costs little completeness. Voxel density keeps exactly the points
    where many layers coincide. This is as much a property of scoring a 20-layer cloud as of surface quality, and
    the bedroom figure shows it: equal completeness, but views reduced to speckle.
  - Filter first, then refine: the second arm below. Not tested: the same comparison on val (11 scenes), deferred
    for time.
## Plan: E45, second arm = filter Déjà View's points first, then refine what remains (10.11.2026)
- **Numbering:** planned and run as E46 in its own session (branch `peter/e46-filter-then-refine`, merged in
  c1be9d6), then folded into E45, whose question it answers in the opposite order. Its files were renamed from
  `e46_*` to `e45_filter_refine*` in the fold; the commands below name them as they are now.
- **Goal (Peter, after E45's first arm):** at the refiner's completeness on dev (3.658 cm), filters of Déjà View's points are
  more accurate than the refiner (voxel density 3.09, confidence 3.52, statistical outlier 3.80, refiner 4.20 cm),
  and E38 traced the refiner's completeness loss to E_mv merging the views' layers. The opposite order is open:
  drop the low-scoring points first, then refine only what remains, so E_mv's consensus is not pulled by them.
  Does filter + refine lie above or below the same filter's own curve (accuracy against completeness), beyond
  noise?
- **Change (one):** a per-pixel keep mask in the refiner (`build_problem` / `refine_scene`). Removed pixels are no
  unknowns of the solve: they stay at the input and come out NaN. Matches whose source pixel or one of the target
  view's four interpolated pixels is removed are dropped; E_n's edges touching a removed pixel are cut; E_d and
  λ_δ act only on kept pixels. What the refiner derives from the input as a whole stays as it is (the scale s,
  Otsu's threshold, each view's neighbors, u, the occlusion contours): the mask only removes pixels from E. An
  all-true mask returns the unmasked solve bit for bit (CPU test).
- **Setting:** the refiner on main (E41's: 5 x 10, u with the own view, BiNI E_n, every weight 1), this machine's
  RTX 3080, the cached inputs of `configs/eval_hypersim.yaml`. Filters: E45's three best on dev, voxel density,
  confidence and statistical outlier, scored on Déjà View's input exactly as in E45 (`src/analysis/filters.py`,
  pooled over the 20 views, ties by the fixed permutation). Each keeps the top fraction q, then the refiner runs
  on the kept pixels; the output is scored by the protocol as is (Umeyama on the kept pixels, accuracy over them,
  completeness over every GT point), as E45 scores a filter. Grid q = 1, 0.9, ..., 0.1 on dev and stress (q = 1
  is the refiner itself: it must repeat E45's run); val once as a check at q = 1, 0.8, 0.6, 0.4, 0.2. No test
  scene. No new constant: the cut is a quantile, the grid is the curve's resolution.
- **Equal completeness:** per scene and grid point, the same filter without refinement is solved to the
  completeness filter + refine reached (E45's bisection, `e45_filters.sweep`, to a gap below 0.0005 cm), and
  accuracy and Chamfer are compared there. The split's numbers are means over scenes. E45's curves without
  refinement (same run, same code) are reused for dev and stress.
- **Rule, fixed before running:** E40's cross-GPU floor, 0.033 / 0.037 / 0.036 cm (accuracy / completeness /
  Chamfer, dev aggregate). At a grid point, filter + refine lies **above** the filter's curve if its accuracy is
  lower by more than 0.033 cm and its Chamfer by more than 0.036; **below** if both are higher by more than that;
  otherwise a tie. The answer per filter is the verdict over the grid on dev, stress as a second look, val as a
  check.
- **Expected, written before running:** the refiner alone (q = 1) lies far below every filter's curve (E45), and
  removing a filter's worst points does not change the refiner's trade, which E38 traced to merging good layers,
  not to outliers pulling them. So voxel density + refine lies below voxel density's curve at every q. For
  confidence, E45 found refine-then-cut above confidence alone at low completeness; in the other order I expect
  the same crossing: below near q = 1, above or a tie at low q. Statistical outlier: below, as voxel density.
- **Report:** curves per filter (with and without refinement, E45's refine-then-cut for reference) per scene and
  as dev / stress means; the Δ at equal completeness per grid point; the bedroom (`ai_006_007`, views 18, 0, 12,
  6): the best filter alone next to filter + refine at the same q, rendered from the same cameras
  (`scene_figures.render_figure`).
- **Cost:** about 3 s per refinement on the RTX 3080: 3 filters x (6 x 10 + 11 x 5) = 345 refinements, about 18
  min of GPU; the bisections run meanwhile on the CPU in 6 processes, about 25 min in all.
- **Amendment after a smoke run (dev, q = 1 and 0.5; 10.11.2026):** q = 1 repeats E45's refiner to 0.000000 cm
  on all three dev scenes. The bisections, though, took 14 min for 18 points while E45's val sweep shared the
  CPU, so the grid is cut to q = 1, 0.9, ..., 0.3 on dev and stress, and val to q = 1, 0.7, 0.4. At q = 0.5
  the smoke run's completeness was already 7-49 cm (dev means), far beyond any operating point. Nothing else
  changes. The smoke run's numbers are not reported.

## E45, second arm: filter first, then refine -> below voxel density's curve, above confidence's and the outlier filter's; none beats voxel density alone (10.11.2026)
- **Goal, change, rule:** as in the plan above, dev only. Stress and val were not run: Peter stopped them for now
  (val takes too long), so the verdicts below are dev's alone, 3 scenes.
- **Commits:** c450183 (plan), a2a421b (keep mask, tests), 7ea7f1b (sweep), cda39f6 (amended grid), 8e115de (results, figures), the fold commit (renamed files, figures redrawn under
  E45's name with the same numbers).
- **Commands** (RTX 3080, clean tree at 7ea7f1b): `e45_filter_refine.py runs/eval_hypersim-20261110-110027 --splits dev
  --grid 1 0.9 0.8 0.7 0.6 0.5 0.4 0.3` -> `results/e45_filter_refine_dev.txt` (33 min, the CPU shared with E45's val sweep);
  `e45_filter_refine_figures.py` on the same run -> `results/e45_filter_refine_figures.txt`,
  `figures/e45_filter_refine_{curves,curves_scenes,bedroom}`.
- **Checks:** q = 1 repeats E45's refiner to 0.000000 cm on all three scenes. The filter alone was matched to
  filter + refine's completeness within 0.0005 cm on 40 of 72 points; the largest gap is 0.091 cm
  (statistical outlier, lobby, q = 0.3), small next to every Δ that decides a verdict.
- **Dev means, cm** (Δ = filter + refine minus the filter alone at the same completeness, per scene and then
  averaged; floor 0.033 accuracy / 0.036 Chamfer):

  | q | voxel density: acc / comp, Δ acc / Δ Ch | confidence: acc / comp, Δ acc / Δ Ch | statistical outlier: acc / comp, Δ acc / Δ Ch |
  |---|---|---|---|
  | 1 (refiner) | 4.199 / 3.658, +1.109 / +0.555 below | 4.199 / 3.658, +0.675 / +0.335 below | 4.199 / 3.658, +0.403 / +0.201 below |
  | 0.9 | 4.079 / 3.423, +0.956 / +0.478 below | 3.151 / 5.987, -0.267 / -0.130 above | 3.440 / 5.340, -0.211 / -0.106 above |
  | 0.8 | 3.929 / 3.149, +0.723 / +0.362 below | 2.886 / 11.467, -0.296 / -0.148 above | 3.268 / 10.447, -0.127 / -0.054 above |
  | 0.7 | 3.705 / 3.058, +0.435 / +0.218 below | 2.625 / 23.108, -0.272 / -0.137 above | 3.137 / 12.938, -0.095 / -0.048 above |
  | 0.6 | 3.382 / 3.133, +0.164 / +0.082 below | 2.324 / 33.757, -0.203 / -0.100 above | 2.940 / 16.538, -0.086 / -0.043 above |
  | 0.5 | 2.918 / 7.363, +0.067 / +0.048 below | 2.069 / 48.758, -0.145 / -0.073 above | 2.642 / 26.959, -0.076 / -0.032 tie |
  | 0.4 | 2.634 / 8.191, +0.001 / +0.001 tie | 1.851 / 62.134, -0.100 / -0.050 above | 2.340 / 63.977, -0.058 / -0.030 tie |
  | 0.3 | 2.247 / 13.527, -0.014 / -0.007 tie | 1.470 / 81.583, -0.085 / -0.046 above | 2.007 / 93.927, -0.053 / -0.013 tie |

- **Per scene:** with confidence and the outlier filter, filter + refine is more accurate than the filter alone
  on all 3 scenes at every q <= 0.9. With voxel density it depends on the scene: the living room
  (ai_010_007) is more accurate from q = 0.7 on (-0.06 to -0.08 cm), the bedroom ties from q = 0.5 on, and the
  lobby (ai_044_001) is below down to q = 0.4 (+2.59 cm at q = 1, +0.49 at q = 0.6) and decides the mean.
- **Against the best curve:** read off the dev means (linear interpolation between E45's grid points, so rough),
  no point of filter + refine beats voxel density alone at its completeness. The closest is confidence + refine at
  q = 0.9: 3.15 cm accuracy at 5.99 cm completeness, against about 3.05 for voxel density alone there.
- **Two side observations:** voxel density + refine is both more accurate and more complete than the refiner
  alone at q = 0.9-0.6 (3.705 / 3.058 cm at q = 0.7 against 4.199 / 3.658): with the scattered points gone, the
  refiner loses less completeness, as E38's account predicts. And refining before the confidence cut (E45's
  "refined + confidence", gray in the curves) and after it (red) trace nearly the same curve on dev.
- **Figure:** `figures/e45_filter_refine_bedroom.jpg` (in `experiments/`), bedroom views 18, 0, 12, 6: the same 70 % of the pixels
  kept by voxel density without and with refinement, each kept pixel's distance to the GT and the change. Over the
  bedroom's kept pixels the refiner moves the mean distance from 2.219 to 2.101 cm (-0.119); 14.6 % of them move
  closer by more than 0.5 cm and 8.6 % farther. That is at the same q, so not at equal completeness (at equal
  completeness the bedroom's q = 0.7 is +0.163 cm, below). In views 18 and 12 the back walls and the door move
  away from the GT (red), and the upper left of view 6 moves toward it (blue). The protocol's render was left out:
  its shading takes normals from neighboring pixels, NaN next to most removed ones.
- **Against the expectation:** voxel density below and confidence crossing from below (q = 1) to above (q <= 0.9)
  came as written; the statistical-outlier filter was expected below, as voxel density, and lies above at
  q = 0.9-0.6. What removing the worst points does to the refiner depends on which points the filter removes.
- **Conclusion:** the opposite order has a split answer on dev. Refining what voxel density keeps lies below
  voxel density's own curve at every q down to 0.5 and ties at 0.4 and 0.3; the refiner still costs accuracy at fixed
  completeness, driven by the lobby. Refining what confidence or the statistical-outlier filter keeps lies above
  their own curves beyond noise (confidence at every q <= 0.9, outlier at q = 0.9-0.6, every scene), but only at
  completeness of 5 cm and more, and those curves stay under voxel density alone. Voxel density without the
  refiner remains the best accuracy-completeness trade on dev. Not checked: stress and val (Peter stopped them).

## Plan: E45, third arm = voxel fusion: bin the points, keep dense voxels, average each into one point (10.11.2026)
- **Goal (Peter, after the second arm):** E45's voxel density only filters: it scores each pixel by the number of
  the scene's points in its voxel and drops the low ones, but moves no point. Peter's variant also fuses: the
  points of a kept voxel are averaged into one, as voxel-grid downsampling does (Open3D's `voxel_down_sample`).
  Does fusing the kept points lie above or below voxel density's curve, and how does it stand against the refiner,
  at equal completeness? Two parts are separated: the voxel's **score** (what decides it is kept) and the
  **averaging** (what happens to its points).
- **Per-pixel form, so the protocol scores it as is:** every kept pixel's point is replaced by the mean of the
  kept points in its voxel (`filters.voxel_means`); removed pixels are NaN. Each pixel still has one point, so
  Umeyama runs on the kept pixel pairs as for every other arm. A voxel's points all land on one location; the
  protocol counts it once per pixel for accuracy and once as a nearest neighbor for completeness.
- **Variants** (voxel side = the scene's median pixel footprint D / f, E45's, unless stated; cut = the top fraction
  q of the pixels by the score, pooled over the 20 views, ties by E45's fixed permutation):

  | variant | score of a pixel (its voxel's) | points |
  |---|---|---|
  | count (E45's voxel density) | number of points | kept as they are (E45's curve, reused) |
  | count + mean | number of points | averaged |
  | Σ confidence | sum of Déjà View's confidence | kept as they are |
  | Σ confidence + mean | sum of Déjà View's confidence | averaged |
  | instance count | points of the pixel's own linked instance in the voxel | kept as they are |
  | instance count + instance mean | points of the pixel's own linked instance in the voxel | averaged per (voxel, instance), so objects do not blend |
  | count + mean, 2 x / 4 x footprint | number of points in a voxel of twice / four times the side | averaged |

  2 x and 4 x read the slope of the voxel size, as E40's grids did; no value is adopted from them. Pixels without
  an instance id count as one instance.
- **Setting:** dev only (Peter, as for the second arm), the E45 run `eval_hypersim-20261110-110027`, CPU only (no
  solve). Grid q = 1, 0.95, ..., 0.05.
- **Readout:** per scene, each variant is solved by bisection to the refiner's completeness (E45's `sweep`, gap
  below 0.0005 cm), and its accuracy and Chamfer there are compared with the refiner's and with voxel density
  alone's (E45's matches); the split's numbers are means over scenes. Curves per variant against voxel density's.
- **Rule, fixed before running:** E40's floor, 0.033 / 0.036 cm (accuracy / Chamfer). Against voxel density
  alone at the refiner's completeness: **above** if both are lower by more than the floor, **below** if both are
  higher by more, else a tie.
- **Expected, written before running:** averaging footprint-sized voxels moves each point by less than a voxel
  side, so count + mean ties voxel density alone; at 2 x and 4 x averaging merges layers as E_mv does, and E38
  predicts the completeness cost: below voxel density's curve, more so at 4 x. Σ confidence ties count (confidence
  and density rank the same points, E45: both lead). Instance count is below count where SAM 3's masks bleed,
  since a bled pixel's voxel holds few points of its own instance; no guess at the sign.
- **Cost:** 7 sweeps x 3 scenes x about 38 scorings of 2 s, in 6 processes: about 10 min.
- **Amendment before running (Peter, 10.11.2026):** only one method is reported, not the seven variants: the
  variant with the lowest mean dev accuracy at the refiner's completeness (E45's rule for its best filter) becomes
  "voxel fusion". All seven stay in the raw output (`results/e45_voxel_fusion_dev.txt`) for traceability. It is
  chosen on the same three dev scenes it is reported on, which flatters it; val, the check for that, was not run.

## E45, third arm: voxel fusion -> averaging never helps; the best variant is instance voxel density, a filter (10.11.2026)
- **Goal, variants, rule:** as in the plan and its amendment above, dev only. One method is reported: the variant
  with the lowest mean dev accuracy at the refiner's completeness of the seven.
- **Commits:** 9f6d287 (plan), 7afd2de (`voxel_sums`, `voxel_means`, `sweep`'s fuse step, tests), 0cf1cec
  (amendment), this commit (`e45_voxel_fusion_figures.py`, results, figures).
- **Commands** (this machine, CPU; git 0cf1cec clean): `e45_voxel_fusion.py runs/eval_hypersim-20261110-110027
  --splits dev` -> `results/e45_voxel_fusion_dev.txt` (all seven variants, per scene; 14.5 min);
  `e45_voxel_fusion_figures.py` on the same run -> `results/e45_voxel_fusion_figures.txt` and the figures below.
- **Checks:** unfused at q = 1, every variant reproduces the run's Déjà View scores to 0.000000 cm on the three
  scenes; every match is within 0.0005 cm of the refiner's completeness.
- **The chosen variant is not a fusion.** The lowest mean accuracy belongs to **instance count**, which keeps the
  points as they are: a pixel's score is the number of points of its own linked SAM 3 instance in its voxel.
  It is reported as **instance voxel density**. Averaging, the part of the idea that fuses, made every variant
  less accurate than the same score without it (count +0.049, Σ confidence +0.012, instance count +0.055 cm at the
  refiner's completeness). At twice and four times the footprint the averaged cloud never reaches the refiner's
  completeness: with every point kept it is already worse (living room 1.675 and 2.792 cm against the refiner's
  1.398), as E38 predicts for merging the views' layers.
- **Instance voxel density at the refiner's completeness, dev** (cm; floor 0.033 / 0.036):

  | method | accuracy | Chamfer | living room | bedroom | lobby |
  |---|---|---|---|---|---|
  | voxel density (first arm) | 3.090 | 3.373 | 1.492 | 1.772 | 6.005 |
  | **instance voxel density** | **3.000** | **3.329** | 1.477 | 1.758 | 5.765 |
  | joint refiner | 4.199 | 3.928 | 1.790 | 2.208 | 8.597 |

  Against voxel density: -0.090 / -0.045 cm, above its curve beyond the floor, better on 3 of 3 scenes, but only
  the lobby by more than the floor (-0.241); the living room (-0.015) and the bedroom (-0.014) tie. Against the
  refiner: -1.199 / -0.600 cm, better on every scene.
- **Chosen on the reported scenes:** it is the best of seven on the same three dev scenes, which flatters it; with
  a margin of 0.09 cm carried by one scene, it may not hold on val, which was not run.
- **Figures** (in `experiments/figures/`): `e45_voxel_fusion_curves.jpg` (per dev scene and the mean: the two
  curves coincide on the steep part and separate only at completeness above ~1.7 cm in the bedroom and ~6.5 cm in
  the lobby, where instance voxel density is lower); `e45_voxel_fusion_bedroom.jpg` (views 18, 0, 12, 6, both cut
  to the refiner's completeness: q = 0.471 and 0.450, kept points 1.770 and 1.755 cm from the GT on average; the
  maps look alike).
- **Conclusion:** the fusion idea does not help: averaging a voxel's points costs completeness and accuracy, at the
  footprint and more so with larger voxels, because the protocol rewards the scattered layers that averaging
  merges. What helps a little is the semantic version of the score: counting only the points of the pixel's own
  instance, so a voxel that two objects share (a bleeding edge) counts less. Its gain over voxel density is within
  noise on two of three dev scenes; worth checking on val before calling it better.

## Plan: E8a = do the segments help the joint refiner, and is SAM 3 the bottleneck (issue #8, 10.11.2026)
- **Question:** the refiner reads the segments in three places: u in E_d's weight, E_mv's gate (a match counts
  only if it lands on its source's instance) and E_n's edges (only within one instance). Does any of that help,
  and if the segments were perfect, would it help more? Prior hints only: E40 found removing u a tie; Peter's
  dev table before the current protocol (05.11, not in this log) had semantics off vs on at completeness 3.86 vs
  3.66 and Chamfer 4.00 vs 3.93 cm.
- **Variants** (one run, every variant on the same scenes; refiner, weights, Gauss-Newton budget, alignment and
  scored labels identical, so only accuracy, completeness and Chamfer can move):
  1. Déjà View, no refinement.
  2. **Off:** no segments: u = 1, E_mv and E_n ungated (`segments: none`, the refiner's default).
  3. **SAM 3:** the linked masks, as now.
  4. **GT:** Hypersim's instance ids, consistent across views by construction; unlabeled and no-mesh pixels have
     none. Walls, floor and ceiling are one instance per class (E29), so GT is an upper bound on identity and
     on object boundaries, not on the boundaries between walls.
  5. **Corrupted SAM 3:** (a) **permuted:** each view's ids permuted among themselves, independently per view (one
     permutation for all views would change nothing): grouping and boundaries within a view stay, cross-view
     identity goes, so u and E_mv's gate break while E_n is unchanged. (b) **shifted:** each view's id map moved
     4 px (about 1% of the width, past E35's 3 px edge band) along a random axis direction: identity stays,
     boundaries move off the depth edges, which touches all three. Seed 0 for both.
- **Reading:** GT ~ Off: the energy cannot use semantics, whatever their quality (stated plainly). GT > Off and
  SAM 3 ~ Off: SAM 3 is the bottleneck. SAM 3 > Off and a corruption ~ Off: semantics help, and which corruption
  takes the gain says whether identity or boundaries do it. "~" is E40's floor (0.033 / 0.037 / 0.036 cm,
  accuracy / completeness / Chamfer, the dev difference of the unchanged refiner between two GPUs); all variants
  run in one process on one GPU, which repeats to 0.00005 cm (E44). No single-surface metric until the team
  decides on one (#25, #32): all three geometry scores are reported.
- **Setting:** `experiments/configs/e8a_segments.yaml`, `scene_eval --splits dev val stress` on the student
  cluster (RTX 5090); dev and val both read, no test scene. One seed per corruption: if a corrupted variant
  lands within twice the floor of a decision, a second seed is run before concluding.
- **Figure:** bedroom views 18, 0, 12, 6: distance to the GT for SAM 3, and Off / GT minus SAM 3 on a shared scale.

## E8a: the segments act as a filter on E_mv's matches, not as semantics: GT gives less than SAM 3, scrambled ids give more (10.11.2026)
- **Question, variants, reading:** as in the plan above (issue #8); run as planned, nothing changed after seeing
  numbers.
- **Commits:** 021f284 (`segments` per variant in `scene_eval`, tests), 33f9deb (plan, config), this commit
  (`e8a_inputs.py`, `e8a_table.py`, `e8a_figures.py`, results, figure).
- **Commands** (student cluster, RTX 5090, clean tree at 33f9deb, job 1669: input check 1:05, all six variants on
  dev + val + stress 5:23): `experiments/scripts/e8a_inputs.py experiments/configs/e8a_segments.yaml --splits dev
  val` -> `results/e8a_inputs.txt`; `scene_eval --cfg experiments/configs/e8a_segments.yaml --splits dev val
  stress` (run `e8a_segments-20261110-130124`); then (CPU job 1670) `e8a_figures.py <run>` -> `results/e8a_figures.txt`, `figures/e8a_bedroom.jpg`; here, from the copied run.yaml,
  `e8a_table.py <run>/run.yaml results/e8a_inputs.txt` -> `results/e8a_table.txt`.
- **What each variant hands the refiner** (mean over scenes; `e8a_inputs.txt`): share of E_mv's matches the
  instance gate keeps / mean u on pixels with a segment.

  | variant | dev | val |
  |---|---|---|
  | Off | 1.000 / 1 | 1.000 / 1 |
  | GT | 0.978 / 0.857 | 0.976 / 0.835 |
  | SAM 3 | 0.914 / 0.780 | 0.902 / 0.716 |
  | shifted 4 px | 0.869 / 0.746 | 0.868 / 0.691 |
  | permuted | 0.129 / 0.162 | 0.217 / 0.167 |

  GT leaves pixels without a segment where Hypersim has no instance: 79% in ai_003_010 (no wall / floor / ceiling
  labels, E34), 40% in ai_004_010, 38% in ai_005_005; there it gates nothing.
- **Results, cm** (accuracy / completeness / Chamfer, lower is better; Δ against Off; floor 0.033 / 0.037 /
  0.036; "worse" = scenes on which the variant is worse than Off):

  | split | variant | aggregate | Δ vs Off | Δ median scene | scenes worse |
  |---|---|---|---|---|---|
  | dev (3) | Déjà View | 4.651 / 2.307 / 3.479 | +0.554 / -1.521 / -0.483 | +0.418 / -0.643 / -0.143 | 3 / 0 / 0 |
  | | Off | 4.097 / 3.828 / 3.962 | | | |
  | | SAM 3 | 4.167 / 3.620 / 3.893 | +0.070 / -0.208 / -0.069 | +0.044 / -0.113 / -0.035 | 3 / 0 / 0 |
  | | GT | 4.146 / 3.737 / 3.941 | +0.049 / -0.091 / -0.021 | +0.014 / -0.074 / -0.027 | 2 / 0 / 0 |
  | | permuted | 4.484 / 2.943 / 3.713 | +0.387 / -0.885 / -0.249 | +0.279 / -0.514 / -0.138 | 3 / 0 / 0 |
  | | shifted | 4.160 / 3.536 / 3.848 | +0.062 / -0.292 / -0.115 | +0.041 / -0.115 / -0.037 | 3 / 0 / 0 |
  | val (11) | Déjà View | 6.065 / 4.147 / 5.106 | +0.430 / -1.704 / -0.637 | +0.376 / -1.044 / -0.232 | 11 / 0 / 0 |
  | | Off | 5.635 / 5.851 / 5.743 | | | |
  | | SAM 3 | 5.688 / 5.559 / 5.624 | +0.053 / -0.292 / -0.119 | +0.057 / -0.114 / -0.026 | 8 / 0 / 5 |
  | | GT | 5.722 / 5.678 / 5.700 | +0.087 / -0.173 / -0.043 | +0.050 / -0.087 / -0.010 | 9 / 0 / 4 |
  | | permuted | 5.817 / 4.978 / 5.397 | +0.182 / -0.873 / -0.346 | +0.240 / -0.621 / -0.167 | 10 / 0 / 0 |
  | | shifted | 5.677 / 5.557 / 5.617 | +0.042 / -0.294 / -0.126 | +0.038 / -0.123 / -0.026 | 10 / 0 / 2 |

  Stress (3 tilt-shift scenes, per scene in `e8a_table.txt`) orders the same way. Labels are the same for every
  variant (dev mIoU / PQ / AP 0.352 / 0.294 / 0.217, val 0.395 / 0.256 / 0.353). Refinement 0.9-1.7 s per scene.
- **Completeness follows the share of matches kept,** whatever the segments are: over the five refined variants,
  the per-scene Spearman correlation between kept share and completeness is median 1.00 (min 0.90) on dev and
  0.90 (min 0.70) on val. Fewer matches -> less of E_mv's merging -> the result stays nearer Déjà View: better
  completeness, worse accuracy, Chamfer better (it weighs completeness's large change more than accuracy's small
  one). Every refined variant stays worse than Déjà View on Chamfer.
- **Figure:** `figures/e8a_bedroom.jpg`, bedroom views 18, 0, 12, 6: SAM 3's segments, the distance to the GT
  refined with them, and Off / GT / permuted minus SAM 3 on ±2 cm. Off and GT differ from SAM 3 in thin lines along
  SAM 3's mask edges and at the bed frame (the matches the gate cuts), elsewhere within ±0.5 cm (bedroom, all 20
  views: 10.5% / 8.7% of pixels change by > 0.5 cm, mean -0.036 / -0.029 cm). Permuted moves whole walls, doors
  and the ceiling by up to 2 cm in both directions (32.6% of pixels, mean +0.195 cm on this scene): with E_mv
  mostly gated off, each view keeps its own Déjà View layer.
- **Reading, by the plan's rule:** GT ~ Off on Chamfer (dev -0.021 within the floor, val -0.043 just beyond it, 4
  of 11 val scenes worse): the energy gains next to nothing from correct segments. SAM 3 beats GT (dev -0.048, val
  -0.076 Chamfer), and both corruptions beat or tie SAM 3, permuted by far. So SAM 3 is not the bottleneck, and
  "the segments help" does not hold in the sense the method claims: SAM 3's gain over Off comes from cutting
  about 9% of E_mv's matches, and a scrambled segmentation that cuts 80-90% gains more. CLAUDE.md's step 5
  (shuffle the semantics; if the gain does not drop, semantics are not what helps) fails.
- **Not separated here:** the gate and u move together (both read the same ids); E40 found removing u a tie, so the
  gate is the likely carrier, not measured. The next step down is a control without semantics: Off with E_mv's
  matches dropped at random at SAM 3's per-scene rate (not run; would need a match-dropout switch in the refiner).
- **Conclusion:** in the current energy, segments are a way to switch off part of E_mv, and E_mv is the term that
  trades accuracy for completeness (E40-E43). Making semantics matter needs a term that uses the identity for
  something E_mv-pruning cannot do, or a refiner that gets nearer the GT than Déjà View at all; tuning which
  segments to use cannot.

## Plan: E8b = is SAM 3's gain over no segments only the number of matches it cuts (issue #8, 10.11.2026)
- **Question:** E8a found completeness ordered by the share of E_mv's matches the segments keep, whatever the
  segments are, and left two things unseparated: (1) does a gate that cuts the same number of matches at random,
  with no segments anywhere else, do what SAM 3 does; (2) of SAM 3's three uses (u in E_d, E_mv's gate, E_n's
  edges), is the gate the one that moves the scores. Same run setup as E8a (refiner, weights, budget, alignment,
  scored labels).
- **Variants:**
  1. Déjà View, no refinement. 2. **Off:** no segments. 3. **SAM 3:** as now (u, gate and E_n edges).
  4. **random @ SAM 3:** Off, then E_mv keeps a uniform random subset of its matches, as many per scene as SAM 3's
     gate keeps (`keep_like: sam3`, dev 0.91, val 0.90 mean), seeds 0 and 1.
  5. **gate only:** SAM 3's ids gate E_mv's matches, nothing else (`segments: none`, `gate: sam3`: u = 1, E_n
     ungated).
  6. **random @ permuted:** as 4, at the count the per-view permuted ids keep (dev 0.13, val 0.22 mean), seed 0:
     whether E8a's largest gain is also just the count.
- **Reading** (floor 0.033 / 0.037 / 0.036 cm as in E8a): random @ SAM 3 within the floor of SAM 3 on all three
  scores -> SAM 3's gain is match pruning, the identity adds nothing (confirmed, stated plainly). SAM 3 better
  than random by more than the floor -> which matches SAM 3 cuts matters; gate only vs SAM 3 then says whether
  the gate carries it or u and E_n do. Random @ permuted near permuted -> the same for the corrupted ids.
- **Setting:** `experiments/configs/e8b_dropout.yaml`, `scene_eval --splits dev val stress`, student cluster RTX
  5090, one job (~6 min, as E8a's). The two seeds of 4 give the seed spread; if it exceeds the floor, it is the
  floor for this comparison.

## E8b: a random cut of SAM 3's size acts like no cut; SAM 3 cuts the matches that pull hardest, at mask edges (10.11.2026)
- **Question, variants, reading:** as in the plan above (issue #8); run as planned, nothing changed after seeing
  numbers.
- **Commits:** d3f75f4 (`MatchSelection`: `gate` and `keep` for `build_problem`, `gate` / `keep_like` per variant in
  `scene_eval`, tests), 7b3f2c1 (plan, config), this commit (`e8b_table.py`, `e8b_matches.py`, `e8b_figures.py`,
  results, figure).
- **Code removed after E8c:** `MatchSelection` (and the `gate` / `keep_like` / `keep_by` variant keys) served only
  E8b and E8c, so it left the solver once they were answered; E8a's `segments` choice stays. To rerun `e8b_dropout.yaml`, check
  out f115eaf. The analysis scripts use only functions still on main.
- **Commands** (student cluster, clean tree at 7b3f2c1): `scene_eval --cfg experiments/configs/e8b_dropout.yaml
  --splits dev val stress` (job 1674, RTX 5090, 6:20 for all seven variants; run `e8b_dropout-20261110-134432`);
  `e8b_matches.py experiments/configs/e8b_dropout.yaml --splits dev val` -> `results/e8b_matches.txt` and
  `e8b_figures.py <run>` -> `results/e8b_figures.txt`, `figures/e8b_bedroom.jpg` (CPU jobs 1678, 1679); here,
  from the copied run.yaml, `e8b_table.py <run>/run.yaml` -> `results/e8b_table.txt`. Déjà View, Off and SAM 3
  repeat E8a's scores to the third decimal on every split, so E8a's permuted variant compares with this run.
- **Results, cm** (accuracy / completeness / Chamfer, lower is better; Δ against the random cut at SAM 3's size,
  seed 0; floor 0.033 / 0.037 / 0.036; "worse" = scenes on which the variant is worse than that random cut).
  Matches kept, share of Off's: dev / val 0.914 / 0.902 for SAM 3, both random cuts and gate only; 0.130 / 0.217
  for random @ permuted.

  | split | variant | aggregate | Δ vs random @ SAM 3 | Δ median scene | scenes worse |
  |---|---|---|---|---|---|
  | dev (3) | Déjà View | 4.651 / 2.307 / 3.479 | +0.551 / -1.485 / -0.467 | +0.417 / -0.625 / -0.134 | 3 / 0 / 0 |
  | | Off | 4.097 / 3.828 / 3.962 | -0.003 / +0.036 / +0.016 | -0.001 / +0.018 / +0.009 | 0 / 3 / 3 |
  | | SAM 3 | 4.167 / 3.620 / 3.893 | +0.067 / -0.172 / -0.053 | +0.042 / -0.094 / -0.026 | 3 / 0 / 0 |
  | | random @ SAM 3, seed 0 | 4.100 / 3.792 / 3.946 | | | |
  | | random @ SAM 3, seed 1 | 4.100 / 3.796 / 3.948 | -0.000 / +0.003 / +0.002 | -0.000 / +0.002 / +0.000 | 1 / 3 / 3 |
  | | gate only | 4.112 / 3.741 / 3.927 | +0.012 / -0.051 / -0.020 | +0.014 / -0.039 / -0.013 | 3 / 1 / 1 |
  | | random @ permuted | 4.245 / 3.286 / 3.766 | +0.145 / -0.506 / -0.180 | +0.161 / -0.342 / -0.128 | 3 / 0 / 0 |
  | | (E8a) permuted | 4.484 / 2.943 / 3.713 | | | |
  | val (11) | Déjà View | 6.065 / 4.147 / 5.106 | +0.431 / -1.675 / -0.622 | +0.374 / -1.036 / -0.228 | 11 / 0 / 0 |
  | | Off | 5.635 / 5.851 / 5.743 | +0.001 / +0.029 / +0.015 | -0.001 / +0.010 / +0.005 | 4 / 11 / 10 |
  | | SAM 3 | 5.688 / 5.559 / 5.624 | +0.054 / -0.263 / -0.104 | +0.054 / -0.099 / -0.016 | 8 / 0 / 4 |
  | | random @ SAM 3, seed 0 | 5.634 / 5.823 / 5.728 | | | |
  | | random @ SAM 3, seed 1 | 5.634 / 5.823 / 5.728 | +0.000 / -0.000 / -0.000 | -0.000 / -0.001 / -0.001 | 4 / 5 / 4 |
  | | gate only | 5.636 / 5.751 / 5.694 | +0.003 / -0.071 / -0.034 | +0.004 / -0.015 / -0.010 | 6 / 0 / 4 |
  | | random @ permuted | 5.718 / 5.524 / 5.621 | +0.084 / -0.299 / -0.107 | +0.100 / -0.256 / -0.072 | 9 / 0 / 0 |
  | | (E8a) permuted | 5.817 / 4.978 / 5.397 | | | |

  The two seeds differ by at most 0.004 cm, far below the floor. Stress orders the same way (`e8b_table.txt`).
  Refinement takes 1.9-2.3 s per scene with a random cut against 0.9-1.3 s without: drawing the subset (a CPU
  permutation of ~16 M matches) is inside the timed solve.
- **Random cut at SAM 3's size = no cut:** against Off it moves dev / val Chamfer by -0.016 / -0.015, within the
  floor. E8a's reading that SAM 3's gain comes from the number of matches it cuts is wrong.
- **Which matches SAM 3 cuts** (`e8b_matches.txt`, on the input, dev / val means): 8.6% / 9.8% of the matches,
  carrying 12.7% / 15.8% of E_mv's initial energy (median |residual| 1.3x / 1.5x the kept matches'); 58% / 44% of
  them start within 3 px of a SAM 3 mask boundary, against 25% / 22% of all matches. The bedroom figure shows two
  kinds: thin lines along mask edges, and whole regions whose ids differ between views (the ceiling in views 12
  and 6, cut almost entirely). Those are the matches that pull hardest, so cutting them leaves the result nearer
  Déjà View: per point of matches cut, SAM 3 moves Chamfer about 3.5x (dev) and 8x (val) as much as the random
  cut at the permuted size.
- **Gate vs the rest:** gate only gets 30% / 27% of SAM 3's completeness gain over the random cut (dev / val) and
  38% / 33% of its Chamfer gain, accuracy almost unchanged. The rest comes from u and E_n's cut edges together,
  not separated here (E40 found removing u alone a tie, on the full method).
- **Random @ permuted vs permuted:** the count explains much of the permuted ids' gain over Off (Chamfer 79% on
  dev, 35% on val; completeness 61% / 37%), but not all: permuted keeps every match of a pixel without a segment
  and cuts by ids, so its cut is not random either.
- **Figure:** `figures/e8b_bedroom.jpg`, bedroom views 18, 0, 12, 6: SAM 3's segments, the share of each pixel's
  matches the gate cuts, the distance to the GT after the random cut, and SAM 3 / gate only minus the random cut on
  ±2 cm. SAM 3 differs from the random cut along mask edges, on the cut ceiling (red, further from the GT) and on
  the chairs and bed frame (both signs); gate only shows the same places, weaker. Bedroom, all 20 views: SAM 3
  minus random mean +0.035 cm (10.3% of pixels change by > 0.5 cm), gate only +0.013 (5.0%), Off -0.001 (0.6%).
- **Reading, by the plan's rule:** SAM 3 differs from the random cut by more than the floor on completeness and
  Chamfer (dev and val), so which matches it cuts matters, and the gate alone carries about a third of it. In the
  direction it acts, nothing changed from E8a: every variant trades accuracy for completeness, SAM 3 is less
  accurate than the random cut (+0.067 / +0.054), and none comes near Déjà View's Chamfer. On val, SAM 3's
  Chamfer gain over the random cut sits in a few scenes (median scene -0.016, 4 of 11 worse).
- **Conclusion:** the segments are not a random filter: they find the matches with the largest residuals, at
  object edges and where the linking disagrees between views, and switching those off is a cheap way to do less
  of E_mv's merging. That is still less E_mv, not semantics that bring the surface nearer the GT. A
  semantics-free check of the same idea is the next step down: cut the 9% of matches with the largest initial
  residual and see whether it lands on SAM 3 (not run).

## Plan: E8c = is SAM 3's gate just a residual detector (issue #8, 10.11.2026)
- **Question:** E8b found SAM 3's gate cuts the matches with the largest initial residuals (12.7% / 15.8% of E_mv's
  energy in 8.6% / 9.8% of the matches, dev / val). Does a cut with no segments, of the same size, that drops the
  matches with the largest initial |residual|, do what SAM 3 does? Same run setup as E8a / E8b (refiner, weights,
  budget, alignment, scored labels).
- **Variants:**
  1. Déjà View, no refinement. 2. **Off:** no segments. 3. **SAM 3:** as now. 4. **random @ SAM 3,** seed 0
     (E8b's control). 5. **gate only** (E8b): SAM 3's ids gate E_mv alone, the fair comparison for a cut that
     touches only E_mv.
  6. **residual @ SAM 3:** Off, then E_mv keeps the matches with the smallest |residual| on the input depth
     (`residuals(d0, scale, scene, m)`, E_mv's own residual), as many per scene as SAM 3's gate keeps
     (`keep_like: sam3`, `keep_by: residual`).
  7. **residual @ permuted:** as 6, at the count the per-view permuted ids keep (dev 0.13, val 0.22 of Off's).
- **Reading** (floor 0.033 / 0.037 / 0.036 cm as in E8a): residual @ SAM 3 within the floor of gate only (and of
  SAM 3) on all three scores -> the segments add nothing beyond a residual test (stated plainly). Clearly worse
  than gate only -> where the segments cut (edges, inconsistently linked regions) matters beyond residual size.
  Clearly better -> a plain residual test beats the segments at their own job.
- **Setting:** `experiments/configs/e8c_residual.yaml`, `scene_eval --splits dev val stress`, student cluster RTX
  5090, one job (~6 min, as E8b's). Déjà View, Off, SAM 3, random and gate only must repeat E8b's scores.

## E8c: a residual test of SAM 3's size does what SAM 3's gate does, while cutting mostly other matches (10.11.2026)
- **Question, variants, reading:** as in the plan above (issue #8); run as planned, nothing changed after seeing
  numbers.
- **Commits:** da8dd77 (`MatchSelection.by`: `random` or `residual`, `keep_by` per variant in `scene_eval`, tests),
  5f34a8d (plan, config), this commit (`e8c_overlap.py`, `e8c_figures.py`, `e8b_table.py` takes the references as
  arguments, results, figure).
- **Code removed after E8c:** `MatchSelection` (and the `gate` / `keep_like` / `keep_by` variant keys) served only
  E8b and E8c, so it left the solver once they were answered; E8a's `segments` choice stays. To rerun `e8c_residual.yaml`, check
  out f115eaf. The analysis scripts use only functions still on main.
- **Commands** (student cluster at 5f34a8d; the run is marked dirty only by the two untracked E8c scripts, which
  `scene_eval` does not import and which match this commit's by md5): `scene_eval --cfg
  experiments/configs/e8c_residual.yaml --splits dev val stress` (job 1700, RTX 5090, 6:18 for all seven variants;
  run `e8c_residual-20261110-153920`); `e8c_overlap.py experiments/configs/e8c_residual.yaml --splits dev val` ->
  `results/e8c_overlap.txt` and `e8c_figures.py <run>` -> `results/e8c_figures.txt`, `figures/e8c_bedroom.jpg`
  (CPU job 1701); here, from the copied run.yaml, `e8b_table.py <run>/run.yaml Off "gate only"` ->
  `results/e8c_table.txt`. Déjà View, Off, SAM 3, random @ SAM 3 and gate only repeat E8b's scores to the third
  decimal on dev and val.
- **Results, cm** (accuracy / completeness / Chamfer, lower is better; Δ against gate only; floor 0.033 / 0.037 /
  0.036; "worse" = scenes on which the variant is worse than gate only). Matches kept, share of Off's: dev / val
  0.914 / 0.902 for SAM 3, random, gate only and residual @ SAM 3; 0.130 / 0.217 for residual @ permuted.

  | split | variant | aggregate | Δ vs gate only | Δ median scene | scenes worse |
  |---|---|---|---|---|---|
  | dev (3) | Déjà View | 4.651 / 2.307 / 3.479 | +0.539 / -1.434 / -0.447 | +0.414 / -0.586 / -0.121 | 3 / 0 / 0 |
  | | Off | 4.097 / 3.828 / 3.962 | -0.015 / +0.087 / +0.036 | -0.015 / +0.058 / +0.021 | 0 / 3 / 2 |
  | | SAM 3 | 4.167 / 3.620 / 3.893 | +0.054 / -0.121 / -0.033 | +0.028 / -0.055 / -0.013 | 3 / 0 / 0 |
  | | random @ SAM 3, seed 0 | 4.100 / 3.792 / 3.946 | -0.012 / +0.051 / +0.020 | -0.014 / +0.039 / +0.013 | 0 / 2 / 2 |
  | | gate only | 4.112 / 3.741 / 3.927 | | | |
  | | residual @ SAM 3 | 4.154 / 3.720 / 3.937 | +0.042 / -0.020 / +0.011 | +0.021 / -0.012 / +0.016 | 3 / 1 / 3 |
  | | residual @ permuted | 4.495 / 2.942 / 3.719 | +0.383 / -0.799 / -0.208 | +0.317 / -0.428 / -0.103 | 3 / 0 / 0 |
  | | (E8a) permuted | 4.484 / 2.943 / 3.713 | | | |
  | val (11) | Déjà View | 6.065 / 4.147 / 5.106 | +0.429 / -1.604 / -0.587 | +0.362 / -1.013 / -0.219 | 11 / 0 / 1 |
  | | Off | 5.635 / 5.851 / 5.743 | -0.001 / +0.100 / +0.049 | -0.005 / +0.029 / +0.020 | 5 / 11 / 7 |
  | | SAM 3 | 5.688 / 5.559 / 5.624 | +0.052 / -0.192 / -0.070 | +0.043 / -0.067 / -0.006 | 11 / 0 / 5 |
  | | random @ SAM 3, seed 0 | 5.634 / 5.823 / 5.728 | -0.003 / +0.071 / +0.034 | -0.004 / +0.015 / +0.010 | 5 / 11 / 7 |
  | | gate only | 5.636 / 5.751 / 5.694 | | | |
  | | residual @ SAM 3 | 5.661 / 5.719 / 5.690 | +0.025 / -0.032 / -0.003 | +0.019 / -0.032 / -0.003 | 8 / 2 / 4 |
  | | residual @ permuted | 5.818 / 5.085 / 5.451 | +0.182 / -0.666 / -0.242 | +0.200 / -0.569 / -0.166 | 9 / 0 / 0 |
  | | (E8a) permuted | 5.817 / 4.978 / 5.397 | | | |

  Stress orders the same way (`e8c_table.txt`, per scene only). Refinement takes 1.1-1.4 s per scene with the
  residual cut (dev / val means), about as long as gate only.
- **Residual @ SAM 3 = gate only** on completeness and Chamfer, within the floor on both splits (Chamfer +0.011 /
  -0.003); accuracy is worse by +0.042 / +0.025, beyond the floor on dev. Against SAM 3 it is worse by +0.044 /
  +0.066 Chamfer, the part E8b put on u and E_n's cut edges.
- **They cut different matches** (`e8c_overlap.txt`, on the input, dev / val means): only 14.9% / 17.8% of SAM 3's
  cut is also in the residual cut (8.6% / 9.8% expected at random). The residual cut carries 48.9% / 58.5% of E_mv's
  initial energy against SAM 3's 12.7% / 15.8%, and lies less at mask edges (35.8% / 31.1% within 3 px of a SAM 3
  boundary, against 58.1% / 44.0%). So the gate is not a residual detector: a test that cuts about four times
  as much of E_mv's initial energy, mostly elsewhere, lands on the same scores.
- **Residual @ permuted vs permuted:** at the permuted ids' count the residual cut gets 98% / 84% of permuted's
  Chamfer gain over Off (dev / val), against 79% / 35% for the random cut there (E8b). E8a's largest gain is the
  count plus a residual-like choice, not the ids.
- **Figure:** `figures/e8c_bedroom.jpg`, bedroom views 18, 0, 12, 6: the share of each pixel's matches SAM 3's
  gate and the residual test cut, the distance to the GT with the gate only, and residual @ SAM 3 / SAM 3 minus
  gate only on ±2 cm. SAM 3 cuts thin mask edges and the inconsistently linked ceiling; the residual test cuts
  whole patches of bed cover, floor and furniture. Residual minus gate only changes broad regions with both signs
  (a wall patch in view 12 moves away from the GT); SAM 3 minus gate only changes edges. Bedroom, all 20 views:
  mean +0.021 cm for residual @ SAM 3 (8.1% of pixels change by > 0.5 cm), +0.022 for SAM 3 (6.6%).
- **Reading, by the plan's rule:** residual @ SAM 3 is within the floor of gate only on completeness and Chamfer,
  so on those scores the segments' gate adds nothing a semantics-free residual test does not, though the two pick
  mostly different matches. What SAM 3 adds beyond either is u and E_n's edges (E8b), and all of it stays the same
  trade of accuracy for completeness, far from Déjà View's Chamfer.
- **Conclusion:** how many of E_mv's matches are cut matters, and so does cutting non-randomly, but the scores
  barely care which non-random cut: segment edges, inconsistent ids and large residuals all buy the same small
  completeness gain. For issue #8: the segments act on the joint refiner as one of several equivalent ways to do
  less of E_mv's merging, not as semantics that move the surface towards the GT.

## E11a: the learned refiner on val with three seeds -> E36's dev gain shrinks to the noise floor; the masks stop mattering (10.11.2026)
- **Goal:** issue #11: does E36's dev gain (Chamfer 3.50 -> 3.20 cm with masks, one seed, three scenes) hold on
  the 11 val scenes and across seeds, and do the masks help there?
- **Setting:** E36's refiner, unchanged (`configs/e11a_learned.yaml` = `configs/learned.yaml` with
  `seeds: [0, 1, 2]` and `predict_splits: [dev, val]`): with masks (`learned-sam3`) and without (`learned-rgb`),
  one model per seed, trained on the 36 + 4 train scenes, written for dev and val. Each model is scored without
  the joint refiner on top. Noise floor as E40 (0.033 / 0.037 / 0.036 cm).
- **Commits:** 424fe7f (`seeds`, `predict_splits`, the config, a config test). Workstation, RTX 3080, clean tree.
- **Data:** val prepared on the workstation for the first time: `hypersim-download --splits val` (576 s),
  `hypersim-prepare` (48 s), `dejaview-predict` (86 s), `sam3-segment --prompts configs/prompts/nyu40.txt
  --threshold 0.3` (228 s), `dsine-predict` (72 s), all 11 scenes.
- **Commands:** `uv run learned-refine --config configs/e11a_learned.yaml` (`learned-20261110-173729`, 2446 s for
  the 6 models), then `scene_eval --splits dev val` with `configs/eval_hypersim.yaml` (`eval_hypersim-20261110-181811`)
  and once per model with a run-local copy of `configs/eval_learned.yaml` (backbone = the model, one variant,
  refiner none; `runs/e11a_configs/`, each copy saved in its run's `run.yaml`): `e11a_<model>-20261110-18*`.
- **Held out (4 train scenes), mean abs log error 3.68%:** with masks 3.43 / 3.43 / 3.46% (seeds 0 / 1 / 2),
  without 3.52 / 3.49 / 3.49%. Seed 0 is close to E36's training (3.43 / 3.50%).
- **Aggregates, cm, mean of the 3 seeds** (seed range of Chamfer in brackets):

  | variant | dev acc / comp | dev Chamfer | val acc / comp | val Chamfer |
  |---|---|---|---|---|
  | Deja View | 4.675 / 2.324 | 3.500 | 6.061 / 4.142 | 5.102 |
  | joint refiner (current) | 4.199 / 3.658 | 3.928 | 5.701 / 5.585 | 5.643 |
  | learned, masks | 4.415 / 2.118 | 3.267 (3.239-3.282) | 6.167 / 3.948 | 5.057 (5.040-5.072) |
  | learned, no masks | 4.514 / 2.193 | 3.353 (3.292-3.411) | 6.167 / 3.956 | 5.062 (5.021-5.106) |

- **Per scene, Chamfer against Deja View** (mean of seeds): dev, with masks better on 3 of 3 (bedroom -0.078,
  living room -0.012, lobby -0.610), without on 2 of 3 (bedroom +0.002). Val, with masks better on 7 of 11,
  without on 6 of 11. Largest val gain ai_004_003 (-0.34 / -0.42); largest loss the mirror bathroom ai_005_005
  (+0.26 / +0.21), whose GT depth goes through the glass.
- **Dev agrees with E36 within the seed spread:** seed 0 with masks 3.239 here against 3.197 in E36 (same seed,
  retrained; E36 was scored on a Mac CPU).
- **Conclusion:**
  - On val the learned refiner improves Chamfer by about 0.04 cm, the size of the noise floor, and by a trade:
    accuracy worse by 0.11 cm, completeness better by 0.19 cm, the mirror image of the joint refiner's trade
    (accuracy -0.36, completeness +1.44). It is better than the joint refiner on val but not clearly better than
    Deja View.
  - The masks help on dev (every seed with masks beats every seed without on Chamfer) but not on val (5.057 vs
    5.062). E36's dev gain was mostly the lobby (-0.61 of the -0.23 mean).
  - Seeds move dev Chamfer by up to 0.12 cm, so a single seed on three scenes, as in E36, cannot decide.

## E21a: BiNI's E_n weight swept and E_n off -> the same accuracy-for-completeness trade; λ_n = 1e-4 stays (10.11.2026)
- **Goal:** issue #21, the parts E42/E43 did not cover: does any weight of BiNI's E_n (λ_n, set to 1e-4 only to fix
  its units) beat the current one, and does E_n off (E42's "no normal term") hold on val.
- **Setting:** `configs/refine.yaml` (5 x 10, u with the own view), only λ_n changed: 0, 1e-5, 3e-5, 1e-4, 3e-4,
  1e-3. Rule as E40-E43: a variant passes if accuracy, completeness and Chamfer are all within the noise floor
  (0.033 / 0.037 / 0.036 cm, E40) or better on dev; a pass is checked once on val. E_n off goes to val regardless.
- **Commits:** 66c24ad (configs `experiments/configs/e21a_sweep.yaml`, `e21a_val.yaml`), run at 499bd18 (adds
  E21b's `normals` option, default `dsine`, so the same inputs). Workstation, RTX 3080, clean tree.
- **Commands:** `scene_eval --cfg experiments/configs/e21a_sweep.yaml --splits dev` (`e21a_sweep-20261110-202136`),
  `scene_eval --cfg experiments/configs/e21a_val.yaml --splits val` (`e21a_val-20261110-202253`). Stress was
  planned but not run: Hypersim's server breaks off the download of ai_015_001 at the same file on every try
  (13 of 504 files missing, `ChunkedEncodingError`), and stress decides nothing.
- **Dev aggregate, cm** (Δ against λ_n = 1e-4):

  | λ_n | accuracy | completeness | Chamfer | Δ acc / comp / Chamfer | passes |
  |---|---|---|---|---|---|
  | Deja View (no refiner) | 4.675 | 2.324 | 3.500 | | |
  | 1e-4 (current) | 4.199 | 3.658 | 3.928 | - | - |
  | 0 (E_n off) | 4.266 | 3.451 | 3.858 | +0.067 / -0.207 / -0.070 | no |
  | 1e-5 | 4.244 | 3.602 | 3.923 | +0.045 / -0.056 / -0.005 | no |
  | 3e-5 | 4.218 | 3.635 | 3.926 | +0.019 / -0.023 / -0.002 | tie |
  | 3e-4 | 4.200 | 3.623 | 3.912 | +0.001 / -0.035 / -0.016 | tie |
  | 1e-3 | 4.245 | 3.508 | 3.876 | +0.045 / -0.150 / -0.052 | no |

  E_n off reproduces E42's "no normal term" row to the last digit. Labels are unchanged (mIoU 0.360).
- **Val, E_n off against λ_n = 1e-4:** accuracy 5.701 -> 5.827 (+0.126), completeness 5.585 -> 5.014 (-0.571),
  Chamfer 5.643 -> 5.420 (-0.223); Deja View 6.061 / 4.142 / 5.102.
- **Conclusion:** no weight improves all three scores: away from 1e-4 in either direction the refiner trades
  accuracy for completeness, as E40 and E42 found for the other settings, and 3e-5 and 3e-4 tie within the floor.
  λ_n = 1e-4 stays. Turning E_n off gives the largest Chamfer gain on dev and val by the same trade, and every
  variant stays far above Deja View's Chamfer.

## E21b: MoGe-3's normals in place of DSINE's -> all three scores 0.02 cm better, inside the noise floor (10.11.2026)
- **Goal:** issues #21 and #20: is E_n limited by DSINE's normals? MoGe-3 (Wang et al. 2026, arXiv 2607.17967,
  MIT) is a newer single-image geometry model that also predicts normals.
- **Setting:** `experiments/scripts/e21b_moge_normals.py` runs MoGe-3 ViT-L (`Ruicheng/moge-3-vitl`, 3 sparse
  refinement steps) on each view with Deja View's field of view and caches its camera-frame normals as
  `moge3_<views>.npz`; `scene_eval` reads them through the new config key `normals: moge3` (default `dsine`,
  so every other config is unchanged). MoGe-3 runs in its own environment (`data/third_party/MoGe` @ 74fbce0,
  `uv sync` there): it needs Triton, so not on macOS. `to_world` fits the normals' axis signs to the depth, as for DSINE (which convention it picked
  was not logged).
  The refiner is otherwise E21a's λ_n = 1e-4. Rule as E21a.
- **Commits:** 499bd18 (script, `normals` key, `experiments/configs/e21b_moge.yaml`). RTX 3080, clean tree.
- **Commands:** `uv run --project data/third_party/MoGe python experiments/scripts/e21b_moge_normals.py <dev views>`
  (188 s for the 3 dev scenes incl. the model download and load), then `scene_eval --cfg
  experiments/configs/e21b_moge.yaml --splits dev` (`e21b_moge-20261110-202806`).
- **Dev aggregate, cm:**

  | normals | accuracy | completeness | Chamfer |
  |---|---|---|---|
  | Deja View (no refiner) | 4.675 | 2.324 | 3.500 |
  | DSINE (E21a, current) | 4.199 | 3.658 | 3.928 |
  | MoGe-3 | 4.179 | 3.635 | 3.907 |
  | Δ | -0.020 | -0.023 | -0.021 |

- **Conclusion:** MoGe-3's normals are better than DSINE's by all three scores, but each difference is inside the
  noise floor, so it does not pass and was not taken to val. Better normals are not what limits the energy: with
  either, the refiner stays at Chamfer 3.9 cm against Deja View's 3.5. MoGe-3 may still help as a depth or detail
  input elsewhere (#20).

## E20a: MoGe-3's depth fitted onto Deja View's, per view or per segment -> no variant beats Deja View (10.11.2026)
- **Goal:** issue #20: Deja View places the surface, MoGe-3 (Wang et al. 2026, arXiv 2607.17967, MIT) shapes it.
  Mircea's E35 puts about half of Deja View's in-domain error at depth edges, where single-image models are sharp.
  Do the segments decide where MoGe-3's shape may be rescaled, and does that beat Deja View?
- **Setting:** `experiments/scripts/e20a_moge_depth.py` runs MoGe-3 ViT-L (`Ruicheng/moge-3-vitl`, 3 sparse
  refinement steps, Deja View's field of view) per view, in MoGe's own environment (`data/third_party/MoGe` @
  74fbce0). `experiments/scripts/e20a_fuse.py` fits MoGe-3's inverse depth onto Deja View's, d ≈ a m + b, by least
  squares refitted once without residuals above 10 % of d, and replaces d with the fit, one (a, b) per view
  (`view`, the control), per per-view SAM 3 segment (`sam3`) or per Hypersim instance (`gt`, the upper bound);
  segments under 200 valid pixels take the view's fit. Cameras, rays and confidence stay Deja View's, and each
  variant is scored as a backbone without refinement (`experiments/configs/e20a_<variant>.yaml`).
- **Checks before running:** with MoGe-3's depth set to Deja View's the output is bit for bit Deja View's; with
  a random (a, b) per SAM 3 segment applied to Deja View's inverse depth, the per-segment fit recovers it exactly
  (relative error 0.0) and the per-view fit leaves 4.2 % (bedroom, view 0).
- **First run (f90aae2), not a result:** a fit that lands just above zero inverse depth put single pixels
  hundreds of metres away (up to 2497 m against Deja View's at most 5 m), and three val scenes scored 17 to 140 cm
  Chamfer. bfd8a10 keeps Deja View's depth wherever the fit departs from d by more than the fit's own 10 % inlier
  band; the identity check still holds. MoGe-3 was not rerun.
- **Commands** (RTX 3080, clean tree, bfd8a10): `uv run --project data/third_party/MoGe python
  experiments/scripts/e20a_moge_depth.py <dev and val views>` (196 s for 14 scenes, valid on 98.3-100 % of pixels),
  `uv run python experiments/scripts/e20a_fuse.py --splits dev val`, then `scene_eval --splits dev val` per
  variant (`e20a_view-20261110-225640`, `e20a_sam3-20261110-225744`, `e20a_gt-20261110-225847`). Deja View's
  numbers are E11a's `eval_hypersim-20261110-181811` (same machine and inputs).
- **MoGe-3 alone is less accurate than Deja View here:** per view, median relative depth error after one scale
  per scene is 1.6-4.4 % for MoGe-3 against 0.5-1.6 % for Deja View, on every dev and val scene.
- **Aggregates, cm:**

  | variant | dev acc / comp | dev Chamfer | val acc / comp | val Chamfer |
  |---|---|---|---|---|
  | Deja View | 4.675 / 2.324 | 3.500 | 6.061 / 4.142 | 5.102 |
  | MoGe-3 fitted per view | 8.276 / 2.290 | 5.283 | 8.705 / 3.679 | 6.192 |
  | MoGe-3 fitted per SAM 3 segment | 5.179 / 2.170 | 3.675 | 6.628 / 3.765 | 5.196 |
  | MoGe-3 fitted per GT instance | 5.958 / 2.110 | 4.034 | 6.696 / 3.588 | 5.142 |

  The guard keeps Deja View's depth on 0.8-2.2 % of the pixels (living room, per-segment and per-view fits).
  Labels are unchanged (mIoU 0.36 dev, 0.40 val).
- **Conclusion:**
  - No variant beats Deja View: each improves completeness a little and loses accuracy by more (per SAM 3 segment
    +0.50 / +0.57 cm on dev / val), the trade every refiner so far makes.
  - The segments matter here (per segment instead of per view: Chamfer -1.61 / -1.00 cm), but only because
    MoGe-3's own shape is worse than Deja View's on Hypersim: the per-segment fit pulls MoGe-3 back towards Deja
    View rather than adding detail Deja View lacks. GT instances are no better than SAM 3's masks on dev.
  - MoGe-2 trained on Hypersim and MoGe-3 starts from it, so this is not a domain gap in Deja View's favour.
    MoGe-3 stays useful as a normal source (E21b ties DSINE) and as a baseline, not as a depth to fuse.

## E41a: one fused voxel grid, a 3D U-Net and a per-pixel depth residual -> worse than Deja View; it overfits 35 scenes (11.11.2026)
- **Goal:** issue #41: every per-view refiner so far trades accuracy for completeness (E11a, E20a, E21a, E40-E45).
  Refine in 3D instead, where a thin object and the wall behind it are not neighbours, and test whether classes
  help there: the same refiner without classes, with SAM 3's per-view classes and with Hypersim's (upper bound).
- **Method** (`src/refinement/volume.py`, `volume-refine`): all pixels of the 20 views lifted along Deja View's
  pinhole rays, in the scene frame divided by the median depth, scatter-averaged into a dense grid of 64 cells along
  the longest side of the padded 1-99 % box: log count, mean confidence and colour, spread of the points per axis
  (the views' layering) and, with classes, the NYU40 histogram. A 3D U-Net (16 / 32 / 64 channels, GroupNorm,
  GELU); each pixel samples its features trilinearly, adds its colour, confidence, offset in the cell, ray direction
  and class embedding, and a 3-layer head predicts r, depth D exp(r). The last layer starts at zero (identity).
  Target: the GT points mapped into Deja View's frame by the Umeyama fit of Deja View's points onto them (the
  evaluation's alignment), projected onto each pixel's ray, r* = log(t* / D), |r*| <= 0.5. L1, AdamW 1e-3
  one-cycle, 3000 steps, 100k pixels per step, one scene per step, seed 0.
- **Checks before running** (`tests/test_volume.py`): untrained refiner bit for bit its input; the grid keeps every
  point and its class histogram sums to one; r* = 0 when the prediction is the GT and recovers a 5 % error on a
  patch the alignment cannot absorb; classes map to NYU40; sampling reads the cell a point lies in. Smoke run
  (20 steps): 1.2 GB peak GPU memory, held-out error 3.16 -> 3.12 %.
- **Data:** `configs/hypersim.yaml`'s 40 train scenes; ai_003_001 is skipped (Deja View's rays sit 11.0 px from
  their pinhole fit, above `MAX_PINHOLE_RESIDUAL`), the last 4 report a held-out error.
- **Commands** (RTX 3080, 8a0ea0b, clean tree): `uv run volume-refine --config configs/e41a_volume.yaml --out
  runs/volume-e41a --reserve-gb 3` (3-4 min per model), `scene_eval --splits dev val` per model with
  `experiments/configs/e41a_volume-<none|sam3|gt>.yaml` (`e41a_volume-*-20261111-00*`). The config file is now in
  the schema of E41b (every view in every step, one seed: the same setting).
- **Held-out error** (mean |r* - r| on the 4 held-out train scenes, 3.16 % before): best at step 300 for all three
  (3.11-3.12 %), then rising to 3.44 / 3.33 / 3.18 % (none / SAM 3 / GT) by step 3000: the refiner fits its 35
  scenes, not the task.
- **Chamfer, cm:** dev 4.262 / 3.604 / 3.873 (none / SAM 3 / GT), val 5.465 / 5.316 / 5.540, against Deja View's
  3.500 / 5.102; accuracy and completeness both worse than Deja View in every case.
- **Conclusion:** worse than Deja View; the model overfits a training set this small. SAM 3's classes beat no
  classes on dev and val, the first time classes help in any of our refiners, but from below Deja View. Next (E41b):
  4x the data, view subsets as augmentation, a larger grid, and the step of least held-out error kept.

## E41b: E41a scaled up (136 train scenes, 128-cell grid, 20k steps, 2 seeds) -> close to Deja View, not better on val (11.11.2026)
- **Goal:** issue #41: does E41a's refiner beat Deja View with enough data, and do the classes help then?
- **Data:** new split `train_extra` in `configs/hypersim.yaml`: the other 113 of the 153 centred scenes of
  Hypersim's official train partition, first public trajectory each (`check_splits` passes). Downloaded with 8
  workers (4043 s, one chunk retried once), prepared, Deja View, SAM 3 (`--prompts configs/prompts/nyu40.txt
  --threshold 0.3`, 1632 s for 98 scenes). `hypersim-prepare` stopped at ai_047_006 (camera orientation not a
  uniformly scaled rotation), so it and the 14 scenes after it in name order were not prepared; with ai_003_001
  and ai_016_009 (rays 11.0 and 13.5 px from their pinhole fit) 17 scenes are skipped and logged: 136 used, the
  last 8 held out.
- **Changes against E41a** (bb1af9f, 3a9b9b8, 58ecfb7; `configs/e41b_volume.yaml`): 128 cells along the longest
  side (128 x 104 x 120 on ai_001_001), U-Net 32 / 64 / 128, head 128, 20000 steps, 200k pixels per step, each
  step's grid built on the GPU from a random 8-20 of the 20 views, U-Net in bfloat16, seeds 0 and 1, scenes kept
  compact on the CPU (uint8 colour and class, float16 rays), held-out error every 1000 steps on 8 scenes and the
  state of least held-out error returned (no dev or val scene takes part).
- **Commands** (RTX 3080, 58ecfb7, clean worktree): `uv run volume-refine --config configs/e41b_volume.yaml --out
  runs/volume-e41b --reserve-gb 7` (35-40 min per model, 4.6 GB peak), then `scene_eval --splits dev val` per
  model with run-local copies of `experiments/configs/e41a_volume-<classes>.yaml`, backbone `volume-<classes>-s<seed>`
  (`runs/e41b_configs/`, `e41b_volume-*-20261111-06*`).
- **Held-out error** (8.25 % before; these 8 scenes are harder than E41a's 4): best 8.18 / 8.17 % (none, seeds 0 /
  1, steps 6000 / 4000), 7.83 / 8.17 % (SAM 3, 20000 / 2000), 8.06 / 7.94 % (GT, 13000 / 20000).
- **Chamfer, cm** (accuracy / completeness in brackets):

  | model | dev seed 0 | dev seed 1 | val seed 0 | val seed 1 |
  |---|---|---|---|---|
  | Deja View | 3.500 (4.675 / 2.324) | | 5.102 (6.061 / 4.142) | |
  | no classes | 3.515 (4.508 / 2.521) | 3.562 (4.452 / 2.671) | 5.249 (6.132 / 4.365) | 5.239 (6.027 / 4.450) |
  | SAM 3 classes | 3.330 (4.266 / 2.393) | 3.487 (4.609 / 2.365) | 5.277 (6.368 / 4.186) | 5.305 (6.296 / 4.313) |
  | GT classes | 3.491 (4.517 / 2.465) | 3.571 (4.621 / 2.521) | 5.148 (6.209 / 4.087) | 5.349 (6.570 / 4.128) |

  Labels are unchanged (mIoU 0.36 dev, 0.39-0.41 val).
- **Conclusion:**
  - The data fixes E41a's overfitting: val Chamfer 5.32-5.54 -> 5.24-5.35, within 0.05-0.25 cm of Deja View,
    and on dev SAM 3's classes beat Deja View (3.41 mean of seeds, accuracy -0.24 cm). On val no model beats
    Deja View (best 5.148, GT seed 0).
  - The classes do not separate the models beyond seed noise: seeds of one model differ by up to 0.20 cm (GT on
    val), more than the gaps between models; GT classes are no better than none on val (5.25 vs 5.24, means).
  - Not run: the 15 unprepared train scenes (fix the prepare crash first), more seeds, and the stress split.

## Plan: E27a = flag, refine, filter: refine every pixel, then drop the ones E45's best filter flags on Déjà View (issue #27, 11.11.2026)
- **Goal:** E45 found that dropping Déjà View's points by instance voxel density beats the refiner at equal
  completeness. Does refining the points a filter keeps add anything on top of the filter? The two outputs keep
  exactly the same pixels, so any difference is the refinement's.
- **Setting:** `experiments/configs/e27a.yaml`, dev (3) and val (11), RTX 3080, one `scene_eval` run. Variants:
  Déjà View, the default refiner (BiNI's E_n, E41) and the tangent term at λ_t = 4e-2 with E_n off (E43's E_t,
  100 x E43's weight), each as is and cut by `keep: 0.5`. The flag is counted on Déjà View's cloud, so the cut
  is the same for all three; the refiners still solve for every pixel.
- **Where the two values come from (not chosen in this run):** both were picked in exploratory work on the
  branch `peter/emb-weighted-tangent`, which is not merged. λ_t: a grid 4e-5 ... 1e-1 on the bedroom, then dev,
  where Chamfer is flat over 3e-2 ... 1e-1 (spread 0.009 cm). q: the lowest median Chamfer of the cut tangent
  output over q = 1, 0.95, ..., 0.3 on dev; two grid steps below it (q 0.45 on dev) Chamfer jumps. That work also
  scored val once with the same weights, so **val is not a clean check here**: this run reproduces a number that
  was already seen, though nothing was tuned on it.
- **Rule:** the tangent output cut at 0.5 against Déjà View cut at 0.5 (the same pixels), per split: better on
  all three scores beyond the floor (0.033 / 0.037 / 0.036 cm) on dev, and the same sign on val, means the
  refinement adds to the filter. BiNI cut at 0.5 is the control: if it does as well, the normal term is not what
  helps. Unfiltered, E_t 4e-2 against BiNI is the trade E21a and E43 rejected (accuracy for completeness); it is
  reported, not decided here (issue #21 owns the default).
- **Check:** `Déjà View, kept 0.5` must reproduce E45's instance voxel density curve at q 0.5 on dev.

## E27a: flag, refine, filter -> refining the kept points with E_t 4e-2 beats the filter alone on all three scores, dev and val; BiNI does not (11.11.2026)
- **Goal, setting, rule:** as in the plan above; run as planned.
- **Commits:** 5b8eccc (one-sided normals on the image border, a rendering fix found here), e8a0f65 (`keep`,
  `filters.instance_voxel_density`, test), 3e6801e (plan, config), this commit (`e27a_figures.py`, results, figure).
- **Commands** (RTX 3080, clean tree at 3e6801e): `scene_eval --cfg experiments/configs/e27a.yaml --splits dev val`
  (`e27a-20261111-112353`, 8 min); `e27a_figures.py runs/e27a-20261111-112353` -> `results/e27a_figures.txt`,
  `figures/e27a_bedroom.jpg`.
- **Checks:** Déjà View, Refined and tangent 4e-2 reproduce E41's and the exploratory run's scores to the third
  decimal on both splits; `Déjà View, kept 0.5` reproduces E45's instance voxel density at q 0.5. The label scores
  are the same for every variant (mIoU 0.360 dev, 0.397 val), as they must be.
- **Results, cm, mean / median over the split's scenes** (accuracy | completeness | Chamfer):

  | variant | dev | val |
  |---|---|---|
  | Déjà View | 4.675 / 2.520 \| 2.324 / 1.172 \| 3.500 / 1.846 | 6.061 / 4.751 \| 4.142 / 2.390 \| 5.102 / 3.971 |
  | Refined (BiNI, E41) | 4.199 / 2.208 \| 3.658 / 1.710 \| 3.928 / 1.959 | 5.701 / 4.456 \| 5.585 / 3.428 \| 5.643 / 4.214 |
  | tangent 4e-2 | 4.494 / 2.465 \| 2.359 / 1.181 \| 3.426 / 1.823 | 5.969 / 4.648 \| 3.702 / 2.635 \| 4.835 / 3.932 |
  | Déjà View, kept 0.5 | 2.998 / 1.897 \| 3.597 / 2.223 \| 3.298 / 1.727 | 3.013 / 2.466 \| 4.621 / 3.357 \| 3.817 / 2.948 |
  | Refined, kept 0.5 | 2.841 / 1.768 \| 4.331 / 2.318 \| 3.586 / 1.824 | 2.872 / 2.356 \| 5.625 / 3.787 \| 4.249 / 3.166 |
  | **tangent 4e-2, kept 0.5** | **2.918 / 1.865 \| 3.500 / 2.129 \| 3.209 / 1.674** | **2.971 / 2.424 \| 4.367 / 3.045 \| 3.669 / 2.889** |

  Each cut output minus `Déjà View, kept 0.5` (the same pixels), mean | median | scenes on which the refined one
  is better:

  | variant | dev | val |
  |---|---|---|
  | tangent 4e-2, kept 0.5 | -0.080 / -0.097 / -0.089 \| -0.070 / -0.094 / -0.082 \| 3 / 3 / 3 of 3 | -0.042 / -0.254 / -0.148 \| -0.041 / -0.112 / -0.077 \| 11 / 10 / 10 of 11 |
  | Refined, kept 0.5 | -0.157 / +0.734 / +0.288 \| -0.129 / +0.388 / +0.130 \| 3 / 0 / 1 of 3 | -0.141 / +1.004 / +0.432 \| -0.111 / +0.580 / +0.220 \| 11 / 0 / 0 of 11 |

- **Figure:** `figures/e27a_bedroom.jpg` (in `experiments/`), bedroom views 18, 0, 12, 6 pulled back, gray under a
  headlight: GT, Déjà View, Déjà View kept 0.5, tangent 4e-2 kept 0.5. Both cuts thin out the flying pixels in
  front of the bed and the dresser (views 18, 12) but leave many. Between the last two, which draw the same pixels,
  the refined walls are more closed (view 0's left wall, view 12's back wall: fewer white gaps between splats, since
  the points of different views move onto one surface) and take on DSINE's faint texture, as E43 saw. Stray points
  near furniture edges stay in both: the flag removes them or it does not, the refinement does not.
- **Conclusion:**
  - Refining the pixels the filter keeps adds to the filter: on dev beyond the floor on all three scores and on every
    scene, on val with the same sign on 10 or 11 of 11 scenes and in the median. The gain is small next to the
    filter's own (dev Chamfer 3.500 -> 3.298 from the cut, -> 3.209 with the refinement), so the filter remains most
    of the effect.
  - The normal term decides it. BiNI's refiner, cut the same way, has the best accuracy but loses completeness on
    every scene, as unfiltered (E21a, E43); E_t at 4e-2 keeps the completeness of the input and still gains
    accuracy. It is the first refined output in this log that beats E45's best filter on all three scores, here
    on the same pixels rather than at equal completeness.
  - Not changed: the default refiner (`configs/refine.yaml`) stays BiNI. Unfiltered, E_t 4e-2 against BiNI is the
    trade E21a and E43 rejected (accuracy +0.295 dev / +0.268 val, completeness -1.299 / -1.883), and issue #21 owns
    that choice. Flag, refine, filter is a `scene_eval` option, not yet a `run-pipeline` stage.
  - Val is not a clean check (see the plan): λ_t and q were chosen on the bedroom and dev, but the exploratory work
    had already scored val once with them.
- **Also tried in the exploratory work, not kept** (branch `peter/emb-weighted-tangent`, not merged; bedroom unless
  said otherwise):
  - E_t's edges weighted by the similarity of SAM 3's or Déjà View's features at both pixels: helped on the bedroom
    only; on val it cost completeness on all 11 scenes (+0.465 cm).
  - E_d weighted by the instance voxel count instead of u: no effect while E_δ is there (E_δ at 1.0 outweighs E_d).
  - Flagging on the refined cloud instead of Déjà View's: ties at q 0.75 (dev and val medians within 0.04 cm), but its
    Chamfer jumps one grid step earlier (q 0.55 dev, 0.45 val), so flagging on the input is the safer cut.
  - MoGe-3's normals in place of DSINE's in this recipe: a tie (Chamfer 1.681 against 1.674 cm at q 0.5), as E21b.
  - A simpler energy, E_d as L1 + E_mv + E_t without E_δ and u, solved to convergence (20 x 50): on its own curve
    it beats the filter from q 0.75 on, but stays 0.02 cm short of this one at equal completeness. Its weights move
    once it converges (5 x 10 acts as early stopping), and it was never run on dev or val.

## Plan: E26a = the direct competitors' geometry on ETH3D, next to ours with and without semantics (issue #26, 11.11.2026)
- **Goal:** issue #26 compares the methods that take unposed RGB and return a labeled point cloud. Our premise
  is that semantics improve the geometry, so this run scores geometry only: accuracy, completeness and Chamfer of
  every method's per-pixel points, and ours with and without the segments. ETH3D, because it is in none of the
  training sets as far as their papers and notices list them (Déjà View: paper Table 6; MUSt3R_512_cvpr, which
  PanSt3R freezes: its NOTICE), while Déjà View saw 0.69 % Hypersim and VGGT, behind IGGT, trained on Hypersim.
  ETH3D has no labels, so no label score.
- **Setting:** all 13 scenes of ETH3D's high-res training set (`configs/eth3d.yaml`: 7 indoor, 6 outdoor), 20
  views each at 504 x 336 (pipes 14, meadow 15), prepared afresh with `eth3d-prepare`; E35's caches of office and
  kicker stay apart in `data/processed/eth3d_e35`. Ours as in E35 (`experiments/configs/e26a.yaml`): Déjà View
  K = 16, SAM 3 image mode nyu40 t0.3 linked, DSINE, `configs/refine.yaml`; variants Déjà View, refined without
  segments and refined with the linked SAM 3 segments (the default). Competitors at their released checkpoints
  and their own demo settings, nothing tuned: PanSt3R v2 (the bedroom check of 11.11 found its geometry is its frozen
  MUSt3R_512_cvpr), EPS3D, IGGT, PE3R. Each gets the same 20 images; its per-pixel points come back onto our
  pixels through its own crop and resize (`e26a_pixels.py`), NaN where it cropped.
- **Scoring:** E35's: per scene one Umeyama Sim(3) on the pixel pairs, `labeled.surface_metrics` against the
  laser-scan depth; then mean and median over the scenes, indoor, outdoor and all.
- **Reading it:** refined with against without segments is the premise; refined against Déjà View is whether
  refinement helps at all out of domain (E35: completeness worse on 13 of 13); the competitors place both. One
  run, no seeds: a difference below E35's per-scene spread is not a difference.

## E26a: the competitors' geometry on ETH3D -> Déjà View leads on the mean; our refinement is worse than its input on 13 of 13 scenes and the semantics do not explain its small edge over no segments (issue #26, 11.11.2026)
- **Goal, setting:** as planned above: all 13 ETH3D high-res training scenes, 20 views at 504 x 336 (pipes 14,
  meadow 15), prepared afresh; ours (Déjà View K = 16, DSINE, SAM 3 nyu40 t0.3 linked, `configs/refine.yaml`)
  without, with and with permuted segments; PanSt3R v2, EPS3D and IGGT at their released checkpoints; PE3R on
  the cluster (it needs more than 10 GB; dropped, see below). Scored as E35: one Sim(3) per output,
  `surface_metrics`.
- **How each competitor was run** (runners in `experiments/scripts/`, each in its own environment under
  `data/third_party/`, nothing tuned):
  - PanSt3R v2 (`panst3r_predict.py`): its demo pipeline, every view a keyframe; 512 x 336 after MUSt3R's
    1.6 % centre crop, mapped back per pixel (`e26a_pixels.py`), the cropped rows without a point.
  - EPS3D (`eps3d_predict.py`): the points its encoder unprojects before voxelizing (VGGT aggregator, camera
    and depth head, weights loaded strictly from its checkpoint) at our 504 x 336, not its 256 x 256 square
    crop, which would drop a third of each view.
  - IGGT (`iggt_predict.py`): its demo's depth unprojected along its predicted cameras, at 504 x 336, its
    default. Two import shims (basicsr's renamed torchvision module; the RMSNorm its release imports but does
    not ship, replaced by `torch.nn.RMSNorm` with apex's eps 1e-5) and its aggregator's weights in bfloat16 to
    fit 10 GB (under its autocast every matmul is bfloat16 anyway).
  - PE3R (`pe3r_predict.py`, `e26a_pe3r_cluster.sh`, branch `peter/e26a-pe3r-cluster`): its own
    `get_reconstructed_scene` with the demo's defaults. Out of memory on the 3080 even with MASt3R held on the
    CPU while it segments. The cluster's 5090 would fit it, but building its environment there was estimated at 1-1.25 h on
    the login node's NFS home, so Peter dropped it; the runner and `e26a_pe3r_cluster.sh` stay for a later run.
- **Commands:** `runs/e26a/inputs2.sh <scenes>` (Déjà View, DSINE, SAM 3, PanSt3R, EPS3D, IGGT per scene,
  about 2 min per scene on the 3080), `uv run python experiments/scripts/e26a_eth3d.py --cfg
  experiments/configs/e26a.yaml` (`runs/e26a-20261111-151031`, 15 min; run on the tree of this commit before
  it was committed, hence the dirty hash), `uv run python experiments/scripts/e26a_figures.py <run>`
  (`experiments/results/e26a_eth3d.txt`, `experiments/figures/e26a_eth3d.png`).
- **Check:** Déjà View and the refiner reproduce E35 on the same scenes: completeness 7.76 -> 9.29 cm here,
  7.77 -> 9.18 there.
- **Results** (mean / median over the 13 scenes, cm; indoor and outdoor in the results file):

  | | accuracy | completeness | Chamfer |
  |---|---|---|---|
  | Déjà View | 12.22 / 8.99 | 7.76 / 5.54 | 9.99 / 7.26 |
  | Refined, no segments | 12.15 / 8.71 | 9.50 / 6.58 | 10.82 / 7.65 |
  | Refined, SAM 3 (ours) | 12.22 / 8.76 | 9.29 / 6.37 | 10.75 / 7.56 |
  | Refined, permuted SAM 3 | 12.32 / 8.80 | 8.89 / 6.18 | 10.60 / 7.49 |
  | PanSt3R v2 | 88.88 / 35.05 | 154.16 / 21.36 | 121.52 / 28.76 |
  | EPS3D | 38.24 / 28.70 | 33.57 / 17.21 | 35.90 / 24.36 |
  | IGGT | 26.56 / 13.33 | 27.09 / 8.37 | 26.83 / 10.85 |

  Ours minus each control, Chamfer: Déjà View +0.765 / +0.286 cm, better on 0 of 13 scenes; no segments
  -0.071 / -0.047, better on 11 of 13; permuted segments +0.149 / +0.161, better on 1 of 13.
- **Conclusion:** out of domain the refiner costs completeness on every scene and buys no accuracy, as E35
  found. The segments' edge over no segments is real in sign but tiny, and permuted segments do better still:
  as on Hypersim (E8b), what helps is cutting matches, not which ones the semantics cut. The premise is not
  supported on ETH3D. Among the competitors Déjà View leads on every mean; IGGT's indoor median beats it
  (Chamfer 3.81 against 6.20, better on office and terrains) but it fails outdoors (meadow 186 cm), and
  PanSt3R's frozen MUSt3R_512_cvpr breaks on 5 scenes (over 2 m) while matching Déjà View on kicker and office.
  None of them reports geometry in its paper, and none is better than its backbone here. PE3R was not run.
- **Addendum, issue #27's result on the same scenes:** E27a's flag-refine-filter (λ_t 4e-2, E_n off, q 0.5 by
  instance voxel density on Déjà View) and the filter alone, which keeps the same pixels
  (`experiments/configs/e26a_frf.yaml`, `runs/e26a_frf-20261111-162646`). Mean / median over 13 scenes, cm:
  filter alone 6.11 / 4.41, 7.86 / 5.73, 6.99 / 5.38 (accuracy, completeness, Chamfer); flag-refine-filter
  6.13 / 4.37, 7.85 / 5.66, 6.99 / 5.38. Against the filter alone its Chamfer is better on 9 of 13 scenes, by a
  median of 0.023 cm and a mean of +0.004 cm (playground +0.81): a tie, inside the noise. Against Déjà View with
  every pixel, Chamfer 9.99 -> 6.99 cm on 13 of 13 scenes, at equal completeness (7.76 -> 7.85). Out of domain,
  what carries flag-refine-filter is the filter; on Hypersim val the refinement added 0.15 cm on top (E27a).

## Plan: E25a = does Déjà View fail per class, beyond what depth, edges and viewing angle explain? (issue #25, 11.11.2026)
- **Goal:** Peter's question on #25: if semantics matter, each class should fail in its own way and by its own
  amount. Measure Déjà View's error per class, controlled for the geometry that makes any pixel hard, and say
  how each class fails. Analysis only: no change to the method, the scorer or the backbones.
- **Scenes:** Hypersim dev (3) + val (11) with its GT NYU40 classes (in domain); ETH3D's 13 scenes (out of domain)
  with SAM 3's classes, as ETH3D has no labels. Déjà View K = 16, 20 views at 504 px, `rays: pinhole`, aligned by
  the protocol's one Umeyama Sim(3) on the pixels (`align_output`). Matterport3D would give labelled out-of-domain
  GT but needs an access request: not used.
- **Per pixel:** error e = 3D distance of the aligned point to its own GT point; relative error r = e / GT depth.
  Expected relative error r̂: the scene's mean r over all its GT pixels in the same cell of edge distance (E35's
  bins: <= 1, 1-3, 3-10, > 10 px from a GT depth edge) x the scene's depth quartile x incidence angle of the GT
  surface (cos <= 0.4, 0.4-0.7, > 0.7; normals from the GT point map). These are the three geometric reasons a
  pixel is hard that have nothing to do with what the object is.
- **Per class** (a class counts in a scene with >= 2000 px there, and in the table with >= 3 scenes):
  - *Excess* = mean r / mean r̂ over the class's pixels, per scene; reported as the geometric mean over scenes with
    the scenes' range. 1 = as hard as its geometry predicts.
  - *Excess error* = the class's (e - ê) summed over its pixels, per pixel of the scene (cm): how much of the
    scene's mean error the class adds beyond its geometry, i.e. the headroom a class-aware method could address.
  - *How it fails:* share of its error within 3 px of a depth edge; cross-view gap (E35: pixel pairs the GT says
    see one surface, |predicted - GT relative gap|, median), the scattered layers of E38; placement vs shape
    (E35's per-instance scale + shift fit, share of the error it removes; object classes only, as Hypersim's
    walls, floor and ceiling are one instance per class); the scorer's per-class accuracy and completeness.
- **Do the classes differ? Rule, fixed before running:** statistic S = standard deviation over classes of the
  per-class mean log excess. Null: per scene, the GT instances' class labels are permuted among the instances
  (pixels, geometry and instance grouping stay), 1000 times. The classes differ if S lies above the null's 95th
  percentile on dev + val together. A class is named as failing in its own way only if its excess is beyond
  1.5x (or below 1/1.5) in at least 2/3 of its scenes.
- **SAM 3's classes (calibration, then ETH3D):** per-view masks of `sam3image-nyu40-t0.3`, unlinked (a pixel's
  class needs no identity), eroded by 3 px (pixels within 3 px of a change of mask id dropped, against leakage
  across edges). On Hypersim, GT classes and SAM 3 classes are compared on the same pixels (SAM 3's eroded,
  labelled ones), so only the labels differ: slope and correlation of SAM 3's per-class log excess against GT's
  say how much label errors blur the differences. ETH3D: the same table, indoor and outdoor apart, with the share
  of pixels SAM 3 labels (NYU40 prompts fit outdoor scenes badly; coarse by design).
- **Checks before reading numbers:** the alignment scale reproduces the protocol's for each scene; r̂'s cells
  reproduce the scene's mean r (excess over all pixels = 1 exactly); with a class map that is one class
  everywhere, S = 0; the permutation null on a scene with shuffled labels already gives S inside its range.
- **Figures:** per-class excess with the scenes as dots and the null band (Hypersim GT, Hypersim SAM 3, ETH3D);
  SAM 3 vs GT excess per class; bedroom `ai_006_007` views 18, 0, 12, 6: image, GT class, relative error, and
  r / r̂ on a shared log scale.
- **Reading:** classes do not differ (S inside the null) -> Déjà View's error is geometry, not category, on
  Hypersim; semantics can then only help through grouping (E8a), and per-class priors have no headroom. Classes
  differ -> the excess-error column says which classes carry it and the failure columns say how.

## E25a: Déjà View's error per class -> no class fails in its own way beyond its geometry, except glass (11.11.2026)
- **Goal, setting, rule:** as in the plan above; run as planned. The checks marked "after the first read" were
  added after the results were seen and are not part of the plan's rule.
- **Commits:** db8a898 (plan), 5eafda3 (`experiments/scripts/e25a_class_split.py`), this commit
  (`e25a_figures.py`, `e25a_checks.py`, results, figures).
- **Commands** (student cluster, CPU job 1838 on VC16, 8 cores, clean tree at 5eafda3, 3 min for both):
  `e25a_class_split.py --cfg configs/eval_hypersim.yaml --splits dev val --out runs/e25a_hypersim --scales
  runs/e8a_segments-20261110-130124/run.yaml` -> `results/e25a_hypersim.txt`; `--cfg configs/eval_eth3d.yaml
  --out runs/e25a_eth3d --maps office` -> `results/e25a_eth3d.txt`; locally `e25a_checks.py runs/e25a_hypersim`
  -> `results/e25a_checks.txt` and `e25a_figures.py runs/e25a_hypersim runs/e25a_eth3d`.
- **Checks:** the alignment scale matches E8a's Déjà View on all 14 scenes to 1e-6; per scene, Σr̂ = Σr and
  Σ(e - ê) = 0 (asserted). Not done: the plan's permutation check on a scene with shuffled labels.
- **Hypersim, GT classes, dev + val (14 scenes; mean per-pixel error 17.0 cm, median 13.2):** excess Σr / Σr̂,
  geometric mean over scenes (range), and excess error in cm per pixel of the scene:

  | class (scenes) | excess | excess error cm | error share within 3 px of an edge | cross-view gap / scene's |
  |---|---|---|---|---|
  | window (3) | 2.23 (1.38-4.99) | +0.235 | 0.11 | 2.28 |
  | otherstructure (10) | 1.17 (0.57-2.45) | +0.190 | 0.70 | 1.70 |
  | lamp (10) | 1.13 (0.48-2.46) | -0.053 | 0.85 | 1.55 |
  | wall (13) | 1.10 (0.88-2.45) | +0.290 | 0.43 | 0.78 |
  | ceiling (10) | 0.98 (0.69-1.60) | -0.202 | 0.35 | 1.07 |
  | table (10) | 0.86 (0.30-1.86) | +0.053 | 0.67 | 2.88 |
  | chair (8) | 0.77 (0.55-1.32) | -0.069 | 0.68 | 1.38 |
  | floor (13) | 0.73 (0.40-1.83) | -0.331 | 0.35 | 0.93 |
  | sofa (6) | 0.64 (0.36-1.09) | -0.168 | 0.54 | 1.13 |

  All 18 classes with >= 3 scenes, the object fit and the scorer's per-class scores are in
  `results/e25a_hypersim.txt`. Only window passes the "beyond 1.5x the same way in >= 2/3 of its scenes" rule.
  Within a class the scenes spread over a factor of 2-5 (lamp 0.48-2.46, table 0.30-1.86, door 0.10-2.12).
- **Do the classes differ (the plan's rule):** S = 0.278 against the permutation null's 95th percentile 0.276
  (median 0.191), p = 0.05: a pass by 0.002.
- **After the first read** (`e25a_checks.py`): without window and mirror (glass, whose GT is the glass surface;
  E35 set it apart) S = 0.192 against a null median 0.187, p = 0.45. Splitting the 14 scenes into random halves
  1000 times, the per-class mean log excess of one half correlates with the other's at median 0.21 (5-95%: -0.14
  to 0.67). The class differences besides glass do not reproduce across scenes.
- **SAM 3's classes on Hypersim, same pixels as GT's (3 px eroded, 67% of the GT pixels kept):** slope 0.57,
  correlation 0.57 of SAM 3's per-class mean log excess on GT's (13 classes); S = 0.249 with SAM 3's labels
  against 0.183 with GT's. The plan expected wrong labels to blur class differences; they add spread instead
  (fewer, noisier pixels per class). SAM 3's per-class spread is therefore an upper bound on the real one.
- **ETH3D, SAM 3 classes (13 scenes; 64% of indoor and 40% of outdoor GT pixels labelled after erosion):** no
  class worse than 1.5x the same way in >= 2/3 of its scenes except lamp indoors (3 scenes, 0.003 of the pixels).
  Ceiling 1.25 (12 scenes, 0.70-2.56, +0.18 cm), wall 0.97, floor 0.84, door 0.77, sofa 0.59 (3 scenes). The
  order matches Hypersim's where both have the class (wall and ceiling near 1, floor, door and sofa below).
- **Figures:** `figures/e25a_class_excess.jpg`: per class the scenes as dots and the geometric mean as a bar,
  Hypersim GT, Hypersim SAM 3, ETH3D; the permutation null with S; SAM 3 vs GT excess. `figures/e25a_bedroom.jpg`:
  bedroom views 18, 0, 12, 6, image, GT class, r, r / r̂. In r / r̂, red and blue areas span class boundaries (the
  right half of view 0's wall, the far wall of view 6), and one class can be red in one view and blue in the
  next; the mirror in view 0 is the one region that is red as a whole.
- **Conclusion:**
  - Once a pixel's edge distance, depth and viewing angle are known, its class adds almost nothing about Déjà
    View's error, in domain (Hypersim, GT classes) and, as far as SAM 3's classes show it, out of domain
    (ETH3D). The one exception is glass: windows are 2.2x worse than their geometry, and how much of that is
    Hypersim's GT definition (the glass surface) rather than Déjà View is not separable here.
  - Classes that look worse or better than their geometry (lamp, otherstructure, floor, sofa) do so in some
    scenes and not in others; the split halves do not agree on them.
  - Headroom: the largest per-class excess error is +0.29 cm (wall) against a mean error of 17.0 cm. A
    per-class correction could at most remove the positive part (about 0.8 cm summed over the classes, about
    5%), and only where it is stable, which outside glass it is not.
  - For the project: per-class priors on Déjà View's output have no headroom on these data. Semantics would have
    to help through grouping or identity (which E8a found the refiner does not use either), not through what a
    class is. Out of domain the answer is coarse: SAM 3 labels few classes outdoors, and no labelled
    out-of-domain GT was used (Matterport3D needs an access request).

## Plan: E25b = how much of Déjà View's error at depth edges could segments remove? An oracle bound (issue #25, 11.11.2026)
- **Goal:** E25a found that a pixel's class says nothing about Déjà View's error beyond its geometry, so the
  remaining route for semantics is boundaries: E35 put 42-48% of Déjà View's error on Hypersim within 3 px of a
  GT depth edge. Bound what segments could do there, and whether that needs semantics or only depth edges.
  Analysis only, on Déjà View's caches: no refiner, no change to the method, the scorer or the backbones.
- **Scenes and protocol:** Hypersim dev (3) + val (11), Déjà View K = 16, 20 views at 504 px, `rays: pinhole`;
  every output aligned onto the GT by the protocol's one Umeyama Sim(3) on its pixels (`align_output`) and
  scored by `surface_metrics`. Glass is not set apart (it is in the scores the project reports).
- **1. What the edge error is.** Band = GT pixels within 3 px of a GT depth edge (E35's edge definition:
  relative jump > 5% to a 4-neighbour). For each band pixel with a GT pixel on the other surface within 3 px
  (E35's `across_edge`), its GT depth z_g, the other surface's z_o and Déjà View's aligned depth z_p in the GT
  camera give t = (z_p - z_g) / (z_o - z_g): t <= 0 away from the other surface, 0-0.25 near its own, 0.25-0.75
  between the two (bleeding), > 0.75 on or past the other surface (the edge drawn in the wrong place). Report
  each bin's pixel share and share of the band's error. A plain offset is symmetric around t = 0, so bleeding
  and misplaced edges show as mass in the last two bins beyond what t <= 0 mirrors.
- **2. Oracle snap.** Per view and segment, pixels within BAND = 3 px of a change of segment id are snapped;
  pixels farther in are the segment's interior. A snapped pixel's inverse depth is replaced by a plane in
  inverse depth over pixel coordinates (exact for a plane seen by a pinhole), least squares on Déjà View's
  inverse depth at the same segment's interior pixels within a 31 x 31 px window around it; with fewer than
  MIN_FIT = 30 such pixels, or a non-positive fit, the pixel keeps Déjà View's depth. Pixels without a segment
  are never snapped. Constants fixed here, not tuned.
- **Segment sources** (one variant each, plus Déjà View unchanged):
  - *GT instances* (Hypersim's; walls, floors and ceilings one instance per class, E29): semantic oracle.
  - *GT depth components*: connected components of the GT pixels after removing GT depth-edge pixels, each
    edge pixel given its nearest component: knows exactly where the depth edges are, no semantics.
  - *SAM 3*: the linked masks of the protocol (`sam3image-nyu40-t0.3`, working association).
  - *Déjà View depth components*: the same components on Déjà View's own depth and edges: realistic, no
    semantics.
- **Scores per variant:** mean per-pixel error (3D distance to its own GT point) over all pixels and in the GT
  band, the share of Déjà View's band error removed, accuracy / completeness / Chamfer, snapped pixel share;
  split means over scenes, and per scene for the count of scenes a variant improves.
- **Checks:** with zero snapped pixels the output is Déjà View bit for bit; on a synthetic plane seen by a
  pinhole the fit returns its inverse depth exactly; the Déjà View row reproduces E8a's scores (same alignment).
- **Reading, fixed before running:**
  - GT instances remove little of the band error (< 10%) -> segments have no headroom at edges on these data;
    the boundary route is closed too -> bring to Umut and Robin.
  - GT instances remove much, GT depth components as much (within 0.05 cm Chamfer, or fewer than 2/3 of the
    scenes in favour of instances) -> the headroom is in knowing the edges, not in semantics: any segmentation
    that follows depth edges does it.
  - GT instances beat GT depth components on Chamfer by more than 0.05 cm in >= 2/3 of the scenes -> semantic
    grouping adds over depth edges; then SAM 3 against GT instances says whether SAM 3 is the bottleneck, and SAM
    3 against Déjà View depth components whether it beats the realistic geometric alternative.
- **Figure:** bedroom `ai_006_007` views 18, 0, 12, 6: image, Déjà View's error, and the error after each snap
  on one shared scale (cm), plus the t histogram of the band.

## E25b: segments at Déjà View's depth edges -> GT instances remove 19% of the edge error, GT depth edges alone 12%, SAM 3 4%; Chamfer barely sees it (11.11.2026)
- **Goal, setting:** as in the plan above, with two changes to the snap made after looking at results (both
  explained in their commit bodies, both applied to all four sources):
  1. *Clamp* (346307a, after the first dev run): unclamped, 0.1-0.5% of the snapped pixels moved by more than 2x
     (up to 464x), the protocol's least-squares Sim(3) followed them and every pixel of the scene moved (SAM 3:
     dev mean error 14.6 -> 18.9 cm, Chamfer 3.48 -> 6.53). A fit now stays within its interior pixels'
     inverse-depth range widened by its own ratio. As-planned dev numbers: `results/e25b_dev_unclamped.txt`
     (4e6aa0b).
  2. *Connected pieces* (f36e368, after the first val run, so val is not an untouched check for E25b): with GT
     instances three val scenes still dragged the Sim(3). Hypersim's walls are one instance, so every segment is
     now split into its 4-connected pieces per view. This did not fix them (val GT-instance Chamfer 9.95 before,
     9.93 after; `results/e25b_hypersim_unsplit.txt`). The cause, seen afterwards on ai_004_010 view 10: Déjà
     View draws the near wall's edge about 100 px left of the GT's, so the far wall's interior holds Déjà View's
     own 10 -> 2.4 m jump and the plane through it extrapolates to 32 m. No third change was made; the three
     scenes are reported both ways below.
- **Commits:** 190a125 (plan), 4e6aa0b (`experiments/scripts/e25b_boundary_oracle.py`), 346307a (clamp),
  f36e368 (pieces), this commit (`e25b_figures.py`, results, figures).
- **Commands** (student cluster, CPU job 2045, 8 cores, clean tree at f36e368, 6 min):
  `e25b_boundary_oracle.py --cfg configs/eval_hypersim.yaml --splits dev val --out runs/e25b --check
  runs/e8a_segments-20261110-130124/run.yaml` -> `results/e25b_hypersim.txt`; locally `e25b_figures.py runs/e25b`.
- **Checks:** the snap returns its input bit for bit with no segment and with one segment, and recovers a plane
  seen by a pinhole to 1e-9 next to a segment edge; the pieces of one id get separate ids (asserted on every run).
  Déjà View's alignment scale and Chamfer reproduce E8a's on all 14 scenes to the printed digits.
- **1. What Déjà View's edge error is** (14 scenes): the band (<= 3 px from a GT depth edge) is 13.5% of the GT
  pixels and carries 42% of the mean per-pixel error (52.96 cm in the band, 17.01 cm overall; E35: 42-48%). For
  the 80% of band pixels with a surface across the edge within 3 px, by t (0 = own surface, 1 = the other):

  | t | pixels | error |
  |---|---|---|
  | away (<= 0) | 0.31 | 0.09 |
  | own (0-0.25) | 0.37 | 0.09 |
  | between (0.25-0.75): bleeding | 0.16 | 0.18 |
  | other (> 0.75): the edge in the wrong place | 0.17 | 0.64 |

  Déjà View's edge error is mostly pixels that take the depth of the surface across the edge (thin parts merged
  into the background, edges drawn a few px off, in ai_004_010 about 100 px), less depth smeared between the two
  surfaces. "Away" and "own" carry similar shares, as a plain offset would.
- **2. Oracle snap,** means over scenes; removed = 1 - band error / Déjà View's, per scene. "11" = without
  ai_004_010, ai_009_009 and ai_031_004, where a snap moved the alignment scale by more than 1% (GT instances on
  all three, GT depth components on ai_009_009); this cut was chosen after looking:

  | variant | snapped px | band error cm (14 / 11) | removed (14 / 11) | Chamfer cm (14 / 11) | Chamfer lower than Déjà View's by > 0.05 cm, scenes of 14 |
  |---|---|---|---|---|---|
  | Déjà View | 0 | 52.96 / 50.25 | - | 4.758 / 3.975 | - |
  | GT instances | 0.24 | 51.84 / 41.81 | -0.07 / 0.19 | 8.523 / 3.849 | 2 |
  | GT depth components | 0.19 | 47.80 / 45.17 | 0.11 / 0.12 | 4.523 / 3.788 | 2 |
  | SAM 3 | 0.26 | 50.64 / 48.12 | 0.04 / 0.04 | 4.774 / 4.008 | 3 |
  | Déjà View depth components | 0.16 | 54.57 / 51.66 | -0.04 / -0.04 | 4.755 / 3.998 | 2 |

  Dev alone (3 scenes, none dragged): removed 0.30 / 0.12 / 0.04 / -0.04 in the order of the table, Chamfer
  3.479 -> 3.374 / 3.469 / 3.417 / 3.491. Val, 8 undragged scenes: removed 0.14 / 0.12 / 0.04 / -0.04. Accuracy
  and completeness are in `results/e25b_hypersim.txt`.
- **Pairs on the 11 undragged scenes** (band error lower in, mean a - b; Chamfer lower by > 0.05 cm in, mean
  a - b): GT instances vs GT depth components 8/11, -3.36 cm; 2/11, +0.061 cm. SAM 3 vs Déjà View depth
  components 11/11, -3.55 cm; 2/11, +0.010 cm. SAM 3 vs GT instances 1/11, +6.31 cm; 3/11, +0.159 cm.
- **Figures:** `figures/e25b_scores.jpg`: per scene each snap's change of band error and of Chamfer against
  Déjà View on one cm axis (the dragged scenes as x), and the t split above. `figures/e25b_bedroom.jpg`: bedroom
  views 18, 0, 12, 6: image, Déjà View's t in the band, the error of Déjà View and of each snap on one 0-30 cm
  scale. GT instances clear most of the bed frame and chair legs (red t: merged into the background); SAM 3
  changes little; Déjà View's own components follow its blurred edges and add errors along them.
- **Reading (the plan's rule):** GT instances remove more than 10% of the band error on the undragged scenes
  (0.19; dev 0.30, val 0.14), so segments have headroom at edges by the per-pixel error. GT depth components,
  which know only where the depth edges are, remove 0.12, and GT instances beat them on Chamfer in 2 of 14
  scenes, below the plan's 2/3: by the rule the headroom is in knowing the edges, not in semantics. By the band
  error itself semantics add some on dev (0.30 vs 0.12) and little on val (0.14 vs 0.12).
- **Conclusion:**
  - Two-thirds of Déjà View's edge error is pixels that sit on the surface across the edge; under a fifth is
    bleeding. Snapping a segment's border onto its interior removes about a fifth of the edge error with perfect
    instances, an eighth with perfect depth edges and 4% with SAM 3; Déjà View's own depth edges make it worse.
  - Chamfer does not see this error: on the 11 scenes GT instances cut the band error by 8.4 cm and Chamfer by
    0.13 cm. A point that takes the depth of the surface behind it still lies on a GT surface, so accuracy does
    not punish it, and other views cover the surface it left. For #32 (which metric decides): a boundary method
    can only show its gain in a per-pixel or own-view metric.
  - A local snap cannot repair an edge that Déjà View misplaces by more than the band, and a plane fitted across
    Déjà View's own jump inside a segment broke the protocol's alignment on 3 of 11 val scenes. A boundary
    method on Déjà View's depth has to expect the wrong edge inside a segment.
  - For the project: the boundary route is open only narrowly. Its oracle headroom is about 8% of the mean
    per-pixel error (a fifth of the 42% at edges), most of it reachable without semantics; SAM 3 reaches a fifth
    of it, and the headline metric does not see it. To bring to Umut and Robin together with E25a.

## Plan: E45a = Déjà View's error per class and at depth edges on KITTI-360, out of domain (issue #45, 11.11.2026)
Unrelated to E45 (the filters of issue #27): the id follows issue #45.
- **Goal:** E25a and E25b answered "does a class fail in its own way" and "what could segments do at depth
  edges" on Hypersim, where Déjà View is in domain, and on ETH3D with SAM 3's classes only. Umut's examples are
  outdoor objects (cars, trees, poles, a wheel with holes). Rerun both analyses unchanged on out-of-domain scenes
  that have GT classes and instances. Analysis only: no change to the method, the scorer or the backbones.
  Tanks and Temples (Truck, Caterpillar, Meeting room; laser GT, no labels) is E45b, planned after this.
- **Data:** KITTI-360 (Liao et al., TPAMI 2022; CC BY-NC-SA 3.0, nothing of it enters git), not in Déjà View's
  29-dataset mix (paper Table 6), which does hold Virtual KITTI 2 and Waymo: driving is seen, this city and
  sensor are not. Left perspective camera, rectified, 1408 x 376, f = 552.55 px, principal point (682.05,
  238.77). Sequences `2013_05_28_drive_0003_sync` (country road; car 1.6%, pole 0.7%, building 1.0%, vegetation
  43% of the labelled pixels) and `_0010_sync` (town; car 7.3%, pole 1.1%, building 16.5%, vegetation 26%),
  chosen as the two smallest downloads that together hold the classes asked about (shares counted from the 2D
  labels of all nine sequences before any image was run).
- **Scenes:** one per accumulated point-cloud window of the two sequences (7 + 17 = 24), none filtered. Views:
  20 labelled frames at a stride of 2 frames, centred in the window (in 0003 the car moves a median 1.6 m per
  frame: 3.2 m between views, 61 m per scene). Uniform sampling over a whole window, as on Hypersim, would put the views
  10-20 m apart with little overlap on anything near the road.
- **Views:** Déjà View assumes a centred principal point (E30), so each image is cropped symmetrically about
  it: 756 x 273 px, resized to 504 x 182 (scale 2/3, f = 368.4 px, 68.7° x 27.8°). The full symmetric crop
  (1364 x 274) would give 504 x 98, 7 patches high and 3 px of edge band on 98 rows; the narrower crop keeps the
  resolution. The crop drops the top 102 rows (sky, tree crowns, upper facades). Fixed before any Déjà View
  output was seen.
- **Ground truth:** the window's static accumulated points (Velodyne + SICK, with semantic and instance ids),
  projected into each view by its `cam0_to_world` pose; per pixel the nearest point whose semantic id equals
  the pixel's 2D label (changed while writing the loader, before any output: the first wording kept a pixel
  only if its nearest point of any class agreed, which empties the pixels next to object outlines, the ones
  E25b is about). This drops moving objects (their pixels would take the road or facade behind them) and
  points seen through a nearer surface of another class. Same-class see-through (vegetation behind
  vegetation) stays. The points are the window's and its two neighbours'. Classes and instances from the 2D maps
  (`semantic`, `instance` = 1000 x class + id, consistent across frames); road, sidewalk, terrain, vegetation
  and the other classes without instance ids are one instance per class, as Hypersim's walls. Sky has no GT.
- **Checks before any number is read** (the ladder, in order): GT depth edges drawn over the image for four
  views per sequence (do edges sit on object outlines; is the see-through filter enough); GT coverage per scene
  and class; the same GT point seen from two views reprojects within a pixel (poses and intrinsics); Déjà View's
  rays against their pinhole fit (`MAX_PINHOLE_RESIDUAL`); E25a's and E25b's own asserts (Σr̂ = Σr, snap identity,
  plane recovery).
- **Run:** Déjà View K = 16, `rays: pinhole`, the protocol's one Umeyama Sim(3) on the pixels, then
  `e25a_class_split.py` and `e25b_boundary_oracle.py` with their constants as committed (E25b with its clamp
  and connected pieces). SAM 3: image mode, t0.3, the working association, prompted with KITTI-360's evaluated
  class names in place of NYU40 (a new vocabulary; the only change to the inputs).
- **What is reported**, per sequence and together, next to Hypersim's numbers from E25a / E25b:
  - E25a: per class excess Σr / Σr̂, excess error in cm, share of error within 3 px of an edge, cross-view gap;
    S against the permutation null (instances' classes permuted, 1000 times); SAM 3's classes against GT's on
    the same pixels.
  - E25b: the band's pixel and error share; the t split (away / own / between / other); band error removed and
    accuracy / completeness / Chamfer for the four snaps (GT instances, GT depth components, SAM 3, Déjà View
    depth components), with the count of scenes whose alignment scale a snap moves by more than 1%.
- **Reading, fixed before running** (E25a's and E25b's rules, unchanged):
  - A class fails in its own way only if its excess is beyond 1.5x (or below 1/1.5) in at least 2/3 of its
    scenes; the classes differ if S is above the null's 95th percentile. Cars, poles or vegetation passing it
    here while nothing but glass did on Hypersim -> class matters out of domain, with the excess-error column
    as its headroom. None passing -> E25a's conclusion holds outdoors.
  - GT instances removing < 10% of the band error -> no headroom at edges; removing more but not beating GT
    depth components on Chamfer by > 0.05 cm in >= 2/3 of the scenes -> the headroom is in the edges, not in
    semantics.
- **Limits known before running:** the GT is sparse where LiDAR is (far range, above the scan lines), so depth
  edges are under-counted as on ETH3D (E35) and the band shares are lower bounds; thin structures (poles,
  fences) have few GT pixels at 504 px; moving cars are not scored; 24 scenes come from 2 drives of one day, so
  scenes are not independent and the permutation test's p is optimistic.
- **Figures:** as E25a / E25b (per-class excess with the null; snap scores and the t split), and for one town
  scene four views with image, GT class, relative error, r / r̂ and the error after each snap on shared scales.
  These are debug visuals; the presentation's bedroom stays the reference.
- **Check of the ground truth, 11.11, before any Déjà View output** (`experiments/scripts/e45a_gt_check.py` on
  all 24 prepared scenes -> `results/e45a_gt_check.txt`, `figures/e45a_gt_check.jpg`; `kitti360-prepare` takes
  8 s per scene on a laptop CPU):
  - Geometry and labels are right: classes and depth sit on their objects in the figure, moving cars have no
    depth, and view i's GT seen in view j agrees to a median 0.4-2.4% (town 78-96% of the pixels within 5%,
    country road 68-97%; the rest is vegetation seen through vegetation and points beyond 100 m).
  - Coverage is 36-59% of the pixels, as single dots near the camera (one pixel per point).
  - **The plan cannot run unchanged.** 50-79% of the GT pixels lie within 3 px of a GT depth edge by E35's
    definition (Hypersim: 13.5%). Two causes, both visible in the figure: on the road, which is 43% of the GT
    and seen at a grazing angle, neighbouring rows differ by more than 5% in depth from about 28 m on, so
    the whole far road is "edge"; and in vegetation the dots alternate between near and far leaves. The
    band is then no longer "the pixels at object outlines", and E25a's edge-distance cell and all of E25b
    lose their meaning. Also, the GT instance ids exist only where there is depth, so GT segments are dotted
    and E25b's snap would treat every pixel as a segment border.
  - One scene (`0010_0000002024_0000002177`) has a 41 m gap between two of its views, as the labelled frames
    have a hole there.
- **Two adaptations, decided with Mircea 11.11 after that check, still before any Déjà View output:**
  - *Loader:* instance ids at every pixel from the 2D labels (sky NO_MESH), whether or not a point fell
    into it; holes interpolated from the surrounding points of the same instance (`fill_holes`: Delaunay on
    the pixels with depth, barycentric in inverse depth, exact on a plane; corners within 3 px, of the hole's
    own instance and within 10% of each other). A plain splat over neighbouring pixels was not used: taking
    the nearest of several points biases a sloped surface towards the camera. Coverage 36-59% -> 54-81%;
    cross-view agreement unchanged (median 0.5-2.5%). `kitti360-prepare` now takes about 45 s per scene.
  - *Edges:* `src/analysis/edges.py`, the plane rule: two neighbours are an edge when each one's
    continuation of its own inverse-depth slope misses the other by more than 5%; with zero slopes it is E35's
    step rule. Tests: a grazing plane has no edges though the step rule marks it, a box standing on it is
    outlined and its foot is not, a pole one pixel wide is found.
  - **Result of the second check** (`results/e45a_gt_check.txt`, `figures/e45a_gt_check.jpg`, step rule and
    plane rule side by side): the plane rule removes the band along the horizon, as designed. But the
    share of GT pixels within 3 px of an edge only falls from 39-67% to 36-64% per scene. Per class, plane
    rule, all scenes: road 33%, sidewalk 38%, building 47%, pole 52%, car 73%, vegetation 78% (Hypersim, all
    classes, step rule: 13.5%). What remains is in the figure: single outlying points scattered over the road,
    cars whose points are sparse and seen through their windows, and vegetation, where neighbouring pixels
    hit near and far leaves.
  - **Reading:** this GT is right on average but not clean enough between neighbouring pixels for "the band
    at depth edges" to mean object outlines. E25b on it would measure LiDAR noise. E25a's per-class excess
    does not need clean edges (its edge-distance cell then also absorbs noisy regions), so it can run.
    **Decided with Mircea, 11.11, before any Déjà View output:** E45a = E25a only on KITTI-360; the E25b
    parts of this plan (the t split, the oracle snaps, their reading rule) are dropped here and go to Tanks
    and Temples as E45b, whose laser scans are dense (2 mm) and hold the thin parts Umut named. Not tried: removing outlying
    points from the GT (another choice stacked on the GT; it would not help cars or vegetation).
- **The plan as it stands after the split, fixed 11.11 before any Déjà View output on KITTI-360** (it
  replaces the Run / What is reported / Reading bullets above where they name E25b):
  - *Run:* Déjà View K = 16, `rays: pinhole`, one Umeyama Sim(3) on the pixels, then
    `e25a_class_split.py --cfg configs/eval_kitti360.yaml --edges plane` with its constants as committed.
    SAM 3 image mode, t0.3, per view and unlinked (as E25a), prompted with `configs/prompts/kitti360.txt`:
    the 18 Cityscapes classes KITTI-360 evaluates, without sky. Scored classes (`VOCABULARIES["kitti360"]`):
    every class KITTI-360 labels but sky, prompted or not. Groups: country (drive 0003, 7 scenes), town
    (0010, 17), all.
  - *Edge rule:* the edge-distance cell and the "error share at edges" column use the plane rule on
    KITTI-360 (`--edges plane`; the step rule marks the far road). So that the comparison with E25a is like
    for like, Hypersim dev + val is rerun under the plane rule and reported next to E25a's step-rule numbers.
    `--edges step` (the default) is E25a as committed: on the dev scenes it reprints E25a's GT-class table
    digit for digit (laptop, from db57fe0 + this change).
  - *Reported:* E25a's tables per group (excess Σr / Σr̂ with its range, scenes beyond 1.5x, excess error in
    cm, error share within 3 px of an edge, cross-view gap, object fit, accuracy / completeness), S against
    the permutation null, SAM 3's classes against GT's on the same pixels; next to Hypersim's E25a numbers.
  - *Reading:* E25a's rule, unchanged (first Reading bullet above). The second, on the band error, is E45b's.
  - *Check of the incidence-angle cell* (`experiments/scripts/e45a_incidence_check.py` on all 24 scenes ->
    `results/e45a_incidence_check.txt`, `figures/e45a_incidence_check.jpg`): the normals of the filled GT
    are sane: road, sidewalk, terrain and parking are 74-79% at |cos| <= 0.4 and 1-2% above 0.7, poles and
    traffic signs 14-20% above 0.7, buildings spread over all three bins; in the figure the road is dark,
    trunks are bright and the facade along the road lies between. 20% of the GT pixels have no normal (a
    4-neighbour without GT; their own bin, as on Hypersim), 40-47% for poles.
  - *Limit this adds:* 69% of all GT pixels fall into one angle bin and the road alone is half the GT, so
    the road's expected error is largely its own mean: its excess is near 1 by construction, and the test
    has power only for the smaller classes (car, pole, vegetation, building), which are the ones asked about.

## E45a, first run: Déjà View on KITTI-360 -> 34% relative error, set by the alignment on a few far pixels; the class tables are not read yet (issue #45, 11.11.2026)
Interim entry: the run is done and checked, the reading of its class tables waits for a decision on the
alignment (below).
- **Setting:** as planned. Commits 1497256 (script, config, prompts), 1881bdb (the tree the cluster ran).
  Inputs: job 2132 (VC15, RTX 5090, 3:16): `dejaview-predict` 86 s and `sam3-segment --threshold 0.3 --prompts
  configs/prompts/kitti360.txt` 109 s for the 24 scenes, peak GPU 3.0 GB for SAM 3. No view failed
  `MAX_PINHOLE_RESIDUAL` at 504 x 182. Analysis: job 2133 (CPU, 3:02):
  - `e25a_class_split.py --cfg configs/eval_kitti360.yaml --edges plane --maps 0010_0000000984_0000001116
    --out runs/e45a_kitti360` -> `results/e45a_kitti360.txt`
  - `e25a_class_split.py --cfg configs/eval_hypersim.yaml --splits dev val --edges plane --out
    runs/e45a_hypersim_plane --scales runs/e8a_segments-20261110-130124/run.yaml` ->
    `results/e45a_hypersim_plane.txt`
- **Checks:** Σr̂ = Σr and Σ(e - ê) = 0 asserted per scene; on Hypersim the alignment scales equal the E8a
  run's to six decimals in all 14 scenes; no non-finite Déjà View point.
- **Hypersim under the plane rule (the comparison promised in the plan):** E25a stands. Plane rule against
  E25a's step rule, dev + val, 14 scenes: GT pixels within 3 px of an edge 13.5% against 13.7%; S = 0.280
  against 0.278, the null's 95th percentile 0.274 against 0.276, p = 0.041 against 0.050; window is the one
  class beyond 1.5x in 2/3 of its scenes under both (2.24 against 2.23). The edge rule is not what E25a's
  result hangs on.
- **KITTI-360, Déjà View under the protocol's alignment:** mean error 5.96 m and mean relative error 33.9%
  over the 24 scenes (Hypersim dev + val: 0.17 m, 3.4%); country 6.18 m / 36.1%, town 5.87 m / 33.0%; per
  scene 0.33-10.78 m, 3.5-79.0%. Median cross-view gap 2.0% (Hypersim 0.6%).
- **Where it comes from** (`experiments/scripts/e45a_alignment_check.py` -> `results/e45a_alignment_check.txt`;
  added after the first read, not in the plan):
  - GT beyond 80 m is 3.3% of the GT pixels (21 of 24 scenes have GT there, reaching up to 352 m) and carries 28.8%
    of the summed error (mean 52.9 m per pixel). There Déjà View's depth is compressed: aligned depth / GT
    depth has a median of 0.41-0.93 beyond 80 m in those scenes.
  - The protocol's Sim(3) is least squares in metres, so these pixels set the scale. In 16 of the 21 scenes
    the aligned depth is then 1.02-1.37x the GT's at every range below 40 m, the same factor in each bin: the
    near scene is blown up as one piece. Pixels at 5-10 m (40% of the GT) get a mean relative error of 40.3%.
  - The three scenes whose GT ends before 80 m or barely passes it (`0003_0000000002`, `0010_0000000353`,
    `0010_0000000549`) are the three best: 3.5%, 5.5%, 13.8%.
  - Depth alone, each view scaled by its median GT / Déjà View depth: 2.1-10.5% in 21 scenes (median 6.4%);
    14.6%, 24.2% and 38.5% in `0003_0000000617`, `0003_0000000508` and `0010_0000001563`, where Déjà View
    itself fails. One Sim(3) per view does not help (13.8-56.9% in the scenes with far GT): the same far pixels
    dominate each view.
  - Déjà View's focal length is short in every scene: 275-359 px against 368 px (2-25%).
- **Reading so far:**
  - Out of domain in a way Hypersim cannot show: Déjà View compresses depth beyond about 80 m and
    under-estimates the focal length on this crop; near geometry is right to about 6% per view.
  - The class tables of `results/e45a_kitti360.txt` (no class but smallpole, 3 scenes and 0.5% of the pixels,
    beyond 1.5x in 2/3 of its scenes; S = 0.172 against a null 95th percentile of 0.195, p = 0.155; SAM 3
    against GT classes slope 1.04, correlation 0.91) are computed on errors of which the largest part is this
    global scale. The depth quartile of r̂ absorbs some of it; how much is not measured. They are reported
    for the record and not interpreted.
  - `figures/e45a_town.jpg` and `e45a_class_excess.jpg` were drawn and looked at (the r maps saturate their
    8% scale everywhere) and are not committed until the alignment is decided.
- **Open, a change to the evaluation and so not made here:** how KITTI-360 is aligned and scored. Candidates:
  drop GT beyond a fixed range before aligning and scoring; align on relative error (weights 1 / depth²);
  keep the protocol and report the far-range failure as the result. The Hypersim numbers do not depend on it.
- **Decided with Mircea, 11.11, after the first run (a change to how KITTI-360 is scored, chosen with its
  result in view):** GT beyond 80 m is dropped before the alignment and every score (`max_gt_depth: 80` in
  `configs/eval_kitti360.yaml`; 3.3% of the GT pixels). 80 m is the bin edge of the alignment check beyond
  which the depth ratio breaks; no other value was tried. Hypersim and ETH3D are untouched (no such key).
  The far-range compression stays reported above as its own finding. Check before the rerun
  (`e45a_alignment_check.py --max-gt-depth 80` -> `results/e45a_alignment_check_80m.txt`): the aligned
  depth / GT depth below 40 m is 0.93-1.06 in 20 scenes (first run: 0.96-1.37), and the relative error
  falls to 3.5-22.5% in those and 21.7% in `0010_0000001245` (ratio 0.82-0.86). The three scenes where
  Déjà View's depth alone is 13-39% off stay at 52-92%, with ratios of 0.54-0.79 at every range: they are
  Déjà View's failures, not the alignment's, and are kept.

## E45a: Déjà View's error per class on KITTI-360 -> poles fail beyond their geometry (1.80x, 14 of 18 scenes), cars and vegetation do not; the headroom is 2 cm of 296 (issue #45, 11.11.2026)
- **Goal:** E25a's question out of domain, on the outdoor objects Umut named: does a class fail in its own
  way, beyond what edge distance, depth and viewing angle predict?
- **Setting:** the plan above as it stands after the split, with GT up to 80 m (decided after the first
  run, see the interim entry). 24 KITTI-360 scenes (7 country, 17 town), 20 views at 504 x 182, Déjà View
  K = 16, one Umeyama Sim(3) on the pixels, plane-rule edges, SAM 3 image mode t0.3 with
  `configs/prompts/kitti360.txt`, per view, eroded 3 px. Commits bb2edcb (cap), 1d7266e (figures script).
- **Command:** `uv run python experiments/scripts/e25a_class_split.py --cfg configs/eval_kitti360.yaml
  --edges plane --maps 0010_0000000984_0000001116 --out runs/e45a_kitti360_80m` ->
  `results/e45a_kitti360_80m.txt`; laptop CPU, 2:17, on the caches of cluster job 2132 (the cluster's queue
  was full; the same script gives the cluster's first-run errors per scene to every printed digit on these
  caches). The run is stamped 1d7266e-dirty: the tree held one untracked results file
  (`e45a_alignment_check_80m.txt`), no code change; a run before that commit printed the same report.
  Figures: `e45a_figures.py runs/e45a_kitti360_80m runs/e45a_hypersim_plane`.
- **Checks:** Σr̂ = Σr and Σ(e - ê) = 0 asserted per scene; no non-finite point; both figures looked at.
- **Déjà View on KITTI-360, GT up to 80 m:** mean error 2.96 m, relative error mean 21.4% and median 14.4%
  over the scenes (country 26.1%, town 19.4%; Hypersim dev + val 3.4%); three scenes at 52-92% (the
  failures of the interim entry), the rest 3.5-22.5%. Median cross-view gap 1.8% (Hypersim 0.6%). GT pixels
  within 3 px of a plane-rule edge 44% (Hypersim 13.5%): the GT's noise, as the GT check said.
- **GT classes, all 24 scenes** (excess = Σr / Σr̂, geometric mean over the scenes where the class has >=
  2000 px; excess error = the class's share of the scene's mean error beyond its geometry):

  | class | scenes | px share | excess (range) | scenes beyond 1.5x | excess error cm | error share at edges |
  |---|---|---|---|---|---|---|
  | pole * | 18 | 0.010 | 1.80 (0.49-3.91) | 14 | +2.06 | 0.51 |
  | fence | 12 | 0.038 | 1.40 (0.70-2.80) | 4 | +12.82 | 0.36 |
  | car | 11 | 0.060 | 1.33 (0.82-2.45) | 3 | +2.71 | 0.73 |
  | traffic sign | 12 | 0.005 | 1.23 (0.32-7.01) | 5 | +0.23 | 0.35 |
  | vegetation | 24 | 0.158 | 1.15 (0.72-2.18) | 3 | +11.07 | 0.82 |
  | sidewalk | 24 | 0.104 | 0.99 (0.67-1.43) | 0 | -4.16 | 0.37 |
  | building | 13 | 0.113 | 0.95 (0.60-1.41) | 1 | -2.08 | 0.53 |
  | road | 24 | 0.519 | 0.87 (0.61-1.00) | 1 | -14.55 | 0.33 |

  \* = beyond 1.5x the same way in >= 2/3 of its scenes. The other six classes (smallpole, wall, terrain,
  ground, parking, box) lie at 0.76-1.26 and none is starred; all rows are in the results file.
  - Per drive: in the town pole is 1.99 (11 of 14 scenes beyond 1.5x) and starred; on the country road it
    has 4 scenes (1.28) and nothing is starred.
  - Do the classes differ? S = 0.240; null (classes permuted among instances, 1000x) median 0.160, 95th
    percentile 0.242; p = 0.054. Not above the null's 95th percentile, by 0.002.
- **SAM 3's classes:** fence is starred (1.47, 6 of 9 scenes), car 1.34, vegetation 1.15, road 0.87. Against
  the GT classes on the same pixels: slope 0.90, correlation 0.95 over 8 classes (Hypersim: 0.55, 0.56); S
  0.180 against GT's 0.196. Pole is not among SAM 3's classes with >= 3 scenes: after eroding 3 px around
  each mask border too little of a pole is left, so the one class that fails is the one SAM 3's eroded
  masks do not see.
- **Against the first run (GT uncapped):** pole 1.44 -> 1.80, 9 -> 14 scenes beyond 1.5x; p 0.155 -> 0.054.
  The class result depends on the cap, which was chosen after seeing the first run.
- **Against Hypersim (E25a, same edge rule):** there window alone is starred (2.24, GT is the glass); here
  pole alone. Hypersim's p is 0.041, here 0.054.
- **Conclusion, by the rule fixed in the plan:**
  - Poles fail in their own way out of domain: 1.8x the error their geometry predicts, in 14 of 18 scenes.
    Cars (1.33, 3 of 11 scenes) and vegetation (1.15, 3 of 24) do not pass; fences come closest (1.40).
  - The headroom is small: the poles' excess is 2.06 cm of a mean error of 296 cm (0.7%), as poles are 1%
    of the pixels. The largest excess errors, fence +12.8 cm and vegetation +11.1 cm, are 4% each.
  - The test that the classes differ at all does not pass (p = 0.054). With E25a: class identity adds
    little beyond geometry indoors and outdoors; the one outdoor exception is thin, upright structures.
  - What dominates out of domain is not per class: a scene-wide error of 14% (median) from cameras and
    scale, depth compressed beyond 80 m, a focal length 2-25% short (interim entry). The town figure shows
    it: r varies smoothly over road and views; only poles stand out in r / r̂.
- **Limits:** the cap is post hoc; 24 scenes from 2 drives of one day are not independent, so p is
  optimistic; poles have the sparsest GT (40% of their pixels without a normal, 52% within 3 px of an
  edge), so part of their excess may be the GT's; the road is half the GT and near 1 by construction;
  moving cars are not scored; per-pixel error to the pixel's own GT point under-measures missing thin parts.
- **Figures:** `figures/e45a_class_excess.jpg` (per-class excess, KITTI-360 GT and SAM 3 classes next to
  Hypersim; null; SAM 3 against GT), `figures/e45a_town.jpg` (debug visual, one town scene: image, GT
  class, r on a 0-30% scale, r / r̂).

## Plan: E45b = Déjà View's error at depth edges on Tanks and Temples, out of domain (issue #45, 11.11.2026)
Unrelated to E45 (the filters of issue #27): the id follows issue #45.
- **Goal:** E25b's two questions on scenes that hold the thin parts Umut named (stake gaps, wheels with
  holes, bars, chair legs) and are out of domain: what is Déjà View's error at depth edges (bleeding, or
  pixels on the surface across the edge), and how much of it could segments remove? KITTI-360's LiDAR was
  too noisy between neighbouring pixels for this (E45a). Analysis only: no refiner, no change to the
  method, the scorer or the backbones.
- **Data:** Tanks and Temples (Knapitsch et al., SIGGRAPH 2017), training scenes Truck, Caterpillar
  (outdoor) and Meeting room (indoor), chosen 11.11 before any was run; not in Déjà View's 29-dataset mix
  (paper Table 6). Per scene: 251 / 383 / 371 frames of 1920 x 1080, a laser scan (8.1 M / 7.3 M / 44.5 M
  points), COLMAP poses and their similarity onto the scan. No labels and no intrinsics. Nothing of the
  data or of depth derived from it enters git.
- **Views:** 20 per scene, uniform over the capture order (`select_frames`, as Hypersim and ETH3D): the
  camera circles the object, so neighbouring views are 13-20 frames apart. Looked at before anything ran:
  no person stands in any of the 60 frames (one does in others of Truck; the scan holds no people). Each
  frame is cut to 1920 x 1066.7 about its centre and resized to 504 x 280.
- **Camera, checked before the loader was written** (`experiments/scripts/e45b_focal_check.py` ->
  `results/e45b_focal_check.txt`): the scan's colours are projected into 8 views per scene with the
  dataset's poses and a candidate camera (centred principal point, focal length f, one radial
  coefficient k1), and their gradients correlated with the photo's.
  - The dataset's page suggests f = 0.7 W; that does not fit (the scan lands visibly larger than the
    truck). Without distortion the best f is 0.5975-0.6000 W in the three scenes, but the inner pixels
    want 0.6000-0.6025 and the outer 0.5950 in all three: a barrel distortion.
  - With k1 the correlation peaks at f = 0.605 W in all three scenes, k1 = -0.04 (Truck, Meeting room)
    and -0.05 (Caterpillar, flat between -0.03 and -0.06), and rises from 0.515 to 0.574 (Truck), 0.419
    to 0.433 and 0.514 to 0.546; inner and outer pixels then agree (0.6050, 0.6050-0.6075). The three
    scenes were filmed with one camera, so one camera is used: f = 0.605 W, k1 = -0.04. Ignoring k1
    would have put the scan about 5 px off the photo at the left and right border at 504 px, more than the
    3 px band this experiment is about.
  - Each frame is resampled to the pinhole of that focal length before it is cut and resized.
- **Ground truth** (`src/loaders/tanksandtemples/`): the scan rendered into each view. Per pixel the front
  is the nearest depth that 3 points share within 1% (a single stray point is no surface), and the pixel
  takes the depth of the one point nearest its centre within 1% behind that front (no mean across an
  edge). Tests: a plane returns its depth, a slanted plane the depth at the pixel centre, stray points
  are ignored, a back surface shows only where the front has no points. What the rule cannot catch: a
  surface the scanner never saw lets the points behind it show; measured below as pixels that another
  view's GT contradicts. No GT outside the scan (background, sky): those pixels are not scored.
- **Segments** (E25b's sources without the first): there are no GT instances, so the semantic oracle
  does not exist here. *GT depth components* (the GT cut at its depth edges: perfect edges, no
  semantics), *SAM 3* (image mode, t0.3, the working association; its prompt list is a new choice and is
  fixed in this entry before any mask of these scenes is seen), *Déjà View depth components*.
- **Run:** Déjà View K = 16, `rays: pinhole`, one Umeyama Sim(3) on the pixels, then
  `e25b_boundary_oracle.py --cfg configs/eval_tanksandtemples.yaml --edges plane` with E25b's constants,
  clamp and connected pieces as committed. Edges by the plane rule (`src/analysis/edges.py`, E45a) for
  the band, the surface across the edge and both depth components. So that the comparison is like for
  like, Hypersim dev + val is rerun under the plane rule and reported next to E25b's step-rule numbers;
  `--edges step` (the default) must first reprint E25b's dev table digit for digit.
- **Checks before any table is read** (the ladder): the focal sweep; the GT figure (depth and edges on
  the image's outlines); GT pixels contradicted by other views; Déjà View's rays against their pinhole
  fit (`MAX_PINHOLE_RESIDUAL`); E45a's alignment check on these scenes (aligned depth over GT depth per
  range bin, Déjà View's focal length against the GT's, error under one Sim(3) per view): the alignment
  must not be what sets the error; the snap's identity and plane recovery (asserted on every run).
- **What is reported,** per scene (three scenes carry no statistics) and as their mean, next to Hypersim:
  the band's pixel and error share; the t split (away / own / between / other); band error removed and
  accuracy / completeness / Chamfer for the three snaps; scenes whose alignment scale a snap moves by
  more than 1%.
- **Reading, fixed before running:**
  - *What the edge error is:* Hypersim's is 64% pixels on the surface across the edge and 18% bleeding.
    The same order here (other > between in all three scenes) -> out of domain the edge error is the
    same kind; otherwise the scenes are named.
  - *Headroom:* GT depth components remove < 10% of the band error in a scene -> perfect edges would not
    help there by a local snap.
  - *SAM 3:* it finds the edges out of domain if it removes at least half of what GT depth components
    remove, in at least 2 of the 3 scenes; it beats the geometric alternative if its band error is below
    Déjà View depth components' in at least 2 of 3.
  - Whether semantics add over edges (E25b's instances against depth components) cannot be answered
    here: no GT instances.
- **Limits known before running:** three scenes; no labels; the camera is ours: f and k1 are read off a
  grid (steps 0.0025 W and 0.01) and trade against each other, so the scan may still sit about a pixel
  off the photo at the border at 504 px, a third of the band; the scans cover the object and the ground
  next to it, not the background; SAM 3's prompts are a new choice.
- **Figures:** per scene four views: image, Déjà View's t in the band, the error of Déjà View and of each
  snap on one cm scale. Debug visuals; the presentation's bedroom stays the reference.
- **Check of the ground truth, 11.11, before any Déjà View output** (`experiments/scripts/e45b_gt_check.py`
  -> `results/e45b_gt_check.txt`, `figures/e45b_gt_check.jpg`; with the fitted camera;
  `tanksandtemples-prepare` takes 7 min for the three scenes on a laptop CPU):

  | scene | GT px | depth m (1%-99%) | GT px within 3 px of an edge, step / plane rule | view i's GT in view j: on j's surface (within 5%), median difference there | GT px contradicted by >= 1 / >= 2 views |
  |---|---|---|---|---|---|
  | truck | 0.53 | 1.2-7.0 | 0.30 / 0.29 | 0.45, 0.16% | 0.019 / 0.008 |
  | caterpillar | 0.37 | 1.6-9.3 | 0.33 / 0.33 | 0.40, 0.12% | 0.019 / 0.006 |
  | meetingroom | 0.95 | 1.5-13.4 | 0.42 / 0.40 | 0.64, 0.14% | 0.032 / 0.009 |

  - Poses and rendering agree across views: where view i's GT lands on the surface view j shows, the
    depths differ by a median 0.12-0.16% (KITTI-360: 0.5-2.5%). The rest of the projections are hidden in
    j, as the camera circles the object.
  - See-through is rare: 2-3% of the GT pixels have another view's GT more than 5% in front of them,
    under 1% by two views. In the figure these are scattered pixels in the plant and around stake gaps,
    and one wedge of the meeting room's ceiling seen at a grazing angle (view 000293, top left), where
    the scan is thin; no whole surface elsewhere.
  - The band is 29-40% of the GT pixels (Hypersim 13.5%, KITTI-360 44%), but here the figure shows the
    edges on outlines: stakes, wheel rims, grille, cab bars, chair backs, roof trusses. The exception is
    the meeting room's plants, where neighbouring pixels alternate between leaves (view 000001: one
    region of edges). The step and the plane rule differ little, as no large plane is seen at a grazing
    angle. Windows have no GT (glass); the ground around Caterpillar is mostly outside its scan.
- **SAM 3's prompts, fixed 12.11 with Mircea before any mask or Déjà View output of these scenes:**
  `configs/prompts/nyu40.txt` (the protocol's list) followed by `configs/prompts/vehicle_parts.txt`, 13
  prompts written from the photos: truck, bulldozer, wheel, tire, bumper, headlight, windshield, grille,
  hood, wooden board, ladder, bucket, ground. One list for all three scenes, so the meeting room gets
  the vehicle prompts too. NYU40 alone names almost nothing on a truck or a loader. The list is a
  choice of ours and is not changed after the masks are seen. Inputs: `configs/eval_tanksandtemples.yaml`
  (`sam3image-nyu40+vehicle_parts-t0.3`), run on the group's RTX 3080 server, as the cluster's queue
  held 31 pending jobs.

## E45b, first run: under the protocol's Sim(3) Déjà View's Truck is 0.54 and its Caterpillar 0.25 of the GT's size; a few outline pixels set the scale (issue #45, 12.11.2026)
Interim entry: the check that the plan puts before any table.
- **Setting:** as planned. Commit d191195 (prompts, eval config). Inputs on the group's RTX 3080 server (clean
  tree at d191195, `runs/e45b_inputs_3080.log`): `dejaview-predict` 21 s and `sam3-segment --threshold 0.3
  --prompts configs/prompts/nyu40.txt --prompts configs/prompts/vehicle_parts.txt` 69 s for the three scenes,
  peak GPU 2.9 GB. No view failed `MAX_PINHOLE_RESIDUAL`.
- **Alignment check** (`e45a_alignment_check.py --dataset tanksandtemples` ->
  `results/e45b_alignment_check.txt`), Déjà View under the protocol's one Umeyama Sim(3):

  | scene | e m, r % | one Sim(3) per view: e m, r % | depth alone r % | focal px, Déjà View / GT | aligned depth / GT depth, every range bin |
  |---|---|---|---|---|---|
  | truck | 1.13, 40.8 | 0.90, 28.6 | 6.9 | 304 / 305 | 0.53-0.54 |
  | caterpillar | 1.75, 52.1 | 1.11, 28.9 | 9.3 | 302 / 305 | 0.25 |
  | meetingroom | 0.59, 11.4 | 0.62, 13.4 | 8.4 | 302 / 305 | 0.96-0.97 |

  Unlike on KITTI-360 the focal length is right (1%) and the ratio is the same at every range: one wrong
  scale, in the two outdoor scenes.
- **Cause** (`experiments/scripts/e45b_alignment.py` -> `results/e45b_alignment.txt`; added after that
  table, not in the plan): each view's depth, scaled by its median GT / Déjà View depth, is off by more
  than 200% on 0.45% (Truck), 0.58% (Caterpillar) and 0.17% (Meeting room) of the GT pixels, always
  because Déjà View is too far, and 97%, 95% and 10% of those pixels lie within 3 px of the GT's border:
  the object's outline, where the scan ends and the photo shows the background behind it, up to about
  77 m away. The Sim(3) is least squares in metres, so these pixels set the scale (the same mechanism as
  E45a's far range, from the other side: there the GT was far, here the prediction is).
- **Decided with Mircea, 12.11, after seeing this (a change to how these scenes are scored, chosen with
  the failure in view):** the Sim(3) is fitted on the interior, the GT pixels more than 3 px from a GT
  depth edge and from the GT's border, as Déjà View's, and applied unchanged to every snap
  (`e25b_boundary_oracle.py --align interior`, commit fe94f86). The pixels under study then cannot set
  the alignment, and a snap cannot move it, which broke three val scenes in E25b. The 3 px are E25b's
  band; no other width was tried. Not chosen: a trimmed Umeyama (needs a threshold), and keeping the
  protocol (the tables would measure the scale). Under the interior fit:

  | scene | scale (protocol) | mean e cm | mean r % | median r % | mean r % over the interior |
  |---|---|---|---|---|---|
  | truck | 5.351 (2.953) | 32.0 | 8.8 | 2.6 | 3.2 |
  | caterpillar | 6.853 (1.843) | 56.7 | 13.0 | 5.9 | 6.7 |
  | meetingroom | 5.196 (5.060) | 50.9 | 9.6 | 2.3 | 3.1 |

  Hypersim's protocol numbers are untouched; the plane-rule rerun is reported under both alignments.

## E45b: Déjà View's error at depth edges on Tanks and Temples -> the same kind as on Hypersim (pixels on the surface across the edge); perfect edges remove 3-12% of it, but outdoors half the error sits on the object's outline (issue #45, 12.11.2026)
- **Goal:** E25b's questions out of domain, on scenes with thin parts: what is Déjà View's error at depth
  edges, and how much could segments remove?
- **Setting:** the plan above with the interior alignment of the interim entry. Truck, Caterpillar,
  Meeting room; 20 views at 504 x 280; GT rendered from the laser scans; Déjà View K = 16; plane-rule
  edges; SAM 3 image mode t0.3, nyu40 + vehicle_parts prompts, the working association. No GT instances
  exist, so E25b's semantic oracle is absent. Commits: 6a3a7ad (loader), 31019a2 (camera), 365e4b6 (plan),
  a658390, cc73153 and fe94f86 (scripts), this commit (`e45b_figures.py`, results, figures).
- **Commands** (laptop CPU, clean tree at fe94f86; the Tanks and Temples run takes 1.5 min):
  - `e25b_boundary_oracle.py --cfg configs/eval_tanksandtemples.yaml --edges plane --align interior --maps
    truck caterpillar meetingroom --out runs/e45b_tanksandtemples` -> `results/e45b_tanksandtemples.txt`
  - `e45b_figures.py runs/e45b_tanksandtemples` -> `results/e45b_tanksandtemples_scenes.txt` and
    `figures/e45b_{truck,caterpillar,meetingroom}.jpg`
  - `e25b_boundary_oracle.py --cfg configs/eval_hypersim.yaml --splits dev val --edges plane --out
    runs/e45b_hypersim_plane --check runs/e8a_segments-20261110-130124/run.yaml` ->
    `results/e45b_hypersim_plane.txt`; the same with `--align interior --out
    runs/e45b_hypersim_plane_interior` -> `results/e45b_hypersim_plane_interior.txt` (Hypersim's val caches
    copied from the cluster)
- **Checks:** the snap's identity and plane recovery on every run; `--edges step --align protocol` reprints
  E25b's dev table without a differing character; under the plane rule and the protocol Déjà View's scale
  and Chamfer equal E8a's on all 14 Hypersim scenes to the printed digits; the three figures looked at.
- **1. Where Déjà View's error is** (per scene; three scenes carry no statistics). Edge band = GT pixels
  within 3 px of a GT depth edge; border band = within 3 px of a pixel without GT (not in the plan: it is
  where the alignment broke, so it is reported next to the edge band; the two overlap):

  | scene | edge band: px, share of the error, error cm | border band: px, share of the error, error cm | error cm elsewhere | error cm overall |
  |---|---|---|---|---|
  | truck | 0.29, 0.46, 50.4 | 0.11, 0.49, 149.7 | 9.3 | 32.0 |
  | caterpillar | 0.33, 0.37, 64.0 | 0.15, 0.55, 206.6 | 22.1 | 56.7 |
  | meetingroom | 0.40, 0.81, 102.7 | 0.05, 0.07, 71.1 | 16.0 | 50.9 |
  | Hypersim, 14 scenes, same rule and alignment | 0.13, 0.46, 52.8 | 0.006, 0.014, 30.3 | 9.4 | 15.7 |

  Outdoors half of Déjà View's error lies in the 3 px along the object's outline against the unscanned
  background, on 11-15% of the GT pixels. In the meeting room, which has GT almost everywhere, 81% lies at
  depth edges (trusses, chairs, plants). Away from both, Truck is as good as Hypersim (9.3 cm);
  Caterpillar is not (22.1 cm): its figure shows a smooth error over whole views (the bucket in view 0,
  the rear in view 10), cameras and scale rather than edges.
- **2. What the edge error is,** by t (0 = own surface, 1 = the surface across the edge), for the 80-84% of
  the edge-band pixels with a GT pixel across the edge; pixels / error:

  | scene | away (<= 0) | own (0-0.25) | between (0.25-0.75): bleeding | other (> 0.75): the edge in the wrong place |
  |---|---|---|---|---|
  | truck | 0.19 / 0.14 | 0.47 / 0.07 | 0.10 / 0.06 | 0.25 / 0.73 |
  | caterpillar | 0.21 / 0.28 | 0.34 / 0.08 | 0.19 / 0.11 | 0.26 / 0.53 |
  | meetingroom | 0.29 / 0.09 | 0.27 / 0.04 | 0.16 / 0.09 | 0.28 / 0.78 |
  | Hypersim (plane rule, protocol) | 0.31 / 0.08 | 0.37 / 0.09 | 0.15 / 0.18 | 0.17 / 0.65 |

  A quarter of the band pixels sit on the surface across the edge (Hypersim: a sixth) and carry 53-78% of
  the band's error; bleeding is 6-11% (Hypersim 18%). In the Truck figure these are the gaps between the
  stakes, which Déjà View closes, and the wheel arches.
- **3. Oracle snaps,** share of Déjà View's error removed in the edge band / in the border band, and
  Chamfer cm (Déjà View's first):

  | scene | GT depth components | SAM 3 | Déjà View depth components | Chamfer: Déjà View, GT depth comp., SAM 3, DV depth comp. |
  |---|---|---|---|---|
  | truck | 0.09 / 0.39 | 0.03 / 0.10 | -0.01 / -0.04 | 8.17, 5.55, 7.46, 8.47 |
  | caterpillar | 0.12 / 0.44 | 0.10 / 0.23 | -0.03 / -0.04 | 18.26, 12.14, 14.69, 18.86 |
  | meetingroom | 0.03 / -0.16 | -0.01 / -0.04 | -0.02 / -0.05 | 5.60, 6.50, 5.80, 5.86 |

  Accuracy carries the Chamfer changes (Truck 14.5 -> 9.3 cm with GT depth components, 13.1 with SAM 3;
  completeness 1.81 -> 1.86 and 1.81); all rows are in `results/e45b_tanksandtemples_scenes.txt`. The
  alignment is fixed, so no snap moved a scale.
- **Hypersim under the plane rule** (dev + val, 14 scenes; E25b's step-rule numbers in brackets): band
  13.3% of the GT pixels (13.5%); t "other" 0.65 of the band's error (0.64), "between" 0.18 (0.18); removed
  -0.06 with GT instances (-0.07), 0.11 with GT depth components (0.11), 0.04 with SAM 3 (0.04), -0.02 with
  Déjà View's (-0.04). E25b does not hang on the edge rule. With the interior alignment no scene is
  dragged: GT instances remove 0.16 on all 14 scenes (dev 0.30, val 0.12; E25b reported 0.19 on its 11
  undragged), GT depth components 0.11, SAM 3 0.04; Chamfer 4.256 (Déjà View), 4.370, 4.175, 4.265.
- **Reading, by the rules fixed in the plan:**
  - *What the edge error is:* "other" exceeds "between" in all three scenes -> out of domain the edge
    error is the same kind as on Hypersim, more so: pixels that take the depth of the surface behind.
  - *Headroom at depth edges:* GT depth components remove 9% (Truck), 12% (Caterpillar) and 3% (Meeting
    room) of the edge-band error: below the plan's 10% in two of three scenes. A local snap onto the
    segment's interior does little even with perfect edges, as on Hypersim (11%).
  - *SAM 3:* it removes at least half of what GT depth components remove in one scene of three
    (Caterpillar), so by the rule it does not find the edges out of domain; its band error is below
    Déjà View depth components' in all three, so it beats the geometric alternative, by 1-8 cm.
  - Whether semantics add over edges cannot be read here (no GT instances).
- **What the plan did not foresee** (measured after the alignment failed, so post hoc): outdoors the
  largest error is not at depth edges inside the scan but on the object's outline. There, knowing the
  outline (GT depth components end where the GT ends) removes 39% and 44% of the error and a third of
  Chamfer; SAM 3's masks remove 10% and 23% (Chamfer 8.17 -> 7.46 and 18.26 -> 14.69 cm). Unlike on
  Hypersim, Chamfer sees this error: a point thrown onto a background that has no GT is far from every
  GT surface. In the meeting room the same snap makes the border worse (-16%): its border is windows
  and scan holes, not an outline.
- **Conclusion:**
  - Out of domain Déjà View's interiors hold (3% of depth on Truck and the meeting room), its edges do not:
    a quarter of the edge pixels land on the surface behind, stake gaps are closed, and an object's
    outline is drawn onto the far background, which also breaks the protocol's least-squares alignment.
  - Segments have little headroom at depth edges by a local snap (3-12% with perfect edges), in or out of
    domain. They have headroom at the outline of an object against a far background: up to 44% of the
    error there with the true outline, about a quarter to a half of that with SAM 3's masks, and this is
    visible in Chamfer. It rests on two scenes and a post hoc measure.
  - For the project (Umut's thin parts, Peter's #25 point): the out-of-domain failure to show is the
    outline and the closed gaps, not a class. A boundary method would have to move pixels across the
    edge, not smooth within a segment.
- **Limits:** three scenes, per-scene numbers only; the interior alignment and the border band are post
  hoc; the border band depends on where the scan ends, not only on the object (the ground patch around
  Truck ends in the open; in the meeting room it is windows and holes); the camera is fitted by us (about
  a pixel at the border); SAM 3's prompts are ours; no GT instances; per-pixel error to the pixel's own GT
  point, so a closed gap counts only where the scan sees through it.
- **Figures:** `figures/e45b_truck.jpg`, `e45b_caterpillar.jpg`, `e45b_meetingroom.jpg` (debug visuals):
  views 0, 5, 10, 15: image, Déjà View's t in the edge band, the error of Déjà View and of each snap on
  one 0-30 cm scale. Look at the stake gaps and the outline in Truck, the smooth error over Caterpillar's
  bucket, the trusses in the meeting room. `figures/e45b_gt_check.jpg`: the ground truth.

## Plan: E51a = token averaging within segments inside Déjà View's loop, GT instances as the oracle (issue #51, 13.11.2026)
- **Goal:** used on Déjà View's output, semantics have little headroom by oracle bounds (E25a, E25b, E45a,
  E45b; #50). Untested is semantics inside the model. The cheapest test: after some of the loop's steps,
  pull each patch token towards the mean of its segment in its view. If GT instances do not improve Déjà
  View's depth this way, token averaging is closed without SAM 3 and #53 (attention bias) is next.
- **Where it acts:** Déjà View runs one shared `LoopedAABlock` (frame attention, then global attention over
  all views) K = 16 times on tokens (V, P, C) = (20, 977, 768): per view 1 camera + 4 register tokens, then
  27 x 36 = 972 patch tokens of 14 px; then decodes depth and rays. A forward hook on that block
  (`recurrent_blocks[0]`) sees every step's output and returns the mixed tokens; dvlt is not changed.
- **Method:** after each chosen step, for every view and segment s, every patch token t of s becomes
  (1 - α) t + α mean_s(t), the mean over s's patch tokens in that view only. Camera and register tokens
  never change. A patch belongs to the id that covers more than half of its 14 x 14 pixels; a patch
  without one, or whose majority is unlabeled (-1), keeps its token. Constants fixed here, not tuned.
- **Segments:** GT instances (Hypersim's; walls, floors and ceilings one instance per class, E29).
- **Grid (coarse):** α in {0.01, 0.03, 0.1, 0.3, 1.0} x steps in {all 16, first 4, last 4} = 15 variants plus
  Déjà View unchanged. Hypersim dev (3 scenes), 20 views at 504 px, K = 16, run on the cluster's 5090; the
  baseline is rerun there in the same process (bf16 differs across GPUs, so not compared to the cache).
- **Control (wrong-scene masks):** permuting ids does not change a within-view average (it depends only on
  the grouping), so the control is the GT instances of another dev scene at the same view index (living
  room -> bedroom -> lobby -> living room): realistic shapes in the wrong place, the same α and steps as the
  best GT variant. Permuting ids per view becomes the control once the mean spans views (E51b).
- **Changed before the run (13.11, after a smoke test on the laptop's 3050, bedroom):** each patch token
  also carries its ray, so mixing bends Déjà View's rays: after the last 4 steps α = 0.1 turns them 1.6°
  (median) and leaves the ray map 6.9 px off a pinhole (1.8 px plain), α = 1 8.6° and 93 px; the
  protocol's pinhole fit fails. A mixed run's depth and confidence are therefore scored on the plain
  run's cameras and rays (the project keeps Déjà View's cameras fixed; the question is the depth), and
  how far the rays and the depth moved is reported per variant. α = 0.1 already moves the depth 1-4%
  (median; Déjà View's median 3D error on the bedroom is 1.1%), so the grid starts at 0.01 instead of 0.1.
- **Scores per variant:** each output, as the protocol, aligned onto the GT by one Umeyama Sim(3) on its
  pixels (`align_output`, pinhole rays `scene_from_geometry`) and scored by `surface_metrics`: accuracy,
  completeness, Chamfer (cm), and the mean per-pixel 3D error (cm); the forward time with and without the
  hook.
- **Checks:** α = 0 on all 16 steps (the hook runs its arithmetic) reproduces Déjà View without a hook bit
  for bit on the same GPU (depth and rays); unit tests on CPU: α = 0 returns the tokens, α = 1 gives each
  segment its mean, special and unlabeled tokens unchanged, the mean of a segment preserved for any α,
  patch ids on a synthetic mask (majority, impure, unlabeled).
- **Reading, fixed before running:**
  - The best GT variant beats Déjà View on Chamfer by less than 0.05 cm, or in fewer than 2 of 3 dev scenes
    -> token averaging does not help even with the oracle: negative result, go to #53.
  - Otherwise the gain must drop by at least half with the wrong-scene masks; then SAM 3's linked masks
    (the protocol's), then val once. If it does not drop, the gain is smoothing, not semantics.
  - The best of 15 on 3 scenes favours noise; val decides whether it holds.
- **Figure:** bedroom `ai_006_007` views 18, 0, 12, 6: image, GT instances, Déjà View's per-pixel error and
  the best variant's on one shared scale (cm).

## E51a: token averaging within GT instances inside Déjà View's loop -> worse than Déjà View for every α and step set on all 3 dev scenes; the route is closed without SAM 3 (issue #51, 13.11.2026)
- **Goal, setting:** as the plan above. GT instances, Hypersim dev (3 scenes), 20 views at 504 px, K = 16,
  α in {0.01, 0.03, 0.1, 0.3, 1} x steps in {all 16, first 4, last 4}; mixed depth and confidence scored
  on the plain run's cameras and rays, the protocol's Sim(3) and `surface_metrics`.
- **Commits, commands:** c6659e6 (the cluster run's commit; reapplied unchanged as c501e5c on the topic
  branch after its revert on main), cluster job 2620 (RTX 5090, 4.6 min):
  `uv run python experiments/scripts/e51a_token_mix.py --cfg configs/eval_hypersim.yaml --splits dev --segments gt --out runs/e51a_gt`;
  figures `uv run python experiments/scripts/e51a_figures.py runs/e51a_gt`; the camera check
  `uv run python experiments/scripts/e51a_own_cameras.py` (laptop 3050). Raw output
  `results/e51a_dev_gt.txt`, `results/e51a_own_cameras.txt`.
- **Rerun at 6cb1b0f:** after this result the hook, its tests and `predict`'s `mix` argument
  left `src` (the scripts need them); #53 brings back what it needs.
- **Checks:** α = 0 on all 16 steps reproduced Déjà View bit for bit (depth, confidence, rays) on all three
  scenes, on the 5090 and on the laptop's 3050; the unit tests of `tests/test_dejaview.py` pass. The plain
  run's Chamfer on the 5090 (3.479 cm, dev mean) is within 0.01 cm of the laptop's (3.488).
  Patches with a segment: living room 98%, bedroom 94%, lobby 84%.
- **Results** (Chamfer, cm; Déjà View: living room 1.606, bedroom 1.837, lobby 6.995, mean 3.479):
  - No variant beats Déjà View on any scene: 0 of 3 for all 15. The loss grows with α in every step set.
  - Mildest variant, α = 0.01 after the first 4 steps: +0.030, +0.024, +0.150 (mean +0.068); accuracy and
    completeness both worse in all three; depth moves 0.15% (median), rays 0.04-0.07°.
  - α = 0.01 after the last 4 steps: +0.18, +0.09, +1.47; after all 16: +0.75, +0.41, +4.02.
  - α = 0.1: +2.6 (first 4), +10.1 (last 4), +24.8 (all) on the mean; α = 1: +32 to +35.
  - Mixing bends the rays: median turn 0.04-0.3° at α = 0.01, 1.2-3.3° at α = 0.1-0.3 (bedroom), up to
    8.8° at α = 1; each patch token carries its ray, as the depth.
  - On their own cameras (first 4 steps, α <= 0.1, where the pinhole fit holds) the variants lose by about
    as much: e.g. α = 0.01: living room 1.615 vs 1.627 on the fixed cameras, bedroom 1.893 vs 1.870, lobby
    7.170 vs 7.165 (laptop; Déjà View 1.589, 1.848, 7.028). The loss is the depth's, not the pairing's.
  - Mixed runs are not bit-reproducible on one GPU: `index_add_` on CUDA sums with atomic adds in no fixed
    order, so a rerun of the camera check moved its mixed Chamfers by up to 0.03 cm (plain Déjà View and
    α = 0 stay bit-identical); far below every loss above.
  - The hook costs 1-12% of the forward pass (0.36 s plain, 0.37-0.41 s mixed, 5090).
- **Reading, by the rules fixed in the plan:** the best GT variant loses 0.068 cm Chamfer on the mean and
  wins 0 of 3 scenes -> token averaging does not help even with the oracle. Negative result; the
  wrong-scene control, SAM 3 and val are not run (the plan runs them only after a gain).
- **What the figures show:** at α = 0.1 after the last 4 steps the error grows over whole walls and the
  ceiling, each one GT instance (Hypersim's walls, floors and ceilings are one instance per class, E29):
  pulling their tokens to one mean pulls their depth towards one value, though a wall's depth varies
  across the image. At α = 0.01 the error maps look like Déjà View's; the loss there is spread thin.
- **Conclusion:**
  - Averaging Déjà View's tokens within GT instances makes its depth worse at every strength and step set
    tried. A token holds more than "what object": its own ray and depth, which differ inside a segment;
    a segment mean overwrites that, and the largest segments (walls, floors, ceilings) lose most.
  - This closes the cheapest route of #50 for within-view means. It does not test segments that keep the
    token's content: #53's attention bias changes only whom a token listens to, which is the next step.
- **Limits:** 3 dev scenes; within-view means only (a cross-view mean, E51b, was planned as a follow-up and
  is not worth running after this); one fixed patch rule (> half the patch); GT walls are one segment per
  class, so smaller structural segments might lose less, untested; mixed depth scored on Déjà View's
  cameras (checked on own cameras for the first 4 steps).
- **Figures:** `figures/e51a_scores.jpg`: Chamfer minus Déjà View's per scene over α, one panel per step
  set (above 0 = worse), and the bedroom's ray turn; `figures/e51a_bedroom.jpg`: bedroom views 18, 0, 12, 6:
  image, GT instances per patch, the per-pixel error of Déjà View, α = 0.01 first 4 and α = 0.1 last 4 on
  one 0-30 cm scale. Look at the walls and ceiling in the last row.

## Plan: E53a = attention bias within segments inside Déjà View's loop, GT instances as the oracle (issue #53, 13.11.2026)
- **Goal:** E51a overwrote what a token holds (its own ray and depth) and lost on every variant. A bias on
  the attention scores leaves the content alone and changes only whom a token listens to. If GT instances
  do not improve Déjà View's depth this way either, semantics inside the frozen model are closed without
  training, and #52 (LoRA) loses its motivation.
- **Where it acts:** the shared `LoopedAABlock` runs frame attention (per view, 977 tokens) and then global
  attention (all 20 views, 19,540 tokens) K = 16 times. Our runs never pass a `block_mask` and use the
  `auto` backend, so both go through plain `F.scaled_dot_product_attention`, not `flex_attention` as #53
  assumed. The wrapper replaces the two attention modules' `forward` with the same computation (qkv,
  q/k norm, RoPE, SDPA, projection) plus an `attn_mask`; dvlt is not changed.
- **Method:** at the chosen steps, +β on the logit of every pair of patch tokens with the same segment id
  (frame: in one view; global: across views), so each such key weighs e^β more. Camera and register tokens
  and unlabelled patches get no bias; all 16 heads the same. Patch ids as E51a (`patch_segments`: the id
  on more than half of the 14 px patch, else none).
- **Step 0, look first:** plain Déjà View on the bedroom at steps 1, 8 and 16: per head, the share of a
  labelled patch query's attention that already falls on keys of its own segment, against the share a
  uniform attention would give, frame and global; and the spread (std) of a query's logits. If the share
  is already near 1 the bias has little to move; the logit spread says whether β in the grid is small or
  large. The grid may be shifted after this, before the run, with the reason noted here.
- **Segments:** Hypersim's GT instances (scene-wide ids, so the same object has one id in every view; walls,
  floors and ceilings one instance per class, E29).
- **Grid (coarse):** β in {0.25, 0.5, 1, 2, 4} x steps in {all 16, first 4, last 4} x attention in {frame,
  global} = 30 variants. Hypersim dev (3 scenes), 20 views at 504 px, K = 16, cluster RTX 5090. Frame and
  global together only if each helps alone.
- **Baselines:** Déjà View plain; and per (steps, attention) the same hook at β = 0, which passes a zero
  mask. A float mask moves SDPA from the flash kernel to the memory-efficient one, so β = 0 is not Déjà
  View bit for bit; each variant is read against its matching β = 0 run so that only β differs, and the
  kernel's own gap to Déjà View is reported.
- **Control (wrong-scene masks):** the bias depends only on which tokens share an id, so permuting ids
  changes nothing; as E51a, the control is the GT instances of the next dev scene at the same view index,
  with the best GT variant's β, steps and attention.
- **Scores per variant:** as E51a (`fixed_cameras`, `score`): depth and confidence on the plain run's
  cameras and rays, the protocol's Sim(3) and `surface_metrics` (accuracy, completeness, Chamfer, cm), mean
  per-pixel 3D error; the rays' median turn and depth's median relative change against the plain run (does
  the bias bend the rays, as mixing did?); forward time.
- **Checks:** the hook with no chosen step (the replaced `forward` without a mask) reproduces Déjà View bit
  for bit on the same GPU (depth, confidence, rays); unit tests on CPU: the replaced `forward` equals
  dvlt's `Attention` bit for bit without a mask, the biased SDPA equals softmax(q kᵀ / √d + B) v written
  out, the bias matrix on synthetic ids (frame vs global, special tokens, unlabelled patches, padding).
- **Reading, fixed before running** (thresholds as E51a):
  - The best GT variant beats its β = 0 run on Chamfer by less than 0.05 cm on the dev mean, or in fewer
    than 2 of 3 scenes -> the attention bias does not help even with the oracle: negative result.
  - Otherwise the gain must drop by at least half with the wrong-scene masks; then SAM 3's linked masks
    (the protocol's), then val once. If it does not drop, the gain is not the segments'.
  - The best of 30 on 3 scenes favours noise; val decides whether it holds.
- **Figure:** bedroom `ai_006_007` views 18, 0, 12, 6: image, GT instances per patch, Déjà View's per-pixel
  error and the best variant's on one shared scale (cm); Chamfer minus β = 0 over β, one panel per step
  set, frame and global as two lines.
- **Step 0 done (14.11, laptop 3050, bedroom, 0c2f90f, `results/e53a_probe.txt`):**
  `uv run python experiments/scripts/e53a_attention_bias.py --cfg configs/eval_hypersim.yaml --probe --scenes ai_006_007_cam_00 --out runs/e53a_probe`.
  A labelled patch query puts 0.26-0.31 of its frame attention on its own GT segment, about what
  uniform attention would (0.26; walls, floor and ceiling are large segments), and 0.36-0.40 of its
  global attention (uniform 0.22, x1.6-1.8); single heads range 0.11-0.82. So Déjà View does not already
  attend within segments, and a bias has room to move. The logits' spread (median std per query) is
  1.1-1.5 in frame and 3.7-5.1 in global attention: β in {0.25, ..., 4} spans a fifth to three times
  a frame std and up to about one global std. The grid stays as planned.

## E53a: attention bias within GT instances inside Déjà View's loop -> no β, step set or attention gains 0.05 cm Chamfer; the best gain (0.014 cm) is the size of the kernel's own noise (issue #53, 14.11.2026)
- **Goal, setting:** as the plan above. GT instances, Hypersim dev (3 scenes), 20 views at 504 px, K = 16;
  +β on same-instance patch pairs in frame or global attention, β in {0.25, 0.5, 1, 2, 4} x steps in {all
  16, first 4, last 4}, each against the same hook at β = 0; biased depth and confidence scored on the
  plain run's cameras and rays, the protocol's Sim(3) and `surface_metrics`.
- **Commits, commands:** 59e8e6f (hook 0c2f90f), cluster job 2722 (RTX 5090, 6.5 min):
  `uv run python experiments/scripts/e53a_attention_bias.py --cfg configs/eval_hypersim.yaml --splits dev --segments gt --out runs/e53a_gt`;
  figures `uv run python experiments/scripts/e53a_figures.py runs/e53a_gt`. Raw output
  `results/e53a_dev_gt.txt` (mean table and every variant per scene), step 0 `results/e53a_probe.txt`.
- **Rerun at 246d203:** after this result the hook (`AttentionBias`), its tests and `predict`'s `bias`
  argument left `src` (the scripts need them).
- **Checks:** the hook without a mask reproduced Déjà View bit for bit (depth, confidence, rays) on all
  three scenes; unit tests in `tests/test_dejaview.py` pass. The plain run's Chamfer (3.479 cm, dev mean)
  equals E51a's on the same GPU. Patches with a segment: living room 98%, bedroom 94%, lobby 84%.
- **The kernel's own noise:** β = 0 (a zero mask: SDPA's memory-efficient kernel instead of flash) moves
  Chamfer by -0.002 to +0.006 cm on the dev mean and by up to 0.02 cm on one scene (bedroom 1.837 ->
  1.854, global all steps), the rays by 0.03-0.04° (median). Gains below about 0.02 cm are not readable.
- **Results** (Chamfer, cm; Déjà View: living room 1.606, bedroom 1.837, lobby 6.995, mean 3.479; change
  against the matching β = 0 run, dev mean, wins of 3):
  - Best: global attention, first 4 steps, β = 0.25: -0.014 (3 of 3: 1.603, 1.823, 6.989 against 1.604,
    1.843, 7.008); frame, first 4, β = 0.25: -0.013 (2 of 3). Against plain Déjà View both are -0.008.
    Every other variant is worse on the mean.
  - Biasing the first 4 steps is mildest: β = 1 +0.05 (global) / +0.10 (frame), β = 4 +0.80 / +1.25.
  - The last 4 steps: β = 0.25 +0.04 (frame) / +0.08 (global), β = 4 +2.8 / +3.7.
  - All 16 steps: global +0.03 (β = 0.25) to +3.6 (β = 4); frame +0.39 to +14.9, the worst of the grid.
  - The bias bends the rays, as mixing did in E51a, though less: median turn (dev mean) 0.04-0.18° on the
    first 4 steps, up to 2.0° on the last 4 and 4.0° on all 16 (frame, β = 4; bedroom 6.0°).
  - Cost: global bias on all steps 0.47 s against 0.36 s plain, 9.8 GB peak against 9.5 (5090); frame
    bias costs nothing measurable.
- **Reading, by the rules fixed in the plan:** the best GT variant gains 0.014 cm Chamfer on its β = 0 run,
  below the 0.05 cm bar, and within the 0.02 cm the kernel switch alone moves a scene -> the attention
  bias does not help even with the oracle. Negative result; the wrong-scene control, SAM 3 and val are not
  run (the plan runs them only after a gain).
- **What the figures show:** the best variant's error maps look like Déjà View's. Frame attention biased
  on the last 4 steps at β = 1 is not worse everywhere: on the bedroom (views 18, 0, 12, 6) the ceiling's
  mean error falls from 8.2 to 6.6 cm and the wall's barely moves (6.6 -> 6.9), but the floor's rises from
  11.6 to 13.9 and the largest furniture instances' by 0.9 to 2.7 cm; overall 8.8 -> 10.0 cm.
- **Conclusion:**
  - Making Déjà View's tokens listen more to their own GT instance, within a view or across views, does
    not improve its depth at any strength or step set tried; mild biases early in the loop change nothing
    measurable, stronger or later ones hurt. Step 0 showed Déjà View does not already attend within
    segments (frame share at uniform), so the bias did change the attention; the model does not use it.
  - With E51a (token averaging) this closes both training-free routes of #50 with GT segments as the
    oracle. It does not bound a trained model: a LoRA or adapter (#52) can learn to read segments, the
    frozen model cannot. #52 names two reasons to run: training-free headroom with GT segments (not met)
    or ruling out that Déjà View cannot use segments without training, which is now the open question.
    The plan above said #52 "loses its motivation"; that was too strong.
- **Limits:** 3 dev scenes; one β for all 16 heads, although heads differ widely (own-segment share
  0.11-0.82); a positive bias only (no repulsion between segments); GT walls, floors and ceilings one
  segment per class (the ceiling gained under a strong frame bias, which a finer split might keep);
  biased depth scored on Déjà View's cameras.
- **Figures:** `figures/e53a_scores.jpg`: Chamfer minus β = 0's over β per step set, frame and global (mean
  line, one dot per scene), and the bedroom's ray turn; `figures/e53a_bedroom.jpg`: bedroom views 18, 0, 12,
  6: image, GT instances per patch, the per-pixel error of Déjà View, of global first 4 β = 0.25 and of
  frame last 4 β = 1 on one 0-30 cm scale. Look at the ceiling against the bed and floor in the last row.

## Plan: Solv3R as a backbone (no issue, 14.11.2026)
- **Goal:** Solv3R (Kwak et al., arXiv 2611.02857) is Déjà View fine-tuned into a stationary looped solver and
  beats it on ETH3D poses (AUC@3 70.4 vs 66.0). Is its geometry better than Déjà View's on Hypersim dev and val,
  and do the filter and flag-refine-filter still help on top of it?
- **Setting:** `src/backbone/solv3r.py` (`solv3r-predict`), released weights (jgkwak/solv3r, NVIDIA License,
  research only), code pinned at c3851d3 in the `solv3r` dependency group. Same views as Déjà View (20 at 504 px),
  cameras fitted to the rays by DVLT's post-processing as for Déjà View. Two solves: 16 calls (`solv3r_c16`) and
  adaptive at 0.02, 4-64 calls (`solv3r_a0.02`), the paper's two settings. DSINE on each backbone's intrinsics.
- **Variants** (`experiments/configs/solv3r.yaml`, weights as chosen for Déjà View in E27a, none retuned):
  Solv3R as is, kept 0.5 by instance voxel density, flag-refine-filter (λ_t = 4e-2, E_n off, q = 0.5).
- **Baselines:** Déjà View dev 3.500 / val 5.102 Chamfer; its filter 3.30 / 3.82, flag-refine-filter 3.21 / 3.67.
- **Commands:** 3d-cluster (RTX 5090): `solv3r-predict <views> [--adaptive 0.02]`, `dsine-predict <views>
  --backbone <b> --name dsine-<b>`, `scene_eval --cfg experiments/configs/solv3r.yaml --splits dev val` per solve.
- **Checks:** tests of the solve (zero calls return the start state, fixed calls reach the known fixed point of a
  contraction, the adaptive stop at the first change under tau, its call bounds); time per scene.

## Solv3R as a backbone -> worse than Deja View on Hypersim, dev and val; the filters help it as much (15.11.2026)
- **Commits:** 66844ff, 1871e30. Run on the 3080 (`~/pouya/solv3r_run.sh`, the cluster was full), same
  views, SAM 3 masks and evaluation as E27a. Times for 14 scenes: predict 93 s (16 calls) / 84 s (adaptive),
  DSINE 53 s, scoring 4.5 min. `experiments/results/solv3r.txt`.
- **Results** (Chamfer in cm; Deja View from E27a):

  | variant | Deja View dev / val | Solv3R 16 calls dev / val | Solv3R adaptive 0.02 dev / val |
  |---|---|---|---|
  | as is | 3.500 / 5.102 | 4.058 / 5.735 | 4.082 / 5.606 |
  | kept 0.5 (instance voxel density) | 3.30 / 3.82 | 3.626 / 4.241 | 3.624 / 4.196 |
  | flag-refine-filter | 3.21 / 3.67 | 3.563 / 4.097 | 3.557 / 4.049 |

- Solv3R is worse on 12 of 14 scenes. The adaptive solve stops after 15 to 25 calls (mean 17.9) and scores
  like 16 calls. PQ is unchanged (0.296 / 0.258 vs 0.292 / 0.258), as the masks are the same.
- **Checks:** the inputs are Solv3R's own preprocessing (RGB in [0, 1], long side 504). The loss is in the
  network's output, not in fitting cameras to the rays: per-view depth after a per-view scale is off by 1.56 %
  (Deja View 0.86 %), camera centres after a Sim(3) by 1.83 cm (0.97 cm), mean over the 14 scenes.
- **Conclusion:** Solv3R's pose gain on ETH3D does not carry over to Hypersim's geometry; Deja View stays the
  backbone. The filter and flag-refine-filter gain about as much on Solv3R (dev -0.43 / -0.50) as on Deja View
  (-0.20 / -0.29), so they are not tied to one backbone.
## Plan: E56a = each view's depth refined from the other views' points projected into it (issue #56, 14.11.2026)
- **Goal:** issue #56: E36's 2D refiner never sees the other views, E41b's volume sees them only at 5 cm cells.
  Does a 2D U-Net that gets, per pixel, the other views' depths along its ray beat Deja View on dev and val, and
  do classes help it?
- **Setting** (`src/refinement/reproject.py`, `configs/e56a_reproject.yaml`): E41b's data, split and protocol (all
  centred train scenes, last 8 held out, state of least held-out error kept, seeds 0 and 1, 20000 steps). Per pixel:
  RGB, log depth, log confidence, a 33-bin soft histogram of log(z / D) over [-0.5, 0.5] of the other views'
  points that land on it, counts in front of and behind that range, their mean log confidence; with classes a
  second histogram of the votes of the pixel's own NYU40 class. E36's U-Net (width 32), 4 target views per step,
  votes from a random 8-20 views. Target and loss: E41b's r* along the pinhole ray, E36's L1 with gradient term.
- **Models:** no classes, SAM 3 classes (per-view masks), GT classes. A permuted-classes control only if SAM 3 or
  GT beats no classes beyond seed noise.
- **Commands:** one SLURM task per model and seed on the 3d-cluster (RTX 5090), `uv run reproject-refine --out
  runs/reproject-e56a --only <model>-s<seed>`, then `scene_eval --splits dev val` per model as in E41b.
- **Checks:** tests (soft histogram keeps mass and mean, consistent views vote for 0 and a view 0.2 farther for
  0.2, votes invariant to scene scale, same-class votes a subset, untrained refiner returns its input bit for bit);
  one scene's step time and memory before the full run.
- **Baselines:** Deja View (dev 3.500, val 5.102), E36 / E11a, E41b (val 5.24 / 5.29 / 5.25).

## E56a: each view refined from the other views' points -> no gain over Deja View (14.11.2026)
- **Commits:** 83ea14c, 8bc6276; 3d-cluster (RTX 5090), one SLURM task per model and seed, about 15 min of
  training each (20k steps), 133 train rooms (15 new ones not yet staged, 2 without a pinhole fit).
- **Results** (Chamfer cm, seeds 0 / 1; Deja View dev 3.500, val 5.102):

  | model | dev | val |
  |---|---|---|
  | no classes | 3.495 / 3.480 | 5.353 / 5.448 |
  | SAM 3 classes | 3.572 / 3.490 | 5.561 / 5.261 |
  | GT classes | 3.552 / 3.522 | 5.505 / 5.202 |

- Held-out error barely moves (0.0824 -> 0.080 at best); seeds differ by up to 0.3 cm, more than the classes.
- **Checks:** unit tests (soft histogram mass and mean, consistent views vote 0, scale invariance, identity at
  zero weights). On three rooms the mean vote correlates with r* at 0.13 to 0.16, and moving each pixel by
  alpha times its mean vote helps at no alpha in {0.1, ..., 1}.
- **Conclusion:** Deja View's views already agree; their remaining error is shared across views, so the other
  views' points cannot reveal it. Same wall as E11a (2D) and E41b (3D). Issue #56 closed.
## Plan: E57a = a keep head on the 3D refiner: a learned filter against instance voxel density (issue #57, 14.11.2026)
- **Goal:** issue #57: E41b's refiner can only move points, while the voxel density filter (E45) and
  flag-refine-filter (E27a, val Chamfer 3.82 and 3.67 at kept 0.5) win by dropping them. Does a learned keep score,
  from the same fused grid, rank Deja View's pixels better than instance voxel density at the same kept fraction,
  and do SAM 3's classes help it?
- **Setting** (`src/refinement/volume.py`, `configs/e57a_volume.yaml`, `configs/e57a_volume_keep.yaml`): E41b's
  data, grid, U-Net, steps, view subsets and seeds; the head gets a second output, a keep logit, trained by binary
  cross-entropy on whether each pixel's remaining error |r* - r| is in the lower half of the step's pixels
  (pixels with ground truth but no valid r* count as errors). Two head sets: keep head alone (the depth stays
  Deja View's) and keep head plus residual. The kept step is the one with the least mean error of the held-out
  pixels its logit keeps (top half). No dev or val scene takes part.
- **Models:** keep alone and keep plus residual, each without and with SAM 3 classes, seeds 0 and 1 (8 models).
  GT classes and a permuted control only if SAM 3 beats no classes beyond seed noise.
- **Evaluation** (`experiments/configs/e57a_volume.yaml`, `scene_eval`'s new `keep_by`): each model's output
  whole, kept 0.5 by instance voxel density and kept 0.5 by its keep logit, on dev and val. For the keep-alone
  models the output is Deja View's depth, so the density cut is E27a's filter on the same pixels.
- **Commands:** one SLURM task per model on the 3d-cluster (RTX 5090), `uv run volume-refine --config <config>
  --out runs/volume-e57a`, then `scene_eval --splits dev val` per model.
- **Checks:** tests (untrained keep head leaves the depth bit for bit and scores all pixels alike, keep labels mark
  the least errors and count invalid pixels as errors, E41b's config has no keep head); E41b's configs train and
  predict exactly as before (the keep head is off unless `keep` is set).
- **Baselines:** Deja View (dev 3.500, val 5.102), instance voxel density kept 0.5 and flag-refine-filter (E27a),
  E41b.

## E57a: a keep head on the 3D refiner -> its cut loses badly to instance voxel density (14.11.2026)
- **Commits:** 1dd9d4d, 4c95ad7; 3d-cluster (RTX 5090), one SLURM task per model and seed, about 25 min each.
- **Results** (val Chamfer cm, range over seeds 0, 1 and none / SAM 3 classes, keep 0.5):

  | cut | keep only | keep + residual |
  |---|---|---|
  | no cut | 5.102 (Deja View) | 5.211 to 5.694 |
  | instance voxel density | 3.817 | 3.991 to 4.443 |
  | keep head | 9.493 to 12.367 | 11.527 to 14.131 |

- The head ranks pixels sensibly (held-out kept error 0.115 -> 0.102; dev accuracy after the cut 2.80 vs 3.00
  cm by density) but deletes whole regions: completeness 15 to 39 cm. In ai_010_007, 8 of the 14 GT instances
  above 0.5 % of the pixels keep under 5 % of them.
- keep + residual picks its state by kept error alone, so the residual is unchecked: dev Chamfer 3.49 to 4.97.
- **Checks:** residuals are applied as in E41b (|r| p99 under 5 %); the logits are not inverted.

## Plan and result: E57b = E57a's logits cut per instance or with density -> still far behind (14.11.2026)
- Planned after E57a, before running: does the per-pixel cut alone cause E57a's holes?
- **Setting:** no retraining; `scene_eval` on E57a's saved logits: `keep_per: instance` cuts every linked SAM 3
  instance to q, `keep_with: density` ranks by `filters.both` of the logit and density. Density per instance
  as the baseline. Commit e015eb0, 3d-cluster job 2740, about 4 min per model.
- **Results** (val Chamfer cm, ranges as above):

  | cut | keep only | keep + residual |
  |---|---|---|
  | density, per instance | 5.190 | 5.312 to 5.774 |
  | keep head, per instance | 6.706 to 7.310 | 6.969 to 7.235 |
  | keep head and density | 6.229 to 7.521 | 6.652 to 8.005 |

- **Conclusion:** at equal kept fraction a learned keep score loses to instance voxel density in every form.
  Density wins by removing redundant points where the cloud is dense, which costs little completeness;
  forced to cut every instance equally it falls to 5.19. An error label is spatially smooth, so cutting by it
  opens holes. Issue #57 closed.
## Plan: E58a = densify after filtering: the flagged pixels filled from their view's kept pixels (issue #58, 14.11.2026)
- **Goal:** issue #58: the instance voxel density filter buys accuracy with completeness (val 6.061 -> 3.013 and
  4.142 -> 4.621 cm at q 0.5, E27a). Does giving the flagged pixels a new depth from the kept ones, instead of
  dropping them, keep the accuracy and win the completeness back, and does filling within an instance beat plain?
- **Setting** (`filters.fill_harmonic`, `scene_eval`'s `fill`, `experiments/configs/e58a.yaml`): E27a's flag (q 0.5,
  instance voxel density on Déjà View's cloud, linked SAM 3 masks). Per view the flagged pixels' inverse depth
  solves the weighted Laplace equation with the kept pixels fixed (exact for planes): `plain` (equal weights),
  `rgb` (Levin et al.'s colour affinity, as NYU Depth v2's fill), `sam3` and `gt` (edges between instances cut to
  1e-3). Flag, refine, fill: E27a's tangent 4e-2 refiner, then the SAM 3 fill from the refined kept pixels.
  Nothing tuned; every pixel keeps a point, so all fills score the full pixel set like Déjà View.
- **Commands:** `scene_eval --cfg experiments/configs/e58a.yaml --splits dev val` (Mac, CPU).
- **Checks:** tests (no flag returns the input bit for bit, a planar hole filled to 1e-4, a hole on an instance's
  border filled from its own plane); `Déjà View, kept 0.5` and `tangent 4e-2, kept 0.5` reproduce E27a.
- **Baselines:** Déjà View (val 5.102, completeness 4.142), the filter (3.817 / 4.621), flag-refine-filter (3.669).

## E58a: densify after filtering -> every fill is worse than the filter alone and than Deja View (14.11.2026)
- **Commits:** 4bbe699; 3080 server, dev and val in one run, 927 s.
- **Results** (Chamfer cm, keep 0.5):

  | variant | dev | val |
  |---|---|---|
  | Deja View | 3.500 | 5.102 |
  | density filter | 3.298 | 3.817 |
  | flag-refine-filter (tangent 4e-2) | 3.209 | 3.669 |
  | filter + fill plain | 5.227 | 6.697 |
  | filter + fill RGB | 4.396 | 6.119 |
  | filter + fill SAM 3 | 4.009 | 5.808 |
  | filter + fill GT instances | 4.213 | 6.037 |
  | flag-refine-filter + fill SAM 3 | 3.962 | 5.720 |

- **Checks** (4 views per scene, dev + val, from the run's saved depths): the fill recovers GT depth (half of
  the GT pixels dropped at random: mean |log error| 0.88 %; the density drop within GT instances 1.02 %, plain
  2.78 %). On the dropped pixels Deja View's own depth (3.20 %) beats every fill, even within GT instances
  (3.43 %; SAM 3 3.61 %, RGB 3.84 %, plain 4.50 %).

## Plan and result: E58b = the fill after a lighter filter -> still worse (14.11.2026)
- Planned after E58a, before running: are E58a's holes just too large to fill?
- **Setting:** `experiments/configs/e58b.yaml`, keep 0.9 and 0.75, plain and SAM 3 fill; commit 648e647, 3080.
- **Results** (Chamfer cm, dev / val):

  | kept | filter alone | + fill plain | + fill SAM 3 |
  |---|---|---|---|
  | 0.9 | 3.430 / 4.932 | 3.536 / 5.150 | 3.520 / 5.139 |
  | 0.75 | 3.293 / 4.576 | 3.654 / 5.257 | 3.565 / 5.209 |

- The fill does not even buy back completeness (val 4.11 to 4.14 cm filter alone, 4.15 to 4.19 with a fill).
- **Conclusion:** a fill from the kept neighbours cannot beat what the backbone already predicts at the flagged
  pixels; filling within an instance helps the fill but not enough. Dropping (or refining) them stays better.
  Issue #58 closed.
## Plan: E59a = where SAM 3 + linking lose panoptic quality (issue #59, 14.11.2026)
- **Goal:** issue #59's first step: split the PQ that SAM 3 + linking lose on Hypersim into wrong class, objects
  split across views (linking), split within a view, merged, missed, and pixels at segment borders, so the fix
  goes where the loss is.
- **Setting:** `experiments/scripts/e59a_pq_loss.py`; the evaluation's labels (SAM 3 image mode, NYU40 prompts at
  0.3, working association, E17) scored per GT pixel by `label_metrics`, all 40 NYU40 classes. Oracles: per-view
  masks unlinked, linked, perfect classes, perfect linking (masks of one GT segment share an id across views),
  perfect grouping (also within a view), pixels within 2 or 5 px of a GT or predicted border made void, and all
  of perfect grouping + classes + 2 px border. A cause for every unmatched GT segment and prediction of the
  linked labels.
- **Scenes:** dev first, then val once. No geometry is refined; Deja View's depth only enters the linking.
- **Checks:** the bedroom figure (`experiments/figures/e59a_bedroom.jpg`), and that `linked` reproduces the
  evaluation's PQ.

## E59a: where SAM 3 + linking lose panoptic quality -> classes first (the unprompted catch-alls), then fragments (14.11.2026)
- **Commits:** 8578486; dev and val on the 3080 server's CPU (`--splits dev val`), about 20 s per scene.
- **Checks:** `linked` reproduces the evaluation's PQ (dev 0.292, val 0.258).
- **Results** (PQ, mean over scenes; `experiments/results/e59a_dev_val.txt`):

  | oracle | dev | val |
  |---|---|---|
  | per-view masks, unlinked | 0.129 | 0.114 |
  | linked (the evaluation) | 0.292 | 0.258 |
  | perfect classes, prompted ones only | 0.411 | 0.342 |
  | perfect classes | 0.436 | 0.462 |
  | perfect linking | 0.372 | 0.342 |
  | perfect grouping | 0.421 | 0.415 |
  | 2 px borders void | 0.405 | 0.362 |
  | grouping + classes + 2 px | 0.779 | 0.724 |

- Val: 317 of 471 GT segments unmatched, 187 of them in the three catch-all classes that are never prompted;
  merged 25, split across views 16, wrong class 11, missed 76. Of 674 unmatched predictions 577 are fragments
  of one GT segment.
- **Conclusion:** classes are the largest share (+0.20 val), most of it the catch-alls (+0.12 of it); then
  grouping fragments (+0.16). Linking alone is +0.08. Next: prompt the catch-alls (E59b).

## Plan: E59b = prompting the catch-all classes (issue #59, 14.11.2026)
- **Goal:** win back E59a's largest share: GT segments of otherstructure, otherfurniture and otherprop that no
  prompt can find.
- **Setting:** SAM 3 image mode at 0.3 with `configs/prompts/nyu40.txt` plus `configs/prompts/catchall.txt`
  (12 generic and frequent words, each mapped to a catch-all), two ways: competing with the NYU40 masks by
  score, and as fallback (`--fallback`: their masks only take pixels no NYU40 mask covers). Scored by
  `e59a_pq_loss.py --segmentation`, same oracles and causes. Defaults unchanged.
- **Scenes:** dev decides, val once.
- **Checks:** NYU40 PQ of the fallback variant on the prompted classes must not drop (it only adds pixels).

## E59b: prompting the catch-all classes -> competing hurts PQ, as fallback PQ is flat, mIoU and AP rise a little (14.11.2026)
- **Commits:** e241b55 (SAM 3 runs), scored at 6c394f5 (adds only the opt-in merge to the script); 3080 server,
  SAM 3 about 21 s per scene and variant, `~/pouya/e59b_run.sh`.
- **Results** (linked labels; `experiments/results/e59b_{competing,fallback}.txt`):

  | prompts | dev PQ / AP / mIoU | val PQ / AP / mIoU |
  |---|---|---|
  | NYU40 (E59a) | 0.292 / 0.216 / 0.405 | 0.258 / 0.345 / 0.349 |
  | + catch-alls, competing | 0.248 / 0.202 / 0.342 | 0.212 / 0.227 / 0.304 |
  | + catch-alls, fallback | 0.298 / 0.254 / 0.444 | 0.254 / 0.351 / 0.365 |

- Val, fallback: unmatched GT segments 317 -> 291, but unmatched predictions 674 -> 2021 (fragments 577 ->
  1757); the generic words fire on parts of objects. Competing, they also take pixels from the NYU40 masks.
- The planned check (PQ of the prompted classes alone) was not computed separately.
- **Conclusion:** the catch-alls are not found by generic words; as fallback they add labeled pixels (mIoU)
  but as many fragments as objects. Prompts stay NYU40 by default.

## Plan and result: E59c = merging touching objects of one class -> PQ +0.03 on dev and val (14.11.2026)
- Planned after E59b, before running, from a split of E59a's 577 val fragments: 530 share a view with another
  prediction on the same GT segment (SAM 3 part masks, which the linking never joins within a view); 182 have
  the GT segment's class, among them 51 GT segments split into same-class parts none of which matches
  (5.7 % of the labeled pixels).
- **Setting:** `associate.merge_touching` after the linking: cubes of 4 pixel footprints, objects of one class
  merge when the share of the smaller one's cubes in or next to the other's is at least the contact share.
  Opt-in (`e59a_pq_loss.py --merge`), defaults unchanged. Commit 6c394f5, 3080 CPU.
- **Dev sweep** (PQ, NYU40 prompts; linked 0.292): contact 0.02 0.286, 0.05 0.289, 0.1 0.304, 0.2 0.308,
  0.3 0.321, 0.4 0.320, 0.6 0.319, 0.8 0.317. 0.3 chosen on dev; val run once.
- **Results** (`experiments/results/e59c_*.txt`):

  | labels | dev PQ / PQ things / AP / mIoU | val PQ / PQ things / AP / mIoU |
  |---|---|---|
  | linked (E59a) | 0.292 / 0.159 / 0.216 / 0.405 | 0.258 / 0.162 / 0.345 / 0.349 |
  | merged 0.3 | 0.321 / 0.193 / 0.239 / 0.405 | 0.286 / 0.198 / 0.347 / 0.349 |
  | fallback catch-alls + merged 0.3 | 0.341 / 0.227 / 0.298 / 0.444 | 0.284 / 0.198 / 0.348 / 0.365 |
  | perfect grouping (oracle) | 0.421 / 0.297 / 0.300 / 0.471 | 0.415 / 0.335 / 0.389 / 0.438 |

- **Checks:** unit test (touching same-class objects merge, separated or other-class ones stay apart). The
  geometry is untouched (labels only), so the surface scores are Deja View's.
- **Conclusion:** the merge closes about a fifth of the gap to perfect grouping (val +0.028 of +0.157), the
  same on dev and val; mIoU cannot change (classes stay). With the fallback catch-alls dev gains more (0.341)
  but val does not (0.284). Still open: the 395 fragments of another class than their object (a part mask
  named differently, e.g. a pillow mask on a sofa) and the catch-alls.

## Plan and result: E59d = absorbing part masks of another class into their host -> val PQ 0.286 -> 0.336 (15.11.2026)
- Planned after E59c, before running, from E59a's 395 val fragments of another class than their GT segment's:
  top pairs sofa/chair 32, shelves/otherstructure 29, desk/table 23, pillow/chair 22; 255 lie on a matched
  prediction (the host), median contact share 0.91, far smaller than the host.
- **Setting:** `associate.absorb_parts` after `merge_touching` (contact 0.3): an object joins the other-class
  object it touches most if its contact share is at least `need` and it has at most `ratio` times the host's
  cubes; the joined object takes the majority class by pixels. Hosts either any class or things only (no
  wall, floor, ceiling). Opt-in (`e59a_pq_loss.py --absorb need:ratio`), defaults unchanged. Commit e44d0ae,
  3080 CPU.
- **Bug fixed before the sweep:** a first version absorbed during the merge, so a part's union with its host
  pulled in the host's whole same-class group (stuff PQ 0.92 -> 0.38 on dev). Now a separate pass on the
  merged objects.
- **Dev sweep** (PQ; merged 0.3 0.321): need 0.5 / 0.8 x ratio 0.01 / 0.03 / 0.1, all hosts 0.356 to 0.370,
  things only 0.343 to 0.373. 0.5:0.1 into things chosen on dev (0.373); val run once, both host modes.
- **Results** (`experiments/results/e59d_{dev_sweep,val}.txt`):

  | labels | dev PQ / PQ things / AP / mIoU | val PQ / PQ things / AP / mIoU |
  |---|---|---|
  | linked (E59a) | 0.292 / 0.159 / 0.216 / 0.405 | 0.258 / 0.162 / 0.345 / 0.349 |
  | merged 0.3 (E59c) | 0.321 / 0.193 / 0.239 / 0.405 | 0.286 / 0.198 / 0.347 / 0.349 |
  | + absorbed 0.5:0.1 into things | 0.373 / 0.244 / 0.230 / 0.418 | 0.336 / 0.252 / 0.325 / 0.381 |
  | + absorbed 0.5:0.1, all hosts | 0.365 / 0.233 / 0.212 / 0.390 | 0.359 / 0.275 / 0.309 / 0.403 |
  | perfect grouping (oracle) | 0.421 / 0.297 / 0.300 / 0.471 | 0.415 / 0.335 / 0.389 / 0.438 |

- **Checks:** unit test (a small pillow on a chair joins it, a sofa does not; hosts restrict). Labels only,
  geometry untouched.
- **Conclusion:** absorbing closes about another third of the gap to perfect grouping on val (0.286 -> 0.336
  of 0.415), with PQ things +0.05 and mIoU +0.03; AP drops 0.02 (absorbed parts lose their own scores). All
  hosts is better on val (0.359) but was not the dev choice. Linked 0.258 -> 0.336 in total from E59c and E59d.
## Plan: E25c = an opt-in fused-surface score, and the filter and flag-refine-filter under it (issue #25's question, 14.11.2026)
- **Goal:** issue #25 (E38) asked whether to score one fused surface next to the per-pixel cloud: Déjà View's
  views leave every surface as scattered layers and completeness takes the nearest one, so the per-pixel score
  rewards a thick shell, and deleting points (E45's filter, E27a's flag-refine-filter) may gain partly from that.
  This is an evaluation addition, off by default: no existing number changes.
- **Setting:** `surface: tsdf` in an evaluation config fuses each output's depth maps, after the run's Umeyama
  alignment, and the GT's depth maps into one Open3D TSDF each (Curless and Levoy 1996), takes the marching-cubes
  vertices as the surface and scores accuracy / completeness / Chamfer between the two (`fused_*` keys). Dropped
  pixels are not fused. Voxel size chosen on dev's GT alone, before any output is fused: the size at which the
  GT's fused surface lies closest to the GT's own pixels; truncation 4 voxels.
- **Variants** (`experiments/configs/e25c_fused.yaml`): Déjà View, Déjà View kept 0.5 (E45's instance voxel
  density filter), tangent 4e-2 kept 0.5 (E27a's flag-refine-filter). Dev, then val.
- **Checks:** tests (a plane fuses onto itself, camera depth inverts the lift, fused scores are 0 on the truth,
  invariant to the output's scale, 3 cm off reads 3 cm, no fused keys without the flag); the per-pixel scores of
  this run must equal E27a's.
- **Baselines (per pixel, E27a):** Chamfer dev 3.500 / 3.30 / 3.21, val 5.102 / 3.82 / 3.67 (Déjà View / filter /
  flag-refine-filter).

## E25c: fused-surface score -> the filter's gain is per pixel only; fused, it loses a third of the surface (15.11.2026)
- **Commits:** f827fc1 (score), d9af412 (the point count is a float, else the split mean failed after all
  scenes were scored), 0ef90ce (fuse only the pixels with GT depth, as the per-pixel score: before, output
  surfaces at windows and missing GT inflated fused accuracy, ai_004_010 31 -> 3.0 cm Chamfer). 3080, about
  10 min with one scoring process (6 ran out of the 61 GB). `experiments/results/e25c_{fused,checks}.txt`.
- **Results** (Chamfer in cm, mean / median over the surface, mean over scenes):

  | variant | dev per pixel | dev fused | val per pixel | val fused |
  |---|---|---|---|---|
  | Deja View | 3.50 / 2.04 | 5.43 / 2.96 | 5.10 / 2.61 | 17.01 / 4.95 |
  | Deja View, kept 0.5 | 3.30 / 1.95 | 8.58 / 3.54 | 3.82 / 1.60 | 18.08 / 4.93 |
  | flag-refine-filter | 3.21 / 1.86 | 8.62 / 3.49 | 3.67 / 1.57 | 17.89 / 4.95 |

  Fused completeness: dev 4.29 -> 13.90 / 13.98, val 22.63 -> 31.88 / 31.44; fused accuracy improves
  (dev 6.57 -> 3.25, val 11.39 -> 4.28 / 4.33).
- **Checks:** the per-pixel scores equal E27a's. Each fused surface lies within 1 to 2 cm (mean) of its own
  pixels, the GT's within 0.4 to 1.4 cm. The GT's own depth with half its pixels dropped at random fuses
  without loss (completeness 0.5 to 1.0 cm), with the filter's half it loses a third of the surface by over 5
  cm (5.5 to 12.3 cm): the loss is where the filter cuts, not a fusion artifact. Val's mean is dominated by
  ai_004_003 (74 cm): GT beyond 5 m (through openings) is about 1 % of its pixels but 12 % of its fused
  surface, and Deja View misses it; the medians are not affected.
- **Conclusion:** on one fused surface the filter and flag-refine-filter trade completeness for accuracy and
  lose on dev (5.43 -> 8.6) and tie on val's median (4.95 -> 4.93). Their per-pixel gain comes mostly from
  removing points that the per-pixel completeness does not miss. The fused score stays opt-in
  (`surface: tsdf`); no existing number changes.

## Plan: #52 = learned segment conditioning inside Déjà View, three steps; E52a first (issue #52, 14.11.2026)
- **Why:** E51a and E53a closed the training-free routes of #50: with GT instances, neither averaging
  tokens within a segment nor a hand-set attention bias improves the frozen model. They do not show that
  Déjà View cannot use segments once something is trained. #52 asks exactly that, with the control that a
  gain from training alone (adapting to Hypersim) is not counted as the segments'.
- **Order** (one change per step, each its own entry; a later step runs only if it still asks a question):
  - **E52a:** Déjà View frozen; only small segment modules are trained, each starting at zero (below).
  - **E52b:** E52a's best module plus a LoRA on the shared block, against the same LoRA without segments.
  - **E52c:** segments only in the loss (agreement within a GT instance across views), no segment input at
    test time, against the same fine-tuning without that term.
  Kept back: segment tokens shared across views (only if E52a/b show a gain and identity across views is
  the open part); class embeddings (no class headroom, E25a/E45a); per-scene test-time training.

## Plan: E52a = learned per-head attention bias and gated segment pooling, Déjà View frozen, GT instances (issue #52, 14.11.2026)
- **Goal:** is there a learned, segment-conditioned change to Déjà View's loop that improves its depth when
  the model itself stays frozen? E53a set one β for all 12 heads and every chosen step by hand, although
  heads differ widely (own-segment share 0.11-0.82); E51a averaged tokens blindly and bent their rays. Here
  gradient descent sets both per head and per step, with the freedom to use a negative bias or none.
- **Modules** (both act inside the released shared `LoopedAABlock`, dvlt unchanged, wrapper as in E53a):
  - **Bias:** β[k, a, h] for loop step k (16), attention a (frame, global) and head h (12): 384 scalars,
    added to the logit of every pair of patch tokens with the same segment id. Exact without an N x N mask:
    each labelled token gets a one-hot code e of its segment, q' = [q, β √d e], k' = [k, e], v padded with
    zeros, SDPA with scale 1/√d, so q'·k'/√d = q·k/√d + β [same segment]. Special and unlabelled tokens get
    e = 0. Codes up to 192 segments per sample (head dim 64 + 192 = 256, flash's limit); beyond that the
    smallest segments are dropped to unlabelled and their count is logged.
  - **Pooling:** after loop step k, x_i += B_k A (m_s(i) - x_i), m_s the mean token of segment s over all
    views (GT ids are scene-wide, so this passes information between views of one object), A: 768 -> 16
    shared, B_k: 16 -> 768 per step and zero at the start (LoRA's init): ~0.21 M parameters.
  - At zero (the start) both return Déjà View: with all parameters exactly zero the wrapper takes the plain
    path (bit for bit); the training path at zero differs only by the kernel (head dim 256), measured and
    reported as noise, as E53a's β = 0.
- **Models** (same steps, data, seeds): bias only, pooling only, both; each with GT instances, and the
  control: the same model trained with **grid segments** (blocks of 3 x 3 patches per view, not linked
  across views, about as many as GT instances): the same parameters and training, no semantics. A gain
  over the grid control is the segments'; a gain over Déjà View only is adaptation.
- **Data:** the 136 Hypersim train scenes of E41b (`train`, `train_extra`; same 17 skipped), the last 8
  held out for model selection; no dev, val or test scene. Views and GT copied from the 3080 server
  (~3 GB) to the cluster. Each step: one scene, a random 8-12 of its 20 views (memory and augmentation, as
  E41b), 504 px, K = 16.
- **Loss:** the conditioned model's points x = o + D d (its own rays and depth), one Umeyama Sim(3) of them
  onto the GT's pixels (detached), L1 of the relative 3D error |S x - x_gt| / depth_gt, pixels with GT,
  errors above 0.5 clipped (misregistered pixels, as E41b's MAX_RESIDUAL). Frame-free, so no cached target
  is needed for a view subset; it trains depth and rays together.
- **Training:** full backprop through the 16 steps, dvlt's checkpointing on the looped block, bf16
  autocast, AdamW (lr 1e-3, no weight decay), held-out error every 250 steps, the state of least held-out
  error kept. Step budget set by step 0's timing, about 1 h per model on the RTX 5090, written here before
  the run.
- **Step 0, look first:** (1) identity: zero parameters equal Déjà View bit for bit on the bedroom; the
  training path at zero, its gap; (2) gradients: finite, non-zero for B and β after one step, zero for
  every frozen weight; (3) peak memory and seconds per step on the 5090 at 8, 12 and 20 views; (4) a
  20-step overfit on one train scene must lower its loss (else the modules cannot act at all).
- **Scores:** dev (3 scenes, 20 views, GT instances), each model as a backbone: the protocol's pinhole
  fit, Sim(3), `surface_metrics` (Chamfer, accuracy, completeness, cm); plus scored on Déjà View's plain
  cameras (as E53a), the rays' median turn, the learned β per head and step, |B_k| per step, time.
- **Checks:** unit tests on CPU: the augmented SDPA equals softmax(q kᵀ / √d + β [same]) v written out per
  head; segment means and the pooling update on synthetic ids; zero parameters return dvlt's output bit
  for bit; the grid segments' layout.
- **Reading, fixed before running:**
  - The best GT model must beat both Déjà View and its grid control by at least 0.05 cm Chamfer on the dev
    mean and in 2 of 3 scenes. Otherwise: negative; E52b's question (does a LoRA let it read segments)
    stays, E52a gives it no lead.
  - If it does: SAM 3's linked masks (the protocol's) in place of GT at test time, then val once; a model
    trained on GT and tested on SAM 3 is stated as such.
  - 6 models on 3 scenes; one seed first, a second seed for any model that passes.
- **Figure:** bedroom `ai_006_007` views 18, 0, 12, 6: Déjà View's per-pixel error, the best GT model's and
  its grid control's on one 0-30 cm scale; the learned β per head over steps (frame, global); held-out
  error over training for all 6 models.
- **Step 0 done (14.11, cluster job 2731, RTX 5090, 12c945d, train scene `ai_001_001`,
  `results/e52a_probe.yaml`):**
  `uv run python experiments/scripts/e52a_segment_adapters.py --cfg configs/e52a_adapters.yaml --probe --out runs/e52a_probe`.
  - Identity: a zero adapter through `predict` is Déjà View bit for bit (depth, confidence, rays). The
    training path at zero (bf16, decoders one view at a time, bias: head dim 256) moves depth by a
    median 0.03-0.04% (max 10-21% on single pixels): gains are read against the trained-at-zero gap,
    not against zero.
  - Gradients finite and non-zero for β and B (A gets none at the first step, as LoRA's down-projection
    while B = 0); no frozen weight has a gradient.
  - Cost per training step (forward + backward): 8 views 0.52 s at 3.1 GB peak, 12 views 0.87 s at
    3.4 GB, 20 views 1.79 s at 4.5 GB. Memory is no constraint; 8-12 views stay for augmentation, as
    planned.
  - 20 steps on the scene's fixed 12 views (relative 3D error, start 0.01270): bias at lr 1e-4 and
    3e-4 does not move (0.01268), at 1e-3 falls slowly and steadily (0.01244, -2%); pooling at 1e-4
    falls to 0.01058 (-17%) and holds, at 3e-4 reaches 0.01035 then rises, at 1e-3 oscillates
    (0.01181 after 20). So the modules can act, pooling far more than the bias, and the planned single
    lr 1e-3 would destabilise pooling (the laptop smoke's rising loss).
  - **Changes before the run, from these numbers:** a learning rate per module, bias 1e-3 and pooling
    1e-4 (the "both" model gets both); 3000 steps per model (~0.7 s per step at 8-12 views plus
    held-out evaluation: about 40 min a model, 6 models about 4 h on one 5090).
  - The one-scene drop of pooling says nothing about segments yet: the grid control can fit a scene
    as well, which the run will show.

## E52a: learned segment adapters inside frozen Déjà View -> no GT model beats Déjà View on any dev scene; the held-out gain came from the 3 scenes where Déjà View fails, pooling diverged and the grid control lost most of its segments (issue #52, 15.11.2026)
- **Goal, setting:** as the plan above, with step 0's changes (lr bias 1e-3, pooling 1e-4, 3000 steps
  per model). 128 train + 8 held-out Hypersim scenes (E41b's 136), 8-12 random views per step at 504 px,
  K = 16; 6 models (bias, pool, both x GT instances, grid); dev: 3 scenes, 20 views, GT instances.
- **Commits, commands:** training at 42fcf97, cluster job 2761 (RTX 5090, 3 h 52 min; per model 27-53
  min, the grid models slowest):
  `uv run python experiments/scripts/e52a_segment_adapters.py --cfg configs/e52a_adapters.yaml --out runs/e52a`;
  check at a0e5c82, job 2788:
  `uv run python experiments/scripts/e52a_segment_adapters.py --cfg configs/e52a_adapters.yaml --check runs/e52a --out runs/e52a_check`;
  figures `uv run python experiments/scripts/e52a_figures.py runs/e52a runs/e52a_check`. Raw output
  `results/e52a_train.yaml` (held-out curves, dev scores), `results/e52a_check.yaml` (loss per scene).
- **Three failed starts before the run of record:**
  - Job 2736 stopped at step 1: two train scenes have fewer than 20 views (`ai_011_010` 9,
    `ai_047_008` 16), and 12 of 9 cannot be drawn. e636823 caps a step's views at the scene's.
  - Job 2737 refused the empty output folder 2736 left behind.
  - Job 2738 ran out of GPU memory after 1.5 h (pool-gt's held-out evaluation, 28 GB allocated):
    `beartype_this_package` decorates and keeps every function it sees, including the closures
    `forward` made anew on each pass, so each pass's segment codes stayed on the GPU (2-4 MB per step,
    diagnostic job 2759). 42fcf97 moves them to module level; memory then stays flat (6.75 GB used on
    the 5090 over the first 3500 steps). The maths was unchanged: bias-gt's held-out curve matches
    2738's within 0.0005 (bf16 kernels).
- **Checks:** unit tests in `tests/test_adapters.py` pass, including that no per-pass state survives a
  pass; every model at step 0 has Déjà View's held-out error (0.14047); a model whose kept state is zero
  scores exactly Déjà View on dev.
- **Results, dev** (Chamfer, cm; living room / bedroom / lobby, mean):
  - Déjà View 1.606 / 1.837 / 6.995, 3.479.
  - bias-gt (kept at step 2250) 1.828 / 1.986 / 7.453, 3.756: worse on 3 of 3 (+0.28 cm on the mean).
    Rays turn 0.15-0.31° (median).
  - both-gt (kept at step 250) 4.320 / 2.788 / 14.849, 7.319: worse on 3 of 3 (+3.8 cm). Rays turn
    0.49-0.86°.
  - pool-gt and all three grid models kept their zero start (no held-out step beat it), so they are
    Déjà View exactly.
- **Held-out error over training** (mean over 8 scenes, 12 fixed views; Déjà View 0.1405):
  - bias-gt falls slowly to 0.1384 at step 2250 (-1.5%); bias-grid rises steadily to 0.1419.
  - Pooling diverges at lr 1e-4 on all 128 scenes, although it was stable on one scene in step 0:
    pool-gt 0.149 after 250 steps and 0.20-0.27 from step 1000, pool-grid 0.16-0.23; both-gt dips to
    0.1312 at step 250 and then jumps to 0.20-0.28.
- **Why the held-out gain did not carry over (the check, 20 fixed views per scene; 12 views give the same
  picture, so the dev scores' 20 views are not the cause):**
  - Déjà View's loss is 0.24-0.43 on 3 of the 8 held-out scenes (`ai_038_004`, `ai_038_005`,
    `ai_039_010`) and 0.02-0.06 on the other 5 and on dev (0.009-0.027). The mean held-out error, and so
    the model choice, is set by those 3.
  - There the models gain: both-gt 0.423 -> 0.348, 0.296 -> 0.215; bias-gt 0.423 -> 0.402.
  - On the other 5 held-out scenes bias-gt is +2 to +13% on 4 and -0.7% on 1 (`ai_046_006`), both-gt
    +23 to +86%; on dev bias-gt is +4 to +19% and both-gt +49 to +148%.
- **The grid control was not the planned control:** at 8-12 views, 3 x 3 tiles give 860-1300 segments
  per pass, and codes hold 192 (the flash kernel's head dimension), so about 80% of tiles were
  unlabelled (2.65 M dropped over 3000 training steps). GT instances never hit the cap. With no GT model ahead of Déjà View
  this does not change the reading, but a grid-control comparison from this run means nothing.
- **What bias-gt learned** (`figures/e52a_beta.jpg`): β between -0.76 and +0.66, differing more by head
  than by step: in global attention heads 3, 6 and 7 pull tokens towards their own segment in the later
  steps (up to +0.66), heads 9 and 11 push them away throughout; frame attention up to ±0.5. Unlike
  E53a's one β for all heads, gradient descent did use the freedom per head, and it still did not help.
- **Reading, by the rule fixed in the plan:** the best GT model must beat Déjà View and its grid control
  by 0.05 cm Chamfer on the dev mean and in 2 of 3 scenes. No GT model beats Déjà View on any dev scene
  -> negative. SAM 3 and val are not run.
- **What the figures show:** on the bedroom (views 18, 0, 12, 6) bias-gt looks like Déjà View (mean
  per-pixel error 9.0 against 8.8 cm); both-gt raises the error over whole walls and the ceiling (11.2
  cm), a smooth change of shape rather than errors at edges.
- **Conclusion:**
  - With Déjà View frozen, a learned per-head attention bias within GT instances, trained on 128 Hypersim
    scenes, does not improve its depth on dev; it learns to fix scenes where Déjà View fails badly, at a
    small cost on the scenes where it works. Pooling as set up here could not be trained stably, so
    pooling is untested, not negative.
  - The run says less than planned, for three reasons of setup, not of the idea: the mean loss over
    scenes lets a few failing scenes decide both the gradient and the model choice; pooling's lr from a
    one-scene sweep was too high for 128 scenes; the grid control was capped. A rerun would need a loss
    and selection per scene relative to Déjà View's (or the failing scenes excluded), a lower pooling
    lr, and a grid with at most 192 tiles per pass.
  - For #52: E52a gives E52b no lead (the plan's rule), and its question (does a LoRA let Déjà View read
    segments) stays open; the same three setup points would apply to it.
- **Limits:** one seed; 3 dev scenes; 8 held-out scenes, 3 of them failures of Déjà View; GT walls,
  floors and ceilings one segment per class; trained with 8-12 views, scored at 20 (the check shows the
  same ordering at 12 and 20 views).
- **Figures:** `figures/e52a_training.jpg`: held-out error over steps for all 6 models, and per scene
  each trained GT model's loss over Déjà View's against Déjà View's own loss (held-out and dev);
  `figures/e52a_beta.jpg`: bias-gt's β per loop step and head, frame and global;
  `figures/e52a_bedroom.jpg`: bedroom views 18, 0, 12, 6: image, GT instances per patch, per-pixel error
  of Déjà View, bias-gt and both-gt on one 0-30 cm scale. Look at the walls and ceiling in the last row.

## Plan: E52a, second run = the same 6 adapters with the first run's three setup flaws fixed (issue #52, 15.11.2026)
- **Goal:** the first run (above) was negative but said less than planned, for three reasons of setup.
  Fix exactly those, keep everything else, and read the result by the same rule.
- **Fix 1, loss and selection relative to Déjà View per scene.** First run: the absolute error, so the 3
  held-out scenes where Déjà View fails (0.24-0.43, others 0.02-0.06) set the gradient and the kept step.
  Now each step's loss is the model's error over Déjà View's on the same scene, views and Umeyama pixel
  subset (one extra no-grad pass of plain Déjà View, about +30% time), and the held-out error is the mean
  over scenes of the same ratio on fixed views (1 at step 0). Chosen over dropping the failing scenes:
  it keeps the training set and E41b's split, and still tests those scenes. Adam's update does not
  depend on the loss's per-scene scale except through its epsilon, so the learning rates carry over.
- **Held-out scenes 8 -> 16** (120 train + 16 held out, the last 16 of the same ordered 136, all with 20
  views), so that no single scene decides the kept step.
- **Fix 2, pooling lr.** 1e-4 (from step 0's one-scene sweep) diverged on 128 scenes. Before the run: a
  multi-scene sweep of pool-gt alone, lr 1e-5, 3e-5 and 1e-4 with the new loss, 500 steps, held-out
  every 50 (one cluster job, ~25 min). The highest lr whose held-out curve does not rise after its
  minimum is used for pool and both, and the step budget is reset if the sweep shows a plateau earlier;
  both are written here before the run. The sweep reads only held-out train scenes, never dev.
- **Fix 3, grid control.** First run: 3 x 3 *patch* blocks, 108 per view, 860-1300 per pass, ~80%
  dropped at the 192-code cap. GT instances give a scene's views a median of 6-54 segments each (~13
  in most of the 17 local dev / val scenes) and 13-169 per 20-view pass, because GT links views.
  The grid now splits each view evenly into 3 x 3 tiles (9 x 12 patches each): 9 per view, near GT's
  per-view count, 180 per pass at dev's 20 views, under the cap; tiles stay unlinked across views, as
  planned. The script asserts that the grid never loses a tile.
- **Kept:** 136 scenes and their order, 8-12 random views per step, bias lr 1e-3, 3000 steps (unless the
  sweep says otherwise), 6 models (bias, pool, both x GT, grid), one seed, dev scoring as before,
  `--check` after training, `e52a_figures.py`.
- **Checks before the run:** unit tests (grid layout, 9 even tiles, apart per view); a laptop smoke run on
  dev / val scenes as train data (not of record) for the ratio at step 0 (= 1) and on the first steps.
- **Reading, unchanged:** the best GT model must beat Déjà View and its grid control by 0.05 cm Chamfer on
  the dev mean and in 2 of 3 scenes; otherwise negative. SAM 3 and val only if it passes.
- **Pooling lr sweep done (15.11, cluster job 2790, RTX 5090, fe8bfff, 18 min,
  `results/e52a2_sweep.yaml`):** pool-gt alone, 400 steps instead of the planned 500 to keep the job
  under 30 min, held-out every 50 (ratio to Déjà View, mean over the 16 held-out scenes):
  `uv run python experiments/scripts/e52a_segment_adapters.py --cfg configs/e52a_adapters.yaml --splits dev --set steps=400 eval_every=50 "segments=[gt]" "models={pool: [false, true]}" lr_pool=<lr> --out runs/e52a2_sweep_<lr>`
  (an earlier job, 2789, stopped at step 1: PyYAML read `1e-5` as a string; `--set` now converts it).
  - lr 1e-5: 0.9997 at step 50, falls slowly and almost steadily to 0.9939 at step 400, still falling.
    Kept step 400: lower than Déjà View on 11 of 16 held-out scenes (ratio 0.965-1.034).
  - lr 3e-5: minimum 0.9955 at step 100, then rises to 1.0114 at step 400.
  - lr 1e-4: rises from step 50 (1.021) to 1.10-1.20, kept its zero start: the first run's divergence
    again, now with the per-scene loss, so it was the lr, not the absolute loss.
  - Time: 0.73 s per step on average, the extra Déjà View pass and the held-out evaluations (16 scenes
    x 12 views, every 50 steps) included.
  - **Changes before the run, by the rule above:** pooling lr 1e-5 (the only lr whose curve does not
    rise after its minimum), for pool and both; 3000 steps stay, since 1e-5 is still falling at 400.
    Six models at ~0.75 s per step: about 4-5 h on one 5090.
  - The sweep also scored dev (it runs the default scoring after training). pool-gt Chamfer at lr 1e-5:
    1.590 / 1.791 / 6.737 cm against Déjà View's 1.606 / 1.837 / 6.995 (mean 3.373 against 3.479,
    lower on 3 of 3), rays turned 0.10-0.13°. The lr was **not** chosen from these numbers, and they
    are not read: without the grid control trained the same way, a gain may be adaptation alone. The
    run of record decides.

## E52a, second run: the first run's setup flaws fixed -> pooling gains ~0.1 cm Chamfer on dev, but the grid control gains as much: adaptation, not segments; every model's training turns unstable after ~500 steps (issue #52, 15.11.2026)
- **Goal, setting:** as the plan above: the first run's 6 models, now with the loss over Déjà View's per
  step, 120 train + 16 held-out scenes, an even 3 x 3 grid control, pooling lr 1e-5 (sweep), bias lr
  1e-3, 3000 steps, one seed. Dev: 3 scenes, 20 views, GT instances.
- **Commits, commands:** training at 0b8a718, split by segment kind because the partition caps a job at
  6 h (each model seeds its own stream, so the split changes no model's training): cluster jobs 2791
  (GT models, 1 h 59 min) and 2792 (grid models, 2 h 18 min), RTX 5090:
  `uv run python experiments/scripts/e52a_segment_adapters.py --cfg configs/e52a_adapters.yaml --splits dev --set "segments=[gt]" --out runs/e52a2_gt`
  (and `"segments=[grid]" --out runs/e52a2_grid`); check, job 2823:
  `uv run python experiments/scripts/e52a_segment_adapters.py --cfg configs/e52a_adapters.yaml --check runs/e52a2_gt --out runs/e52a2_check`;
  figures `uv run python experiments/scripts/e52a2_figures.py runs/e52a2_gt runs/e52a2_grid runs/e52a2_check`.
  Raw output `results/e52a2_train.yaml` (both jobs), `results/e52a2_check.yaml`.
- **Checks:** both jobs scored Déjà View identically on dev (1.606 / 1.837 / 6.995 cm); a model whose kept
  state is zero equals it exactly; the grid lost no tile (0 dropped in training and at dev's 20 views),
  GT none either.
- **Held-out error over training** (over Déjà View's, mean of 16 scenes, 12 fixed views; 1 = Déjà View):
  - pool-gt 0.9971 (step 250), 0.9950 (500), then up: 1.011 (1000), 1.086 (2000), 2.63 (3000).
  - both-gt 0.9958, 0.9947 (500), then 1.011, 1.056, 1.21.
  - pool-grid 0.9968 (250), 1.000 (500), then 1.076, 2.67, 4.08.
  - bias-gt never below 1: 1.001 (250) rising to 1.015; bias-grid 1.009 rising to 1.084; both-grid 1.004
    rising to 5.32.
  - The training loss rises with the held-out error in every model, so this is unstable optimisation,
    not overfitting: at the sweep's 1e-5 pooling still diverges, only later (the sweep stopped at 400).
    Kept: pool-gt and both-gt step 500, pool-grid step 250, bias-gt, bias-grid and both-grid their zero
    start (= Déjà View).
- **Results, dev** (Chamfer, cm; living room / bedroom / lobby, mean; Déjà View 1.606 / 1.837 / 6.995,
  3.479):
  - pool-gt 1.619 / 1.778 / 6.734, **3.377** (-0.102; 2 of 3 better). Rays turn 0.13-0.15°.
  - pool-grid 1.588 / 1.802 / 6.694, **3.362** (-0.117; 3 of 3 better). Rays turn 0.09-0.11°.
  - both-gt 1.634 / 1.787 / 6.808, 3.409 (-0.070; 2 of 3). Its grid control both-grid kept zero.
  - Most of each gain is the lobby (-0.19 to -0.30 cm), Déjà View's worst dev scene.
  - Depth alone does not carry it: each pooling model's depth on Déjà View's plain cameras scores
    4.19-4.25 cm (mean), worse than Déjà View; the gain needs its own, slightly turned rays.
- **Per scene (the check, GT models, 20 fixed views):** pool-gt and both-gt change the training loss by
  -6% to +5% on the 16 held-out scenes (lower on 11 and 9 of 16), 0 to -2% on Déjà View's 3 failing
  scenes (first run: up to -27%); on dev -7 to -8% on the lobby, -1.4 to -1.7% on the bedroom, -3% and
  +4% on the living room (pool-gt, both-gt).
- **Reading, by the rule fixed in the plan:** the best GT model (pool-gt, 3.377) beats Déjà View by
  0.10 cm in 2 of 3 scenes, but not its grid control (pool-grid 3.362, better on 2 of 3 scenes than
  pool-gt) -> **negative for segments**. both-gt beats its control (zero) by 0.07 cm in 2 of 3, but that
  control never trained; pool-grid shows the same gain without segments, so both-gt's is not read as the
  segments' either.
- **What the figures show:** `figures/e52a2_training.jpg`: left, every curve dips by at most 0.5% before
  rising; right, per dev scene the pooling models' Chamfer change with GT (solid) and grid (hatched) is
  of the same size and sign. `figures/e52a2_bedroom.jpg` (views 18, 0, 12, 6): mean per-pixel error 8.8
  cm (Déjà View) -> 8.7 cm (both models); the change is smooth over whole surfaces, ±2-4 cm, both signs
  (ceiling and floor of view 6 better, its back wall worse), not along instance boundaries.
- **Conclusion:**
  - With the setup fixed, a pooling adapter inside frozen Déjà View improves dev Chamfer by ~3% (0.10-0.12
    cm), mostly on one scene, and it does so equally with GT instances and with unlinked grid tiles. The
    gain is adaptation of Déjà View to Hypersim through a small learned module, not information from
    segments. With E51a and E53a (training-free, negative) and the first run, GT segments inside the
    frozen model have now shown no effect beyond a segment-free control in any form we tried.
  - The adaptation gain is a lead of its own, not a result: one seed, 3 dev scenes, kept after 250-500
    steps of a training that then diverges, val not run. It belongs with E52b's segment-free LoRA
    control.
  - The instability (training loss rising after ~500 steps at bias lr 1e-3 and pooling lr 1e-5, also in
    the bias models, which were stable in the first run's absolute loss) is not explained. A guess, not
    tested: the ratio to Déjà View's error weights scenes where Déjà View is near-perfect (error ~0.01)
    up to ~45x more than the failing ones (0.009 against 0.42), so their noise dominates the gradient.
  - So "GT equals grid" holds for briefly trained adapters only; a stable training (lower lr, warm-up,
    or the ratio bounded) could still separate them. Not planned without a team decision.
- **Limits:** one seed; 3 dev scenes, one of which carries most of the gain; kept states at 250-500 steps;
  the sweep's 400 steps did not reveal 1e-5's later divergence; GT walls, floors and ceilings one segment
  per class.

## Plan: E52a, second run, stability probe = can the adapters train past ~500 steps? (issue #52, 15.11.2026)
- **Goal:** the second run's "GT equals grid" holds only for adapters kept at 250-500 steps before every
  model's training loss rose. Find a setting under which both-gt trains stably for 1000 steps; only
  then is a full GT-against-grid comparison worth running.
- **Arms** (both-gt, so both modules are tested; 1000 steps, held-out every 100, no dev scoring):
  - A, lower lrs: bias 3e-4, pooling 3e-6 (the second run's / 3.3), plain ratio.
  - B, the second run's lrs (1e-3, 1e-5), loss = error / max(Déjà View's error, 0.03): Déjà View's error
    is 0.009-0.43 per scene, so the plain ratio weights a near-solved scene up to ~45x a failing one; the
    floor (about the typical held-out error) caps that at ~3x. A log-ratio loss would not do this: its
    gradient ∇e / e equals the ratio's ∇e / b while the model stays near Déjà View (e ≈ b).
- **Reading:** an arm is stable if its held-out ratio (plain, as before) does not rise after its minimum
  and its training loss shows no upward trend over the last 500 steps. If one is, run pool and both x GT,
  grid with it (~3 h, ask first); if neither, PR as is, the instability stated as a limit.

## E52a, second run, stability probe: lower lrs train both-gt without divergence for 1000 steps, but its held-out gain stays at 0.5%; the floored ratio diverges like the second run (issue #52, 16.11.2026)
- **Setting, commits, commands:** as the plan above, at 7e278dc; cluster jobs 2834 (A, 15 min 51 s) and
  2835 (B, 15 min 15 s), RTX 5090:
  `uv run python experiments/scripts/e52a_segment_adapters.py --cfg configs/e52a_adapters.yaml --splits --set steps=1000 eval_every=100 "segments=[gt]" "models={both: [true, true]}" lr_bias=3e-4 lr_pool=3e-6 --out runs/e52a2_probeA`
  (B: `lr_bias`, `lr_pool` from the config and `ratio_floor=0.03`, `--out runs/e52a2_probeB`). Raw output
  `results/e52a2_probe.yaml`.
- **Checks:** no segment dropped; held-out reference (step 0) is Déjà View's error, as in the second run.
- **Held-out error** (over Déjà View's, mean of 16 scenes, 12 fixed views), steps 100 to 1000:
  - A: 0.9993, 0.9987, 0.9987, 0.9975, **0.9949** (500), 0.9970, 0.9971, 0.9961, 0.9962, 0.9950.
  - B: 1.0012, 1.0024, 0.9996, 0.9970, **0.9968** (500), 1.0034, 1.0090, 1.0093, 1.0135, 1.0168.
  - Per scene at step 1000: A lower on 12 of 16 (median 0.998, range 0.965-1.027); B on 7 of 16
    (median 1.009, range 0.967-1.11).
- **Training loss** (mean over each 100 steps): A 0.9991 falling to 0.9818, last 500 steps 0.9887,
  0.9914, 0.9921, 0.9886, 0.9818: no upward trend. B (floored, so not on A's scale) 0.75-0.82 throughout,
  no trend in either direction while its held-out error rises.
- **Reading, by the plan's rule:** B is unstable. A meets the training-loss part; its held-out ratio rose
  by 0.002 after its minimum at step 500 and was back to it at step 1000 (0.9950 against 0.9949), about
  the step-to-step spread of the curve. For comparison, the second run's both-gt was at 1.011 by step
  1000. A is read as **stable, but not improving after step 500**: its gain (0.5% held-out) is the size
  of the second run's kept states.
- **Conclusion:**
  - The second run's divergence is the learning rate, not the ratio's weighting: lowering both lrs by
    3.3x removes it, while bounding the weights (B) does not. The untested guess in the second run's
    entry is refuted.
  - A stable training does not by itself buy a larger gain: within 1000 steps A plateaus where the
    second run's early-kept models were. A full GT-against-grid run at A's lrs would read the same
    comparison without the early-stopping caveat; whether it is worth ~3 h is open.
- **Limits:** one seed, one model (both-gt), 1000 steps; no dev scoring and no grid arm, so this says
  nothing yet about segments.

## Plan: E52a, third run = pool and both x GT, grid at the probe's stable lrs (issue #52, 16.11.2026)
- **Goal:** the second run's "GT equals grid" was read on models kept after 250-500 steps of a training
  that then diverged. Arm A of the stability probe trains stably; rerun the comparison at its lrs, so
  that the kept states are chosen from a training that does not diverge.
- **Setting:** the second run's (120 train + 16 held-out scenes, loss over Déjà View's, plain ratio,
  3 x 3 grid, 3000 steps, held-out every 250, one seed, dev 3 scenes at 20 views), with lr_bias 3e-4 and
  lr_pool 3e-6 (probe arm A). Models pool and both x {gt, grid} = 4; the bias-only models are left out
  (never below Déjà View in either run). Two jobs (GT, grid; the partition caps 6 h), then `--check` of
  the GT models. Estimated ~45 min per model, ~1.5 h per job.
- **Reading (the second run's rule, unchanged):** the best GT model must beat Déjà View AND its grid
  control by >= 0.05 cm dev mean Chamfer and on 2 of 3 dev scenes; else negative for segments.
  Stability is reported per model with the probe's rule (held-out not rising after its minimum,
  training loss without upward trend over the last 500 steps).

## E52a, third run: at the probe's lower lrs all four adapters still diverge, only later; GT instances again gain no more than the grid control -> negative for segments (issue #52, 16.11.2026)
- **Setting, commits, commands:** as the plan above, at cbeb5d0; cluster jobs 2863 (GT models, 1 h 18
  min), 2864 (grid models, 1 h 27 min), 2865 (check of the GT models, 1 min), RTX 5090:
  `uv run python experiments/scripts/e52a_segment_adapters.py --cfg configs/e52a_adapters.yaml --splits dev --set "segments=[gt]" "models={pool: [false, true], both: [true, true]}" lr_bias=3e-4 lr_pool=3e-6 --out runs/e52a3_gt`
  (and `"segments=[grid]" ... --out runs/e52a3_grid`); check
  `uv run python experiments/scripts/e52a_segment_adapters.py --cfg configs/e52a_adapters.yaml --check runs/e52a3_gt --out runs/e52a3_check`
  (the check's config records the default lrs; it trains nothing); figures
  `uv run python experiments/scripts/e52a2_figures.py runs/e52a3_gt runs/e52a3_grid runs/e52a3_check --run e52a3`.
  Raw output `results/e52a3_train.yaml` (both jobs), `results/e52a3_check.yaml`.
- **Checks:** both jobs scored Déjà View identically on dev (1.606 / 1.837 / 6.995 cm); no segment dropped
  (GT or grid); the second run's figures redrawn by the extended script are byte-identical.
- **Held-out error over training** (over Déjà View's, mean of 16 scenes, every 250 steps):
  - pool-gt min **0.9945** (1000), 0.9947 (1500), then up to 1.0001 (2250), 1.0061 (3000).
  - both-gt min **0.9935** (1500), then up to 1.0003 (2250), 1.0040 (3000).
  - pool-grid min **0.9950** (1000), then up to 1.0042 (2000), 1.0470 (3000).
  - both-grid min **0.9983** (500), then up from step 750 to 1.0459 (3000).
  - The training loss rises with them (pool-grid 0.9875 at 1000 -> 1.0357 at 3000, both-grid 0.9902 ->
    1.0499; GT models 0.98 -> 0.99). So no model is stable by the probe's rule: lowering the lrs by 3.3x
    delays the divergence by 500-1500 steps, it does not remove it. The probe's 1000 steps ended before
    it, as the sweep's 400 had in the second run. **This corrects the probe's conclusion** ("the
    divergence is the learning rate"): its cause is not known; neither lower lrs nor the floored ratio
    removes it.
- **Results, dev** (Chamfer, cm; living room / bedroom / lobby, mean; Déjà View 1.606 / 1.837 / 6.995,
  3.479):
  - pool-gt 1.590 / 1.791 / 6.809, **3.397** (-0.083; 3 of 3 better). Rays turn 0.08-0.12°.
  - both-gt 1.601 / 1.795 / 6.857, 3.418 (-0.062; 3 of 3). Rays 0.16-0.19°.
  - pool-grid 1.564 / 1.810 / 6.826, **3.400** (-0.079; 3 of 3). Rays 0.13-0.15°.
  - both-grid 1.603 / 1.778 / 6.634, **3.338** (-0.141; 3 of 3). Rays 0.10-0.13°.
  - Again most of each gain is the lobby (-0.14 to -0.36 cm). On Déjà View's plain cameras every model is
    worse than Déjà View (3.53-4.19 cm mean): the gain needs the model's own, slightly turned rays.
  - both-grid, the model with the weakest held-out gain (0.998), is the best on dev: model-to-model
    differences of ~0.1 cm on these 3 scenes are not resolved by the held-out selection.
- **Per scene (the check, GT models):** training loss over Déjà View's on the 16 held-out scenes -3.6% to
  +3.6% (lower on 9-13 of 16, at 12 and 20 views), 0 to -2% on Déjà View's 3 failing scenes; on dev at 20
  views pool-gt -2.4 / -1.0 / -5.8% (living room, bedroom, lobby), both-gt +1.7 / -1.1 / -7.7%.
- **Reading, by the rule fixed in the plan:** the best GT model (pool-gt, 3.397) beats Déjà View by 0.08
  cm on 3 of 3 scenes, but its grid control (pool-grid, 3.400) by 0.003 cm, under the 0.05 bar; both-gt
  is 0.08 cm worse than its control both-grid -> **negative for segments**, the same reading as the
  second run, now with all four controls trained.
- **What the figures show:** `figures/e52a3_training.jpg`: left, all four curves dip to 0.994-0.998 and
  rise after 500-2000 steps, the grid models earlier; right, per dev scene the four models change Chamfer
  in the same direction, the grid bars as large as the GT bars or larger. `figures/e52a3_bedroom.jpg`
  (views 18, 0, 12, 6): mean per-pixel error 8.8 cm (Déjà View) -> 8.8 / 8.7 cm (both-gt / pool-gt);
  the change is ±2-4 cm smooth over whole surfaces (view 6's back wall worse, its ceiling and floor
  better), not along instance boundaries, as in the second run.
- **Conclusion:**
  - Three runs of E52a, the last two with fixed setup flaws and trained controls, give the same answer:
    adapters inside frozen Déjà View improve dev Chamfer by 2-4% (0.06-0.14 cm, 3 of 3 scenes in this run),
    and an even 3 x 3 grid of unlinked tiles does so as much as GT instances. The gain is adaptation to
    Hypersim, not segment information. With E51a and E53a, GT segments injected into frozen Déjà View
    have shown no effect beyond a segment-free control in any form we tried.
  - No model trains stably past 500-2000 steps at any setting tried (two lrs, plain and floored ratio),
    so every kept state is early. Whether a stable training would separate GT from grid stays open;
    E52a stops here: finding the instability's cause is a project of its own, and a larger, longer
    training is what E52b (LoRA, with its segment-free control) would test anyway.
- **Limits:** one seed; 3 dev scenes, the lobby carrying most of each gain; kept states early in a
  diverging training; val not run (the plan's rule: no positive GT result); GT walls, floors and ceilings
  one segment per class.

## Plan: E52d = why do E52a's adapters diverge? Diagnostics and four one-change arms (issue #52, 16.11.2026)
- **Goal:** in E52a's three runs every adapter's held-out and training loss rose after 500-2000 steps;
  lower lrs only delayed it, a floored ratio did not help. E52b (LoRA) and E52c train the same way, so
  find the cause first. A stable setting is what those runs start from.
- **Model and setting:** pool-grid at the third run's lr (pooling 3e-6), 3000 steps, held-out every 250,
  no dev scoring (`--splits`): its divergence was the largest (held-out 0.9950 at 1000 -> 1.0470 at
  3000) and it has one module; grid, so no segment question is mixed in. Each step now records the
  gradient norm per parameter and Déjà View's error on the step's scene; every 50 steps each parameter's
  norm.
- **Arms** (one change each against A; the defaults train as E52a did):
  - A, as the third run, diagnostics only. Also the identity check: its curve should repeat the third
    run's pool-grid (same seed; not bit for bit across runs on one GPU, E51a).
  - B, `autocast=false`: fp32 forward instead of bf16 (16 loop steps of bf16 rounding in the gradient).
  - C, `weight_decay_up=100`: AdamW's decoupled decay on B_k only. Adam moves each coordinate by about
    the lr per step, so the decay caps |B_k| near 1/wd = 0.01, ~6x the kept state's largest entry
    (1.7e-3 at step 1000); its pull lr x wd = 3e-4 per step halves an unused weight in ~2300 steps.
  - D, `clip_factor=3`: gradient norm clipped at 3x the median of the last 100 steps' (the norm's scale
    is not known before this run, so the limit is relative).
  - E, `lr_decay=cosine`: lr from 3e-6 to 0 over 3000 steps. Not in the first list; added because the
    third run's kept B_k already moved 58% of the most Adam allows (lr x steps) in 1000 steps: Adam's
    step does not shrink with the gradient, so at a constant lr the weights keep walking once the loss
    is near its floor, which a decay stops and a lower lr only delays.
- **Reading:** an arm is stable, as the probe's rule, if its held-out ratio does not rise after its
  minimum and its training loss has no upward trend over the last 500 steps. A's diagnostics are read
  against the rise: gradient spikes before it point to D, parameter norms growing at a flat gradient to
  C / E, a change only in fp32 to B. If an arm is stable, pool-gt and pool-grid are rerun with it (ask
  first) and E52b starts from it; if none is, the cause stays open and E52b uses E (decay) by default,
  stated as a limit.
- **Cost:** five cluster jobs on one RTX 5090 each, ~40 min each in bf16 (third run: 1 h 27 for two
  models plus dev scoring), B perhaps 2-3x longer.


## E52d: no gradient spikes and no precision effect; B_k moves steadily in a direction that raises both losses, and only slowing (weight decay) or stopping (cosine lr decay) that motion limits the rise (issue #52, 17.11.2026)

- **Setting, commits, commands:** as the plan above, at 2f7a2c7; cluster jobs 2889-2893 (arms A-E), RTX

  5090, 35 min each, B (fp32) 1 h 39 min:

  `uv run python experiments/scripts/e52a_segment_adapters.py --cfg configs/e52a_adapters.yaml --splits --set "segments=[grid]" "models={pool: [false, true]}" lr_bias=3e-4 lr_pool=3e-6 <arm> --out runs/e52d_<arm>`

  with `<arm>` empty (A), `autocast=false` (B), `weight_decay_up=100` (C), `clip_factor=3` (D),

  `lr_decay=cosine` (E); figure and raw output

  `uv run python experiments/scripts/e52d_figures.py runs/e52d runs/e52a3_grid`. Raw output

  `results/e52d_train.yaml` (per-step series as 50-step means; the per-step ones stay in the runs).

- **Checks:** identity: A repeats the third run's pool-grid (held-out 0.99503 against 0.99502 at step

  1000, 1.0426 against 1.0470 at 3000; not bit for bit, as expected on another node). Every arm draws the

  same scenes and views per step (same seed), so training losses compare step by step across arms.

- **Held-out error over training** (over Déjà View's, mean of 16 scenes; minimum (step), at 3000):

  A 0.9950 (1000), 1.0426; B 0.9945 (1250), 1.0492; C 0.9938 (1250), 1.0155; D 0.9946 (1250), 1.0524;

  **E 0.9935 (1250), 0.9973**. Training loss, mean of the last 500 steps (trend over them): A 1.031

  (+0.007), B 1.035 (+0.019), C 1.002 (-0.011), D 1.038 (+0.016), E 0.989 (-0.014); paired against A on

  the same draws, C -0.029 and E -0.042, B +0.005, D +0.008.

- **Diagnostics:**

  - No gradient spikes before the rise: the gradient norm of B_k has p99 / median 2.8-3.7 in every 500-step

    window of every arm. Single-step outliers (up to 45x the median) sit at the same steps in every arm

    (same draws) and also in C and E, which do not diverge; D clipped 101 of 3000 steps (3%) and diverged

    as A. Clipping is not the fix.

  - Precision is not the cause: fp32 (B) follows A within the run-to-run spread.

  - In A the norm of B_k grows almost linearly (0.07 at 250, 0.32 at 1500, 0.68 at 3000), not like the

    square root of steps that a random walk of Adam's steps would give, and its gradient norm grows with it

    (median 2.4 -> 4.9). So Adam moves B_k in a consistent direction while the training loss on the same

    draws rises: the averaged gradient points uphill of the loss it is measured with. This corrects the

    plan's reasoning for E (a random walk near the loss floor).

  - Weight decay (C) slows B_k's growth (0.48 at 3000) and with it the rise; cosine decay (E) stops the

    growth at 0.32 and with it the rise. Neither removes the cause: both only limit how far B_k moves.

- **Reading, by the plan's rule:** no arm is strictly stable; E comes closest (held-out up 0.004 after its

  minimum, then flat; training loss falling over the last 500 steps), C rises 0.022. The best kept states

  are equal within 0.0015 (0.9935-0.9950): no arm finds more gain, they differ in how they end.

- **Conclusion:**

  - Not bf16 rounding, not gradient spikes. The gradient moves B_k steadily in a direction that raises the

    loss it was computed from, which a correct gradient at a small lr does not do on average. The likeliest

    cause is a mismatch between the gradient and the loss: the Sim(3) is detached and fit by least squares,

    while the loss is the clipped L1 of the relative error, so the alignment's own response to B_k (zero

    only when the alignment minimizes the same loss) is missing. Untested; a recomputation mismatch under

    checkpointing would look the same (tested at the start of E52a, but only for the attention patch).

  - Next, cheap and decisive (one scene, minutes): at A's kept state, compare the loss change along the

    computed gradient (re-aligned at each point) with the one it predicts, for the detached and for a

    differentiable Sim(3). If the detached gradient is not a descent direction and the differentiable one

    is, train with the differentiable one. Until then, E52b uses cosine decay (E), stated as a limit.

- **Limits:** one model (pool-grid), one lr, one seed; training losses are per-draw means and share a bump

  at steps 2000-2750 in every arm (same draws), so the held-out curves are the reading.


## Plan: E52d, gradient check = is the adapters' training gradient a descent direction of the loss it is read with? (issue #52, 17.11.2026)
- **Goal:** E52d found B_k moving steadily while the training loss on the same draws rose, so the averaged
  gradient points uphill. Test the likeliest cause before LoRA trains the same way: training's error
  detaches the Sim(3) S, fit by least squares, while the loss is the clipped L1 of the relative error; the
  measured loss re-fits S at every step, and its gradient has the extra term ∂e/∂S ∂S/∂θ, zero only if S
  minimized e.
- **Setting:** `experiments/scripts/e52d_gradient_check.py`, laptop RTX 3050 (a few minutes; the cluster
  queue held E52d's jobs for hours), fp32 forward, 4 views per draw, the 3 dev scenes x 3 draws per state
  (dev only as data for a gradient check: nothing is tuned or scored). States: B_k zero, arm A's kept state
  (step 1000), that state with B_k doubled (stand-in for the diverging phase). Per draw: the detached
  gradient g and the exact one g* (differentiable Umeyama, checked against `umeyama` on every draw), their
  cosine, and central finite differences along -g and -g* at step lengths 1e-3 and 1e-2 (parameter-vector
  norm; one Adam step at lr 3e-6 is ~1.3e-3), as a ratio to the slope the gradient predicts:
  - autograd: S frozen, along -g: tests backprop through the checkpointed loop (expect 1);
  - detached: S re-fit, along -g: does training's step lower the measured loss?
  - exact: S re-fit, along -g*: tests g* (expect 1).
  Also the cosine of the mean gradients per state, as training follows the average.
- **Reading:** autograd and exact ratios near 1 make the check valid (else the bug is elsewhere, e.g. the
  checkpoint recomputation). If the detached ratio is near 1 and cos(g, g*) > ~0.9, the alignment is not
  the cause and the drift needs another explanation. If the detached ratio is small or negative, or the
  mean gradients disagree (cosine well below 1), most clearly at the later states, train with g* (the
  differentiable Sim(3)) and rerun arm A with it (cluster, ~35 min, ask first).

## E52d, gradient check: backprop is right, but the detached Sim(3) makes the training gradient a weak and, averaged over draws, increasingly wrong descent direction (issue #52, 17.11.2026)
- **Setting, commits, commands:** as the plan above, at fbd4c7f, laptop RTX 3050, 15 min:
  `uv run python experiments/scripts/e52d_gradient_check.py --cfg configs/e52a_adapters.yaml --state runs/e52d_A/pool-grid.pt --out runs/e52d_gradcheck`;
  figure `uv run python experiments/scripts/e52d_figures.py runs/e52d runs/e52a3_grid --gradcheck runs/e52d_gradcheck`
  (also redraws `e52d_training.jpg`, byte-identical). Raw output `results/e52d_gradcheck.yaml`.
- **Checks:** the differentiable Umeyama equals `umeyama` on every draw (scale to 1e-9 relative, rotation to
  1e-9); detached and re-fit losses agree at the start point. Backprop through the checkpointed loop matches
  finite differences on all 27 draws (autograd ratio 0.997-1.012 at step 1e-3), and so does g* (exact
  ratio 0.994-1.016): the check is valid, and a recomputation mismatch is ruled out.
- **Results** (9 draws per state: 3 dev scenes x 3 draws of 4 views; median (min-max)):
  | state | cos(g, g*) per draw | cos of the mean gradients | detached ratio | abs(g) / abs(g*) |
  |---|---|---|---|---|
  | B_k = 0 | 0.65 (0.21-0.83) | **0.62** | 0.41 (0.06-0.92) | 1.49 |
  | kept (step 1000) | 0.42 (0.33-0.97) | **0.38** | 0.31 (0.13-1.02) | 1.49 |
  | kept, B_k x 2 | 0.60 (0.13-0.78) | **0.24** | 0.35 (0.03-0.83) | 1.84 |
  - The detached ratio (measured descent of the re-fit loss along -g over the descent g promises) is
    positive on all 27 draws: on its own draw, a step along the training gradient never raises the loss.
    But it gives only a third of what it promises, and as little as 3%; it equals cos(g, g*) times
    abs(g*) / abs(g) (as it must when g* is the true gradient). The detached gradient is 1.5-1.8x longer
    than the exact one: most of it is a component the re-fitted Sim(3) cancels.
  - The lobby (ai_044_001) is the worst scene at every state (cosine 0.13-0.71, ratio 0.03-0.35); the
    bedroom the best.
  - Averaged over draws, as training follows it, the two gradients turn apart as B_k grows (0.62 -> 0.38
    -> 0.24); the per-draw cosines do not fall in the same way, so the drift is in what the draws share.
- **What the figure shows:** `figures/e52d_gradcheck.jpg`: left, per-draw cosines spread 0.1-1 at every
  state while the mean-gradient bar falls; right, grey (backprop) and green (exact) sit on 1, orange
  (training's detached gradient) spreads 0.03-1.02 below them.
- **Conclusion:**
  - The likeliest cause named in E52d is confirmed as a defect of the loss, not of the backprop: with the
    Sim(3) detached, the training gradient is only weakly aligned with the gradient of the loss it is read
    with, longer than it, and its average turns further away as B_k grows. That fits E52d's B_k growing
    steadily without a gain. It does not show directly that the training steps go uphill: per draw the
    detached gradient still descends, and Adam rescales it per coordinate, which this check does not
    measure. The rise could come from the cancelled component (steps that change the output in ways the
    Sim(3) undoes, until second-order effects add up); untested.
  - Next, by the plan's rule: train with the exact gradient (differentiable Umeyama in `step_loss`) and
    rerun arm A with it (cluster, ~35 min, ask first). If its held-out error no longer rises, E52b's LoRA
    trains the same way.
- **Limits:** dev scenes as data (nothing tuned or scored); 4 views per draw (training draws 8-12); 9
  draws per state; the doubled state is a stand-in for the diverging phase, not a state training reached.

## Plan: E52d, exact-gradient arm F = arm A trained through a differentiable Sim(3) (issue #52, 17.11.2026)
- **Goal:** the gradient check found training's detached Sim(3) gradient only weakly aligned with the
  gradient of the loss it is read with (mean-gradient cosine 0.62 -> 0.24 as B_k grows). Does training
  with the exact gradient stop arm A's rise? If so, E52b's LoRA trains that way.
- **Change:** `align: exact` (`configs/e52a_adapters.yaml`, default `detached` = every earlier run):
  `step_loss` fits s, R, t with `umeyama_torch` (moved from the check script to
  `src/evaluation/labeled.py`, tested against `umeyama` and by finite differences) on the same pixel
  subset, in float64, with grad. The loss value is the same as with the detached fit; only the gradient
  gains the term through S. Déjà View's base pass and the held-out passes run without grad, so they are
  unchanged.
- **Arm F:** arm A (pool-grid, lr_pool 3e-6, 3000 steps, held-out every 250, `--splits` none) with
  `align=exact`, nothing else changed.
- **Machine:** the 3080 server (`ssh tum`, user's choice: the cluster queue held ~35 jobs), bf16 as A.
  A ran on a 5090, so F's curve is compared to A's with that caveat (bf16 numbers differ across GPUs;
  E52d's identity check showed run-to-run spread on one GPU). If F is borderline, A is rerun on the 3080
  too (another ~1.5-2 h, ask first).
- **Checks before the run:** laptop smoke (few steps, 4 views): same loss value for `detached` and
  `exact` at one state, finite gradients.
- **Reading:** the probe's rule: stable if the held-out ratio does not rise after its minimum and the
  training loss has no upward trend over the last 500 steps; also |B_k| against A's linear growth. Stable
  -> LoRA (E52b's control) trains with `align=exact`. Rises as A -> the detached Sim(3) is not the
  cause; the cause stays open and E52b uses cosine decay (arm E) as planned.
- **Cost:** ~1.5-2 h on the 3080 (A: ~40 min on the 5090, the 3080 ~3x slower); peak memory ~3.4 GB.

## E52d, exact-gradient arm F: training through the differentiable Sim(3) removes the rise, and the held-out gain is about three times that of every detached arm (issue #52, 17.11.2026)
- **Setting, commits, commands:** as the plan above, at 4e10e7d (clean), on the 3080 server, bf16,
  1 h 56 min (A: 34 min on a 5090):
  `uv run python experiments/scripts/e52a_segment_adapters.py --cfg configs/e52a_adapters.yaml --splits --set "segments=[grid]" "models={pool: [false, true]}" lr_bias=3e-4 lr_pool=3e-6 align=exact --out runs/e52d_F`;
  figure and raw output `uv run python experiments/scripts/e52d_figures.py runs/e52d runs/e52a3_grid`
  (F added to `e52d_training.jpg` and `results/e52d_train.yaml`; arms A-E unchanged).
- **Checks:** laptop smoke before the run: `detached` and `exact` give the same loss value at one state
  (0.01302994), the exact gradient is 0.77x as long, cosine 0.64 (as in the gradient check). F draws the
  same scenes and views per step as A-E: Déjà View's per-step error on the 3080 matches A's on the 5090
  to 1e-3, so training losses still pair step by step across arms.
- **Held-out error over training** (over Déjà View's, mean of 16 scenes, 12 fixed views):
  | step | 500 | 1000 | 1500 | 2000 | 2500 | 2750 | 3000 |
  |---|---|---|---|---|---|---|---|
  | A (detached, 5090) | 0.9964 | **0.9950** | 0.9964 | 1.0041 | 1.0188 | 1.0266 | 1.0426 |
  | E (cosine decay, 5090) | 0.9966 | 0.9943 | 0.9944 | 0.9961 | 0.9973 | 0.9974 | 0.9973 |
  | F (exact, 3080) | 0.9904 | 0.9861 | 0.9838 | 0.9831 | 0.9822 | **0.9818** | 0.9823 |
  - F falls at every eval up to step 2750 and rises 0.0004 at 3000. A rose 0.047 after its minimum, E
    0.004.
  - Training loss over the last 500 steps (linear trend over them; paired against A on the same draws):
    F 0.957 (**-0.020**; -0.074), against A 1.031 (+0.007), C 1.002 (-0.011; -0.029), E 0.989 (-0.014;
    -0.042). F shares the other arms' bump at steps 2250-2750 (same draws); the uptick at 2500 in its log
    is that bump.
  - Norm of B_k: 0.07 / 0.11 / 0.18 / 0.23 at steps 500 / 1000 / 2000 / 3000, growing slower over time
    (A: 0.12 / 0.22 / 0.42 / 0.68, linear). The gradient norm of B_k stays at 1.9-2.4 (median per 250
    steps; A 2.4 -> 4.9).
- **What the figure shows:** `figures/e52d_training.jpg`: the red curve (F) leaves the others from step
  250 on and keeps falling in both loss panels while A-D rise; its B_k norm bends over, and its gradient
  norm stays flat.
- **Reading, by the plan's rule:** the training half holds (no upward trend). The held-out half is missed
  by one eval: +0.0004 at step 3000. That is above the eval's own spread (two runs of A's setting on two
  5090 nodes agreed to 1e-5 at step 1000) but 1% of A's rise and a tenth of E's, after 750 steps within
  0.0013 of each other. Read as stable to about 0.0005 over 3000 steps; whether it stays flat beyond 3000
  is untested. The plan's A rerun on the 3080 is not needed: F and A differ by 0.06 at step 3000, 15x the
  spread of A's repeats across nodes at that step (0.004).
- **Conclusion:**
  - The detached Sim(3) caused E52a's and E52d's divergence: with the exact gradient through the
    alignment, and nothing else changed, the rise is gone and B_k no longer grows linearly. With the
    gradient check this closes E52d's question.
  - The exact gradient also finds more: held-out minimum 0.9818 (-1.8%) against 0.9935-0.9950 (-0.5 to
    -0.65%) for every detached arm. E52a's gains were held back by the wrong gradient, not only cut short
    by the divergence.
  - This gain is adaptation (pool-grid: grid tiles, no segments) on held-out training scenes; it is not a
    dev Chamfer result (`--splits` none) and says nothing yet about segments.
  - Next, by the plan: E52b's LoRA control trains with `align=exact`. E52a's verdict (GT = grid) came
    from runs trained with the detached gradient, so the pool-gt vs pool-grid comparison is rerun with
    `align=exact` and dev scoring before it is relied on.
- **Limits:** one model (pool-grid), one lr, one seed, 3000 steps; F on another GPU than A-E (checked
  above: same draws, spread far below the effect); runtime not comparable (3080 vs 5090; the float64
  Umeyama's share is not timed).

## Plan: E52a, fourth run = pool x {GT, grid} trained with the exact gradient, the kept state chosen on val, after the ladder's overfit rungs (issue #52, 17.11.2026)
- **Goal:** E52a's verdict "GT instances gain no more than the grid control" was read on states kept
  early in trainings that diverged, and E52d found the cause: the detached Sim(3). Trained with the
  exact gradient (arm F) the rise is gone and the held-out gain triples. Rerun the comparison that way:
  do GT instances, given to the pooling adapter, beat the same adapter given grid tiles?
- **Changes against the third run (one in training, one in selection):**
  - Training: `align=exact` (E52d, arm F). Nothing else in training changes: same 120 train scenes,
    same draws per step (the training random stream is untouched), lr_pool 3e-6, 3000 steps, bf16.
  - Selection (`docs/experiments.md`, PR #73): every eval also scores Hypersim's 11 `val` scenes the
    same way as the 16 held-out train scenes (ratio to Déjà View's error, 12 fixed views), and the kept
    state is the one of least mean **val** ratio (`select: val`). The held-out curve stays logged, so
    both selections can be compared afterwards; selection does not change the training trajectory, so
    pool-grid is also a rerun of arm F (its held-out curve should match F's).
  - Logging: TensorBoard under `runs/<out>/tb/<model>-<segments>`: training ratio, Déjà View's error and
    gradient norms every step, parameter norms every 50, held-out and val ratios every 250, and one
    random val scene's ratio every 10 steps (`probe_every: 10`; every step would add ~25% runtime for
    a curve the 250-step eval already gives at less noise). The val probe draws from its own stream.
  - Models: pool x {gt, grid}. bias and both are dropped (bias stayed at zero in every run, both-grid
    diverged first in the third run), so this is the ladder's smallest model: A 768 x 16 + B_k 16 x 16 x
    768, ~0.2 M parameters, constant lr, no weight decay, no dropout or normalization of its own.
  - Code (`experiments/scripts/e52a_segment_adapters.py`, `configs/e52a_adapters.yaml`): `val_splits`,
    `select`, `probe_every`, `train_scenes` (named scenes for rungs 3/4), `held_out: 0` allowed. The
    defaults (`select: held_out`, `probe_every: 0`) reproduce every earlier run's training and choice;
    only the extra val evals are new.
- **Ladder** (`docs/experiments.md`):
  - Rungs 0-2 (setup, data, speed): passed by E52a's step 0 (identity bit for bit, finite gradients,
    0.5-1.8 s per step at 8-20 views on the 5090; data is held in RAM, so a step's time is the forward
    and backward through 16 loop steps of Déjà View) and by the smoke run of this code (laptop, bedroom,
    4 steps, `runs/e52a4_smoke`, not of record). Over 1 s per step on the 3080 (~2.3 s, arm F) is the
    rung-2 warning; it is the frozen backbone's cost, not a fixable bottleneck.
  - Rung 3, overfit the bedroom `ai_006_007` (laptop, 8 views per step drawn from its 20, lr_pool 1e-4
    = step 0's overfit lr, 200 steps, pool-gt and pool-grid). An adapter on a frozen Déjà View can only
    bend its output, so "loss about 0" cannot happen. **Pass:** training ratio over the last 50 steps
    below 0.90 (step 0: -17% in 20 steps on fixed views) and no rise over the last 100 steps.
  - Rung 4, overfit dev (3 scenes; same settings, 300 steps). **Pass:** the last 100 steps' ratio below
    0.95 on each of the 3 scenes, i.e. the adapter fits all three, not one.
  - Rungs 3 and 4 are diagnostics; no score from them is reported. Fail -> bug checklist before rung 5.
  - Rung 5: the run below.
- **Rung 5, command** (3080 server, bf16, the machine of arm F):
  `uv run python experiments/scripts/e52a_segment_adapters.py --cfg configs/e52a_adapters.yaml --splits dev --set "models={pool: [false, true]}" lr_pool=3e-6 align=exact select=val probe_every=10 --out runs/e52a4`
- **Reading:**
  - Positive for segments (E52a's rule, unchanged): pool-gt beats Déjà View AND pool-grid by >= 0.05 cm
    dev mean Chamfer, on 2 of 3 dev scenes. With one seed this is a first positive, to be repeated with
    seeds 1 and 2 before it is reported as a result.
  - Otherwise "not yet working" for segments, not negative: missing are 3 seeds and a tuned baseline
    (the grid control is a control, not a tuned baseline; E52b's LoRA without segments is planned).
  - Also reported: stability by the probe's rule for each model; whether held-out and val choose the
    same step; pool-grid's held-out curve against arm F's.
- **Cost:** rungs 3/4 on the laptop, ~15 and ~25 min. Rung 5 ~2 h per model on the 3080 (F: 1 h 56 min;
  the val evals and probes add ~10 min), ~4-4.5 h for both, plus ~5 min dev scoring. Ask before it.

## E52a, fourth run: trained with the exact gradient, both adapters keep improving and gain on all 3 dev scenes, but GT instances gain no more than grid tiles -> not yet working for segments (issue #52, 18.11.2026)
- **Goal, setting:** the plan entry above. Rung 5 ran on the 3080 at aa3e8a7 (clean), 17.11 23:56 to
  18.11 03:47 (pool-gt 6847 s, pool-grid 6867 s, 2.3 s per step, then dev scoring), one seed (0).
  `--out runs/e52a4`, log `runs/e52a4/e52a4.log`, TensorBoard under `runs/e52a4/tb/`. One CUDA
  allocator OOM warning during dev scoring (03:46); the allocator retried and every dev score was
  written, so no score is missing.
- **Rungs 3 and 4** (on the 3080 instead of the laptop, after the laptop's rung 3 ran 10 s per step;
  aa3e8a7 clean, 29 min both): passed. Rung 3, bedroom: last-50 training ratio pool-gt 0.834,
  pool-grid 0.820 (< 0.90; previous 50: 0.875 / 0.875, still falling). Rung 4, dev, last-100 ratio per
  scene, gt / grid: bedroom 0.909 / 0.911, living room 0.787 / 0.777, lobby 0.707 / 0.696 (all < 0.95).
  Diagnostics only, no scores reported. Runs `runs/e52a4_rung{3,4}`.
- **Training (rung 5), figure `e52a4_training.jpg`, raw `results/e52a4_train.yaml`:**

  | | held-out min (step) | held-out @3000 | val min (step) | train trend, last 500 steps |
  |---|---|---|---|---|
  | pool-gt | 0.9803 (3000) | 0.9803 | 0.9710 (3000) | -0.014 |
  | pool-grid | 0.9822 (2750) | 0.9838 | 0.9762 (2500) | -0.017 |
  | E52d arm F (pool-grid, same draws) | 0.9818 (2750) | 0.9823 | - | -0.020 |

  Ratios are the model's error over Déjà View's, mean over scenes, 12 fixed views. Stability by the
  probe's rule: pool-gt passes (held-out at its minimum at the last eval, training ratio falling);
  pool-grid misses the held-out half by one eval (+0.0016 at 3000 after its 2750 minimum), as F did
  (+0.0004); its training ratio still falls. No divergence in either model, unlike every detached run.
- **Selection:** val and held-out choose the same step for pool-gt (3000), not for pool-grid (val
  2500, held-out 2750). Kept states are val's (plan).
- **Rerun noise (the run's own check):** pool-grid repeats arm F's training on the same draws on the
  same GPU (Déjà View's per-step error equal at every step), yet the training ratio differs from step 1
  on (median |difference| per step 0.003-0.007; bf16 kernels are not deterministic, and Adam's first
  steps amplify small gradient differences). Held-out: pool-grid lies above F by up to 0.0032 (step
  750). pool-gt lies below pool-grid by up to 0.0035, and within 0.0027 of F at every eval. So the
  held-out gap between GT and grid is as large as the gap between two runs of the same model. On val
  pool-gt is below pool-grid by 0.003-0.007 at every eval from step 1000 on; the rerun spread on val is not measured
  (F had no val evals).
- **Dev, Chamfer (cm, 20 views, 3080; Déjà View here differs from the 5090's 1.606 / 1.837 / 6.995 by
  bf16 kernels, so every model is read against Déjà View on the same machine):**

  | | living room | bedroom | lobby | mean |
  |---|---|---|---|---|
  | Déjà View | 1.601 | 1.846 | 7.053 | 3.500 |
  | pool-gt (step 3000) | 1.567 | 1.800 | 6.622 | 3.330 |
  | pool-grid (step 2500) | 1.561 | 1.813 | 6.604 | 3.326 |
  | pool-gt minus pool-grid | +0.007 | -0.013 | +0.018 | +0.004 |

  Both models beat Déjà View on 3 of 3 scenes (mean -0.17 cm; the lobby carries 0.43-0.45 of it). GT
  beats grid on the bedroom only, by 0.013 cm, and loses on the mean.
- **Reading (the plan's rule):** pool-gt beats Déjà View (3/3) but not pool-grid by >= 0.05 cm (0/3,
  mean +0.004 cm) -> not positive. **Not yet working for segments**, not negative: missing are 3
  seeds and a tuned segment-free baseline (E52b's LoRA without segments). Against E52a's third run
  (detached gradient, 5090): the gain over Déjà View grows from 0.08 to 0.17 cm mean, for GT and grid
  alike, so the exact gradient fixes the training but adds nothing segment-specific.
- **Conclusion:** with the training fixed, E52a's earlier verdict holds: the pooling adapter's gain is
  adaptation of Déjà View to Hypersim, which grid tiles give as well as GT instances. The only sign
  for GT is val's 0.003-0.007 lower ratio, inside a range two runs of one model already span on
  held-out. Per the agreed order for #52, no more seeds for E52a (it is not positive); next is E52b's
  control: LoRA on Déjà View's shared LoopedAABlock, no segments, `align=exact`, through the ladder.
- **Commands:** rung 5 as in the plan. Figure and raw:
  `uv run python experiments/scripts/e52a4_figures.py runs/e52a4 runs/e52d_F`.

## Plan: E52b, control = LoRA on Déjà View's shared loop block, no segments: plain fine-tuning on Hypersim, the control for E52b and E52c (issue #52, 18.11.2026)
- **Goal:** E52a's pooling adapter gains over Déjà View with GT instances and with grid tiles alike
  (fourth run, 3080: dev Chamfer 3.500 -> 3.330 / 3.326 cm), so that gain is adaptation to Hypersim,
  not segments. E52b (LoRA + segment pooling, GT vs grid) and E52c (LoRA + a cross-view instance loss)
  need the adaptation a segment-free model of their kind gets: here a LoRA on the shared loop block
  with no segments anywhere. How much does plain LoRA fine-tuning gain over Déjà View on dev, and
  against E52a's pool-grid, trained the same way? This is the tuned segment-free baseline that
  `docs/experiments.md` asks for before any segment result can be read.
- **Model:** LoRA (Hu et al. 2022) of rank 16 on the `qkv` (768 -> 2304) and `proj` (768 -> 768)
  linears of both attentions (frame and global) of the one `LoopedAABlock` that Déjà View runs 16
  times: W x + U D x, D as nn.Linear's init, U zero, so the model starts as Déjà View; one LoRA for
  all 16 steps, as the block is shared. ~0.15 M parameters, near the pooling adapter's ~0.2 M, so the
  two differ in where the capacity sits, not in how much. The MLPs, the image encoder and the
  decoders stay frozen. Scale alpha / r = 1 (with Adam, tuning it is about tuning the lr; Hu et al.,
  Sec. 4.1). Rung 0's smallest model: constant lr, no weight decay, no dropout or normalization of
  its own.
- **Training, as E52a's fourth run except the model:** 120 train scenes, the 16 held-out ones and
  Hypersim's 11 `val` scenes evaluated every 250 steps (ratio to Déjà View's error, 12 fixed views),
  kept state of least val ratio, 3000 steps, 8-12 views per step, loss = error over Déjà View's on
  the same views, `align=exact`, bf16, seed 0, val probe every 10 steps, TensorBoard under
  `runs/<out>/tb/lora-none`. Segment kind `none`: every token without segment.
- **Code:** `SegmentAdapter(lora=r)` in `src/backbone/adapters.py`: swapped into the attentions'
  linears inside each checkpointed loop step (as the attention bias), so the recompute in backward
  uses it too. Tests: zero LoRA = Déjà View on the training path; LoRA = Déjà View with the merged
  weight W + U D (known solution); gradients reach U and no frozen weight; checkpointed gradients =
  unchecked ones. Script: model flags (bias, pool, lora), `lora_rank`, `lr_lora`, kind `none`, peak
  GPU memory per model. Old configs' two flags became three (`lora` false), same training.
- **Ladder** (`docs/experiments.md`), all on the 3080 (`no-long-laptop-runs`):
  - Rungs 0-2: tests above; laptop smoke (bedroom, 4 views, 4 steps, `runs/e52b_smoke`, not of
    record): runs end to end, gradients finite, D's gradient 0 at step 1 as U = 0. Memory and s/step at
    8-12 views read from rung 3 (pooling: 3.1-3.4 GB on the 5090; the 3080 has 10 GB). Over 1 s per
    step is the rung-2 warning; it is the frozen backbone's cost (pooling: 2.3 s on the 3080).
  - Rung 3, overfit the bedroom `ai_006_007` (8 views drawn per step, 200 steps), at lr_lora 1e-4 and
    1e-5. **Pass:** last-50 training ratio below 0.90 and no rise over the last 100 steps (the pooling
    adapter: 0.834 / 0.820).
  - Rung 4, overfit dev (3 scenes, 300 steps) at the larger rung-3 lr that passed. **Pass:** last-100
    ratio below 0.95 on each of the 3 scenes.
  - Rung 5, lr on val (the baseline must be tuned at least as much as what it controls): 500 steps
    at lr_lora 1e-5, 3e-5, 1e-4, the lr of least val ratio at step 500 (no dev scoring). Caveat: E52a's
    400-step sweep missed a later divergence, but that came from the detached Sim(3) (E52d); with
    `align=exact` neither E52a model rose. Then 3000 steps at that lr, dev scored.
  - Rungs 3/4 and the sweep are diagnostics or choices; only rung 5's dev scores are reported.
- **Rung 5, command** (3080, bf16; `<lr>` = the sweep's):
  `uv run python experiments/scripts/e52a_segment_adapters.py --cfg configs/e52a_adapters.yaml --splits dev --set "models={lora: [false, false, true]}" "segments=[none]" lr_lora=<lr> align=exact select=val probe_every=10 --out runs/e52b_control`
- **Reading:**
  - Reported: dev Chamfer (cm) of lora-none against Déjà View and against E52a's fourth-run pool-gt /
    pool-grid (3.330 / 3.326; same 3080, so comparable), per scene and mean; val and held-out curves;
    stability by the probe's rule (held-out not rising after its minimum, training ratio not rising
    over the last 500 steps).
  - It has no positive or negative for segments: it is the bar. E52b's rule: LoRA + pool-gt beats
    Déjà View AND this control AND LoRA + pool-grid by >= 0.05 cm dev mean, on 2 of 3 scenes.
  - If lora-none gains no more than pool-grid, LoRA's capacity adds nothing over a pooling adapter
    for adaptation; E52b still runs (its question is GT vs grid on top of LoRA). If it fails rung 3/4:
    bug checklist, no rung 5.
  - One seed: a control result, not a claim about LoRA in general.
- **Cost:** rungs 3/4 ~10 + ~15 min (2 + 1 runs; s/step unknown until rung 3, est. 2-3 s); sweep
  3 x 500 steps ~1 h; rung 5 ~2-2.5 h + ~5 min dev scoring. Ask before the sweep and before rung 5.

## E52b, control: LoRA on Déjà View's loop block without segments gains 0.35 cm Chamfer over Déjà View on dev (3/3 scenes) and 0.17 cm more than E52a's pool adapters: the bar for E52b and E52c (issue #52, 18.11.2026)
- **Goal, setting:** the plan entry above. All rungs on the 3080 at 3cc909a (clean), one seed (0),
  lr_lora 1e-4 (from the sweep below). Rung 5: 18.11 11:02 to 13:01 (7011 s training, 2.3 s per
  step, then dev scoring), peak 2.21 GB. `--out runs/e52b_control`, log
  `runs/e52b_control/e52b_control.log`, TensorBoard under `runs/e52b_control/tb/`. One CUDA allocator
  OOM warning during dev scoring, as in E52a's fourth run; the allocator retried and every dev score
  was written.
- **Rungs 3 and 4** (18.11 09:33-09:54, `runs/e52b_rung3_1e-4`, `runs/e52b_rung3_1e-5`,
  `runs/e52b_rung4_1e-4`): passed. Rung 3, bedroom, last-50 training ratio: lr 1e-4 0.843 (previous
  50: 0.891), passes; lr 1e-5 0.962 (previous 0.981) falls but misses < 0.90 (too slow, not a bug).
  Rung 4 at 1e-4, dev, last-100 ratio per scene: bedroom 0.910, living room 0.775, lobby 0.706 (all
  < 0.95; the pool adapter: 0.909 / 0.787 / 0.707). 1.71 s per step at 8 views. Diagnostics only.
- **lr sweep** (500 steps each, 09:59-11:00, `runs/e52b_sweep_<lr>`), ratio at step 250 / 500:

  | lr_lora | held-out | val |
  |---|---|---|
  | 1e-5 | 0.9941 / 0.9881 | 0.9906 / 0.9799 |
  | 3e-5 | 0.9870 / 0.9790 | 0.9817 / 0.9661 |
  | 1e-4 | 0.9792 / 0.9752 | 0.9729 / 0.9627 |

  The rule (least val ratio at 500) picks 1e-4. Its margin over 3e-5 (0.0034) is smaller than the
  rerun spread measured below (0.007), so the sweep separates 1e-5 from the other two, not 3e-5 from
  1e-4.
- **Training (rung 5), figure `e52b_control.jpg`, raw `results/e52b_control_train.yaml`:**

  | | held-out min (step) | held-out @3000 | val min (step) | train trend, last 500 steps |
  |---|---|---|---|---|
  | lora-none | 0.9385 (3000) | 0.9385 | 0.9410 (2500) | -0.054 |
  | pool-gt (E52a4) | 0.9803 (3000) | 0.9803 | 0.9710 (3000) | -0.014 |
  | pool-grid (E52a4) | 0.9822 (2750) | 0.9838 | 0.9762 (2500) | -0.017 |

  Stable by the probe's rule (held-out at its minimum at the last eval, training ratio falling). Not
  monotone: held-out and val rise at steps 750-1000 (val 0.956 -> 0.976) and fall again from 1250 on;
  val moves by up to 0.014 between evals (2500 -> 3000: 0.941 -> 0.955 while held-out falls 0.959 ->
  0.939). The training ratio is much noisier than the pool adapters' (50-step means 0.84-0.97): the
  LoRA's gain varies far more from scene to scene. Kept state: step 2500 (val).
- **Rerun noise:** rung 5's first 500 steps repeat the sweep's 1e-4 run (same config, draws, GPU).
  Val ratio at 250: 0.9715 vs 0.9729; at 500: 0.9558 vs 0.9627 (0.007 apart); held-out at 500: 0.9722
  vs 0.9752. This is the rerun spread on val that E52a's fourth run had not measured.
- **Dev, Chamfer (cm, 20 views, 3080, against the same Déjà View run as E52a4):**

  | | living room | bedroom | lobby | mean |
  |---|---|---|---|---|
  | Déjà View | 1.601 | 1.846 | 7.053 | 3.500 |
  | pool-gt (E52a4, step 3000) | 1.567 | 1.800 | 6.622 | 3.330 |
  | pool-grid (E52a4, step 2500) | 1.561 | 1.813 | 6.604 | 3.326 |
  | lora-none (step 2500) | 1.580 | 1.693 | 6.189 | **3.154** |
  | lora-none minus pool-grid | +0.019 | -0.120 | -0.416 | -0.172 |

  lora-none beats Déjà View on 3 of 3 scenes (mean -0.35 cm; the lobby carries 0.86 of the 1.04 cm
  sum), and pool-grid on 2 of 3 (living room +0.019). It moves Déjà View's rays by 0.13-0.24°
  (mean per scene); scored on Déjà View's own cameras, living room is worse than Déjà View (2.267),
  bedroom 1.953, lobby 6.187, so the gain needs its own rays, as pooling's did in E52a's second run.
- **Reading (the plan's rule):** this is the control, with no verdict on segments. lora-none gains
  more than pool-grid (-0.17 cm mean, 2/3 scenes), so the LoRA's capacity adds to adaptation beyond a
  pooling adapter of the same size, mostly on the lobby. One seed and one lr chosen inside the rerun
  spread: a control result, not a claim about LoRA in general.
- **Conclusion:** the bar for #52 moves to 3.154 cm: E52b (LoRA + pool GT vs grid) needs LoRA +
  pool-gt to beat Déjà View AND this control AND LoRA + pool-grid by >= 0.05 cm dev mean on 2 of 3
  scenes. With val moving by up to 0.014 between evals and 0.007 between reruns, a 0.05 cm dev gap is
  only readable with more than one seed; the E52b plan should budget seeds for the control too. Next
  per the agreed order: E52c (LoRA + a cross-view instance-consistency loss), then E52b.
- **Commands:** rung 5 as in the plan with `lr_lora=1e-4`; sweep as in the plan (`--splits` none,
  `steps=500`, `--out runs/e52b_sweep_<lr>`). Figure and raw:
  `uv run python experiments/scripts/e52b_figures.py runs/e52b_control runs/e52a4 runs/e52b_sweep`.

## Plan: E57c = E57a's keep head, trained to predict the distance to the ground-truth surface (issue #57, 17.11.2026)
- **Goal:** issue #57 asks whether a learned filter beats the instance voxel density filter at an equal kept
  fraction. E57a (Pouya) trained a keep logit on a binary label (error in the lower half) and lost badly: it
  ranked pixels well but deleted whole regions. Peter's variant: the head regresses each pixel's distance to the
  ground-truth surface and the points of largest predicted distance are dropped. Does that version beat density,
  and does the answer change what a perfect distance would give?
- **Setting** (`src/refinement/volume.py`, `keep.target: distance`; configs `e57c_overfit_bedroom.yaml`,
  `e57c_overfit_dev.yaml`, `e57c_volume.yaml`): E57a's keep-only model (E41b's data, grid, U-Net, steps, view
  subsets; the depth stays Déjà View's) with the second output trained by L1 on the nearest-GT-point distance of
  the pixel's point in Déjà View's frame (Umeyama of its points onto the GT, the evaluation's alignment), in units
  of the scene's median depth. L1 gives a conditional median, so a point behind a wall does not dominate; no
  threshold or label quantile is involved. Every pixel has a target, including those without GT depth. The kept
  step minimizes the held-out distance L1 (E57a: the kept error); `scene_eval` ranks by the negated prediction at
  keep 0.5, as for E57a. U-Net 1.4M parameters (E41b's; over the ladder's 1M for rung 0, kept to compare with
  E57a).
- **Ladder** (`docs/experiments.md`): rung 1, the target on the bedroom as an image; rung 2, time per step; rung 3,
  overfit `ai_006_007` (1000 steps); rung 4, overfit `dev` (2000 steps); rung 5, train on train + train_extra,
  validate on `val` (curves in TensorBoard under `runs/`, 3 seeds, none and SAM 3 classes). Rungs 3 and 4 are
  diagnostics: nothing from them is a score.
- **Oracle arm** (needs no training): the same cut by the *true* distance on dev and val. It is the ceiling of any
  distance head at keep 0.5. If the ceiling does not beat density in Chamfer, no head can.
- **Evaluation** (`experiments/configs/e57c.yaml`, `e57c_oracle.yaml`; `scene_eval`'s `keep_by`): Déjà View whole,
  kept 0.5 by instance voxel density, kept 0.5 by the head, kept 0.5 by the oracle, on dev and val; Chamfer,
  accuracy, completeness as E57a reports them.
- **Baselines:** Déjà View (dev 3.500, val 5.102 cm), instance voxel density kept 0.5 (val 3.817), flag-refine-filter
  (3.669), E57a's keep head (val 9.49 to 12.37 cm).
- **Expected, written before running:** the head ranks well (rank correlation with the true distance clearly above
  0) but loses to density in Chamfer, as E57a did: the true distance is as smooth over a surface as its error
  label, so the cut still opens holes. The oracle shows how much of the gap is ranking and how much is the cut.
- **Checks:** tests (the target is 0 on the GT and bounded by the shift of a displaced patch, the score drops the
  largest predicted distance, the head overfits a smooth distance on CPU); the untrained head leaves the depth
  bit for bit.

## E57c: a distance head, rungs 0-4 and the oracle -> a perfect distance ranking only ties density on val; rung 5 (the learned head on val) not run (issue #57, 17.11.2026)
- **Commits:** 3461588 (head, configs, tests), af740c1 (plan), this commit (oracle, figure, results); RTX 3080
  shared with another job (5.4 GB taken), so the evaluation ran on the CPU. The runs' `git` says dirty: two
  unrelated E27a figure files were modified in the tree, nothing of E57c.
- **Commands:**
  - `volume-refine --config configs/e57c_overfit_bedroom.yaml --out runs/e57c/overfit_bedroom --reserve-gb 0`, then
    the same with `e57c_overfit_dev.yaml` (rungs 3 and 4; TensorBoard under `runs/e57c/*/tensorboard/`).
  - `uv run python -m experiments.scripts.e57c_oracle` (the true distance of every dev and val pixel, 14 scenes),
    `CUDA_VISIBLE_DEVICES= scene_eval --cfg experiments/configs/e57c_oracle.yaml --splits dev val`,
    `e57c_table.py runs/e57c_oracle-20261117-200725 > results/e57c_oracle.txt`, `e57c_figures.py --head
    volume-dist-none > results/e57c_figures.txt`.
- **Ladder:** rung 0, 1.44M parameters (E41b's U-Net, kept to compare with E57a; over the 1M the ladder asks for);
  rung 1, the target as an image (below): zero on well-reconstructed surfaces, bright at furniture outlines and
  on whole wall patches Déjà View misplaces; rung 2, 0.19 s per step (batch 100 000 pixels, 8-20 views, 128
  cells, 20 evaluations included), target computation 1.6 s per scene (Umeyama 0.3 s, nearest GT point 1.3 s);
  rung 3 (bedroom, 1000 steps): held-out = train distance L1 0.00719 -> 0.00085 (median depths), rank correlation
  with the true distance 0.94, still falling; rung 4 (dev, 2000 steps): 0.00639 -> 0.00102, correlation 0.89,
  still falling. Both overfit (an 8x and 6x fall, not 0, in 1000 and 2000 steps), and the outputs differ per
  scene (the three dev scenes' mean distances are 0.3, 0.7 and 0.9 %). Diagnostics: not scores.
- **Oracle results** (Chamfer cm, accuracy / completeness / Chamfer, means over scenes, Déjà View K = 16, 20 views
  at 504 px, kept 0.5 per scene; every Déjà View and density number repeats E27a and E57a):

  | split | Déjà View | instance voxel density | true distance (oracle) |
  |---|---|---|---|
  | dev (3 scenes) | 4.675 / 2.324 / 3.500 | 2.998 / 3.597 / 3.298 | 1.521 / 3.931 / 2.726 |
  | val (11 scenes) | 6.061 / 4.142 / 5.102 | 3.013 / 4.621 / 3.817 | 1.510 / 6.401 / 3.955 |

  The oracle beats density in Chamfer on 3 of 3 dev scenes and on 5 of 11 val scenes; on val it is 0.14 cm worse in
  the mean (ai_009_009 alone 2.4 cm worse, ai_004_003 0.8). A perfect ranker is far the more accurate (1.51 against
  3.01 cm) and loses it in completeness (6.40 against 4.62).
- **Figure:** `figures/e57c_bedroom.jpg` (views 18, 0, 12, 6): the true distance and the rung 4 head's prediction
  (trained on this scene) agree closely; the pixels the true distance drops are whole surfaces (the wall and the
  door in views 0 and 6, whole walls), while density drops pixels scattered over the image. Dropping the least
  accurate half therefore leaves holes the GT then misses; dropping the densest half removes redundancy.
- **Conclusion:**
  - What is established: the cut at keep 0.5 by the *true* distance gains 0.57 cm on dev and loses 0.14 cm on
    val against instance voxel density at the same q. That is one q; the accuracy oracle's curve on dev and val (`results/e57c_oracle_curve_dev_val.txt`, same script's first version, 4 min 57 s) shows its
    Chamfer is lowest at q 0.7 (dev) and 0.8 (val), where it is 29% and 36% below Déjà View, and density was
    not run at those q, so "a distance head cannot beat density" is not shown. It stands at q 0.5 only.
  - What is not established: that the trained head loses. Rung 5 has not run: this checkout holds dev, val and
    three other scenes, no train scene, and the job (3 seeds, 20k steps, ~1.5 h each on a shared GPU) is past
    the 30-minute limit. Until it runs, the learned head is "not yet run", and the ceiling is a bound on what
    ranking by distance can do, not a measurement of the head.
  - The oracle bounds only rankings by distance; a head trained for completeness-aware cuts is not covered.
- **Limits:** one cut (0.5); dev and val only (test untouched); the oracle's val gap to density (0.14 cm) is
  inside what 11 scenes with one very hard scene (ai_009_009, Chamfer 11 to 16 cm) can move; the ladder's rungs 3
  and 4 were one seed each.

## Plan: E57c, three oracles = Déjà View cut by each pixel's true accuracy, completeness and Chamfer (issue #57, 17.11.2026)
- **Goal:** the accuracy oracle (true distance of the pixel's point to the GT) lowers Déjà View's Chamfer by about a
  third on dev and val (best q 0.7 to 0.8), but Chamfer rises again below that q. Peter's question: rank by the
  quantity the evaluation scores. How far do a completeness oracle and a Chamfer oracle get?
- **Definitions** (Déjà View's full, unfiltered cloud, aligned by one Umeyama onto the GT as the evaluation does;
  every pixel has one prediction and one GT point, so both map 1:1 to pixels): accuracy a_i = distance of pixel i's
  predicted point to the nearest GT point; completeness c_i = distance of pixel i's GT point to the nearest
  predicted point; Chamfer (a_i + c_i) / 2. Each oracle drops the pixels of largest value: scores -a_i, -c_i,
  -(a_i + c_i) / 2 (higher keeps). Completeness here is read off the unfiltered cloud, not recomputed for the kept
  set; recomputing it per q is the later step.
- **Setting:** `experiments/scripts/e57c_oracle_curve.py`, dev only (3 scenes), q = 1, 0.9 ... 0.1; each kept set is
  realigned and scored as `scene_eval` scores the scene (accuracy, completeness, Chamfer). Ranking over the pixels
  that have a GT point (all but 0.01% on dev). No density, no label scores.
- **Expected, written before running:** the accuracy oracle repeats the previous run (best Chamfer 2.479 at q 0.7).
  The completeness oracle drops pixels the cloud does not cover well, which are mostly bad pixels too, so it lowers
  Chamfer as well, but less than the accuracy oracle on accuracy. The Chamfer oracle is best or tied at most q.

## E57c, three oracles: ranking by the pixel's completeness or Chamfer is worse than by its accuracy at every q (dev, 17.11.2026)
- **Commands:** `CUDA_VISIBLE_DEVICES= uv run python experiments/scripts/e57c_oracle_curve.py --splits dev --per-scene`
  -> `results/e57c_oracles_dev.txt`, 4 min 9 s on the CPU (3 scenes, 3 oracles x 10 q, each kept set realigned and
  scored); the tree is dirty (unrelated E27a figure files, this script and plan uncommitted).
- **Results** (dev, mean of 3 scenes; Chamfer cm, accuracy / completeness in cm in the file; q 1.0 is Déjà View, 3.500):

  | q | accuracy oracle | completeness oracle | Chamfer oracle |
  |---|---|---|---|
  | 0.9 | 2.724 | 3.323 | 2.813 |
  | 0.8 | 2.514 | 3.493 | 2.762 |
  | 0.7 | **2.479** | 3.742 | 2.851 |
  | 0.6 | 2.546 | 3.966 | 3.052 |
  | 0.5 | 2.726 | 4.207 | 3.375 |
  | 0.3 | 3.453 | 5.051 | 4.463 |
  | 0.1 | 5.444 | 6.702 | 7.786 |

  The accuracy oracle repeats the earlier run (best 2.479 at q 0.7). The completeness oracle is below Déjà View only
  at q 0.9 (3.323) and above it from q 0.7 on; its accuracy at q 0.5 is 2.70 cm (accuracy oracle 1.52) and its
  completeness 5.72 (3.93). The Chamfer oracle's best is 2.762 at q 0.8, 0.28 cm above the accuracy oracle's.
- **Earlier run, accuracy oracle only, dev and val** (`results/e57c_oracle_curve_dev_val.txt`; its own mean line
  pools both splits, the split means are computed from its per-scene lines): Chamfer at best q, dev 2.479 (q 0.7),
  val 3.270 (q 0.8), against Déjà View's 3.500 and 5.102; every scene's best q is between 0.6 and 0.9.
- **Against what was expected (written before running):** the completeness oracle was expected to lower Chamfer
  at moderate q, and the Chamfer oracle to be best or tied; neither holds. 
- **Why (read from the definitions, not tested):** c_i of the unfiltered cloud is large where the GT is not covered
  by any prediction nearby; those pixels are the ones whose neighbours' points are the nearest points of the
  uncovered GT, so dropping them uncovers more GT. It measures a coverage defect, not what removing the pixel
  gains. Averaging it with a_i dilutes a_i, the signal that works.
- **Limits:** dev only (3 scenes); completeness read off the unfiltered cloud, not recomputed per q (the planned next
  step); one scene's pixels without GT (0.01%) are not ranked; no density row, so no equal-q comparison with it.

## Plan: E57c, marginal oracle = drop a point iff its accuracy gain exceeds the completeness it costs (issue #57, 17.11.2026)
- **Goal:** Peter's simple Chamfer oracle: per point, the accuracy gain from dropping it minus the completeness
  drop. The earlier (a_i + c_i) / 2 oracle (c_i of the unfiltered cloud) lost to the accuracy oracle at every q.
- **Definition** (full aligned cloud, one removal at a time): dropping point i lowers the accuracy mean by
  (a_i - mean a) / (K - 1) and raises the completeness mean by m_i / N, m_i = sum of (d2 - d1) over the GT points
  whose nearest predicted point is i (d1, d2 their nearest and second nearest). With K ~ N it helps Chamfer iff
  a_i - mean a > m_i. Score (higher keeps) m_i - a_i, cut at q as before; also the cut it defines itself, a_i - mean a
  > m_i, which needs no q. First order: neighbours that cover each other both have m_i ~ 0, and dropping both uncovers
  more than the sum.
- **Setting:** `e57c_oracle_curve.py --oracles marginal`, dev, q = 1 ... 0.1, no density.
- **Expected, written before running:** below the accuracy oracle at fixed q (it drops redundant neighbour pairs
  together); its own cut keeps a share of the pixels near q 0.7 to 0.9 and lands close to the accuracy oracle's best.

## E57c, marginal oracle: dropping a point iff its accuracy gain exceeds the completeness it costs -> dev Chamfer 3.500 -> 2.202 (q 0.6), 2.238 at its own cut, below the accuracy oracle at every q <= 0.9 (dev, 17.11.2026)
- **Commands:** `CUDA_VISIBLE_DEVICES= uv run python experiments/scripts/e57c_oracle_curve.py --splits dev --per-scene
  --oracles marginal` -> `results/e57c_oracle_marginal_dev.txt`, 1 min 40 s (CPU). Tree dirty as before.
- **Results** (dev, mean of 3 scenes; accuracy / completeness / Chamfer, cm; Déjà View 4.675 / 2.324 / 3.500):

  | q | marginal oracle | accuracy oracle (Chamfer only) |
  |---|---|---|
  | 0.9 | 3.009 / 2.324 / 2.667 | 2.724 |
  | 0.7 | 1.987 / 2.526 / 2.257 | 2.479 |
  | 0.6 | 1.720 / 2.683 / **2.202** | 2.546 |
  | 0.5 | 1.530 / 2.899 / 2.214 | 2.726 |
  | 0.3 | 1.295 / 3.516 / 2.406 | 3.453 |
  | 0.1 | 1.280 / 4.358 / 2.819 | 5.444 |

  Its own cut (a_i - mean a > m_i, no q chosen) keeps 0.678 of the pixels (0.706, 0.653, 0.674 per scene) and scores
  1.914 / 2.563 / **2.238**; per scene Chamfer 1.00, 1.31, 4.40 against Déjà View's 1.60, 1.85, 7.05.
- **Against what was expected (written before running):** it was expected below the accuracy oracle at fixed q because
  neighbours that cover each other get dropped together; it is above it (better) at every q from 0.9 to 0.1, and its
  own cut lands at q 0.68, as expected. The first-order marginals cost little here.
- **Reading:** a point costs completeness only if it is the nearest of GT that nothing else is near, so dropping
  the points with m_i ~ 0 and large a_i first leaves completeness at 2.32 cm down to q 0.9, where the accuracy oracle
  already loses 0.1. The best Chamfer is 37% below Déjà View's (accuracy oracle's best: 29%).
- **Limits:** dev only (3 scenes), no val; the optimum is unknown (the exact problem is a facility-location
  problem, NP-hard in general, and this is the first step of its drop heuristic), so 2.202 is not a lower bound; no density row
  at equal q; oracle scores use GT.

## Plan: E57c, oracle zoo = every ranking of Déjà View's pixels we could think of, top 5 swept against density and confidence (issue #57, 17.11.2026)
- **Goal (Peter, overnight, no questions possible):** list all oracles (rankings that use the GT, no training), rank
  them, sweep the top 5 over q on dev, and compare with the two real rankings that exist today, instance voxel density
  (E45) and Déjà View's own confidence, at equal q (neither was measured at q != 0.5 so far).
- **Brainstorm and literature.** Sparsification plots with an oracle that sorts by the true error are standard for
  uncertainty (Ilg et al. 2018 for flow; AUSE = area between the method's curve and the oracle's): our accuracy
  oracle. MVS point filters are learned confidences or classifiers (DeepC-MVS, Kuhn et al. 3DV 2020; DELS-MVS,
  Sormann et al.; per-class outlier classifiers for urban MVS, ISPRS 2018). Subset selection for Chamfer is
  k-median / facility location (monotone submodular variants have a 1 - 1/e greedy guarantee, not this signed
  objective); no paper found that selects a point subset by Chamfer. Facility-location facts and these references
  come from a web search, not read in full.
- **Candidates** (pixel i, a_i accuracy, m_i completeness cost of dropping it alone):
  1. accuracy -a_i (done: dev best 2.479); 2. completeness -c_i (done, worse); 3. (a_i + c_i)/2 (done, worse);
  4. marginal m_i - a_i, one shot (done: 2.202 at q 0.6); 5. reverse greedy on the exact gain, recomputed every round
  (10% steps; ends in `greedy-own` when each round drops all positive-gain pixels); 6. same-pixel GT error e_i
  (what Déjà View's confidence loss targets, conf* = alpha / (gamma E[e_i])); 7. window-averaged accuracy (what a head
  that sees neighbourhoods but not single pixels could reach: separates variance from smooth bias); 8. F-score at
  tolerance tau: optimal is a hard cut a_i > tau (a pixel with a_i > tau covers no GT within tau), the same ranking as 1;
  9. exact subset: k-median with opening costs, NP-hard in general, only the greedy path is tried; 10. swap local
  search from the greedy solution: a refinement of 5, not run; 11. oracles by view or instance (drop a whole
  segment): not a pixel ranking, left out.
- **Ranking of the candidates before running** (potential for Chamfer, and how learnable the score is from the
  prediction alone): 5 and 4 best potential; 1 the basis of any distance head and the sparsification standard; 6 the
  only one a confidence loss can reach as it stands; 7 the realistic cap of a head with a receptive field; 2, 3
  measured worse than 1, 8 equals 1, 9 and 10 cost more than they tell. **Top 5 swept: accuracy, marginal, greedy
  (10% steps) with greedy-own, pixel, smooth (7 x 7).** Rankings run with them: density, confidence.
- **Setting:** `e57c_oracle_curve.py --splits dev --per-scene`, CPU, q = 1 ... 0.1, Chamfer averaged over q as an
  AUSE-like area. Tree dirty.
- **Expected, written before running:** greedy <= marginal <= accuracy in Chamfer at every q; pixel close to
  accuracy (slightly worse); smooth clearly worse than accuracy at small q if flying points are single pixels, close to
  it if the errors are smooth; confidence between density and accuracy oracle; density no better than the confidence
  at q <= 0.7. The plan was written a few minutes after the run was started; nothing had been read from it yet.

## E57c, oracle zoo: a greedy Chamfer oracle reaches 2.0 cm on dev (-43%), the accuracy oracle 2.48, density 3.19, Déjà View's confidence never beats the uncut cloud (dev, 17.11.2026)
- **Commands:** `CUDA_VISIBLE_DEVICES= uv run python experiments/scripts/e57c_oracle_curve.py --splits dev
  --per-scene` -> `results/e57c_oracle_zoo_dev.txt` (11 min, CPU; 200 s per scene, almost all of it the density
  baseline and SAM 3 linking); `--rankings greedy --grid 0.7 0.5 0.3 --step 0.05` and `--grid 0.6 0.5 0.4 --step 0.025`
  -> `results/e57c_greedy_steps_dev.txt` (30 s per scene). Dev (3 scenes, 20 views at 504 px, Déjà View K = 16), one
  Umeyama per cut, cm. Tree dirty as before.
- **Results** (dev mean; Chamfer in cm by q, Déjà View uncut 3.500 = accuracy 4.675 / completeness 2.324):

  | ranking | q 0.9 | 0.7 | 0.6 | 0.5 | 0.4 | 0.3 | 0.1 | best | mean over q 0.1-1 |
  |---|---|---|---|---|---|---|---|---|---|
  | greedy, 10% steps (oracle) | 2.667 | 2.181 | 2.081 | 2.048 | 2.057 | 2.099 | 2.277 | **2.048** (0.5) | 2.343 |
  | marginal, one shot (oracle) | 2.667 | 2.257 | 2.202 | 2.214 | 2.283 | 2.406 | 2.819 | 2.202 (0.6) | 2.528 |
  | accuracy (oracle) | 2.724 | 2.479 | 2.546 | 2.726 | 3.028 | 3.453 | 5.444 | 2.479 (0.7) | 3.254 |
  | pixel: same-pixel GT error (oracle) | 2.791 | 2.722 | 3.013 | 3.560 | 4.359 | 5.347 | 10.549 | 2.713 (0.8) | 4.549 |
  | smooth: accuracy, 7 x 7 mean (oracle) | 2.848 | 2.972 | 3.334 | 3.925 | 4.781 | 6.777 | 17.094 | 2.769 (0.8) | 5.770 |
  | instance voxel density | 3.429 | 3.277 | 3.188 | 3.299 | 5.307 | 7.757 | 53.873 | 3.188 (0.6) | 10.221 |
  | Déjà View confidence | 4.289 | 12.753 | 17.884 | 25.376 | 31.955 | 41.512 | 99.769 | 3.500 (1.0) | 30.827 |

  Greedy step size (K rounds to q 0.5): one shot (K = 1) 2.214, 10% (K = 5) 2.048, 5% (K = 10) 2.020, 2.5% (K = 20)
  1.993: converged near 2.0 (-43%). `greedy-own` (drop all positive gains each round, 16 rounds) ends at q 0.159 with
  2.348, worse: simultaneous drops overshoot. The one-shot marginal's own cut: 2.238 at q 0.678. Per scene
  (ai_010_007 / ai_006_007 / ai_044_001; uncut 1.601 / 1.846 / 7.053) the best Chamfer of greedy 0.87 / 1.04 / 4.12,
  marginal 0.93 / 1.21 / 4.39, accuracy 1.07 / 1.41 / 4.92, density 1.44 / 1.69 / 6.36, confidence 1.60 / 1.81 / 6.98:
  the order greedy < marginal < accuracy < density < confidence holds in each scene.
- **Against what was expected (written before running):** greedy <= marginal <= accuracy at every q: yes. Pixel
  close to accuracy: no, much worse (2.713 best, 3.56 at q 0.5 against 2.73). Smooth clearly worse at small q: yes,
  but its accuracy hardly suffers (q 0.5: 1.626 against 1.521); the cost is completeness (6.22 against 3.93).
  Confidence between density and the accuracy oracle: wrong, it is far below density; its accuracy is better than
  density's at equal q (q 0.5: 2.220 against 3.001, as in E45) but its completeness collapses (48.5 cm at q 0.5), so
  at no q does it beat the uncut cloud in Chamfer.
- **Reading.**
  1. The step from accuracy (2.48) to marginal/greedy (2.0-2.2) is the completeness term m_i: what a pixel contributes
     that no other pixel, in any view, covers. Pixel-wise scores (accuracy, same-pixel error, confidence) cannot see it; only
     a score that depends on the other views' points can.
  2. Confidence and density drop whole surfaces (completeness up to 100+ cm at q 0.1) where the oracles drop redundant
     points of the same views; a plausible cause, not tested here, is that their low scores sit on the same surface in every view, so a cut removes what nothing else covers.
  3. Same-pixel error is a worse ranking than distance to the nearest GT point (a point sliding along a surface is not
     wrong): the target of Déjà View's confidence loss, taken to its limit (conf* = alpha / (gamma E[e_i]), E57c
     discussion), ranks like `pixel` at best, not like `accuracy`, and not like greedy.
  4. The accuracy gain survives 7 x 7 smoothing, the completeness does not: accuracy errors are spatially coherent
     enough for a network with context; deciding what is redundant needs per-pixel resolution across views.
- **Design consequence (a recommendation, not a measurement).** A better confidence loss for the backbone would stay
  pointwise, so it cannot reach m_i, and it needs the backbone retrained; a post-hoc filter head sees every view's
  points at once (E41b's volume features do), works on any backbone's output (Déjà View, Solv3R, SegVGGT), and
  can be trained on the one-shot marginal label (`a_i - mean a > m_i`, binary, its own cut ~ q 0.68, no q to choose; the
  greedy path as a soft target is a refinement) or the greedy gain. Next experiment: E57a's head with that label,
  rungs 3-5, against density at equal q on val, with GT labels computed in about 1 min per scene.
- **Limits:** dev only (3 scenes), no val, one seed (oracles have none); every oracle uses GT and none is trained, so
  none is a result of a method. The GT cloud is the depth of the same 20 views, so completeness is measured on points other
  views also cover: the m_i term, and with it greedy's edge, may be smaller against a mesh of the whole scene.
  Greedy is a heuristic: 1.993 is not a lower bound. Facility-location and literature statements above are from a web
  search, not read in full. DV's training config was read from the installed `dvlt` package, not the released checkpoint's.

## E57c, clouds: the bedroom's points after each ranking's cut at its best q (dev q, bedroom ai_006_007, 18.11.2026)
- **What:** `uv run python -m experiments.scripts.e57c_clouds > experiments/results/e57c_clouds.txt` (1 min, GPU
  for the splats) -> `figures/e57c_clouds_{dollhouse_colors,dollhouse_shaded,view18_pulledback_colors,view0_...,view12_...,view6_...}.jpg`.
  Each panel: the pixels a ranking keeps of Déjà View's 20-view cloud, aligned onto the GT, in image colors (the
  dollhouse also shaded), same camera in every panel. q = the ranking's best dev-mean q from
  `results/e57c_oracle_zoo_dev.txt` (greedy 0.5, marginal 0.6, accuracy 0.7, same-pixel error 0.8, smoothed accuracy
  0.8, density 0.6). Déjà View's confidence is cut as `dvlt.viz.pointcloud.build_pointcloud` does (drop the pixels
  below a percentile of the confidence; the viewer's default is 50), not at a best q, because on dev no q beat the
  uncut cloud. I took that from the installed package's viewer, not from the paper, which I have not read.
- **Bedroom Chamfer of each panel (cm; Déjà View uncut 1.846):** greedy 1.135, marginal 1.276, accuracy 1.429,
  same-pixel 1.595, smooth 1.570, density 1.725, confidence at 0.5 **6.293**. The order matches dev's.
- **What the figures show** (looked at the dollhouse in colors and shaded and the interior view 18; views 0, 12, 6 saved, not
  looked at): the oracles remove the debris above the floor and in front of the bed; confidence at 0.5 leaves
  white holes in the floor, the right wall and the ceiling side, i.e. whole surfaces, while the bed stays; density keeps most
  of the floor debris; the streaks between the bedpost and the camera in view 18 survive every cut, the greedy one
  included. Why was not looked into.
- **Limits:** one scene, chosen as the presentation scene before this run, not a sample; dev's best q, not the bedroom's own.

## Plan: E57c, confidence in the greedy oracle = (A) greedy with the accuracy estimated as kappa / conf, (B) confidence cut first, then the oracle's points back (issue #57, 18.11.2026)
- **Goal (Peter):** (A) how much of the greedy oracle's 2.05 cm survives when only its accuracy term is estimated from Déjà View's confidence
  (conf* = alpha / (gamma E[error]) -> error estimate kappa / conf; m_i still true); (B) cut by confidence, then add back the pixels the
  greedy oracle keeps if they lie farther than delta from the confidence-kept cloud. The gaps this leaves in the surface are for later.
- **Setting:** `e57c_hybrid.py --splits dev`, CPU, dev 3 scenes, 20 views, cm, the evaluation's cuts. A: kappa one scalar per scene,
  mean(kappa / conf) = the scene's true mean accuracy (a GT statistic, disclosed), also x0.5 and x2 (sensitivity, not tuning:
  all three reported), q = 0.9 ... 0.1, with the true-a greedy in the same run. B: q_c in {0.5, 0.7, 0.9} (0.5 = the viewer's default),
  greedy q_g = 0.5, delta in {0, 1, 2, 4, 8, inf} cm (0 = union, inf = confidence alone); control `fill`: every pixel farther than
  delta is added, oracle or not (shows whether the oracle's information matters beyond hole filling).
- **Expected, written before running:** A: between the accuracy oracle (best 2.479) and true-a greedy (2.048), say 2.3-2.8, so below density
  (3.188); not better than the true-a greedy at any q. B: far below the confidence cut's 25.4 at q_c 0.5; the best delta at 2-4 cm with
  Chamfer near 2.5-3.5; `fill` clearly worse than the oracle's points at the same delta; the union (delta 0) keeps the oracle's
  accuracy gain but is the worst for the share kept.

## E57c, confidence in the greedy oracle: (A) kappa / conf as the accuracy gives 2.70-2.88 cm (true-a greedy 2.05, density 3.19); (B) confidence first and the oracle's points back never beats the oracle alone (dev, 18.11.2026)
- **Commands:** `CUDA_VISIBLE_DEVICES= uv run python -m experiments.scripts.e57c_hybrid --splits dev` ->
  `results/e57c_hybrid_dev.txt` (9 min, CPU, 3 min per scene, mostly density and SAM 3 linking in `rank_scores`). Dev, cm,
  Déjà View uncut 3.500 (4.675 / 2.324). Tree dirty.
- **A. Greedy with the accuracy from confidence** (true m_i; kappa per scene from the true mean accuracy; Chamfer, best q in brackets):

  | accuracy used | q 0.9 | 0.7 | 0.5 | 0.3 | best (q) |
  |---|---|---|---|---|---|
  | true a_i (greedy oracle) | 2.667 | 2.181 | 2.048 | 2.099 | 2.048 (0.5) |
  | kappa / conf x0.5 | 2.930 | 2.763 | 2.727 | 2.806 | 2.704 (0.6) |
  | kappa / conf x1 | 3.021 | 2.941 | 2.919 | 3.068 | 2.882 (0.6) |
  | kappa / conf x2 | 3.141 | 3.162 | 3.176 | 3.401 | 3.117 (0.6) |
  | for reference: accuracy oracle / density | 2.724 / 3.429 | 2.479 / 3.277 | 2.726 / 3.299 | 3.453 / 7.757 | 2.479 / 3.188 |

  At q 0.5, x1: accuracy 2.19 (true-a greedy 1.53), completeness 3.65 (2.56): the estimated error ranks worse, and the true m_i
  cannot make up for it. All three kappa scales are below density and the uncut cloud at every q <= 0.9, none below the accuracy
  oracle's best (2.479, true a, no m). Smaller kappa (more weight on m_i) is better at all three values; the trend ends at least
  below x0.5, not searched.
- **B. Confidence cut at q_c, then the greedy oracle's pixels (q_g 0.5) farther than delta from the kept cloud** (kept share, Chamfer):

  | q_c | delta 0 (union) | 1 cm | 2 cm | 4 cm | 8 cm | inf (confidence alone) | `fill` (any pixel) delta 1 / 8 cm |
  |---|---|---|---|---|---|---|---|
  | 0.5 | 0.696, 2.524 | 0.654, 2.551 | 0.625, 2.589 | 0.603, 2.644 | 0.588, 2.716 | 0.500, 25.376 | 0.920, 3.541 / 0.756, 3.652 |
  | 0.7 | 0.806, 2.666 | 0.781, 2.684 | 0.762, 2.704 | 0.747, 2.733 | 0.737, 2.781 | 0.700, 12.753 | 0.950, 3.517 / 0.839, 3.552 |
  | 0.9 | 0.926, 2.889 | 0.920, 2.893 | 0.914, 2.900 | 0.909, 2.913 | 0.906, 2.939 | 0.900, 4.289 | 0.987, 3.496 / 0.941, 3.390 |

  Per scene, q_c 0.5 union: 1.08 / 1.40 / 5.09 (ai_010_007 / ai_006_007 / ai_044_001), against uncut 1.60 / 1.85 / 7.05 and the greedy
  oracle alone at q 0.5: 0.87-1.04-4.12 (best per scene).
- **Against what was expected (written before running):** A: 2.3-2.8, between the accuracy oracle and the true-a greedy, below density:
  x0.5 inside (2.70), x1 just outside (2.88), x2 outside (3.12); all below density, none below the true-a greedy, as expected. B: far
  below confidence's 25.4: yes; best delta 2-4 cm, Chamfer 2.5-3.5: wrong, the best delta is 0 (every oracle pixel back, 2.524) and
  Chamfer rises monotonically with delta; `fill` clearly worse than the oracle's points: yes (3.4-3.65 against 2.5-2.9, close to the
  uncut 3.50).
- **Reading.**
  1. Confidence as the accuracy estimate keeps about half of the greedy oracle's gain over the uncut cloud at its best q
     ((3.50 - 2.70) / (3.50 - 2.05) = 55% at x0.5, 43% at x1) and beats density (3.19) once the completeness term is known; the
     completeness term must be estimated for it to count, and this run does not do that.
  2. B is dominated by the oracle alone at equal share (union 0.696: 2.524; greedy at 0.7: 2.181): the confidence-kept pixels
     contain many that the oracle would drop (union accuracy 2.45 against 1.99 at the same share). The more is cut by confidence
     first, the better the union (q_c 0.5 < 0.7 < 0.9), towards the oracle alone. Confidence as a first stage costs; as a feature
     of a learned oracle-label head it may still help.
  3. Filling the holes by distance alone brings the uncut cloud back (3.4-3.65, uncut 3.50): the distance carries no information about
     which pixels are right; what helps is the oracle's selection, so the gaps in the surface are the oracle's job, not a geometric one.
- **Limits:** dev only (3 scenes), no val; A's kappa uses one GT statistic per scene; the scale was not searched beyond three values;
  B's oracle points use the GT, so it is a ceiling of "confidence first", not a method; the gaps in the surface that B leaves were not
  measured separately from completeness.

## E57c, clouds of the confidence hybrids: the bedroom after A and B (18.11.2026)
- **What:** `uv run python -m experiments.scripts.e57c_clouds_hybrid > experiments/results/e57c_clouds_hybrid.txt` ->
  `figures/e57c_clouds_hybrid_{dollhouse_colors,dollhouse_shaded,view18_pulledback_colors,view0_...,view12_...,view6_...}.jpg`; same cameras and splats as
  `e57c_clouds` (its drawing is now `render`). Bedroom Chamfer (cm; uncut 1.846): greedy true a q 0.5 1.135; A kappa/conf x0.5 q 0.6 1.447,
  x1 q 0.6 1.500; B confidence 0.5 + oracle, delta 0: 1.400 (kept 0.70), delta 8 cm: 1.646 (0.57); control, any pixel farther than 8 cm:
  2.064 (0.71); confidence alone 6.293 (0.50). The same order as dev's.
- **Looked at** the dollhouse in colors and the interior view 18 (views 0, 12, 6 saved, not looked at): A's clouds are as clean as the
  oracle's above the floor and in front of the bed, with a few holes in the bed cover in view 18; B with delta 8 cm shows the gaps Peter
  expected as thin white seams, 8 cm wide, along the confidence cut's boundary (a ring around the wall hole in view 18), and B's delta 0 is
  seamless; the control brings the debris above the floor back. One scene, dev's q.

## Plan: E57c, verifier = the SAM 3 masks carry the confident pixels across the confidence holes, Déjà View's own points come back where they agree (issue #57, 18.11.2026)
- **Goal (Peter):** fill the confidence cut's gaps and the band seams of `e57c_hybrid` B with the segmentation masks, without worsening what the
  confidence cut already covers; visualize confidence with the cut as an isoline next to the oracle values.
- **Idea:** E58a filled the flagged pixels' depth from the kept ones (weighted Laplace membrane, SAM 3 edges cut) and lost: Déjà View's own
  depth at those pixels (3.20 % error) beats every fill (3.61 % SAM 3). So no new depth: the membrane only verifies. A dropped pixel returns with
  its own Déjà View point iff |membrane depth / Déjà View depth - 1| <= eps; a known (confident) pixel is never touched; a flying point off the
  extrapolated surface stays out. No GT, no oracle in the rule.
- **Setting:** `e57c_verify.py`; dev, cm, the evaluation's cuts; q_c in {0.5 (viewer default), 0.7}, eps in {0.5, 1, 2, 4} % (all reported, not
  tuned; E58a's fill error within SAM 3 instances is about 1 %); diagnostics: mean a_i of the pixels that come back, their share inside the greedy
  oracle's keep set (q 0.5). `--figure`: bedroom views 18, 0, 12, 6, columns image + SAM 3 borders, confidence, a_i, m_i, greedy keep, membrane
  residual, the isoline in each.
- **Expected, written before running:** Chamfer far below the confidence cut alone (dev 25.4 at q_c 0.5, 12.8 at 0.7) and the best eps at 1-2 %;
  not below the oracle fill's union (2.524, 2.666), and about at density (3.19); the pixels that come back more accurate than the dropped ones
  that stay out, and more often in the oracle's set than the 0.5 base rate. Failure mode I expect: planes only, so objects and the bed
  stay holes, and flying points close to a wall pass the test.
- **Added before the dev run, after looking at the bedroom figure** (`results/e57c_verify_bedroom.txt`, `figures/e57c_verify_bedroom.jpg`: bedroom
  Chamfer of the confidence cut at 0.5 6.293 -> 3.546 at eps 2 %, 3.109 at 4 %, uncut 1.846; the pixels back are in the oracle's set 0.45-0.47 of the time):
  the confidence isoline differs strongly between views (view 6's room is almost all on one side, view 0's almost all on the other), and the
  membrane's residual is small near the known pixels and large far from them (extrapolation to the image border). So two more confidence cuts are
  scored: confidence ranked within each view (q 0.5) and within each SAM 3 instance (q 0.5), same eps. Expected: both lose less completeness than
  the global cut (dev 48.5 at q 0.5); the verifier brings Chamfer down further but not below the uncut 3.500.

## E57c, verifier result: the masks' membrane does not rescue the confidence cut on dev; per-instance ranking helps most, no cut reaches the uncut 3.500 (dev, 18.11.2026)
- **Commands:** `uv run python -m experiments.scripts.e57c_verify --figure` (bedroom, 1 min) and `CUDA_VISIBLE_DEVICES= uv run python
  -m experiments.scripts.e57c_verify --splits dev` (5 min) -> `results/e57c_verify_{bedroom,dev}.txt`, `figures/e57c_verify_bedroom.jpg`. Dev, cm. Tree dirty.
- **Results** (dev mean Chamfer, kept share in brackets; uncut 3.500, density best 3.188, oracle union 2.524):

  | confidence cut | alone | + verifier eps 1 % | 2 % | 4 % |
  |---|---|---|---|---|
  | global q 0.5 | 25.376 (0.50) | 15.893 (0.57) | 14.973 (0.61) | 14.037 (0.65) |
  | global q 0.7 | 12.753 (0.70) | 10.092 (0.75) | 9.650 (0.78) | 8.883 (0.81) |
  | per view q 0.5 | 23.695 (0.50) | 16.958 (0.59) | 14.353 (0.63) | 13.099 (0.69) |
  | per instance q 0.5 | 5.354 (0.50) | 4.798 (0.61) | 4.674 (0.65) | 4.490 (0.71) |

  Per scene (eps 2 %; ai_010_007 / ai_006_007 / ai_044_001): global 0.5 4.38 / 3.55 / 36.99, global 0.7 3.34 / 2.37 / 23.25, per instance 2.24 / 2.45 / 9.34;
  uncut 1.60 / 1.85 / 7.05. The mean accuracy of the pixels that come back is 5.0-5.7 cm; the dropped ones average 7.1 (global 0.5) and 5.75 (per
  instance) cm, derived from the uncut 4.675 and the kept accuracy, so the check is somewhat selective for the global cut and not for the per-instance one.
- **Against what was expected:** Chamfer far below the confidence alone: yes (25.4 -> 14.0); best eps 1-2 %: no, it falls monotonically to 4 %, the
  largest tried; not below the oracle union and about at density: no, far above both and above the uncut cloud; per view / per instance lose less
  completeness: per instance yes (7.1 cm against 48.5), per view barely (45.1); the verifier below uncut: no, at no setting.
- **Reading** (bedroom figure, all four presentation views): the confidence is a smooth per-view map and its median isoline cuts whole regions per view
  (view 6's room mostly on one side, view 0's on the other); the oracle drops most of views 0 and 6 and keeps other views of the same surfaces (a view
  selection per surface patch, m_i being the cross-view redundancy); [correction, see the seams entry below: its keep mask is not blockwise per (view, instance),
  only 15.5 % of the pixels lie in blocks it keeps < 10 % or > 90 % of]. The membrane residual is small only near the
  known pixels and grows to > 30 % toward the image border (extrapolation), so the verifier fills bands and holes between known pixels, not large
  regions, and ai_044_001, where the confidence leaves most of the scene out, stays at 9-37 cm.
- **Limits:** dev, 3 scenes; eps not tuned beyond four values and still improving at the edge; no val.

## Plan: E57c, confidence ranked within each voxel (issue #57, 18.11.2026)
- **Goal:** the verifier above fails because a global confidence cut removes surfaces that every view sees low; the oracle thins redundant views instead.
  Peter asked for an elegant version that fills the gaps without worsening surfaces the confidence already covers. Rule, no GT, no mask test: in each
  voxel of the 20-view cloud keep the most confident round(q n) of its n pixels (`keep_top_within`; n = 1 is dropped for q <= 0.5). A surface is thinned where several
  views cover it and kept where one does: no hole, no band; the pixels dropped first in a voxel are the least confident ones.
- **Setting:** `e57c_voxel_conf.py --splits dev` (started before this entry was written, nothing read from it yet); voxel side = `pixel_footprint` x {1, 2, 4}, grouped
  by SAM 3 instance or not, q in {0.9, 0.7, 0.5, 0.3}; dev, cm, the evaluation's cuts. Baselines at equal q: density (q 0.9 3.429, 0.7 3.277, 0.5 3.299, 0.3 7.757), global
  confidence (4.289 / 12.753 / 25.376 / 41.512), accuracy oracle (2.724 / 2.479 / 2.726 / 3.453), greedy oracle (2.667 / 2.181 / 2.048 / 2.099).
- **Expected, written before running:** completeness stays near the uncut 2.3-3 cm down to q 0.5 for every side (a voxel keeps pixels); Chamfer below density at
  q 0.9 and 0.7 (it is thinning by confidence, which was the better accuracy ranker), the best side x2 or x4; not below the accuracy oracle; at q 0.3 it
  drops to where voxels hold few pixels and gets worse (singletons dropped).

## E57c, confidence ranked within each voxel: completeness is kept, accuracy is not gained -> Chamfer 3.30-3.70 at every setting, density's level at best (dev, 18.11.2026)
- **Commands:** `CUDA_VISIBLE_DEVICES= uv run python -m experiments.scripts.e57c_voxel_conf --splits dev` -> `results/e57c_voxel_conf_dev.txt` (6 min, CPU). Dev, cm,
  uncut 3.500 (4.675 / 2.324). Tree dirty.
- **Results** (dev mean; kept share, accuracy / completeness / Chamfer; no grouping by instance, the per-instance rows within 0.03 cm of them):

  | voxel side | q 0.9 | q 0.7 | q 0.5 | q 0.3 |
  |---|---|---|---|---|
  | x1 footprint | 0.974, 4.729 / 2.332 / 3.530 | 0.792, 4.979 / 2.381 / 3.680 | 0.298, 3.026 / 7.219 / 5.123 | 0.216, 3.170 / 7.284 / 5.227 |
  | x2 | 0.924, 4.757 / 2.347 / 3.552 | 0.713, 4.877 / 2.447 / 3.662 | 0.465, 3.817 / 2.834 / **3.326** | 0.298, 3.940 / 2.983 / 3.462 |
  | x4 | 0.902, 4.698 / 2.365 / 3.532 | 0.702, 4.676 / 2.485 / 3.580 | 0.496, 4.233 / 2.699 / 3.466 | 0.300, 4.273 / 2.970 / 3.621 |

  Density at equal q: 3.429 / 3.277 / 3.299 / 7.757; global confidence 4.289 / 12.753 / 25.376 / 41.512; accuracy oracle 2.724 / 2.479 / 2.726 / 3.453.
- **Against what was expected (written before running):** completeness near uncut down to q 0.5: yes for x2 and x4 (2.7-2.8), no for x1 (7.2: its share is 0.30, not 0.5, as it drops voxels
  with one pixel; how many pixels those are was not counted); Chamfer below density at q 0.9 and 0.7: no (3.55 / 3.66 against 3.43 / 3.28); not below the accuracy oracle: yes;
  worse at q 0.3: yes for x1, mild for the others.
- **Reading.** The cut no longer removes surfaces, and it no longer gains accuracy: at share ~0.5 the kept accuracy is 3.8-4.2 cm (x2, x4) against 2.22 for the global
  confidence cut and 3.00 for density. Within a voxel the points sit on one patch of surface and their confidences say little about which are the accurate ones; what
  the global confidence knows is which regions are bad, and that is exactly what removes whole surfaces. So no confidence-only rule here (global, per view, per instance,
  membrane-verified, per voxel) gets below the uncut 3.500 by more than 0.2 cm; the oracle's 2.0-2.5 needs a signal that separates accurate from inaccurate points of the
  same patch, which Déjà View's confidence does not provide.
- **Not tried (candidates, not results):** cross-view consensus inside a voxel (each point's depth against the other views' points of its voxel, E45's MVS-consistency
  filter lost to density by 0.13 cm as a global filter); the within-voxel ranking by the true a_i as the ceiling of any per-voxel thinning; the learned head on the greedy label
  with confidence and voxel statistics as inputs (what the volumetric head already sees).
- **Limits:** dev only; x1/x2/x4 and q on four values; one grouping rule; no val.

## E57c, seams: the confidence's drop order and the oracle's side by side with the SAM 3 masks, and the pixels of the 8 cm gate (bedroom, 18.11.2026)
- **What:** `uv run python -m experiments.scripts.e57c_seams > experiments/results/e57c_seams.txt` (1 min) -> `figures/e57c_seams_views.jpg` (views 18, 0, 12, 6)
  and `figures/e57c_seams_mask.jpg` (the same, cropped to each view's largest SAM 3 mask, white outside). It replaces the verifier figure as the picture for deriving a
  combination (Peter's request: both rankings per pixel, the masks, and the combination at 8 cm). Columns: confidence's drop order v; the greedy oracle's drop order v
  (the largest kept fraction at which the pixel is still kept, from the greedy path in 10 % rounds: the oracle's own value); SAM 3 masks on the image; the combination of
  `e57c_hybrid` B (q_c 0.5, q_g 0.5, delta 8 cm): kept by confidence, added from the oracle (> 8 cm from kept), kept by the oracle but within 8 cm (the seam), dropped.
- **Shares of the GT pixels:** dropped 0.305, kept by confidence 0.500, oracle added 0.070, seam 0.125. Of the 0.5 the oracle keeps, 0.30 lie in the confidence's kept set.
- **Looked at** both figures. Inside a large mask (the walls) the confidence is a smooth field, green in the middle of the wall and red toward its borders, and its isoline cuts the
  wall in one curve; the oracle's order on the same wall is speckled at pixel scale with a large-scale trend (view 6's wall almost all red, view 12's left part green); the seam
  (orange) is a band on the dropped side of the isoline, up to the width of the gate, with the blue pixels behind it.
- **Check:** the share of GT pixels lying in a (view, SAM 3 instance) block that a cut keeps < 10 % or > 90 % of (449 blocks): confidence q 0.5 0.379, greedy oracle q 0.5 0.155. The
  oracle is not an all-or-nothing choice per (view, mask).
- **Limits:** one scene; the order v of the oracle is the greedy path's, with 10 % rounds; the purity check is a count, not a test of any method.

## E57c, view 18 in four colors: the gate's seam is the thick blue band, the reintroduced pixels sit only far from the confidence's kept set (bedroom, 18.11.2026)
- **What:** `uv run python -m experiments.scripts.e57c_four_colors > experiments/results/e57c_four_colors.txt` -> `figures/e57c_four_colors_view18.jpg`. Confidence q_c 0.5, greedy
  oracle q_g 0.5, gate 8 cm as `e57c_hybrid` B. White: kept by the confidence cut; red: dropped by confidence and by the oracle (dropped whatever the gate); green: dropped by confidence,
  kept by the oracle, > 8 cm from kept (reintroduced); blue: the same but within 8 cm (the seam). My reading of Peter's four colors: his "white, filtered regardless of threshold"
  cannot be a fourth class beside "red, dropped by confidence" unless white is the kept set, so white is the kept set; to be confirmed.
- **Shares** (of view 18 / of all GT pixels): kept 0.525 / 0.500; red 0.255 / 0.305; green 0.089 / 0.070; blue 0.131 / 0.125.
- **Looked at:** the blue band follows the confidence's kept boundary on the wall, around the dresser and the bed frame, 10-30 pixels wide; green is the left wall away from the kept region;
  red is the right wall and the bed frame, where the oracle also drops.

## Plan and result: E57c, two ways to close the gate's seam, bedroom view 18 (18.11.2026)
- **Rules** (Peter): (1) the oracle decides only in a confidence band, 0.5 < v_conf <= 0.8, no distance gate; beyond the band dropped. (2) the gate's blue pixels come back
  if they lie on a straight run (row, column, both diagonals of the view's image; one pass) of blue pixels of one SAM 3 instance, with a white (confidence-kept) pixel at one end
  and a green (oracle-reintroduced) pixel at the other. `e57c_two_fills.py` -> `figures/e57c_two_fills_view18.jpg`, `results/e57c_two_fills.txt`; bedroom only, nothing tuned (0.8 and
  the four directions picked before running).
- **Expected, written before running:** (1) Chamfer between the union (1.400) and the 8 cm gate (1.646), no seam; (2) few blue pixels come back (the seam is bounded by white on one side and
  by dropped pixels on the other), Chamfer within 0.05 cm of the gate's 1.646.
- **Result** (bedroom, all 20 views; uncut 1.846, confidence alone 6.293, B gate 8 cm 1.646 at kept 0.570, B union 1.400 at 0.695): (1) band: kept 0.626, Chamfer **2.481**;
  (2) lines: kept 0.586, Chamfer **1.608**; on view 18 rule 2 brings back 2.7 % of the pixels (gold) and leaves 10.5 % blue.
- **Against what was expected:** (1) wrong: 2.481 is worse than the gate, not between union and gate; the pixels the oracle keeps beyond the band (v_conf > 0.8, the least confident
  fifth) are lost, and they are coverage nothing else gives (view 18: 5.2 % of its pixels are oracle-kept beyond the band); the seam is closed, the completeness is not. (2) as expected:
  within 0.05 cm of the gate (1.646 -> 1.608), few pixels come back, the strip it fills on the left wall is the one between white and green; the seam stays where red, not green, lies behind it.
- **Looked at** `figures/e57c_two_fills_view18.jpg`. Limits: one scene; the band edge 0.8 and the four directions were not varied; rule 2 is one pass; both rules use the oracle's keep.
- **Band edge, Peter's correction:** "a band of confidence between 0.3 and 0.5", read as the confidence percentile 0.3-0.5, i.e. v in (0.5, 0.7] (his reading of the band is narrower than my 0.8; the band
  on the kept side, 0.3 < v <= 0.5, would only let the oracle drop pixels, no seam to close). Bedroom sweep of rule 1's band edge (kept, Chamfer cm): 0.6 0.541, 4.560; **0.7 0.585, 3.250**;
  0.8 0.626, 2.481; 0.9 0.664, 1.749; 1.0 (= union) 0.695, 1.400; the narrower the band, the worse, monotonically. `figures/e57c_two_fills_view18.jpg` now shows 0.7 (the 0.8 version:
  `..._band08.jpg`, `results/e57c_two_fills_band08.txt`); rule 2 unchanged (1.608). Expected before running: worse than 0.8's 2.481, because every oracle-kept pixel beyond the band is lost; confirmed.

## Plan and result: E57c, the 8 cm gate with a confidence band, view 18 (18.11.2026)
- **Correction of my reading (Peter):** the last plots were not what he meant. He expects the same white (confidence percentile c > 0.5) as in `e57c_four_colors`, and the seam (oracle-kept, within 8 cm of the kept
  set) split by the confidence: blue, c <= 0.3, not reintroduced; yellow, 0.3 < c <= 0.5, reintroduced; green as before. `e57c_band_gate.py` -> `figures/e57c_band_gate_view18.jpg`,
  `results/e57c_band_gate.txt`; bedroom, the band's lower edge also swept (0.4, 0.3, 0.2, 0.1, 0 = every near oracle pixel back).
- **Expected, written before running:** Chamfer between the gate's 1.646 and the union's 1.400, falling as the lower edge goes to 0; yellow a thin strip along the white's border on view 18, blue the thicker, farther part of the old seam.
- **Result** (bedroom, all 20 views; uncut 1.846, 8 cm gate 1.646 at kept 0.570, union 1.400 at 0.695): lower band edge c = 0.4: kept 0.605, Chamfer 1.558; **0.3: 0.637, 1.488**; 0.2: 0.660, 1.449;
  0.1: 0.680, 1.422; 0: 0.695, 1.400. Shares of view 18: white 0.525, red 0.255, green 0.089, yellow 0.080, blue 0.051 (all GT pixels: 0.500, 0.305, 0.070, 0.066, 0.059).
- **Against what was expected:** between the gate and the union, falling towards the union as the edge goes to 0: yes. Yellow a strip along the white's border, blue the part farther from it (bed frame, floor edge): yes
  (`figures/e57c_band_gate_view18.jpg`, looked at). Limits: one scene; the Chamfer falls monotonically with the share kept and no equal-share comparison with the greedy oracle or density was made here.

## E57c, clouds of the seam fills and the union: the bedroom, same views (18.11.2026)
- **What:** `uv run python -m experiments.scripts.e57c_clouds_fills > experiments/results/e57c_clouds_fills.txt` -> `figures/e57c_clouds_fills_{dollhouse_colors,dollhouse_shaded,view18_pulledback_colors,view0_...,view12_...,view6_...}.jpg`.
  Panels: ground truth, Déjà View, confidence q 0.5, confidence + oracle with the 8 cm gate, + straight-run fill, + confidence band 0.3 < c <= 0.5, union (no gate), greedy oracle q 0.5, density q 0.6.
  Chamfer of the bedroom ai_006_007 (cm, all 20 views, one scene, not the dev mean): 1.846, 6.293, 1.646, 1.608, 1.488, 1.400, 1.135, 1.725.
- **Looked at** the dollhouse in colors and view 18 (views 0, 12, 6 and the shaded dollhouse saved, not looked at): the gate leaves the white ring on the wall in view 18, the straight-run fill leaves part of it
  (right side), the band version and the union show a continuous wall; the remaining holes in the band version are the wardrobe window and bits of the bed cover.

## Plan: E57c, debugging the white squiggles of "gate + straight-run fill" in the pulled-back view 18 (18.11.2026)
- **Goal (Peter):** the straight-run fill is the most complete-looking cloud, but thin white lines remain on the wall (right of the door). Find where they come from: inside SAM 3 masks or at their borders,
  at the border of a source view, salt-and-pepper confidence, a lone non-blue pixel stopping a run; then fixes (dilation / closing before the fill?).
- **Method** (`e57c_squiggles.py`, bedroom, no new scores): render the cloud from the same camera twice with an id buffer (`render.splat` with the flat pixel index as color): the kept pixels
  and all pixels. A *gap* pixel of the render: no kept splat, but a dropped pixel is visible there; it names the dropped source pixel (view, y, x) and its category (blue: the oracle keeps it,
  the fill did not bring it back; red: the oracle drops it too). Figure 1: the render with the gaps colored by category, by source view, and a zoom on the right wall. Figure 2: the source
  views of the zone in image space (categories, SAM 3 borders, confidence percentile). Counts for the zone: category, source view, distance to the image border and to a SAM 3 border, isolated
  or one-pixel-wide dropped pixels, confidence percentile, and for the blue ones why no run closed (the run's two ends: white, red, green, gold, image border, other instance).
- **Expected, written before running:** most gap pixels are blue (the fill's rule fails, the oracle wants them); the dominant failure is a run closed by white at both ends (a thin seam between two
  confidence-kept regions, no green seed), then runs ending at red; few at image borders or SAM 3 borders.

## E57c, squiggles [WRONG, corrected in the entry below]: the white lines of the straight-run fill are the oracle's own drops, bad points that no other view's patch closes (bedroom, pulled-back view 18, 18.11.2026)
- **Commands:** `uv run python -m experiments.scripts.e57c_squiggles > experiments/results/e57c_squiggles.txt` (1.5 min) -> `figures/e57c_squiggles_render.jpg` (the render, gaps by category and by source
  view, zoom), `figures/e57c_squiggles_sources.jpg` (the zone's three main source views: categories, SAM 3 borders, confidence percentile). Looked at both. The zone is a box on the right wall
  (render rows 30-260, columns 400-560): it holds the long squiggle and the wardrobe window, so its counts mix both.
- **Results** (gap pixel = no kept splat, a dropped pixel visible behind it): of 13 925 gap pixels of the render 11 898 are **red** (dropped by the confidence cut and by the oracle) and 2 027 blue
  (the oracle keeps them, the fill did not); in the zone 3 525 red, 57 blue. The zone's 1 732 source pixels come from 14 views (view 7: 750, 6: 294, 12: 236). Of them: blue 36; within 2 px of a
  SAM 3 border 727 (42 %; all pixels of those views: 16.8 %); instance -1 234; within 8 px of the image border 21; all four neighbors kept 0; one pixel wide between kept pixels 10; confidence
  percentile c > 0.3 365 (21 %). True accuracy a_i of the source pixels: median 5.25 cm, 90th percentile 19.7 (all pixels: median 1.78); 3D distance to the nearest kept pixel: median 5.87 cm,
  90th 18.6. Across a gap pixel in the render: different source views on the two sides 1 512, same view 477, no kept pixel within 4 px on both sides 1 593. The few blue ones: the shortest run ends in
  (dropped, green) 12, (green, other instance) 9, (dropped, dropped) 8, (dropped, other instance) 7.
- **Against what was expected:** wrong. Expected mostly blue gap pixels and runs closed by white at both ends: the squiggles are red, and the blue ones fail on runs that end in dropped pixels, not (white, white).
  Few image-border or SAM 3 border cases was right for the image border (21 of 1 732), not for SAM 3 borders (2.5 x their base rate).
- **Reading.** The squiggle is a strip of bad points (median 5 cm off) that the oracle removes for accuracy and that costs it no completeness, because the kept patches from different source views
  on both sides lie within a few mm of the GT points there; in the splat render the patches do not touch and a hairline of background (or of the bad points) shows. The Chamfer is blind to it (a GT point
  in the sliver has a kept point at most a splat radius away). It is not salt-and-pepper (no isolated pixel), not the image border, and a lone non-blue pixel stopping a run only matters for the 15 % blue gaps,
  so dilation or closing before the fill would not touch it.
- **Candidate fixes** (not run): (a) a coverage-aware completeness in the oracle: m_i counts the GT points whose nearest kept pixel would be farther than the splat radius (a threshold, F-score like), so
  a sliver wider than a splat costs; (b) a closing in 3D of the kept set at the pixel footprint (re-add dropped pixels that bridge two kept patches, within a tolerance of their surfaces; the source pixels are
  6 cm off, so a pure distance test would bring bad points back, and the test must use the kept patches' extrapolated surface); (c) cosmetic: a larger splat radius for kept pixels (closes hairlines in the
  render, changes no score). Limits: one scene, one camera; zone chosen by eye; a_i uses the GT.

## E57c, squiggles, correction: they are the blue pixels the fill did not bring back, the oracle-alone cloud covers them (bedroom, 18.11.2026)
- **What was wrong (Peter's objection: the greedy oracle alone shows no such lines, so "dropped by both" is odd):** I identified each gap pixel with the *nearest* dropped pixel of the all-pixels render. That is an
  occluder in front of the gap (a bad point, hence "red", a_i median 5 cm), not what is missing behind it. The right question is which oracle-kept pixel would cover the gap. The kept set differs from the oracle's
  set only by the blue pixels (oracle-kept, not brought back) and by pixels the confidence keeps and the oracle drops, so the oracle-alone cloud covers a gap pixel only through a blue one.
- **Check** (`e57c_squiggles.py` now renders the blue pixels and the oracle-alone cloud as id buffers; `results/e57c_squiggles_blue.txt`; the first run's output stays in `results/e57c_squiggles.txt`): of the render's 13 925 gap
  pixels 8 438 (61 %) are covered by the oracle-alone cloud, all through blue pixels, 5 487 by neither (holes of the oracle alone too: the wardrobe window, the bed cover); in the right-wall zone 1 150 of 3 582 (32 %)
  and 2 432 by neither. The long squiggle is of the first kind (blue in `figures/e57c_squiggles_render.jpg`).
- **The squiggle's source pixels** (586 blue pixels from 8 views: 12: 213, 14: 203, 6: 62, 8: 58, 7: 43): true accuracy median 1.42 cm (90th percentile 1.72; all pixels 1.78), so good points; 3D distance to the nearest kept
  pixel median 3.8 cm, max 6.3 (inside the 8 cm gate); within 2 px of a SAM 3 border 61 (10 %, base rate 16.9 %), image border 11, instance -1 16, isolated 0; confidence percentile c > 0.3: 258 (44 %).
- **Why the straight-run rule missed them** (shortest run of blue pixels of one instance through each pixel; its two ends): (dropped, green) 247, (dropped, dropped) 214, (dropped, other instance) 28, (dropped, kept) 19,
  (gold, kept) 15, (dropped, gold) 14, others 49. 523 of 586 (89 %) have a dropped (red) end; only 19 (3 %) have white at one end. In the source view the squiggle is a thin blue strip between a red region (the oracle
  drops it) and a green one: the 8 cm gate measures proximity in 3D to the kept pixels of *other* views, the rule needs white and green neighbours in the *same* view's image. The two do not coincide.
- **Not salt-and-pepper, not the image border, not mainly SAM 3 borders** (10 % against 16.9 % base rate): erosion or dilation before the fill would not close it either; what is missing is a rule attaching these pixels
  to what is kept, e.g. a blue pixel on a run with a kept (green or white) end, or a smaller gate than 8 cm (the squiggle's pixels lie at 2-6 cm; at delta 2 cm they are back, as the earlier delta sweep shows).
- **Limits:** one scene and camera; the zone chosen by eye; a_i uses the GT. Candidate rules not run.

## Plan: E57c, the product of the remaining confidence and the oracle's keep order (issue #57, 18.11.2026)
- **Idea (Peter):** no distance gate, only the two rankings. (1) Cut by confidence: pixels with percentile c > 0.5 stay. (2) For the others, u = c / 0.5 in [0, 1] (the remaining confidence) times
  k in [0, 1], the greedy oracle's keep order (the position in its removal sequence / N; 1 = removed last; ties of a round broken by that round's score); pixel kept iff u k >= tau. Plot u k on the
  bedroom's view 18 (confidence-kept pixels white) and try thresholds.
- **Setting:** `e57c_product.py`, bedroom, all 20 views, greedy in rounds of 5 % of the pixels down to 5 %. Thresholds given as the kept share (0.52 ... union 0.695), Chamfer (cm) of the product cut, of the
  oracle alone at the same share (the cost of using confidence at all), and the earlier variants as references (8 cm gate 1.646 at 0.570, band 0.3 < c <= 0.5 1.488 at 0.637, union 1.400 at 0.695, oracle alone q 0.5 1.135).
- **Expected, written before running:** the product is no fix by itself: Chamfer at equal share within 0.1 cm of the band/gate variants at the same share, above the oracle alone; it multiplies away the pixels the oracle ranks
  high and the confidence low, which are the seam's coverage. No render of the squiggles in this step.

## E57c, product of the remaining confidence and the oracle order: it does not fix the performance, at equal share it is 0.6-1.3 cm worse than the gate and band variants (bedroom, 18.11.2026)
- **Commands:** `uv run python -m experiments.scripts.e57c_product > experiments/results/e57c_product.txt` (1.5 min) -> `figures/e57c_product_view18.jpg` (looked at). Thresholds as quantiles (Peter's request): the top
  fraction of the confidence-dropped pixels by u k, equal to the kept share; u and k are ranks already (u = c / 0.5, k the position in the greedy removal sequence / N, rounds of 5 %).
- **Results** (bedroom, all 20 views; Chamfer cm; the oracle alone at the same share is the cost of the confidence at all; references at near shares: 8 cm gate 1.646 at 0.570, band 0.3 < c <= 0.5 1.488 at 0.637,
  union 1.400 at 0.695; confidence cut alone 6.293 at 0.500):

  | top fraction of the dropped | kept share | tau | product cut | oracle alone |
  |---|---|---|---|---|
  | 0.04 | 0.520 | 0.705 | 4.210 | 1.130 |
  | 0.10 | 0.550 | 0.551 | 3.289 | 1.152 |
  | 0.14 | 0.570 | 0.479 | 2.912 | 1.167 |
  | 0.20 | 0.600 | 0.396 | 2.537 | 1.192 |
  | 0.28 | 0.640 | 0.312 | 2.252 | 1.225 |
  | 0.36 | 0.680 | 0.247 | 1.993 | 1.261 |
  | 0.39 | 0.695 | 0.227 | 1.911 | 1.275 |

- **Against what was expected (written before running):** within 0.1 cm of the band/gate variants at equal share: wrong, it is 0.6-1.3 cm worse (0.570: 2.912 against the gate's 1.646; 0.637 to 0.640: about 2.25 against the band's 1.488; 0.695: 1.911 against the union's 1.400); above
  the oracle alone: yes; it multiplies away the pixels the oracle ranks high and the confidence low: consistent with the maps.
- **Reading** (view 18): u k is a ramp that decays away from the confidence border, so the cut is seam-free (green touches white) but regional: it reintroduces the left wall next to the white and drops the pixels
  further out and in the bed frame, as confidence does. Among the confidence-dropped pixels u and k are close to independent (the joint histogram is nearly flat, apart from a row at k = 0 where the oracle drops pixels
  of every confidence): the confidence says little about which dropped pixels the oracle wants, so a product lets it outvote the oracle. No threshold of this product recovers the union or the gate at equal share.
- **Not run:** an exponent on u (u^a k, a = 0 is the oracle alone), the oracle's order with the confidence as tie-breaker, min / rank-sum combinations. Limits: one scene, no render of the clouds, no val.

## Plan: E57c, two gradients and a quantile cut of the undecided pixels (issue #57, 18.11.2026)
- **Idea (Peter):** confidence gradient g_c = 1 for the percentile c >= 0.5, linear to 0 at c <= 0.25; oracle gradient g_o = 1 for an oracle-kept pixel more than 10 cm from the confidence-kept set, linear
  to 0 at 0 cm (an oracle-dropped pixel: 0). Pixels with c >= 0.5 and oracle-kept pixels farther than 10 cm are decided (kept) and do not count to the quantile. The undecided ones (c < 0.5, not far-oracle)
  are ranked by p = g_c g_o and the top fraction q of them is kept, q = 0, 0.1, ..., 1.0 (q = 0: confidence plus the far oracle pixels, the 8 cm gate with 10 cm; q = 1: everything). Ties (p = 0: c <= 0.25
  or oracle-dropped) broken by the seeded random order of `filters.ranks`.
- **Setting:** `e57c_gradients.py`, bedroom, all 20 views, greedy oracle q_g 0.5 in 10 % rounds as in the gate variants; per q the kept share and Chamfer (cm), the oracle alone at the same share (5 % rounds) as the
  floor, and the references: gate 8 cm 1.646 at 0.570, band 0.3 < c <= 0.5 1.488 at 0.637, union 1.400 at 0.695, uncut 1.846, product u k at 0.640 2.252.
- **Expected, written before running:** q = 0 near the gate (1.6-1.7, share about 0.56); the Chamfer falls as q rises, not below the band variant at equal share (the p = 0 ties, the low-confidence quarter and the
  oracle-dropped pixels, enter late and by chance), and the curve returns to the uncut 1.846 at q = 1. A minimum near 1.4-1.6 at q 0.3-0.6. No render of the clouds in this step.

## E57c, two gradients and a quantile cut: best 1.462 cm at q 0.2 (kept 0.652), where the pixels with p > 0 run out; beyond it random ties raise it to the uncut 1.846 (bedroom, 18.11.2026)
- **Commands:** `uv run python -m experiments.scripts.e57c_gradients > experiments/results/e57c_gradients.txt` (1.5 min) -> `figures/e57c_gradients_view18.jpg` (looked at). Bedroom, all 20 views, cm.
- **Setup facts:** decided and kept 0.565 (confidence 0.500, oracle-kept farther than 10 cm 0.065); undecided 0.435, of which 81.2 % have p = 0 (c <= 0.25: 47.9 %, oracle-dropped: 70.0 %), so only 18.8 % of the pool,
  8.2 % of all pixels, have p > 0, and the ties are broken by the seeded random order.
- **Results** (kept share, accuracy / completeness / Chamfer; the oracle alone at the same share in brackets):

  | q | kept | accuracy / completeness / Chamfer | oracle alone |
  |---|---|---|---|
  | 0.0 | 0.565 | 1.668 / 1.692 / 1.680 | 1.163 |
  | 0.1 | 0.608 | 1.622 / 1.426 / 1.524 | 1.198 |
  | **0.2** | 0.652 | 1.604 / 1.320 / **1.462** | 1.235 |
  | 0.3 | 0.695 | 1.767 / 1.253 / 1.510 | 1.276 |
  | 0.5 | 0.782 | 2.040 / 1.212 / 1.626 | 1.364 |
  | 0.7 | 0.869 | 2.262 / 1.191 / 1.726 | 1.470 |
  | 1.0 | 1.000 | 2.520 / 1.172 / 1.846 | 1.846 |

  References: 8 cm gate 1.646 at 0.570 (q 0 here, 10 cm: 1.680 at 0.565), band 0.3 < c <= 0.5 1.488 at 0.637, union 1.400 at 0.695 (here q 0.3: 1.510), uncut 1.846, product u k 1.911 at 0.695.
- **Against what was expected (written before running):** q = 0 near the gate: yes (1.680); a minimum near 1.4-1.6 at q 0.3-0.6: it is at q 0.2 (1.462), earlier than expected because the pool has few p > 0 pixels; not below the band variant at
  equal share: 1.462 at 0.652 against 1.488 at 0.637, no clear difference; returns to 1.846 at q = 1: yes.
- **Reading:** the useful part of the curve ends at q about 0.19, where the p > 0 pixels (oracle-kept, c > 0.25, d < 10 cm; a ribbon along the confidence border in view 18) run out; the rest of the sweep keeps tie pixels at random
  (the speckled maps at q 0.4 and 0.6), so accuracy worsens steadily and Chamfer drifts to the uncut value. At equal share the cut is 0.1 cm above the union (1.510 against 1.400 at 0.695) and 0.2-0.3 cm above the oracle alone.
  The gradients are the band rule with a soft edge; they add no information beyond c > 0.25 and the oracle's keep.
- **Limits:** one scene; ties by random order (a deterministic tie-break, for example the oracle's keep order k, was not tried); no render of the clouds.

## Plan: E57c, the two-gradient cut rerun lean, with g_c from 0 to 0.5, and its quantile isolines (issue #57, 18.11.2026)
- **Change (Peter):** the confidence gradient goes from 0 at c = 0 to 1 at c >= 0.5 (g_c = min(c / 0.5, 1)); the oracle gradient and the rest as in the entry above. Rerun lean: no SAM 3 linking (`load_inputs` does it), no density,
  nothing but Chamfer; the per-scene arrays (points, a_i, the greedy order, c, d) cached in `runs/e57c/` so a rerun only evaluates the cuts. Stage times are printed. Plot: the product p with the thresholds of the quantiles q as
  isolines, next to the heatmap of the quantile (rank of p among the undecided pixels, ties at their mean rank), isolines at q = 0.1 ... 0.9, and the Chamfer against q.
- **Expected, written before running:** the same shape as with 0.25 (best below the gate and near the band variant, back to 1.846 at q = 1), the minimum at q 0.3 or earlier, where the pixels with p > 0 (oracle-kept near pixels with c > 0, about 30 % of the
  undecided ones) run out, 1.45-1.5; first run well under the 1.5 min of the last one, later runs only the evaluations.

## E57c, two gradients with g_c from 0 to 0.5, lean: the best cut is the union itself, 1.400 cm at q 0.3; the lean run takes 30 s (bedroom, 18.11.2026)
- **Commands:** `uv run python -m experiments.scripts.e57c_gradients --floor > experiments/results/e57c_gradients_v2.txt` -> `figures/e57c_gradients_isolines_view18.jpg` (looked at). Stage times of the first run: load 1.3 s, arrays 24.6 s
  (computed once, cached in `runs/e57c/`; 0.5 s after), gradients 0.5 s, sweep 22 s for the 11 Chamfers (44 s with the oracle-alone column `--floor`): 70 s first, 30 s later, against about 90 s for the version that
  linked SAM 3 and computed the density. The Chamfers dominate: two KD-trees of up to 3.8 M and 2.5 M points per cut, realigned as `evaluate` does.
- **Results** (bedroom, all 20 views, cm; decided and kept 0.565, undecided 0.435 of which p = 0: 70.0 %, all oracle-dropped):

  | q | kept | accuracy / completeness / Chamfer | oracle alone |
  |---|---|---|---|
  | 0.0 | 0.565 | 1.668 / 1.692 / 1.680 | 1.163 |
  | 0.1 | 0.608 | 1.624 / 1.382 / 1.503 | 1.198 |
  | 0.2 | 0.652 | 1.584 / 1.285 / 1.435 | 1.235 |
  | **0.3** | 0.695 | 1.547 / 1.253 / **1.400** | 1.276 |
  | 0.4 | 0.739 | 1.736 / 1.201 / 1.469 | 1.318 |
  | 0.5 | 0.782 | 1.903 / 1.191 / 1.547 | 1.364 |
  | 0.7 | 0.869 | 2.189 / 1.181 / 1.685 | 1.470 |
  | 1.0 | 1.000 | 2.520 / 1.172 / 1.846 | 1.846 |

- **Against what was expected (written before running):** same shape: yes; minimum at q 0.3 or earlier: at 0.3; Chamfer 1.45-1.5 there: it is 1.400, below the range, because with c > 0 every oracle-kept undecided pixel
  has p > 0, so q = 0.3 (30 % of the undecided = exactly those pixels) is the union of the confidence cut and the oracle: the same 0.695 and 1.400 as before. Beyond it the cut adds pixels the oracle drops, by the random tie-break.
- **Reading:** the gradient orders the pixels of the union; its prefixes (q 0.1, 0.2: 1.503, 1.435) are worse than the whole, so ordering by p adds nothing in Chamfer over the union and over the oracle alone at the same share
  (1.276 at 0.695). In view 18 the gradient is a ribbon next to the confidence border (white to green through yellow-orange); 70 % of the undecided pixels are red (p = 0).
- **Figure** (Peter's coloring): white = kept by confidence, green = oracle-kept > 10 cm, the undecided pixels red (p = 0) through yellow to green (p = 1), black isolines at the p thresholds of the quantiles q (none beyond q 0.3: ties); the
  quantile heatmap (ties at their mean rank) and Chamfer against q next to it. Limits: one scene; no render of the clouds; ties by seeded random order.

## Plan: E57c, the mean of the two gradients instead of their product (issue #57, 18.11.2026)
- **Idea (Peter):** p = (g_c + g_o) / 2, because the product penalizes values near zero too much. Everything else as in the two-gradient entry (g_c = min(c / 0.5, 1), g_o = clip(d / 10 cm, 0, 1) for oracle-kept pixels, 0 for the others;
  decided: c >= 0.5 and oracle-kept farther than 10 cm; q of the undecided, 0 ... 1). `e57c_gradients.py --combine mean`.
- **Expected, written before running:** no ties any more (oracle-dropped pixels are ranked by g_c / 2 < 0.5); they enter among the oracle-kept ones, so the high-confidence oracle-dropped pixels just below c = 0.5 come in before the
  low-confidence oracle-kept ones; Chamfer no better than the union (1.400 at 0.695), the minimum at a q of 0.2-0.4 at 1.40-1.55, and above the product's curve for q beyond the union's 0.3, below it before.

## E57c, mean of the two gradients: a flat curve at 1.54 cm for q 0.1-0.6, no ties, never the union's 1.400 (bedroom, 18.11.2026)
- **Commands:** `uv run python -m experiments.scripts.e57c_gradients --combine mean --floor > experiments/results/e57c_gradients_mean.txt` (30 s + 22 s for the floor, cached arrays) ->
  `figures/e57c_gradients_mean_isolines_view18.jpg` (looked at). Bedroom, all 20 views, cm.
- **Results** (kept, Chamfer; the product's value at the nearest kept share in brackets):

  | q | kept | accuracy / completeness / Chamfer | product (same q) |
  |---|---|---|---|
  | 0.0 | 0.565 | 1.668 / 1.692 / 1.680 | 1.680 |
  | 0.1 | 0.608 | 1.646 / 1.446 / 1.546 | 1.503 |
  | 0.2 | 0.652 | 1.706 / 1.384 / 1.545 | 1.435 |
  | 0.3 | 0.695 | 1.755 / 1.326 / 1.541 | 1.400 |
  | 0.5 | 0.782 | 1.843 / 1.239 / 1.541 | 1.547 |
  | 0.7 | 0.869 | 1.963 / 1.164 / 1.563 | 1.685 |
  | 0.9 | 0.956 | 2.146 / 1.129 / 1.638 | 1.797 |
  | 1.0 | 1.000 | 2.520 / 1.172 / 1.846 | 1.846 |

  Minimum 1.541 at q 0.3-0.5 (kept 0.70-0.78); oracle alone at 0.695: 1.276; union 1.400; band variant 1.488 at 0.637.
- **Against what was expected (written before running):** no ties: yes (p = 0 for none); no better than the union: yes (1.541 against 1.400); minimum 1.40-1.55: 1.541; above the product's curve for q beyond 0.3 and below it before: wrong, the product is
  better up to q 0.4 and the mean from q 0.5 or 0.6 on (equal at 0.5).
- **Reading:** the mean is flat (1.541-1.546 from q 0.1 to 0.6) and degrades slowly towards the uncut value (1.638 at 0.956 kept, 0.2 cm below uncut), where the product reaches its 1.400 only at its q 0.3 and climbs
  linearly afterwards (1.797 at 0.956 kept): the mean is the more stable curve and the product the better minimum. The oracle-dropped pixels are now ranked by confidence (g_c / 2 < 0.5), the high-confidence ones just below the cut come in
  among the oracle-kept ones, which costs the 0.14 cm against the union. The p map is a smooth gradient over the whole undecided area (figure), as asked for earlier. Limits: one scene; no render of the clouds.

## Plan: E57c, clouds of the two-gradient cuts at their best q, next to the references (issue #57, 18.11.2026)
- **What (Peter):** RGB clouds of the bedroom in the usual views: ground truth, Déjà View, confidence only (q 0.5), greedy oracle only (q 0.5), the straight-run fusion (gate 8 cm + straight runs within a SAM 3 instance),
  the product and the mean of the two gradients at the q of their lowest bedroom Chamfer (picked among q = 0, 0.1, ..., 1.0 in the script, so on the bedroom itself: a presentation scene, not a tuned value). `e57c_clouds_gradients.py`;
  `e57c_clouds.render` now takes any number of panels.
- **Expected, written before running:** the product's best cut is the union (q 0.3, kept 0.695, 1.400) and looks like the straight-run fusion with the seam closed; the mean's best (q about 0.3-0.5, 1.541) keeps more low-confidence debris
  in the flat floor regions (it ranks the oracle-dropped pixels by confidence) and looks the noisiest of the three fusions; the hairlines of the straight-run fusion stay.
- **Result** (`results/e57c_clouds_gradients.txt`; figures `e57c_clouds_gradients_{dollhouse_colors,dollhouse_shaded,view18_pulledback_colors,view0_...,view12_...,view6_...}.jpg`; looked at the dollhouse in colors and view 18): best q on the bedroom: product
  0.3 (1.400, kept 0.695, the union), sum 0.4 (1.541, kept 0.739; q 0.3 and 0.5 give 1.541 too). Chamfer / kept: Déjà View 1.846 / 1.00, confidence only 6.293 / 0.50, oracle only 1.135 / 0.50, straight-run fusion 1.608 / 0.586.
- **Against what was expected:** the product's best looks like the fusion with the seam closed: yes, the wall in view 18 is continuous; the straight-run fusion keeps its white hairline on the wall: yes; the sum's best the noisiest of the three: not visible, it looks
  like the product's (smooth wall and floor, the bed frame debris as in the oracle-only panel); the differences between the two gradient cuts are small at this size.
- **Limits:** one scene; q picked on the bedroom's own Chamfer (a presentation scene, not a tuned value); views 0, 12, 6 and the shaded dollhouse saved, not looked at.

## Plan: E57c, all the methods of the last clouds on dev (issue #57, 18.11.2026)
- **Goal (Peter):** score on dev (3 scenes, 20 views at 504 px, Déjà View K = 16, SAM 3 image mode linked with the working association, cm, the evaluation's cuts, one Umeyama per cut) the panels of `e57c_clouds_gradients`: Déjà View, confidence only (q 0.5),
  greedy oracle only (q 0.5), the straight-run fusion (gate 8 cm + straight runs in a SAM 3 instance), the product and the mean of the two gradients; plus instance voxel density (q 0.5 and 0.6) as the one real filter. `e57c_dev_methods.py --splits dev`.
- **The q of the gradient cuts:** the full sweep q = 0 ... 1 is reported; "best" is the q with the lowest dev mean (chosen on dev, three scenes, so optimistic: it is not a held-out number); the bedroom's q (product 0.3, mean 0.4) as the second row. No val.
- **Expected, written before running (Peter expects the oracle first):** greedy oracle best (about 2.05 as in the zoo run); product at its best q second at about 2.5-3.0 (on the bedroom it recovers 63 % of the oracle's gain over Déjà View); the mean about 2.7-3.2; straight-run
  fusion about 2.8-3.4; density 3.2-3.3; Déjà View 3.500; confidence only far the worst (25.4). The ai_044_001 scene (7.05 uncut) may order the middle methods differently.

## E57c, all methods on dev: the greedy oracle is first by 0.45 cm, the product of the gradients second (2.498), then the mean (2.626) and the straight-run fusion (2.680), density 3.188, Déjà View 3.500, confidence only 25.376 (dev, 18.11.2026)
- **Commands:** `CUDA_VISIBLE_DEVICES= uv run python -m experiments.scripts.e57c_dev_methods --splits dev > experiments/results/e57c_dev_methods.txt` (4 min, CPU, 80 s per scene). Dev, 3 scenes, cm, the evaluation's cuts. Tree dirty.
- **Results** (dev mean; kept share, accuracy / completeness / Chamfer; per scene Chamfer ai_010_007 / ai_006_007 / ai_044_001):

  | method | kept | accuracy / completeness / Chamfer | per scene |
  |---|---|---|---|
  | **greedy oracle only q 0.5** | 0.500 | 1.533 / 2.562 / **2.048** | 0.88 / 1.14 / 4.13 |
  | product of the gradients, q 0.3 (best on dev = the bedroom's) | 0.708 | 2.632 / 2.364 / 2.498 | 1.09 / 1.40 / 5.00 |
  | mean of the gradients, q 0.2 (best on dev) | 0.667 | 2.596 / 2.655 / 2.626 | 1.19 / 1.54 / 5.15 |
  | mean, q 0.4 (the bedroom's) | 0.750 | 2.818 / 2.514 / 2.666 | 1.23 / 1.54 / 5.23 |
  | straight-run fusion (gate 8 cm) | 0.603 | 2.513 / 2.848 / 2.680 | 1.21 / 1.61 / 5.22 |
  | density q 0.6 | 0.600 | 3.550 / 2.827 / 3.188 | 1.48 / 1.73 / 6.36 |
  | density q 0.5 | 0.500 | 3.001 / 3.596 / 3.299 | 1.73 / 1.69 / 6.47 |
  | Déjà View | 1.000 | 4.675 / 2.324 / 3.500 | 1.60 / 1.85 / 7.05 |
  | confidence only q 0.5 | 0.500 | 2.220 / 48.532 / 25.376 | 8.20 / 6.29 / 61.63 |

  The product's sweep (Chamfer by q): 0.0 2.747, 0.1 2.600, 0.2 2.545, **0.3 2.498**, 0.4 2.640, 0.5 2.797, 0.7 3.103, 0.9 3.377, 1.0 3.500; the mean's: 0.0 2.747, 0.1 2.627, 0.2 2.626, 0.3 2.638, 0.4 2.666, 0.5 2.704, 0.7 2.829, 0.9 3.062, 1.0 3.500.
- **Against what was expected (written before running):** the oracle first: yes, and the order oracle < product < mean < straight-run fusion < density < Déjà View < confidence holds in each of the three scenes; oracle 2.05: yes (2.048); product 2.5-3.0: 2.498, at the edge; mean
  2.7-3.2: 2.626, below the range; fusion 2.8-3.4: 2.680, below; density 3.2-3.3: 3.188 and 3.299; Déjà View 3.500; confidence 25.4: 25.376.
- **Reading:** against Déjà View the oracle is -41 %, the product -29 %, the mean -25 %, the fusion -23 %, density -9 %. The product and the fusion beat density by 0.5-0.7 cm in every scene. All of them except Déjà View, density and confidence use the greedy oracle, so
  they are not methods: the product, the mean and the fusion are ceilings of "confidence cut first, oracle's pixels back". The oracle alone beats every one of them, so adding the confidence cut costs 0.45-0.6 cm; the same gap as on the bedroom (1.135 against 1.400).
  The mean is the flatter curve (2.63-2.70 for q 0.1-0.5, the product 2.50-2.80) and the product the lower minimum, as on the bedroom.
- **Limits:** dev only, 3 scenes, no val, no seeds; the q of the product and the mean chosen on dev (the product's 0.3 also the bedroom's; the mean's best 0.2 differs from the bedroom's 0.4) so their best rows are optimistic; oracles use the GT.

## Plan: E57c, the squiggle in category colors, from the pulled-back camera and in its source views (issue #57, 18.11.2026)
- **Question (Peter):** how can the top right of the pulled-back view 18 be filled while a squiggle of uniform width persists, if a blue pixel (the oracle keeps it, the straight-run fill did not bring it back) lies there? Render it like the four-color view 18.
- **Hypothesis, written before running:** the strip is the gate's own width. Green pixels are by definition more than 8 cm from every confidence-kept (white) pixel, blue ones less than 8 cm: a blue strip between a white patch and a green patch is about 8 cm wide in 3D whatever the views,
  hence the uniform width; and the straight-run rule looks for white and green neighbours in the same source view's image, while the white and the green flank the strip in other views. Check: the 3D distance of the strip's blue source pixels to the nearest white and to the nearest green
  pixel should be about 4 cm each (their sum near 8 cm; the earlier median to the nearest kept was 3.8 cm).
- **Method** (`e57c_squiggle_categories.py`, bedroom): (1) the pulled-back view-18 camera, every pixel colored by its category (white kept by confidence, green oracle beyond 8 cm, gold blue-brought-back, blue oracle-kept not brought back, red dropped), nearest point wins,
  with and without the red ones; zoom on the zone; (2) the zone's main source views in image space in the same colors, the strip's source pixels marked.
- **Result** (`results/e57c_squiggle_categories.txt`; `figures/e57c_squiggle_categories_render.jpg`, `..._sources.jpg`, looked at both): in the pulled-back camera, with the red points left out, the zone shows a blue band (the oracle keeps it, not brought back) between a white patch
  (confidence-kept) on the left and a green patch (oracle-kept, more than 8 cm from white) on the right; with the red points included the nearest points are red almost everywhere, in front of them. The strip's 586 blue source pixels: 3D distance to the nearest white pixel median 4.54 cm
  (10th-90th percentile 2.96-6.27), to the nearest green 6.61 (2.93-15.88), their sum median 10.81 (8.53-19.71); the gate is 8 cm. The pixels sit in views 12, 14, 6 as thin lines along the border of a red and a green region, not next to white (`..._sources.jpg`).
- **Against the hypothesis (written before running):** a strip of the gate's width between white and green: yes, in the render. The sum of the distances is at least 8 cm by construction (green is more than 8 cm from white); its median is 10.8 and its 10th percentile 8.5, so the blue pixels
  lie on short bridges between a white and a green pixel, as expected; the individual distances (4.5 and 6.6 cm) are not both about 4, the strip being nearer to white. The uniform width is therefore not another view's coverage: the strip is the set of oracle-kept pixels closer than the gate to the white patch and not green, and it is
  missing from the fusion because the straight-run rule needs white and green neighbors in the source view's image, which the strip's source pixels do not have.
- **Implication:** the gate creates a band of its own width wherever a white and a green patch meet in 3D; the union (no gate) has none, which is also what the product's best q, 0.3, reproduces. A 3D bridge rule (a blue pixel with a white and a green pixel each within the gate) is a candidate, not run.

## Plan: E57c, the squiggle colored in the views, and the SAM 3 instance-id maps (issue #57, 18.11.2026)
- **Peter asked which I had colored: the 3D render or the view.** The first figure of the previous entry (`e57c_squiggle_categories_render.jpg`) colors the 3D render from the pulled-back camera (every pixel of all 20 views splatted); its second figure (`..._sources.jpg`)
  colors views 12, 14, 6 in image space. He wants the views: `e57c_view_maps.py` draws view 18 and the strip's three main source views in image space: the categories of the straight-run fusion, the SAM 3 instance-id maps colored by id (ids are linked across views, the largest labeled with
  their number) and the image, with the strip marked: in the source views its own source pixels, in view 18 the strip's 3D points projected into the view.
- **Checks printed:** the categories of view 18's own pixels where the strip projects, and whether view 18's instance id there equals the id of the strip's source pixel (cross-view identity).
- **Expected, written before running:** in view 18 the strip projects onto pixels that view 18 itself keeps (white) or drops (red), mostly of the same instance id as the strip's source pixels (the wall).
- **Result** (`results/e57c_view_maps.txt`, `figures/e57c_view_maps.jpg`, looked at): the strip (586 source pixels: views 12: 213, 14: 203, 6: 62, 8: 58, 7: 43, 18: 3) projected into view 18 lands on 583 pixels inside its image, on the right part of the wall (instance 170, the largest). View 18's own category at those pixels: red 425,
  white 90, blue 58, green 8, gold 2: where the squiggle appears in the pulled-back render, view 18's own pixels are mostly dropped by confidence and oracle; the coverage the gap lacks is the other views' blue pixels. The strip pixels' own instance ids: 170 for 505 of 586 (the wall), 100 for 58, -1 for 16;
  view 18's instance id at the projected pixels equals the strip pixel's id for 72.6 % (the ids are linked across views: the wall is 170 in views 18, 12, 14 and 6). In the source views the strip is a thin line inside the wall mask at the border between the red and the green or white regions of the category map.
- **Against what was expected (written before running):** the strip projects onto view 18 pixels that it keeps (white) or drops (red): mostly red (73 %), white 15 %; mostly the same instance id: 73 %, the wall. Limits: one scene; the projection uses the GT camera pose and the pixel-corner convention is ignored (a pixel of error at most).
- **Correction after Peter's question (is the squiggle another id?):** the magenta squiggle in the instance-id column was my overlay of the strip's pixels, not a segment. `e57c_view_maps.py` now draws no overlay on the maps; the strip is a fourth column of its own. Under it: instance 170 (the wall) for 505 of the 586 strip
  pixels (views 12: 213, 14: 195, 6: 59, 7: 33, 19: 3, 18: 2), instance 100 for 58 (all from view 8: this instance has 0 pixels in views 18, 12, 14, 6, so view 8 gives that part of the wall another id), no instance for 16, 135 for 4. In the maps of views 18, 12, 14, 6 the strip lies inside the wall's mask; no thin region of its own id.

## Plan: E57c, reprojected white as seeds for the straight-run rule (issue #57, 18.11.2026)
- **Idea (Peter):** where a view's pixels are red (dropped by confidence and oracle), let the confidence-kept (white) points of other views with the SAME SAM 3 instance id inpaint "True" into that view's keep mask by reprojection, each point coloring exactly one pixel
  (no splat diameter); then connect white with green as before (`e57c_two_fills` rule 2). Reason: the squiggle's source pixels (views 12, 14, 6) are blue strips between red and green in their own images; the white that flanks the strip in 3D lies in another view (view 18), so the rule
  found no white end. Reprojecting view 18's white into view 12 puts a white end next to the strip.
- **Rule** (`e57c_reproject.py`): for each target view v, each white pixel of the other views with an instance id >= 0: project its aligned 3D point (GT camera pose, pixel = floor of the projection) into v; if it lands in the image on a RED pixel of v with the same instance id, that pixel is a *seed*.
  No depth test, no radius. Straight-run rule with the seeds counted as white (real white and seeds at one end, green at the other, one instance, blue between). Two kept sets: **A, seeds as white for the rule only** (the seeds themselves are not kept: only blue pixels come back),
  **B, literal reading: the seeds are kept too** (view v's own red points at the seed pixels). Peter's wording ("inpaint True into the red areas of the keep mask") is B; A is the variant without the red points.
- **Checks:** seeds per view; the share of the strip's 586 source pixels brought back (baseline without seeds, A); the gap pixels of the pulled-back render inside the zone that the oracle-alone cloud covers (baseline, A, B); bedroom Chamfer; figures: the pulled-back clouds and the views 18, 12, 14, 6 before / seeds / after.
- **Expected, written before running:** the seeds land mostly on the red wall pixels of views 12, 14, 6 next to the strip; A brings back most of the strip's source pixels (the baseline none or few) and closes the squiggle; B adds redundant pixels and costs Chamfer
  (the oracle dropped them), A is at 1.40-1.60 (baseline 1.608, union 1.400); wrong seeds (occluded projections of the same wall id) are possible since there is no depth test.

## E57c, reprojected white as seeds: the squiggle shrinks by 72 % and the bedroom Chamfer improves to 1.575 cm; keeping the seeds too (B) is worse, 1.674 (bedroom, 18.11.2026)
- **Commands:** `uv run python -m experiments.scripts.e57c_reproject > experiments/results/e57c_reproject.txt` (2 min) -> `figures/e57c_reproject_clouds_{dollhouse_colors,dollhouse_shaded,view18_pulledback_colors,view0_...,view12_...,view6_...}.jpg`,
  `figures/e57c_reproject_views.jpg` (views 18, 12, 14, 6: before, seeds, after, instance ids). Looked at the view 18 clouds and the views figure. Bedroom, all 20 views, cm.
- **Results:** seeds per view 327 to 50 935 (of 15 308 to 95 934 red pixels); blue pixels brought back: 59 368 without seeds, 102 153 with (of 477 291 blue). The strip's 586 source pixels brought back: 0 % without seeds, 45.6 % with. Zone gap pixels of the pulled-back render that the oracle-alone cloud covers:
  1 150 without seeds, 317 with A, 266 with B, 0 for the union. Chamfer (kept): fusion without seeds 1.608 (0.586), **A 1.575 (0.597)**, B 1.674 (0.703), union 1.400 (0.695), oracle alone 1.135 (0.500).
- **Against what was expected (written before running):** seeds land next to the strip: yes, and over large red areas of the door and wall in views 14 and 6 (the same instance, white in another view); A brings back most of the strip: 46 %, not most; closes the squiggle: the hairline on the wall of view 18 is gone in the render, 72 % fewer gap pixels in the zone, 317 remain;
  B costs Chamfer: yes (+0.10 against A, flying debris back in the render); A at 1.40-1.60: 1.575, at the upper edge, 0.03 cm below the fusion without seeds.
- **Reading:** the reprojected white gives the straight-run rule the white end the strip's source view lacked; the remaining gap pixels are strips whose runs still end in red or in another instance. The seeds have no depth test, so a white point seen from behind an occluder can seed a red pixel of the same id (not checked how often).
  The union is still the best Chamfer and has no gap at all. Limits: bedroom only, no dev, the zone chosen by eye, the strip's 586 pixels defined by the baseline.

## Plan: E57c, the greedy oracle on Déjà View's depth lifted through the ground-truth cameras (issue #57, 18.11.2026)
- **Idea (Peter):** keep only Déjà View's depth maps and replace its rays and cameras by the ground truth's (GT intrinsics and pose, pixel centers +0.5), then apply the greedy oracle; he expects an even lower Chamfer.
- **Setting** (`e57c_gt_rays.py --splits dev`, lean: the cached views, GT and geometry only, no SAM 3, no density): per view the points are `lift(s d, K_GT, c2w_GT)`, d Déjà View's z-depth, s the one global scale of the Umeyama alignment of the usual cloud (Déjà View's depth is in its own scale,
  the GT cameras in metres; no other fit of the depth); then the evaluation's single Umeyama per cut as for everything else. Compared in the same run with the usual cloud (Déjà View's ray map replaced by its pinhole fit, `scene_from_geometry`), uncut and the greedy oracle (10 % rounds) at q = 0.9 ... 0.1, dev, 3 scenes, cm.
  This uses the GT cameras: a diagnostic of what the camera error costs, not a method.
- **Expected, written before running (E35: one Sim(3) per view removes 4-7 % of Déjà View's error on Hypersim):** a modest gain, not a large one: uncut 3.500 -> 3.2-3.4, the oracle's best 2.048 -> 1.8-2.0 (5-10 %), the same q (0.5) best; the gain larger on ai_044_001 (the scene of the largest error).

## E57c, Déjà View's depth through the GT cameras: no lower Chamfer, uncut 3.576 against 3.500 and the oracle's best 2.041 against 2.048 (dev, 18.11.2026)
- **Commands:** `CUDA_VISIBLE_DEVICES= uv run python -m experiments.scripts.e57c_gt_rays --splits dev > experiments/results/e57c_gt_rays_dev.txt` (3 min, CPU, 65 s per scene). Dev, 3 scenes, cm, the evaluation's cuts. Tree dirty.
- **Results** (dev mean; kept share, accuracy / completeness / Chamfer; per scene Chamfer ai_010_007 / ai_006_007 / ai_044_001):

  | cloud | cut | accuracy / completeness / Chamfer | per scene |
  |---|---|---|---|
  | usual (Déjà View's pinhole-fit rays) | uncut | 4.675 / 2.324 / 3.500 | 1.60 / 1.85 / 7.05 |
  | GT rays, depth x global scale | uncut | 4.578 / 2.574 / 3.576 | 1.64 / 1.71 / 7.38 |
  | usual | greedy oracle q 0.5 (best) | 1.533 / 2.562 / 2.048 | 0.88 / 1.14 / 4.13 |
  | GT rays | greedy oracle q 0.5 | 1.480 / 2.688 / 2.084 | 0.91 / 1.09 / 4.26 |
  | GT rays | greedy oracle q 0.3 (best) | 1.173 / 2.908 / 2.041 | 0.91 / 1.10 / 4.11 |
  | usual | oracle q 0.3 | 1.295 / 2.903 / 2.099 | 0.88 / 1.04 / 4.38 |

  The best Chamfer over q per scene: usual 0.87 / 1.04 / 4.12, GT rays 0.90 / 1.08 / 4.10.
- **Against what was expected (written before running):** uncut 3.2-3.4: wrong, 3.576, 0.076 above the usual cloud; the oracle's best 1.8-2.0: wrong, 2.041, 0.007 below the usual cloud's 2.048 and at another q (0.3, not 0.5); a larger gain on ai_044_001: its uncut error rises (7.05 -> 7.38),
  its oracle's best moves 4.12 -> 4.10.
- **Reading:** the GT cameras improve the accuracy (uncut 4.675 -> 4.578, at q 0.3 the oracle's 1.295 -> 1.173, -9 %) and lose completeness (uncut 2.324 -> 2.574, +0.25 cm), so the Chamfer does not move. Hypotheses, not tested: the one global scale s fits Déjà View's depth to the GT metric only on average, and its per-view
  scale and the depth errors are consistent with its own cameras (the usual cloud uses them), so GT camera centres expose them; the ray grid of the pinhole fit is already removed in the usual cloud. Together with E35 (camera error 4-7 % of the error on Hypersim) the camera is not where the Chamfer is lost.
- **Limits:** dev only, 3 scenes, no val; the scale s taken from the usual cloud's Umeyama (a per-view or a depth-only fit to the GT was not tried, and would use more GT); the oracle uses the GT; no render.

## Plan: E57c, seeds without any instance information, and a summary of all approaches with their runtimes (issue #57, 18.11.2026)
- **Last idea (Peter):** variant A (reprojected white as seeds for the straight-run rule, the seeds not kept) with the whole scene as one instance: the seeds need no instance match, the runs need no same-instance check. Run on dev and on the bedroom (render, gap pixels of the squiggle zone, views before / seeds / after).
  Control not asked for, needed to separate the effects: the fusion without seeds and without instances.
- **Summary run** (`e57c_summary.py --splits dev`): one pass over dev that scores every approach (Déjà View, confidence only q 0.5, density q 0.6, greedy oracle q 0.5, the one-shot marginal oracle q 0.6, the union of confidence and oracle, the fusion without seeds with and without instances, A with and without instances,
  the product and the mean of the gradients at their dev-best q) and times the stages of each per scene (CPU, 16 cores, one run per stage: the algorithm without loading, SAM 3, the Chamfer scoring), with the optimized variants: `e57c_fast.greedy_fast` (exactly the reference's masks, checked) and the confidence cut by a partition instead of a sort.
- **Expected, written before running:** without instances A seeds red pixels of other instances and bridges more of the blue strips: more of the strip comes back, the squiggle shrinks further, the Chamfer within 0.05 cm of A with instances (bedroom 1.575, dev not yet run) and visual quality the same or slightly more debris;
  every algorithm's own runtime is seconds per scene (the oracle's greedy rounds the largest, 5-9 s on the bedroom); the oracle-free ones (confidence 0.01 s, density 4 s) are cheaper, and none needs a GPU.

## E57c, summary run: seeds without instance information are the best fusion at 2.621 cm on dev; every approach's own runtime is seconds per scene (dev, bedroom, 18.11.2026)
- **Commands:** `uv run python -m experiments.scripts.e57c_summary --splits dev > experiments/results/e57c_summary_dev.txt` (7 min: 117-150 s per scene, almost all of it the 33 Chamfer scorings; figures `e57c_summary_clouds_*.jpg` and `e57c_summary_views_one_instance.jpg` from the bedroom inside the same run; looked at the view 18 clouds and the views figure).
- **Metrics** (dev mean, cm; kept, accuracy / completeness / Chamfer; per scene ai_010_007 / ai_006_007 / ai_044_001): Déjà View 1.000, 4.675 / 2.324 / 3.500 (1.60 / 1.85 / 7.05); confidence only q 0.5 0.500, 2.220 / 48.532 / 25.376; density q 0.6 0.600, 3.550 / 2.827 / 3.188 (1.48 / 1.73 / 6.36);
  greedy oracle q 0.5 0.500, 1.533 / 2.562 / **2.048** (0.88 / 1.14 / 4.13); one-round marginal oracle q 0.6 0.600, 1.720 / 2.683 / 2.202; union (no gate) 0.696, 2.446 / 2.602 / 2.524; product q 0.3 0.708, 2.632 / 2.364 / 2.498; mean q 0.2 0.667, 2.596 / 2.655 / 2.626; fusion without seeds, ids 0.603, 2.513 / 2.848 / 2.680;
  fusion without seeds, one instance 0.608, 2.505 / 2.832 / 2.668; A with ids 0.616, 2.507 / 2.796 / 2.651 (1.17 / 1.58 / 5.20); **A without instances 0.629, 2.497 / 2.744 / 2.621** (1.16 / 1.54 / 5.16). All earlier rows reproduce the earlier dev run.
- **Bedroom** (Chamfer, kept, squiggle-zone gap pixels the oracle-alone cloud covers): fusion without seeds 1.608, 0.586, 1 150; same, one instance 1.597, 0.589, 1 061; A with ids 1.575, 0.597, 317; A without instances **1.539**, 0.610, **202**; union 1.400, 0.695, 0.
- **Against what was expected (written before running):** more of the strip comes back and the squiggle shrinks further: yes (202 against 317 gap pixels); Chamfer within 0.05 cm of A with instances: yes (dev 2.621 against 2.651, bedroom 1.539 against 1.575); visual quality the same or slightly more debris: the same in the view 18 render (the wall continuous, the wardrobe window and the bed cover
  still have holes). The control separates the effects: dropping the instances alone moves the fusion by 0.012 cm on dev (2.680 -> 2.668); the seeds do the work (2.668 -> 2.621 without ids, 2.680 -> 2.651 with them), the instances were not needed.
- **Stage times** (s per scene, dev mean of 3, CPU 16 cores, one run per stage, 3.8 M pixels (20 x 378 x 504); without the loading, the SAM 3 linking and the Chamfer scoring): lift 0.04; confidence cut by sort 1.02, by partition 0.01; density 3.73; Umeyama 0.54; accuracy a_i 1.36; greedy reference 9.26, incremental (`greedy_fast`, identical masks) 5.71; one marginal round 1.01; gate 8 cm 0.53;
  straight runs 1.61 (ids) / 1.66 (one instance); seeds 2.71 / 2.81; runs with seeds 1.67 / 1.71; distance to the kept 0.54; gradient ranking 0.24. **Totals of the optimized variants:** confidence only 0.01; density 3.74; greedy oracle 7.61 (reference 11.16); marginal oracle 2.92; union 7.62; fusion without seeds 9.76; A with ids 12.53; A without instances 12.67;
  product / mean 9.41 (the percentile by a sort, 1.02 of it). Everything except confidence only, density and Déjà View needs the GT (the oracle: align, a_i, greedy), so these totals are the cost of computing the oracle and the fusion, not of a deployable method; a learned head would replace the oracle's 7.6 s by its forward pass (not measured).
- **Limits:** dev only, 3 scenes, no val, no seeds; the q of the product and the mean chosen on dev; the bedroom's visuals are one scene; times from one run per stage on a shared machine; the density's 3.7 s is the reference implementation (not optimized).

## E57c, the figures of the issue-#76 report (18.11.2026)
- **What:** `uv run python -m experiments.scripts.e57c_figures_issue76` (all five, about 3 minutes; `--only algorithms anatomy peeling steps mechanism`); design tokens and layout in `e57c_style.py` (one palette, laid out in inches). Bedroom `ai_006_007`, numbers the dev means of `results/e57c_summary_dev.txt`
  (the accuracy, same-pixel and smoothed oracles from `e57c_oracle_zoo_dev.txt`; the greedy's q curve from the same file). Looked at every figure.
  1. `figures/e57c_issue76_algorithms.png` (and `_dollhouse.png`): all fifteen measured approaches as clouds, grouped (no GT / oracle / oracle + confidence), with the dev Chamfer, Δ against Déjà View, kept share, runtime and a bar on a common 0-3.5 cm scale.
  2. `e57c_issue76_oracle_anatomy.png`: view 18 in six panels: the image, the error (a_i minus the mean), the coverage a point provides (m_i), their net, the greedy order, the kept set at q 0.5.
  3. `e57c_issue76_oracle_peeling.png`: six clouds at q 1.0 ... 0.1 over the dev curves of accuracy, completeness and Chamfer, minimum marked.
  4. `e57c_issue76_fusion_steps.png`: Fusion A without instance information in four steps (Déjà View, confidence threshold, + the oracle's pixels, + the bridge) with a zoom on the seam.
  5. `e57c_issue76_fusion_mechanism.png`: the bridge in image space (view 12): before, the seeds, after.
- **Design decisions:** the saturated colors of the debugging maps are replaced by one palette with one meaning each (steel blue: kept by confidence; teal: kept by the oracle; amber: bridged; rose: the seam; violet: seed; slate: error; light gray: dropped); the cloud ground is slate gray so that holes in white walls show.
  The clouds of figure 1 are bedroom renders and the numbers dev means: both are named in the subtitle.
- **Limits:** figures from one scene; the stage times are those of the dev summary; figure 5's crop is fixed by eye around the strip's source pixels of view 12.

## E57c, what the pull request carries (18.11.2026)
- **Committed:** the oracle scripts and their results (`e57c_oracle_curve`, `e57c_fast`), the fusion and its seams (`e57c_seams`, `e57c_hybrid`, `e57c_two_fills`, `e57c_squiggles`, `e57c_squiggle_categories`, `e57c_reproject`), the gradient cuts and the confidence-only fixes that failed (`e57c_gradients`, `e57c_product`, `e57c_verify`,
  `e57c_voxel_conf`, `e57c_gt_rays`), the summary with the stage times (`e57c_summary`), the figures of the report on issue #76 (`e57c_style`, `e57c_figures_issue76`, six JPEGs) and the text of that report (`results/e57c_issue76_comment.md`); tests for the cost of a point, the greedy and its incremental version, the straight-run rule and the seeds.
- **Not committed:** the debug renders and the one-off scripts of the entries above (the first clouds, the clouds of the hybrids, fills and gradients, the four-color view, the band gate, the view maps, `e57c_dev_methods`, `e57c_clouds`): superseded by `e57c_summary` and the report's figures, their numbers stay in the entries as recorded; the figures they wrote are not in git.
  Result files of versions of scripts that have changed since (`e57c_gradients.txt`, `e57c_two_fills_band08.txt`) are not committed either; `results/e57c_summary_dev.txt` reproduces every dev row of `e57c_dev_methods` and of the earlier runs (rerun after the cleanup: all Chamfers identical, stage times within 0.2 s). `e57c_oracles_dev.txt` and `e57c_oracle_curve_dev_val.txt` come from earlier revisions of `e57c_oracle_curve` (`--rankings accuracy completeness chamfer marginal` reproduces them).
## Plan: E52c = the E52b control's LoRA trained with a cross-view point-to-plane loss whose partners lie in one GT instance, against the same loss with any partner (issue #52, 18.11.2026)
- **Goal:** E52a's segment adapters gained no more with GT instances than with grid tiles, and the
  E52b control (LoRA, no segments) set the bar at 3.154 cm dev Chamfer. Can instances help when
  they only shape the training loss, with no segments at test time? E52c asks the LoRA to make each
  point lie on its own instance's surface as the other views predict it. The same loss with any
  partner, ignoring instances, is the control: the CLAUDE.md rule's "same method without semantics".
- **Loss** (`cross_view_distance`, `map_normals` in `src/backbone/adapters.py`, module docstring):
  loss = GT ratio (as the control) + λ c / c_DV, with c_DV Déjà View's c on the same draw (one
  extra no-grad pass, as the GT ratio), floored at 1e-3. c = mean over 6000 random GT-valid pixels
  of the step's views (the Sim(3)-aligned points of the exact alignment) of
  min(|n_q · (x_p - x_q)| / z_p, 0.05). q is the nearest point of another view and of the same GT
  instance (`cons_gate: instance`) or any (`none`). n_q is the predicted map's normal (central
  differences 2 px apart, detached as ICP's target normals), and z_p is the GT depth. Partners and
  normals come from the prediction and the instance id alone. GT correspondences would make the
  term the GT loss reweighted. Both arms average over the same points (instances seen in 2+ views).
- **Probe before any run** (`experiments/scripts/e52c_probe.py`, figure `e52c_probe.jpg`, raw
  `results/e52c_probe.yaml`; laptop, Déjà View on each dev scene's 12 fixed views; c capped at 0.05):

  | | living room | bedroom | lobby |
  |---|---|---|---|
  | point to point, GT points | 0.0172 | 0.0246 | 0.0238 |
  | point to point, Déjà View | 0.0181 | 0.0268 | 0.0256 |
  | point to plane, GT points, gate / any | 0.0013 / 0.0014 | 0.0048 / 0.0054 | 0.0047 / 0.0055 |
  | point to plane, Déjà View, gate / any | 0.0040 / 0.0038 | 0.0104 / 0.0103 | 0.0086 / 0.0091 |

  The first draft (point to point) scored perfect geometry within 10% of Déjà View. It measures
  how far apart 6000 samples lie, and its gradient would pull points together, so it was dropped
  before any run. Point to plane scores Déjà View 2-3x above GT on all three scenes; normals 7 px
  apart change it by at most 0.0012. The gate changes Déjà View's c by at most 6% (lobby
  0.0086 vs 0.0091). On GT it lowers the floor by up to 15%, as it drops partners across instance
  boundaries.
- **Hypothesis and the numbers that would refute it:** the gated arm beats Déjà View, the control
  and the any-partner arm on dev mean Chamfer, each by more than the larger of 0.05 cm and the
  control's seed range (max - min over seeds 0-2, running now on the 3080), and on 2 of 3 scenes.
  Gated within that margin of any-partner refutes "instances add to this loss". Any-partner within
  it of the control refutes "a cross-view term adds to GT supervision" (no verdict on instances
  then). **Prediction from the probe:** the gate changes what the loss asks of Déjà View by under
  6%, so the two arms end within the spread of each other.
- **Training:** as the control (LoRA r16 on qkv and proj of both attentions of the loop block, lr
  1e-4, 3000 steps, 8-12 views, `align=exact`, bf16, state kept on val, probe every 10 steps).
  Only `cons_weight` and `cons_gate` change. `train_loss` keeps the GT ratio alone, and
  `train_cons_ratio` logs c / c_DV. Laptop smoke (bedroom + living room, 4 views, 4 steps, both
  gates, `runs/e52c_smoke_<gate>`, not of record): runs end to end at 1.43 GB and 3.5 s per step.
  c / c_DV is 1.01-1.02 at step 1 with a zero LoRA: the known gap between the grad (checkpointed)
  and the no-grad path, larger on c than on the GT ratio (~0.999) because c is small.
- **Ladder** (`docs/experiments.md`; 3080, after the control seeds), λ = 1 in rungs 3/4, both arms:
  - Rung 3, bedroom, 200 steps. **Pass:** last-50 GT ratio below 0.90 (the control: 0.843) and
    last-50 c / c_DV below 0.90.
  - Rung 4, dev, 300 steps. **Pass:** last-100 GT ratio below 0.95 on each scene, and c / c_DV
    falling.
  - Sweep λ in {0.1, 0.3, 1} for both arms (500 steps, `--splits` none; least val ratio at step 500).
    The control's lr was swept the same way. Ask before (6 runs, ~2 h).
  - Rung 5, 3000 steps per arm at its λ, dev scored. The seeds per arm are set from the control's
    seed range before these runs (asked; ~2 h per run).
- **Reading:** dev Chamfer (cm) per scene and mean against Déjà View 3.500, the control (mean over
  its seeds) and the other arm; val curves; stability by the probe rule. With rungs 3/4 passed and
  both arms swept alike, a gated arm within the margin of any-partner is "fails for this loss",
  stated for this term only, with the budget and spread logged.
- **Commands** (3080): rung 3
  `uv run python experiments/scripts/e52a_segment_adapters.py --cfg configs/e52a_adapters.yaml --splits --set "models={lora: [false, false, true]}" "segments=[none]" "train_scenes=[ai_006_007_cam_00]" held_out=0 select=val "view_subset=[8, 8]" steps=200 eval_every=200 align=exact cons_weight=1.0 cons_gate=<instance|none> --out runs/e52c_rung3_<gate>`;
  rung 4 as rung 3 with the 3 dev scenes and `steps=300 eval_every=300` (as E52b's rungs); rung 5 as the control's command plus
  `cons_weight=<λ> cons_gate=<gate> seed=<s>`. Probe: `uv run python experiments/scripts/e52c_probe.py`.

## E52c, before the sweep: the control's three seeds span 0.080 cm dev mean, the bar becomes 3.155 cm; both arms pass the GT-ratio rungs at λ = 1, the cross-view ratio falls but misses rung 3's 0.90, and the arms are indistinguishable (issue #52, 18.11.2026)
- **Setting:** the plan entry above. 3080. Control seeds 1, 2 at 813c3e1 (clean), the E52b control's
  command plus `seed=<s>`, `--out runs/e52b_control_s<s>`, 18.11 14:31-18:33 (~2.37 s per step).
  Rungs at bd10710 (clean), 18:33-19:04, λ = 1, `runs/e52c_rung{3,4}_{instance,none}`. Figure
  `e52c_rungs.jpg`, raw `results/e52c_rungs.yaml`.
- **Control seeds, dev Chamfer (cm, 20 views, same Déjà View run):**

  | | living room | bedroom | lobby | mean | kept step |
  |---|---|---|---|---|---|
  | Déjà View | 1.601 | 1.846 | 7.053 | 3.500 | |
  | seed 0 (E52b) | 1.580 | 1.693 | 6.189 | 3.154 | 2500 |
  | seed 1 | 1.470 | 1.678 | 6.438 | 3.195 | 1500 |
  | seed 2 | 1.517 | 1.571 | 6.257 | 3.115 | 3000 |
  | range | 0.110 | 0.122 | 0.249 | 0.080 | |

  Every seed beats Déjà View on every scene, so the control's gain holds across seeds. The bar is
  the seed mean, 3.155 cm. By the plan's rule, E52c's margin is max(0.05, 0.080) = 0.080 cm. The
  per-scene ranges (0.11-0.25 cm) are larger than most per-scene gaps so far, so "2 of 3 scenes" is
  only readable from seed means. The kept step varies from 1500 to 3000: the choice on val is noisy.
- **Rungs, λ = 1:**

  | | gate: same instance | any partner | bar |
  |---|---|---|---|
  | rung 3 GT ratio, last 50 | 0.842 | 0.843 | < 0.90: pass |
  | rung 3 c / c_DV, first 50 -> last 50 | 0.984 -> 0.943 | 0.990 -> 0.940 | < 0.90: missed |
  | rung 4 GT ratio, last 100 (living / bed / lobby) | 0.776 / 0.905 / 0.710 | 0.779 / 0.905 / 0.709 | < 0.95: pass |
  | rung 4 c / c_DV, first -> last 100, lobby | 0.942 -> 0.910 | 0.948 -> 0.907 | falling: pass |

  The GT ratio matches the control's rungs (0.843; 0.910 / 0.775 / 0.706). The cross-view ratio
  falls on every scene in both arms but, on the bedroom, flattens at ~0.94 after step 150. That
  misses rung 3's bar, so the term is not shown to reach a known answer yet. Part of c cannot go:
  GT points score 0.46 of Déjà View's c on the bedroom (probe), so c / c_DV has a floor near 0.46,
  plus normal noise. At λ = 1 the GT ratio may set the gradient. The two arms differ by at most
  0.006 on any number above, as the probe predicted.
- **Reading:** not yet a pass for the term: rung 3 of the cross-view loss is open. Next, before the
  sweep: rung 3 on the bedroom at λ = 10, both arms (~6 min each). If c / c_DV then falls well below
  0.90, the term trains as intended and the sweep should cover larger λ than {0.1, 0.3, 1}.
  Otherwise, check the gradient path (detached normals, partner choice, the cap's share of points).
- **Commands:** as in the plan; figure and raw:
  `uv run python experiments/scripts/e52c_rungs_figures.py`.

## E52c, rung 3 at λ = 10: ten times the weight moves the cross-view ratio from 0.94 to 0.91 and costs the GT ratio 0.84 -> 0.92; neither bar is met, the arms stay indistinguishable (issue #52, 18.11.2026)
- **Setting:** rung 3 of the E52c plan (bedroom, 200 steps, both arms) with `cons_weight=10.0`, at
  bd10710 (clean), 3080, 18.11 22:17-22:29 (~350 s per arm), `runs/e52c_rung3_{instance,none}_l10`.
  Hypothesis (entry above): if the term trains as intended, c / c_DV falls well below 0.90 at
  λ = 10. Refuted if it stays near 0.90 or above.
- **Result (bedroom, 20 views drawn as 8 per step, ratios to Déjà View on the same draw):**

  | | λ = 1, same instance | λ = 1, any partner | λ = 10, same instance | λ = 10, any partner | bar |
  |---|---|---|---|---|---|
  | GT ratio, last 50 | 0.842 | 0.843 | 0.918 | 0.931 | < 0.90 |
  | c / c_DV, first 50 -> last 50 | 0.984 -> 0.943 | 0.990 -> 0.940 | 0.981 -> 0.909 | 0.988 -> 0.907 | < 0.90 |

  Ten times the weight buys 0.03 of c / c_DV and gives back 0.08-0.09 of the GT ratio. At λ = 10
  the two ratios fall together and dip at the same step (180, the seeded draw, also seen at
  λ = 1), so c behaves like a second, weaker copy of the GT error rather than a term the model
  can drive down on its own. The probe's floor (GT scores 0.46 of Déjà View's c) is far below
  0.91. The gate again changes nothing (<= 0.013 on any number). No step was clipped in any of
  the four runs. The median LoRA-up gradient norm grows 2.3 -> 8.2 (λ = 1 -> 10), 3.5x for 10x
  the weight, so the term's gradient is small next to the GT ratio's; since Adam normalises the
  step size, λ only shifts the update's direction towards c, and c still barely moves.
- **Reading:** rung 3 of the cross-view term is still open; the hypothesis is refuted, so the λ
  sweep is not run yet. The term is "not yet working", not failed: what is missing is a check
  of its gradient path. Candidates: the normals are detached and taken from the prediction, so
  moving a point along its partner's normal is the only free direction; nearest partners are
  re-chosen every step, so the target moves with the prediction; the 0.05 cap may silence the
  points that carry most of c. Next: log the capped share and the per-step gradient norm of the
  term against the GT ratio's, and run rung 3 with c alone (needs a GT-ratio weight key, which
  the script lacks), which must reach c / c_DV well below 0.90 if the term can be minimised at
  all.
- **Commands:** as the rung-3 command of the plan with `cons_weight=10.0` and
  `--out runs/e52c_rung3_<gate>_l10`; figure (fourth panel added) and raw:
  `uv run python experiments/scripts/e52c_rungs_figures.py`.

## E52c, plan: rung 3 with the cross-view term alone, the check that it can be minimised at all (issue #52, 18.11.2026)
- **Why:** at λ = 10 c / c_DV stopped at 0.91 while the GT ratio got worse (entry above). Either
  the GT ratio holds c up, or the term cannot be driven down by the LoRA at all.
- **Change:** `gt_weight` (default 1.0, earlier runs unchanged) scales the GT ratio in the loss;
  `gt_weight=0` trains on c / c_DV alone. `train_loss` still records the GT ratio.
- **Hypothesis:** with the term alone, c / c_DV on the bedroom falls well below 0.90 within 200
  steps (last 50 < 0.80). Refuted if it stays >= 0.90: then the term's gradient path is the
  problem (detached normals, partners re-chosen every step, the cap), not its weight. The GT
  ratio is expected to rise; how far shows how much c alone conflicts with the GT.
- **Command** (3080, both gates, ~6 min each): the plan's rung-3 command with
  `cons_weight=1.0 gt_weight=0.0`, `--out runs/e52c_rung3_<gate>_conly`.

## E52c, rung 3 with the cross-view term alone: c / c_DV stops at 0.89-0.90 while the GT ratio climbs to 1.13-1.23, so the term itself cannot be driven down by the LoRA; the λ sweep is dropped until its gradient path is understood (issue #52, 18.11.2026)
- **Setting:** the plan entry above, at 34f1183 (clean), 3080, 18.11 22:43-22:56 (~347 s per arm),
  `runs/e52c_rung3_{instance,none}_conly`.
- **Result (bedroom, 200 steps, ratios to Déjà View on the same draw; λ = 1 and λ = 10 from the
  entries above):**

  | | λ = 1 | λ = 10 | term alone | bar |
  |---|---|---|---|---|
  | c / c_DV, last 50, same instance | 0.943 | 0.909 | 0.902 | < 0.80 (term alone) |
  | c / c_DV, last 50, any partner | 0.940 | 0.907 | 0.890 | |
  | GT ratio, last 50, same instance | 0.842 | 0.918 | 1.133 | |
  | GT ratio, last 50, any partner | 0.843 | 0.931 | 1.228 | |
  | median LoRA-up gradient norm, same instance | 2.32 | 8.19 | 0.80 | |
  | val ratio at step 200, same instance / any partner | 1.002 / 0.999 | 1.002 / 1.001 | 1.086 / 1.134 | |

  Without the GT ratio, c / c_DV ends where it ended at λ = 10 (0.89-0.90; the 20-step means
  flatten from step 160), so the GT ratio was not what held it at 0.91. Meanwhile the geometry
  moves away from the GT: the GT ratio passes 1.0 at step ~90 and keeps rising (20-step peak
  1.22 / 1.33), and val worsens by 9-13%. The term's own gradient is small: its median norm
  alone is 0.80, about a third of the GT ratio's (2.3 at λ = 1), and 10 x 0.80 matches the 8.2
  of λ = 10. The probe's floor (GT scores 0.46 of Déjà View's c) is far away, so the LoRA lowers
  c by distorting the scene, not by moving it towards the GT. The gate changes c by 0.012.
- **Reading:** refuted: the term alone does not reach a known answer (rung 3), so a λ sweep or
  seeds would only measure a term that does not work as intended. The term is "not yet
  working": what is missing is a solver-level check that removes the network. Next, on cached
  Déjà View output: minimise c directly over per-pixel depth (or a per-view affine) of the
  bedroom's 20 views and see whether it reaches the GT's c (0.46) and moves the points towards
  the GT. If it does, the LoRA's parameterisation limits the term; if not, the term (detached
  prediction normals, partners re-chosen every step, the 0.05 cap) has minima away from the GT.
- **Commands:** the plan's command; figure (rung-3 panel for the term alone added) and raw:
  `uv run python experiments/scripts/e52c_rungs_figures.py`.

## E52c, plan: solver check of the cross-view term without the network (issue #52, 19.11.2026)
- **Why:** trained alone, the term stopped at c / c_DV 0.89-0.90 while the GT error grew past
  Déjà View's (entry above). Either the LoRA cannot express the change the term asks for, or the
  term's minima lie away from the GT.
- **Setting:** `experiments/scripts/e52c_solver.py`. Déjà View's cached depth of the bedroom's 12
  probe views (pinhole rays), Adam on a log-scale per 14 px patch (bilinear) or a scale and
  offset per view, 1000 steps, lr 1e-3, the exact Sim(3) re-fit every step, 6000 drawn pixels.
  Loss: c gated by instance, c with any partner, or the GT error (the solver's known answer).
  lr 1e-2 raised c in every arm within 4 steps (depth steps of 1% are twice Déjà View's c), so
  1e-3 was set on a 50-step laptop smoke test before this entry (patch arms: c 0.75, GT ratio
  1.14 at step 50; GT arm: GT ratio 0.71, c 0.86).
- **Hypothesis:** the term's minimum lies near the GT: minimising c on patches reaches c / c_DV
  near GT's (0.49 gated on this draw) and the GT ratio ends below 1. Refuted if the GT ratio
  ends above 1 while c falls (the term's minima lie away from the GT; then a different term,
  not a different λ or network, is needed). The GT arm must reach a GT ratio well below 1 for
  the check to count.
- **Command** (3080, ~1-2 min per arm): `uv run python experiments/scripts/e52c_solver.py`.

## E52c, solver check: minimising the cross-view term on Déjà View's own depth moves away from the GT, first directly, then by shrinking the alignment; the term's minima lie away from the GT (issue #52, 19.11.2026)
- **Setting:** the plan entry above. Run of record at b91def6 (clean), 3080, 19.11 00:50-00:54
  (~40 s per arm). Two earlier runs at 95adc6d and 3076a05 gave the same ratios to the third
  decimal; b91def6 only adds the Sim(3) scale and wider depth quantiles to the record. The
  figure's labels were redrawn from the yaml with `--plot` (no new numbers). GT's own c / c_DV on
  the fixed draw: 0.492 (same instance), 0.550 (any partner).
- **Result (bedroom, 12 views, ratios to Déjà View; scale = the Sim(3)'s scale over Déjà View's;
  depth = final over start depth, 0.1 / 5 / 50 / 95 / 99.9% quantiles):**

  | free variables, loss | GT ratio | c / c_DV (own rule) | scale | depth quantiles |
  |---|---|---|---|---|
  | patch, c same instance, step 100 | 1.43 | 0.696 | 0.985 | |
  | patch, c same instance, step 1000 | 17.26 | 0.378 | 0.466 | 0.89 / 0.97 / 1.00 / 1.05 / 4.17 |
  | patch, c any partner, step 1000 | 17.24 | 0.368 | 0.467 | 0.89 / 0.96 / 1.00 / 1.05 / 4.17 |
  | patch, GT error (known answer) | 0.679 | 0.854 | 0.996 | 0.90 / 0.98 / 1.00 / 1.02 / 1.08 |
  | view, c same instance | 0.983 | 0.937 | 0.998 | 0.95 / 0.99 / 1.00 / 1.01 / 1.01 |
  | view, c any partner | 0.983 | 0.938 | 0.999 | 0.95 / 0.99 / 1.00 / 1.01 / 1.01 |
  | view, GT error (known answer) | 0.929 | 0.971 | 0.996 | 0.97 / 0.99 / 1.00 / 1.01 / 1.01 |

  The solver passes its own check: fitting the GT on patches lowers the GT ratio to 0.68. Two
  ways c is lowered away from the GT:
  1. Directly: in the first 100 steps c falls 30% while the GT error rises 43%, with the fit's
     scale still at 0.985. Déjà View's depth is not near a minimum of c that is near the GT:
     c's descent direction from there moves the points off the GT.
  2. Through the alignment: afterwards 0.1% of the pixels are pushed to ~4x their depth, the
     least-squares Sim(3) shrinks the cloud to 0.47 of its size, and c, divided by the fixed GT
     depth, shrinks with it, to 0.38, below GT's own 0.49. 90% of the depth moves by under 5%.
  Fitting the GT lowers c only to 0.85, far from GT's 0.49: most of the c that separates Déjà
  View from GT lies below the 14 px patch scale. Per-view scale and offset barely move either
  number. The gate again makes no difference (0.011).
- **Reading:** the hypothesis is refuted: the cross-view term's minima lie away from the GT on
  the bedroom, so no λ, network or seed can make it a proxy for GT error. This explains rung 3
  with the term alone (c 0.89-0.90, GT ratio 1.13-1.23); whether the LoRA also used the scale
  route was not recorded there. Limits: one scene, 12 views, patch-scale freedom. A term that
  could work has to (a) be invariant to the alignment's scale, e.g. divided by the predicted
  depth or computed before the Sim(3), which removes route 2 only, and (b) descend towards the
  GT from Déjà View's output; the cheap test for (b) is the cosine between the gradients of c
  and of the GT error at Déjà View's depth, before any training.
- **Commands:** `uv run python experiments/scripts/e52c_solver.py` (3080, ~4 min);
  `--plot` redraws the figure from the yaml. Figure `e52c_solver.jpg`, raw
  `results/e52c_solver.yaml`.

## E52c closed: segments in the training loss are not yet working; the cross-view term has to be replaced, not tuned (issue #52, 19.11.2026)
- **What was tested:** LoRA on frozen Déjà View with the GT ratio plus a cross-view
  point-to-plane term whose partners lie in the same GT instance, against the same term with
  any partner (the semantics-free twin). Bar: the LoRA control's seed mean, dev 3.155 cm,
  margin 0.080 cm (entries above).
- **Where it stopped:** before the λ sweep and before rung 5, so no dev score of E52c exists.
  The term never reached a known answer: c / c_DV on the bedroom stayed at 0.94 (λ = 1),
  0.91 (λ = 10) and 0.89-0.90 (term alone), and the solver check showed its minima lie away
  from the GT (c falls while the GT error rises, then by shrinking the alignment). The instance
  gate changed no number by more than 0.013, as the probe predicted (<= 6%).
- **Status:** "not yet working", not "fails": missing are a term whose descent agrees with the
  GT error at Déjà View's output (test: gradient cosine, scale-invariant c) and then the sweep,
  seeds and ablations. The E52b control (LoRA without segments, dev 3.500 -> 3.155 cm over 3
  seeds) stays the bar for any segment method in #52.

## E52e, plan: does a scale-invariant cross-view term point towards the GT at Déjà View's output; gradient cosine before any training (issue #52, 19.11.2026)
- **Why:** E52c closed with the cross-view term's minima away from the GT, by two routes: c fell
  while the GT error rose in the first 100 solver steps, then fell further by shrinking the
  Sim(3) (c was divided by the fixed GT depth after the fit). E52c's reading named the cheap
  test for a replacement: the cosine between the gradients of the term and of the GT error at
  Déjà View's depth. A positive cosine is the first-order condition for the term to lower the
  GT error; without it no λ, network or seed can help.
- **Setting:** `experiments/scripts/e52e_gradient_cosine.py`, the solver check's problem
  (bedroom, 12 probe views, pinhole rays, log-scale per 14 px patch at zero = Déjà View's
  depth). Terms: E52c's c as built (after the exact Sim(3), over GT depth), and an invariant c:
  the same point-to-plane distance on the raw points before any alignment, over the predicted
  depth, so a similarity leaves it unchanged (asserted at run time, drift < 1e-3) and route 2
  is gone. Each gated by GT instance and with any partner. Target: the GT error's gradient
  over all GT-valid pixels. A term's gradient is the mean over 20 draws of 6000 px; the GT
  error on the same draws is the ceiling, 200 random directions the floor; per-draw cosines
  are recorded as one training step would see them. Added after a first run had
  shown a positive cosine (cosines unchanged by the addition): the cosine of sign(gradient), the
  direction of Adam's first steps, and the GT error after steps along each direction with an
  rms of 0.1%, 1% and 3% of log-depth, against a random direction of the same size.
- **Hypothesis:** the invariant c descends towards the GT: its mean-gradient cosine with the
  GT error's is above 0.3. Refuted if it is at or below 0.1 (or within 2 std of the random
  floor): then the term's direction at Déjà View's output is not the GT's, and route 1 is a
  property of cross-view agreement itself, not of the normalisation. Expected from E52c's
  route 1 (c down 30%, GT error up 43% at scale 0.985): the built c's cosine is near zero or
  negative; the gate again changes nothing (<= 0.05).
- **Command** (laptop 3050, ~20 s, no training):
  `uv run python experiments/scripts/e52e_gradient_cosine.py`.

## E52e: the cross-view term points slightly towards the GT at Déjà View's output (cosine 0.19, with or without scale invariance), but a step of 1% of depth along it already raises the GT error almost as much as a random step; not a usable objective (issue #52, 19.11.2026)
- **Setting:** the plan entry above. Run of record at 677f81a (clean), laptop 3050, 17 s. A first
  run before the commit gave the same numbers to the third decimal. 11664 patch log-scales;
  random-direction cosine std 0.0097; the invariant term changed by 3.6e-7 under a random
  similarity.
- **Result (bedroom, 12 views; cosine with the GT error's gradient over all GT pixels; GT ratio =
  GT error after a step along minus the direction / before, step rms in log-depth):**

  | direction | cosine, 20-draw mean gradient | cosine, one draw | cosine of sign | GT ratio, step 0.1% | 1% | 3% |
  |---|---|---|---|---|---|---|
  | GT error, same 20 draws (ceiling) | 0.974 | 0.697 +- 0.005 | 0.858 | 0.964 | 0.850 | 1.532 |
  | E52c's c, same instance | 0.193 | 0.108 +- 0.011 | 0.203 | 0.995 | 1.047 | 1.472 |
  | E52c's c, any partner | 0.193 | 0.107 +- 0.011 | 0.194 | 0.995 | 1.047 | 1.470 |
  | invariant c, same instance | 0.191 | 0.107 +- 0.011 | 0.227 | 0.995 | 1.048 | 1.471 |
  | invariant c, any partner | 0.191 | 0.106 +- 0.011 | 0.217 | 0.995 | 1.048 | 1.469 |
  | random direction | 0 +- 0.010 | | | 1.001 | 1.059 | 1.397 |

  1. The cosine is positive, about 20 random-direction stds above zero: at Déjà View's output
     the term's descent does point towards the GT, and so does its sign (Adam's direction).
     The plan's expectation for E52c's term (near zero or negative) was wrong.
  2. It is weak: 0.19 means 98% of a step's length is orthogonal to the GT error's descent.
     A step of 0.1% of depth lowers the GT error by 0.5% (the GT's own step: 3.6%); at 1% it
     raises it by 4.8%, against 5.9% for a random step and -15% for the GT's. One 6000-px draw,
     what one training step sees, gives 0.11.
  3. Scale invariance changes nothing at first order (0.191 vs 0.193): E52c's route 2 (shrinking
     the alignment) is what many steps do, not the starting direction. The gate changes no
     cosine by more than 0.003, a fourth time.
- **Reading:** by the plan's numbers the hypothesis is neither supported (0.19 < 0.3) nor refuted
  (> 0.1). The step check settles what the cosine left open: the useful component is real but
  is lost by a step of 1% of depth, so the term cannot drive refinement on its own (as in E52c's
  term-alone rung and solver), and with the GT loss it would mostly add an orthogonal push.
  This is consistent with the team's reprojection refiner on val this week (cross-view votes vs
  error r ~ 0.15): Déjà View's views agree with each other, so cross-view disagreement carries
  little of its error, whatever the normalisation or the gate. Limits: one scene, 12 views,
  first order at Déjà View's output, patch log-scales, not the LoRA's weight space. Status of
  segments in the training loss (#52): "not yet working"; missing is a signal whose direction
  is substantially the GT's, which cross-view agreement within an instance is not here.
- **Commands:** `uv run python experiments/scripts/e52e_gradient_cosine.py` (laptop, 17 s);
  `--plot` redraws the figure from the yaml. Figure `e52e_gradient_cosine.jpg`, raw
  `results/e52e_gradient_cosine.yaml`.
- **Code removed after E52e:** `cross_view_distance` and `map_normals` (src/backbone/adapters.py,
  with their tests) and the `cons_*` / `gt_weight` keys of `e52a_segment_adapters.py` served only
  E52c and E52e, so they left src once both were answered; the E52a/E52b training script and
  config are back to their state on main. To rerun E52c's rungs, probe and solver check or E52e,
  check out 33bca5f.

## Plan: E84a, a keep head trained on the expected Chamfer of the kept cloud itself, not on the oracle's mask (issue #84, 20.11.2026)
- **Why:** #84 asks for a model that predicts the greedy oracle's keep set. Imitating a mask hides the real
  objective, the minimal Chamfer of the kept cloud, and the greedy is only a heuristic (K = 5 / 20 rounds: 2.048 /
  1.993 on dev, not a lower bound). E84a trains on the objective itself: one head, one loss, the expected Chamfer of
  the cloud that keeps prediction i with probability s_i.
- **Head:** a copy of the confidence branch of Déjà View's convolutional depth head (`dvlt.model.dvlt.heads.DecoderHead`,
  `output_block`: Conv 3 x 3 (64 + 2 UV -> 32), ReLU, Conv 1 x 1 -> out), one output, fed with the features that
  `output_block` reads in the frozen model (66 channels per pixel, 378 x 504). s = sigmoid(logit). Fresh initialisation
  (logit 0, s about 0.5), not DV's confidence weights, so DV's confidence is not a starting point. Only the head
  trains. Cached per scene as fp16; the backbone never reruns after the cache.
- **Loss** (per scene, N = pixels with GT depth, prediction i and GT point j share a pixel; cm; the predictions
  aligned once by Umeyama on all N, scene_eval's own path):
  A(s) = sum_i s_i a_i / (sum_i s_i + eps), C(s) = 1/N sum_j [sum_{r <= R} d_jr s_pi_j(r) prod_{r' < r} (1 - s_pi_j(r')) +
  d_j,R+1 prod_r (1 - s_pi_j(r))], L = (A + C) / 2, R = 8 in the config, eps = 1e-6. a_i: distance to the nearest GT
  point; d_jr: the r-th nearest prediction of GT point j. C is the exact expectation of the completeness when every
  prediction is kept independently with probability s, if one of the R probes is kept; where all R are dropped it
  charges d_j,R+1, a lower bound of the true distance. A is a ratio of expectations, not the expectation of the ratio.
  No other term.
- **Minibatches:** none of the loss. A and C are whole-scene sums, so a step is one scene, whole: s of all pixels is
  computed without grad in chunks of views, L and dL/ds are computed on the full s, and the chunks are re-run with
  grad and backpropagated with their slice of dL/ds (gradient caching). The gradient equals the full-batch one
  (tested), no sub-sampling, no rescaling; a step over several scenes averages the scenes' L, as the dev mean does.
- **Evaluation** (dev, never test): per scene and mean, A, C, Chamfer of (a) the exact Chamfer of the kept cloud in the
  loss's fixed alignment (KD-tree, not the loss's probe lists), next to the loss value L and the kept share, for s >= t
  (thresholded) and for Bernoulli(s) draws (seeds 0-4, mean and spread); (b) scene_eval's protocol, one Umeyama per
  output on the kept pixels, for the same cuts, Déjà View uncut, and Déjà View's confidence at the thresholded kept
  share. The cut t is picked on dev only. Completeness is against the depth of the same 20 views. A falling L with a
  rising exact completeness means the loss is gamed.
- **Rungs** (docs/experiments.md): 3 overfits `ai_006_007_cam_00` (20 views, one step = the scene), 4 the three dev
  scenes. Adam, lr 1e-3 constant, no weight decay, no scheduler, seed 0, bf16 off. Rung 5 (train scenes, val) is not
  part of E84a: it exceeds 30 min and needs the train scenes, so a result here is a diagnostic, not a score.
- **Hypothesis, written before running:** the head's thresholded (t = 0.5) exact Chamfer on the bedroom ends below the
  accuracy oracle's 1.41 cm (a pointwise ranking by a_i with the GT, E57c) and at best near the greedy's 1.04 (uncut
  1.846); the head sees only 3 x 3 pixels of DV's features, so I expect 1.1 to 1.4, and exceeding 1.04 is possible
  because the greedy is not a bound and E84a optimises soft keep. **Refuted** if the bedroom's exact thresholded
  Chamfer is above 1.41 after the step budget: then a pixelwise head on this objective cannot even match the accuracy
  oracle, and #84's premise of needing all views at once stands. Rung 4: refuted if the dev mean (scene_eval) is above
  the accuracy oracle's 2.479 (greedy 2.048, density 3.188, uncut 3.500). Gaming: refuted if exact completeness at
  the threshold exceeds the uncut cloud's while L falls. Bernoulli: its mean exact Chamfer within 10 % of L when s is
  near binary; DV's confidence at the same share far worse (E57c: 25.4 at q 0.5 on dev).
- **Ablation, if the main run works:** the sigmoid replaced by clamp(logit, 0, 1) (same bias-free head, logit
  initialised at 0.5 by the bias); not run before the main result is in.
- **Commands:** `uv run python -m experiments.scripts.e84a_train --cfg configs/e84a_overfit_bedroom.yaml`, then
  `configs/e84a_overfit_dev.yaml`; evaluation `uv run python -m experiments.scripts.e84a_eval --run runs/<run>`
  and `scene_eval` with `configs/e84a_eval.yaml`.

## E84a: not yet working; the expected-Chamfer loss as specified has its global minimum at the empty cloud, and rung 3 trains straight into it (bedroom, 20.11.2026)
- **Code:** `src/refinement/keep_chamfer.py` (head, loss, probes, exact Chamfer, gradient caching), `tests/test_keep_chamfer.py`,
  `experiments/scripts/e84a_{train,loss_profile,figures}.py`, `configs/e84a_overfit_{bedroom,dev}.yaml`. The head is
  a copy of `DecoderHead.output_block` (checked against the installed dvlt in a test) on the 66 channels it reads, 19.3 k
  parameters; features cached as fp16 (503 MB per scene), the backbone re-run once per scene (its depth asserted to
  stay within 1e-2 median relative of the cached geometry). Whole-scene sums, no minibatch bias: s of all pixels without
  grad, L and dL/ds on the full s, then the head re-run per chunk of views with grad and backpropagated with its slice
  of dL/ds; equals the full-batch gradient (test, chunks 1, 2, 5). Tests (CPU, seconds): at binary s with every GT point
  keeping a probe L equals the brute-force Chamfer; C equals the enumerated expectation over all 2^7 draws (R = n - 1);
  `exact_chamfer` equals brute force with R = 2 where the fallback query runs. The all-kept loss is 1.8458 cm,
  scene_eval's Déjà View bedroom 1.846.
- **Commands** (RTX 3080, tree dirty = the new files, base 5012dda): `uv run python experiments/scripts/e84a_train.py
  --cfg configs/e84a_overfit_bedroom.yaml` -> `runs/e84a_overfit_bedroom-20261120-084816` (R = 8, eps 1e-6, Adam 1e-3
  constant, 1000 steps, seed 0; one step = the scene's 3.81 M pixels, 0.22 s median, 3.0 GB peak, 2.2 s to load the cached
  scene, the first-run caches (re-run backbone, KD-tree probes) took about 15 s); `e84a_loss_profile.py` ->
  `results/e84a_loss_profile.txt`; `e84a_figures.py` -> `figures/e84a_{curves,s_views}.jpg`. The same trajectory came
  out in three runs; the first two crashed at step 225 in my own evaluation, which asserted a non-empty cloud, and
  were rerun with the evaluation fixed.
- **Result, rung 3** (bedroom ai_006_007; cm, fixed alignment; "exact" = KD-tree Chamfer of one Bernoulli(s) draw,
  seed = step; Déjà View uncut 1.846 = A 2.520 / C 1.172):

  | step | L (loss) | A / C of the loss | mean s | exact Chamfer, Bernoulli | exact completeness, Bernoulli | exact Chamfer, s >= 0.5 |
  |---|---|---|---|---|---|---|
  | 0 | 1.926 | 2.559 / 1.293 | 0.538 | 1.877 | 1.283 | 1.831 (everything kept) |
  | 25 | 1.566 | 1.324 / 1.809 | 0.195 | 2.782 | 4.247 | 51.5 |
  | 100 | 1.490 | 1.164 / 1.815 | 0.180 | 7.090 | 13.015 | 47.6 |
  | 200 | 1.455 | 1.056 / 1.855 | 0.088 | 14.218 | 27.403 | 53.8 |
  | 225 | 1.190 | 0.455 / 1.926 | 0.000 | 152.99 | 305.58 | nothing kept |
  | 600-1000 | 0.963 | 0.000 / 1.926 | 0.000 | nothing kept | - | nothing kept |

  The loss ends at 0.963 cm, below the greedy oracle's 1.04 and the uncut 1.846, with an empty cloud.
- **Why** (`results/e84a_loss_profile.txt`, uniform s = c): for c >= 0.5 the loss equals the exact Bernoulli Chamfer to 0.1 %
  (c = 0.5: 1.917 against 1.918; the share of GT points without a kept probe is 0.4 %), so the loss is right where its R
  probes suffice. As c falls the share rises (0.17 at c = 0.2, 0.92 at c = 0.01) and the loss charges those points
  d_j,R+1, a lower bound, so it undershoots (c = 0.01: 2.21 against 3.06). The empty cloud is the limit: A -> 0
  (sum s -> 0 with eps = 1e-6 in the denominator) and C = mean d_j,R+1 = 1.927 cm, L(0) = 0.964 cm; R = 16, 32, 64 give
  L(0) = 1.14, 1.38, 1.70 cm, all below the uncut cloud. So no finite R fixes it: switching points off is always
  cheaper than the truncation charge, and the accurate-points-only A term pulls the same way. `figures/e84a_s_views.jpg`
  (views 18, 0, 12, 6): at step 50 the head keeps the near floor and bed, where a_i is small, and switches off the
  walls, where a_i is 3-5+ cm and where a dropped list is charged only d_9; by step 200 almost nothing is above 0.5.
- **Against what was expected (written before running):** the bedroom's thresholded exact Chamfer below the accuracy
  oracle's 1.41: **refuted** (51 cm at step 25, then an empty cloud; the only value near the uncut one is step 0's, where
  every s is above 0.5). The gaming criterion (L falls while the exact completeness rises) fired from step 25 on: L
  1.926 -> 0.963 while the Bernoulli completeness goes 1.28 -> 4.2 -> 305 cm. Bernoulli within 10 % of L: true only where
  s >= 0.5 holds almost everywhere (step 0: 1.877 against 1.926), false after step 25.
- **Reading:** not yet working, and not a failure of the head, the optimizer or the data: the objective is wrong where
  it is cheapest to be wrong. The expected completeness is exact only while every GT point keeps a probe among its R
  nearest; the loss gives the optimizer a way out of that regime that pays. This is a loss defect in rung 3's sense
  (docs/experiments.md: targets, loss or optimizer), so the ladder stops here: rung 4 (dev) and the dev evaluation against
  Déjà View, its confidence at the same kept share and the oracle were not run, since the head keeps nothing. The next
  step is a change at this rung, one change: E84b charges a GT point whose R probes are all dropped an upper bound of
  its distance instead of the lower bound d_j,R+1, so leaving the covered regime costs more than the point is worth.
- **Limits:** one scene, one seed, R = 8 (R = 16-64 only through L(0), not trained), one learning rate; the sigmoid-vs-clamp
  ablation not run (the sigmoid is not what fails). The exact Chamfer is at the loss's fixed alignment, scene_eval's
  per-cut Umeyama was not used since no cut survived. The profile's Bernoulli numbers are means of 3 draws.

## Plan: E84b, E84a's loss with an upper-bound charge for a GT point whose R probes are all dropped (issue #84, 20.11.2026)
- **Why:** E84a trained into the empty cloud because a miss (all R probes of a GT point dropped) is charged d_j,R+1, a
  lower bound of the true distance, so the loss undershoots exactly where the head switches points off, and
  L(0) = mean d_R+1 / 2 = 0.96 cm lies below every real cloud. One change, nothing else: the miss is charged an upper
  bound U of the distance from any GT point to any prediction, U = 100 (max_j |t_j - c| + max_i |p_i - c|) cm with c the
  GT mean (triangle inequality; a scene constant, no tuned value). The loss is then an upper bound of the
  expected Chamfer that is exact where every GT point keeps a probe, and L(0) = U / 2 (empty cloud: all points miss).
  Config key `miss: next` (E84a) or `bound` (E84b); R = 8, eps, head, optimizer, seed, steps as in E84a rung 3.
- **Hypothesis, written before running:** the head no longer collapses: mean s stays above 0.05 for all 1000 steps and the
  Bernoulli exact Chamfer stays within 10 % of L after step 100, because a binary-ish s covers every GT point's probe list
  (the oracle's structure) and the loss is then exact. The bedroom's thresholded (t = 0.5) exact Chamfer falls below the
  accuracy oracle's 1.41 cm (uncut 1.846, greedy 1.04). **Refuted** if (a) it collapses again (nothing kept), (b) it stays
  at s -> 1, the uncut cloud's 1.846 (the loss then has a local minimum at keeping everything), or (c) the thresholded
  exact Chamfer ends above 1.41. A different outcome, e.g. a fractional s everywhere with Bernoulli far from the
  threshold, is reported as such. If (c) fails but the others hold, the pixelwise head's 3 x 3 field is the limit, E84a's
  question.
- **Command:** `uv run python experiments/scripts/e84a_train.py --cfg configs/e84b_overfit_bedroom.yaml`; rung 4 and the dev
  evaluation only if rung 3 passes.

## E84b: charging a miss an upper bound removes the collapse, and the head is stable and honest but stays at 97.5 % kept; bedroom thresholded Chamfer 1.747 against the uncut 1.846, still falling at step 1000 (bedroom, 20.11.2026)
- **Command** (RTX 3080, tree dirty = the new files, base 25868a7): `uv run python experiments/scripts/e84a_train.py
  --cfg configs/e84b_overfit_bedroom.yaml` -> `runs/e84b_overfit_bedroom-20261120-085532`: E84a's rung 3 with
  `miss: bound` (U = 100 (max |t - c| + max |p - c|) cm), nothing else. Figures `figures/e84b_{curves,s_views}.jpg`,
  looked at. A test shows the loss with this charge is an upper bound of the brute-force completeness and equal to it
  when every GT point keeps a probe.
- **Result** (cm, fixed alignment; Déjà View uncut 1.846 = A 2.520 / C 1.172; exact = KD-tree on the cloud):

  | step | L | mean s | s in (0.05, 0.95) | exact Chamfer, Bernoulli draw (A / C, kept) | exact Chamfer, s >= 0.5 (A / C, kept) |
  |---|---|---|---|---|---|
  | 0 | 3.099 | 0.538 | 1.00 | 1.902 (2.554 / 1.250, 0.66) | 1.846 (all kept) |
  | 25 | 1.843 | 0.973 | 0.05 | 1.842 (2.508 / 1.177, 0.97) | 1.846 |
  | 100 | 1.793 | 0.919 | 0.56 | 1.784 (2.368 / 1.201, 0.92) | 1.795 (2.410 / 1.179, 0.99) |
  | 500 | 1.753 | 0.904 | 0.41 | 1.745 (2.285 / 1.204, 0.90) | 1.770 (2.358 / 1.181, 0.985) |
  | 1000 | 1.733 | 0.901 | 0.36 | 1.723 (2.246 / 1.201, 0.90) | **1.747** (2.311 / 1.183, 0.975) |

  The loss tracks the exact Chamfer of its own cloud within 0.6 % from step 25 on (1.733 against 1.723 at step 1000), the
  gaming of E84a is gone, and the exact completeness never leaves 1.17-1.23 cm. 0.34 s per step (E84a 0.22: the
  GPU was shared).
- **Against what was expected (written before running):** (a) collapse: no; the head keeps 97.5 % at s >= 0.5. (b) stays
  at the uncut cloud: nearly; step 25 is the all-keep state (mean s 0.973, 1.842) and the loss then falls only 0.1 cm in
  975 steps, 1.747 thresholded and 1.723 Bernoulli, 5.4 % and 6.7 % below the uncut cloud, and it is still falling at
  the end (-0.006 cm per 125 steps). (c) thresholded Chamfer below the accuracy oracle's 1.41: **refuted** at this
  budget (1.747). Bernoulli within 10 % of L: yes (0.6 %). The loss and the exact Chamfer agree, so what fails is the
  descent, not the objective.
- **Reading** (hypotheses, not tested): the first 25 steps push every s to 1 to remove the miss charge (U P(miss) is 1000
  times a centimetre), and from there a fractional s costs: at s = 0.6 a GT point misses with 0.4^8 = 7e-4, times U
  of about 10 m. Dropping a point is cheap only when its GT neighbours' other probes sit at s = 1, a structured
  (binary) state the gradient at a uniform s = 0.9 does not point to. Figure: after 1000 steps s is low (0.4-0.6) on
  the far walls of views 12 and 6, where a_i is largest, and dips on depth edges, so the head learned the
  accuracy ordering, but nowhere low enough to pay. The sigmoid saturating near 1 (s = 0.97 has a gradient factor of 0.03)
  also slows it. Whether 10 times the learning rate escapes this (optimization) or not (landscape, or the 3 x 3 field) is
  the next, single change: E84c.
- **Limits:** one scene, one seed, one learning rate, 1000 steps; rung 3 not passed, so no dev rung and no dev evaluation.

## Plan: E84c, E84b with ten times the learning rate (issue #84, 20.11.2026)
- **Change:** `lr: 1.0e-2` (E84b: 1.0e-3), `configs/e84c_overfit_bedroom.yaml`, 1000 steps, all else as E84b; one change.
- **Hypothesis, written before running:** E84b is optimization-limited, not landscape-limited: with 10 times the step the
  bedroom's thresholded exact Chamfer falls below 1.41 cm within 1000 steps and the kept share below 80 %.
  **Refuted** if it ends above 1.41 (then the descent stalls in the landscape or the 3 x 3 field and a larger lr does not
  help; I expect then 1.6-1.75), or if it collapses or diverges (nothing kept, NaN). If it reaches 1.41 only at a
  fractional s with Bernoulli far from the threshold, reported as such.
