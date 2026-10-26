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

Earlier entries (E4, E7, E9) first reported precision and recall with scorer v1; the values above supersede them
(differences: recall +0.01 from ignoring the "SPLIT" label, precision +0.02 to +0.05 from ignoring masks on
unlabeled GT). id consistency, switches and merges are unchanged.

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
