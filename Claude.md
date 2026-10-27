# CLAUDE.md

Training-free refinement of frozen feed-forward 3D reconstructions, guided by multi-view panoptic segments.
Goal: fused surfaces that are more accurate and sharper than the raw prediction, and than the same refinement
without semantics, at a small multiple of the forward pass. Supervisors: Umut Kocasarı, Robin Borth (Niessner Lab).
Title: Semantic-Guided Refinement for Feed-Forward 3D Reconstruction.
Team (TUM, in this order; write the names exactly so, diacritics included):
Pouya Shekarchizadeh Esfahani (data loading, Déjà View inference; branch `pouya`), Peter Hofinger,
Mircea-Alexandru Răuțoiu.

## Bar
This repository is a public portfolio piece, read by screeners and interviewers for research-engineering roles.
They skim the README, open two files and scroll the history; every one of those must hold up to a critical read.
- Substance over signal: a measured claim with its baseline beats any badge, buzzword or abstraction.
- Every number names its baseline, its scenes and how it was measured. Negative results are reported as plainly
  as positive ones; a clean "this did not work, because" is a strong signal, an inflated gain is a fatal one.
- Peter must be able to defend any line in an interview. No code that only works by accident, no trick without
  a reason; non-obvious choices are explained in the docstring or the commit body.
- Credit precisely: teammates' work, the papers an idea comes from, and third-party code with its licence.

## Method
Per view i: depth D_i, confidence C_i, ray map (origin o, direction d); points x = o + D·d. Cameras stay fixed.
Refine inverse depth per segment s: d̂_i(p) = a_{i,s}·d⁰_i(p) + b_{i,s} + δ_i(p), minimizing with L-BFGS

  E = E_d + λ_mv·E_mv + λ_bd·E_bd + λ_pr·E_pr + λ_δ·‖δ‖²

- E_d: Huber anchor to the prediction, weighted by confidence × fraction of views that agree on the segment.
- E_mv: cross-view depth agreement, only where the projection lands in the same segment.
- E_bd: smoothing within segments only, so boundaries stay sharp.
- E_pr: planarity for structural classes (wall, floor, table).

Scope: monocular refinement of one view (`refine.py`), and joint refinement of all views of a scene with E_mv
(`multiview.py`, no segments yet, solved by Gauss-Newton and CG).

Status: δ only (`refine.py`), no segments, no a or b; verified on a synthetic room with GT. `refine.py` also
takes BiNI's normal term E_n (log depth, cut on the prior's occlusion contours) under the same L-BFGS. On real data,
Déjà View outputs for one view of each of two ScanNet++ scenes (office 7b6477cb95 view 6, kitchen 5ee7c22ba0
view 13). Study (`normals.py`): DSINE normals integrated with BiNI against the Déjà View depth, judged by the
normals of the result (patch grid, edges) and their angle to DSINE's. ScanNet++'s laser scans need a token we lack;
ground truth comes from ETH3D office and kicker (`eth3d.py`, data CC BY-NC-SA, never committed): `evaluate.py`
scores three views per scene by accuracy against the laser scan; `multiview.py` scores the fused cloud of all 20
views (accuracy on 200k fixed pixels, thickness and normal angle on a 5 mm voxel grid).

## Models (frozen; we only optimize their outputs)
- Déjà View: one shared block (frame then global attention) looped K times; default 16, collapses beyond ~30.
  Outputs ray map, depth and confidence in the first view's frame, up to scale. 117M parameters.
  Its ray map deviates from a pinhole in patch-sized steps (0.6-0.9 px median): every normal map lifted through it
  shows a 14 px grid that its depth does not have. Use `rays: pinhole` (same camera, fitted K).
- DA3: depth plus a ray map at H/2×W/2; cameras are decoded from the rays.
- VGGT: cameras, depth and point maps with confidences; first view's frame, up to scale.
- SAM 3: instance masks with identities that persist across frames. Plain SAM masks are per view only.
- DSINE: per-view surface normals from one image and intrinsics.

## Commands
- `uv sync --all-groups` · `uv add <pkg>` · `uv run <cmd>`. Never pip, never bare `python`.
- `uv run python -m semantic_3d_refinement.<entry> --cfg configs/<name>.yaml`
- `uv run pytest -q` · `uv run ruff check --fix && uv run ruff format`

## Code
- Minimal and dense, from first principles. Fewer files, fewer abstractions; delete before adding.
- Design by contract: assert shapes, dtypes, ranges and finiteness on entry and before return.
- Type hints everywhere, enforced at runtime by `beartype_this_package()` in `__init__.py`.
- Module docstrings state the idea, the equation and the paper it comes from; function docstrings state the
  contract in one line. No other comments except ones that structure code; names carry meaning.
- No dead code, no commented-out code, no TODOs, no debug prints. Ruff `ALL` stays clean; every `noqa` names
  its rule and is justified by the line it sits on.
- Every mathematical claim has a test: identity, a known solution, an invariance. Tests run on CPU in seconds.
- Hyperparameters live in configs. Every run saves its config and git hash; a dirty tree is marked as such.
- Backbone outputs and masks are cached on disk; refinement never reruns a backbone.
- Third-party code is never copied in: it lives under `data/third_party/` (git-ignored), its commit is pinned in
  the config, and its licence is respected (BiNI is GPL-3, DSINE non-commercial). Our own reimplementations say
  what they follow and where they differ.
- No secrets in the tree: tokens live in `.env`, which is git-ignored.

## Git
History is part of the portfolio: it should read like a lab notebook kept by a careful engineer.
- One logical change per commit, and every commit passes `ruff` and `pytest`, so the history stays bisectable.
  Code, the configs it needs and its tests land together; figures land in the commit that reports them.
- Subject: imperative, ≤ 72 characters, no trailing period, and says what changes ("Fit the metric scale on all
  cameras"), never "fix", "update" or "wip". Body, wrapped at 72: why, and what was measured.
- Fold fixups into the commit they fix before pushing (`git commit --fixup` + autosquash). Never rewrite
  pushed history, and never commit to someone else's branch.
- Work on topic branches; merge to `main` through a PR that states motivation, change, result (table or figure)
  and how it was tested.
- `runs/` and `data/` never enter git. A figure directory superseded by a new run is deleted in the same commit.

## Experiments
Climb in order; when something fails, step back down instead of tuning blindly.
1. Look at inputs and outputs, and check the initial energy is sane.
2. Identity: zero iterations or zero weights return the input bit for bit.
3. Synthetic scene with known ground truth: error goes to ≈ 0 when the prior matches the truth.
4. One real scene, a few, then the benchmark. Semantic and non-semantic refinement get the same budget.
5. Shuffle or corrupt the semantics: if the gain doesn't drop, semantics aren't what helps.
6. Only then tune weights, coarse grid first.

Smallest scene, resolution and view count that could work. One change per run; time every stage.
Fixed scenes, metrics, alignment and seeds; never tune on test scenes, and say so when a value was.

## Visuals
Robin: show visuals, tables and graphs every week; time belongs to ideation and direction, not implementation.
- Everything that can be visualized is visualized, as PNG.
  `uv run python -m semantic_3d_refinement.visualize --run runs/<run>` writes `figures/<run>/` with a README table.
- Before and after side by side, on shared scales, with units and a caption that says what to look at.
- Each figure README ends with findings that stand on their own, without this conversation.
- Commit and push `figures/`; `runs/` and `data/` stay local.
- Look at every figure before claiming anything from it.

## README
The front page a screener reads in under a minute. Top to bottom: one sentence on the problem, the headline
figure, a results table against the baselines, reproduction in at most three commands, then design decisions and
limitations. It is updated in the same commit as any result it reports.

## Ask first
Changing the evaluation, jobs over 30 min or on more than one GPU, deleting caches or data, pushing, rewriting
history, touching another person's branch.
