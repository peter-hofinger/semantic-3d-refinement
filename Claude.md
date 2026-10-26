# CLAUDE.md

Training-free refinement of frozen feed-forward 3D reconstructions, guided by multi-view panoptic segments.
Goal: fused surfaces that are more accurate and sharper than the raw prediction, and than the same refinement
without semantics, at a small multiple of the forward pass. Supervisors: Umut Kocasarı, Robin Borth (Niessner Lab).

## Method
Per view i: depth D_i, confidence C_i, ray map (origin o, direction d); points x = o + D·d. Cameras stay fixed.
Refine inverse depth per segment s: d̂_i(p) = a_{i,s}·d⁰_i(p) + b_{i,s} + δ_i(p), minimizing with L-BFGS

  E = E_d + λ_mv·E_mv + λ_bd·E_bd + λ_pr·E_pr + λ_δ·‖δ‖²

- E_d: Huber anchor to the prediction, weighted by confidence × fraction of views that agree on the segment.
- E_mv: cross-view depth agreement, only where the projection lands in the same segment.
- E_bd: smoothing within segments only, so boundaries stay sharp.
- E_pr: planarity for structural classes (wall, floor, table).

Status: monocular PoC in `refine.py`; δ only, no segments, no a or b.

## Models (frozen; we only optimize their outputs)
- Déjà View: one shared block (frame then global attention) looped K times; default 16, collapses beyond ~30.
  Outputs ray map, depth and confidence in the first view's frame, up to scale. 117M parameters.
- DA3: depth plus a ray map at H/2×W/2; cameras are decoded from the rays.
- VGGT: cameras, depth and point maps with confidences; first view's frame, up to scale.
- SAM 3: instance masks with identities that persist across frames. Plain SAM masks are per view only.

## Commands
- `uv sync` · `uv add <pkg>` · `uv run <cmd>`. Never pip, never bare `python`.
- `uv run python -m semantic_3d_refinement.<entry> --cfg configs/<name>.yaml`
- `uv run pytest -q` · `uv run ruff check --fix && uv run ruff format`

## Code
- Minimal and dense, from first principles. Fewer files, fewer abstractions; delete before adding.
- Design by contract: assert shapes, dtypes, ranges and finiteness on entry and before return.
- Type hints everywhere, enforced at runtime by `beartype_this_package()` in `__init__.py`.
- No comments except ones that structure code; names carry meaning.
- Hyperparameters live in configs. Every run saves its config and git hash.
- Backbone outputs and masks are cached on disk; refinement never reruns a backbone.

## Experiments
Climb in order; when something fails, step back down instead of tuning blindly.
1. Look at inputs and outputs, and check the initial energy is sane.
2. Identity: zero iterations or zero weights return the input bit for bit.
3. Synthetic scene with known ground truth: error goes to ≈ 0 when the prior matches the truth.
4. One real scene, a few, then the benchmark. Semantic and non-semantic refinement get the same budget.
5. Shuffle or corrupt the semantics: if the gain doesn't drop, semantics aren't what helps.
6. Only then tune weights, coarse grid first.

Smallest scene, resolution and view count that could work. One change per run; time every stage.
Fixed scenes, metrics, alignment and seeds; never tune on test scenes.

## Visuals
Robin: show visuals, tables and graphs every week; time belongs to ideation and direction, not implementation.
- Everything that can be visualized is visualized, as PNG.
  `uv run python -m semantic_3d_refinement.visualize --run runs/<run>` writes `figures/<run>/` with a README table.
- Before and after side by side, on shared scales.
- Commit and push `figures/`; `runs/` and `data/` stay local.
- Look at every figure before claiming anything from it.

## Ask first
Changing the evaluation, jobs over 30 min or on more than one GPU, deleting caches or data.
