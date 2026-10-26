# CLAUDE.md

Semantic-guided refinement of feed-forward 3D reconstructions: predict geometry from images
with a frozen model, then refine it using semantic cues so that surfaces are less noisy, views
agree more, and boundaries and details are preserved while staying faithful to the images.
Supervisors: Umut Kocasarı, Robin Borth (Niessner Lab). Ask before changing eval.

## Milestones
1. Run a feed-forward baseline (VGGT / DejaView) and measure surface quality.
2. Extract semantics (SAM masks, categories, features) and lift them onto the 3D geometry consistently across views.
3. Semantic-guided post-optimization: region-aware smoothing, boundary preservation, category-dependent geometric priors.
4. Evaluate accuracy, surface quality, completeness and extra compute against the same refinement *without* semantics.

## Models
- **VGGT**: DINO patch tokens plus a camera token per frame, then a transformer that alternates frame-wise and
  global attention. Heads output camera parameters, depth and point maps (each with a confidence map), and tracks,
  all in one forward pass. Output is in the first camera's frame and up to scale.
- **DA3 (Depth Anything 3)**: a single plain DINO ViT. The first L_s layers attend within each view, the last L_g
  layers also across views. A Dual-DPT head outputs depth (N×H×W) and ray maps (N×H/2×W/2×6: origin and direction).
  Points = origin + depth · direction. Camera token: encoded from known cameras if available, learnable otherwise;
  camera decoded from it. Losses: depth (teacher-distilled), ray, points, camera.
- **SAM**: promptable or automatic 2D masks. They are per-view and not consistent across views; association happens in 3D.
- **DéjàView** (Burzio et al., NVIDIA): DINOv2 ViT-B tokens, plus 4 register tokens and a camera token per view, then
  *one* shared block (frame attention, then global attention) looped K times (trained on K ∈ [8, 16], default 16),
  gated by time conditioning. Heads output ray maps (6ch), depth plus confidence, and a camera MLP.
  Points = origin + depth · direction; cameras come from the rays as in DA3; first-view frame. 117M parameters,
  ~10× smaller than VGGT at similar accuracy. More K buys accuracy; beyond ~30 it collapses.
- Feed-forward models stay frozen; we only optimize their outputs.

## Commands
- `uv sync` · `uv add <pkg>` · `uv run <cmd>`. Never pip, never bare `python`.
- `uv run python -m <pkg>.<entry> --cfg configs/<name>.yaml`
- `uv run pytest -q` · `uv run ruff check --fix && uv run ruff format`

## Code
- Tinygrad-style: minimal, dense, first principles. Derive what is necessary, write only that.
  Fewer files, fewer abstractions, no speculative generality. Delete before adding.
- Design by contract: `assert` preconditions on entry, postconditions before return,
  invariants in loops. Shapes, dtypes, ranges, finiteness. Fail loud and early.
- Type hints everywhere, enforced at runtime: `from beartype.claw import beartype_this_package`
  then `beartype_this_package()` in the package `__init__.py`.
- No comments except ones that structure code. Names carry meaning.
- Hyperparameters live in configs; each run saves config + git hash.
- Cache feed-forward outputs and masks to disk; refinement experiments must not rerun the backbones.

## Visuals
Everything that can be visualized should be visualized. Supervisor (Robin): "Theoretically, you should have some
visuals, tables, graphs to show every week because we expect you to do lots of different experiments/results and
with nice visuals, you can show them. We are in the era of agents and you should spend most of the time on ideation,
creative direction finding, potential contributions to the field. The implementations dont take a lot of time
anymore, any paper can be implemented from scratch in a day maybe. So, I dont think you will ever spend most of the
time on implementation."
- Every run gets figures and a metrics table: `uv run python -m semantic_3d_refinement.visualize --run runs/<run>`
  writes `figures/<run>/` (PNGs + README.md with the table). Commit and push `figures/`; `runs/` and `data/` stay local.
- Inputs, outputs, residuals, errors vs GT, energy curves, point clouds, before/after side by side on shared scales.
- Look at every figure before drawing conclusions from it.

## Debugging ladder: in order, never skip
1. Look at the inputs and outputs: depth, points, confidences, masks, lifted labels. Check the initial loss/energy is sane.
2. Identity: with zero iterations or zero weights, refinement returns the input bit-for-bit.
3. Overfit one synthetic scene with known GT: the error must go to ≈ 0 when the prior matches the truth.
4. One real scene, then a few, then the whole benchmark. Semantic and non-semantic refinement get the same budget.
5. Shuffle or corrupt the semantics: if the gain doesn't drop, the semantics aren't what is helping.
6. Only then tune weights and step size (coarse grid first) and add complexity.

When something fails, go back down this ladder; don't tune blindly.

## Rules
- Smallest scene, resolution and view count that could work. Debug cycles in minutes.
- One change per run. Time each stage (backbone, segmentation, lifting, refinement).
- Fixed scenes, metrics, alignment (scale/pose) and seeds across runs. Never tune on the test scenes.
- Evidence before claims: numbers plus renders or meshes, side by side with the baseline.
- Ask before launching a job that runs longer than 30 min or uses more than one GPU, and before deleting caches or data.
