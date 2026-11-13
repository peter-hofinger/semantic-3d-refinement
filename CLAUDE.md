# CLAUDE.md

Semantic-Guided Refinement for Feed-Forward 3D Reconstruction (TUM, supervisors Umut Kocasarı and Robin Borth, Niessner Lab). Team, in this order and spelled exactly so: Pouya Shekarchizadeh Esfahani, Peter Hofinger, Mircea-Alexandru Răuțoiu. Everyone works on every part of the repo.

## Bar
This repo is a public portfolio piece for research-engineering roles; screeners skim the README, open two files and scroll the history.
- Substance over signal: a measured claim with its baseline beats any badge, buzzword or abstraction.
- Every number names its baseline, scenes and how it was measured. Report negative results as plainly as positive ones; an inflated gain is fatal.
- The authors must be able to defend any line in an interview: no code that works by accident, no trick without a reason; non-obvious choices are explained in the docstring or the commit body.
- Credit precisely: teammates, the papers an idea comes from, third-party code and its licence.

## Idea
A frozen feed-forward model (Déjà View; SegVGGT is wired in too) predicts per-view depth, confidence and rays. We refine its output, guided by multi-view panoptic segments (SAM 3 masks linked across views), towards fused surfaces that are more accurate than the raw prediction and than the same refinement without semantics, at a small multiple of the forward pass. The method is not settled. The default refiner optimizes inverse depth per pixel of all views jointly (Gauss-Newton + CG) under a data anchor, cross-view agreement within one segment and a normal prior; learned refiners are explored next to it. `src/refinement/multiview.py`'s docstring is the reference for the energy, `configs/refine.yaml` holds its settings and `experiments/LOG.md` says why each choice was made.

Pipeline: Views -> backbone -> SAM 3 + linking -> refiner -> labeled point cloud, one point per input pixel (`run-pipeline`).

## Pitfalls
- Déjà View's ray map deviates from a pinhole in patch-sized steps; normals lifted through it show a 14 px grid. Use `rays: pinhole`.
- Déjà View assumes a centred principal point: evaluate on Hypersim's centred-camera scenes (stress is the tilt-shift exception).
- Hypersim's walls, floors and ceilings have no instance ids; each such class becomes one instance.

## Evaluation
- `src/analysis/scene_eval.py` on Hypersim (`configs/hypersim.yaml`): dev decides, val checks once, test is only reported, stress is a second look. Never tune on val or test, and say so when a value was.
- Each output is aligned onto the GT by one Umeyama Sim(3) on its pixels, then scored by `src/evaluation/labeled.py`.

## Commands
- `uv sync --all-groups` · `uv add <pkg>` · `uv run <cmd>`; never pip or bare `python`.
- `uv run python -m src.analysis.scene_eval --cfg configs/eval_hypersim.yaml --splits dev`
- `uv run run-pipeline <views> --geometry <dejaview npz> --segmenter sam3 --refiner joint`
- `uv run pytest -q` · `uv run ruff check --fix && uv run ruff format`

## Code
- Minimal and dense, from first principles: fewer files and abstractions; delete before adding.
- Design by contract: assert shapes, dtypes, ranges and finiteness on entry and before return.
- Type hints everywhere, enforced at runtime by `beartype_this_package` (`src/__init__.py`): no `TYPE_CHECKING` imports, interfaces are `@runtime_checkable` protocols, hot helpers live at module level.
- Module docstrings state the idea, the equation and the paper; function docstrings the contract in one line. No other comments except structuring ones.
- No dead or commented-out code, TODOs or debug prints. Ruff `ALL` stays clean; every `noqa` names its rule and is justified where it sits.
- Every mathematical claim has a test (identity, known solution, invariance), on CPU in seconds.
- Hyperparameters live in configs; every run saves its config and git hash, marked dirty if the tree is.
- Backbone outputs and masks are cached on disk; refinement never reruns a backbone.
- Third-party code is never copied in: it lives in `data/third_party/` (git-ignored) at a pinned commit, licence respected (BiNI GPL-3, DSINE non-commercial). Secrets live in `.env`.

## Git
- One logical change per commit; every commit passes ruff and pytest. Code, configs and tests land together; figures with the commit that reports them.
- Subject imperative, ≤ 72 characters, no period, says what changes (never "fix", "update", "wip"); body wrapped at 72: why, and what was measured.
- Fold fixups before pushing; never rewrite pushed history or commit to someone else's branch.
- Topic branches, merged through a PR that states motivation, change, result and how it was tested.
- `runs/` and `data/` never enter git.

## Experiments
Climb in order and step back down when something fails: look at inputs and outputs; identity (zero iterations or weights return the input bit for bit); a synthetic scene with known truth; one real scene, then the benchmark; corrupt the semantics (if the gain doesn't drop, semantics aren't what helps); only then tune, coarse grid first.
- An experiment is E<issue><letter> (E8a, E8b, ...); work without an issue gets one first. E1-E45 keep their numbers.
- Each gets a plan entry in `experiments/LOG.md` before it runs and a result entry after: goal, setting, commits, commands, checks, results with baselines, conclusion.
- Smallest scene, resolution and view count that could work; one change per run; time every stage.

## Visuals
- Visualize everything, as image files, and look at every figure before claiming anything from it.
- Presentation visuals use only the bedroom `ai_006_007`, views 18, 0, 12, 6 (issue #7); debug visuals may use any scene.
- Before and after side by side, on shared scales, with units and a caption that says what to look at.

## README
The front page: problem in one sentence, headline figure, results table against the baselines, reproduction in at most three commands, design decisions and limitations. Update it in the same commit as any result it reports.

## Ask first
Changing the evaluation, jobs over 30 min or on more than one GPU, deleting caches or data, pushing, rewriting history, touching another person's branch.
