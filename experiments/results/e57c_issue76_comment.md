## Oracle filter for minimal Chamfer: results (E57c)

### TL;DR

- **A subset of Déjà View's own points has a much lower Chamfer than all of them.** A greedy oracle that keeps half the pixels reaches **2.048 cm on dev, −41 % against Déjà View's 3.500**. With finer rounds it reaches 1.993 cm (−43 %); that is an upper bound on the minimum, since any subset we can exhibit is feasible.
- **No closed form, as the issue suspected.** The exact problem is a facility-location / k-median-type subset selection. I derived a first-order per-point gain and a reverse greedy that approximates it; I did not prove hardness for this instance class.
- **Most of the headroom is redundancy across views, not "which points are wrong".** Ranking by each point's true error alone reaches only 2.479 cm (−29 %); adding what a point costs the GT coverage (a point is cheap to drop when other views cover the same GT) gives the rest.
- **Everything that combines the oracle with Déjà View's confidence lands at 2.50–2.68 cm (−23 to −29 %) and still needs the oracle.** Every rule that uses confidence alone fails: the confidence cut at q 0.5 scores 25.4 cm (completeness 48.5 cm) and is never below the uncut cloud at any q.

![Fifteen ways to pick a subset of Déjà View's cloud, as point clouds with their dev Chamfer](e57c_issue76_algorithms.jpg)

*Every measured approach on one scene (bedroom `ai_006_007`), grouped by what it needs; the number under each name is the dev mean over three scenes, the bar a common 0–3.5 cm scale (full bar = Déjà View whole).*

### Setting

| | |
|---|---|
| Question | Which subset of all points achieves minimal Chamfer? (issue text) |
| Data | Hypersim **dev**: `ai_010_007`, `ai_006_007`, `ai_044_001`; 20 views each at 504 px |
| Predictions | Déjà View K = 16, one point per pixel, 3.8 M points per scene (20 × 378 × 504) |
| Metric | Chamfer = mean of accuracy (point to nearest GT point) and completeness (GT point to nearest kept point), cm, `src/evaluation/labeled.py`; one Umeyama Sim(3) per cut, on the kept points; pixels with a GT point |
| GT cloud | the GT depth of the same 20 views (so completeness is measured on points other views also cover) |
| Baselines | Déjà View whole (3.500); instance voxel density at q 0.6 (#27, E45), the best predictor-only filter we have (3.188) |
| Status | every row below except density and Déjà View uses the GT; none is a method |

### The oracle

For a kept set S, Chamfer = ½ (mean a_i over S + mean c_j over the GT points j), with a_i the distance of point i to the nearest GT point and c_j the distance of GT point j to its nearest kept point. Dropping one point i changes it by

```
Δ = (a_i − ā) / (|S| − 1)  −  m_i / N_GT,      m_i = Σ (d2 − d1) over the GT points whose nearest kept point is i
```

(d1, d2: their nearest and second nearest kept point). So a point is worth dropping iff its accuracy gain exceeds the completeness it costs; two neighbours that cover each other both have m_i ≈ 0, which a one-shot rule cannot see and a **reverse greedy** (drop the lowest-gain 10 % of the pixels per round, recompute m_i on the kept set) can. An incremental top-two update keeps it exact and makes it 38 % faster (5.7 s per scene against 9.3 s; the masks are identical, asserted on every scene).

Rounds of the greedy at q 0.5, dev Chamfer: 1 round (the one-shot rule) 2.214 · 5 rounds (10 %) 2.048 · 10 rounds (5 %) 2.020 · 20 rounds (2.5 %) **1.993**: it converges near 2.0 cm. That is a heuristic; 1.993 is not a lower bound on the optimum.

![How the oracle decides, pixel by pixel: error, coverage, net change, greedy order, kept set](e57c_issue76_oracle_anatomy.jpg)

*One decision per pixel: the error of a point (how much worse than average) against the coverage only it provides. Dark = dropping helps, teal = keep. The greedy repeats this after every round.*

![Peeling the cloud: clouds at six kept fractions above the accuracy, completeness and Chamfer curves](e57c_issue76_oracle_peeling.jpg)

*Accuracy improves all the way down; completeness stays flat while the redundancy is removed and then rises; the Chamfer has its minimum at q = 0.5 (2.048 cm, −41 %).*

### Results: every variant that beats Déjà View (dev mean; cm)

Δ is against Déjà View's 3.500. Time is seconds per scene for the optimized variant, CPU (16 cores), without loading, SAM 3 and the Chamfer scoring. Per scene: `ai_010_007 / ai_006_007 / ai_044_001`.

**Oracles (rank by the GT alone)**

| Variant | q / kept | Accuracy / completeness / **Chamfer** | Δ | per scene | s |
|---|---|---|---|---|---|
| **Greedy**, 10 % rounds | 0.5 | 1.533 / 2.562 / **2.048** | −41.5 % | 0.88 / 1.14 / 4.13 | 7.6 |
| Marginal, one round | 0.6 | 1.720 / 2.683 / 2.202 | −37.1 % | 0.94 / 1.28 / 4.39 | 2.9 |
| Accuracy (true distance to the GT) | 0.7 | 1.983 / 2.974 / 2.479 | −29.2 % | – | 1.9† |
| Same-pixel GT error | 0.8 | 2.738 / 2.688 / 2.713 | −22.5 % | – | – |
| Accuracy averaged over 7 × 7 pixels | 0.8 | 2.451 / 3.087 / 2.769 | −20.9 % | – | – |

**Oracle + Déjà View's confidence (c = confidence percentile)**

| Variant | q / kept | Accuracy / completeness / **Chamfer** | Δ | per scene | s |
|---|---|---|---|---|---|
| Product of two gradients, best q | 0.3 / 0.71 | 2.632 / 2.364 / 2.498 | −28.6 % | 1.09 / 1.40 / 5.00 | 9.4 |
| Union (confidence ≥ 0.5 **or** greedy keeps) | – / 0.70 | 2.446 / 2.602 / 2.524 | −27.9 % | 1.08 / 1.40 / 5.09 | 7.6 |
| **Fusion A**, seeds, no instance information | – / 0.63 | 2.497 / 2.744 / 2.621 | −25.1 % | 1.16 / 1.54 / 5.16 | 12.7 |
| Mean of two gradients, best q | 0.2 / 0.67 | 2.596 / 2.655 / 2.626 | −25.0 % | 1.19 / 1.54 / 5.15 | 9.4 |
| Fusion A, seeds, SAM 3 instance ids | – / 0.62 | 2.507 / 2.796 / 2.651 | −24.3 % | 1.17 / 1.58 / 5.20 | 12.5 |
| Fusion, no seeds, one instance | – / 0.61 | 2.505 / 2.832 / 2.668 | −23.8 % | 1.21 / 1.60 / 5.20 | 9.8 |
| Fusion, no seeds, SAM 3 instance ids | – / 0.60 | 2.513 / 2.848 / 2.680 | −23.4 % | 1.21 / 1.61 / 5.22 | 9.8 |

**Reference without the GT**

| Variant | q | Accuracy / completeness / Chamfer | Δ | per scene | s |
|---|---|---|---|---|---|
| Instance voxel density (#27) | 0.6 | 3.550 / 2.827 / 3.188 | −8.9 % | 1.48 / 1.73 / 6.36 | 3.7 |
| Déjà View | 1.0 | 4.675 / 2.324 / 3.500 | – | 1.60 / 1.85 / 7.05 | – |

† derived: Umeyama 0.54 s + accuracy 1.36 s. The order oracle < confidence fusions < density < Déjà View holds in every one of the three scenes. Every q in the tables is the dev-best of a sweep (greedy, marginal, accuracy, same-pixel, smoothed, density; product and mean over q = 0 … 1), so those rows are optimistic; the union and the fusions have no q of their own (confidence q_c 0.5, oracle q_g 0.5).

<details>
<summary>How the confidence variants are built</summary>

- **Union:** keep a pixel if its confidence percentile is ≥ 0.5 or the greedy oracle (q 0.5) keeps it.
- **Fusion (no seeds):** confidence cut q_c 0.5 ("white"); oracle-kept pixels farther than 8 cm from every white pixel are added ("green"); oracle-kept pixels nearer than 8 cm ("blue") come back when they lie on a straight run (row, column or diagonal of one view) of blue pixels between a white and a green pixel.
- **Fusion A:** the same, with white points of the other views, reprojected into a view (one pixel per point, no radius), counting as white where they land on a pixel dropped by both rankings. "No instance information": the whole scene is one instance; with SAM 3 instance ids the reprojection and the runs must stay inside one instance.
- **Product / mean of two gradients:** g_c = min(c / 0.5, 1); g_o = clip(d / 10 cm, 0, 1) for oracle-kept pixels, 0 for the others (d: distance to the nearest confidence-kept point). Pixels with c ≥ 0.5 or g_o = 1 are decided and kept; the rest are ranked by g_c g_o (or by (g_c + g_o) / 2) and the top fraction q is kept.

</details>

### Fusion A, step by step

![Fusion A in four steps: Déjà View, confidence threshold, plus the oracle's pixels, plus the bridge, with a zoom on the seam](e57c_issue76_fusion_steps.jpg)

![Why the seam closes: view 12 before, with the seeds, and after the bridge](e57c_issue76_fusion_mechanism.jpg)

*Top: the cloud after each step, with a zoom on the wall right of the door; the hairline of step 3 is the seam the 8 cm gate leaves. Bottom: the same in view 12's image space: a kept point of another view, reprojected onto a dropped pixel, is a seed that lets a straight run bridge the seam.*

### Findings

1. **The oracle alone beats every confidence combination.** On dev it is 0.45–0.63 cm below them (2.048 against 2.498–2.680), at a smaller kept share (0.50 against 0.60–0.71); at equal share on the bedroom it is still 0.12 cm below the union (1.276 against 1.400 at 0.695). The confidence cut costs coverage.
2. **The completeness term is worth 0.3–0.4 cm.** Greedy 2.048 and the one-round rule 2.202 against the accuracy-only oracle 2.479: dropping a point is cheap when another view covers its GT.
3. **The oracle is not blockwise per (view, SAM 3 instance).** On the bedroom only 15.5 % of the pixels lie in a block that it keeps < 10 % or > 90 % of, against 37.9 % for the confidence cut: a per-mask or per-view decision cannot express it.
4. **Confidence is a decent accuracy ranker and a poor selector.** At q 0.5 the kept accuracy is 2.220 (oracle 1.521, density 3.001), but it drops whole surfaces (completeness 48.5 cm); a plausible cause, not tested here, is that the low confidences sit on the same surface in every view. Using κ/conf as the accuracy term inside the greedy with the true completeness cost gives 2.70–2.88 cm: 43–55 % of the greedy's gain.
5. **Déjà View's confidence loss cannot reach this ranking.** Its optimum conf* = α / (γ E[error]) is monotone in the expected error of the pixel's own prediction, a pointwise quantity; the completeness term depends on other views' points. Read from the installed `dvlt` package, not from the released checkpoint's config.
6. **The 8 cm gate creates a seam, and reprojected white closes most of it.** Between a white and a green patch the blue pixels form a band about as wide as the gate (the strip's source pixels lie a median 4.5 cm from the nearest white and 6.6 cm from the nearest green). On the bedroom, the squiggle's gap pixels fall 1 150 (no seeds) → 317 (seeds, ids) → 202 (seeds, no instance information); dev Chamfer 2.680 → 2.651 → 2.621. Dropping the instance ids alone moves it by 0.012 cm: **the seeds do the work, the semantics were not needed.**
7. **Chamfer is blind to hairlines.** A 1–2 px sliver between two kept patches costs no completeness (the GT points there have a kept neighbour within a splat radius) but shows in a render; visual checks stay necessary.
8. **Camera error is not where the Chamfer is lost.** Déjà View's depth lifted through the GT cameras: uncut 3.576 against 3.500, the oracle's best 2.041 against 2.048.

### What did not work

| Variant | dev Chamfer | Note |
|---|---|---|
| Confidence cut alone, q 0.5 | 25.376 | never below the uncut cloud at any q (q 0.9: 4.289); completeness 48.5 |
| Confidence per view / per SAM 3 instance, q 0.5 | 23.695 / 5.354 | per instance loses least, still above uncut |
| Membrane verifier (SAM 3 masks, inverse depth) | 4.49 – 14.0 | fills bands, not large regions; always above uncut |
| Confidence ranked inside each voxel | 3.33 best | completeness kept, accuracy not gained; ties density |
| GT camera rays + Déjà View depth, then the oracle | 2.041 | no gain over 2.048 (accuracy −9 %, completeness +0.25 cm) |
| Product of the *ranks* of confidence and oracle (bedroom) | 1.911 at 0.70 | against 1.400 for the union: the two rankings are nearly independent among the dropped pixels |

### Visual check (bedroom `ai_006_007`, views 18, 0, 12, 6 and the dollhouse; my reading of one scene)

Confidence only is clean where it keeps points but leaves large holes in the floor and walls. The greedy oracle alone is clean with some bed-frame debris. The fusion with seeds is the most complete and clean-looking cloud among the real fusions (the wall right of the door is continuous; the wardrobe window and the bed cover still have holes); the union and the gradient cuts look the same. Figures: `experiments/figures/e57c_summary_clouds_*.jpg`, `e57c_summary_views_one_instance.jpg`, `e57c_clouds*.jpg`.

### Limitations

- Dev only (3 scenes), one run, no val for the new oracles (the accuracy oracle was run on val: best 3.270 at q 0.8 against Déjà View 5.102), no seeds. Every q in the tables was chosen on dev (best of a sweep).
- Every number except Déjà View and density uses the GT; the fusions are ceilings of "confidence first, oracle's pixels back", not methods.
- The GT cloud is the depth of the same 20 views, so the completeness term, and with it the greedy's edge, may be smaller against a mesh of the whole scene.
- The greedy is a heuristic; the facility-location and sparsification statements (Ilg et al. 2018 for oracle sparsification and AUSE; DeepC-MVS, Kuhn et al. 2020 for learned MVS confidence filters) come from a web search, not full reads.
- Runtimes: one run per stage on a shared machine; density's 3.7 s is the reference implementation. A learned head would replace the oracle's 7.6 s by its forward pass (not measured).

### Implications for #57 and next steps

1. **Train the head on the oracle's keep, not on a thresholded error or a distance.** Candidate labels: the one-round marginal keep (`a_i − ā > m_i`, no q to choose, own cut ≈ q 0.68, dev 2.238) or the greedy order. Pouya's E57a head (error label) does not beat density at equal q; the true-distance oracle at q 0.5 ties it on val (3.955 against 3.817) and beats it on dev; the learned distance head went through rungs 0–4 only (rung 5, the val comparison, not run).
2. **The input needs cross-view context** (what else covers this surface, in which view it is most accurate), which a per-pixel confidence cannot give: the volumetric head's voxel statistics, with the confidence as one input feature.
3. **Validate on val at equal q against density**, with the ladder of `docs/experiments.md` (overfit the bedroom, then dev, then train → val, three seeds before any "fails").
4. If a label is learned well, fusion A (reprojected seeds, no semantics) is a cheap way to close the seams of the cut.

### Reproduction

Code, tests, results and figures are in the pull request from branch `peter/e57c-distance-head` (base `795a1df`); `uv run pytest -q` covers the cost of a point, the greedy and its incremental version, the straight-run rule and the seeds.

```bash
uv run python -m experiments.scripts.e57c_summary --splits dev            # every row above, with the stage times
uv run python experiments/scripts/e57c_oracle_curve.py --splits dev --per-scene   # the oracles' q sweeps
```

Results: `experiments/results/e57c_summary_dev.txt`, `e57c_oracle_zoo_dev.txt`, `e57c_greedy_steps_dev.txt`, `e57c_hybrid_dev.txt`, `e57c_verify_dev.txt`, `e57c_voxel_conf_dev.txt`, `e57c_gt_rays_dev.txt`. The plan, the pre-registered expectations and the result of each step are in `experiments/LOG.md` (E57c entries of 17–18 November).

Credit: Déjà View is the frozen backbone; the instance voxel density filter is E45 (#27); the E57a keep head is Pouya's (#57).
