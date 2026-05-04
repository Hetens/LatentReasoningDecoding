# Extended Experiments: Activation Patching, Latent Geometry, and Candidate-Set Analysis

## Table of Contents

1. [Activation Patching: What It Tells Us](#1-activation-patching-what-it-tells-us)
2. [Candidate-Set Size and the Backtracking F1 Paradox](#2-candidate-set-size-and-the-backtracking-f1-paradox)
3. [Latent Geometry: PCA, UMAP, and the Corner Plot](#3-latent-geometry-pca-umap-and-the-corner-plot)
4. [The CKA Heatmap (Corrected)](#4-the-cka-heatmap-corrected)
5. [Synthesizing the Full Picture](#5-synthesizing-the-full-picture)
6. [Suggested Next Steps](#6-suggested-next-steps)

---

## 1. Activation Patching: What It Tells Us

### 1.1 Background

Activation patching is a **causal** intervention. While probing asks "is
information X *present* in z?", patching asks "does the model *use*
information X from z to produce its output?" This is the difference between
correlation and causation.

Two interventions were run at two recursion indices:


|                           | (T=1, i=4) — best probe F1 | (T=2, i=5) — weakest probe F1 |
| ------------------------- | -------------------------- | ----------------------------- |
| **Cross-puzzle swap**     | +0.106 [-0.259, +0.465]    | **+0.602 [+0.215, +1.018]**   |
| **Within-puzzle shuffle** | +0.207 [-0.125, +0.555]    | **-0.914 [-1.291, -0.567]**   |


Delta CE > 0 means the intervention **hurts** the model (higher cross-entropy =
worse predictions). Delta CE < 0 means the intervention **helps** the model.

### 1.2 Interpretation

**At (T=1, i=4)** — the step where the probe performs best:

Both CIs straddle zero. This means swapping in a completely different puzzle's
z_L, or shuffling cell positions within the same puzzle's z_L, does **not
significantly change** the model's output at this early step. This is actually
expected: (T=1, i=4) is only the 4th inner step of the *first* outer cycle.
The model still has two more full outer cycles (T=2 and T=3) to refine and
overwrite this information before producing logits. Any corruption at (1,4)
gets corrected by subsequent recursion.

**At (T=2, i=5)** — the step where the probe performs weakest:

Two significant results:

1. **Cross-puzzle swap: +0.602 [+0.215, +1.018]** — Swapping z_L from a
  different puzzle significantly increases cross-entropy. The model
   **causally depends** on puzzle-specific information encoded at this step.
   Because (2,5) is late in the second outer cycle, there is only one more
   outer cycle (T=3) to recover from the corruption, and the model cannot
   fully compensate.
2. **Within-puzzle shuffle: -0.914 [-1.291, -0.567]** — Shuffling cell
  positions within z_L significantly **decreases** cross-entropy (improves
   output). This is the surprising result. It suggests that by step (2,5),
   the model has developed positional structure in z_L that is *misaligned*
   with the output head's expectations for these specific cells. Shuffling
   disrupts this misalignment and, in expectation, produces a distribution
   closer to what downstream processing handles well.
   An alternative reading: the z_L vectors at (2,5) encode useful information
   in their *values* (hence the cross-puzzle swap hurts), but the specific
   cell-to-position mapping has become noisy or entangled through two cycles
   of recursive processing. The model may rely more on aggregate statistics
   of z_L across cells rather than precise per-cell assignments at this stage.

### 1.3 Is This Useful for the Paper?

**Yes**, but frame it carefully. The key takeaways for the paper are:

- **Causal necessity grows with recursion depth.** Early steps (T=1) can be
corrupted without consequence because the model self-corrects. Later steps
(T=2 onward) are causally necessary — the model cannot recover. This shows
the recursion is not redundant; later cycles carry information the model
needs.
- **The model uses z_L's content more than its positional structure at
deeper steps.** The cross-puzzle swap (corrupting content) hurts, while the
shuffle (disrupting position) helps or is neutral. This suggests the model
has learned to encode cell-independent features (e.g., aggregate constraint
satisfaction patterns) rather than strict per-cell mappings at deeper
recursion stages.
- **There is an asymmetry between "what is encoded" and "what is used."**
(T=1, i=4) has the *highest* probe F1 but zero causal effect. (T=2, i=5)
has the *weakest* probe F1 but strong causal effect. This is a common and
important finding in mechanistic interpretability: representational richness
(probing) and causal importance (patching) are not the same thing.

---

## 2. Candidate-Set Size and the Backtracking F1 Paradox

### 2.1 The Numbers

From the |S_c| analysis:


| Statistic                 | Easy (b=0) | Hard (b=1)         |
| ------------------------- | ---------- | ------------------ |
| Number of puzzles         | 1,806      | 3,194              |
| Mean                      | S_c        | (all cells)        |
| Median                    | S_c        | (all cells)        |
| Fraction of given cells ( | S_c        | =1)                |
| Mean                      | S_c        | (blank cells only) |


### 2.2 Why Hard Puzzles Have Higher Probe F1

The `f1_by_backtracking.png` plot shows MLP F1 ~0.83-0.84 for hard puzzles
vs. ~0.75 for easy puzzles. The |S_c| analysis explains this completely:

**Easy puzzles have 65% of cells already given** (|S_c|=1 — only one candidate,
the answer). For given cells, predicting the candidate set is trivial (it's just
the digit itself), but these cells contribute few true positives per cell to the
F1 numerator (only 1 TP per cell vs. potentially 3-5 for blank cells). Meanwhile,
easy puzzles have very few blank cells, and those blank cells have relatively
small candidate sets (mean 3.43).

**Hard puzzles have only 33% given cells** — most cells are blank with larger
candidate sets (mean 3.93). Each blank cell contributes more true positives when
the probe gets it right. The micro-averaged F1 metric sums TP/FP/FN across all
cells and digits, so puzzles with more blank cells and larger candidate sets
provide more "signal" per cell for the probe to exploit.

**This is a metric artifact, not a deeper finding.** The model is not "better"
at encoding hard-puzzle constraints. The label distribution simply favours
higher F1 when there are more positive labels per cell. For a fair comparison,
you would need to compute F1 **stratified by |S_c|** — comparing easy vs. hard
puzzles only among cells with the same candidate-set size.

### 2.3 Paper Implication

Include the |S_c| analysis as a confound explanation. This is methodologically
important: it shows you understand that the difficulty split is confounded by
label frequency, and that the F1 difference does not reflect a difference in
representational quality. The `candidate_size_analysis.png` plot is a clean
visual to include.

---

## 3. Latent Geometry: PCA, UMAP, and the Corner Plot

### 3.1 What the PCA 2D Plot Shows

The PCA 2D scatter (`pca_2d.png`) reveals a striking spatial structure:

- **T=3 steps (cyan/teal) form tight, separated clusters** in the bottom-right
and right portions of the plot. These are clearly distinct from earlier steps.
- **T=1 and T=2 steps are intermixed** in the upper-left cloud, with high
overlap between different inner steps.

This matches the CKA heatmap perfectly: T=1 and T=2 have moderate CKA between
them (~~0.5-0.6), so their PCA projections overlap. T=3 has low CKA with T=1
(~~0.3), so it projects to a completely different region.

**Key insight:** The third outer cycle (T=3) induces a *qualitative phase
transition* in the representation. The model's latent state after two outer
cycles of refinement occupies a fundamentally different region of representation
space. This is the model's "final reasoning mode."

### 3.2 What the UMAP 2D Plot Shows

The UMAP (`umap_2d.png`) is dominated by T=3 clusters in the center, with all
other steps forming a diffuse ring. This happens because UMAP preserves local
neighbourhood structure: the T=3 vectors are internally coherent (tight
clusters) while T=1/T=2 vectors are more spread out and interleaved with each
other.

The small isolated clusters visible in the upper-left of the UMAP are likely
cells with extreme or distinctive candidate sets (e.g., cells with very large
or very small |S_c|) at T=3 steps.

### 3.3 The Corner Plot

The corner plot (`pca_corner.png`) shows pair-wise relationships between the
top 5 PCA components. The most informative panels are:

- **PC1 vs PC3**: Shows the clearest separation of T=3 clusters from the
T=1/T=2 cloud.
- **PC1 histogram**: Bimodal distribution — T=3 steps occupy a separate peak
around PC1 = +5, while T=1/T=2 centre around PC1 = -8.
- **PC4 and PC5**: These components capture within-cluster variance and show
less obvious structure, suggesting the outer-cycle distinction is captured
in the first 3 components.

### 3.4 Explained Variance

From `pca_explained_variance.png`:


| PCs  | Cumulative variance |
| ---- | ------------------- |
| 1-2  | 20.0%               |
| 1-3  | 27.3%               |
| 1-5  | 38.4%               |
| 1-10 | 60.4%               |


10 components capture only 60% of the variance, meaning the latent space is
genuinely high-dimensional. The information is distributed across many
directions, not concentrated in a low-dimensional subspace. This is consistent
with the probe results: a linear probe works (the information is linearly
accessible), but the representation is spread across many dimensions rather
than being cleanly low-rank.

### 3.5 Why the Plots Look Cluttered

Two reasons:

1. **18 groups with 5,000 points each** makes for very dense scatter plots.
  The inner steps within each outer cycle are hard to distinguish because they
   overlap heavily (as CKA confirms: within-cycle CKA is high).
2. **The colour palette**: 18 colours on a scatter plot with semi-transparent
  points makes individual groups hard to track.

**Recommended improvements:**

- **Group by outer cycle only** (T=1 vs T=2 vs T=3) using just 3 bold colours.
This is the dominant structure the data shows, and 3-colour plots are far
more readable.
- **Faceted plots**: One PCA panel per outer cycle T, showing only the 6
inner steps within that cycle. This reveals within-cycle structure that is
currently invisible.
- **Subsample more aggressively**: 200 puzzles × 5 cells = 1,000 points per
group gives 18,000 total, which is still dense but more readable.
- **Density contours** instead of scatter: Replace point clouds with 2D KDE
contour lines, one per group. Much cleaner for publication.

### 3.6 How to Proceed Toward Human-Readable Puzzle States

The current visualizations show you *where* different recursion steps live in
latent space, but not *what* the model is representing. To bridge that gap,
there are several productive directions:

**A. Colour by puzzle property instead of (T,i).**
Re-use the same PCA/UMAP coordinates but colour points by:

- |S_c| (candidate-set size) — reveals whether the geometry encodes constraint
information
- Cell position (row/column/box) — reveals whether spatial structure is encoded
- Whether the cell's answer is correct — reveals where errors cluster in
latent space

This is the most direct path to "human-readable" interpretation: if cells with
|S_c|=2 cluster separately from |S_c|=5, you know the geometry encodes
candidate-set size, and you can read that from the plot.

**B. Compute PCA on *changes* between steps (delta-z).**
Instead of visualizing z at each step, visualize `z[T, i+1] - z[T, i]` (the
update per inner step) or `z[T+1, 1] - z[T, L]` (the update per outer cycle).
This shows *what changes* at each recursion step, not just where things are.

**C. Targeted per-cell decoding.**
For a small set of puzzles (e.g., 10), extract the z vector for each cell,
project into PCA space, and annotate each point with its puzzle position, given
digit (if any), and candidate set. This creates an interpretable "map" of one
puzzle in latent space.

---

## 4. The CKA Heatmap (Corrected)

The corrected CKA heatmap (`cka_heatmap.png`, now with proper (1,1)-(3,6)
labels) is one of the strongest results in your experiment set.

### 4.1 What It Shows

The 18x18 heatmap has clear **block-diagonal structure with three 6x6 blocks**:

- **Within each outer cycle** (the bright yellow/orange diagonal blocks):
CKA ~0.7-0.9. The 6 inner steps within one outer cycle maintain similar
representational geometry. The model refines *values* within a
consistent geometric framework during each inner loop.
- **Between T=1 and T=2**: CKA drops to ~0.4-0.6 (orange-red).
The second outer cycle reshapes the geometry moderately.
- **Between T=1 and T=3** (and T=2 and T=3): CKA drops further to ~0.2-0.4
(dark red-purple). The third outer cycle creates a substantially
different representation space.
- **Within T=2**: An interesting sub-pattern — pairs like (2,1)-(2,2) and
(2,3)-(2,4) have especially high CKA (bright yellow spots), suggesting the
inner steps come in pairs that are geometrically very similar.

### 4.2 Paper Takeaway

This is strong evidence that **each outer cycle (T) constitutes a distinct
computational phase**, not just a repeated application of the same operation.
The model uses a different representational geometry at each stage, analogous
to how a human solver might first do "easy eliminations" (T=1), then
"constraint propagation" (T=2), then "final inference" (T=3). The inner
steps (i) refine within each phase without changing the underlying geometry.

This directly supports interpreting the H_cycles as **qualitatively different
reasoning stages**, not just additional iterations of the same computation.

---

## 5. Synthesizing the Full Picture

Putting all experiments together, here is the story your results tell:

### The narrative

1. **The TRM's latent state z_L encodes Sudoku candidate-set structure at
  every recursion step** (F1 ~0.77-0.82, far above the permutation null of
   ~0.15). This is the foundational finding.
2. **The three outer cycles are qualitatively different computational phases.**
  CKA shows block-diagonal structure with decreasing cross-block similarity.
   PCA shows T=3 occupying a geometrically distinct region. The model is not
   simply iterating the same computation three times — each outer cycle
   transforms the representation into a new regime.
3. **Within each outer cycle, inner steps refine without restructuring.**
  CKA is high within each 6x6 block. Probe F1 fluctuates but does not
   monotonically increase. The inner loop appears to adjust values within a
   fixed geometric framework rather than progressively building up constraint
   information.
4. **Causal necessity grows with depth.** Patching at (T=1, i=4) has no
  significant effect (the model self-corrects). Patching at (T=2, i=5) has a
   strong causal effect. This means the later cycles carry irreplaceable
   information.
5. **The model relies on z_L's content, not its positional structure, at
  deeper steps.** Cross-puzzle swaps hurt; within-puzzle shuffles help or
   are neutral. The model aggregates information across cells rather than
   maintaining strict per-cell assignments.
6. **The backtracking F1 gap is a metric artifact** driven by candidate-set
  size distribution: hard puzzles have more blank cells with larger |S_c|,
   inflating micro-averaged F1.

### What this means for the paper

The strongest contributions for your report are:

- **CKA block structure** — clean, visual, interpretable evidence of
multi-phase reasoning.
- **Patching asymmetry** — shows the difference between representational
richness and causal importance, a nuanced finding.
- **Probe results + null comparison** — solid evidence that candidate-set
information is genuinely encoded (F1 ~0.81 vs null ~0.15).
- **|S_c| confound analysis** — shows methodological rigor.

---

## 6. Suggested Next Steps

### High-priority (recommended for the paper)

1. **Cleaner PCA/UMAP plots grouped by T only.** Reduce to 3 colours (T=1,
  T=2, T=3) and use density contours. This makes the phase-transition
   finding visually clear and publishable.
2. **PCA coloured by |S_c| and by cell position.** Same coordinates, different
  colouring. This directly shows whether the geometry encodes candidate-set
   structure and spatial relationships.
3. **F1 stratified by |S_c|.** Compute F1 separately for cells with
  |S_c|=2, 3, 4, 5 to get a fair easy-vs-hard comparison that controls for
   the label-frequency confound.

### Medium-priority (extends the story)

1. **Delta-z analysis.** Visualize the *change* in z between consecutive
  steps to understand what each step adds.
2. **Patch at all 18 (T,i) positions.** Currently you have 2 data points.
  A full 18-point patching sweep would show how causal importance varies
   across the full recursion, and pair directly with the probe F1 heatmap.
3. **Per-puzzle latent trajectory.** For a few individual puzzles, plot
  the trajectory of z through PCA space across all 18 steps. This would
   show the "reasoning path" the model takes for specific inputs.

### Lower-priority (nice to have)

1. **Cross-checkpoint CKA.** If you have multiple training checkpoints, compare
  how the block structure emerges during training.
2. **Probe on z_H instead of z_L.** The high-level latent may show different
  patterns, particularly across outer cycles where it is updated.
3. **Attention pattern analysis.** Examine which cells attend to which other
  cells at different recursion steps — this would complement the latent-space
   view with a mechanistic view of information flow.

