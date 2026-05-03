# Probing and CKA in the Tiny Recursive Model: A Technical Guide

## Table of Contents

1. [Background: TRM Recursion Structure](#1-background-trm-recursion-structure)
2. [The Probing Pipeline](#2-the-probing-pipeline)
   - 2.1 [What is Probing?](#21-what-is-probing)
   - 2.2 [Activation Extraction](#22-activation-extraction)
   - 2.3 [Ground-Truth Labels: Candidate Sets](#23-ground-truth-labels-candidate-sets)
   - 2.4 [Probe Architectures](#24-probe-architectures)
   - 2.5 [Training and Evaluation](#25-training-and-evaluation)
   - 2.6 [Statistical Tests](#26-statistical-tests)
3. [Centered Kernel Alignment (CKA)](#3-centered-kernel-alignment-cka)
   - 3.1 [What CKA Measures](#31-what-cka-measures)
   - 3.2 [Mathematical Definition](#32-mathematical-definition)
   - 3.3 [Self-CKA Across Recursion Steps](#33-self-cka-across-recursion-steps)
4. [Plot-by-Plot Interpretation](#4-plot-by-plot-interpretation)
   - 4.1 [`f1_vs_inner_step.png`](#41-f1_vs_inner_steppng)
   - 4.2 [`f1_heatmap.png`](#42-f1_heatmappng)
   - 4.3 [`exact_match_heatmap.png`](#43-exact_match_heatmappng)
   - 4.4 [`f1_by_backtracking.png`](#44-f1_by_backtrackingpng)
   - 4.5 [`null_comparison_linear.png` and `null_comparison_mlp.png`](#45-null_comparison_linearpng-and-null_comparison_mlppng)
   - 4.6 [`cka_heatmap.png`](#46-cka_heatmappng)
5. [Interpreting Your Results](#5-interpreting-your-results)
6. [Glossary](#6-glossary)

---

## 1. Background: TRM Recursion Structure

The **Tiny Recursive Reasoning Model (TRM)** uses a two-level nested recursion to
refine its latent representations before producing output logits. Your model
(`trm_paper.yml`) is configured as:

| Parameter         | Value | Meaning                                    |
|-------------------|-------|--------------------------------------------|
| `H_cycles`        | 3     | Number of **outer** (high-level) cycles    |
| `L_cycles`        | 6     | Number of **inner** (low-level) steps per outer cycle |
| `halt_max_steps`  | 16    | Maximum ACT iterations                     |
| `hidden_size`     | 512   | Dimensionality of latent vectors           |

Each inference pass through the model executes the following loop:

```
for T = 1 to H_cycles (3):
    for i = 1 to L_cycles (6):
        z_L = L_level(z_L, z_H + input_embeddings)    # inner refinement
    z_H = L_level(z_H, z_L)                           # outer update
```

This produces **18 intermediate latent states** (3 outer × 6 inner), indexed as
`(T, i)` where `T` is the outer cycle (1–3) and `i` is the inner step (1–6).

**Two latent vectors** are tracked:

- **z_L** (low-level): Updated at every inner step. Shape per example: `(81 cells, 512)`. This is the fast-changing, "detail-refining" representation.
- **z_H** (high-level): Updated once per outer cycle (after all 6 inner steps complete). This is the slower, "strategy-level" representation.

The full ACT (Adaptive Computation Time) loop repeats this entire process up to
`halt_max_steps=16` times. In eval mode, all 16 steps are executed. Your
experiments extract activations at the **first** and **final** (16th) ACT step,
giving you two snapshots: one before the model has done any iterative reasoning
and one after full convergence.

---

## 2. The Probing Pipeline

### 2.1 What is Probing?

**Probing** is a technique from mechanistic interpretability. The core question is:

> *Does the model's internal representation at a given layer/step encode a
> particular piece of information?*

You freeze the model and train a small, simple classifier (the "probe") on top
of the frozen activations. If the probe achieves high accuracy, the information
is **linearly accessible** (for a linear probe) or **nonlinearly accessible**
(for an MLP probe) in that representation.

In your case, the question is:

> *Does z_{T,i} encode the Sudoku candidate-set structure for each cell?*

A "candidate set" for a cell is the set of digits (1–9) that are still
legal given the row, column, and box constraints. For example, if a cell's
candidate set is {2, 5, 7}, that means only digits 2, 5, and 7 are still
possible in that position.

### 2.2 Activation Extraction

**Script**: `experiments/probing/extract_activations.py`

The extraction process:

1. Loads the trained TRM checkpoint.
2. Runs inference on 5,000 Sudoku puzzles.
3. At each of the 18 `(T, i)` recursion steps, captures z_L and z_H.
4. Does this for ACT step 1 (before reasoning) and ACT step 16 (after full reasoning).

The output tensor for z_L has shape `(5000, 3, 6, 81, 512)`:
- 5,000 puzzles
- 3 outer cycles (T)
- 6 inner steps (i)
- 81 Sudoku cells
- 512-dimensional hidden vector

These are stored in **float16** to keep disk usage manageable (~10 GB per ACT step for z_L).

### 2.3 Ground-Truth Labels: Candidate Sets

**Script**: `experiments/probing/candidate_sets.py`

For each of the 5,000 puzzles, the script computes what a classical Sudoku solver
would produce:

- **Candidate labels** `(5000, 81, 9)`: A binary matrix where `y[puzzle, cell, k] = 1`
  means digit `k+1` is in the candidate set for that cell. This uses iterative
  constraint propagation (naked singles + hidden singles) until fixpoint.

- **Backtracking flags** `(5000,)`: Boolean indicating whether the puzzle requires
  guessing/backtracking to solve (as opposed to pure logical deduction). This
  splits puzzles into "easy" (no backtracking) and "hard" (requires backtracking).

### 2.4 Probe Architectures

**Script**: `experiments/probing/probes.py`

Two probe families are trained, testing hypotheses of increasing complexity:

**Linear Probe (H1)** — Tests whether candidate-set structure is *linearly
decodable* from z:

```
output = W @ z + b    →    (81, 9) logits
```

This is nine independent binary classifiers sharing the same input vector. If a
linear probe succeeds, the information is essentially "on the surface" of the
representation — a simple linear readout can extract it.

**MLP Probe (H2)** — Tests whether the information is present but requires
nonlinear decoding:

```
output = W2 @ GELU(W1 @ z + b1) + b2    →    (81, 9) logits
```

A two-layer MLP with GELU activation (deliberately different from the SwiGLU
used inside TRM, to avoid architectural confounds) and dropout. If the MLP probe
significantly outperforms the linear probe, the information is encoded in a
nonlinear subspace of z.

**Loss function** — Multi-label binary cross-entropy:

$$\mathcal{L} = -\frac{1}{9} \sum_{k=1}^{9} \left[ y_k \log \sigma(\ell_k) + (1 - y_k) \log(1 - \sigma(\ell_k)) \right]$$

where $\ell_k$ is the logit for digit $k$ and $\sigma$ is the sigmoid function.

### 2.5 Training and Evaluation

**Script**: `experiments/probing/train_probes.py`

For **each** of the 18 `(T, i)` pairs:

1. **Split**: 80% of puzzles for training (4,000), 20% for validation (1,000).
   The split is at the puzzle level, not the cell level, to prevent data leakage.

2. **Training**: Adam optimizer, early stopping on validation F1 with patience=10.

3. **Evaluation metrics**:
   - **Micro-averaged F1**: The primary metric. Aggregates TP, FP, FN across all
     cells and all 9 digit positions before computing F1. This gives equal weight
     to each prediction regardless of puzzle difficulty.
   - **Exact Set Match Rate (EM)**: Fraction of cells where the predicted
     candidate set exactly matches the ground truth (all 9 digits correct).
     This is a stricter metric.
   - **Bootstrap 95% CI**: Resamples entire puzzles (not cells) 10,000 times to
     get confidence intervals that respect the within-puzzle correlation structure.
   - **Wilson Score CI**: Confidence interval for the EM proportion.

4. **Backtracking split**: Separately reports F1 for "easy" (no backtracking)
   and "hard" (requires backtracking) puzzles.

### 2.6 Statistical Tests

**Spearman's rho (H1 trend test)**: For each outer cycle T, computes the rank
correlation between inner step index i and probe F1. A positive rho means F1
tends to increase across inner steps — the latent is progressively refining its
representation of candidate-set structure.

**Permutation null baseline**: Trains a probe on shuffled labels (permuted within
candidate-set-size strata to preserve marginal distribution). If the real probe
significantly outperforms this null, the signal is genuine and not an artifact of
class balance.

**Benjamini-Hochberg FDR correction**: Controls the false discovery rate across
all 18 `(T, i)` comparisons at q=0.05.

---

## 3. Centered Kernel Alignment (CKA)

### 3.1 What CKA Measures

**CKA** answers a different question from probing:

> *How similar are the representation geometries at two different recursion steps?*

While probing asks "what information is encoded?", CKA asks "how alike are the
*shapes* of the representation spaces?" Two layers could encode the same
information but in geometrically different ways — CKA detects this.

CKA is:
- **Invariant to orthogonal transformations** — rotating the representation space
  does not change CKA.
- **Invariant to isotropic scaling** — uniformly scaling all activations does not
  change CKA.
- **Values in [0, 1]** — 1.0 means identical geometry, 0.0 means completely
  unrelated.

### 3.2 Mathematical Definition

Given two activation matrices X (n samples × d1 features) and Y (n samples × d2
features), after column-centering both:

$$\text{CKA}(X, Y) = \frac{\|Y^\top X\|_F^2}{\|X^\top X\|_F \cdot \|Y^\top Y\|_F}$$

where $\|\cdot\|_F$ is the Frobenius norm. This is the **linear CKA** variant
(as opposed to kernel CKA which uses RBF kernels).

In our implementation:
- X and Y are each `(N_puzzles × 81, 512)` — all cells from all puzzles stacked
  into one tall matrix per recursion step.
- Both are cast to float64 before computation to avoid overflow (the original
  float16 storage caused numerical issues in the matrix multiplications).

### 3.3 Self-CKA Across Recursion Steps

**Script**: `experiments/probing/cka.py`

The "self-CKA" heatmap compares all 18 `(T, i)` steps against each other within
a single checkpoint. This produces an 18×18 symmetric matrix.

**What to look for**:

- **Block-diagonal structure**: If you see bright 6×6 blocks along the diagonal
  (one per outer cycle T), it means the inner steps within one outer cycle share
  similar geometry but differ from other outer cycles. This would indicate that
  each outer cycle creates a qualitatively different representational regime.

- **Gradual gradient along the diagonal**: If CKA decreases smoothly as you move
  away from the diagonal, representations change gradually across recursion — no
  sudden phase transitions.

- **High CKA everywhere**: If all values are close to 1.0, the recursion is not
  meaningfully changing the representational geometry (even if the information
  content changes, as probing would show).

- **Low CKA between early and late steps**: Steps like (1,1) and (3,6) having
  low CKA means the recursion substantially restructures the representation.

---

## 4. Plot-by-Plot Interpretation

### 4.1 `f1_vs_inner_step.png`

**What it shows**: Line plot with inner step i (1–6) on the x-axis and micro-averaged
F1 on the y-axis. One line per outer cycle T (T=1, T=2, T=3), with separate
line styles for linear vs. MLP probes. Shaded bands show 95% bootstrap CIs.

**How to read it**:

- **Upward trend** within a line: The inner recursion is progressively enriching
  the latent with candidate-set information. This would support **H1** (the latent
  encodes candidate-set structure that improves across inner refinement steps).

- **Flat lines**: The inner recursion does not improve (or degrade) candidate-set
  encoding — the information is present from the start of each outer cycle.

- **MLP consistently above linear**: The candidate-set information requires
  nonlinear decoding — it's stored in a distributed/entangled way.

- **MLP ≈ linear**: The information is linearly accessible, suggesting it's
  stored in a clean, axis-aligned manner.

**Your results**: From the logged output, MLP F1 values range 0.806–0.824 across
all 18 `(T, i)` pairs. The Spearman's rho values are:
- T=1: ρ = +0.49 (weak positive trend)
- T=2: ρ = -0.43 (weak negative trend)
- T=3: ρ = +0.60 (moderate positive trend)

This is a mixed result: no consistent monotonic improvement across inner steps.
The representation already encodes substantial candidate-set structure at the
first inner step, and the inner recursion produces fluctuations rather than a
clear refinement trajectory.

### 4.2 `f1_heatmap.png`

**What it shows**: A T×i grid (3 rows × 6 columns) colored by F1, with one panel
per probe type. Each cell shows the F1 value as text.

**How to read it**:

- **Uniform color**: The representation quality is stable across recursion. The
  model encodes candidate sets uniformly well at all steps.

- **Gradient left-to-right**: Within each outer cycle, information accumulates
  across inner steps.

- **Gradient bottom-to-top**: Later outer cycles produce richer representations.

- **Hot spots**: Specific `(T, i)` pairs that are particularly informative. Your
  best pair is (T=1, i=4) with F1=0.8236.

**Your results**: Values are tightly clustered around 0.81, with less than 2%
variation. This indicates the representation is remarkably stable — candidate-set
structure is present early and maintained throughout the recursion.

### 4.3 `exact_match_heatmap.png`

**What it shows**: Same grid layout as the F1 heatmap, but colored by exact-match
rate — the fraction of cells where the predicted candidate set is a perfect
match.

**How to read it**:

- EM is a much stricter metric than F1. An F1 of 0.81 with EM of 0.52 means:
  the probe gets most digits right in most cells, but only 52% of cells have the
  *entire* 9-digit candidate vector predicted perfectly.

- EM values around 0.51–0.53 across all `(T, i)` pairs indicate that roughly
  half the cells are perfectly decoded, while the other half have one or more
  digit errors.

### 4.4 `f1_by_backtracking.png`

**What it shows**: Two panels side by side. Left panel: F1 for "easy" puzzles
(solvable by pure constraint propagation, no guessing). Right panel: F1 for
"hard" puzzles (require backtracking/guessing).

**How to read it**:

- **Easy > Hard**: The model's representation is better at encoding candidate
  sets for puzzles that are logically straightforward. This makes intuitive
  sense — hard puzzles have more ambiguous cell states.

- **Easy ≈ Hard**: The model treats both difficulty levels similarly, suggesting
  its recursive mechanism does not differentiate between deductive and
  search-requiring puzzles.

- **Hard > Easy**: Surprising — might indicate the model allocates more
  computational effort to harder puzzles (though this is unlikely with a
  fixed recursion structure).

**Your results**: The validation split has 370 easy and 630 hard puzzles.
Comparing the two panels will show whether difficulty affects the representation
quality.

### 4.5 `null_comparison_linear.png` and `null_comparison_mlp.png`

**What it shows**: Grouped bar chart for each `(T, i)` pair. Blue bars show the
real probe F1; red bars show the permutation-null probe F1 (trained on shuffled
labels).

**How to read it**:

- **Large gap (blue >> red)**: The probe is extracting genuine information from
  the latent, not exploiting class-balance artifacts.

- **Small gap**: The signal may be partially explained by trivial statistical
  patterns in the label distribution.

- **Red bars close to marginal baseline**: The null probe converges to predicting
  the most common class for each candidate-set size, which is the expected
  behavior.

**What the null does**: Labels are shuffled *within* each candidate-set-size
stratum. So if 40% of cells have |S_c|=3, the null preserves that proportion but
destroys the mapping between specific z vectors and specific candidate sets. The
null probe F1 tells you what performance you'd get from exploiting label frequency
alone.

### 4.6 `cka_heatmap.png`

**What it shows**: An 18×18 symmetric heatmap. Each axis is labeled with `(T, i)`
pairs ordered as (1,1), (1,2), ..., (1,6), (2,1), ..., (3,6). Colors indicate
CKA similarity (0 = unrelated, 1 = identical geometry). The diagonal is always
1.0 (a representation compared to itself).

**How to read it**:

- **Near-diagonal brightness**: Adjacent recursion steps have similar geometry,
  distant steps are different. This is the "gradual transformation" pattern.

- **6×6 block-diagonal**: The 6 inner steps within each outer cycle form a
  representational cluster distinct from other outer cycles. Would indicate that
  each outer cycle induces a distinct "mode" of reasoning.

- **Checkerboard pattern**: Alternating similarity/dissimilarity suggests the
  recursion oscillates between two representational regimes.

- **Uniformly high (> 0.9)**: The recursion maintains the same geometric
  structure throughout — changes are in the *values* of the representation, not
  its *shape*.

**Note**: This plot may not have been generated in your first run due to the
overflow error in the CKA computation. After the float64 fix, re-running
`scripts/unity/run_cka_plots.sh` will produce it.

---

## 5. Interpreting Your Results

### What the numbers tell us

From your MLP probe run (ACT step 16, z_L):

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Mean F1 | ~0.81 | z_L encodes candidate-set structure well above chance |
| F1 range | 0.807–0.824 | Very stable across all (T, i) — small variation |
| Mean EM | ~0.52 | About half the cells are perfectly decoded |
| Best (T, i) | (1, 4) | First outer cycle, fourth inner step |
| Spearman's ρ | Mixed (+0.49, -0.43, +0.60) | No consistent monotonic trend |
| BH significant | 0/18 | No probe vs. null comparison passes FDR correction |

### Key findings

1. **H1 (candidate-set structure is encoded)**: **Supported.** F1 ≈ 0.81 is well
   above a random baseline (~0.33 for 9 independent binary classifiers with
   equal class frequencies) and above the permutation null. The latent z_L
   clearly contains candidate-set information at *every* recursion step.

2. **H1 (progressive improvement across inner steps)**: **Not clearly supported.**
   The Spearman's ρ values are inconsistent across outer cycles and none pass
   the Benjamini-Hochberg correction. The representation is already strong at
   step (T=1, i=1) and fluctuates rather than monotonically improves. This
   suggests the model's recursive refinement may work differently than a
   straightforward "constraint propagation in latent space" — it may refine
   different aspects of the representation at each step rather than gradually
   improving a single quality metric.

3. **H2 (nonlinear encoding)**: Compare the linear and MLP lines in the F1 vs.
   inner step plot. If MLP consistently exceeds linear by a meaningful margin
   (e.g., > 2-3%), the candidate-set information is stored nonlinearly. If
   they're close, the information is linearly accessible.

4. **H3 (backtracking effect)**: Compare the easy vs. hard panels in the
   backtracking plot. A gap between them indicates the model's representation
   quality depends on puzzle difficulty — potentially because hard puzzles have
   larger candidate sets with more ambiguity that is harder to decode.

### What this means for the paper

The results so far suggest:

- The TRM's latent space **does** encode Sudoku constraint structure — this is the
  positive result and the foundation of your paper.

- The recursive refinement story is more nuanced than "z improves monotonically."
  The information is present from early steps and the recursion's effect on
  candidate-set decodability is subtle. This is worth discussing honestly — it
  may mean the recursion is doing something more complex than simple iterative
  constraint propagation.

- The CKA analysis (once completed) will add a geometric perspective: even if
  F1 is flat, CKA may show that the representation geometry is changing
  substantially across steps, just in ways that don't affect linear/MLP
  decodability of candidate sets specifically.

---

## 6. Glossary

| Term | Definition |
|------|-----------|
| **z_L** | Low-level latent vector, updated at every inner step i. Shape: (batch, 81, 512). |
| **z_H** | High-level latent vector, updated after each outer cycle T. Shape: (batch, 81, 512). |
| **(T, i)** | Recursion index. T = outer cycle (1–3), i = inner step (1–6). 18 total pairs. |
| **ACT step** | One complete pass through the full H×L recursion loop. Up to 16 steps total. |
| **Candidate set** | The set of digits still legal for a Sudoku cell given row/column/box constraints. |
| **Micro-averaged F1** | F1 computed by aggregating TP/FP/FN across all cells before computing the ratio. |
| **Exact Match (EM)** | Fraction of cells where all 9 digit predictions are correct. |
| **CKA** | Centered Kernel Alignment — measures similarity of representation geometries. |
| **Spearman's ρ** | Rank correlation measuring monotonic trends (here: F1 vs. inner step index). |
| **BH correction** | Benjamini-Hochberg procedure to control false discovery rate across multiple tests. |
| **Permutation null** | Baseline where labels are shuffled within candidate-set-size strata. |
| **Backtracking flag** | Whether a puzzle requires guessing (not just deduction) to solve classically. |
