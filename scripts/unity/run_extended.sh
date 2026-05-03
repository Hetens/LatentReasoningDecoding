#!/bin/bash
# ============================================================================
# run_extended.sh — Extended probing experiments (Phase 2).
#
# Assumes Phase 1 (run_probing.sh) has already completed and results exist
# in results/probing/{activations,labels,probe_results,cka}.
#
# This script runs:
#   1. |S_c| analysis per difficulty group
#   2. Activation patching at best (T=1,i=4) and weakest (T=2,i=5)
#   3. PCA / UMAP / corner-plot latent visualizations
#   4. Re-generate all plots with fixed CKA labels
#   5. Interpret patching results
#
# Submit from repo root:
#     sbatch scripts/unity/run_extended.sh
# ============================================================================

#SBATCH --job-name=TRM-Extended
#SBATCH --partition=gpu
#SBATCH --time=03:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --constraint=l40s
#SBATCH --output=logs/extended_%j.out
#SBATCH --error=logs/extended_%j.err

set -euo pipefail

REPO_DIR="$(pwd)"

# =====================  EDIT THESE  =====================
CHECKPOINT="checkpoints/Sudoku-extreme-1k-aug-1000-ACT-torch/TinyRecursiveReasoningModel_ACTV1 analytic-cobra/step_65100.pt"
CONFIG="trm_base/config_pretrain_paper.yml"
DATA_PATH="data/sudoku-extreme-1k-aug-1000"
# ========================================================

OUT_ROOT="results/probing"
ACT_DIR="$OUT_ROOT/activations"
LABEL_DIR="$OUT_ROOT/labels"
PROBE_DIR="$OUT_ROOT/probe_results"
CKA_DIR="$OUT_ROOT/cka"
PATCH_DIR="$OUT_ROOT/patching"
PLOT_DIR="$OUT_ROOT/plots"

mkdir -p logs "$PATCH_DIR" "$PLOT_DIR"

echo "=== TRM Extended Experiments (Phase 2) ==="
echo "Job ID:  $SLURM_JOB_ID"
echo "Node:    $(hostname)"
echo "Date:    $(date)"
nvidia-smi
echo ""

# ---- Modules & venv ----
module load python/3.11.7
module load cuda/12.6
source "$HOME/venvs/tinyllm/bin/activate"

# ---- Ensure new deps ----
pip install --quiet scikit-learn umap-learn

# ---- Environment ----
cd "$REPO_DIR"
export PYTHONPATH="$REPO_DIR/trm_base:$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"

# ---- Step 1: |S_c| analysis (CPU, fast) ----
echo ">>> [1/5] Candidate-set size analysis …"
python -m experiments.probing.analyze_results \
    --labels-dir "$LABEL_DIR" \
    --output-dir "$PLOT_DIR"

# ---- Step 2a: Activation patching at best (T=1, i=4) (GPU) ----
echo ">>> [2/5] Activation patching at (T=1, i=4) …"
python -m experiments.probing.activation_patching \
    --config "$CONFIG" \
    --checkpoint "$CHECKPOINT" \
    --data-path "$DATA_PATH" \
    --output-dir "$PATCH_DIR" \
    --target-T 1 --target-i 4 \
    --n-pairs 200 \
    --batch-size 32 \
    --seed 0

# ---- Step 2b: Activation patching at weakest (T=2, i=5) (GPU) ----
echo ">>> [3/5] Activation patching at (T=2, i=5) …"
python -m experiments.probing.activation_patching \
    --config "$CONFIG" \
    --checkpoint "$CHECKPOINT" \
    --data-path "$DATA_PATH" \
    --output-dir "$PATCH_DIR" \
    --target-T 2 --target-i 5 \
    --n-pairs 200 \
    --batch-size 32 \
    --seed 0

# ---- Step 3: Latent visualization (PCA + UMAP + corner) ----
echo ">>> [4/5] Latent visualizations …"
python -m experiments.probing.visualize_latents \
    --activations-dir "$ACT_DIR" \
    --output-dir "$PLOT_DIR" \
    --latent z_L --act-step last \
    --max-puzzles 500 --cells-per-puzzle 10 \
    --pca-components 10 --corner-components 5 \
    --seed 42

# ---- Step 4: Re-generate all plots (with fixed CKA labels) ----
echo ">>> [5/5] Re-generating plots (CKA labels fixed) …"
python -m experiments.probing.plot_results \
    --probe-dir "$PROBE_DIR" \
    --cka-dir "$CKA_DIR" \
    --output-dir "$PLOT_DIR" \
    --h-cycles 3 --l-cycles 6

# ---- Step 5: Interpret patching results ----
echo ">>> Interpreting patching results …"
python -m experiments.probing.analyze_results \
    --labels-dir "$LABEL_DIR" \
    --patching-dir "$PATCH_DIR" \
    --output-dir "$PLOT_DIR"

echo ""
echo ">>> All extended experiments finished at $(date)"
echo ">>> Results in: $OUT_ROOT"
