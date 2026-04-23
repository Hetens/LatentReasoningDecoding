#!/bin/bash
# ============================================================================
# run_probing.sh — Run the full probing experiment pipeline on Unity.
#
# Expects a trained TRM paper checkpoint.  Edit CHECKPOINT below to point
# at your best checkpoint .pt file.
#
# Submit from repo root:
#     sbatch scripts/unity/run_probing.sh
# ============================================================================

#SBATCH --job-name=TRM-Probing
#SBATCH --partition=gpu
#SBATCH --time=10:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --constraint=l40s
#SBATCH --output=logs/probing_%j.out
#SBATCH --error=logs/probing_%j.err

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

mkdir -p logs "$ACT_DIR" "$LABEL_DIR" "$PROBE_DIR" "$CKA_DIR" "$PATCH_DIR" "$PLOT_DIR"

echo "=== TRM Probing Experiments ==="
echo "Job ID:  $SLURM_JOB_ID"
echo "Node:    $(hostname)"
echo "Date:    $(date)"
nvidia-smi
echo ""

# ---- Modules & venv ----
module load python/3.11.7
module load cuda/12.6
source "$HOME/venvs/tinyllm/bin/activate"

# ---- Environment ----
cd "$REPO_DIR"
export PYTHONPATH="$REPO_DIR/trm_base:$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"

# ---- Step 1: Extract activations (GPU) ----
# 5000 examples × 2 ACT steps × (z_L + z_H) ≈ 21 GB on disk
echo ">>> [1/6] Extracting activations …"
python -m experiments.probing.extract_activations \
    --config "$CONFIG" \
    --checkpoint "$CHECKPOINT" \
    --data-path "$DATA_PATH" \
    --output-dir "$ACT_DIR" \
    --split test \
    --batch-size 128 \
    --max-examples 5000

# ---- Step 2: Compute candidate sets (CPU-bound) ----
echo ">>> [2/6] Computing candidate sets & backtracking flags …"
python -m experiments.probing.candidate_sets \
    --activations-dir "$ACT_DIR" \
    --output-dir "$LABEL_DIR" \
    --mode cp

# ---- Step 3a: Train linear probes (H1) ----
echo ">>> [3/6] Training linear probes …"
python -m experiments.probing.train_probes \
    --activations-dir "$ACT_DIR" \
    --labels-dir "$LABEL_DIR" \
    --output-dir "$PROBE_DIR" \
    --probe linear --latent z_L --act-step last \
    --run-null --seed 0

# ---- Step 3b: Train MLP probes (H2) ----
echo ">>> [4/6] Training MLP probes …"
python -m experiments.probing.train_probes \
    --activations-dir "$ACT_DIR" \
    --labels-dir "$LABEL_DIR" \
    --output-dir "$PROBE_DIR" \
    --probe mlp --latent z_L --act-step last \
    --run-null --seed 0

# ---- Step 4: Self-CKA heatmap ----
echo ">>> [5/6] Computing self-CKA …"
# Find the last ACT step file
LAST_ACT=$(ls "$ACT_DIR"/z_L_act*.pt | sort -t 't' -k2 -n | tail -1)
python -m experiments.probing.cka self \
    --file "$LAST_ACT" \
    --latent z_L \
    --output-dir "$CKA_DIR"

# ---- Step 5: Plots ----
echo ">>> [6/6] Generating plots …"
python -m experiments.probing.plot_results \
    --probe-dir "$PROBE_DIR" \
    --cka-dir "$CKA_DIR" \
    --output-dir "$PLOT_DIR"

echo ""
echo ">>> All probing experiments finished at $(date)"
echo ">>> Results in: $OUT_ROOT"
