#!/bin/bash -l
# ============================================================================
# run_patching_sweep.sh - Rebuttal experiment P0-a.
#
# Systematic activation patching over ALL 18 recursion indices (T, i),
# both interventions (cross-puzzle swap, within-puzzle shuffle), with
# DeltaCE and DeltaAcc per step. Extends the two-index analysis behind
# Table 2 into a full causal map, as requested by all three reviewers.
#
# Submit from repo root:
#     sbatch scripts/unity/run_patching_sweep.sh
# ============================================================================

#SBATCH --job-name=TRM-PatchSweep
#SBATCH --partition=gpu
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --constraint=l40s
#SBATCH --output=logs/patch_sweep_%j.out
#SBATCH --error=logs/patch_sweep_%j.err

set -euo pipefail

REPO_DIR="$(pwd)"

# =====================  EDIT THESE  =====================
CHECKPOINT="checkpoints/Sudoku-extreme-1k-aug-1000-ACT-torch/TinyRecursiveReasoningModel_ACTV1 analytic-cobra/step_65100.pt"
CONFIG="trm_base/config_pretrain_paper.yml"
DATA_PATH="data/sudoku-extreme-1k-aug-1000"
# ========================================================

OUT_DIR="results/probing/patching_sweep"
mkdir -p logs "$OUT_DIR"

echo "=== TRM Patching Sweep (rebuttal P0-a) ==="
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

echo ">>> Patching sweep over all 18 (T, i) ..."
python -m experiments.probing.activation_patching_sweep \
    --config "$CONFIG" \
    --checkpoint "$CHECKPOINT" \
    --data-path "$DATA_PATH" \
    --output-dir "$OUT_DIR" \
    --n-pairs 200 \
    --batch-size 32 \
    --seed 0

echo ""
echo "=== Done: $(date) ==="
