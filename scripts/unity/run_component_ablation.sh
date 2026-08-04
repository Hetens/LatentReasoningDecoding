#!/bin/bash -l
# ============================================================================
# run_component_ablation.sh - Rebuttal experiment P0-b.
#
# Component-level ablations (attention vs. MLP per block, plus the five
# constraint-routing heads), scoped per outer cycle. Tests the paper's
# claim that the phase restructuring originates in MLP/value computation,
# and measures the downstream necessity of the routing heads.
#
# Submit from repo root:
#     sbatch scripts/unity/run_component_ablation.sh
# ============================================================================

#SBATCH --job-name=TRM-CompAblate
#SBATCH --partition=gpu
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --constraint=l40s
#SBATCH --output=logs/comp_ablate_%j.out
#SBATCH --error=logs/comp_ablate_%j.err

set -euo pipefail

REPO_DIR="$(pwd)"

# =====================  EDIT THESE  =====================
CHECKPOINT="checkpoints/Sudoku-extreme-1k-aug-1000-ACT-torch/TinyRecursiveReasoningModel_ACTV1 analytic-cobra/step_65100.pt"
CONFIG="trm_base/config_pretrain_paper.yml"
DATA_PATH="data/sudoku-extreme-1k-aug-1000"
# ========================================================

OUT_DIR="results/probing/component_ablation"
mkdir -p logs "$OUT_DIR"

echo "=== TRM Component Ablation (rebuttal P0-b) ==="
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

for MODE in zero mean; do
    echo ">>> Component ablation, mode=$MODE ..."
    python -m experiments.probing.component_ablation \
        --config "$CONFIG" \
        --checkpoint "$CHECKPOINT" \
        --data-path "$DATA_PATH" \
        --output-dir "$OUT_DIR" \
        --n-puzzles 200 \
        --batch-size 32 \
        --seed 0 \
        --mode "$MODE"
done

echo ""
echo "=== Done: $(date) ==="
