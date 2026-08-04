#!/bin/bash -l
# ============================================================================
# run_rebuttal_round2.sh — patching follow-ups for the NeurIPS 29918 response.
#
#   z_H patching, shuffle-permutation distribution, difficulty-matched donor
#   pairs, and the clean-run stratified error analysis.
#
# Submit from repo root:
#     sbatch scripts/unity/run_rebuttal_round2.sh
# ============================================================================

#SBATCH --job-name=TRM-Rebuttal2
#SBATCH --partition=gpu-preempt,gpu
#SBATCH --time=00:45:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --constraint=bf16
#SBATCH --output=logs/rebuttal2_%j.out
#SBATCH --error=logs/rebuttal2_%j.err

set -euo pipefail

REPO_DIR="$(pwd)"

CHECKPOINT="checkpoints/Sudoku-extreme-1k-aug-1000-ACT-torch/TinyRecursiveReasoningModel_ACTV1 analytic-cobra/step_65100.pt"
CONFIG="trm_base/config_pretrain_paper.yml"
DATA_PATH="data/sudoku-extreme-1k-aug-1000"
OUT_DIR="results/probing/patching_extras"

mkdir -p logs "$OUT_DIR"

echo "=== TRM rebuttal round 2 ==="
echo "Job ID:  ${SLURM_JOB_ID:-none}"
echo "Node:    $(hostname)"
echo "Date:    $(date)"

module load python/3.11.7
module load cuda/12.6
source "$HOME/venvs/tinyllm/bin/activate"

cd "$REPO_DIR"
export PYTHONPATH="$REPO_DIR/trm_base:$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"

python -m experiments.probing.patching_extras \
    --config "$CONFIG" \
    --checkpoint "$CHECKPOINT" \
    --data-path "$DATA_PATH" \
    --output-dir "$OUT_DIR" \
    --n-pairs 200 --n-stratify 1000 --n-perms 20 \
    --batch-size 32 --seed 0 \
    --do zh,shufdist,matched,stratify

echo ""
echo ">>> Round 2 finished at $(date)"
