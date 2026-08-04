#!/bin/bash -l
# ============================================================================
# run_geometry_controls.sh — clustering controls + convergence statistic.
#
# Reviewer Vy5n Q2 (given-vs-blank, solution-digit, position controls,
# |Sc|-balanced resample) and MKju W4 (quantitative convergence statistic).
#
# Submit from repo root:
#     sbatch scripts/unity/run_geometry_controls.sh
# ============================================================================

#SBATCH --job-name=TRM-GeomBlank
#SBATCH --partition=gpu-preempt,gpu
#SBATCH --time=01:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --constraint=bf16
#SBATCH --output=logs/geomblank_%j.out
#SBATCH --error=logs/geomblank_%j.err

set -euo pipefail

REPO_DIR="$(pwd)"
CHECKPOINT="checkpoints/Sudoku-extreme-1k-aug-1000-ACT-torch/TinyRecursiveReasoningModel_ACTV1 analytic-cobra/step_65100.pt"
CONFIG="trm_base/config_pretrain_paper.yml"
DATA_PATH="data/sudoku-extreme-1k-aug-1000"
OUT_DIR="results/probing/geometry_controls"

mkdir -p logs "$OUT_DIR"

echo "=== TRM geometry controls ==="
echo "Job ID:  ${SLURM_JOB_ID:-none}"
echo "Node:    $(hostname)"
echo "Date:    $(date)"

module load python/3.11.7
module load cuda/12.6
source "$HOME/venvs/tinyllm/bin/activate"

cd "$REPO_DIR"
export PYTHONPATH="$REPO_DIR/trm_base:$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"

python -m experiments.probing.geometry_controls \
    --config "$CONFIG" \
    --checkpoint "$CHECKPOINT" \
    --data-path "$DATA_PATH" \
    --output-dir "$OUT_DIR" \
    --n-puzzles 400 --batch-size 32 --pca-dim 50 --k 20 --seed 42 --blank-only

echo ""
echo ">>> Geometry controls finished at $(date)"
