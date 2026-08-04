#!/bin/bash -l
# ============================================================================
# run_error_distribution_sudoku.sh - Does error consolidation replicate?
#
# The Maze run showed training SORTS instances into basins rather than
# reducing error: exact accuracy rose 227x while mean per-puzzle error
# fell 4.6%, bimodality rose 0.443 -> 0.595, and on puzzles unsolved at
# both endpoints errors GREW +6.9 cells (paired, p=2.8e-10).
#
# This asks whether the same signature appears on Sudoku-Extreme, the task
# where the two-clock fixed-point result already replicates. If it does,
# error consolidation is a property of the recursion rather than of mazes.
#
# Uses the canonical run (analytic-cobra), which every other analysis
# script references. seq_len is 81, so this is far cheaper than the maze
# sweep and can run alongside the maze continuation.
#
# Submit from repo root:
#     sbatch scripts/unity/run_error_distribution_sudoku.sh
# ============================================================================

#SBATCH --job-name=TRM-ErrDistSudoku
#SBATCH --partition=gpu
#SBATCH --time=03:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --constraint=l40s
#SBATCH --output=logs/error_dist_sudoku_%j.out
#SBATCH --error=logs/error_dist_sudoku_%j.err

set -euo pipefail

REPO_DIR="$(pwd)"
CKPT_DIR="checkpoints/Sudoku-extreme-1k-aug-1000-ACT-torch/TinyRecursiveReasoningModel_ACTV1 analytic-cobra"
OUT_DIR="results/probing/error_dist_sudoku"

mkdir -p logs "$OUT_DIR"

module load python/3.11.7
module load cuda/12.6
source "$HOME/venvs/tinyllm/bin/activate"
cd "$REPO_DIR"
export PYTHONPATH="$REPO_DIR/trm_base:$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"

echo "=== Per-puzzle error distribution across Sudoku training ==="
echo "Job ${SLURM_JOB_ID:-none} on $(hostname) at $(date)"
echo ""

python -u -m experiments.probing.error_distribution \
    --config trm_base/config_pretrain_paper.yml \
    --data-path data/sudoku-extreme-1k-aug-1000 \
    --output-dir "$OUT_DIR" \
    --n-puzzles 2000 --n-segments 16 --batch-size 64 \
    --checkpoints \
        "$CKPT_DIR/step_6510.pt" \
        "$CKPT_DIR/step_13020.pt" \
        "$CKPT_DIR/step_26040.pt" \
        "$CKPT_DIR/step_39060.pt" \
        "$CKPT_DIR/step_52080.pt" \
        "$CKPT_DIR/step_65100.pt"

echo ""
echo "=== Done: $(date) ==="
