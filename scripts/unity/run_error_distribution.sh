#!/bin/bash -l
# ============================================================================
# run_error_distribution.sh - Does training consolidate errors or reduce them?
#
# Sweeps the Maze-Hard checkpoints across the grokking transition and
# measures the PER-PUZZLE error distribution at each. The fixed-point
# framing predicts the histogram goes bimodal (a spike at zero errors plus
# a tail of badly-wrong puzzles) rather than sliding left as one bump.
#
# Motivation: over the whole ramp, route-cell accuracy moved only
# 65.6% -> 68.4% while exact match went 0.000 -> 0.220. Total error hardly
# moved, so the change must be in its distribution across puzzles.
#
# Runs on the ORIGINAL run's checkpoints, so it needs no new training and
# can run alongside the continuation jobs.
#
# Submit from repo root:
#     sbatch scripts/unity/run_error_distribution.sh
# ============================================================================

#SBATCH --job-name=TRM-ErrDist
#SBATCH --partition=gpu
#SBATCH --time=03:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --constraint=l40s
#SBATCH --output=logs/error_dist_%j.out
#SBATCH --error=logs/error_dist_%j.err

set -euo pipefail

REPO_DIR="$(pwd)"
CKPT_DIR="checkpoints/Maze-30x30-hard-1k-ACT-torch/TinyRecursiveReasoningModel_ACTV1 tuscan-roadrunner"
OUT_DIR="results/probing/error_dist_maze"

mkdir -p logs "$OUT_DIR"

module load python/3.11.7
module load cuda/12.6
source "$HOME/venvs/tinyllm/bin/activate"
cd "$REPO_DIR"
export PYTHONPATH="$REPO_DIR/trm_base:$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"

echo "=== Per-puzzle error distribution across the maze grokking ramp ==="
echo "Job ${SLURM_JOB_ID:-none} on $(hostname) at $(date)"
echo ""

# Sample the ramp: flat region, onset, mid-transition, end.
python -u -m experiments.probing.error_distribution \
    --config trm_base/config_pretrain_maze.yml \
    --data-path data/maze-30x30-hard-1k \
    --output-dir "$OUT_DIR" \
    --n-puzzles 1000 --n-segments 16 --batch-size 16 \
    --checkpoints \
        "$CKPT_DIR/step_31248.pt" \
        "$CKPT_DIR/step_70308.pt" \
        "$CKPT_DIR/step_93744.pt" \
        "$CKPT_DIR/step_117180.pt" \
        "$CKPT_DIR/step_132804.pt" \
        "$CKPT_DIR/step_156240.pt"

echo ""
echo "=== Done: $(date) ==="
