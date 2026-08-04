#!/bin/bash -l
# ============================================================================
# run_two_clock.sh - Tier A: two-clock trajectory analysis.
#
# Traces the recursion across BOTH axes (ACT segments x inner steps),
# recording residual norms, per-decode answers, commit times, and the
# q-head halt signal. Runs 4x past the training horizon (64 vs 16
# segments) to test the fixed-point property directly.
#
# Submit from repo root:
#     sbatch scripts/unity/run_two_clock.sh
# ============================================================================

#SBATCH --job-name=TRM-TwoClock
#SBATCH --partition=gpu
#SBATCH --time=03:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --constraint=l40s
#SBATCH --output=logs/two_clock_%j.out
#SBATCH --error=logs/two_clock_%j.err

set -euo pipefail

REPO_DIR="$(pwd)"

CHECKPOINT="checkpoints/Sudoku-extreme-1k-aug-1000-ACT-torch/TinyRecursiveReasoningModel_ACTV1 analytic-cobra/step_65100.pt"
CONFIG="trm_base/config_pretrain_paper.yml"
DATA_PATH="data/sudoku-extreme-1k-aug-1000"
OUT_DIR="results/probing/two_clock"

mkdir -p logs "$OUT_DIR"

module load python/3.11.7
module load cuda/12.6
source "$HOME/venvs/tinyllm/bin/activate"
cd "$REPO_DIR"
export PYTHONPATH="$REPO_DIR/trm_base:$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"

echo "=== Two-clock trajectory (Tier A) ==="
echo "Job ${SLURM_JOB_ID:-none} on $(hostname) at $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

echo ">>> [1/2] sanity run (8 puzzles, 20 segments) ..."
python -m experiments.probing.two_clock_trajectory \
    --config "$CONFIG" --checkpoint "$CHECKPOINT" --data-path "$DATA_PATH" \
    --output-dir /tmp/two_clock_sanity \
    --n-puzzles 8 --n-segments 20 --batch-size 8 --seed 0
echo "SANITY_OK"

echo ""
echo ">>> [2/2] full run (200 puzzles, 64 segments = 4x training horizon) ..."
python -m experiments.probing.two_clock_trajectory \
    --config "$CONFIG" --checkpoint "$CHECKPOINT" --data-path "$DATA_PATH" \
    --output-dir "$OUT_DIR" \
    --n-puzzles 200 --n-segments 64 --batch-size 32 --seed 0

echo ""
echo "=== Done: $(date) ==="
