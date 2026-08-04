#!/bin/bash -l
# ============================================================================
# run_two_clock_maze.sh - Cross-task replication of the two-regime finding.
#
# Runs the two-clock trajectory analysis on the freshly trained Maze-Hard
# checkpoint. This doubles as the accuracy check for that run, since the
# script reports per-cell accuracy and solved fraction at the training
# horizon (wandb was offline, so no metrics were logged elsewhere).
#
# Question: does the Sudoku result replicate on a task with completely
# different constraint structure (path connectivity vs all-different)?
#   - solved instances contract to an exact fixed point,
#   - unsolved instances never converge,
#   - accuracy keeps improving past the training horizon with nothing lost.
#
# seq_len is 900 vs Sudoku's 81, so batch is reduced to 16.
#
# Submit from repo root:
#     sbatch scripts/unity/run_two_clock_maze.sh
# ============================================================================

#SBATCH --job-name=TRM-TwoClockMaze
#SBATCH --partition=gpu
#SBATCH --time=06:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --constraint=l40s
#SBATCH --output=logs/two_clock_maze_%j.out
#SBATCH --error=logs/two_clock_maze_%j.err

set -euo pipefail

REPO_DIR="$(pwd)"
# Override to analyse a continuation leg, e.g.
#   CHECKPOINT=checkpoints/.../TRM-maze-cont1/step_200000.pt \
#   OUT_DIR=results/probing/two_clock_maze_cont1 \
#       sbatch scripts/unity/run_two_clock_maze.sh
CKPT_DIR="checkpoints/Maze-30x30-hard-1k-ACT-torch/TinyRecursiveReasoningModel_ACTV1 tuscan-roadrunner"
CHECKPOINT="${CHECKPOINT:-$CKPT_DIR/step_156240.pt}"
CONFIG="trm_base/config_pretrain_maze.yml"
DATA_PATH="data/maze-30x30-hard-1k"
OUT_DIR="${OUT_DIR:-results/probing/two_clock_maze}"

mkdir -p logs "$OUT_DIR"

if [ ! -f "$CHECKPOINT" ]; then
    echo "FATAL: checkpoint not found: $CHECKPOINT" >&2
    exit 1
fi

module load python/3.11.7
module load cuda/12.6
source "$HOME/venvs/tinyllm/bin/activate"
cd "$REPO_DIR"
export PYTHONPATH="$REPO_DIR/trm_base:$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"

echo "=== Two-clock trajectory on Maze-Hard ==="
echo "Job ${SLURM_JOB_ID:-none} on $(hostname) at $(date)"
echo "checkpoint: $CHECKPOINT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

echo ">>> [1/2] sanity run (4 puzzles, 18 segments) ..."
python -u -m experiments.probing.two_clock_trajectory \
    --config "$CONFIG" --checkpoint "$CHECKPOINT" --data-path "$DATA_PATH" \
    --output-dir /tmp/two_clock_maze_sanity \
    --n-puzzles 4 --n-segments 18 --batch-size 4 --seed 0
echo "SANITY_OK"

echo ""
echo ">>> [2/2] full run (200 puzzles, 64 segments = 4x training horizon) ..."
python -u -m experiments.probing.two_clock_trajectory \
    --config "$CONFIG" --checkpoint "$CHECKPOINT" --data-path "$DATA_PATH" \
    --output-dir "$OUT_DIR" \
    --n-puzzles 200 --n-segments 64 --batch-size 16 --seed 0

echo ""
echo "=== Done: $(date) ==="
