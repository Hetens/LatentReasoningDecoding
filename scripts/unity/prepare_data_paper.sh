#!/bin/bash
# ============================================================================
# prepare_data_paper.sh — Build the Sudoku dataset matching the TRM paper:
#   1,000 base puzzles x 1,000 augmentations each.
#
# Submit from repo root:
#     sbatch scripts/unity/prepare_data_paper.sh
# ============================================================================

#SBATCH --job-name=tinyllm-data-paper
#SBATCH --partition=cpu-preempt
#SBATCH --time=01:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/prepare_data_paper_%j.out
#SBATCH --error=logs/prepare_data_paper_%j.err

set -euo pipefail

REPO_DIR="$(pwd)"

mkdir -p logs

echo "=== TinyLLM paper dataset preparation ==="
echo "Job ID:  $SLURM_JOB_ID"
echo "Node:    $(hostname)"
echo "Date:    $(date)"
echo "Config:  1000 base puzzles, 1000 augmentations each"
echo ""

# ---- Modules & venv ----
module load python/3.11.7
source "$HOME/venvs/tinyllm/bin/activate"

# ---- Run the dataset builder ----
cd "$REPO_DIR"
export PYTHONPATH="$REPO_DIR/trm_base${PYTHONPATH:+:$PYTHONPATH}"

echo ">>> Building Sudoku dataset (paper config) …"
python trm_base/build_sdku_data.py \
    --output-dir data/sudoku-extreme-1k-aug-1000 \
    --subsample-size 1000 \
    --num-aug 1000

echo ""
echo ">>> Data preparation complete."
ls -lh data/sudoku-extreme-1k-aug-1000/train/
ls -lh data/sudoku-extreme-1k-aug-1000/test/
