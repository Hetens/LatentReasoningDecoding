#!/bin/bash -l
# ============================================================================
# prepare_data_einstein.sh - Generate Einstein / zebra logic-grid puzzles.
#
# 5 houses x 5 attributes, up to 20 clues -> seq_len 125 (25 solution
# cells + 100 clue tokens), vocab 17. Comparable in size to Sudoku's 81
# so the TRM recipe transfers with minimal retuning.
#
# Output: data/einstein-5x5/{train,test}/
#
# Submit from repo root:
#     sbatch scripts/unity/prepare_data_einstein.sh
# ============================================================================

#SBATCH --job-name=TRM-EinsteinData
#SBATCH --partition=cpu
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=16
#SBATCH --output=logs/prepare_einstein_%j.out
#SBATCH --error=logs/prepare_einstein_%j.err

set -euo pipefail

REPO_DIR="$(pwd)"
mkdir -p logs

module load python/3.11.7
source "$HOME/venvs/tinyllm/bin/activate"
cd "$REPO_DIR"
export PYTHONPATH="$REPO_DIR/trm_base:$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"

echo "=== Einstein puzzle generation ==="
echo "Job ${SLURM_JOB_ID:-none} on $(hostname) at $(date)"

python trm_base/build_einstein_data.py \
    --n-houses 5 --n-attrs 5 \
    --n-train 100000 --n-test 5000 \
    --max-clues 20 \
    --n-workers 16 \
    --seed 42 \
    --output-dir data/einstein-5x5

echo ""
echo "=== Contents ==="
find data/einstein-5x5 -type f | head -20
du -sh data/einstein-5x5
echo "=== Done: $(date) ==="
