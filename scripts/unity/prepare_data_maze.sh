#!/bin/bash -l
# ============================================================================
# prepare_data_maze.sh - Download + preprocess Maze-Hard (30x30) for TRM.
#
# Output: data/maze-30x30-hard-1k/{train,test}/
# Needs CPU + internet (HuggingFace download).
#
# Submit from repo root:
#     sbatch scripts/unity/prepare_data_maze.sh
# ============================================================================

#SBATCH --job-name=TRM-MazeData
#SBATCH --partition=cpu
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/prepare_maze_%j.out
#SBATCH --error=logs/prepare_maze_%j.err

set -euo pipefail

REPO_DIR="$(pwd)"
mkdir -p logs

module load python/3.11.7
source "$HOME/venvs/tinyllm/bin/activate"

cd "$REPO_DIR"
export PYTHONPATH="$REPO_DIR/trm_base:$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"

echo "=== Maze-Hard data preparation ==="
echo "Job ID: ${SLURM_JOB_ID:-none}   Node: $(hostname)   Date: $(date)"

# aug=true gives 8x dihedral augmentation on train (1k -> 8k puzzles),
# matching the HRM/TRM recipe for this dataset.
python trm_base/build_maze_data.py \
    --source-repo sapientinc/maze-30x30-hard-1k \
    --output-dir data/maze-30x30-hard-1k \
    --aug

echo ""
echo "=== Contents ==="
find data/maze-30x30-hard-1k -type f | head -20
du -sh data/maze-30x30-hard-1k
echo "=== Done: $(date) ==="
