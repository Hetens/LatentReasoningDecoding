#!/bin/bash -l
# ============================================================================
# train_trm_maze.sh - Train TRM on Maze-Hard (30x30) for the ICLR track.
#
# seq_len 900 (vs 81 for Sudoku), so batch is cut to 64. Resumable:
# set RESUME_CKPT to chain a follow-on job from the latest checkpoint.
#
# Submit from repo root:
#     sbatch scripts/unity/train_trm_maze.sh
# ============================================================================

#SBATCH --job-name=TRM-MazeTrain
#SBATCH --partition=gpu
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --constraint=l40s
#SBATCH --output=logs/train_maze_%j.out
#SBATCH --error=logs/train_maze_%j.err

set -euo pipefail

REPO_DIR="$(pwd)"
mkdir -p logs

module load python/3.11.7
module load cuda/12.6
source "$HOME/venvs/tinyllm/bin/activate"
cd "$REPO_DIR"
export PYTHONPATH="$REPO_DIR/trm_base:$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=8
export WANDB_MODE=offline

echo "=== TRM Maze-Hard training ==="
echo "Job ${SLURM_JOB_ID:-none} on $(hostname) at $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

# To resume: RESUME_CKPT=path sbatch scripts/unity/train_trm_maze.sh
EXTRA=""
if [ -n "${RESUME_CKPT:-}" ]; then
    echo ">>> Resuming from $RESUME_CKPT"
    EXTRA="load_checkpoint=$RESUME_CKPT"
fi

python trm_base/pretrain.py --config trm_base/config_pretrain_maze.yml $EXTRA

echo ""
echo "=== Done: $(date) ==="
