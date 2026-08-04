#!/bin/bash -l
# ============================================================================
# train_trm_maze_v5.sh - Maze-Hard, first run with RoPE actually working.
#
# Config is byte-for-byte v4's hyperparameters. The change is in the code:
# trm_base/layers.py rotate_half was missing its negation, so RoPE was not a
# rotation and q.k did not depend on relative position. See
# config_pretrain_maze_v5.yml for the measurements. Every earlier run in this
# repo, v1 through v4, trained with scrambled positional information on a
# 916-token spatial task.
#
# Requires the adam-atan2 CUDA extension. Build it once with
# scripts/unity/setup_adam_atan2.sh. Compiled for arch 8.0/8.6/8.9, hence the
# a100|a40|l40s constraint: V100 (7.0) will not work.
#
# Runs on ONE GPU. A 4-GPU request sat PENDING on Priority indefinitely;
# a single GPU schedules quickly and the gpu partition allows 14 days.
#
# 65,104 optimizer steps = 390,625 micro-batches, so ~51h, same as v2.
#
# NPROC=4 still works if you ever get a 4-GPU allocation: pretrain.py reads
# LOCAL_RANK, shards the dataloader by rank, and all-reduces gradients at each
# optimizer step, with the loss pre-scaled by 1/global_batch_size.
#
# Submit from repo root:
#     sbatch scripts/unity/train_trm_maze_v5.sh
# To resume:
#     RESUME_CKPT=path RUN_NAME=name sbatch scripts/unity/train_trm_maze_v5.sh
# ============================================================================

#SBATCH --job-name=TRM-MazeV5
#SBATCH --partition=gpu
#SBATCH --time=96:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --constraint="a100|a40|l40s"
#SBATCH --output=logs/train_maze_v5_%j.out
#SBATCH --error=logs/train_maze_v5_%j.err

set -euo pipefail

REPO_DIR="$(pwd)"
mkdir -p logs

module load python/3.11.7
module load cuda/12.6
source "$HOME/venvs/tinyllm/bin/activate"
cd "$REPO_DIR"
export PYTHONPATH="$REPO_DIR/trm_base:$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=4
export WANDB_MODE=offline

RUN_NAME="${RUN_NAME:-TRM-maze-v5}"

echo "=== TRM Maze-Hard v5 (RoPE fixed) ==="
echo "Job ${SLURM_JOB_ID:-none} on $(hostname) at $(date)"
echo "run name : $RUN_NAME"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo ""

EXTRA=()
if [ -n "${RESUME_CKPT:-}" ]; then
    if [ ! -f "$RESUME_CKPT" ]; then
        echo "FATAL: checkpoint not found: $RESUME_CKPT" >&2
        exit 1
    fi
    echo ">>> Resuming from $RESUME_CKPT"
    EXTRA+=("load_checkpoint=$RESUME_CKPT")
fi

# ${EXTRA[@]+...} so an empty array does not trip `set -u`.
torchrun --nproc-per-node "${NPROC:-1}" --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nnodes=1 \
    trm_base/pretrain.py \
    --config trm_base/config_pretrain_maze_v5.yml \
    "run_name=$RUN_NAME" \
    ${GBS:+"global_batch_size=$GBS"} \
    ${MICRO:+"micro_batch_size=$MICRO"} \
    ${EPOCHS:+"epochs=$EPOCHS"} \
    ${EVAL_INTERVAL:+"eval_interval=$EVAL_INTERVAL"} \
    ${EXTRA[@]+"${EXTRA[@]}"}

echo ""
echo "=== Done: $(date) ==="
