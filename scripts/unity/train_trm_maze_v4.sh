#!/bin/bash -l
# ============================================================================
# train_trm_maze_v4.sh - Maze-Hard with AdamATan2, the upstream optimizer.
#
# Differs from v3 in exactly one setting: optimizer=adam_atan2 instead of
# adamw. v3 already had the batch size (768 via gradient accumulation), the
# puzzle embedding (16 recurrent prefix registers) and halt_exploration_prob
# right. See config_pretrain_maze_v4.yml for why the optimizer matters:
# AdamW's update collapses as gradients shrink and per-step weight decay takes
# over, which is the frozen-lm_loss equilibrium v2 died in.
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
#     sbatch scripts/unity/train_trm_maze_v4.sh
# To resume:
#     RESUME_CKPT=path RUN_NAME=name sbatch scripts/unity/train_trm_maze_v4.sh
# ============================================================================

#SBATCH --job-name=TRM-MazeV4
#SBATCH --partition=gpu
#SBATCH --time=96:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --constraint="a100|a40|l40s"
#SBATCH --output=logs/train_maze_v4_%j.out
#SBATCH --error=logs/train_maze_v4_%j.err

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

RUN_NAME="${RUN_NAME:-TRM-maze-v4}"

echo "=== TRM Maze-Hard v4 (AdamATan2, the upstream optimizer) ==="
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
    --config trm_base/config_pretrain_maze_v4.yml \
    "run_name=$RUN_NAME" \
    ${GBS:+"global_batch_size=$GBS"} \
    ${MICRO:+"micro_batch_size=$MICRO"} \
    ${EPOCHS:+"epochs=$EPOCHS"} \
    ${EVAL_INTERVAL:+"eval_interval=$EVAL_INTERVAL"} \
    ${EXTRA[@]+"${EXTRA[@]}"}

echo ""
echo "=== Done: $(date) ==="
