#!/bin/bash -l
# ============================================================================
# train_trm_maze_v3.sh - Maze-Hard, third attempt.
#
# v2 (job 62270703) completed all 390,625 steps and finished at 0.157 exact
# accuracy against the paper's ~0.85, flat-to-declining from step 117k on,
# with lm_loss frozen at ~0.07 since step 19k. See config_pretrain_maze_v3.yml
# for the full diagnosis.
#
# The fix that matters: upstream trains at global_batch_size=768, not 128.
# AdamW applies decoupled weight decay per STEP, so v2's 6x step count meant
# 6x the cumulative regularization at the same nominal weight_decay=1.0.
# 768 does not fit on one L40S, so this uses gradient accumulation:
# micro_batch_size=128 x 6 = 768, verified numerically equal to one 768 batch.
# Also restores the puzzle embedding (16 recurrent prefix registers) and
# halt_exploration_prob=0.1.
#
# Runs on ONE GPU. A 4-GPU request sat PENDING on Priority indefinitely;
# a single L40S schedules quickly and the gpu partition allows 14 days.
#
# 65,104 optimizer steps = 390,625 micro-batches, so ~51h, same as v2.
#
# NPROC=4 still works if you ever get a 4-GPU allocation: pretrain.py reads
# LOCAL_RANK, shards the dataloader by rank, and all-reduces gradients at each
# optimizer step, with the loss pre-scaled by 1/global_batch_size.
#
# Submit from repo root:
#     sbatch scripts/unity/train_trm_maze_v3.sh
# To resume:
#     RESUME_CKPT=path RUN_NAME=name sbatch scripts/unity/train_trm_maze_v3.sh
# ============================================================================

#SBATCH --job-name=TRM-MazeV3
#SBATCH --partition=gpu
#SBATCH --time=96:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --constraint=l40s
#SBATCH --output=logs/train_maze_v3_%j.out
#SBATCH --error=logs/train_maze_v3_%j.err

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

RUN_NAME="${RUN_NAME:-TRM-maze-v3}"

echo "=== TRM Maze-Hard v3 (batch 768 via grad accum + puzzle emb) ==="
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
    --config trm_base/config_pretrain_maze_v3.yml \
    "run_name=$RUN_NAME" \
    ${GBS:+"global_batch_size=$GBS"} \
    ${MICRO:+"micro_batch_size=$MICRO"} \
    ${EPOCHS:+"epochs=$EPOCHS"} \
    ${EVAL_INTERVAL:+"eval_interval=$EVAL_INTERVAL"} \
    ${EXTRA[@]+"${EXTRA[@]}"}

echo ""
echo "=== Done: $(date) ==="
