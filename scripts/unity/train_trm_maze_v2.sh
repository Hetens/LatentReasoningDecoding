#!/bin/bash -l
# ============================================================================
# train_trm_maze_v2.sh - Maze-Hard with the TRM repo's actual hyperparameters.
#
# The first attempt plateaued at ~0.185 exact accuracy against the paper's
# ~85%. Diagnosis was NOT insufficient training: regressing the last 8
# evals over 73k steps gives slope +0.021/100k, p=0.60, i.e. flat. The
# config was wrong in three places (see config_pretrain_maze_v2.yml), the
# important one being weight_decay 0.1 where upstream uses 1.0.
#
# Runs on ONE GPU. A 4-GPU request sat PENDING on Priority indefinitely;
# a single L40S schedules quickly and the gpu partition allows 14 days, so
# the slower wall-clock is the cheaper trade.
#
# global_batch_size 128 is the measured ceiling for an L40S at seq_len 900
# (30.6 of 44.4 GiB; 256 OOMs), and upstream notes single-GPU 128 costs no
# noticeable accuracy.
# total_steps = 50000 * 1000 / 128 = 390,625, ~67h at the measured
# 206 samples/s.
#
# NPROC=4 still works if you ever get a 4-GPU allocation: pretrain.py reads
# LOCAL_RANK, shards the dataloader by rank, and all-reduces gradients, with
# the loss pre-scaled by 1/global_batch_size so a SUM all-reduce is correct.
#
# Submit from repo root:
#     sbatch scripts/unity/train_trm_maze_v2.sh
# To resume:
#     RESUME_CKPT=path RUN_NAME=name sbatch scripts/unity/train_trm_maze_v2.sh
# ============================================================================

#SBATCH --job-name=TRM-MazeV2
#SBATCH --partition=gpu
#SBATCH --time=96:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --constraint=l40s
#SBATCH --output=logs/train_maze_v2_%j.out
#SBATCH --error=logs/train_maze_v2_%j.err

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

RUN_NAME="${RUN_NAME:-TRM-maze-v2}"

echo "=== TRM Maze-Hard v2 (paper hyperparameters) ==="
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
    --config trm_base/config_pretrain_maze_v2.yml \
    "run_name=$RUN_NAME" \
    "global_batch_size=${GBS:-128}" \
    ${EPOCHS:+"epochs=$EPOCHS"} \
    ${EVAL_INTERVAL:+"eval_interval=$EVAL_INTERVAL"} \
    ${EXTRA[@]+"${EXTRA[@]}"}

echo ""
echo "=== Done: $(date) ==="
