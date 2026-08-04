#!/bin/bash -l
# ============================================================================
# train_trm_sudoku_v2.sh - Sudoku-Extreme (attention) with the corrected recipe.
#
# The maze v4 recipe retargeted at Sudoku: AdamATan2, weight_decay 1.0, the 16
# recurrent prefix registers, halt_exploration_prob 0.1. See
# config_pretrain_sudoku_v2.yml for the full diff against the old
# config_pretrain_paper.yml and for why the optimizer and the weight decay
# have to move together.
#
# Requires the adam-atan2 CUDA extension. Build it once with
# scripts/unity/setup_adam_atan2.sh. Compiled for arch 8.0/8.6/8.9, hence the
# a100|a40|l40s constraint: V100 (7.0) will not work.
#
# Runs on ONE GPU. A 4-GPU request sat PENDING on Priority indefinitely;
# a single GPU schedules quickly and the gpu partition allows 14 days.
#
# 65,104 optimizer steps, no gradient accumulation (seq_len is 97, not the
# maze's 916, so the full 768-row batch fits).
#
# Submit from repo root:
#     sbatch scripts/unity/train_trm_sudoku_v2.sh
# To resume:
#     RESUME_CKPT=path RUN_NAME=name sbatch scripts/unity/train_trm_sudoku_v2.sh
# ============================================================================

#SBATCH --job-name=TRM-SudokuV2
#SBATCH --partition=gpu
#SBATCH --time=96:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --constraint="a100|a40|l40s"
#SBATCH --output=logs/train_sudoku_v2_%j.out
#SBATCH --error=logs/train_sudoku_v2_%j.err

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

RUN_NAME="${RUN_NAME:-TRM-sudoku-v2}"

echo "=== TRM Sudoku-Extreme v2 (AdamATan2, wd=1.0, puzzle embeddings) ==="
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
    --config trm_base/config_pretrain_sudoku_v2.yml \
    "run_name=$RUN_NAME" \
    ${GBS:+"global_batch_size=$GBS"} \
    ${MICRO:+"micro_batch_size=$MICRO"} \
    ${EPOCHS:+"epochs=$EPOCHS"} \
    ${EVAL_INTERVAL:+"eval_interval=$EVAL_INTERVAL"} \
    ${MAXTEST:+"max_test_samples=$MAXTEST"} \
    ${EXTRA[@]+"${EXTRA[@]}"}

echo ""
echo "=== Done: $(date) ==="
