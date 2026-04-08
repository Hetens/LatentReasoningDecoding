#!/bin/bash
# ============================================================================
# train_trm.sh — SLURM job: train the TRM base model (single GPU).
#
# Submit from repo root:
#     sbatch scripts/unity/train_trm.sh
#
# Requests 1 A100 GPU (80 GB VRAM).  The model is small (~512 hidden) so a
# single GPU is sufficient.  Uses gpu-preempt (max 2 h); for longer runs
# switch to  --partition=gpu  and increase --time.
# ============================================================================

#SBATCH --job-name=tinyllm-train
#SBATCH --partition=gpu-preempt
#SBATCH --time=02:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --constraint=a100
#SBATCH --output=logs/train_trm_%j.out
#SBATCH --error=logs/train_trm_%j.err

set -euo pipefail

REPO_DIR="$(pwd)"

mkdir -p logs

echo "=== TinyLLM TRM training ==="
echo "Job ID:  $SLURM_JOB_ID"
echo "Node:    $(hostname)"
echo "Date:    $(date)"
nvidia-smi
echo ""

# ---- Modules & venv ----
module load python/3.11.7
module load cuda/12.6
source "$HOME/venvs/tinyllm/bin/activate"

# ---- Environment ----
cd "$REPO_DIR"
export PYTHONPATH="$REPO_DIR/trm_base${PYTHONPATH:+:$PYTHONPATH}"

# Enable torch.compile on A100
export FORCE_TORCH_COMPILE=1

# W&B — set to offline if you have no internet on compute nodes, or
# keep online if Unity compute nodes can reach the internet.
# export WANDB_MODE=offline

echo ">>> Starting training …"
python trm_base/pretrain.py --config trm_base/config_pretrain_unity.yml

echo ""
echo ">>> Training finished at $(date)"
