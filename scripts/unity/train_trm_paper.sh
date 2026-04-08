#!/bin/bash
# ============================================================================
# train_trm_paper.sh — Train TRM with paper hyperparams on 1x L40S.
#
# Paper says ~18-20 hours total.  gpu-preempt has a 2-hour limit, so this
# job will get killed and you resubmit to resume from the last checkpoint.
#
# Submit from repo root:
#     sbatch scripts/unity/train_trm_paper.sh
# ============================================================================

#SBATCH --job-name=tinyllm-paper
#SBATCH --partition=gpu-preempt
#SBATCH --time=02:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --constraint=l40s
#SBATCH --output=logs/train_paper_%j.out
#SBATCH --error=logs/train_paper_%j.err

set -euo pipefail

REPO_DIR="$(pwd)"

mkdir -p logs

echo "=== TinyLLM TRM paper training ==="
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
export FORCE_TORCH_COMPILE=1

echo ">>> Starting paper-config training …"
python trm_base/pretrain.py --config trm_base/config_pretrain_paper.yml

echo ""
echo ">>> Training finished at $(date)"
