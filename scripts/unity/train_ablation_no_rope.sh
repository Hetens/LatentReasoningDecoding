#!/bin/bash
# ============================================================================
# train_ablation_no_rope.sh — Ablation: paper config WITHOUT RoPE.
#
# Same as train_trm_paper.sh but uses pos_encodings: none.
# Compare W&B project "ablation-rope-paper-scale" against the paper run.
#
# Submit from repo root:
#     sbatch scripts/unity/train_ablation_no_rope.sh
# ============================================================================

#SBATCH --job-name=trm-no-rope
#SBATCH --partition=gpu
#SBATCH --time=24:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --constraint=l40s
#SBATCH --output=logs/train_no_rope_%j.out
#SBATCH --error=logs/train_no_rope_%j.err

set -euo pipefail

REPO_DIR="$(pwd)"

mkdir -p logs

echo "=== TRM Ablation: No RoPE (paper scale) ==="
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

echo ">>> Starting no-RoPE ablation training …"
python trm_base/pretrain.py --config experiments/ablation/configs/paper_no_rope.yml

echo ""
echo ">>> Training finished at $(date)"
