#!/bin/bash -l
# ============================================================================
# run_train_split_check.sh - Was the old maze run overfitting?
#
# The v1 maze config used weight_decay=0.1 where the TRM repo uses 1.0, on
# a dataset of 8,000 examples (1,000 mazes x 8 dihedral) that each get seen
# ~1,250 times. If under-regularization is what capped it at 0.185 test
# exact accuracy, the model should score far higher on its own TRAINING
# data.
#
# The `train/exact_accuracy` logged during training cannot answer this: it
# is measured on the raw (non-EMA) weights mid-ACT, so it is not comparable
# to the EMA eval. This runs the SAME eval code on the train split.
#
#   train >> test  -> overfitting, weight_decay 0.1 is the culprit
#   train ~= test  -> underfitting, something else is wrong
#
# Uses the v1 checkpoint and v1 config (L_cycles=6), which is what that
# checkpoint was trained with.
#
# Submit from repo root:
#     sbatch scripts/unity/run_train_split_check.sh
# ============================================================================

#SBATCH --job-name=TRM-TrainSplit
#SBATCH --partition=gpu
#SBATCH --time=02:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --constraint=l40s
#SBATCH --output=logs/train_split_check_%j.out
#SBATCH --error=logs/train_split_check_%j.err

set -euo pipefail

REPO_DIR="$(pwd)"
CKPT="checkpoints/Maze-30x30-hard-1k-ACT-torch/TinyRecursiveReasoningModel_ACTV1 tuscan-roadrunner/step_156240.pt"

mkdir -p logs

module load python/3.11.7
module load cuda/12.6
source "$HOME/venvs/tinyllm/bin/activate"
cd "$REPO_DIR"
export PYTHONPATH="$REPO_DIR/trm_base:$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"

echo "=== Overfitting check: train split vs test split, v1 maze checkpoint ==="
echo "Job ${SLURM_JOB_ID:-none} on $(hostname) at $(date)"
echo ""

for SPLIT in train test; do
    echo ">>> split=$SPLIT"
    python -u -m experiments.probing.error_distribution \
        --config trm_base/config_pretrain_maze.yml \
        --data-path data/maze-30x30-hard-1k \
        --output-dir "results/probing/train_split_check/$SPLIT" \
        --split "$SPLIT" \
        --n-puzzles 1000 --n-segments 16 --batch-size 16 \
        --checkpoints "$CKPT"
    echo ""
done

echo "=== Done: $(date) ==="
