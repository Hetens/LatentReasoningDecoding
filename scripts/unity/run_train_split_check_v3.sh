#!/bin/bash -l
# ============================================================================
# run_train_split_check_v3.sh - Is maze v3 memorizing its 1,000 mazes?
#
# Before spending another ~51h on an augmented-data run, test the hypothesis
# that motivates it. Our maze train split is 1,000 fixed mazes with NO
# dihedral augmentation, each seen ~65,000 times. If that lack of variety is
# what caps us at 0.297 against the paper's ~0.85, the model should score far
# higher on the mazes it trained on than on held-out ones.
#
#   train >> test  -> memorizing 1,000 instances; build the aug8 set and rerun
#   train ~= test  -> underfitting; augmentation will not help, look elsewhere
#
# The same check on the v1 checkpoint returned 0.2270 on BOTH splits, i.e. no
# memorization at all. But v1 ran a different config (weight_decay 0.1,
# L_cycles 6, no puzzle embeddings) and scored lower, so it does not settle
# the question for v3. This reruns it on v3's final checkpoint with v3's
# config.
#
# Note the train/exact_accuracy logged during training cannot answer this: it
# is measured on the raw (non-EMA) weights mid-ACT, so it is not comparable to
# the EMA eval. This runs the same eval code on the train split.
#
# Short and small: good backfill candidate while the big runs are queued.
#
# Submit from repo root:
#     sbatch scripts/unity/run_train_split_check_v3.sh
# ============================================================================

#SBATCH --job-name=TRM-SplitV3
#SBATCH --partition=gpu
#SBATCH --time=01:30:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --constraint="a100|a40|l40s"
#SBATCH --output=logs/train_split_check_v3_%j.out
#SBATCH --error=logs/train_split_check_v3_%j.err

set -euo pipefail

REPO_DIR="$(pwd)"
CKPT="checkpoints/Maze-30x30-hard-1k-ACT-torch/TRM-maze-v3/step_65103.pt"

mkdir -p logs

module load python/3.11.7
module load cuda/12.6
source "$HOME/venvs/tinyllm/bin/activate"
cd "$REPO_DIR"
export PYTHONPATH="$REPO_DIR/trm_base:$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"

if [ ! -f "$CKPT" ]; then
    echo "FATAL: checkpoint not found: $CKPT" >&2
    exit 1
fi

echo "=== Memorization check: train vs test split, v3 final checkpoint ==="
echo "Job ${SLURM_JOB_ID:-none} on $(hostname) at $(date)"
echo "ckpt: $CKPT"
echo ""

for SPLIT in train test; do
    echo ">>> split=$SPLIT"
    python -u -m experiments.probing.error_distribution \
        --config trm_base/config_pretrain_maze_v3.yml \
        --data-path data/maze-30x30-hard-1k \
        --output-dir "results/probing/train_split_check_v3/$SPLIT" \
        --split "$SPLIT" \
        --n-puzzles 1000 --n-segments 16 --batch-size 16 \
        --checkpoints "$CKPT"
    echo ""
done

echo "=== Done: $(date) ==="
