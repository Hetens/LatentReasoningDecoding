#!/bin/bash -l
# ============================================================================
# run_rebuttal_round1.sh — z_H probing + checkpoint-robustness patching sweeps.
#
# Round 1 of the NeurIPS 29918 discussion-period experiments:
#   (a) linear probes on z_H at its 3 cycle points (reviewer Vy5n Q5),
#   (b) the 18-index z_L patching sweep at three earlier checkpoints of the
#       same paper run (reviewer MKju: "single checkpoint").
#
# Submit from repo root:
#     sbatch scripts/unity/run_rebuttal_round1.sh
# ============================================================================

#SBATCH --job-name=TRM-CkptSweep
#SBATCH --partition=gpu-preempt,gpu
#SBATCH --time=00:40:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --constraint=bf16
#SBATCH --output=logs/ckptsweep_%j.out
#SBATCH --error=logs/ckptsweep_%j.err

set -euo pipefail

REPO_DIR="$(pwd)"

CKPT_DIR="checkpoints/Sudoku-extreme-1k-aug-1000-ACT-torch/TinyRecursiveReasoningModel_ACTV1 analytic-cobra"
CONFIG="trm_base/config_pretrain_paper.yml"
DATA_PATH="data/sudoku-extreme-1k-aug-1000"
OUT_ROOT="results/probing"

mkdir -p logs "$OUT_ROOT/probe_results" "$OUT_ROOT/patching_sweep_ckpt"

echo "=== TRM rebuttal round 1 ==="
echo "Job ID:  ${SLURM_JOB_ID:-none}"
echo "Node:    $(hostname)"
echo "Date:    $(date)"
nvidia-smi || true
echo ""

module load python/3.11.7
module load cuda/12.6
source "$HOME/venvs/tinyllm/bin/activate"

cd "$REPO_DIR"
export PYTHONPATH="$REPO_DIR/trm_base:$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"

# ---- patching sweep at earlier checkpoints ----------------------------
for STEP in 39060 52080 58590; do
    echo ""
    echo ">>> [2/2] Patching sweep at step_${STEP} …"
    python -m experiments.probing.activation_patching_sweep \
        --config "$CONFIG" \
        --checkpoint "$CKPT_DIR/step_${STEP}.pt" \
        --data-path "$DATA_PATH" \
        --output-dir "$OUT_ROOT/patching_sweep_ckpt/step_${STEP}" \
        --n-pairs 200 --batch-size 32 --seed 0
done

echo ""
echo ">>> Round 1 finished at $(date)"
