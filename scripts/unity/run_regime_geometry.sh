#!/bin/bash -l
# ============================================================================
# run_regime_geometry.sh - Recompute the paper's geometry per dynamical regime.
#
# Tests whether the "three geometrically distinct phases" result survives
# when solved and unsolved instances are separated, rather than pooled.
#
# Submit from repo root:
#     sbatch scripts/unity/run_regime_geometry.sh
# ============================================================================

#SBATCH --job-name=TRM-RegimeGeom
#SBATCH --partition=gpu
#SBATCH --time=03:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --constraint=l40s
#SBATCH --output=logs/regime_geom_%j.out
#SBATCH --error=logs/regime_geom_%j.err

set -euo pipefail

REPO_DIR="$(pwd)"
CHECKPOINT="checkpoints/Sudoku-extreme-1k-aug-1000-ACT-torch/TinyRecursiveReasoningModel_ACTV1 analytic-cobra/step_65100.pt"
CONFIG="trm_base/config_pretrain_paper.yml"
DATA_PATH="data/sudoku-extreme-1k-aug-1000"
OUT_DIR="results/probing/regime_geometry"

mkdir -p logs "$OUT_DIR"

module load python/3.11.7
module load cuda/12.6
source "$HOME/venvs/tinyllm/bin/activate"
cd "$REPO_DIR"
export PYTHONPATH="$REPO_DIR/trm_base:$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"

echo "=== Regime-split geometry ==="
echo "Job ${SLURM_JOB_ID:-none} on $(hostname) at $(date)"
echo ""

python -m experiments.probing.regime_split_geometry \
    --config "$CONFIG" --checkpoint "$CHECKPOINT" --data-path "$DATA_PATH" \
    --output-dir "$OUT_DIR" \
    --n-puzzles 400 --cells-per-puzzle 8 --batch-size 32 --seed 42

echo ""
echo "=== Done: $(date) ==="
