#!/bin/bash -l
# Tiny CPU smoke test of the two rebuttal experiment scripts.
# GPU jobs are chained with --dependency=afterok on this job.

#SBATCH --job-name=TRM-Smoke
#SBATCH --partition=cpu
#SBATCH --time=00:45:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/smoke_%j.out
#SBATCH --error=logs/smoke_%j.err

set -euo pipefail

REPO_DIR="$(pwd)"
CHECKPOINT="checkpoints/Sudoku-extreme-1k-aug-1000-ACT-torch/TinyRecursiveReasoningModel_ACTV1 analytic-cobra/step_65100.pt"
CONFIG="trm_base/config_pretrain_paper.yml"
DATA_PATH="data/sudoku-extreme-1k-aug-1000"

mkdir -p logs

module load python/3.11.7
source "$HOME/venvs/tinyllm/bin/activate"
cd "$REPO_DIR"
export PYTHONPATH="$REPO_DIR/trm_base:$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"

echo ">>> Smoke: patching sweep (2 pairs, CPU) ..."
python -m experiments.probing.activation_patching_sweep \
    --config "$CONFIG" --checkpoint "$CHECKPOINT" --data-path "$DATA_PATH" \
    --output-dir /tmp/smoke_sweep --n-pairs 2 --batch-size 2 --seed 0 --device cpu
echo "SMOKE_SWEEP_OK"

echo ">>> Smoke: component ablation (2 puzzles, CPU, zero mode) ..."
python -m experiments.probing.component_ablation \
    --config "$CONFIG" --checkpoint "$CHECKPOINT" --data-path "$DATA_PATH" \
    --output-dir /tmp/smoke_ablate --n-puzzles 2 --batch-size 2 --seed 0 --mode zero
echo "SMOKE_ABLATE_OK"
