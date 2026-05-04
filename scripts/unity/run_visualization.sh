#!/bin/bash
# ============================================================================
# run_visualization.sh — Improved latent visualizations (Phase 3).
#
# Assumes Phase 1 results exist in results/probing/{activations,labels}.
# Produces all v2 plots: T-grouped, density contours, faceted, property-
# coloured (|S_c|, row/col/box), delta-z, and per-cell decoding maps.
#
# Submit from repo root:
#     sbatch scripts/unity/run_visualization.sh
# ============================================================================

#SBATCH --job-name=TRM-Viz
#SBATCH --partition=gpu
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --constraint=l40s
#SBATCH --output=logs/viz_%j.out
#SBATCH --error=logs/viz_%j.err

set -euo pipefail

REPO_DIR="$(pwd)"

OUT_ROOT="results/probing"
ACT_DIR="$OUT_ROOT/activations"
LABEL_DIR="$OUT_ROOT/labels"
PLOT_DIR="$OUT_ROOT/plots"
CKA_DIR="$OUT_ROOT/cka"
PROBE_DIR="$OUT_ROOT/probe_results"

mkdir -p logs "$PLOT_DIR"

echo "=== TRM Visualization (Phase 3) ==="
echo "Job ID:  $SLURM_JOB_ID"
echo "Node:    $(hostname)"
echo "Date:    $(date)"
echo ""

# ---- Modules & venv ----
module load python/3.11.7
module load cuda/12.6
source "$HOME/venvs/tinyllm/bin/activate"

pip install --quiet scikit-learn umap-learn

cd "$REPO_DIR"
export PYTHONPATH="$REPO_DIR/trm_base:$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"

# ---- V2 visualizations ----
echo ">>> [1/2] Running v2 latent visualizations …"
python -m experiments.probing.visualize_latents_v2 \
    --activations-dir "$ACT_DIR" \
    --labels-dir "$LABEL_DIR" \
    --output-dir "$PLOT_DIR" \
    --latent z_L --act-step last \
    --max-puzzles 200 --cells-per-puzzle 5 \
    --n-decode-puzzles 10 \
    --seed 42

# ---- Re-generate original plots with fixed CKA labels ----
echo ">>> [2/2] Re-generating standard plots (fixed CKA) …"
python -m experiments.probing.plot_results \
    --probe-dir "$PROBE_DIR" \
    --cka-dir "$CKA_DIR" \
    --output-dir "$PLOT_DIR" \
    --h-cycles 3 --l-cycles 6

echo ""
echo ">>> All visualizations finished at $(date)"
echo ">>> Results in: $PLOT_DIR"
