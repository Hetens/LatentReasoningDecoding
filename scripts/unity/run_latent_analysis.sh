#!/bin/bash
# ============================================================================
# run_latent_analysis.sh — Deep latent space analysis: clustering, TriMAP/
# PaCMAP visualization, and induction head extraction.
#
# Assumes Phase 1 activations exist in results/probing/{activations,labels}.
# The induction head experiment needs the checkpoint and data path.
#
# Submit from repo root:
#     sbatch scripts/unity/run_latent_analysis.sh
# ============================================================================

#SBATCH --job-name=TRM-LatentAnalysis
#SBATCH --partition=gpu
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --constraint=l40s
#SBATCH --output=logs/latent_analysis_%j.out
#SBATCH --error=logs/latent_analysis_%j.err

set -euo pipefail

REPO_DIR="$(pwd)"

# =====================  EDIT THESE  =====================
CHECKPOINT="checkpoints/Sudoku-extreme-1k-aug-1000-ACT-torch/TinyRecursiveReasoningModel_ACTV1 analytic-cobra/step_65100.pt"
CONFIG="trm_base/config_pretrain_paper.yml"
DATA_PATH="data/sudoku-extreme-1k-aug-1000"
# ========================================================

OUT_ROOT="results/probing"
ACT_DIR="$OUT_ROOT/activations"
LABEL_DIR="$OUT_ROOT/labels"
PLOT_DIR="$OUT_ROOT/plots"
ATTN_DIR="$OUT_ROOT/attention"

mkdir -p logs "$PLOT_DIR" "$ATTN_DIR"

echo "=== TRM Deep Latent Analysis ==="
echo "Job ID:  $SLURM_JOB_ID"
echo "Node:    $(hostname)"
echo "Date:    $(date)"
nvidia-smi
echo ""

# ---- Modules & venv ----
module load python/3.11.7
module load cuda/12.6
source "$HOME/venvs/tinyllm/bin/activate"

pip install --quiet scikit-learn pacmap trimap hdbscan

cd "$REPO_DIR"
export PYTHONPATH="$REPO_DIR/trm_base:$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"

# ---- 1. Multidimensional Clustering ----
echo ">>> [1/3] Running multidimensional clustering …"
python -m experiments.probing.cluster_latents \
    --activations-dir "$ACT_DIR" \
    --labels-dir "$LABEL_DIR" \
    --output-dir "$PLOT_DIR" \
    --max-puzzles 200 --cells-per-puzzle 5 \
    --pca-dims 50

# ---- 2. TriMAP / PaCMAP Visualization ----
echo ">>> [2/3] Running TriMAP/PaCMAP visualization …"
python -m experiments.probing.visualize_trimap_pacmap \
    --activations-dir "$ACT_DIR" \
    --labels-dir "$LABEL_DIR" \
    --output-dir "$PLOT_DIR" \
    --max-puzzles 200 --cells-per-puzzle 5

# ---- 3. Induction Head Analysis ----
echo ">>> [3/3] Running induction head analysis …"
python -m experiments.probing.induction_heads \
    --config "$CONFIG" \
    --checkpoint "$CHECKPOINT" \
    --data-path "$DATA_PATH" \
    --output-dir "$ATTN_DIR" \
    --max-examples 100 \
    --batch-size 10 \
    --split test

echo ""
echo ">>> All analyses finished at $(date)"
echo ">>> Clustering/visualization results in: $PLOT_DIR"
echo ">>> Attention results in: $ATTN_DIR"
