#!/bin/bash
# Rerun only CKA + plots (steps 1–4 already completed).
#SBATCH --job-name=TRM-CKA
#SBATCH --partition=gpu
#SBATCH --time=02:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --constraint=l40s
#SBATCH --output=logs/cka_plots_%j.out
#SBATCH --error=logs/cka_plots_%j.err

set -euo pipefail
REPO_DIR="$(pwd)"

OUT_ROOT="results/probing"
ACT_DIR="$OUT_ROOT/activations"
PROBE_DIR="$OUT_ROOT/probe_results"
CKA_DIR="$OUT_ROOT/cka"
PLOT_DIR="$OUT_ROOT/plots"

mkdir -p logs "$CKA_DIR" "$PLOT_DIR"

module load python/3.11.7
module load cuda/12.6
source "$HOME/venvs/tinyllm/bin/activate"

cd "$REPO_DIR"
export PYTHONPATH="$REPO_DIR/trm_base:$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"

echo ">>> Computing self-CKA …"
LAST_ACT=$(ls "$ACT_DIR"/z_L_act*.pt | sort -t 't' -k2 -n | tail -1)
python -m experiments.probing.cka self \
    --file "$LAST_ACT" \
    --latent z_L \
    --output-dir "$CKA_DIR"

echo ">>> Generating plots …"
python -m experiments.probing.plot_results \
    --probe-dir "$PROBE_DIR" \
    --cka-dir "$CKA_DIR" \
    --output-dir "$PLOT_DIR"

echo ">>> Done at $(date)"
