#!/bin/bash -l
#SBATCH --job-name=TRM-DiagCKA
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --constraint=l40s
#SBATCH --time=01:30:00
#SBATCH --mem=80G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/diag_cka_%j.out
#SBATCH --error=logs/diag_cka_%j.err
set -euo pipefail
module load python/3.11.7
module load cuda/12.6
source "$HOME/venvs/tinyllm/bin/activate"
cd "$HOME/LatentReasoningDecoding"
export PYTHONPATH="$PWD/trm_base:$PWD${PYTHONPATH:+:$PYTHONPATH}"
python -u -m experiments.probing.diagnose_cka_mismatch \
  --act-dir results/probing/activations \
  --stored-grid results/probing/cka/self_cka_z_L.npy \
  --n-puzzles 200
