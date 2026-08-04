#!/bin/bash -l
#SBATCH --job-name=TRM-ReconCKA
#SBATCH --partition=gpu
#SBATCH --time=01:30:00
#SBATCH --mem=120G
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --constraint=l40s
#SBATCH --output=logs/recon_cka_%j.out
#SBATCH --error=logs/recon_cka_%j.err
set -euo pipefail
module load python/3.11.7
module load cuda/12.6
source "$HOME/venvs/tinyllm/bin/activate"
cd "$HOME/LatentReasoningDecoding"
export PYTHONPATH="$PWD/trm_base:$PWD${PYTHONPATH:+:$PYTHONPATH}"
python -u -m experiments.probing.reconcile_cka \
  --activations results/probing/activations/z_L_act16.pt \
  --stored-grid results/probing/cka/self_cka_z_L.npy
