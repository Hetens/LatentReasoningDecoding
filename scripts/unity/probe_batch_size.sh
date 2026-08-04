#!/bin/bash -l
# ============================================================================
# probe_batch_size.sh - Largest maze batch that fits, and its throughput.
#
# The maze run plateaued at ~0.185 exact accuracy (slope over the last 8
# evals is +0.021/100k steps, p=0.60, indistinguishable from flat). Eval
# std is 0.0224 against a binomial noise floor of 0.0123, so the weights
# themselves wander between evals: the signature of a gradient-noise floor
# from batch 64 with constant LR 1e-4. The paper uses batch 768.
#
# pretrain.py has no gradient accumulation, so the usable batch is capped
# by one L40S (46 GiB) at seq_len 900. This measures where that cap is.
#
# Submit from repo root:
#     sbatch scripts/unity/probe_batch_size.sh
# ============================================================================

#SBATCH --job-name=TRM-ProbeBatch
#SBATCH --partition=gpu
#SBATCH --time=00:30:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --constraint=l40s
#SBATCH --output=logs/probe_batch_%j.out
#SBATCH --error=logs/probe_batch_%j.err

set -euo pipefail

REPO_DIR="$(pwd)"
mkdir -p logs

module load python/3.11.7
module load cuda/12.6
source "$HOME/venvs/tinyllm/bin/activate"
cd "$REPO_DIR"
export PYTHONPATH="$REPO_DIR/trm_base:$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"

python -u probe_batch.py
