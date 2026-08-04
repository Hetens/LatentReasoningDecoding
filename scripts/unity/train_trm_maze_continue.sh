#!/bin/bash -l
# ============================================================================
# train_trm_maze_continue.sh - Continue Maze-Hard training past the cutoff.
#
# The first run (job 62187673, 156,240 steps) stopped in the MIDDLE of the
# grokking transition, not after it. Per-cell accuracy sat at ~0.958 the
# whole time (that is the trivial "copy the maze" solution, since only
# ~12% of the 900 cells are route cells), while exact_accuracy went
#
#   step   7.8k  0.000     step 117.2k  0.095
#   step  93.7k  0.033     step 132.8k  0.195
#   step 109.4k  0.038     step 156.2k  0.220   <- still climbing steeply
#
# Extrapolating the logit of exact_accuracy (slope ~3.4e-5 per step over
# the ramp) puts ~85%, the number TRM reports for Maze-Hard, around
# +90k steps. This script runs 200k more per leg to clear it with margin.
#
# Deliberately NOT changed: global_batch_size (64), lr (1e-4), and the
# architecture. The transition is already underway; batch size sets the
# gradient noise scale, which is exactly what drives a grokking
# transition, so perturbing it mid-run risks stalling the thing we are
# trying to finish.
#
# Caveats of resuming, both benign here:
#   - save_train_state stores only the EMA weights, no optimizer state, so
#     Adam moments restart and lr_warmup_steps (2000) is re-served. LR is
#     constant after warmup (lr_min_ratio=1.0), so nothing else resets.
#   - the step counter restarts at 0, so each leg gets its own run_name
#     and checkpoint directory rather than clobbering the previous one.
#
# Usage (from repo root):
#     RESUME_CKPT="checkpoints/.../step_156240.pt" RUN_NAME=TRM-maze-cont1 \
#         sbatch scripts/unity/train_trm_maze_continue.sh
#
# Or chain both legs unattended:
#     bash scripts/unity/submit_maze_chain.sh
# ============================================================================

#SBATCH --job-name=TRM-MazeCont
#SBATCH --partition=gpu
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --constraint=l40s
#SBATCH --output=logs/train_maze_cont_%j.out
#SBATCH --error=logs/train_maze_cont_%j.err

set -euo pipefail

REPO_DIR="$(pwd)"
mkdir -p logs

: "${RESUME_CKPT:?set RESUME_CKPT to the checkpoint to continue from}"
: "${RUN_NAME:?set RUN_NAME to a unique name for this leg}"

# 12800 epochs * 1000 groups / 64 batch = 200,000 steps, ~18.3h at the
# 3.04 steps/s the first run sustained. eval_interval must divide epochs;
# 640 gives 20 checkpoints, one every 10,000 steps.
EPOCHS="${EPOCHS:-12800}"
EVAL_INTERVAL="${EVAL_INTERVAL:-640}"

module load python/3.11.7
module load cuda/12.6
source "$HOME/venvs/tinyllm/bin/activate"
cd "$REPO_DIR"
export PYTHONPATH="$REPO_DIR/trm_base:$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=8
export WANDB_MODE=offline

echo "=== TRM Maze-Hard continuation ==="
echo "Job ${SLURM_JOB_ID:-none} on $(hostname) at $(date)"
echo "resume from : $RESUME_CKPT"
echo "run name    : $RUN_NAME"
echo "epochs      : $EPOCHS (eval every $EVAL_INTERVAL)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

if [ ! -f "$RESUME_CKPT" ]; then
    echo "FATAL: checkpoint not found: $RESUME_CKPT" >&2
    exit 1
fi

python -u trm_base/pretrain.py \
    --config trm_base/config_pretrain_maze.yml \
    "load_checkpoint=$RESUME_CKPT" \
    "run_name=$RUN_NAME" \
    "epochs=$EPOCHS" \
    "eval_interval=$EVAL_INTERVAL"

echo ""
echo "=== Done: $(date) ==="
