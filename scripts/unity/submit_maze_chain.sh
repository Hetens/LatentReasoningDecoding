#!/bin/bash -l
# ============================================================================
# submit_maze_chain.sh - Chain two Maze-Hard continuation legs.
#
# Leg 1 resumes the interrupted grokking transition from step_156240 and
# runs 200k more steps; leg 2 (--dependency=afterok) adds another 200k as
# insurance and to confirm the curve plateaus rather than merely stopping.
#
# Cumulative: 156,240 + 200,000 + 200,000 = 556,240 steps.
#
# Leg 2 is cheap insurance: cancel it with `scancel <id>` once leg 1's
# EVAL lines show exact_accuracy flattening near the TRM paper's ~0.85.
#
# Run from repo root:  bash scripts/unity/submit_maze_chain.sh
# ============================================================================

set -euo pipefail

BASE="checkpoints/Maze-30x30-hard-1k-ACT-torch"
LEG0_CKPT="$BASE/TinyRecursiveReasoningModel_ACTV1 tuscan-roadrunner/step_156240.pt"

if [ ! -f "$LEG0_CKPT" ]; then
    echo "FATAL: starting checkpoint not found: $LEG0_CKPT" >&2
    exit 1
fi

# 12800 epochs -> 200,000 steps, so each leg's final checkpoint is step_200000.pt.
LEG1_CKPT="$BASE/TRM-maze-cont1/step_200000.pt"

JOB1=$(RESUME_CKPT="$LEG0_CKPT" RUN_NAME=TRM-maze-cont1 \
    sbatch --parsable scripts/unity/train_trm_maze_continue.sh)
echo "leg 1 submitted: $JOB1  (resumes step_156240, -> $BASE/TRM-maze-cont1)"

JOB2=$(RESUME_CKPT="$LEG1_CKPT" RUN_NAME=TRM-maze-cont2 \
    sbatch --parsable --dependency=afterok:"$JOB1" scripts/unity/train_trm_maze_continue.sh)
echo "leg 2 submitted: $JOB2  (afterok:$JOB1, -> $BASE/TRM-maze-cont2)"

echo ""
echo "Watch the transition with:"
echo "  grep EVAL logs/train_maze_cont_${JOB1}.out"
