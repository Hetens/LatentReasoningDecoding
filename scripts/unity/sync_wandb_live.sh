#!/bin/bash -l
# ============================================================================
# sync_wandb_live.sh - Push an offline wandb run to the cloud periodically.
#
# Compute nodes have no outbound network, so training always runs with
# WANDB_MODE=offline and writes to wandb/offline-run-*/. This resyncs that
# directory from a node that does have internet, on an interval, until the
# SLURM job leaves the queue. Re-syncing resumes into the same run ID, so the
# dashboard just keeps extending.
#
# "unexpected EOF" on a live run is normal and not an error: wandb sync reads
# the transaction log up to whatever training has flushed so far.
#
# Usage (from repo root, on a login node):
#     ./scripts/unity/sync_wandb_live.sh <job_id> <offline-run-dir> [interval_s]
# ============================================================================

set -uo pipefail

JOB_ID="${1:?usage: sync_wandb_live.sh <job_id> <offline-run-dir> [interval_s]}"
RUN_DIR="${2:?usage: sync_wandb_live.sh <job_id> <offline-run-dir> [interval_s]}"
INTERVAL="${3:-7200}"

if [ ! -d "$RUN_DIR" ]; then
    echo "FATAL: no such run dir: $RUN_DIR" >&2
    exit 1
fi

source "$HOME/venvs/tinyllm/bin/activate"

while true; do
    STATE=$(sacct -j "$JOB_ID" --format=State -Pn 2>/dev/null | head -1)
    wandb sync "$RUN_DIR" >/dev/null 2>&1
    echo "$(date '+%F %T') synced $RUN_DIR (job $JOB_ID: ${STATE:-unknown})"

    case "$STATE" in
        RUNNING*|PENDING*|REQUEUED*|SUSPENDED*|"") ;;
        *)
            # One final pass so the tail of the run is not lost.
            sleep 30
            wandb sync "$RUN_DIR" >/dev/null 2>&1
            echo "$(date '+%F %T') final sync done, job state $STATE"
            exit 0
            ;;
    esac
    sleep "$INTERVAL"
done
