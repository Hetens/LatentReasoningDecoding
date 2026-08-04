"""Print eval metrics from an offline wandb run directory.

wandb runs in offline mode on Unity, so eval metrics never reach a UI and,
for runs started before the EVAL-print fix, never reach stdout either.
This reads the .wandb datastore directly and emits one line per eval.

Usage:
    python scripts/unity/dump_wandb_eval.py wandb/offline-run-20260726_185602-zjxgerf1
"""

import glob
import os
import sys

from wandb.proto import wandb_internal_pb2 as pb
from wandb.sdk.internal import datastore


def eval_rows(run_dir: str):
    files = glob.glob(os.path.join(run_dir, "*.wandb"))
    if not files:
        raise FileNotFoundError(f"no .wandb file under {run_dir}")
    ds = datastore.DataStore()
    ds.open_for_scan(files[0])
    rows = []
    while True:
        data = ds.scan_data()
        if data is None:
            break
        rec = pb.Record()
        rec.ParseFromString(data)
        if rec.WhichOneof("record_type") != "history":
            continue
        # History items key on nested_key (a repeated field), not key.
        row = {("/".join(it.nested_key) if it.nested_key else it.key): it.value_json
               for it in rec.history.item}
        if any(k.startswith("all/") for k in row):
            rows.append(row)
    return rows


def main() -> None:
    run_dir = sys.argv[1] if len(sys.argv) > 1 else max(
        glob.glob("wandb/offline-run-*"), key=os.path.getmtime)
    for row in eval_rows(run_dir):
        metrics = {k: float(v) for k, v in row.items() if k.startswith("all/")}
        print("EVAL step={} | {}".format(
            row.get("_step", "?"),
            "  ".join(f"{k}={v:.4f}" for k, v in sorted(metrics.items()))))


if __name__ == "__main__":
    main()
