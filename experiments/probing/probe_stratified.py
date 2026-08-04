"""
Probe decoding quality broken down by candidate-set size and by digit.

Reviewer hi4P asked for error analysis beyond aggregate probe F1: where does
the probe fail, on high-ambiguity cells or on particular digits? This trains
the same probes as train_probes.py (identical split, seed and protocol) at
selected recursion indices and reports micro F1 per |Sc| stratum and per
digit on the held-out puzzles.

Usage (from repo root):
    python -m experiments.probing.probe_stratified \
        --activations-dir results/probing/activations \
        --labels-dir      results/probing/labels \
        --output-dir      results/probing/probe_stratified \
        --probe mlp --latent z_L --act-step 16 --indices 1,4 2,5 3,6
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(__file__)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
_PROJECT_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from probes import LinearProbe, MLPProbe  # noqa: E402
from experiments.probing.train_probes import (  # noqa: E402
    train_probe,
    micro_f1_from_counts,
    _counts,
)


def main() -> None:
    ap = argparse.ArgumentParser(description="Stratified probe evaluation.")
    ap.add_argument("--activations-dir", required=True)
    ap.add_argument("--labels-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--probe", choices=["linear", "mlp"], default="mlp")
    ap.add_argument("--latent", choices=["z_L", "z_H"], default="z_L")
    ap.add_argument("--act-step", default="16")
    ap.add_argument("--indices", nargs="+", default=["1,4", "2,5", "3,6"],
                    help="Recursion indices T,i in paper numbering (1-based).")
    ap.add_argument("--train-frac", type=float, default=0.8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--max-epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--mlp-hidden", type=int, default=128)
    ap.add_argument("--mlp-dropout", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    tag = f"act{args.act_step}"
    z = torch.load(os.path.join(args.activations_dir, f"{args.latent}_{tag}.pt"),
                   map_location="cpu", weights_only=True).float()
    y = np.load(os.path.join(args.labels_dir, "candidate_labels.npy"))  # (N, 81, 9)
    N, D = z.shape[0], z.shape[-1]

    # Identical split to train_probes.py at the same seed.
    idx = np.arange(N)
    rng = np.random.default_rng(args.seed)
    rng.shuffle(idx)
    n_train = int(N * args.train_frac)
    train_idx, val_idx = idx[:n_train], idx[n_train:]
    print(f"N={N} train={len(train_idx)} val={len(val_idx)} latent={args.latent} {tag}")

    yt = torch.from_numpy(y[train_idx].reshape(-1, 9))
    yv_np = y[val_idx].reshape(-1, 9)
    yv = torch.from_numpy(yv_np)
    set_sizes = yv_np.sum(axis=1).astype(int)     # |Sc| per validation cell

    results = {"latent": args.latent, "act_step": args.act_step,
               "probe": args.probe, "n_val_puzzles": int(len(val_idx)), "indices": []}

    for spec in args.indices:
        T, i = [int(v) for v in spec.split(",")]
        if args.latent == "z_L":
            z_ti = z[:, T - 1, i - 1]
        else:
            z_ti = z[:, T - 1]
        X_train = z_ti[train_idx].reshape(-1, D)
        X_val = z_ti[val_idx].reshape(-1, D)

        probe = (LinearProbe(D) if args.probe == "linear"
                 else MLPProbe(D, d_hidden=args.mlp_hidden, dropout=args.mlp_dropout))
        probe = train_probe(probe, X_train, yt, X_val, yv,
                            lr=args.lr, weight_decay=args.weight_decay,
                            batch_size=args.batch_size, max_epochs=args.max_epochs,
                            patience=args.patience, device=device)

        probe.eval()
        with torch.no_grad():
            pred = (probe(X_val.to(device)).sigmoid() >= 0.5).cpu().numpy().astype(np.float32)

        tp, fp, fn = _counts(pred, yv_np)
        entry = {"T": T, "i": i, "f1_overall": micro_f1_from_counts(tp, fp, fn),
                 "by_set_size": [], "by_digit": []}

        for s in range(1, 10):
            m = set_sizes == s
            if m.sum() == 0:
                continue
            tp, fp, fn = _counts(pred[m], yv_np[m])
            entry["by_set_size"].append({
                "set_size": int(s), "n_cells": int(m.sum()),
                "f1": micro_f1_from_counts(tp, fp, fn),
                "prevalence": float(yv_np[m].mean()),
            })
        for d in range(9):
            tp, fp, fn = _counts(pred[:, d:d + 1], yv_np[:, d:d + 1])
            entry["by_digit"].append({
                "digit": d + 1, "f1": micro_f1_from_counts(tp, fp, fn),
                "prevalence": float(yv_np[:, d].mean()),
            })

        results["indices"].append(entry)
        print(f"(T={T},i={i}) overall F1 {entry['f1_overall']:.4f}")
        print("   by |Sc|:", [(e["set_size"], round(e["f1"], 3)) for e in entry["by_set_size"]])
        print("   by digit:", [(e["digit"], round(e["f1"], 3)) for e in entry["by_digit"]])

    out = os.path.join(args.output_dir, f"probe_stratified_{args.probe}_{tag}_{args.latent}.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
