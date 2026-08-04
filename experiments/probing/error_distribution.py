"""Per-puzzle error distribution across training checkpoints.

Tests a prediction that falls out of the fixed-point framing. If the
recursion is an iterated map converging to attractors, then improvement
during training should look like instances discretely falling into the
correct basin, not like a uniform reduction of per-cell error. The
observable difference:

  attractor picture -> the per-puzzle error histogram becomes BIMODAL,
      a growing spike at zero errors plus a tail of badly-wrong puzzles;
  uniform-refinement -> a single bump that slides left as training
      proceeds.

Maze-Hard makes this sharp. Route cells are only 12.5% of the grid, so
copying the input already scores 0.875 per-cell, and across the whole
grokking ramp route-cell accuracy moved just 65.6% -> 68.4% while exact
match went 0.000 -> 0.220. Total error barely changed; its distribution
across puzzles changed a great deal. This script measures that directly.

Reports per checkpoint: the error histogram, fraction solved exactly,
mean/median errors among unsolved puzzles, and a bimodality coefficient.

Usage (from repo root):
    python -m experiments.probing.error_distribution \
        --config trm_base/config_pretrain_maze.yml \
        --data-path data/maze-30x30-hard-1k \
        --output-dir results/probing/error_dist_maze \
        --checkpoints "ckpt/step_31248.pt" "ckpt/step_156240.pt" \
        --n-puzzles 1000 --batch-size 16
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List

import numpy as np
import torch
from tqdm import tqdm

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_TRM_BASE = os.path.join(_PROJECT_ROOT, "trm_base")
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
if _TRM_BASE not in sys.path:
    sys.path.insert(0, _TRM_BASE)

from experiments.probing.extract_activations import (  # noqa: E402
    load_trm_model,
    load_test_data,
)


@torch.no_grad()
def predict_batch(model, batch: Dict[str, torch.Tensor], device: torch.device,
                  n_segments: int) -> torch.Tensor:
    """Run n_segments ACT segments and return argmax predictions (B, S)."""
    inner = model.inner
    cfg = model.config
    B = batch["inputs"].shape[0]

    carry = inner.empty_carry(B)
    carry.z_H = carry.z_H.to(device)
    carry.z_L = carry.z_L.to(device)
    carry = inner.reset_carry(torch.ones(B, dtype=torch.bool, device=device), carry)

    seq_info = dict(cos_sin=inner.rotary_emb() if hasattr(inner, "rotary_emb") else None)
    input_emb = inner._input_embeddings(batch["inputs"], batch["puzzle_identifiers"])
    z_H, z_L = carry.z_H, carry.z_L

    for _s in range(n_segments):
        for _T in range(cfg.H_cycles):
            for _i in range(cfg.L_cycles):
                z_L = inner.L_level(z_L, z_H + input_emb, **seq_info)
            z_H = inner.L_level(z_H, z_L, **seq_info)

    logits = inner.lm_head(z_H)[:, inner.puzzle_emb_len:]
    return logits.float().argmax(dim=-1)


def bimodality_coefficient(x: np.ndarray) -> float:
    """Sarle's bimodality coefficient: (skew^2 + 1) / kurtosis.

    Above ~0.555 (the value for a uniform distribution) is conventionally
    read as bimodal; a normal distribution gives 0.333.
    """
    x = x.astype(np.float64)
    n = len(x)
    if n < 4 or x.std() == 0:
        return float("nan")
    m2 = ((x - x.mean()) ** 2).mean()
    m3 = ((x - x.mean()) ** 3).mean()
    m4 = ((x - x.mean()) ** 4).mean()
    skew = m3 / m2 ** 1.5
    kurt = m4 / m2 ** 2
    # Sample-corrected form.
    g1 = skew * np.sqrt(n * (n - 1)) / (n - 2)
    g2 = (n - 1) * ((n + 1) * (kurt - 3) + 6) / ((n - 2) * (n - 3))
    return float((g1 ** 2 + 1) / (g2 + 3 * (n - 1) ** 2 / ((n - 2) * (n - 3))))


def main() -> None:
    ap = argparse.ArgumentParser(description="Per-puzzle error distribution by checkpoint.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--data-path", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--checkpoints", nargs="+", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--n-puzzles", type=int, default=1000)
    ap.add_argument("--n-segments", type=int, default=16)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    raw = load_test_data(args.data_path, args.split, max_examples=args.n_puzzles)
    N = min(args.n_puzzles, len(raw["inputs"]))
    print(f"{N} puzzles, seq_len {raw['inputs'].shape[1]}")

    # Copy-the-input baseline: on Maze this is 0.875 per-cell, which is why
    # raw per-cell accuracy is nearly uninformative about progress.
    copy_err = (raw["inputs"][:N] != raw["labels"][:N]).sum(axis=1)
    print(f"copy baseline: mean {copy_err.mean():.1f} wrong cells per puzzle")

    results: Dict[str, dict] = {"n_puzzles": int(N), "copy_baseline_mean_errors": float(copy_err.mean()),
                                "checkpoints": {}}
    per_ckpt_errors: Dict[str, List[int]] = {}

    for ckpt in args.checkpoints:
        tag = os.path.basename(ckpt).replace(".pt", "")
        print(f"\n=== {tag} ===", flush=True)
        model = load_trm_model(args.config, ckpt, device, args.data_path, args.split)
        model.eval()

        errs = []
        for start in tqdm(range(0, N, args.batch_size), desc=tag):
            sel = np.arange(start, min(start + args.batch_size, N))
            batch = {
                k: torch.from_numpy(raw[k][sel].astype(np.int32)).to(device)
                for k in ("inputs", "labels", "puzzle_identifiers")
            }
            pred = predict_batch(model, batch, device, args.n_segments)
            labels = batch["labels"]
            mask = labels != -100
            wrong = ((pred != labels.long()) & mask).sum(dim=1)
            errs.append(wrong.cpu().numpy())

        e = np.concatenate(errs)
        per_ckpt_errors[tag] = e.tolist()
        unsolved = e[e > 0]
        stats = {
            "exact_acc": float((e == 0).mean()),
            "mean_errors": float(e.mean()),
            "median_errors": float(np.median(e)),
            "mean_errors_unsolved": float(unsolved.mean()) if len(unsolved) else 0.0,
            "median_errors_unsolved": float(np.median(unsolved)) if len(unsolved) else 0.0,
            "frac_le_5_errors": float((e <= 5).mean()),
            "frac_gt_20_errors": float((e > 20).mean()),
            "bimodality": bimodality_coefficient(e),
        }
        results["checkpoints"][tag] = stats
        for k, v in stats.items():
            print(f"  {k}: {v:.4f}")
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print("\n=== SUMMARY (the attractor prediction) ===")
    print(f"{'checkpoint':>18} {'exact':>7} {'mean_err':>9} {'err|unsolved':>13} {'bimodal':>8}")
    for tag, s in results["checkpoints"].items():
        print(f"{tag:>18} {s['exact_acc']:>7.3f} {s['mean_errors']:>9.1f} "
              f"{s['mean_errors_unsolved']:>13.1f} {s['bimodality']:>8.3f}")
    print("\n  Attractor picture: exact rises sharply while mean_err falls only")
    print("  slightly, err|unsolved stays flat or GROWS, bimodality rises.")
    print("  Uniform refinement: mean_err and err|unsolved both fall steadily.")

    with open(os.path.join(args.output_dir, "error_distribution.json"), "w") as f:
        json.dump(results, f, indent=2)
    np.savez_compressed(
        os.path.join(args.output_dir, "error_counts.npz"),
        **{k: np.array(v) for k, v in per_ckpt_errors.items()})
    print(f"\nSaved -> {args.output_dir}/error_distribution.json + error_counts.npz")


if __name__ == "__main__":
    main()
