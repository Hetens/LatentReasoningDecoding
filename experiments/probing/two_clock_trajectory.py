"""
Two-clock trajectory analysis: the recursion as a dynamical system.

TRM has two nested recursion axes and the paper analyzes only one of them:
  - ACT segments s = 1..halt_max (16)   <- not analyzed in the submission
  - inner indices (T, i) = 1..18        <- analyzed, but only at s = 16

This script traces both. Crucially it stores only scalars (residual norms)
and argmax decodes, never the 512-dim states, so the full 16 x 18 grid
costs megabytes rather than the ~290 GB a naive extraction would need.

Produces, per puzzle:
  - residual curve  r_t = ||z_t - z_{t-1}|| / ||z_{t-1}||  for z_L (every
    inner step) and z_H (every outer cycle), across all segments;
  - the decoded answer after every z_H update (3 per segment), giving an
    "answer trajectory" of length 3 * n_segments;
  - per-cell commit time (the last decode index at which the predicted
    token changed), which separates gradual refinement from the
    "grokking"/guessing dynamics reported for HRM by Ren & Liu (2601.10679);
  - the model's own q-head halt logits, to compare its halting signal
    against actual convergence.

Extended recursion (--n-segments > halt_max) tests the fixed-point
property directly: does the state stay put, drift, or keep improving when
run past its training horizon?

Usage (from repo root):
    python -m experiments.probing.two_clock_trajectory \
        --config trm_base/config_pretrain_paper.yml \
        --checkpoint "checkpoints/.../step_65100.pt" \
        --data-path data/sudoku-extreme-1k-aug-1000 \
        --output-dir results/probing/two_clock \
        --n-puzzles 200 --n-segments 64 --batch-size 32
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict

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


def _rel_residual(new: torch.Tensor, old: torch.Tensor) -> torch.Tensor:
    """||new - old|| / ||old||, per example (flattened over cells and dims)."""
    d = (new - old).flatten(1).float().norm(dim=1)
    n = old.flatten(1).float().norm(dim=1).clamp(min=1e-6)
    return d / n


@torch.no_grad()
def trace_batch(
    model,
    batch: Dict[str, torch.Tensor],
    device: torch.device,
    n_segments: int,
) -> Dict[str, np.ndarray]:
    """Run n_segments ACT segments, recording residuals and decodes.

    Returns arrays:
      res_L   (B, n_segments * L_cycles * H_cycles)
      res_H   (B, n_segments * H_cycles)
      preds   (B, n_segments * H_cycles, seq_len)   int8 argmax decodes
      qhalt   (B, n_segments * H_cycles)            halt logit
    """
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
    res_L, res_H, preds, qhalt = [], [], [], []

    for _s in range(n_segments):
        for _T in range(cfg.H_cycles):
            for _i in range(cfg.L_cycles):
                z_L_new = inner.L_level(z_L, z_H + input_emb, **seq_info)
                res_L.append(_rel_residual(z_L_new, z_L).cpu())
                z_L = z_L_new
            z_H_new = inner.L_level(z_H, z_L, **seq_info)
            res_H.append(_rel_residual(z_H_new, z_H).cpu())
            z_H = z_H_new

            # Decode the answer at this point in the trajectory.
            logits = inner.lm_head(z_H)[:, inner.puzzle_emb_len:]
            preds.append(logits.argmax(dim=-1).to(torch.int8).cpu())
            qhalt.append(inner.q_head(z_H[:, 0]).float()[..., 0].cpu())

    return {
        "res_L": torch.stack(res_L, dim=1).numpy(),
        "res_H": torch.stack(res_H, dim=1).numpy(),
        "preds": torch.stack(preds, dim=1).numpy(),
        "qhalt": torch.stack(qhalt, dim=1).numpy(),
    }


def commit_times(preds: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Per-cell commit time: 1 + last decode index at which the prediction
    changed. 0 means the cell never changed from its first decode.

    preds: (N, D, S) int8 decodes; mask: (N, S) valid-cell mask.
    Returns (N, S) int array; invalid cells set to -1.
    """
    N, D, S = preds.shape
    final = preds[:, -1, :][:, None, :]
    changed = preds != final                      # (N, D, S)
    idx = np.arange(D)[None, :, None]
    last_change = np.where(changed, idx, -1).max(axis=1) + 1
    last_change[~mask] = -1
    return last_change


def main() -> None:
    ap = argparse.ArgumentParser(description="Two-clock trajectory analysis.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data-path", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--n-puzzles", type=int, default=200)
    ap.add_argument("--n-segments", type=int, default=64,
                    help="ACT segments to run; > halt_max tests the fixed-point property.")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    print("Loading model ...")
    model = load_trm_model(args.config, args.checkpoint, device, args.data_path, args.split)
    model.eval()
    cfg = model.config
    halt_max = cfg.halt_max_steps
    print(f"  H_cycles={cfg.H_cycles} L_cycles={cfg.L_cycles} halt_max={halt_max}")
    print(f"  tracing {args.n_segments} segments ({args.n_segments / halt_max:.1f}x training horizon)")

    print("Loading data ...")
    raw = load_test_data(args.data_path, args.split, max_examples=args.n_puzzles)
    N = min(args.n_puzzles, len(raw["inputs"]))
    order = rng.permutation(len(raw["inputs"]))[:N]

    chunks = {k: [] for k in ("res_L", "res_H", "preds", "qhalt")}
    labels_all = []
    for start in tqdm(range(0, N, args.batch_size), desc="batches"):
        idx = order[start:start + args.batch_size]
        batch = {
            k: torch.from_numpy(raw[k][idx].astype(np.int32)).to(device)
            for k in ("inputs", "labels", "puzzle_identifiers")
        }
        out = trace_batch(model, batch, device, args.n_segments)
        for k in chunks:
            chunks[k].append(out[k])
        labels_all.append(batch["labels"].cpu().numpy())

    res_L = np.concatenate(chunks["res_L"])
    res_H = np.concatenate(chunks["res_H"])
    preds = np.concatenate(chunks["preds"])
    qhalt = np.concatenate(chunks["qhalt"])
    labels = np.concatenate(labels_all)
    mask = labels != -100

    # ---- Accuracy trajectory over decode points ----
    correct = (preds == labels[:, None, :]) & mask[:, None, :]
    cell_acc = correct.sum(axis=2) / np.maximum(mask.sum(axis=1)[:, None], 1)   # (N, D)
    solved = cell_acc == 1.0                                                    # (N, D)

    D = preds.shape[1]
    train_D = halt_max * cfg.H_cycles       # decode index where training horizon ends
    ct = commit_times(preds, mask)

    # ---- Grokking statistic: is progress concentrated at one step? ----
    # Fraction of a puzzle's total accuracy gain contributed by its single
    # largest one-step jump. 1.0 = pure step function, ~1/D = gradual.
    # Only meaningful for puzzles that actually moved: dividing by a
    # near-zero net gain otherwise explodes the ratio (seen on Maze, where
    # most puzzles are static and the mean blew up to ~1e6).
    gains = np.diff(cell_acc, axis=1)
    net_gain = cell_acc[:, -1] - cell_acc[:, 0]
    moved = np.abs(net_gain) > 0.01
    max_jump_frac = np.full(len(cell_acc), np.nan)
    max_jump_frac[moved] = gains[moved].max(axis=1) / net_gain[moved]

    final_solved = solved[:, min(train_D, D) - 1]
    summary = {
        "n_puzzles": int(len(preds)),
        "n_segments": args.n_segments,
        "halt_max": int(halt_max),
        "decodes_per_segment": int(cfg.H_cycles),
        "n_decodes": int(D),
        "train_horizon_decode_idx": int(train_D),
        "cell_acc_at_train_horizon": float(cell_acc[:, min(train_D, D) - 1].mean()),
        "cell_acc_at_end": float(cell_acc[:, -1].mean()),
        "solved_frac_at_train_horizon": float(final_solved.mean()),
        "solved_frac_at_end": float(solved[:, -1].mean()),
        "residual_L_first": float(res_L[:, 0].mean()),
        "residual_L_at_train_horizon": float(res_L[:, min(train_D * cfg.L_cycles, res_L.shape[1]) - 1].mean()),
        "residual_L_at_end": float(res_L[:, -1].mean()),
        "residual_H_at_end": float(res_H[:, -1].mean()),
        "max_jump_fraction_mean": float(np.nanmean(max_jump_frac)),
        "max_jump_n_moved": int(np.isfinite(max_jump_frac).sum()),
        "max_jump_fraction_solved": float(np.nanmean(max_jump_frac[final_solved])),
        "max_jump_fraction_unsolved": float(np.nanmean(max_jump_frac[~final_solved])),
        "mean_commit_time_all": float(ct[mask].mean()),
        "mean_commit_time_solved": float(ct[final_solved][mask[final_solved]].mean()),
        "mean_commit_time_unsolved": float(ct[~final_solved][mask[~final_solved]].mean())
        if (~final_solved).any() else float("nan"),
    }

    # Extended-recursion verdict.
    if D > train_D:
        acc_after = cell_acc[:, train_D:].mean(axis=0)
        drift = float(acc_after.max() - acc_after.min())
        summary["extended_acc_delta"] = float(cell_acc[:, -1].mean() - cell_acc[:, train_D - 1].mean())
        summary["extended_acc_drift"] = drift
        summary["extended_verdict"] = (
            "improves" if summary["extended_acc_delta"] > 0.005 else
            "degrades" if summary["extended_acc_delta"] < -0.005 else
            "stable"
        )

    print("\n=== SUMMARY ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    with open(os.path.join(args.output_dir, "two_clock_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    np.savez_compressed(
        os.path.join(args.output_dir, "two_clock_trajectories.npz"),
        res_L=res_L.astype(np.float32),
        res_H=res_H.astype(np.float32),
        cell_acc=cell_acc.astype(np.float32),
        commit_times=ct.astype(np.int16),
        qhalt=qhalt.astype(np.float32),
        max_jump_frac=max_jump_frac.astype(np.float32),
        final_solved=final_solved,
    )
    print(f"\nSaved -> {args.output_dir}/two_clock_summary.json + two_clock_trajectories.npz")


if __name__ == "__main__":
    main()
