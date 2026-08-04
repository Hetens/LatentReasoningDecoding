"""
Reconcile three CKA computations that currently disagree for the cycle 1
to 3 block of the 18x18 self-CKA grid:

  (a) paper text and abstract:                        0.2 - 0.4
  (b) the grid stored in results/probing/cka/:        ~0.637
  (c) a fresh extraction at 400 puzzles x 8 cells:    ~0.903

The extraction path has already been ruled out by inspection (the paper's
ACT loop resets the carry only on step 1, then runs 16 segments
continuously and captures the 16th, which is what _warmup_carry plus
_capture_all_snapshots does). The remaining hypothesis is sample size:
linear CKA is biased upward when the row count is not large relative to
the 512 feature dimensions, and (b) uses 405,000 rows against (c)'s 3,200.

This script sweeps the row count directly and reports the resulting block
means, so the bias curve is measured rather than assumed.

Efficiency notes: the naive pairwise loop recomputes X^T X for every pair.
Here each state is centered once, its self-Gram Frobenius norm is cached,
and only the cross term is computed per pair, on GPU. Output is flushed
so a timeout still leaves a partial record.

Usage (from repo root):
    python -u -m experiments.probing.reconcile_cka \
        --activations results/probing/activations/z_L_act16.pt \
        --stored-grid results/probing/cka/self_cka_z_L.npy
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_TRM_BASE = os.path.join(_PROJECT_ROOT, "trm_base")
for p in (_PROJECT_ROOT, _TRM_BASE):
    if p not in sys.path:
        sys.path.insert(0, p)


def log(*a):
    print(*a, flush=True)


def block_means(grid: np.ndarray, n_T: int = 3) -> dict:
    per = grid.shape[0] // n_T

    def blk(a, b):
        return float(grid[a * per:(a + 1) * per, b * per:(b + 1) * per].mean())

    return {
        "within": round(float(np.mean([blk(t, t) for t in range(n_T)])), 3),
        "c1_c2": round(blk(0, 1), 3),
        "c1_c3": round(blk(0, n_T - 1), 3),
        "c2_c3": round(blk(1, n_T - 1), 3),
        "min": round(float(grid.min()), 3),
    }


@torch.no_grad()
def cka_grid_gpu(states: torch.Tensor, device: torch.device) -> np.ndarray:
    """states: (K, n, d) on CPU. Returns (K, K) linear CKA grid.

    Uses CKA(X,Y) = ||Yc^T Xc||_F^2 / (||Xc^T Xc||_F * ||Yc^T Yc||_F),
    caching the per-state self terms so each pair costs one matmul.
    """
    K = states.shape[0]
    Xs, self_norms = [], []
    for k in range(K):
        Xc = states[k].to(device, torch.float32)
        Xc = Xc - Xc.mean(dim=0, keepdim=True)
        Xs.append(Xc)
        g = Xc.T @ Xc
        self_norms.append(torch.linalg.matrix_norm(g.double(), ord="fro"))

    grid = np.eye(K)
    for a in range(K):
        for b in range(a + 1, K):
            cross = Xs[b].T @ Xs[a]                      # (d, d)
            num = (torch.linalg.matrix_norm(cross.double(), ord="fro") ** 2)
            den = self_norms[a] * self_norms[b]
            v = float((num / den).item()) if den > 0 else 0.0
            grid[a, b] = grid[b, a] = v
    del Xs
    torch.cuda.empty_cache() if device.type == "cuda" else None
    return grid


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--activations", required=True)
    ap.add_argument("--stored-grid", required=True)
    ap.add_argument("--output", default="results/probing/cka/reconcile_cka.json")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    rng = np.random.default_rng(args.seed)
    out = {}

    stored = np.load(args.stored_grid)
    out["stored_grid"] = block_means(stored)
    log("=== (b) stored grid ===")
    log("   ", out["stored_grid"])

    log("\nLoading activations ...")
    t0 = time.time()
    z = torch.load(args.activations, map_location="cpu")
    if isinstance(z, dict):
        z = z.get("z_L", next(iter(z.values())))
    log(f"  shape {tuple(z.shape)} dtype {z.dtype}  ({time.time()-t0:.0f}s)")

    # The stored tensor is (N, H, L, S, D): puzzles FIRST. Getting this
    # order wrong silently yields K = N*H "states" instead of 18, so assert.
    N, H, L, S, D = z.shape
    assert (H, L, D) == (3, 6, 512), f"unexpected layout {tuple(z.shape)}"
    K = H * L
    log(f"  parsed as N={N} puzzles, {H}x{L}={K} states, S={S}, D={D}")

    # Sweep the row count to measure the small-sample bias directly.
    schemes = [
        ("8 cells x 400 puzzles   (my regime run)", 400, 8),
        ("8 cells x 2000 puzzles", 2000, 8),
        ("81 cells x 200 puzzles", 200, None),
        ("81 cells x 1000 puzzles", 1000, None),
        ("81 cells x 5000 puzzles (paper)", 5000, None),
    ]
    out["schemes"] = {}
    for name, n_puz, n_cells in schemes:
        n_puz = min(n_puz, N)
        pidx = torch.from_numpy(rng.choice(N, n_puz, replace=False))
        if n_cells is not None:
            cidx = rng.integers(0, S, size=(n_puz, n_cells))
            rows = torch.from_numpy(np.repeat(np.arange(n_puz), n_cells))
            cols = torch.from_numpy(cidx.reshape(-1))
        # Slice state by state to avoid materialising a permuted copy of
        # the whole 7.5 GB tensor.
        mats = []
        for T in range(H):
            for i in range(L):
                sub = z[pidx, T, i]                     # (n_puz, S, D)
                mats.append(sub.reshape(-1, D) if n_cells is None
                            else sub[rows, cols, :])
        mat = torch.stack(mats)                          # (18, n, D)
        assert mat.shape[0] == K, f"expected {K} states, got {mat.shape[0]}"
        t0 = time.time()
        g = cka_grid_gpu(mat, device)
        bm = block_means(g)
        bm["n_rows"] = int(mat.shape[1])
        out["schemes"][name] = bm
        log(f"\n=== {name}  (n_rows={mat.shape[1]}) ===")
        log("   ", bm, f"[{time.time()-t0:.0f}s]")
        del mat, mats

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    log(f"\nSaved -> {args.output}")

    log("\n=== READING ===")
    log("  If c1_c3 falls toward ~0.64 as n_rows grows, the fresh-extraction")
    log("  value was small-sample bias and the regime split must be rerun")
    log("  with all 81 cells. The stored-vs-paper gap (0.637 vs 0.2-0.4) is")
    log("  independent of sampling and remains a discrepancy in the paper.")


if __name__ == "__main__":
    main()
