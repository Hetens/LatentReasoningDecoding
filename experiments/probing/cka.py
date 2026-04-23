"""
Linear Centered Kernel Alignment (CKA) for comparing latent geometries
across checkpoints or recursion indices.

    CKA(X, Y) = ||Y^T X||_F^2 / (||X^T X||_F  ·  ||Y^T Y||_F)

where X, Y are column-centered activation matrices of shape (n, d).

Usage (from repo root):
    python -m experiments.probing.cka \
        --file-a results/probing/activations/z_L_act16.pt \
        --file-b results/probing/activations_v2/z_L_act16.pt \
        --output-dir results/probing/cka \
        --latent z_L
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Tuple

import numpy as np
import torch
from tqdm import tqdm


# ---------------------------------------------------------------------------
# CKA
# ---------------------------------------------------------------------------

def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Compute linear CKA between two column-centered activation matrices.

    Args:
        X: (n, d1) activation matrix.
        Y: (n, d2) activation matrix (same n).

    Returns:
        CKA value in [0, 1].
    """
    X = X.astype(np.float64)
    Y = Y.astype(np.float64)
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)

    YtX = Y.T @ X
    XtX = X.T @ X
    YtY = Y.T @ Y

    num = np.linalg.norm(YtX, "fro") ** 2
    denom = np.linalg.norm(XtX, "fro") * np.linalg.norm(YtY, "fro")
    return float(num / denom) if denom > 0 else 0.0


def bootstrap_cka_ci(
    X: np.ndarray,
    Y: np.ndarray,
    n_puzzles: int,
    cells_per_puzzle: int = 81,
    n_resamples: int = 10_000,
    seed: int = 42,
    alpha: float = 0.05,
) -> Tuple[float, float]:
    """Puzzle-level bootstrap CI for linear CKA.

    Rows of X and Y are ordered as ``(puzzle0_cell0, puzzle0_cell1, …,
    puzzle1_cell0, …)``.  The bootstrap resamples entire puzzles.
    """
    rng = np.random.default_rng(seed)
    cka_samples = np.empty(n_resamples)

    for i in range(n_resamples):
        puz_idx = rng.integers(0, n_puzzles, size=n_puzzles)
        row_idx = np.concatenate(
            [np.arange(p * cells_per_puzzle, (p + 1) * cells_per_puzzle) for p in puz_idx]
        )
        cka_samples[i] = linear_cka(X[row_idx], Y[row_idx])

    lo = float(np.percentile(cka_samples, 100 * alpha / 2))
    hi = float(np.percentile(cka_samples, 100 * (1 - alpha / 2)))
    return lo, hi


# ---------------------------------------------------------------------------
# Self-CKA heatmap (across recursion indices within one checkpoint)
# ---------------------------------------------------------------------------

def compute_self_cka_grid(
    z: np.ndarray,
    is_z_L: bool = True,
) -> np.ndarray:
    """CKA between every pair of (T, i) recursion steps from one checkpoint.

    Args:
        z: Loaded tensor reshaped to numpy.
           z_L: (N, H, L, 81, D)  or  z_H: (N, H, 81, D).

    Returns:
        grid: (K, K) CKA matrix where K = H*L (z_L) or H (z_H), with
              rows/cols ordered by flattened (T, i) index.
    """
    if is_z_L:
        N, H, L, C, D = z.shape
        K = H * L
        flat = z.reshape(N, K, C, D)
    else:
        N, H, C, D = z.shape
        K = H
        flat = z.reshape(N, K, C, D)

    # Stack all (puzzle, cell) pairs per index → (N*C, D).
    mats = [flat[:, k, :, :].reshape(-1, D) for k in range(K)]
    grid = np.eye(K)
    for a in tqdm(range(K), desc="CKA grid"):
        for b in range(a + 1, K):
            val = linear_cka(mats[a], mats[b])
            grid[a, b] = grid[b, a] = val
    return grid


# ---------------------------------------------------------------------------
# Cross-checkpoint CKA (matched (T, i) pairs)
# ---------------------------------------------------------------------------

def compute_cross_cka(
    z_a: np.ndarray,
    z_b: np.ndarray,
    is_z_L: bool = True,
) -> np.ndarray:
    """CKA between matched (T, i) pairs from two checkpoints.

    Both arrays must cover the same puzzles in the same order.

    Returns:
        cka_vec: (K,) array of CKA values per (T, i) index.
    """
    if is_z_L:
        N, H, L, C, D = z_a.shape
        K = H * L
        flat_a = z_a.reshape(N, K, C, D)
        flat_b = z_b.reshape(N, K, C, D)
    else:
        N, H, C, D = z_a.shape
        K = H
        flat_a = z_a.reshape(N, K, C, D)
        flat_b = z_b.reshape(N, K, C, D)

    cka_vec = np.empty(K)
    for k in tqdm(range(K), desc="Cross CKA"):
        Xa = flat_a[:, k, :, :].reshape(-1, D)
        Xb = flat_b[:, k, :, :].reshape(-1, D)
        cka_vec[k] = linear_cka(Xa, Xb)
    return cka_vec


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Compute CKA between TRM latent representations.")
    sub = parser.add_subparsers(dest="cmd")

    # Self-CKA (within one checkpoint)
    p_self = sub.add_parser("self", help="Self-CKA heatmap across (T,i) pairs.")
    p_self.add_argument("--file", required=True, help="Path to z_L_act*.pt or z_H_act*.pt")
    p_self.add_argument("--latent", choices=["z_L", "z_H"], default="z_L")
    p_self.add_argument("--output-dir", required=True)
    p_self.add_argument("--max-examples", type=int, default=None)

    # Cross-checkpoint CKA
    p_cross = sub.add_parser("cross", help="CKA between matched (T,i) from two checkpoints.")
    p_cross.add_argument("--file-a", required=True)
    p_cross.add_argument("--file-b", required=True)
    p_cross.add_argument("--latent", choices=["z_L", "z_H"], default="z_L")
    p_cross.add_argument("--output-dir", required=True)
    p_cross.add_argument("--max-examples", type=int, default=None)

    args = parser.parse_args()
    if args.cmd is None:
        parser.print_help()
        return

    os.makedirs(args.output_dir, exist_ok=True)
    is_z_L = (args.latent == "z_L")

    if args.cmd == "self":
        z = torch.load(args.file, map_location="cpu", weights_only=True).numpy()
        if args.max_examples and args.max_examples < z.shape[0]:
            z = z[: args.max_examples]
        print(f"Self-CKA on {args.latent}  shape={z.shape}")
        grid = compute_self_cka_grid(z, is_z_L)
        out_path = os.path.join(args.output_dir, f"self_cka_{args.latent}.npy")
        np.save(out_path, grid)
        print(f"Saved {out_path}  shape={grid.shape}")

    elif args.cmd == "cross":
        z_a = torch.load(args.file_a, map_location="cpu", weights_only=True).numpy()
        z_b = torch.load(args.file_b, map_location="cpu", weights_only=True).numpy()
        n = min(z_a.shape[0], z_b.shape[0])
        if args.max_examples:
            n = min(n, args.max_examples)
        z_a, z_b = z_a[:n], z_b[:n]
        print(f"Cross-CKA on {args.latent}  n={n}")
        vec = compute_cross_cka(z_a, z_b, is_z_L)
        out_path = os.path.join(args.output_dir, f"cross_cka_{args.latent}.npy")
        np.save(out_path, vec)
        print(f"Saved {out_path}  shape={vec.shape}")

        n_puzzles = n
        print("\nWith bootstrap CIs (puzzle-level):")
        if is_z_L:
            N, H, L, C, D = z_a.shape
            K = H * L
        else:
            N, H, C, D = z_a.shape
            K = H
        for k in range(K):
            if is_z_L:
                Xa = z_a[:, k // L, k % L].reshape(-1, D)
                Xb = z_b[:, k // L, k % L].reshape(-1, D)
            else:
                Xa = z_a[:, k].reshape(-1, D)
                Xb = z_b[:, k].reshape(-1, D)
            lo, hi = bootstrap_cka_ci(Xa, Xb, n_puzzles, cells_per_puzzle=C)
            print(f"  k={k}: CKA={vec[k]:.4f}  [{lo:.4f}, {hi:.4f}]")

    print("Done.")


if __name__ == "__main__":
    main()
