"""
Recompute the paper's geometry results separately for the two dynamical
regimes found by two_clock_trajectory.py.

Motivation: the submission's CKA blocks, PCA separation and cluster
purities are measured on a pooled population that is ~70% already
converged by the analysed segment. If the "three geometrically distinct
phases" claim is an artifact of averaging a converged sub-population with
a wandering one, it has to be revised. This script tests that directly.

For each regime (ends solved / ends unsolved) it recomputes:
  - the 18x18 self-CKA grid over (T, i) at the final ACT step;
  - PCA explained variance and the T-separation along PC1;
  - K-Means cluster purity w.r.t. outer cycle T and candidate-set size,
    with an |Sc|-balanced control (a reviewer request in its own right).

Usage (from repo root):
    python -m experiments.probing.regime_split_geometry \
        --config trm_base/config_pretrain_paper.yml \
        --checkpoint "checkpoints/.../step_65100.pt" \
        --data-path data/sudoku-extreme-1k-aug-1000 \
        --output-dir results/probing/regime_geometry \
        --n-puzzles 400 --cells-per-puzzle 8
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
from experiments.probing.activation_patching_sweep import (  # noqa: E402
    _warmup_carry,
    _capture_all_snapshots,
)
from experiments.probing.cka import linear_cka  # noqa: E402
from experiments.probing.candidate_sets import (  # noqa: E402
    inputs_to_puzzle_string,
    compute_cp_candidates,
)


def _candidate_set_sizes(inputs: np.ndarray) -> np.ndarray:
    """(B, 81) candidate-set size per cell, via the same CP solver the
    paper uses for its |Sc| labels.

    Sudoku only: the CP solver parses a 9x9 grid. Callers must gate on
    seq_len, since the cross-task runs (Maze, Einstein) have no |Sc|.
    """
    out = np.zeros(inputs.shape[:2], dtype=np.int16)
    for b in range(inputs.shape[0]):
        cands = compute_cp_candidates(inputs_to_puzzle_string(inputs[b]))
        out[b] = np.array([len(c) for c in cands], dtype=np.int16)
    return out


def _purity(labels: np.ndarray, clusters: np.ndarray) -> float:
    total = 0
    for c in np.unique(clusters):
        m = clusters == c
        if m.sum() == 0:
            continue
        vals, counts = np.unique(labels[m], return_counts=True)
        total += counts.max()
    return total / len(labels)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data-path", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--n-puzzles", type=int, default=400)
    ap.add_argument("--cells-per-puzzle", type=int, default=8,
                    help="cells sampled per puzzle; 0 or 81 means all cells "
                         "(needed for CKA values comparable to the paper, "
                         "which uses all 81)")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--pca-dim", type=int, default=50)
    ap.add_argument("--act-step", choices=["first", "last"], default="last",
                    help="which ACT segment to analyse; the paper's CKA grid "
                         "turned out to be ACT step 1 (first) while its probes "
                         "are step 16 (last), so both are worth reporting")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA

    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    print("Loading model ...")
    model = load_trm_model(args.config, args.checkpoint, device, args.data_path, args.split)
    model.eval()
    inner, cfg = model.inner, model.config
    n_T, n_I = cfg.H_cycles, cfg.L_cycles
    targets = [(T, i) for T in range(n_T) for i in range(n_I)]

    raw = load_test_data(args.data_path, args.split, max_examples=args.n_puzzles)
    N = min(args.n_puzzles, len(raw["inputs"]))
    idx_all = np.arange(N)

    # The CP solver behind |Sc| only understands a 9x9 Sudoku grid.
    has_sc = raw["inputs"].shape[1] == 81
    if not has_sc:
        print(f"seq_len {raw['inputs'].shape[1]} is not Sudoku: skipping |Sc| labels, "
              f"reporting cycle-T geometry only")

    # states[(T,i)] -> list of (cells_sampled, 512) blocks; parallel arrays for labels
    states: Dict = {t: [] for t in targets}
    solved_flags: List[np.ndarray] = []
    sc_sizes: List[np.ndarray] = []

    print(f"Extracting {len(targets)} states for {N} puzzles ...")
    for start in tqdm(range(0, N, args.batch_size), desc="batches"):
        sel = idx_all[start:start + args.batch_size]
        batch = {
            k: torch.from_numpy(raw[k][sel].astype(np.int32)).to(device)
            for k in ("inputs", "labels", "puzzle_identifiers")
        }
        labels = batch["labels"]
        mask = labels != -100

        if args.act_step == "first":
            # Geometry from ACT step 1: no warm-up, run the first segment
            # from the init state.
            carry = inner.empty_carry(len(sel))
            carry.z_H = carry.z_H.to(device)
            carry.z_L = carry.z_L.to(device)
            carry = inner.reset_carry(
                torch.ones(len(sel), dtype=torch.bool, device=device), carry)
            snaps, _ = _capture_all_snapshots(inner, carry, batch)
            # The regime label must still be the FINAL outcome, not "solved
            # after one segment", so take the solved flag from the full run.
            final_carry = _warmup_carry(inner, batch, device)
            _, clean_logits = _capture_all_snapshots(inner, final_carry, batch)
        else:
            carry = _warmup_carry(inner, batch, device)
            snaps, clean_logits = _capture_all_snapshots(inner, carry, batch)

        pred = clean_logits.float().argmax(dim=-1)
        per_puzzle_solved = ((pred == labels.long()) | ~mask).all(dim=1).cpu().numpy()

        # Ground-truth candidate-set sizes for the sampled cells. |Sc| is a
        # Sudoku construct; on Maze/Einstein leave it constant so the T
        # analysis still runs and the |Sc| columns are reported as n/a.
        if has_sc:
            sc_size = _candidate_set_sizes(raw["inputs"][sel])   # (B, 81)
        else:
            sc_size = np.zeros(labels.shape, dtype=np.int16)

        B, S_ = len(sel), labels.shape[1]
        if args.cells_per_puzzle in (0, S_):
            rows = np.repeat(np.arange(B), S_)
            cols = np.tile(np.arange(S_), B)
        else:
            cell_idx = rng.integers(0, S_, size=(B, args.cells_per_puzzle))
            rows = np.repeat(np.arange(B), args.cells_per_puzzle)
            cols = cell_idx.reshape(-1)
        n_per = len(cols) // B

        for t in targets:
            z = snaps[t].float().cpu().numpy()       # (B, 81, 512)
            states[t].append(z[rows, cols, :])
        solved_flags.append(np.repeat(per_puzzle_solved, n_per))
        sc_sizes.append(sc_size[rows, cols])

    solved = np.concatenate(solved_flags)
    scs = np.concatenate(sc_sizes)
    X = {t: np.concatenate(states[t]) for t in targets}
    n_pts = len(solved)
    print(f"\n{n_pts} cell-states; solved-regime {solved.mean():.1%}, "
          f"unsolved {1-solved.mean():.1%}")

    results = {"n_points": int(n_pts), "act_step": args.act_step,
               "solved_frac": float(solved.mean()), "regimes": {}}

    # "pooled" is the control: it must reproduce the paper's published
    # numbers (within-cycle 0.7-0.9, cycle 1 to 3 dropping to 0.2-0.4).
    # Without it, a per-regime difference could just mean our extraction
    # differs from the paper's rather than revealing a mixture effect.
    for regime, m in (("pooled", np.ones_like(solved, dtype=bool)),
                      ("solved", solved), ("unsolved", ~solved)):
        if m.sum() < 200:
            print(f"[{regime}] too few points ({m.sum()}), skipping")
            continue
        print(f"\n===== regime: {regime}  (n={int(m.sum())}) =====")

        # ---- 18x18 self-CKA ----
        grid = np.zeros((len(targets), len(targets)))
        for a, ta in enumerate(targets):
            for b, tb in enumerate(targets):
                if b < a:
                    grid[a, b] = grid[b, a]
                else:
                    grid[a, b] = linear_cka(X[ta][m], X[tb][m])

        def block(Ta, Tb):
            ia = [k for k, (T, _) in enumerate(targets) if T == Ta]
            ib = [k for k, (T, _) in enumerate(targets) if T == Tb]
            return float(grid[np.ix_(ia, ib)].mean())

        within = float(np.mean([block(T, T) for T in range(n_T)]))
        cross13 = block(0, n_T - 1)
        print(f"  CKA within-cycle {within:.3f}   between cycle 1 and {n_T}: {cross13:.3f}")

        # ---- PCA + clustering in PCA space ----
        allX = np.concatenate([X[t][m] for t in targets])
        Tlab = np.concatenate([np.full(m.sum(), T) for (T, _i) in targets])
        Slab = np.tile(scs[m], len(targets))

        pca = PCA(n_components=min(args.pca_dim, allX.shape[1]), random_state=args.seed)
        Z = pca.fit_transform(allX)
        ev2 = float(pca.explained_variance_ratio_[:2].sum())
        evk = float(pca.explained_variance_ratio_.sum())

        km = KMeans(n_clusters=20, n_init=10, random_state=args.seed).fit(Z)
        pT = _purity(Tlab, km.labels_)
        pS = _purity(Slab, km.labels_) if has_sc else float("nan")

        print(f"  PCA: 2 PCs {ev2:.1%} var, {args.pca_dim} PCs {evk:.1%}")

        if has_sc:
            # |Sc|-balanced control: equal counts per |Sc| stratum. On Sudoku
            # this is what shows the raw |Sc| purity is mostly class imbalance
            # while the cycle-T purity is not.
            vals, counts = np.unique(Slab, return_counts=True)
            per = int(min(counts.min(), 2000))
            keep = np.concatenate([
                rng.choice(np.where(Slab == v)[0], per, replace=False) for v in vals
            ])
            kmb = KMeans(n_clusters=20, n_init=10, random_state=args.seed).fit(Z[keep])
            pTb = _purity(Tlab[keep], kmb.labels_)
            pSb = _purity(Slab[keep], kmb.labels_)
            print(f"  purity T={pT:.3f} (chance {1/n_T:.3f})   |Sc|={pS:.3f} (chance {1/9:.3f})")
            print(f"  |Sc|-balanced: T={pTb:.3f}  |Sc|={pSb:.3f}  ({per}/stratum)")
        else:
            pTb = pSb = float("nan")
            print(f"  purity T={pT:.3f} (chance {1/n_T:.3f})   |Sc|=n/a (not Sudoku)")

        results["regimes"][regime] = {
            "n": int(m.sum()),
            "cka_within_cycle": within,
            "cka_cycle1_to_last": cross13,
            "cka_grid": grid.tolist(),
            "pca_var_2": ev2,
            "pca_var_k": evk,
            "purity_T": pT,
            "purity_Sc": pS,
            "purity_T_balanced": pTb,
            "purity_Sc_balanced": pSb,
            "chance_T": 1.0 / n_T,
            "chance_Sc": 1.0 / 9 if has_sc else None,
        }

    out = os.path.join(args.output_dir, f"regime_geometry_{args.act_step}.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved -> {out}")

    R = results["regimes"]
    if {"pooled", "solved", "unsolved"} <= set(R):
        print("\n=== VERDICT ===")
        print(f"{'regime':>10} {'CKA within':>12} {'CKA 1->last':>13}")
        for k in ("pooled", "solved", "unsolved"):
            print(f"{k:>10} {R[k]['cka_within_cycle']:>12.3f} {R[k]['cka_cycle1_to_last']:>13.3f}")
        print("\n  Paper reports pooled: within 0.7-0.9, cycle 1 to 3 = 0.2-0.4.")
        print("  If pooled here reproduces that while each regime alone does not,")
        print("  the three-phase structure is a mixture effect.")
        print("  If pooled here does NOT reproduce it, our extraction differs from")
        print("  the paper's and the comparison is not yet valid.")


if __name__ == "__main__":
    main()
