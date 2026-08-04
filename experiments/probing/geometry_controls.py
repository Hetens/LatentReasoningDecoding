"""
Clustering controls and a convergence statistic for the latent geometry
(NeurIPS 29918 discussion period).

Reviewer Vy5n Q2 asks whether the cluster structure reported in Table 1 is
explained by trivial cell properties, and MKju W4 asks for a quantitative
replacement for the visual "geometry converges toward solved cells" claim.

Protocol matches Table 1: cell states from the final ACT step, pooled over
all (T, i), reduced to 50 PCA dimensions, K-Means with k=20. For every cell
property we report
  * raw purity,
  * a label-permutation baseline (the property labels shuffled), which is
    the empirical chance level given the realised cluster-size distribution,
  * purity under a class-balanced resample of that property.

Properties: candidate-set size |Sc| after constraint propagation, given
versus blank cell, solution digit, grid box, grid row, and the outer cycle T
of the state itself.

Convergence statistic: mean distance from cells with |Sc| > 1 to the
centroid of the |Sc| = 1 (propagation-determined) cells, per (T, i), scaled
by the RMS norm of the cloud so cycles are comparable, with puzzle-level
bootstrap CIs.

Usage (from repo root):
    python -m experiments.probing.geometry_controls \
        --config trm_base/config_pretrain_paper.yml \
        --checkpoint "checkpoints/.../step_65100.pt" \
        --data-path data/sudoku-extreme-1k-aug-1000 \
        --output-dir results/probing/geometry_controls \
        --n-puzzles 400
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
from experiments.probing.activation_patching_sweep import (  # noqa: E402
    _warmup_carry,
    _capture_all_snapshots,
    _masked_mean,
    _per_cell_correct,
)
from experiments.probing.activation_patching import per_cell_cross_entropy  # noqa: E402
from experiments.probing.candidate_sets import (  # noqa: E402
    inputs_to_puzzle_string,
    compute_cp_candidates,
)


def _purity(prop: np.ndarray, clusters: np.ndarray) -> float:
    vals = []
    for c in np.unique(clusters):
        m = clusters == c
        _, counts = np.unique(prop[m], return_counts=True)
        vals.append(counts.max() / counts.sum())
    return float(np.mean(vals)) if vals else float("nan")


def _balanced_indices(prop: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Indices of a class-balanced resample of *prop* (without replacement)."""
    classes, counts = np.unique(prop, return_counts=True)
    per = int(counts.min())
    keep = []
    for c in classes:
        idx = np.where(prop == c)[0]
        keep.append(rng.choice(idx, per, replace=False))
    return np.concatenate(keep)


def main() -> None:
    ap = argparse.ArgumentParser(description="Clustering controls for the latent geometry.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data-path", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--n-puzzles", type=int, default=400)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--pca-dim", type=int, default=50)
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--blank-only", action="store_true",
                    help="Restrict the point cloud to cells the model must fill in "
                         "(givens excluded), so purities cannot be carried by copied inputs.")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    print("Loading model ...")
    model = load_trm_model(args.config, args.checkpoint, device, args.data_path, args.split)
    model.eval()
    inner = model.inner
    cfg = model.config
    n_T, n_I = cfg.H_cycles, cfg.L_cycles

    raw = load_test_data(args.data_path, args.split, max_examples=args.n_puzzles)
    n = len(raw["inputs"])
    print(f"{n} puzzles, {n_T}x{n_I} states per puzzle")

    # ---- cell properties -------------------------------------------------
    set_sizes = np.zeros((n, 81), dtype=np.int32)
    for k in tqdm(range(n), desc="candidate sets"):
        cands = compute_cp_candidates(inputs_to_puzzle_string(raw["inputs"][k]))
        set_sizes[k] = np.array([max(1, len(s)) for s in cands], dtype=np.int32)
    is_given = (raw["inputs"][:n] > 1).astype(np.int32)
    digits = (raw["labels"][:n].astype(np.int32) - 1)
    rows = np.tile(np.arange(81) // 9, (n, 1))
    cols = np.tile(np.arange(81) % 9, (n, 1))
    boxes = (rows // 3) * 3 + (cols // 3)

    # ---- collect states --------------------------------------------------
    states = np.zeros((n, n_T, n_I, 81, cfg.hidden_size), dtype=np.float32)
    solved = np.zeros(n, dtype=bool)
    for s in tqdm(range(0, n, args.batch_size), desc="forward"):
        e = min(s + args.batch_size, n)
        b = {
            "inputs": torch.from_numpy(raw["inputs"][s:e].astype(np.int32)).to(device),
            "labels": torch.from_numpy(raw["labels"][s:e].astype(np.int32)).to(device),
            "puzzle_identifiers": torch.from_numpy(
                raw["puzzle_identifiers"][s:e].astype(np.int32)).to(device),
        }
        carry = _warmup_carry(inner, b, device)
        snaps, logits = _capture_all_snapshots(inner, carry, b)
        mask = b["labels"] != -100
        acc = _masked_mean(_per_cell_correct(logits.float(), b["labels"]), mask)
        solved[s:e] = (acc >= 1.0).cpu().numpy()
        for T in range(n_T):
            for i in range(n_I):
                states[s:e, T, i] = snaps[(T, i)].float().cpu().numpy()

    print(f"clean run solves {solved.sum()}/{n} puzzles completely")

    # ---- pooled point cloud over all (T, i) ------------------------------
    D = cfg.hidden_size
    X = states.transpose(1, 2, 0, 3, 4).reshape(n_T * n_I * n * 81, D)
    cell_keep = np.ones(n * 81, dtype=bool)
    if args.blank_only:
        cell_keep = (is_given.ravel() == 0)
        print(f"blank-only: keeping {cell_keep.sum()}/{len(cell_keep)} cells per state")
        X = X[np.tile(cell_keep, n_T * n_I)]
    props: Dict[str, np.ndarray] = {}
    tile = lambda a: np.tile(a.ravel()[cell_keep], n_T * n_I)  # noqa: E731
    props["set_size"] = tile(set_sizes)
    props["given_or_blank"] = tile(is_given)
    props["solution_digit"] = tile(digits)
    props["box"] = tile(boxes)
    props["row"] = tile(rows)
    props["outer_cycle_T"] = np.concatenate(
        [np.full(int(cell_keep.sum()), T + 1) for T in range(n_T) for _ in range(n_I)])

    print(f"clustering {X.shape[0]} points in {args.pca_dim}D ...")
    pca = PCA(n_components=args.pca_dim, random_state=args.seed)
    Z = pca.fit_transform(StandardScaler().fit_transform(X))
    ev2 = float(pca.explained_variance_ratio_[:2].sum())
    evk = float(pca.explained_variance_ratio_.sum())
    km = KMeans(n_clusters=args.k, n_init=10, random_state=args.seed).fit(Z)
    clusters = km.labels_

    results = {
        "n_puzzles": int(n), "n_points": int(X.shape[0]), "k": args.k,
        "pca_dim": args.pca_dim, "pca_var_2": ev2, "pca_var_k": evk,
        "solved_fraction": float(solved.mean()),
        "blank_only": bool(args.blank_only),
        "purities": {},
    }
    for name, prop in props.items():
        raw_p = _purity(prop, clusters)
        shuffled = prop.copy()
        rng.shuffle(shuffled)
        perm_p = _purity(shuffled, clusters)
        keep = _balanced_indices(prop, rng)
        kmb = KMeans(n_clusters=args.k, n_init=10, random_state=args.seed).fit(Z[keep])
        bal_p = _purity(prop[keep], kmb.labels_)
        bal_shuf = prop[keep].copy()
        rng.shuffle(bal_shuf)
        bal_perm = _purity(bal_shuf, kmb.labels_)
        results["purities"][name] = {
            "raw": raw_p, "label_permuted": perm_p,
            "balanced": bal_p, "balanced_label_permuted": bal_perm,
            "n_classes": int(len(np.unique(prop))),
        }
        print(f"  {name:>16}: raw {raw_p:.3f} (perm {perm_p:.3f})   "
              f"balanced {bal_p:.3f} (perm {bal_perm:.3f})")

    # ---- convergence toward the propagation-determined cells -------------
    # Per (T, i): mean distance from |Sc|>1 cells to the |Sc|=1 centroid,
    # scaled by the RMS norm of the centred cloud at that state.
    conv = []
    sc = set_sizes
    for T in range(n_T):
        for i in range(n_I):
            S = states[:, T, i]                       # (n, 81, D)
            det = S[sc == 1]
            amb_mask = sc > 1
            centroid = det.mean(axis=0)
            scale = np.sqrt(((S.reshape(-1, D) - S.reshape(-1, D).mean(0)) ** 2).sum(1).mean())
            per_puzzle = np.array([
                np.linalg.norm(S[p][amb_mask[p]] - centroid, axis=1).mean() / scale
                if amb_mask[p].any() else np.nan
                for p in range(n)
            ])
            vals = per_puzzle[~np.isnan(per_puzzle)]
            boot = np.array([
                vals[rng.integers(0, len(vals), len(vals))].mean() for _ in range(2000)
            ])
            conv.append({
                "T": T + 1, "i": i + 1,
                "mean_scaled_distance": float(vals.mean()),
                "ci": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
            })
            print(f"  (T={T+1},i={i+1}) distance to solved-cell centroid "
                  f"{conv[-1]['mean_scaled_distance']:.4f} "
                  f"[{conv[-1]['ci'][0]:.4f}, {conv[-1]['ci'][1]:.4f}]")
    results["convergence_to_solved_centroid"] = conv

    suffix = "_blank_only" if args.blank_only else ""
    out = os.path.join(args.output_dir, f"geometry_controls{suffix}.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
