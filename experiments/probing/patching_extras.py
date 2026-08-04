"""
Discussion-period follow-ups to the 18-index patching sweep (NeurIPS 29918).

Four analyses, each writing its own JSON into --output-dir:

  zh        Patch z_H (the state the output head reads) at the end of each
            outer cycle, with the same two interventions and metrics as the
            z_L sweep. Reviewer Vy5n Q5.

  shufdist  Repeat the within-puzzle shuffle with many independent
            permutations per puzzle at selected (T, i), to show whether the
            negative dCE at mid-recursion is a property of the intervention
            or of one lucky permutation draw. Reviewer Vy5n Q4.

  matched   Re-run the cross-puzzle swap with donors matched to originals on
            puzzle difficulty (backtracking requirement and number of givens)
            instead of drawn at random, to remove the donor-difficulty
            confound. Reviewer hi4P.

  stratify  Clean-run error analysis: per-cell accuracy stratified by
            candidate-set size |Sc|, by solution digit and by given/blank,
            plus puzzle-level solve rate against classical-solver difficulty.
            Reviewer hi4P.

Metrics match the sweep exactly: dCE is the change in mean per-cell
cross-entropy (patched minus clean, nats, positive = worse) and dAcc the
change in mean per-cell accuracy (negative = worse), both averaged over
puzzles with puzzle-level bootstrap 95% CIs, computed in float32.

Usage (from repo root):
    python -m experiments.probing.patching_extras \
        --config trm_base/config_pretrain_paper.yml \
        --checkpoint "checkpoints/.../step_65100.pt" \
        --data-path data/sudoku-extreme-1k-aug-1000 \
        --output-dir results/probing/patching_extras \
        --do zh,shufdist,matched,stratify
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import torch
from tqdm import tqdm

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_TRM_BASE = os.path.join(_PROJECT_ROOT, "trm_base")
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
if _TRM_BASE not in sys.path:
    sys.path.insert(0, _TRM_BASE)

from trm import (  # noqa: E402
    TinyRecursiveReasoningModel_ACTV1InnerCarry,
    TinyRecursiveReasoningModel_ACTV1_Inner,
)

from experiments.probing.extract_activations import (  # noqa: E402
    load_trm_model,
    load_test_data,
)
from experiments.probing.activation_patching import (  # noqa: E402
    _inner_forward_patched,
    per_cell_cross_entropy,
)
from experiments.probing.activation_patching_sweep import (  # noqa: E402
    _warmup_carry,
    _capture_all_snapshots,
    _masked_mean,
    _per_cell_correct,
    _bootstrap_mean_ci,
)
from experiments.probing.candidate_sets import (  # noqa: E402
    inputs_to_puzzle_string,
    compute_cp_candidates,
)
from sudoku.util import sudoku_metrics  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def _capture_zh_snapshots(
    inner: TinyRecursiveReasoningModel_ACTV1_Inner,
    carry: TinyRecursiveReasoningModel_ACTV1InnerCarry,
    batch: Dict[str, torch.Tensor],
) -> Tuple[Dict[int, torch.Tensor], torch.Tensor]:
    """Run the final ACT step once, capturing z_H after each outer-cycle
    update. The capture point matches where _inner_forward_patched applies
    patch_z_H."""
    cfg = inner.config
    seq_info = dict(cos_sin=inner.rotary_emb() if hasattr(inner, "rotary_emb") else None)
    input_emb = inner._input_embeddings(batch["inputs"], batch["puzzle_identifiers"])

    snaps: Dict[int, torch.Tensor] = {}
    z_H, z_L = carry.z_H, carry.z_L
    for T in range(cfg.H_cycles):
        for _i in range(cfg.L_cycles):
            z_L = inner.L_level(z_L, z_H + input_emb, **seq_info)
        z_H = inner.L_level(z_H, z_L, **seq_info)
        snaps[T] = z_H.clone()

    clean_logits = inner.lm_head(z_H)[:, inner.puzzle_emb_len:]
    return snaps, clean_logits


def _shuffle_cells(z: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    """Permute cell positions independently per example."""
    perm = torch.stack([
        torch.randperm(z.shape[1], device=z.device, generator=generator)
        for _ in range(z.shape[0])
    ])
    return torch.gather(z, dim=1, index=perm.unsqueeze(-1).expand_as(z))


def _summary(ce: np.ndarray, acc: np.ndarray, seed: int) -> Dict:
    lo, hi = _bootstrap_mean_ci(ce, seed=seed)
    alo, ahi = _bootstrap_mean_ci(acc, seed=seed)
    return {
        "delta_ce_mean": float(ce.mean()),
        "delta_ce_ci": [lo, hi],
        "delta_acc_mean": float(acc.mean()),
        "delta_acc_ci": [alo, ahi],
        "n": int(len(ce)),
    }


def _batches(indices: np.ndarray, bs: int):
    for s in range(0, len(indices), bs):
        yield indices[s: s + bs]


def _puzzle_difficulty(inputs: np.ndarray) -> Dict[str, np.ndarray]:
    """Classical-solver difficulty features for each puzzle."""
    n = len(inputs)
    givens = np.zeros(n, dtype=np.int32)
    guesses = np.zeros(n, dtype=np.int32)
    backtracks = np.zeros(n, dtype=np.int32)
    needs_bt = np.zeros(n, dtype=bool)
    for k in range(n):
        m = sudoku_metrics(inputs_to_puzzle_string(inputs[k]))
        givens[k] = m.num_givens
        guesses[k] = m.num_guesses
        backtracks[k] = m.num_backtracks
        needs_bt[k] = (m.num_guesses > 0) or (m.num_backtracks > 0)
    return {"givens": givens, "guesses": guesses,
            "backtracks": backtracks, "needs_backtracking": needs_bt}


def _cp_set_sizes(inputs: np.ndarray) -> np.ndarray:
    """Candidate-set size per cell after constraint propagation, (n, 81)."""
    out = np.zeros((len(inputs), 81), dtype=np.int32)
    for k in range(len(inputs)):
        cands = compute_cp_candidates(inputs_to_puzzle_string(inputs[k]))
        out[k] = np.array([max(1, len(s)) for s in cands], dtype=np.int32)
    return out


# ---------------------------------------------------------------------------
# Analyses
# ---------------------------------------------------------------------------

def run_zh(model, raw, orig_idx, donor_idx, device, args) -> Dict:
    """Patch z_H at the end of each outer cycle."""
    inner = model.inner
    cfg = model.config
    n_T = cfg.H_cycles
    gen = torch.Generator(device=device).manual_seed(args.seed)

    d_ce = {T: [] for T in range(n_T)}
    d_acc = {T: [] for T in range(n_T)}
    d_ce_s = {T: [] for T in range(n_T)}
    d_acc_s = {T: [] for T in range(n_T)}

    for oi, di in tqdm(list(zip(_batches(orig_idx, args.batch_size),
                                _batches(donor_idx, args.batch_size))),
                       desc="z_H patching"):
        b_o = _to_batch(raw, oi, device)
        b_d = _to_batch(raw, di, device)
        labels, mask = b_o["labels"], b_o["labels"] != -100

        carry_o = _warmup_carry(inner, b_o, device)
        snaps_o, clean_logits = _capture_zh_snapshots(inner, carry_o, b_o)
        carry_d = _warmup_carry(inner, b_d, device)
        snaps_d, _ = _capture_zh_snapshots(inner, carry_d, b_d)

        clean_logits = clean_logits.float()
        ce_c = _masked_mean(per_cell_cross_entropy(clean_logits, labels), mask)
        acc_c = _masked_mean(_per_cell_correct(clean_logits, labels), mask)

        for T in range(n_T):
            for patch, ce_store, acc_store in (
                (snaps_d[T], d_ce[T], d_acc[T]),
                (_shuffle_cells(snaps_o[T], gen), d_ce_s[T], d_acc_s[T]),
            ):
                out = _inner_forward_patched(
                    inner, carry_o, b_o,
                    target_T=T, target_i=-1, patch_z_H=patch,
                ).float()
                ce_store.extend(
                    (_masked_mean(per_cell_cross_entropy(out, labels), mask) - ce_c).cpu().tolist())
                acc_store.extend(
                    (_masked_mean(_per_cell_correct(out, labels), mask) - acc_c).cpu().tolist())

    results = {"n_pairs": int(len(orig_idx)), "H_cycles": n_T, "cycles": []}
    for T in range(n_T):
        results["cycles"].append({
            "T": T + 1,
            "cross_puzzle": _summary(np.array(d_ce[T]), np.array(d_acc[T]), args.seed + 1),
            "within_shuffle": _summary(np.array(d_ce_s[T]), np.array(d_acc_s[T]), args.seed + 1),
        })
        c = results["cycles"][-1]
        print(f"  z_H after cycle T={T+1}: cross dCE {c['cross_puzzle']['delta_ce_mean']:+.3f} "
              f"{c['cross_puzzle']['delta_ce_ci']}  dAcc {c['cross_puzzle']['delta_acc_mean']:+.4f}"
              f" | shuffle dCE {c['within_shuffle']['delta_ce_mean']:+.3f} "
              f"{c['within_shuffle']['delta_ce_ci']}")
    return results


def run_shufdist(model, raw, orig_idx, device, args, targets) -> Dict:
    """Distribution of the within-puzzle shuffle effect over permutations."""
    inner = model.inner
    K = args.n_perms
    per_perm_ce = {t: [[] for _ in range(K)] for t in targets}
    per_perm_acc = {t: [[] for _ in range(K)] for t in targets}

    for oi in tqdm(list(_batches(orig_idx, args.batch_size)), desc="shuffle distribution"):
        b_o = _to_batch(raw, oi, device)
        labels, mask = b_o["labels"], b_o["labels"] != -100
        carry_o = _warmup_carry(inner, b_o, device)
        snaps_o, clean_logits = _capture_all_snapshots(inner, carry_o, b_o)
        clean_logits = clean_logits.float()
        ce_c = _masked_mean(per_cell_cross_entropy(clean_logits, labels), mask)
        acc_c = _masked_mean(_per_cell_correct(clean_logits, labels), mask)

        for (T, i) in targets:
            for k in range(K):
                gen = torch.Generator(device=device).manual_seed(
                    args.seed * 1000 + k * 17 + int(oi[0]))
                z_shuf = _shuffle_cells(snaps_o[(T, i)], gen)
                out = _inner_forward_patched(
                    inner, carry_o, b_o, target_T=T, target_i=i, patch_z_L=z_shuf).float()
                per_perm_ce[(T, i)][k].extend(
                    (_masked_mean(per_cell_cross_entropy(out, labels), mask) - ce_c).cpu().tolist())
                per_perm_acc[(T, i)][k].extend(
                    (_masked_mean(_per_cell_correct(out, labels), mask) - acc_c).cpu().tolist())

    results = {"n_pairs": int(len(orig_idx)), "n_perms": K, "targets": []}
    for (T, i) in targets:
        perm_means_ce = np.array([np.mean(v) for v in per_perm_ce[(T, i)]])
        perm_means_acc = np.array([np.mean(v) for v in per_perm_acc[(T, i)]])
        pooled_ce = np.concatenate([np.array(v) for v in per_perm_ce[(T, i)]])
        entry = {
            "T": T + 1, "i": i + 1,
            "perm_mean_ce": float(perm_means_ce.mean()),
            "perm_sd_ce": float(perm_means_ce.std(ddof=1)),
            "perm_min_ce": float(perm_means_ce.min()),
            "perm_max_ce": float(perm_means_ce.max()),
            "frac_perms_ce_negative": float((perm_means_ce < 0).mean()),
            "perm_mean_acc": float(perm_means_acc.mean()),
            "perm_sd_acc": float(perm_means_acc.std(ddof=1)),
            "pooled_ce_mean": float(pooled_ce.mean()),
        }
        results["targets"].append(entry)
        print(f"  (T={T+1},i={i+1}) over {K} permutations: dCE {entry['perm_mean_ce']:+.3f} "
              f"+/- {entry['perm_sd_ce']:.3f} (range {entry['perm_min_ce']:+.3f} to "
              f"{entry['perm_max_ce']:+.3f}), {entry['frac_perms_ce_negative']:.0%} negative; "
              f"dAcc {entry['perm_mean_acc']:+.4f}")
    return results


def _match_donors(orig_idx: np.ndarray, donor_idx: np.ndarray, diff: Dict[str, np.ndarray],
                  seed: int) -> np.ndarray:
    """Greedy assignment of donors to originals, matching on backtracking
    requirement first and number of givens second. Returns donors reordered
    to align with orig_idx."""
    rng = np.random.default_rng(seed)
    available = list(donor_idx)
    matched = np.empty_like(orig_idx)
    order = rng.permutation(len(orig_idx))
    for pos in order:
        o = orig_idx[pos]
        best, best_cost = None, None
        for d in available:
            cost = (0 if diff["needs_backtracking"][d] == diff["needs_backtracking"][o] else 100)
            cost += abs(int(diff["givens"][d]) - int(diff["givens"][o]))
            if best_cost is None or cost < best_cost:
                best, best_cost = d, cost
        matched[pos] = best
        available.remove(best)
    return matched


def run_matched(model, raw, orig_idx, donor_idx, device, args) -> Dict:
    """Cross-puzzle swap with difficulty-matched donors, at every (T, i)."""
    inner = model.inner
    cfg = model.config
    targets = [(T, i) for T in range(cfg.H_cycles) for i in range(cfg.L_cycles)]

    diff = _puzzle_difficulty(raw["inputs"])
    matched_idx = _match_donors(orig_idx, donor_idx, diff, args.seed)

    gap_random = np.abs(diff["givens"][orig_idx] - diff["givens"][donor_idx]).mean()
    gap_matched = np.abs(diff["givens"][orig_idx] - diff["givens"][matched_idx]).mean()
    bt_agree_random = float((diff["needs_backtracking"][orig_idx]
                             == diff["needs_backtracking"][donor_idx]).mean())
    bt_agree_matched = float((diff["needs_backtracking"][orig_idx]
                              == diff["needs_backtracking"][matched_idx]).mean())
    print(f"  mean |givens gap|: random {gap_random:.2f} -> matched {gap_matched:.2f}; "
          f"backtracking-flag agreement {bt_agree_random:.2f} -> {bt_agree_matched:.2f}")

    d_ce = {t: [] for t in targets}
    d_acc = {t: [] for t in targets}

    for oi, mi in tqdm(list(zip(_batches(orig_idx, args.batch_size),
                                _batches(matched_idx, args.batch_size))),
                       desc="matched-pair patching"):
        b_o = _to_batch(raw, oi, device)
        b_m = _to_batch(raw, mi, device)
        labels, mask = b_o["labels"], b_o["labels"] != -100

        carry_o = _warmup_carry(inner, b_o, device)
        snaps_o, clean_logits = _capture_all_snapshots(inner, carry_o, b_o)
        carry_m = _warmup_carry(inner, b_m, device)
        snaps_m, _ = _capture_all_snapshots(inner, carry_m, b_m)
        clean_logits = clean_logits.float()
        ce_c = _masked_mean(per_cell_cross_entropy(clean_logits, labels), mask)
        acc_c = _masked_mean(_per_cell_correct(clean_logits, labels), mask)

        for (T, i) in targets:
            out = _inner_forward_patched(
                inner, carry_o, b_o, target_T=T, target_i=i,
                patch_z_L=snaps_m[(T, i)]).float()
            d_ce[(T, i)].extend(
                (_masked_mean(per_cell_cross_entropy(out, labels), mask) - ce_c).cpu().tolist())
            d_acc[(T, i)].extend(
                (_masked_mean(_per_cell_correct(out, labels), mask) - acc_c).cpu().tolist())

    results = {
        "n_pairs": int(len(orig_idx)),
        "givens_gap_random": float(gap_random),
        "givens_gap_matched": float(gap_matched),
        "backtracking_agreement_random": bt_agree_random,
        "backtracking_agreement_matched": bt_agree_matched,
        "steps": [],
    }
    for (T, i) in targets:
        entry = {"T": T + 1, "i": i + 1,
                 "cross_puzzle_matched": _summary(np.array(d_ce[(T, i)]),
                                                  np.array(d_acc[(T, i)]), args.seed + 1)}
        results["steps"].append(entry)
        s = entry["cross_puzzle_matched"]
        print(f"  (T={T+1},i={i+1}) matched cross dCE {s['delta_ce_mean']:+.3f} "
              f"[{s['delta_ce_ci'][0]:+.3f},{s['delta_ce_ci'][1]:+.3f}]  "
              f"dAcc {s['delta_acc_mean']:+.4f}")
    return results


def run_stratify(model, raw, device, args) -> Dict:
    """Clean-run accuracy stratified by |Sc|, digit, given/blank, difficulty."""
    inner = model.inner
    n = min(args.n_stratify, len(raw["inputs"]))
    idx = np.arange(n)

    print(f"  computing candidate sets and solver difficulty for {n} puzzles ...")
    set_sizes = _cp_set_sizes(raw["inputs"][:n])
    diff = _puzzle_difficulty(raw["inputs"][:n])

    correct = np.zeros((n, 81), dtype=np.float32)
    valid = np.zeros((n, 81), dtype=bool)
    digits = np.zeros((n, 81), dtype=np.int32)
    is_given = raw["inputs"][:n] > 1  # 0 = PAD, 1 = blank, 2..10 = digits

    for bi in tqdm(list(_batches(idx, args.batch_size)), desc="clean run"):
        b = _to_batch(raw, bi, device)
        labels = b["labels"]
        mask = labels != -100
        carry = _warmup_carry(inner, b, device)
        _, logits = _capture_all_snapshots(inner, carry, b)
        logits = logits.float()
        c = _per_cell_correct(logits, labels)
        correct[bi] = c.cpu().numpy()
        valid[bi] = mask.cpu().numpy()
        digits[bi] = (labels.cpu().numpy() - 1)

    flat_ok = valid.ravel()
    acc_flat = correct.ravel()[flat_ok]
    sc_flat = set_sizes.ravel()[flat_ok]
    dg_flat = digits.ravel()[flat_ok]
    gv_flat = is_given.ravel()[flat_ok]

    by_sc = []
    for s in range(1, int(sc_flat.max()) + 1):
        m = sc_flat == s
        if m.sum() == 0:
            continue
        by_sc.append({"set_size": int(s), "n_cells": int(m.sum()),
                      "accuracy": float(acc_flat[m].mean())})
    by_digit = []
    for d in range(1, 10):
        m = dg_flat == d
        if m.sum() == 0:
            continue
        by_digit.append({"digit": int(d), "n_cells": int(m.sum()),
                         "accuracy": float(acc_flat[m].mean())})
    by_given = [
        {"cells": "given", "n_cells": int(gv_flat.sum()),
         "accuracy": float(acc_flat[gv_flat].mean())},
        {"cells": "blank", "n_cells": int((~gv_flat).sum()),
         "accuracy": float(acc_flat[~gv_flat].mean())},
    ]

    # Puzzle-level: fully solved vs classical-solver difficulty.
    puzzle_acc = np.array([correct[k][valid[k]].mean() for k in range(n)])
    solved = np.array([bool(correct[k][valid[k]].all()) for k in range(n)])
    bt = diff["needs_backtracking"]
    results = {
        "n_puzzles": int(n),
        "mean_cell_accuracy": float(puzzle_acc.mean()),
        "solved_fraction": float(solved.mean()),
        "dataset_difficulty": {
            "frac_needing_backtracking": float(bt.mean()),
            "mean_givens": float(diff["givens"].mean()),
            "min_givens": int(diff["givens"].min()),
            "max_givens": int(diff["givens"].max()),
            "mean_guesses": float(diff["guesses"].mean()),
            "mean_guesses_backtracking_only": float(diff["guesses"][bt].mean()) if bt.any() else 0.0,
        },
        "solve_rate_by_backtracking": {
            "needs_backtracking": {"n": int(bt.sum()),
                                   "solved_fraction": float(solved[bt].mean()) if bt.any() else 0.0,
                                   "cell_accuracy": float(puzzle_acc[bt].mean()) if bt.any() else 0.0},
            "propagation_only": {"n": int((~bt).sum()),
                                 "solved_fraction": float(solved[~bt].mean()) if (~bt).any() else 0.0,
                                 "cell_accuracy": float(puzzle_acc[~bt].mean()) if (~bt).any() else 0.0},
        },
        "givens_solved": float(diff["givens"][solved].mean()) if solved.any() else 0.0,
        "givens_unsolved": float(diff["givens"][~solved].mean()) if (~solved).any() else 0.0,
        "guesses_solved": float(diff["guesses"][solved].mean()) if solved.any() else 0.0,
        "guesses_unsolved": float(diff["guesses"][~solved].mean()) if (~solved).any() else 0.0,
        "accuracy_by_set_size": by_sc,
        "accuracy_by_digit": by_digit,
        "accuracy_by_given": by_given,
    }
    print(json.dumps({k: v for k, v in results.items()
                      if k not in ("accuracy_by_set_size", "accuracy_by_digit")}, indent=2))
    print("  accuracy by |Sc|:", [(e["set_size"], round(e["accuracy"], 3)) for e in by_sc])
    print("  accuracy by digit:", [(e["digit"], round(e["accuracy"], 3)) for e in by_digit])
    return results


def _to_batch(raw, indices: np.ndarray, device) -> Dict[str, torch.Tensor]:
    return {
        "inputs": torch.from_numpy(raw["inputs"][indices].astype(np.int32)).to(device),
        "labels": torch.from_numpy(raw["labels"][indices].astype(np.int32)).to(device),
        "puzzle_identifiers": torch.from_numpy(
            raw["puzzle_identifiers"][indices].astype(np.int32)).to(device),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Patching follow-ups for the rebuttal.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data-path", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--n-pairs", type=int, default=200)
    ap.add_argument("--n-stratify", type=int, default=1000)
    ap.add_argument("--n-perms", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--do", default="zh,shufdist,matched,stratify")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    todo = [s.strip() for s in args.do.split(",") if s.strip()]

    print("Loading model ...")
    model = load_trm_model(args.config, args.checkpoint, device, args.data_path, args.split)
    model.eval()

    # Pairing identical to activation_patching_sweep at the same seed.
    rng = np.random.default_rng(args.seed)
    max_examples = max(args.n_pairs * 2, args.n_stratify if "stratify" in todo else 0)
    raw = load_test_data(args.data_path, args.split, max_examples=max_examples)
    N_pair_pool = min(len(raw["inputs"]), args.n_pairs * 2)
    order = rng.permutation(N_pair_pool)
    n_pairs = min(args.n_pairs, N_pair_pool // 2)
    orig_idx = order[:n_pairs]
    donor_idx = order[n_pairs: 2 * n_pairs]

    if "zh" in todo:
        print("\n=== z_H patching ===")
        out = run_zh(model, raw, orig_idx, donor_idx, device, args)
        _save(out, args.output_dir, "zh_patching.json")

    if "shufdist" in todo:
        print("\n=== shuffle permutation distribution ===")
        cfg = model.config
        targets = [(0, 3), (1, 4), (2, cfg.L_cycles - 1)]  # (1,4), (2,5), (3,6) in paper indexing
        out = run_shufdist(model, raw, orig_idx, device, args, targets)
        _save(out, args.output_dir, "shuffle_distribution.json")

    if "matched" in todo:
        print("\n=== difficulty-matched donors ===")
        out = run_matched(model, raw, orig_idx, donor_idx, device, args)
        _save(out, args.output_dir, "matched_pairs.json")

    if "stratify" in todo:
        print("\n=== clean-run stratified error analysis ===")
        out = run_stratify(model, raw, device, args)
        _save(out, args.output_dir, "error_stratification.json")


def _save(obj: Dict, out_dir: str, name: str) -> None:
    path = os.path.join(out_dir, name)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
    print(f"Saved -> {path}")


if __name__ == "__main__":
    main()
