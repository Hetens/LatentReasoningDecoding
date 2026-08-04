"""
Systematic activation patching sweep over ALL 18 recursion indices (T, i).

Extends experiments.probing.activation_patching (which patches a single
(T, i) per invocation) to produce a full causal map DeltaCE(T, i) in one
job. Per batch, the donor and original carries are warmed up once, all 18
z_L snapshots are captured in a single final-ACT-step pass, and each
(T, i) is then patched with a cheap single-pass forward. This is ~10x
faster than looping the single-target script.

Interventions per (T, i), mirroring the original protocol exactly:
  - cross-puzzle swap: replace z_L with the donor puzzle's z_L.
  - within-puzzle shuffle: permute cell positions of the original z_L.

Metrics per (T, i) and intervention:
  - DeltaCE:  mean change in per-cell cross-entropy (paper metric).
  - DeltaAcc: mean change in per-cell accuracy (task metric; added for
    the rebuttal to test whether effects survive beyond CE).

With --seed 0 --n-pairs 200 the puzzle pairing is identical to the runs
behind Table 2, so (T=1,i=4) and (T=2,i=5) serve as sanity anchors.

Usage (from repo root):
    python -m experiments.probing.activation_patching_sweep \
        --config trm_base/config_pretrain_paper.yml \
        --checkpoint "checkpoints/.../step_65100.pt" \
        --data-path data/sudoku-extreme-1k-aug-1000 \
        --output-dir results/probing/patching_sweep \
        --n-pairs 200 --batch-size 32 --seed 0
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
    TinyRecursiveReasoningModel_ACTV1,
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


# ---------------------------------------------------------------------------
# Carry warm-up and snapshot capture
# ---------------------------------------------------------------------------

@torch.no_grad()
def _warmup_carry(
    inner: TinyRecursiveReasoningModel_ACTV1_Inner,
    batch: Dict[str, torch.Tensor],
    device: torch.device,
) -> TinyRecursiveReasoningModel_ACTV1InnerCarry:
    """Run ACT steps 1 .. halt_max-1 and return the carry entering the final
    ACT step. Mirrors the warm-up inside _run_clean_and_patched."""
    cfg = inner.config
    batch_size = batch["inputs"].shape[0]

    inner_carry = inner.empty_carry(batch_size)
    inner_carry.z_H = inner_carry.z_H.to(device)
    inner_carry.z_L = inner_carry.z_L.to(device)
    inner_carry = inner.reset_carry(
        torch.ones(batch_size, dtype=torch.bool, device=device), inner_carry
    )

    seq_info = dict(cos_sin=inner.rotary_emb() if hasattr(inner, "rotary_emb") else None)
    input_emb = inner._input_embeddings(batch["inputs"], batch["puzzle_identifiers"])

    z_H, z_L = inner_carry.z_H, inner_carry.z_L
    for _act_step in range(1, cfg.halt_max_steps):
        for _T in range(cfg.H_cycles):
            for _i in range(cfg.L_cycles):
                z_L = inner.L_level(z_L, z_H + input_emb, **seq_info)
            z_H = inner.L_level(z_H, z_L, **seq_info)
    return TinyRecursiveReasoningModel_ACTV1InnerCarry(z_H=z_H.detach(), z_L=z_L.detach())


@torch.no_grad()
def _capture_all_snapshots(
    inner: TinyRecursiveReasoningModel_ACTV1_Inner,
    carry: TinyRecursiveReasoningModel_ACTV1InnerCarry,
    batch: Dict[str, torch.Tensor],
) -> Tuple[Dict[Tuple[int, int], torch.Tensor], torch.Tensor]:
    """Run the final ACT step once, capturing z_L after its update at every
    (T, i). The snapshot point matches _inner_forward_patched's patch point.

    Returns (snapshots, clean_logits)."""
    cfg = inner.config
    seq_info = dict(cos_sin=inner.rotary_emb() if hasattr(inner, "rotary_emb") else None)
    input_emb = inner._input_embeddings(batch["inputs"], batch["puzzle_identifiers"])

    snaps: Dict[Tuple[int, int], torch.Tensor] = {}
    z_H, z_L = carry.z_H, carry.z_L
    for T in range(cfg.H_cycles):
        for i in range(cfg.L_cycles):
            z_L = inner.L_level(z_L, z_H + input_emb, **seq_info)
            snaps[(T, i)] = z_L.clone()
        z_H = inner.L_level(z_H, z_L, **seq_info)

    clean_logits = inner.lm_head(z_H)[:, inner.puzzle_emb_len:]
    return snaps, clean_logits


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _masked_mean(per_cell: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (per_cell * mask).sum(dim=-1) / mask.sum(dim=-1).clamp(min=1)


def _per_cell_correct(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return (logits.argmax(dim=-1) == labels.long()).float()


def _bootstrap_mean_ci(arr: np.ndarray, seed: int, n_boot: int = 10_000, alpha: float = 0.05):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(arr), size=(n_boot, len(arr)))
    means = arr[idx].mean(axis=1)
    return (
        float(np.percentile(means, 100 * alpha / 2)),
        float(np.percentile(means, 100 * (1 - alpha / 2))),
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Patching sweep over all (T, i).")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--n-pairs", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    print("Loading model ...")
    model = load_trm_model(args.config, args.checkpoint, device, args.data_path, args.split)
    model.eval()
    cfg = model.config
    inner = model.inner
    n_T, n_I = cfg.H_cycles, cfg.L_cycles
    targets: List[Tuple[int, int]] = [(T, i) for T in range(n_T) for i in range(n_I)]
    print(f"  H_cycles={n_T}  L_cycles={n_I}  halt_max={cfg.halt_max_steps}")
    print(f"  Sweeping {len(targets)} targets x 2 interventions, {args.n_pairs} pairs")

    print("Loading data ...")
    raw = load_test_data(args.data_path, args.split, max_examples=args.n_pairs * 2)
    N = len(raw["inputs"])
    n_pairs = min(args.n_pairs, N // 2)

    # Identical pairing to the single-target script at the same seed.
    order = rng.permutation(N)
    orig_idx = order[:n_pairs]
    donor_idx = order[n_pairs: 2 * n_pairs]

    # Accumulators: per target, per intervention, per puzzle.
    d_ce_cross = {t: [] for t in targets}
    d_ce_shuf = {t: [] for t in targets}
    d_acc_cross = {t: [] for t in targets}
    d_acc_shuf = {t: [] for t in targets}
    # Clean-run per-puzzle stats, for post-hoc splits (saturated vs. not).
    clean_ce_all: list = []
    clean_acc_all: list = []

    bs = args.batch_size
    for start in tqdm(range(0, n_pairs, bs), desc="batches"):
        end = min(start + bs, n_pairs)
        oi = orig_idx[start:end]
        di = donor_idx[start:end]

        def _to_batch(indices: np.ndarray) -> Dict[str, torch.Tensor]:
            return {
                "inputs": torch.from_numpy(raw["inputs"][indices].astype(np.int32)).to(device),
                "labels": torch.from_numpy(raw["labels"][indices].astype(np.int32)).to(device),
                "puzzle_identifiers": torch.from_numpy(
                    raw["puzzle_identifiers"][indices].astype(np.int32)
                ).to(device),
            }

        batch_orig = _to_batch(oi)
        batch_donor = _to_batch(di)
        labels = batch_orig["labels"]
        mask = labels != -100

        # One warm-up + one capture pass per side.
        carry_orig = _warmup_carry(inner, batch_orig, device)
        snaps_orig, clean_logits = _capture_all_snapshots(inner, carry_orig, batch_orig)
        carry_donor = _warmup_carry(inner, batch_donor, device)
        snaps_donor, _ = _capture_all_snapshots(inner, carry_donor, batch_donor)
        # Metrics in float32: the model runs bf16 and small CE deltas
        # quantize to zero at bf16 resolution.
        clean_logits = clean_logits.float()

        ce_clean = per_cell_cross_entropy(clean_logits, labels)
        acc_clean = _per_cell_correct(clean_logits, labels)
        ce_clean_m = _masked_mean(ce_clean, mask)
        acc_clean_m = _masked_mean(acc_clean, mask)
        clean_ce_all.extend(ce_clean_m.cpu().tolist())
        clean_acc_all.extend(acc_clean_m.cpu().tolist())

        for (T, i) in targets:
            # --- Cross-puzzle swap ---
            patched = _inner_forward_patched(
                inner, carry_orig, batch_orig,
                target_T=T, target_i=i,
                patch_z_L=snaps_donor[(T, i)],
            ).float()
            ce_p = _masked_mean(per_cell_cross_entropy(patched, labels), mask)
            acc_p = _masked_mean(_per_cell_correct(patched, labels), mask)
            d_ce_cross[(T, i)].extend((ce_p - ce_clean_m).cpu().tolist())
            d_acc_cross[(T, i)].extend((acc_p - acc_clean_m).cpu().tolist())

            # --- Within-puzzle shuffle (permute cell positions of own z_L) ---
            z = snaps_orig[(T, i)]
            perm = torch.stack(
                [torch.randperm(z.shape[1], device=device) for _ in range(z.shape[0])]
            )
            z_shuf = torch.gather(z, dim=1, index=perm.unsqueeze(-1).expand_as(z))
            patched_s = _inner_forward_patched(
                inner, carry_orig, batch_orig,
                target_T=T, target_i=i,
                patch_z_L=z_shuf,
            ).float()
            ce_s = _masked_mean(per_cell_cross_entropy(patched_s, labels), mask)
            acc_s = _masked_mean(_per_cell_correct(patched_s, labels), mask)
            d_ce_shuf[(T, i)].extend((ce_s - ce_clean_m).cpu().tolist())
            d_acc_shuf[(T, i)].extend((acc_s - acc_clean_m).cpu().tolist())

    # ---- Aggregate, report, save ----
    results = {
        "n_pairs": n_pairs,
        "seed": args.seed,
        "checkpoint": args.checkpoint,
        "H_cycles": n_T,
        "L_cycles": n_I,
        "steps": [],
    }
    print(f"\n{'(T,i)':>8} {'dCE cross':>22} {'dCE shuffle':>22} {'dAcc cross':>12} {'dAcc shuf':>12}")
    npz_payload = {}
    for (T, i) in targets:
        entry = {"T": T + 1, "i": i + 1}
        for name, store, acc_store in (
            ("cross_puzzle", d_ce_cross, d_acc_cross),
            ("within_shuffle", d_ce_shuf, d_acc_shuf),
        ):
            ce_arr = np.array(store[(T, i)])
            acc_arr = np.array(acc_store[(T, i)])
            lo, hi = _bootstrap_mean_ci(ce_arr, seed=args.seed + 1)
            alo, ahi = _bootstrap_mean_ci(acc_arr, seed=args.seed + 1)
            entry[name] = {
                "delta_ce_mean": float(ce_arr.mean()),
                "delta_ce_ci": [lo, hi],
                "delta_acc_mean": float(acc_arr.mean()),
                "delta_acc_ci": [alo, ahi],
            }
            npz_payload[f"dce_{name}_T{T+1}_i{i+1}"] = ce_arr
            npz_payload[f"dacc_{name}_T{T+1}_i{i+1}"] = acc_arr
        results["steps"].append(entry)
        c, s = entry["cross_puzzle"], entry["within_shuffle"]
        print(
            f"  (T={T+1},i={i+1})"
            f"  {c['delta_ce_mean']:+.3f} [{c['delta_ce_ci'][0]:+.3f},{c['delta_ce_ci'][1]:+.3f}]"
            f"  {s['delta_ce_mean']:+.3f} [{s['delta_ce_ci'][0]:+.3f},{s['delta_ce_ci'][1]:+.3f}]"
            f"  {c['delta_acc_mean']:+.4f}"
            f"  {s['delta_acc_mean']:+.4f}"
        )

    npz_payload["clean_ce"] = np.array(clean_ce_all)
    npz_payload["clean_acc"] = np.array(clean_acc_all)
    npz_payload["orig_idx"] = orig_idx
    npz_payload["donor_idx"] = donor_idx

    out_json = os.path.join(args.output_dir, "patching_sweep.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    out_npz = os.path.join(args.output_dir, "patching_sweep_per_puzzle.npz")
    np.savez_compressed(out_npz, **npz_payload)
    print(f"\nSaved -> {out_json}")
    print(f"Saved -> {out_npz}")


if __name__ == "__main__":
    main()
