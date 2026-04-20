"""
Activation patching: replace z at a target (T, i) with activations from a
donor puzzle and measure the shift in the model's cell logits.

Two intervention types (Section 4 of the report):
  - **cross-puzzle swap**: replace z with z from a different puzzle.
  - **within-puzzle shuffle**: randomly permute cell positions in z within
    the same puzzle.

For each intervention the metric is Δ = mean change in per-cell
cross-entropy toward the donor puzzle's ground-truth labels.

Usage (from repo root):
    python -m experiments.probing.activation_patching \
        --config  trm_base/config_pretrain_paper.yml \
        --checkpoint checkpoints/.../step_50000.pt \
        --data-path data/sudoku-extreme-full \
        --output-dir results/probing/patching \
        --target-T 3 --target-i 4 \
        --n-pairs 200 \
        --seed 0
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
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


# ---------------------------------------------------------------------------
# Patched forward pass
# ---------------------------------------------------------------------------

@torch.no_grad()
def _inner_forward_patched(
    inner: TinyRecursiveReasoningModel_ACTV1_Inner,
    carry: TinyRecursiveReasoningModel_ACTV1InnerCarry,
    batch: Dict[str, torch.Tensor],
    *,
    target_T: int,
    target_i: int,
    patch_z_L: Optional[torch.Tensor] = None,
    patch_z_H: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Run one inner forward pass, patching z_L and/or z_H at (target_T, target_i).

    Returns the output logits only.
    """
    cfg = inner.config
    seq_info = dict(cos_sin=inner.rotary_emb() if hasattr(inner, "rotary_emb") else None)
    input_emb = inner._input_embeddings(batch["inputs"], batch["puzzle_identifiers"])

    z_H = carry.z_H
    z_L = carry.z_L

    for T in range(cfg.H_cycles):
        for i in range(cfg.L_cycles):
            z_L = inner.L_level(z_L, z_H + input_emb, **seq_info)
            if T == target_T and i == target_i and patch_z_L is not None:
                z_L = patch_z_L
        z_H = inner.L_level(z_H, z_L, **seq_info)
        if T == target_T and patch_z_H is not None:
            z_H = patch_z_H

    return inner.lm_head(z_H)[:, inner.puzzle_emb_len:]


@torch.no_grad()
def _run_clean_and_patched(
    model: TinyRecursiveReasoningModel_ACTV1,
    batch: Dict[str, torch.Tensor],
    device: torch.device,
    target_T: int,
    target_i: int,
    patch_z_L: Optional[torch.Tensor] = None,
    patch_z_H: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run the full ACT loop to warm up the carry, then run the final step
    twice: once clean and once with the patch applied.

    Returns:
        clean_logits:  (batch, seq_len, vocab)
        patched_logits: (batch, seq_len, vocab)
    """
    inner = model.inner
    cfg = model.config
    halt_max = cfg.halt_max_steps
    batch_size = batch["inputs"].shape[0]

    # Initialise ACT carry.
    inner_carry = inner.empty_carry(batch_size)
    inner_carry.z_H = inner_carry.z_H.to(device)
    inner_carry.z_L = inner_carry.z_L.to(device)
    steps = torch.zeros(batch_size, dtype=torch.int32, device=device)
    halted = torch.ones(batch_size, dtype=torch.bool, device=device)
    current_data = {k: torch.empty_like(v) for k, v in batch.items()}

    # Run ACT steps 1 … halt_max-1 normally (warm-up).
    for act_step in range(1, halt_max):
        inner_carry = inner.reset_carry(halted, inner_carry)
        steps = torch.where(halted, torch.zeros_like(steps), steps)
        current_data = {
            k: torch.where(halted.view((-1,) + (1,) * (batch[k].ndim - 1)), batch[k], v)
            for k, v in current_data.items()
        }
        seq_info = dict(cos_sin=inner.rotary_emb() if hasattr(inner, "rotary_emb") else None)
        input_emb = inner._input_embeddings(current_data["inputs"], current_data["puzzle_identifiers"])
        z_H, z_L = inner_carry.z_H, inner_carry.z_L
        for _T in range(cfg.H_cycles):
            for _i in range(cfg.L_cycles):
                z_L = inner.L_level(z_L, z_H + input_emb, **seq_info)
            z_H = inner.L_level(z_H, z_L, **seq_info)
        inner_carry = TinyRecursiveReasoningModel_ACTV1InnerCarry(z_H=z_H.detach(), z_L=z_L.detach())
        steps = steps + 1
        halted = steps >= halt_max

    # Prepare final step carry.
    inner_carry = inner.reset_carry(halted, inner_carry)
    current_data = {
        k: torch.where(halted.view((-1,) + (1,) * (batch[k].ndim - 1)), batch[k], v)
        for k, v in current_data.items()
    }

    # Clean forward on the last ACT step.
    clean_logits = _inner_forward_patched(
        inner, inner_carry, current_data,
        target_T=-1, target_i=-1,  # no patch
    )

    # Patched forward.
    patched_logits = _inner_forward_patched(
        inner, inner_carry, current_data,
        target_T=target_T, target_i=target_i,
        patch_z_L=patch_z_L, patch_z_H=patch_z_H,
    )

    return clean_logits, patched_logits


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def per_cell_cross_entropy(
    logits: torch.Tensor, labels: torch.Tensor,
) -> torch.Tensor:
    """Per-cell cross-entropy (ignoring IGNORE_LABEL_ID=-100)."""
    return F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        labels.reshape(-1).long(),
        ignore_index=-100,
        reduction="none",
    ).reshape(labels.shape)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Activation patching experiments.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--target-T", type=int, required=True, help="1-indexed outer cycle.")
    parser.add_argument("--target-i", type=int, required=True, help="1-indexed inner step.")
    parser.add_argument("--n-pairs", type=int, default=200, help="Number of puzzle pairs to test.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    # Convert to 0-indexed.
    tgt_T = args.target_T - 1
    tgt_i = args.target_i - 1

    print("Loading model …")
    model = load_trm_model(args.config, args.checkpoint, device, args.data_path, args.split)
    model.eval()
    cfg = model.config
    print(f"  H_cycles={cfg.H_cycles}  L_cycles={cfg.L_cycles}  halt_max={cfg.halt_max_steps}")

    print("Loading data …")
    raw = load_test_data(args.data_path, args.split, max_examples=args.n_pairs * 2)
    N = len(raw["inputs"])
    n_pairs = min(args.n_pairs, N // 2)

    # Pair up puzzles: (original, donor).
    order = rng.permutation(N)
    orig_idx = order[:n_pairs]
    donor_idx = order[n_pairs: 2 * n_pairs]

    delta_cross_puzzle: list[float] = []
    delta_shuffle: list[float] = []

    print(f"Running {n_pairs} patching pairs at (T={tgt_T+1}, i={tgt_i+1}) …")
    bs = args.batch_size

    for start in tqdm(range(0, n_pairs, bs)):
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

        # --- Extract donor z_L at the target step ---
        inner = model.inner
        inner_carry = inner.empty_carry(len(di))
        inner_carry.z_H = inner_carry.z_H.to(device)
        inner_carry.z_L = inner_carry.z_L.to(device)
        inner_carry = inner.reset_carry(torch.ones(len(di), dtype=torch.bool, device=device), inner_carry)

        seq_info = dict(cos_sin=inner.rotary_emb() if hasattr(inner, "rotary_emb") else None)
        input_emb = inner._input_embeddings(batch_donor["inputs"], batch_donor["puzzle_identifiers"])
        z_H_d, z_L_d = inner_carry.z_H, inner_carry.z_L

        # Run the full ACT warm-up on donor to get its z.
        halt_max = cfg.halt_max_steps
        for act_step in range(1, halt_max + 1):
            z_H_d_old, z_L_d_old = z_H_d, z_L_d
            for T in range(cfg.H_cycles):
                for i in range(cfg.L_cycles):
                    z_L_d = inner.L_level(z_L_d, z_H_d + input_emb, **seq_info)
                    if act_step == halt_max and T == tgt_T and i == tgt_i:
                        donor_z_L = z_L_d.clone()
                z_H_d = inner.L_level(z_H_d, z_L_d, **seq_info)

        # --- Cross-puzzle swap ---
        clean, patched = _run_clean_and_patched(
            model, batch_orig, device,
            target_T=tgt_T, target_i=tgt_i,
            patch_z_L=donor_z_L,
        )
        labels = batch_orig["labels"]
        ce_clean = per_cell_cross_entropy(clean, labels)
        ce_patched = per_cell_cross_entropy(patched, labels)
        mask = labels != -100
        delta = ((ce_patched - ce_clean) * mask).sum(dim=-1) / mask.sum(dim=-1).clamp(min=1)
        delta_cross_puzzle.extend(delta.cpu().tolist())

        # --- Within-puzzle shuffle ---
        perm = torch.stack([torch.randperm(donor_z_L.shape[1], device=device) for _ in range(len(oi))])
        shuffled_z_L = torch.gather(
            clean.new_zeros(*donor_z_L.shape).copy_(donor_z_L),
            dim=1,
            index=perm.unsqueeze(-1).expand_as(donor_z_L),
        )
        # Actually re-use original z, but shuffle cell positions within it.
        # We need z_L from the original, not the donor, and then shuffle.
        # Let's get original z_L at target step via a separate pass:
        inner_carry_o = inner.empty_carry(len(oi))
        inner_carry_o.z_H = inner_carry_o.z_H.to(device)
        inner_carry_o.z_L = inner_carry_o.z_L.to(device)
        inner_carry_o = inner.reset_carry(
            torch.ones(len(oi), dtype=torch.bool, device=device), inner_carry_o,
        )
        input_emb_o = inner._input_embeddings(batch_orig["inputs"], batch_orig["puzzle_identifiers"])
        z_H_o, z_L_o = inner_carry_o.z_H, inner_carry_o.z_L
        for act_step in range(1, halt_max + 1):
            for T in range(cfg.H_cycles):
                for i in range(cfg.L_cycles):
                    z_L_o = inner.L_level(z_L_o, z_H_o + input_emb_o, **seq_info)
                    if act_step == halt_max and T == tgt_T and i == tgt_i:
                        orig_z_L = z_L_o.clone()
                z_H_o = inner.L_level(z_H_o, z_L_o, **seq_info)

        perm_o = torch.stack([torch.randperm(orig_z_L.shape[1], device=device) for _ in range(len(oi))])
        shuffled_orig_z_L = torch.gather(
            orig_z_L,
            dim=1,
            index=perm_o.unsqueeze(-1).expand_as(orig_z_L),
        )
        _, patched_shuf = _run_clean_and_patched(
            model, batch_orig, device,
            target_T=tgt_T, target_i=tgt_i,
            patch_z_L=shuffled_orig_z_L,
        )
        ce_shuf = per_cell_cross_entropy(patched_shuf, labels)
        delta_s = ((ce_shuf - ce_clean) * mask).sum(dim=-1) / mask.sum(dim=-1).clamp(min=1)
        delta_shuffle.extend(delta_s.cpu().tolist())

    # ---- Report ----
    delta_cross = np.array(delta_cross_puzzle)
    delta_shuf = np.array(delta_shuffle)

    def _bootstrap_mean_ci(arr: np.ndarray, n_boot: int = 10_000, alpha: float = 0.05):
        rng2 = np.random.default_rng(args.seed + 1)
        means = np.empty(n_boot)
        for i in range(n_boot):
            means[i] = rng2.choice(arr, size=len(arr), replace=True).mean()
        return float(np.percentile(means, 100 * alpha / 2)), float(np.percentile(means, 100 * (1 - alpha / 2)))

    cross_mean = float(delta_cross.mean())
    cross_lo, cross_hi = _bootstrap_mean_ci(delta_cross)
    shuf_mean = float(delta_shuf.mean())
    shuf_lo, shuf_hi = _bootstrap_mean_ci(delta_shuf)

    print(f"\n--- Activation Patching at (T={tgt_T+1}, i={tgt_i+1}) ---")
    print(f"Cross-puzzle swap:   Δ CE = {cross_mean:.4f}  [{cross_lo:.4f}, {cross_hi:.4f}]")
    print(f"Within-puzzle shuf:  Δ CE = {shuf_mean:.4f}  [{shuf_lo:.4f}, {shuf_hi:.4f}]")

    results = {
        "target_T": tgt_T + 1,
        "target_i": tgt_i + 1,
        "n_pairs": n_pairs,
        "cross_puzzle": {"mean": cross_mean, "ci_lo": cross_lo, "ci_hi": cross_hi},
        "within_shuffle": {"mean": shuf_mean, "ci_lo": shuf_lo, "ci_hi": shuf_hi},
    }

    out_file = os.path.join(args.output_dir, f"patching_T{tgt_T+1}_i{tgt_i+1}.json")
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved → {out_file}")


if __name__ == "__main__":
    main()
