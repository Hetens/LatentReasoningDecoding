"""
Induction head analysis for TRM.

In standard language models, induction heads copy tokens that follow similar
prior contexts. In Sudoku, the analogous mechanism is attention heads that
route information along constraint-relevant dimensions: cells in the same
row, column, or 3x3 box share constraints and must coordinate.

This script:
  1. Hooks into the Attention modules to capture raw attention weights
     (bypasses scaled_dot_product_attention which does not return them).
  2. Records attention patterns at every (T, i) recursion step.
  3. Computes "constraint attention scores": fraction of total attention
     mass that each head places on same-row, same-col, same-box cells.
  4. Identifies heads that consistently attend to constraint-relevant
     positions above a random baseline.
  5. Produces per-head, per-layer, per-(T,i) analysis and plots.

Architecture recap (trm_paper.yml):
  - L_layers=2 blocks in L_level, each with 8 attention heads (head_dim=64)
  - Same L_level is used for z_L updates and z_H updates (weight-shared)
  - 18 inner calls to L_level for z_L (3 outer x 6 inner) + 3 calls for z_H
  - Total: 21 calls per ACT step, each passing through 2 layers x 8 heads

Usage (from repo root):
    python -m experiments.probing.induction_heads \
        --config  trm_base/config_pretrain_paper.yml \
        --checkpoint checkpoints/.../step_50000.pt \
        --data-path data/sudoku-extreme-full \
        --output-dir results/probing/attention \
        --max-examples 100 --split test
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_TRM_BASE = os.path.join(_PROJECT_ROOT, "trm_base")
if _TRM_BASE not in sys.path:
    sys.path.insert(0, _TRM_BASE)

from layers import RotaryEmbedding, Attention, apply_rotary_pos_emb  # noqa: E402
from trm import (  # noqa: E402
    TinyRecursiveReasoningModel_ACTV1,
    TinyRecursiveReasoningModel_ACTV1Config,
    TinyRecursiveReasoningModel_ACTV1InnerCarry,
    TinyRecursiveReasoningModel_ACTV1_Inner,
)

_EXPERIMENTS_ROOT = os.path.abspath(os.path.dirname(__file__))
if _EXPERIMENTS_ROOT not in sys.path:
    sys.path.insert(0, _EXPERIMENTS_ROOT)
from extract_activations import load_trm_model, load_test_data  # noqa: E402


# ===================================================================
# Sudoku constraint masks (81 x 81)
# ===================================================================

def build_constraint_masks(seq_len: int = 81) -> Dict[str, np.ndarray]:
    """Build boolean masks for same-row, same-col, same-box relationships."""
    same_row = np.zeros((seq_len, seq_len), dtype=bool)
    same_col = np.zeros((seq_len, seq_len), dtype=bool)
    same_box = np.zeros((seq_len, seq_len), dtype=bool)
    any_constraint = np.zeros((seq_len, seq_len), dtype=bool)

    for c1 in range(seq_len):
        r1, col1 = c1 // 9, c1 % 9
        box1 = (r1 // 3) * 3 + (col1 // 3)
        for c2 in range(seq_len):
            r2, col2 = c2 // 9, c2 % 9
            box2 = (r2 // 3) * 3 + (col2 // 3)
            if c1 != c2:
                if r1 == r2:
                    same_row[c1, c2] = True
                if col1 == col2:
                    same_col[c1, c2] = True
                if box1 == box2:
                    same_box[c1, c2] = True
                if r1 == r2 or col1 == col2 or box1 == box2:
                    any_constraint[c1, c2] = True

    return {
        "same_row": same_row,
        "same_col": same_col,
        "same_box": same_box,
        "any_constraint": any_constraint,
    }


# ===================================================================
# Attention extraction via hooks
# ===================================================================

class AttentionCaptureHook:
    """Forward hook that intercepts the Attention module to compute and
    store attention weights (which scaled_dot_product_attention discards)."""

    def __init__(self):
        self.attention_weights: List[torch.Tensor] = []
        self._call_idx = 0

    def reset(self):
        self.attention_weights = []
        self._call_idx = 0

    def hook_fn(self, module: Attention, args, kwargs, output):
        """Post-forward hook. We recompute attention weights from Q, K
        since SDPA does not return them."""
        # Recompute QKV from the stored hidden_states
        # We need to intercept before SDPA. Instead, use a pre-hook
        # approach: register a wrapper that computes attn weights.
        pass

    def __call__(self, module: Attention, args, output):
        pass


def compute_attention_weights_manual(
    attn_module: Attention,
    hidden_states: torch.Tensor,
    cos_sin: Optional[Tuple[torch.Tensor, torch.Tensor]],
) -> torch.Tensor:
    """Manually compute attention weights from an Attention module.

    Returns: (batch, num_heads, seq_len, seq_len) attention probabilities.
    """
    batch_size, seq_len, _ = hidden_states.shape
    qkv = attn_module.qkv_proj(hidden_states)
    qkv = qkv.view(batch_size, seq_len,
                    attn_module.num_heads + 2 * attn_module.num_key_value_heads,
                    attn_module.head_dim)
    query = qkv[:, :, :attn_module.num_heads]
    key = qkv[:, :, attn_module.num_heads:attn_module.num_heads + attn_module.num_key_value_heads]

    if cos_sin is not None:
        cos, sin = cos_sin
        query, key = apply_rotary_pos_emb(query, key, cos, sin)

    # (B, S, H, D) -> (B, H, S, D)
    query = query.permute(0, 2, 1, 3)
    key = key.permute(0, 2, 1, 3)

    scale = query.shape[-1] ** -0.5
    attn_weights = torch.matmul(query, key.transpose(-2, -1)) * scale
    attn_weights = F.softmax(attn_weights.float(), dim=-1)  # (B, H, S, S)

    return attn_weights


@torch.no_grad()
def extract_attention_patterns(
    inner: TinyRecursiveReasoningModel_ACTV1_Inner,
    batch: Dict[str, torch.Tensor],
    puzzle_emb_len: int,
    carry: Optional[TinyRecursiveReasoningModel_ACTV1InnerCarry] = None,
) -> Dict[str, torch.Tensor]:
    """Run one inner forward pass and capture attention weights at every
    (T, i) step for each layer and head.

    Args:
        carry: If provided, uses this carry state (e.g., from a previous ACT
               step). If None, initializes fresh carry (for single-step use).

    Returns dict with keys like "zL_T{t}_i{i}_layer{l}" and "zH_T{t}_layer{l}",
    each mapping to attention weights of shape (B, num_heads, S, S) restricted
    to the puzzle cells (excluding puzzle embedding prefix).
    """
    cfg = inner.config
    cos_sin = inner.rotary_emb() if hasattr(inner, "rotary_emb") else None
    input_emb = inner._input_embeddings(batch["inputs"], batch["puzzle_identifiers"])

    if carry is None:
        carry = inner.empty_carry(batch["inputs"].shape[0])
        carry.z_H = carry.z_H.to(batch["inputs"].device)
        carry.z_L = carry.z_L.to(batch["inputs"].device)
        carry = inner.reset_carry(
            torch.ones(batch["inputs"].shape[0], dtype=torch.bool, device=batch["inputs"].device),
            carry,
        )

    z_H, z_L = carry.z_H, carry.z_L
    attn_patterns = {}

    L_level = inner.L_level  # the shared reasoning module

    for T in range(cfg.H_cycles):
        # Inner loop: z_L updates
        for i in range(cfg.L_cycles):
            hs = z_L + (z_H + input_emb)  # input_injection
            for layer_idx, layer in enumerate(L_level.layers):
                if hasattr(layer, "self_attn"):
                    aw = compute_attention_weights_manual(layer.self_attn, hs, cos_sin)
                    # Trim puzzle embedding prefix
                    if puzzle_emb_len > 0:
                        aw = aw[:, :, puzzle_emb_len:, puzzle_emb_len:]
                    key = f"zL_T{T+1}_i{i+1}_layer{layer_idx}"
                    attn_patterns[key] = aw.cpu()
                # Still need to run the actual forward for correct z_L
                hs = layer(cos_sin=cos_sin, hidden_states=hs)
            z_L = hs  # after all layers

        # Outer cycle: z_H update (uses same L_level)
        hs = z_H + z_L  # input_injection for z_H update
        for layer_idx, layer in enumerate(L_level.layers):
            if hasattr(layer, "self_attn"):
                aw = compute_attention_weights_manual(layer.self_attn, hs, cos_sin)
                if puzzle_emb_len > 0:
                    aw = aw[:, :, puzzle_emb_len:, puzzle_emb_len:]
                key = f"zH_T{T+1}_layer{layer_idx}"
                attn_patterns[key] = aw.cpu()
            hs = layer(cos_sin=cos_sin, hidden_states=hs)
        z_H = hs

    return attn_patterns


# ===================================================================
# Analysis
# ===================================================================

def compute_constraint_scores(
    attn_weights: torch.Tensor,
    masks: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    """For each head, compute fraction of attention on constraint-related cells.

    Args:
        attn_weights: (B, H, S, S) where S=81
        masks: constraint boolean masks, each (81, 81)

    Returns:
        Dict mapping constraint name to (B, H) array of scores.
    """
    aw = attn_weights.numpy()  # (B, H, 81, 81)
    scores = {}
    for name, mask in masks.items():
        # mask: (81, 81) -> broadcast over (B, H)
        constraint_mass = (aw * mask[None, None, :, :]).sum(axis=(-2, -1))
        total_mass = aw.sum(axis=(-2, -1))
        scores[name] = constraint_mass / (total_mass + 1e-12)
    return scores


def random_baseline_scores(masks: Dict[str, np.ndarray], seq_len: int = 81) -> Dict[str, float]:
    """Expected fraction of attention on constraint cells under uniform attention."""
    baselines = {}
    for name, mask in masks.items():
        baselines[name] = mask.sum() / (seq_len * (seq_len - 1))
    return baselines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-examples", type=int, default=100)
    parser.add_argument("--split", default="test")
    parser.add_argument("--batch-size", type=int, default=10)
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model using the shared helper from extract_activations
    print("Loading config and model...")
    model = load_trm_model(args.config, args.checkpoint, device, args.data_path, args.split)
    inner = model.inner
    puzzle_emb_len = inner.puzzle_emb_len

    print(f"Model loaded. H_cycles={model.config.H_cycles}, L_cycles={model.config.L_cycles}")
    print(f"L_layers={model.config.L_layers}, num_heads={model.config.num_heads}, head_dim={model.config.hidden_size // model.config.num_heads}")

    # Load data using the shared helper from extract_activations
    print("Loading data...")
    raw_data = load_test_data(args.data_path, args.split, max_examples=args.max_examples)
    n_examples = raw_data["inputs"].shape[0]
    constraint_masks = build_constraint_masks(81)
    baselines = random_baseline_scores(constraint_masks)
    print(f"Random baselines: {baselines}")

    # Accumulate scores across batches
    all_scores = defaultdict(lambda: defaultdict(list))

    for start in tqdm(range(0, n_examples, args.batch_size), desc="Extracting attention"):
        end = min(start + args.batch_size, n_examples)
        batch = {
            k: torch.tensor(raw_data[k][start:end], dtype=torch.long, device=device)
            for k in ("inputs", "labels", "puzzle_identifiers")
        }

        # Run full ACT to final step, extract attention at that step
        # For simplicity, extract attention from a single inner forward at the final ACT step.
        # First run ACT steps 1..15 normally, then extract at step 16.
        halt_max = model.config.halt_max_steps
        carry = inner.empty_carry(batch["inputs"].shape[0])
        carry.z_H = carry.z_H.to(device)
        carry.z_L = carry.z_L.to(device)
        halted = torch.ones(batch["inputs"].shape[0], dtype=torch.bool, device=device)

        for act_step in range(1, halt_max + 1):
            carry = inner.reset_carry(halted, carry)
            current_data = {
                k: torch.where(
                    halted.view((-1,) + (1,) * (batch[k].ndim - 1)),
                    batch[k], batch[k],
                ) for k in batch
            }

            if act_step < halt_max:
                # Normal forward (no extraction)
                seq_info = dict(cos_sin=inner.rotary_emb() if hasattr(inner, "rotary_emb") else None)
                input_emb = inner._input_embeddings(current_data["inputs"], current_data["puzzle_identifiers"])
                z_H, z_L = carry.z_H, carry.z_L
                for _T in range(inner.config.H_cycles):
                    for _i in range(inner.config.L_cycles):
                        z_L = inner.L_level(z_L, z_H + input_emb, **seq_info)
                    z_H = inner.L_level(z_H, z_L, **seq_info)
                carry = TinyRecursiveReasoningModel_ACTV1InnerCarry(z_H=z_H.detach(), z_L=z_L.detach())
                halted = torch.zeros_like(halted)
            else:
                # Final ACT step: extract attention patterns using carry from step 15
                # Reset carry for examples that halted (all should have halted=False here)
                patterns = extract_attention_patterns(inner, current_data, puzzle_emb_len, carry=carry)
                for key, aw in patterns.items():
                    scores = compute_constraint_scores(aw, constraint_masks)
                    for constraint_name, score_arr in scores.items():
                        all_scores[key][constraint_name].append(score_arr)

    # Aggregate scores: mean across batches
    print("\nAggregating scores across batches...")
    aggregated = {}
    for step_key in sorted(all_scores.keys()):
        aggregated[step_key] = {}
        for constraint_name in all_scores[step_key]:
            arr = np.concatenate(all_scores[step_key][constraint_name], axis=0)  # (total_B, H)
            mean_per_head = arr.mean(axis=0)  # (H,)
            aggregated[step_key][constraint_name] = mean_per_head.tolist()

    # Save raw results
    results_path = os.path.join(args.output_dir, "induction_head_scores.json")
    with open(results_path, "w") as f:
        json.dump({"baselines": baselines, "scores": aggregated}, f, indent=2)
    print(f"Saved scores to {results_path}")

    # ===================================================================
    # Plotting
    # ===================================================================
    num_heads = model.config.num_heads
    num_layers = model.config.L_layers
    H_cycles = model.config.H_cycles
    L_cycles = model.config.L_cycles

    # 1. Per-head constraint attention heatmap for z_L steps
    constraint_types = ["same_row", "same_col", "same_box", "any_constraint"]

    for ct in constraint_types:
        baseline = baselines[ct]

        # Build matrix: rows = (T, i) steps, cols = layer x head
        zL_keys = [f"zL_T{T}_i{i}_layer{l}"
                   for T in range(1, H_cycles + 1)
                   for i in range(1, L_cycles + 1)
                   for l in range(num_layers)]
        zL_keys = [k for k in zL_keys if k in aggregated]

        if not zL_keys:
            continue

        n_steps = H_cycles * L_cycles
        n_head_slots = num_layers * num_heads
        matrix = np.zeros((n_steps, n_head_slots))

        step_labels = []
        for T in range(1, H_cycles + 1):
            for i in range(1, L_cycles + 1):
                step_labels.append(f"({T},{i})")
                row_idx = (T - 1) * L_cycles + (i - 1)
                for l in range(num_layers):
                    key = f"zL_T{T}_i{i}_layer{l}"
                    if key in aggregated:
                        scores = aggregated[key][ct]
                        for h in range(num_heads):
                            col_idx = l * num_heads + h
                            matrix[row_idx, col_idx] = scores[h]

        head_labels = [f"L{l}H{h}" for l in range(num_layers) for h in range(num_heads)]

        fig, ax = plt.subplots(figsize=(14, 8))
        im = ax.imshow(matrix - baseline, aspect="auto", cmap="RdBu_r",
                       vmin=-0.1, vmax=0.1)
        ax.set_xlabel("Layer / Head")
        ax.set_ylabel("Recursion step (T, i)")
        ax.set_xticks(range(len(head_labels)))
        ax.set_xticklabels(head_labels, rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(len(step_labels)))
        ax.set_yticklabels(step_labels, fontsize=7)
        ax.set_title(f"Constraint attention: {ct} (relative to {baseline:.3f} baseline)")
        fig.colorbar(im, ax=ax, label="Attention fraction minus baseline")
        fig.tight_layout()
        fig.savefig(os.path.join(args.output_dir, f"induction_{ct}_heatmap.png"), dpi=150)
        plt.close(fig)

    # 2. Summary bar chart: top heads by any_constraint score
    if "any_constraint" in constraint_types:
        head_means = np.zeros(n_head_slots)
        count = 0
        for T in range(1, H_cycles + 1):
            for i in range(1, L_cycles + 1):
                for l in range(num_layers):
                    key = f"zL_T{T}_i{i}_layer{l}"
                    if key in aggregated:
                        scores = aggregated[key]["any_constraint"]
                        for h in range(num_heads):
                            head_means[l * num_heads + h] += scores[h]
                        count += 1
        head_means /= max(count, 1)

        baseline_any = baselines["any_constraint"]
        relative = head_means - baseline_any

        fig, ax = plt.subplots(figsize=(10, 5))
        colors = ["#e74c3c" if r > 0 else "#3498db" for r in relative]
        ax.bar(range(len(relative)), relative, color=colors, edgecolor="none")
        ax.axhline(0, color="k", linewidth=0.5)
        ax.set_xticks(range(len(head_labels)))
        ax.set_xticklabels(head_labels, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Mean attention on constraints minus baseline")
        ax.set_title(f"Constraint attention per head (baseline = {baseline_any:.3f})")
        fig.tight_layout()
        fig.savefig(os.path.join(args.output_dir, "induction_head_summary.png"), dpi=150)
        plt.close(fig)

    # 3. Attention heatmap for top constraint head on a single puzzle
    if zL_keys and n_examples > 0:
        top_head_idx = np.argmax(head_means)
        top_layer = top_head_idx // num_heads
        top_h = top_head_idx % num_heads
        print(f"\nTop constraint head: Layer {top_layer}, Head {top_h} "
              f"(mean constraint attn = {head_means[top_head_idx]:.4f}, "
              f"baseline = {baseline_any:.4f})")

        # Re-run single puzzle to get attention pattern for visualization
        single_batch = {
            k: torch.tensor(raw_data[k][0:1], dtype=torch.long, device=device)
            for k in ("inputs", "labels", "puzzle_identifiers")
        }
        patterns = extract_attention_patterns(inner, single_batch, puzzle_emb_len)
        key = f"zL_T{H_cycles}_i{L_cycles}_layer{top_layer}"
        if key in patterns:
            aw = patterns[key][0, top_h].numpy()  # (81, 81)
            fig, ax = plt.subplots(figsize=(8, 7))
            im = ax.imshow(aw, cmap="hot", aspect="equal")
            ax.set_xlabel("Key cell")
            ax.set_ylabel("Query cell")
            ax.set_title(f"Attention pattern: Layer {top_layer}, Head {top_h} at (T={H_cycles}, i={L_cycles})")

            # Overlay grid lines for 3x3 boxes
            for g in [9, 18, 27, 36, 45, 54, 63, 72]:
                ax.axhline(g - 0.5, color="cyan", linewidth=0.5, alpha=0.5)
                ax.axvline(g - 0.5, color="cyan", linewidth=0.5, alpha=0.5)
            for g in [27, 54]:
                ax.axhline(g - 0.5, color="cyan", linewidth=1.5)
                ax.axvline(g - 0.5, color="cyan", linewidth=1.5)

            fig.colorbar(im, ax=ax, label="Attention weight")
            fig.tight_layout()
            fig.savefig(os.path.join(args.output_dir, "induction_top_head_pattern.png"), dpi=150)
            plt.close(fig)

    print("\nDone. All plots saved to", args.output_dir)


if __name__ == "__main__":
    main()
