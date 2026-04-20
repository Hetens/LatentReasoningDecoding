"""
Extract TRM latent activations z_L and z_H at each recursion step (T, i).

Runs the full ACT inference loop and records intermediate latent states at
the first and final ACT steps.  Saves results as .pt files for downstream
probing, CKA, and activation-patching experiments.

Usage (from repo root):
    python -m experiments.probing.extract_activations \
        --config  trm_base/config_pretrain_paper.yml \
        --checkpoint checkpoints/.../step_50000.pt \
        --data-path data/sudoku-extreme-full \
        --output-dir results/probing/activations \
        --split test
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from typing import Dict, Optional, Set, Tuple

import numpy as np
import torch
import yaml
from tqdm import tqdm

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_TRM_BASE = os.path.join(_PROJECT_ROOT, "trm_base")
if _TRM_BASE not in sys.path:
    sys.path.insert(0, _TRM_BASE)

from layers import RotaryEmbedding  # noqa: E402
from trm import (  # noqa: E402
    IGNORE_LABEL_ID,
    TinyRecursiveReasoningModel_ACTV1,
    TinyRecursiveReasoningModel_ACTV1Config,
    TinyRecursiveReasoningModel_ACTV1Carry,
    TinyRecursiveReasoningModel_ACTV1InnerCarry,
    TinyRecursiveReasoningModel_ACTV1_Inner,
)
from losses import ACTLossHead  # noqa: E402
from pretrain import load_composed_config, PretrainConfig  # noqa: E402
from metadata import PuzzleDatasetMetadata  # noqa: E402


# ---------------------------------------------------------------------------
# Core extraction
# ---------------------------------------------------------------------------

@dataclass
class ExtractionResult:
    """Activations recorded during one inner forward pass."""
    z_L: torch.Tensor   # [H_cycles, L_cycles, batch, seq_len, hidden]
    z_H: torch.Tensor   # [H_cycles, batch, seq_len, hidden]
    logits: torch.Tensor # [batch, seq_len, vocab]


@torch.no_grad()
def inner_forward_with_extraction(
    inner: TinyRecursiveReasoningModel_ACTV1_Inner,
    carry: TinyRecursiveReasoningModel_ACTV1InnerCarry,
    batch: Dict[str, torch.Tensor],
) -> Tuple[TinyRecursiveReasoningModel_ACTV1InnerCarry, ExtractionResult]:
    """Run one full inner forward pass recording z_L at every (T, i) and z_H
    after every outer cycle T.

    Mirrors ``TinyRecursiveReasoningModel_ACTV1_Inner.forward`` but captures
    intermediate states instead of discarding them.
    """
    cfg = inner.config
    seq_info = dict(
        cos_sin=inner.rotary_emb() if hasattr(inner, "rotary_emb") else None,
    )
    input_embeddings = inner._input_embeddings(
        batch["inputs"], batch["puzzle_identifiers"],
    )

    z_H = carry.z_H
    z_L = carry.z_L
    B, S, D = z_H.shape

    z_L_all = torch.empty(
        cfg.H_cycles, cfg.L_cycles, B, S, D,
        dtype=z_H.dtype, device=z_H.device,
    )
    z_H_all = torch.empty(
        cfg.H_cycles, B, S, D,
        dtype=z_H.dtype, device=z_H.device,
    )

    for T in range(cfg.H_cycles):
        for i in range(cfg.L_cycles):
            z_L = inner.L_level(z_L, z_H + input_embeddings, **seq_info)
            z_L_all[T, i] = z_L
        z_H = inner.L_level(z_H, z_L, **seq_info)
        z_H_all[T] = z_H

    new_carry = TinyRecursiveReasoningModel_ACTV1InnerCarry(
        z_H=z_H.detach(), z_L=z_L.detach(),
    )
    logits = inner.lm_head(z_H)[:, inner.puzzle_emb_len:]

    return new_carry, ExtractionResult(
        z_L=z_L_all,
        z_H=z_H_all,
        logits=logits,
    )


@torch.no_grad()
def run_act_with_extraction(
    model: TinyRecursiveReasoningModel_ACTV1,
    batch: Dict[str, torch.Tensor],
    device: torch.device,
    extract_at_steps: Optional[Set[int]] = None,
) -> Dict[int, ExtractionResult]:
    """Run the full ACT inference loop, extracting latents at selected steps.

    Args:
        model: The TRM model (unwrapped from ACTLossHead).
        batch: Dict with keys ``inputs``, ``labels``, ``puzzle_identifiers``,
               each a tensor on *device*.
        device: Torch device.
        extract_at_steps: 1-indexed ACT step numbers at which to extract.
            Defaults to ``{1, halt_max_steps}``.

    Returns:
        Mapping from ACT step number to :class:`ExtractionResult`.
    """
    halt_max = model.config.halt_max_steps
    if extract_at_steps is None:
        extract_at_steps = {1, halt_max}

    batch_size = batch["inputs"].shape[0]
    inner = model.inner

    # Initialise ACT carry (all halted → will be reset on step 1).
    inner_carry = inner.empty_carry(batch_size)
    inner_carry.z_H = inner_carry.z_H.to(device)
    inner_carry.z_L = inner_carry.z_L.to(device)
    steps = torch.zeros(batch_size, dtype=torch.int32, device=device)
    halted = torch.ones(batch_size, dtype=torch.bool, device=device)
    current_data = {k: torch.empty_like(v) for k, v in batch.items()}

    results: Dict[int, ExtractionResult] = {}

    for act_step in range(1, halt_max + 1):
        # Reset carry for examples that halted on the previous step.
        inner_carry = inner.reset_carry(halted, inner_carry)
        steps = torch.where(halted, torch.zeros_like(steps), steps)
        current_data = {
            k: torch.where(
                halted.view((-1,) + (1,) * (batch[k].ndim - 1)),
                batch[k], v,
            )
            for k, v in current_data.items()
        }

        if act_step in extract_at_steps:
            inner_carry, ext = inner_forward_with_extraction(
                inner, inner_carry, current_data,
            )
            results[act_step] = ext
        else:
            seq_info = dict(
                cos_sin=inner.rotary_emb() if hasattr(inner, "rotary_emb") else None,
            )
            input_emb = inner._input_embeddings(
                current_data["inputs"], current_data["puzzle_identifiers"],
            )
            z_H, z_L = inner_carry.z_H, inner_carry.z_L
            for _T in range(inner.config.H_cycles):
                for _i in range(inner.config.L_cycles):
                    z_L = inner.L_level(z_L, z_H + input_emb, **seq_info)
                z_H = inner.L_level(z_H, z_L, **seq_info)
            inner_carry = TinyRecursiveReasoningModel_ACTV1InnerCarry(
                z_H=z_H.detach(), z_L=z_L.detach(),
            )

        steps = steps + 1
        halted = steps >= halt_max

    return results


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_trm_model(
    config_path: str,
    checkpoint_path: str,
    device: torch.device,
    data_path: str,
    split: str = "test",
) -> TinyRecursiveReasoningModel_ACTV1:
    """Reconstruct a TRM model from its YAML config and load a checkpoint."""

    raw_config = load_composed_config(config_path)
    pretrain_cfg = PretrainConfig.model_validate(raw_config)

    meta_file = os.path.join(data_path, split, "dataset.json")
    with open(meta_file, "r") as f:
        meta = PuzzleDatasetMetadata(**json.load(f))

    world_size = 1
    model_cfg = dict(
        **pretrain_cfg.arch.__pydantic_extra__,
        batch_size=pretrain_cfg.global_batch_size // world_size,
        vocab_size=meta.vocab_size,
        seq_len=meta.seq_len,
        num_puzzle_identifiers=meta.num_puzzle_identifiers,
    )

    with torch.device(device):
        model = TinyRecursiveReasoningModel_ACTV1(model_cfg)

    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)

    # Strip torch.compile ``_orig_mod.`` prefix and the ACTLossHead ``model.``
    # wrapper so the keys align with the bare TRM.
    cleaned: dict = {}
    for k, v in state_dict.items():
        new_k = k.replace("_orig_mod.", "")
        if new_k.startswith("model."):
            new_k = new_k[len("model."):]
        cleaned[new_k] = v

    model.load_state_dict(cleaned, strict=False)
    model.to(device).eval()
    return model


# ---------------------------------------------------------------------------
# Data loading (lightweight — no PuzzleDataset dependency)
# ---------------------------------------------------------------------------

def load_test_data(
    data_path: str,
    split: str = "test",
    max_examples: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """Load inputs, labels, and puzzle_identifiers from the on-disk split."""
    base = os.path.join(data_path, split)

    meta_file = os.path.join(base, "dataset.json")
    with open(meta_file) as f:
        meta = json.load(f)
    set_name = meta["sets"][0]

    data = {}
    for field in ("inputs", "labels", "puzzle_identifiers"):
        arr = np.load(os.path.join(base, f"{set_name}__{field}.npy"))
        data[field] = arr

    if max_examples is not None and max_examples < len(data["inputs"]):
        for k in data:
            data[k] = data[k][:max_examples]

    return data


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract TRM latent activations at each (T, i) recursion step.",
    )
    parser.add_argument("--config", required=True, help="Path to pretrain YAML config.")
    parser.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint.")
    parser.add_argument("--data-path", required=True, help="Path to processed Sudoku data dir.")
    parser.add_argument("--output-dir", required=True, help="Directory to save activations.")
    parser.add_argument("--split", default="test", help="Data split (default: test).")
    parser.add_argument("--max-examples", type=int, default=None, help="Cap on number of examples.")
    parser.add_argument("--batch-size", type=int, default=128, help="Inference batch size.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading model …")
    model = load_trm_model(args.config, args.checkpoint, device, args.data_path, args.split)
    cfg = model.config
    print(f"  H_cycles={cfg.H_cycles}  L_cycles={cfg.L_cycles}  hidden={cfg.hidden_size}")
    halt_max = cfg.halt_max_steps
    print(f"  halt_max_steps={halt_max}")

    print("Loading data …")
    raw = load_test_data(args.data_path, args.split, args.max_examples)
    N = len(raw["inputs"])
    print(f"  {N} examples loaded.")

    puzzle_emb_len = model.inner.puzzle_emb_len
    seq_len = cfg.seq_len  # 81

    extract_steps = {1, halt_max}

    # Pre-allocate output tensors (float16 to save disk).
    z_L_out = {
        s: torch.empty(N, cfg.H_cycles, cfg.L_cycles, seq_len, cfg.hidden_size, dtype=torch.float16)
        for s in extract_steps
    }
    z_H_out = {
        s: torch.empty(N, cfg.H_cycles, seq_len, cfg.hidden_size, dtype=torch.float16)
        for s in extract_steps
    }

    print("Extracting activations …")
    bs = args.batch_size
    for start in tqdm(range(0, N, bs)):
        end = min(start + bs, N)
        batch = {
            "inputs": torch.from_numpy(raw["inputs"][start:end].astype(np.int32)).to(device),
            "labels": torch.from_numpy(raw["labels"][start:end].astype(np.int32)).to(device),
            "puzzle_identifiers": torch.from_numpy(
                raw["puzzle_identifiers"][start:end].astype(np.int32)
            ).to(device),
        }

        results = run_act_with_extraction(model, batch, device, extract_steps)

        for step, ext in results.items():
            # Strip puzzle-embedding prefix positions (if any).
            zl = ext.z_L[:, :, :, puzzle_emb_len:, :]  # [H, L, B, 81, D]
            zh = ext.z_H[:, :, puzzle_emb_len:, :]      # [H, B, 81, D]
            # Permute to [B, H, L, 81, D] / [B, H, 81, D] and store.
            z_L_out[step][start:end] = zl.permute(2, 0, 1, 3, 4).cpu().half()
            z_H_out[step][start:end] = zh.permute(1, 0, 2, 3).cpu().half()

    # Save
    for step in extract_steps:
        tag = f"act{step}"
        torch.save(z_L_out[step], os.path.join(args.output_dir, f"z_L_{tag}.pt"))
        torch.save(z_H_out[step], os.path.join(args.output_dir, f"z_H_{tag}.pt"))
        print(f"Saved z_L_{tag}.pt  shape={tuple(z_L_out[step].shape)}")
        print(f"Saved z_H_{tag}.pt  shape={tuple(z_H_out[step].shape)}")

    # Also save the raw inputs for candidate-set computation.
    np.save(os.path.join(args.output_dir, "inputs.npy"), raw["inputs"])
    np.save(os.path.join(args.output_dir, "labels.npy"), raw["labels"])
    print("Done.")


if __name__ == "__main__":
    main()
