"""
Component-level ablation: localize which computation (attention vs. MLP,
and which heads) carries the model's output and the late-cycle causal load.

Requested by all three reviewers: the paper argues by elimination that the
multi-phase restructuring must originate in MLP/value computation because
attention routing is static. This script tests that directly.

Conditions (each scoped to outer cycle T=1, T=2, T=3, or all cycles, and
applied during the final ACT step to both the z_L and z_H updates of the
scoped cycles, since the block is weight-shared across both):
  - attn_L{0,1}: remove the attention residual contribution of block 0/1.
  - mlp_L{0,1}:  remove the MLP residual contribution of block 0/1.
  - head_L{l}H{h}: remove one head's slice before o_proj, for the paper's
    constraint-routing heads (L0H7, L1H0, L1H4, L1H6, L1H7).
  - heads_joint: all five constraint-routing heads at once.

Ablation modes:
  - zero: replace the component output with zeros.
  - mean: replace with the clean-run batch-mean output at the same call
    site (controls for off-distribution shift from zeroing).

Metrics per condition: DeltaCE and DeltaAcc vs. the clean run over the
same puzzles, with bootstrap CIs (protocol matches activation patching).

Usage (from repo root):
    python -m experiments.probing.component_ablation \
        --config trm_base/config_pretrain_paper.yml \
        --checkpoint "checkpoints/.../step_65100.pt" \
        --data-path data/sudoku-extreme-1k-aug-1000 \
        --output-dir results/probing/component_ablation \
        --n-puzzles 200 --batch-size 32 --seed 0 --mode zero
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

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
from experiments.probing.activation_patching import (  # noqa: E402
    per_cell_cross_entropy,
)
from experiments.probing.activation_patching_sweep import (  # noqa: E402
    _warmup_carry,
    _masked_mean,
    _per_cell_correct,
    _bootstrap_mean_ci,
)

CONSTRAINT_HEADS: List[Tuple[int, int]] = [(0, 7), (1, 0), (1, 4), (1, 6), (1, 7)]
HEAD_DIM = 64


class AblationController:
    """Mutable state read by the hooks.

    type: None | 'attn' | 'mlp' | 'heads'
    layer: block index the ablation applies to (ignored for 'heads', which
           carries (layer, head) pairs in `heads`).
    active: toggled by the manual recursion loop per scoped cycle.
    capture: when True, hooks record clean batch-mean outputs per call site.
    call_site: incremented by the loop for every L_level invocation so that
           mean-ablation replays the mean captured at the same site.
    """

    def __init__(self) -> None:
        self.type: Optional[str] = None
        self.layer: int = -1
        self.heads: List[Tuple[int, int]] = []
        self.active = False
        self.capture = False
        self.mode = "zero"
        self.call_site = -1
        self.means: Dict[Tuple[str, int, int], torch.Tensor] = {}


def _register_hooks(inner, ctrl: AblationController):
    handles = []
    layers = inner.L_level.layers

    for l_idx, block in enumerate(layers):
        def attn_hook(module, args, output, l_idx=l_idx):
            key = ("attn", l_idx, ctrl.call_site)
            if ctrl.capture:
                ctrl.means[key] = output.mean(dim=0, keepdim=True).detach()
                return None
            if ctrl.active and ctrl.type == "attn" and ctrl.layer == l_idx:
                if ctrl.mode == "mean" and key in ctrl.means:
                    return ctrl.means[key].expand_as(output).clone()
                return torch.zeros_like(output)
            return None

        def mlp_hook(module, args, output, l_idx=l_idx):
            key = ("mlp", l_idx, ctrl.call_site)
            if ctrl.capture:
                ctrl.means[key] = output.mean(dim=0, keepdim=True).detach()
                return None
            if ctrl.active and ctrl.type == "mlp" and ctrl.layer == l_idx:
                if ctrl.mode == "mean" and key in ctrl.means:
                    return ctrl.means[key].expand_as(output).clone()
                return torch.zeros_like(output)
            return None

        def oproj_pre_hook(module, args, l_idx=l_idx):
            if not (ctrl.active and ctrl.type == "heads"):
                return None
            x = args[0]
            target = [h for (ll, h) in ctrl.heads if ll == l_idx]
            if not target:
                return None
            x = x.clone()
            for h in target:
                x[..., h * HEAD_DIM: (h + 1) * HEAD_DIM] = 0.0
            return (x,)

        handles.append(block.self_attn.register_forward_hook(attn_hook))
        handles.append(block.mlp.register_forward_hook(mlp_hook))
        handles.append(block.self_attn.o_proj.register_forward_pre_hook(oproj_pre_hook))
    return handles


@torch.no_grad()
def _final_step_forward(
    inner,
    carry,
    batch: Dict[str, torch.Tensor],
    ctrl: AblationController,
    scope_T: Optional[int],
) -> torch.Tensor:
    """Final ACT step with the controller toggled on for the scoped cycle(s).

    scope_T: 0-indexed outer cycle, or None for all cycles."""
    cfg = inner.config
    seq_info = dict(cos_sin=inner.rotary_emb() if hasattr(inner, "rotary_emb") else None)
    input_emb = inner._input_embeddings(batch["inputs"], batch["puzzle_identifiers"])

    z_H, z_L = carry.z_H, carry.z_L
    site = 0
    for T in range(cfg.H_cycles):
        in_scope = scope_T is None or T == scope_T
        for i in range(cfg.L_cycles):
            ctrl.active = in_scope
            ctrl.call_site = site
            z_L = inner.L_level(z_L, z_H + input_emb, **seq_info)
            site += 1
        ctrl.active = in_scope
        ctrl.call_site = site
        z_H = inner.L_level(z_H, z_L, **seq_info)
        site += 1
    ctrl.active = False
    return inner.lm_head(z_H)[:, inner.puzzle_emb_len:]


def main() -> None:
    parser = argparse.ArgumentParser(description="Component-level ablations.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--n-puzzles", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--mode", choices=["zero", "mean"], default="zero")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    print("Loading model ...")
    model = load_trm_model(args.config, args.checkpoint, device, args.data_path, args.split)
    model.eval()
    inner = model.inner
    cfg = model.config
    n_T = cfg.H_cycles

    ctrl = AblationController()
    ctrl.mode = args.mode
    handles = _register_hooks(inner, ctrl)

    print("Loading data ...")
    raw = load_test_data(args.data_path, args.split, max_examples=args.n_puzzles)
    N = len(raw["inputs"])
    n_puzzles = min(args.n_puzzles, N)
    # Same puzzle selection recipe as the patching scripts at the same seed.
    order = rng.permutation(N)
    use_idx = order[:n_puzzles]

    # Conditions: (name, type, layer, heads)
    conditions = []
    for l in (0, 1):
        conditions.append((f"attn_L{l}", "attn", l, []))
        conditions.append((f"mlp_L{l}", "mlp", l, []))
    for (l, h) in CONSTRAINT_HEADS:
        conditions.append((f"head_L{l}H{h}", "heads", -1, [(l, h)]))
    conditions.append(("heads_joint", "heads", -1, list(CONSTRAINT_HEADS)))

    scopes = [(f"T{t+1}", t) for t in range(n_T)] + [("all", None)]

    d_ce = {(c[0], s[0]): [] for c in conditions for s in scopes}
    d_acc = {(c[0], s[0]): [] for c in conditions for s in scopes}

    bs = args.batch_size
    for start in tqdm(range(0, n_puzzles, bs), desc="batches"):
        end = min(start + bs, n_puzzles)
        idx = use_idx[start:end]
        batch = {
            "inputs": torch.from_numpy(raw["inputs"][idx].astype(np.int32)).to(device),
            "labels": torch.from_numpy(raw["labels"][idx].astype(np.int32)).to(device),
            "puzzle_identifiers": torch.from_numpy(
                raw["puzzle_identifiers"][idx].astype(np.int32)
            ).to(device),
        }
        labels = batch["labels"]
        mask = labels != -100

        carry = _warmup_carry(inner, batch, device)

        # Clean pass; in mean mode this also captures per-site means.
        ctrl.capture = args.mode == "mean"
        ctrl.means = {}
        clean_logits = _final_step_forward(inner, carry, batch, ctrl, scope_T=None).float()
        ctrl.capture = False

        ce_clean = _masked_mean(per_cell_cross_entropy(clean_logits, labels), mask)
        acc_clean = _masked_mean(_per_cell_correct(clean_logits, labels), mask)

        for (name, ctype, layer, heads) in conditions:
            ctrl.type = ctype
            ctrl.layer = layer
            ctrl.heads = heads
            for (sname, sT) in scopes:
                logits = _final_step_forward(inner, carry, batch, ctrl, scope_T=sT).float()
                ce = _masked_mean(per_cell_cross_entropy(logits, labels), mask)
                acc = _masked_mean(_per_cell_correct(logits, labels), mask)
                d_ce[(name, sname)].extend((ce - ce_clean).cpu().tolist())
                d_acc[(name, sname)].extend((acc - acc_clean).cpu().tolist())
        ctrl.type = None

    for h in handles:
        h.remove()

    # ---- Aggregate, report, save ----
    results = {
        "n_puzzles": n_puzzles,
        "seed": args.seed,
        "mode": args.mode,
        "checkpoint": args.checkpoint,
        "conditions": [],
    }
    print(f"\n{'condition':>14} {'scope':>6} {'dCE':>22} {'dAcc':>10}")
    for (name, _t, _l, _h) in conditions:
        for (sname, _sT) in scopes:
            ce_arr = np.array(d_ce[(name, sname)])
            acc_arr = np.array(d_acc[(name, sname)])
            lo, hi = _bootstrap_mean_ci(ce_arr, seed=args.seed + 1)
            alo, ahi = _bootstrap_mean_ci(acc_arr, seed=args.seed + 1)
            entry = {
                "condition": name,
                "scope": sname,
                "delta_ce_mean": float(ce_arr.mean()),
                "delta_ce_ci": [lo, hi],
                "delta_acc_mean": float(acc_arr.mean()),
                "delta_acc_ci": [alo, ahi],
            }
            results["conditions"].append(entry)
            print(
                f"{name:>14} {sname:>6}"
                f"  {entry['delta_ce_mean']:+.3f} [{lo:+.3f},{hi:+.3f}]"
                f"  {entry['delta_acc_mean']:+.4f}"
            )

    out_json = os.path.join(args.output_dir, f"component_ablation_{args.mode}.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved -> {out_json}")


if __name__ == "__main__":
    main()
