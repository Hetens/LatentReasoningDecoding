"""Verify grad accumulation == one big batch, and that step/EMA bookkeeping is right.

Runs on CPU-sized fake maze batches. The check that matters: gradients after
6 accumulated micro-batches of 8 must equal gradients from one batch of 48,
because train_batch scales every micro loss by 1/global_batch_size.
"""
import json, sys, copy, torch
sys.path.insert(0, "trm_base"); sys.path.insert(0, ".")
from pretrain import (load_composed_config, apply_overrides, PretrainConfig,
                      create_model, resolve_batching, TrainState, train_batch)
from puzzle_dataset import PuzzleDatasetMetadata

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEQ, VOCAB = 60, 6
META = PuzzleDatasetMetadata(pad_id=0, ignore_label_id=0, blank_identifier_id=0,
                             vocab_size=VOCAB, seq_len=SEQ, num_puzzle_identifiers=1,
                             total_groups=48, mean_puzzle_examples=1.0,
                             total_puzzles=48, sets=["all"])

GLOBAL, MICRO = 48, 8
ACCUM = GLOBAL // MICRO


def make_cfg(micro):
    raw = apply_overrides(load_composed_config("trm_base/config_pretrain_maze_v2.yml"), [
        f"global_batch_size={GLOBAL}",
        f"micro_batch_size={micro}" if micro else "seed=0",
        "run_name=TEST", "epochs=48", "eval_interval=48",
        "arch.hidden_size=64", "arch.num_heads=2", "arch.L_layers=1",
        "arch.H_cycles=1", "arch.L_cycles=1", "arch.halt_max_steps=2",
        "arch.forward_dtype=float32", "ema=false",
    ])
    cfg = PretrainConfig.model_validate(raw)
    cfg.checkpoint_path = None
    return cfg


def fixed_batches(n, bs, seed=0):
    g = torch.Generator().manual_seed(seed)
    out = []
    for _ in range(n):
        out.append({
            "inputs": torch.randint(1, VOCAB, (bs, SEQ), generator=g).to(torch.int32),
            "labels": torch.randint(1, VOCAB, (bs, SEQ), generator=g).to(torch.int32),
            "puzzle_identifiers": torch.zeros(bs, dtype=torch.int32),
        })
    return out


def run(micro, batches):
    torch.manual_seed(1234)
    cfg = make_cfg(micro)
    m, opts, lrs = create_model(cfg, META, rank=0, world_size=1)
    _, accum = resolve_batching(cfg)
    ts = TrainState(model=m, optimizers=opts, optimizer_lrs=lrs, carry=None,
                    step=0, total_steps=100, grad_accum=accum)
    grads, steps = None, 0
    for b in batches:
        b = {k: v.to(DEV) for k, v in b.items()}
        _, stepped = train_batch(cfg, ts, b, rank=0, world_size=1)
        if stepped:
            steps += 1
            # grads are zeroed by optim.step(); capture params instead
            grads = {n: p.detach().clone() for n, p in m.named_parameters()}
    return ts, grads, steps


# One batch of 48, no accumulation
big = fixed_batches(1, GLOBAL, seed=7)
ts_big, p_big, steps_big = run(None, big)

# Same 48 rows split into 6 micro-batches of 8
micro_batches = [{k: v[i * MICRO:(i + 1) * MICRO] for k, v in big[0].items()}
                 for i in range(ACCUM)]
ts_acc, p_acc, steps_acc = run(MICRO, micro_batches)

print(f"grad_accum          : {ts_acc.grad_accum} (expected {ACCUM})")
print(f"optimizer steps     : big={steps_big}  accum={steps_acc}  (both must be 1)")
print(f"train_state.step    : big={ts_big.step}  accum={ts_acc.step}")
print(f"micro_step reset    : {ts_acc.micro_step} (expected 0)")

worst, worst_name = 0.0, None
for n in p_big:
    d = (p_big[n] - p_acc[n]).abs().max().item()
    if d > worst:
        worst, worst_name = d, n
print(f"max |param diff|    : {worst:.3e}  ({worst_name})")
print("RESULT:", "PASS" if (worst < 1e-5 and steps_big == steps_acc == 1
                            and ts_acc.micro_step == 0) else "FAIL")
