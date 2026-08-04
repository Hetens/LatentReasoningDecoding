"""Find the largest maze batch that fits, and its throughput."""
import json, sys, time, torch
sys.path.insert(0, "trm_base"); sys.path.insert(0, ".")
from pretrain import load_composed_config, apply_overrides, PretrainConfig, create_model
from puzzle_dataset import PuzzleDatasetMetadata

meta = PuzzleDatasetMetadata(**json.load(open("data/maze-30x30-hard-1k/train/dataset.json")))
dev = torch.device("cuda")
print(f"GPU: {torch.cuda.get_device_name(0)}  {torch.cuda.get_device_properties(0).total_memory/2**30:.1f} GiB")

for B in (64, 128, 256, 384, 512, 768):
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    try:
        raw = apply_overrides(load_composed_config("trm_base/config_pretrain_maze.yml"),
                              [f"global_batch_size={B}", "run_name=PROBE"])
        cfg = PretrainConfig.model_validate(raw); cfg.checkpoint_path = None
        model, opts, lrs = create_model(cfg, meta, rank=0, world_size=1)
        batch = {
            "inputs": torch.randint(1, 6, (B, 900), device=dev, dtype=torch.int32),
            "labels": torch.randint(1, 6, (B, 900), device=dev, dtype=torch.int32),
            "puzzle_identifiers": torch.zeros(B, device=dev, dtype=torch.int32),
        }
        # initial_carry allocates on the ambient device, so scope it
        # exactly as train_batch does or the carry lands on CPU.
        with torch.device(dev):
            carry = model.initial_carry(batch)
        t0 = None
        for it in range(6):
            if it == 2:
                torch.cuda.synchronize(); t0 = time.time()
            carry, loss, metrics, _, _ = model(carry=carry, batch=batch, return_keys=[])
            ((1 / B) * loss).backward()
            for o in opts: o.step(); o.zero_grad()
        torch.cuda.synchronize()
        dt = (time.time() - t0) / 4
        peak = torch.cuda.max_memory_allocated() / 2**30
        print(f"  B={B:>4}  OK   {dt*1000:7.0f} ms/step  {B/dt:7.0f} samples/s  peak {peak:5.1f} GiB", flush=True)
        del model, opts, carry, batch, loss
    except torch.cuda.OutOfMemoryError:
        print(f"  B={B:>4}  OOM", flush=True)
        break
    except Exception as e:
        print(f"  B={B:>4}  ERR {type(e).__name__}: {str(e)[:200]}", flush=True)
        break
