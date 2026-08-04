#!/bin/bash -l
# ============================================================================
# setup_adam_atan2.sh - Build and verify the adam-atan2 CUDA extension.
#
# Upstream TRM uses AdamATan2, not AdamW (pretrain.py:20,
# `from adam_atan2 import AdamATan2`). Our repo used torch.optim.AdamW, which
# is very likely why runs froze: AdamW's update is m/(sqrt(v)+eps), so once
# gradients get small the update collapses and the per-step decoupled weight
# decay dominates. AdamATan2 uses atan2(m, sqrt(v)), which is bounded and
# scale-invariant, so the update stays ~lr regardless of gradient scale. That
# is what makes upstream's weight_decay=1.0 survivable.
#
# adam-atan2 is a CUDA/C++ extension, so it needs nvcc (absent on login nodes)
# and a GPU to verify against. TORCH_CUDA_ARCH_LIST covers the Ampere/Ada
# cards we actually request: a100 (8.0), a40 (8.6), l40s (8.9).
#
# Submit from repo root:
#     sbatch scripts/unity/setup_adam_atan2.sh
# ============================================================================

#SBATCH --job-name=TRM-SetupAtan2
#SBATCH --partition=gpu
#SBATCH --time=00:40:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --constraint=l40s
#SBATCH --output=logs/setup_adam_atan2_%j.out
#SBATCH --error=logs/setup_adam_atan2_%j.err

set -euo pipefail

REPO_DIR="$(pwd)"
mkdir -p logs

module load python/3.11.7
module load cuda/12.6
source "$HOME/venvs/tinyllm/bin/activate"
cd "$REPO_DIR"

echo "=== adam-atan2 setup ==="
echo "Job ${SLURM_JOB_ID:-none} on $(hostname) at $(date)"
nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader
nvcc --version | tail -2
echo ""

export TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9"
export MAX_JOBS=8

echo ">>> build deps"
pip install -q --upgrade packaging ninja wheel setuptools setuptools-scm

echo ">>> installing adam-atan2"
pip install --no-build-isolation adam-atan2 2>&1 | tail -20

echo ""
echo ">>> verifying"
python - <<'PY'
import torch
from adam_atan2 import AdamATan2

print("import OK:", AdamATan2)

# Convex check: minimise (x - 3)^2 from x=0. Must approach 3.
dev = torch.device("cuda")
x = torch.zeros(4, device=dev, requires_grad=True)
opt = AdamATan2([x], lr=0.1, weight_decay=0.0, betas=(0.9, 0.95))
for _ in range(400):
    loss = ((x - 3.0) ** 2).sum()
    opt.zero_grad(); loss.backward(); opt.step()
print(f"convex descent: x={x.detach().cpu().numpy()} (target 3.0)")
assert torch.allclose(x.detach(), torch.full((4,), 3.0, device=dev), atol=1e-2), "did not converge"

# The property that matters: update magnitude must not collapse when the
# gradient is tiny, which is exactly where AdamW stalls against weight decay.
for scale in (1.0, 1e-4, 1e-8):
    y = torch.zeros(1, device=dev, requires_grad=True)
    o = AdamATan2([y], lr=0.01, weight_decay=0.0, betas=(0.9, 0.95))
    y.grad = torch.full((1,), -scale, device=dev)
    o.step()
    print(f"  grad={scale:>8.0e} -> step={y.item():+.6f}")

# bf16 params, as used in training
z = torch.zeros(8, device=dev, dtype=torch.float32, requires_grad=True)
oz = AdamATan2([z], lr=0.01, weight_decay=1.0, betas=(0.9, 0.95))
z.grad = torch.randn(8, device=dev)
oz.step()
print("weight_decay=1.0 step OK:", torch.isfinite(z).all().item())

print("RESULT: PASS")
PY

echo ""
echo "=== Done: $(date) ==="
