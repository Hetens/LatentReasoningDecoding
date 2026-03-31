# TinyLLM — Spatial Reasoning in Tiny Recursive Models

This repository contains two transformer implementations trained on Sudoku puzzles, plus an **ablation study** to identify the architectural source of spatial reasoning in the Tiny Recursive Model (TRM).

## Repository Structure

```
TinyLLM/
├── core/                        # Demo Transformer (GPT-2 style)
│   ├── attention.py             #   Causal multi-head attention, KV-cache
│   ├── config.py                #   Config & TransformerTrainingArgs
│   ├── layers.py                #   Embed, PosEmbed (learned), LayerNorm, Unembed
│   ├── mlp.py                   #   GELU feed-forward
│   ├── sampler.py               #   Autoregressive generation
│   ├── trainer.py               #   Training loop
│   └── transformer.py           #   DemoTransformer model
│
├── trm_base/                    # Tiny Recursive Model (TRM / ACT v1)
│   ├── arch/trm.yml             #   Architecture config (RoPE, hidden size, …)
│   ├── trm.py                   #   TinyRecursiveReasoningModel_ACTV1
│   ├── layers.py                #   CastedLinear, CastedEmbedding, RoPE, Attention, SwiGLU
│   ├── losses.py                #   ACTLossHead, stablemax cross-entropy
│   ├── common.py                #   Truncated-normal init
│   ├── sparse_embedding.py      #   CastedSparseEmbedding + SignSGD optimizer
│   ├── pretrain.py              #   Training loop (single-GPU / DDP)
│   ├── puzzle_dataset.py        #   PuzzleDataset (iterable, group-based batching)
│   ├── build_sdku_data.py       #   Builds on-disk Sudoku data from HuggingFace
│   ├── metadata.py              #   Dataset metadata types
│   ├── functions.py             #   Dynamic model-class loader
│   └── config_pretrain.yml      #   Default pretrain hyperparameters
│
├── sudoku/                      # Sudoku task (data + tokenizer for demo transformer)
│   ├── main_sudoku.py           #   Train DemoTransformer on Sudoku
│   └── sudoku_tokenizer.py      #   11-token Sudoku tokenizer
│
├── experiments/ablation/        # ** Ablation study — RoPE vs CastedLinear **
│   ├── configs/
│   │   ├── arch/trm.yml         #   Base TRM arch (copied from trm_base)
│   │   ├── baseline_rope_casted.yml
│   │   ├── ablation_learned_casted.yml
│   │   ├── ablation_rope_standard.yml
│   │   └── ablation_learned_standard.yml
│   └── run_ablation.py          #   Experiment runner
│
├── scripts/                     # Entry-point scripts
│   ├── train_demo.py            #   Train DemoTransformer on TinyStories
│   └── train_sudoku.py          #   Train DemoTransformer on Sudoku
│
├── docs/                        # Documentation & notes
├── archive/                     # Legacy / old code
├── data/                        # On-disk datasets (gitignored)
├── checkpoints/                 # Saved model weights (gitignored)
└── requirements.txt
```

## Key Architectural Differences

| Aspect | Demo Transformer (`core/`) | TRM (`trm_base/`) |
|---|---|---|
| **Positional encoding** | Learned absolute (`PosEmbed`) | **RoPE** (default) or learned |
| **Linear layers** | Standard `nn.Parameter` matmuls | **CastedLinear** (dtype-casting + trunc-normal init) |
| **Embeddings** | Standard `nn.Parameter` | **CastedEmbedding** (dtype-casting) |
| **Attention** | Causal, per-head W_Q/W_K/W_V/W_O | Non-causal, fused QKV, `scaled_dot_product_attention` |
| **FFN** | 2-layer GELU MLP | **SwiGLU** |
| **Normalization** | Pre-norm LayerNorm | Post-norm **RMS norm** (no learned params) |
| **Recursion** | Single forward pass | **H/L cycles** with carry + **ACT halting** |
| **Task** | Autoregressive next-token | Non-autoregressive (predict all 81 cells in parallel) |

## Ablation Study: Source of Spatial Reasoning

The TRM achieves spatial reasoning on Sudoku. Two candidate mechanisms are:

1. **RoPE** — Rotary Position Embeddings inject relative-position information directly into Q/K dot products, potentially encoding spatial structure of the 9×9 grid.
2. **CastedLinear** — dtype-casting linear layers with truncated-normal initialization may provide better gradient flow or numerical stability that aids spatial learning.

### Experiment Matrix

| Experiment | Positional Encoding | Linear Layer | What it tests |
|---|---|---|---|
| `baseline_rope_casted` | RoPE | CastedLinear | Control (default TRM) |
| `ablation_learned_casted` | **Learned** | CastedLinear | Does removing RoPE hurt spatial reasoning? |
| `ablation_rope_standard` | RoPE | **nn.Linear** | Does removing Casted layers hurt spatial reasoning? |
| `ablation_learned_standard` | **Learned** | **nn.Linear** | Does removing both degrade further? |

### Running the Ablation

```bash
# Run all four experiments sequentially
python experiments/ablation/run_ablation.py

# Run a single experiment
python experiments/ablation/run_ablation.py --experiment ablation_learned_casted

# Dry-run (print commands only)
python experiments/ablation/run_ablation.py --dry-run
```

Results are logged to the W&B project `ablation-spatial-reasoning`. Compare `exact_accuracy` across runs to determine which component drives spatial reasoning.

### Interpreting Results

- If **`ablation_learned_casted`** degrades significantly vs baseline → **RoPE is key**
- If **`ablation_rope_standard`** degrades significantly vs baseline → **CastedLinear is key**
- If **`ablation_learned_standard`** degrades more than either single ablation → **both contribute**
- If neither single ablation degrades much → the mechanism is elsewhere (e.g., SwiGLU, RMS norm, ACT recursion)

## Getting Started

### Installation

```bash
pip install -r requirements.txt
```

### Train the Demo Transformer (Sudoku)

```bash
python scripts/train_sudoku.py
```

### Train the TRM (Sudoku)

```bash
# First build the on-disk dataset
python trm_base/build_sdku_data.py

# Then train
python trm_base/pretrain.py
```

## License

This project is open source and available under the MIT License.
