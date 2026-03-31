"""Main entry point for training and running inference with the DemoTransformer."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import torch as t
import datasets
from torch.utils.data import DataLoader
from transformer_lens import HookedTransformer
from transformer_lens.utils import tokenize_and_concatenate
from core.config import Config, TransformerTrainingArgs, device
from core.transformer import DemoTransformer
from core.trainer import TransformerTrainer
from core.sampler import TransformerSampler


def main():
    # Print device info
    print(f"Using device: {device}")
    if device == "cuda":
        print(f"CUDA capability: {t.cuda.get_device_capability()}")

    # Load reference GPT-2 for tokenizer and vocab size
    print("Loading reference GPT-2 model for tokenizer...")
    reference_gpt2 = HookedTransformer.from_pretrained(
        "gpt2-small",
        fold_ln=False,
        center_unembed=False,
        center_writing_weights=False,
    )

    # Create model configuration
    model_cfg = Config(
        debug=False,
        d_model=32,
        n_heads=16,
        d_head=2,
        d_mlp=32 * 4,
        n_layers=4,
        n_ctx=128,
        d_vocab=reference_gpt2.cfg.d_vocab,
    )

    # Create model
    model = DemoTransformer(model_cfg).to(device)
    print(f"Created DemoTransformer with {sum(p.numel() for p in model.parameters())} parameters")

    # Training arguments
    args = TransformerTrainingArgs()

    # Load and tokenize dataset
    print("Loading TinyStories dataset...")
    dataset = datasets.load_dataset("roneneldan/TinyStories", split="train")
    tokenized_dataset = tokenize_and_concatenate(
        dataset,
        reference_gpt2.tokenizer,
        streaming=False,
        max_length=model.cfg.n_ctx,
        column_name="text",
        add_bos_token=True,
        num_proc=4,
    )

    # Create data loaders
    dataset_dict = tokenized_dataset.train_test_split(test_size=1000)
    train_loader = DataLoader(
        dataset_dict["train"],
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    test_loader = DataLoader(
        dataset_dict["test"],
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # Create trainer and train
    print("Starting training...")
    trainer = TransformerTrainer(
        args=args,
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        tokenizer=reference_gpt2.tokenizer,
    )
    trainer.train()

    # Save checkpoint
    checkpoint_path = "./saved_models/model_checkpoint.pt"
    trainer.save_model(checkpoint_path)

    # # Run inference loop
    # run_inference_loop(model, reference_gpt2.tokenizer)


def run_inference_loop(model: DemoTransformer, tokenizer):
    """Run an interactive inference loop."""
    print("\n" + "=" * 50)
    print("INFERENCE MODE")
    print("Enter a prompt to generate text, or 'quit' to exit.")
    print("=" * 50 + "\n")

    sampler = TransformerSampler(model, tokenizer)

    while True:
        try:
            prompt = input("Prompt: ").strip()
            if prompt.lower() in ("quit", "exit", "q"):
                print("Goodbye!")
                break
            if not prompt:
                print("Please enter a prompt.")
                continue

            print("\nGenerating...")
            generated = sampler.sample(
                prompt,
                max_tokens_generated=100,
                verbose=False,
                temperature=0.8,
                top_p=0.9,
            )
            print(f"\n{generated}\n")

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except EOFError:
            print("\nGoodbye!")
            break


def load_and_run_inference(checkpoint_path: str = "model_checkpoint.pt"):
    """Load a checkpoint and run inference."""
    # Load reference GPT-2 for tokenizer
    reference_gpt2 = HookedTransformer.from_pretrained(
        "gpt2-small",
        fold_ln=False,
        center_unembed=False,
        center_writing_weights=False,
    )

    # Load checkpoint to get config
    checkpoint = t.load(checkpoint_path, map_location=device)
    model_config = checkpoint["model_config"]

    # Create model with saved config
    cfg = Config(
        d_model=model_config["d_model"],
        n_heads=model_config["n_heads"],
        d_head=model_config["d_head"],
        d_mlp=model_config["d_mlp"],
        n_layers=model_config["n_layers"],
        n_ctx=model_config["n_ctx"],
        d_vocab=model_config["d_vocab"],
    )
    model = DemoTransformer(cfg).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"Loaded model from {checkpoint_path}")
    run_inference_loop(model, reference_gpt2.tokenizer)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--inference":
        # Run inference only from a saved checkpoint
        checkpoint = sys.argv[2] if len(sys.argv) > 2 else "./saved_models/model_checkpoint.pt"
        load_and_run_inference(checkpoint)
    else:
        # Full training
        main()
