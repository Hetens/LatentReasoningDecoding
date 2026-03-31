import torch as t
import math
import os
import numpy as np
import sys
from collections import defaultdict
from tqdm import tqdm
from dataclasses import dataclass
import datasets
import einops       
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import Tensor
from jaxtyping import Float, Int
from torch.nn import functional as F
import transformer_lens
from transformer_lens.utils import gelu_new, tokenize_and_concatenate
from transformer_lens import HookedTransformer
from transformers.models.gpt2.tokenization_gpt2_fast import GPT2TokenizerFast

#initialize cuda and see if it works
device = 'cuda' if t.cuda.is_available() else 'cpu'
print(device)  # Should be True
print(t.cuda.get_device_capability())  # Should show (6, 1)

#lets define a config for the model
@dataclass
class Config:
    d_model: int = 768
    debug: bool = True
    layer_norm_eps: float = 1e-5
    d_vocab: int = 50257
    init_range: float = 0.02 #for normal initialization the range
    n_ctx: int = 1024
    d_head: int = 64
    d_mlp: int = 3072
    n_layers: int = 12
    n_heads: int = 12

cfg = Config()
#Layer Norm
class LayerNorm(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.w = nn.Parameter(t.ones(cfg.d_model))
        self.b = nn.Parameter(t.zeros(cfg.d_model))
    def forward(self, residual: Float [Tensor,"batch posn d_model"]) -> Float [Tensor,"batch posn d_model"]:
        residual_mean = residual.mean(dim=-1, keepdim=True)
        residual_var = residual.var(dim=-1, keepdim=True, unbiased=False)
        residual = (residual - residual_mean) / (residual_var + self.cfg.layer_norm_eps).sqrt()
        return residual * self.w + self.b

class Embed(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.W_E = nn.Parameter(t.empty(cfg.d_vocab, cfg.d_model))
        nn.init.normal_(self.W_E, std=cfg.init_range)
    def forward(self, tokens: Int[Tensor, "batch posn"])-> Float[Tensor, "batch posn d_model"]:
        return self.W_E[tokens]
    
class PosEmbed(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.W_pos = nn.Parameter(t.empty(cfg.d_model, cfg.n_ctx))
        nn.init.normal_(self.W_pos, std=cfg.init_range)
    def forward(self, tokens: Int[Tensor, "batch posn"])-> Float[Tensor, "batch posn d_model"]:
        batch, seq_len = tokens.shape
        return einops.repeat(self.W_pos[:seq_len],"seq_len d_model -> batch seq_len d_model", batch = batch)

# Attention 
class Attention(nn.Module):
    IGNORE: Float[Tensor, ""]

    def __init__(self, cfg:Config):
        super().__init__()
        self.cfg = cfg
        self.W_Q = nn.Parameter(t.empty((cfg.n_heads, cfg.d_model, cfg.d_head)))
        self.W_K = nn.Parameter(t.empty((cfg.n_heads, cfg.d_model, cfg.d_head)))
        self.W_V = nn.Parameter(t.empty((cfg.n_heads, cfg.d_model, cfg.d_head)))
        self.W_O = nn.Parameter(t.empty((cfg.n_heads, cfg.d_head, cfg.d_model)))
        self.b_Q = nn.Parameter(t.zeros((cfg.n_heads, cfg.d_head)))
        self.b_K = nn.Parameter(t.zeros((cfg.n_heads, cfg.d_head)))
        self.b_V = nn.Parameter(t.zeros((cfg.n_heads, cfg.d_head)))
        self.b_O = nn.Parameter(t.zeros((cfg.d_model)))
        nn.init.normal_(self.W_Q, std=self.cfg.init_range)
        nn.init.normal_(self.W_K, std=self.cfg.init_range)
        nn.init.normal_(self.W_V, std=self.cfg.init_range)
        nn.init.normal_(self.W_O, std=self.cfg.init_range)
        self.register_buffer("IGNORE", t.tensor(float("-inf"), dtype=t.float32, device=device))

    def forward(self, normalized_resid_pre: Float[Tensor, "batch posn d_model"])-> Float[Tensor, "batch posn d_model"]:
        q = (
            einops.einsum(
                normalized_resid_pre,
                self.W_Q,
                "batch posn d_model, nheads d_model d_head -> batch posn nheads d_head",
            )
            + self.b_Q
        )
        k = (
            einops.einsum(
                normalized_resid_pre,
                self.W_K,
                "batch posn d_model, nheads d_model d_head -> batch posn nheads d_head",
            )
            + self.b_K
        )
        v = (
            einops.einsum(
                normalized_resid_pre,
                self.W_V,
                "batch posn d_model, nheads d_model d_head -> batch posn nheads d_head",
            )
            + self.b_V
        )

        # Calculate attention scores, then scale and mask, and apply softmax to get probabilities
        attn_scores = einops.einsum(
            q,
            k,
            "batch posn_Q nheads d_head, batch posn_K nheads d_head -> batch nheads posn_Q posn_K",
        )
        attn_scores_masked = self.apply_causal_mask(attn_scores / self.cfg.d_head**0.5)
        attn_pattern = attn_scores_masked.softmax(-1)

        # Take weighted sum of value vectors, according to attention probabilities
        z = einops.einsum(
            v,
            attn_pattern,
            "batch posn_K nheads d_head, batch nheads posn_Q posn_K -> batch posn_Q nheads d_head",
        )

        # Calculate output (by applying matrix W_O and summing over heads, then adding bias b_O)
        attn_out = (
            einops.einsum(
                z,
                self.W_O,
                "batch posn_Q nheads d_head, nheads d_head d_model -> batch posn_Q d_model",
            )
            + self.b_O
        )

        return attn_out
        
    def apply_causal_mask(self, attn_scores: Float[Tensor, "batch n_heads query_pos key_pos"])-> Float[Tensor, "batch n_heads query_pos key_pos"]:
        all_ones = t.ones(attn_scores.size(-2), attn_scores.size(-1), device=attn_scores.device)
        mask = t.triu(all_ones, diagonal=1).bool()
        attn_scores.masked_fill_(mask, self.IGNORE)
        return attn_scores

class MLP(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.W_in = nn.Parameter(t.empty((cfg.d_model, cfg.d_mlp)))
        self.W_out = nn.Parameter(t.empty((cfg.d_mlp, cfg.d_model)))
        self.b_in = nn.Parameter(t.zeros((cfg.d_mlp)))
        self.b_out = nn.Parameter(t.zeros((cfg.d_model)))
        nn.init.normal_(self.W_in, std=self.cfg.init_range)
        nn.init.normal_(self.W_out, std=self.cfg.init_range)

    def forward(
        self, normalized_resid_mid: Float[Tensor, "batch posn d_model"]
    ) -> Float[Tensor, "batch posn d_model"]:
        pre = (
            einops.einsum(
                normalized_resid_mid,
                self.W_in,
                "batch position d_model, d_model d_mlp -> batch position d_mlp",
            )
            + self.b_in
        )
        post = gelu_new(pre)
        mlp_out = (
            einops.einsum(
                post, self.W_out, "batch position d_mlp, d_mlp d_model -> batch position d_model"
            )
            + self.b_out
        )
        return mlp_out

class TransformerBlock(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.attn = Attention(cfg)
        self.mlp = MLP(cfg)
        self.ln1 = LayerNorm(cfg)
        self.ln2 = LayerNorm(cfg)
    def forward(self, resid_pre: Float[Tensor, "batch posn d_model"])-> Float[Tensor, "batch posn d_model"]:
        resid_mid = self.attn(self.ln1(resid_pre)) + resid_pre
        resid_post = self.mlp(self.ln2(resid_mid)) + resid_mid
        return resid_post


class Unembed(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.W_U = nn.Parameter(t.empty((cfg.d_model, cfg.d_vocab)))
        nn.init.normal_(self.W_U, std=self.cfg.init_range)
        self.b_U = nn.Parameter(t.zeros((cfg.d_vocab), requires_grad=False))
    def forward(self, normalized_resid_final: Float[Tensor, "batch posn d_model"])-> Float[Tensor, "batch posn d_vocab"]:
        return (
            einops.einsum(
                normalized_resid_final,
                self.W_U,
                "batch posn d_model, d_model d_vocab -> batch posn d_vocab",
            )
            + self.b_U
        )

### Making the Model
class DemoTransformer(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.embed = Embed(cfg)
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.unembed = Unembed(cfg)
    def forward(self, tokens: Int[Tensor, "batch posn"])-> Float[Tensor, "batch posn d_vocab"]:
        resid_pre = self.embed(tokens)
        for block in self.blocks:
            resid_pre = block(resid_pre)
        return self.unembed(resid_pre) #logits


reference_gpt2 = HookedTransformer.from_pretrained(
    "gpt2-small",
    fold_ln=False,
    center_unembed=False,
    center_writing_weights=False,  # you'll learn about these arguments later!
)

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
model = DemoTransformer(model_cfg)


@dataclass
class TransformerTrainingArgs:
    batch_size: int = 16
    epochs: int = 10
    max_steps_per_epoch: int = 500
    lr: float = 1e-3
    weight_decay: float = 1e-2
    wandb_project: str | None = "day1-demotransformer"
    wandb_name: str | None = None


args = TransformerTrainingArgs()

dataset = datasets.load_dataset("roneneldan/TinyStories", split="train")
# print(dataset)
# print(dataset[0]["text"])
tokenized_dataset = tokenize_and_concatenate(
    dataset,
    reference_gpt2.tokenizer,
    streaming=False,
    max_length=model.cfg.n_ctx,
    column_name="text",
    add_bos_token=True,
    num_proc=4,
)

dataset_dict = tokenized_dataset.train_test_split(test_size=1000)
train_loader = DataLoader(
    dataset_dict["train"], batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True
)
test_loader = DataLoader(
    dataset_dict["test"], batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True
)




class TransformerSampler:
    def __init__(self, model: DemoTransformer, tokenizer: GPT2TokenizerFast):
        self.model = model
        self.cfg = model.cfg
        self.tokenizer = tokenizer

    @t.inference_mode()
    def sample(self, prompt: str, max_tokens_generated=100, verbose=False, **kwargs) -> str:
        """
        Returns a string of autoregressively generated text, starting from the prompt.

        Sampling terminates at max_tokens_generated, or when the model generates an end-of-sequence token. kwargs are
        passed to sample_next_token, to give detailed instructions on how new tokens are chosen.
        """
        self.model.eval()
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(device)[0]

        for i in range(max_tokens_generated):
            # Get new logits (make sure we don't pass in more tokens than the model's context length)
            logits = self.model(input_ids[None, -self.cfg.n_ctx :])
            # We only take logits for the last token, because this is what we're sampling
            logits = logits[0, -1]
            # Get next token (as a tensor of size (1, 1) so we can concat it to input_ids)
            next_token = t.tensor([TransformerSampler.sample_next_token(input_ids, logits, **kwargs)], device=device)
            # Create new input ids string, with shape (1, old_seq_len + 1)
            input_ids = t.cat([input_ids, next_token], dim=-1)
            # Print out results, if required
            if verbose:
                print(self.tokenizer.decode(input_ids), end="\r")
            # If our new token was the end-of-text token, stop
            if next_token == getattr(self.tokenizer, "eos_token_id", None):
                break

        return self.tokenizer.decode(input_ids)

    @staticmethod
    def sample_next_token(
        input_ids: Int[Tensor, "seq_len"],
        logits: Float[Tensor, "d_vocab"],
        temperature=1.0,
        top_k=0,
        top_p=0.0,
        frequency_penalty=0.0,
        seed=None,
    ) -> int:
        assert input_ids.ndim == 1, "input_ids should be a 1D sequence of token ids"
        assert temperature >= 0, "Temperature should be non-negative"
        assert 0 <= top_p <= 1.0, "Top-p must be a probability"
        assert 0 <= top_k, "Top-k must be non-negative"
        assert not (top_p != 0 and top_k != 0), "At most one of top-p and top-k supported"

        # Set random seeds for reproducibility
        if seed is not None:
            t.manual_seed(seed)
            np.random.seed(seed)

        # Apply all the specialized sampling methods
        if temperature == 0:
            return TransformerSampler.greedy_search(logits)
        elif temperature != 1.0:
            logits = TransformerSampler.apply_temperature(logits, temperature)
        if frequency_penalty != 0.0:
            logits = TransformerSampler.apply_frequency_penalty(
                input_ids, logits, frequency_penalty
            )
        if top_k > 0:
            return TransformerSampler.sample_top_k(logits, top_k)
        if top_p > 0.0:
            return TransformerSampler.sample_top_p(logits, top_p)
        return TransformerSampler.sample_basic(logits)

    @staticmethod
    def greedy_search(logits: Float[Tensor, "d_vocab"]) -> int:
        """
        Returns the most likely token (as an int).
        """
        return t.argmax(logits).item()

    @staticmethod
    def apply_temperature(
        logits: Float[Tensor, "d_vocab"], temperature: float
    ) -> Float[Tensor, "d_vocab"]:
        """
        Applies temperature scaling to the logits.
        """
        return logits / temperature

    @staticmethod
    def apply_frequency_penalty(
        input_ids: Int[Tensor, "seq_len"], logits: Float[Tensor, "d_vocab"], freq_penalty: float
    ) -> Float[Tensor, "d_vocab"]:
        """
        Applies a frequency penalty to the logits.
        """
        d_vocab = logits.size(0)
        id_freqs = t.bincount(input_ids, minlength=d_vocab)
        return logits - freq_penalty * id_freqs

    @staticmethod
    def sample_basic(logits: Float[Tensor, "d_vocab"]) -> int:
        """
        Samples from the distribution defined by the logits.
        """
        sampled_token = t.distributions.categorical.Categorical(logits=logits).sample()
        return sampled_token.item()

    @staticmethod
    def sample_top_k(logits: Float[Tensor, "d_vocab"], k: int) -> int:
        """
        Samples from the top k most likely tokens.
        """
        top_k_logits, top_k_ids = t.topk(logits, k)
        sampled_token = t.distributions.categorical.Categorical(logits=top_k_logits).sample()
        return top_k_ids[sampled_token].item()

    @staticmethod
    def sample_top_p(
        logits: Float[Tensor, "d_vocab"], top_p: float, min_tokens_to_keep: int = 1
    ) -> int:
        """
        Samples from the most likely tokens which make up at least p cumulative probability.
        """
       # Sort logits, and get cumulative probabilities
        logits_sorted, indices = logits.sort(descending=True, stable=True)
        cumul_probs = logits_sorted.softmax(-1).cumsum(-1)
        # Choose which tokens to keep, in the set we sample from
        n_keep = t.searchsorted(cumul_probs, top_p, side="left").item() + 1
        n_keep = max(n_keep, min_tokens_to_keep)
        keep_idx = indices[:n_keep]
        keep_logits = logits[keep_idx]
        # Perform the sampling
        sample = t.distributions.categorical.Categorical(logits=keep_logits).sample()
        return keep_idx[sample].item()


class TransformerTrainer:
    def __init__(self, args: TransformerTrainingArgs, model: DemoTransformer):
        super().__init__()
        self.model = model
        self.args = args
        self.sampler = TransformerSampler(self.model, reference_gpt2.tokenizer)
        self.optimizer = t.optim.AdamW(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        self.step = 0

        self.train_loader = train_loader
        self.test_loader = test_loader

    def get_log_probs(
        self, logits: Float[Tensor, "batch posn d_vocab"], tokens: Int[Tensor, "batch posn"]
    ) -> Float[Tensor, "batch posn-1"]:
        log_probs = logits.log_softmax(dim=-1)
        # Get logprobs the first seq_len-1 predictions (so we can compare them with the actual next tokens)
        log_probs_for_tokens = (
            log_probs[:, :-1].gather(dim=-1, index=tokens[:, 1:].unsqueeze(-1)).squeeze(-1)
        )

        return log_probs_for_tokens

    def training_step(self, batch: dict[str, Int[Tensor,"batch seq"]]) -> Float[Tensor, ""]:
        tokens = batch["tokens"].to(device)
        logits = self.model(tokens)
        loss = -self.get_log_probs(logits, tokens).mean()
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.step += 1
        return loss
        
    @t.inference_mode()
    def evaluate(self) -> float:
        self.model.eval()
        totally_correct, total_samples = 0, 0
        for batch in tqdm(self.test_loader, desc="Evaluating"):
            tokens = batch["tokens"].to(device)
            logits: Tensor = self.model(tokens)[:, :-1]
            predicted_tokens = logits.argmax(dim=-1)
            totally_correct += (predicted_tokens == tokens[:,1:]).sum().item()
            total_samples += tokens.size(0) * (tokens.size(1) - 1)
        accuracy = totally_correct / total_samples
        self.model.train()
        return accuracy
    
    def train(self):
        accuracy = np.nan
        progress_bar = tqdm(total = self.args.max_steps_per_epoch * self.args.epochs, desc="Training")

        for epoch in range(self.args.epochs):
            for i, batch in enumerate(self.train_loader):
                loss = self.training_step(batch)
                progress_bar.update()
                progress_bar.set_description(f"Epoch {epoch}, loss: {loss:.3f},accuracy:{accuracy:.3f}")
                if i>= self.args.max_steps_per_epoch:
                    break
            accuracy = self.evaluate()
            sample_text = self.sampler.sample("Once upon a time", max_tokens_generated = 50)
            print(sample_text)
    
    def save_model(self, filepath: str):
        """
        Save model checkpoint including model state, optimizer state, and training progress.
        """
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'step': self.step,
            'model_config': {
                'd_model': self.model.cfg.d_model,
                'n_heads': self.model.cfg.n_heads,
                'd_head': self.model.cfg.d_head,
                'd_mlp': self.model.cfg.d_mlp,
                'n_layers': self.model.cfg.n_layers,
                'n_ctx': self.model.cfg.n_ctx,
                'd_vocab': self.model.cfg.d_vocab,
            }
        }
        t.save(checkpoint, filepath)
        print(f"Model saved to {filepath}")

model = DemoTransformer(model_cfg).to(device)
args = TransformerTrainingArgs()
trainer = TransformerTrainer(args, model)
trainer.train()
trainer.save_model("model_checkpoint.pt")
                