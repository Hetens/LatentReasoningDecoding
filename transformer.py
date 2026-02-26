"""Transformer block and main DemoTransformer model."""
import torch.nn as nn
from torch import Tensor
from jaxtyping import Float, Int
from config import Config
from layers import LayerNorm, Embed, PosEmbed, Unembed
from attention import Attention
from mlp import MLP


class TransformerBlock(nn.Module):
    """Single transformer block with attention and MLP."""
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.attn = Attention(cfg)
        self.mlp = MLP(cfg)
        self.ln1 = LayerNorm(cfg)
        self.ln2 = LayerNorm(cfg)

    def forward(
        self,
        resid_pre: Float[Tensor, "batch posn d_model"],
        kv_cache: dict | None = None,
        cache_position: int | None = None,
    ) -> Float[Tensor, "batch posn d_model"]:
        resid_mid = self.attn(self.ln1(resid_pre), kv_cache=kv_cache, cache_position=cache_position) + resid_pre
        resid_post = self.mlp(self.ln2(resid_mid)) + resid_mid
        return resid_post


class DemoTransformer(nn.Module):
    """Main transformer model."""
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.embed = Embed(cfg)
        self.pos_embed = PosEmbed(cfg)
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.unembed = Unembed(cfg)

    def forward(
        self,
        tokens: Int[Tensor, "batch posn"],
        kv_caches: list[dict] | None = None,
        cache_position: int | None = None,
    ) -> Float[Tensor, "batch posn d_vocab"]:
        """
        Forward pass with optional KV caches.

        Args:
            tokens: Input token IDs.
            kv_caches: List of per-layer KV cache dicts (one per block).
                       If None, runs without caching (training mode).
            cache_position: Total sequence length including current tokens.
        """
        # Original was simply: resid_pre = self.embed(tokens) + self.pos_embed(tokens)
        # Now we double check if there is an offset and calculate the position embedding from there if needed save time
        pos_offset = (cache_position - tokens.size(1)) if cache_position is not None else 0
        resid_pre = self.embed(tokens) + self.pos_embed(tokens, offset=pos_offset)
        for i, block in enumerate(self.blocks):
            layer_cache = kv_caches[i] if kv_caches is not None else None
            resid_pre = block(resid_pre, kv_cache=layer_cache, cache_position=cache_position)
        return self.unembed(resid_pre)  # logits

    def create_kv_caches(self) -> list[dict]:
        """Create empty KV caches for each layer."""
        return [{"k": None, "v": None} for _ in range(self.cfg.n_layers)]
