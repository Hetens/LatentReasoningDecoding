"""
Probe architectures for the candidate-set classification task.

Linear probe  (H1): Nine independent binary classifiers sharing the same input.
    ℓ_k(h) = w_k^T h + b_k,  k = 1, …, 9.

MLP probe     (H2): Two-layer MLP with GELU activations and a 9-dim sigmoid
    output.  SwiGLU is deliberately avoided (it matches TRM internals).

Both probes are trained with the same multi-label binary cross-entropy loss
(Eq. 2 in the report).
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class LinearProbe(nn.Module):
    """Nine binary classifiers: ``W h + b`` → (batch, 9) logits."""

    def __init__(self, d_input: int) -> None:
        super().__init__()
        self.linear = nn.Linear(d_input, 9)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.linear(h)


class MLPProbe(nn.Module):
    """Two-layer MLP probe: h → Linear → GELU → Linear → (batch, 9) logits.

    Hidden width defaults to 128; GELU chosen to differ from SwiGLU used
    inside TRM itself.
    """

    def __init__(self, d_input: int, d_hidden: int = 128, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_input, d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, 9),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h)


def probe_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """Multi-label binary cross-entropy (Eq. 2 in the report).

    Args:
        logits: (batch, 9) raw logits.
        targets: (batch, 9) binary labels.
    """
    return F.binary_cross_entropy_with_logits(logits, targets, reduction="mean")
