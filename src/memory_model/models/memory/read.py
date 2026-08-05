from __future__ import annotations

import torch
import torch.nn as nn

from .state import MemoryState


class MemoryReader(nn.Module):
    """Read values from a key-value fast-weight matrix."""

    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, query: torch.Tensor, state: MemoryState) -> torch.Tensor:
        if query.ndim != 3:
            raise ValueError("query must have shape [batch, sequence, key_dim]")
        batch_size, _, key_dim = query.shape
        value_dim = state.matrix.size(-1)
        state.validate(batch_size, key_dim, value_dim)
        numerator = torch.einsum("bkv,btk->btv", state.matrix, query)
        denominator = torch.einsum(
            "bk,btk->bt", state.normalizer, query.abs()
        ).clamp_min(self.eps)
        return numerator / denominator.unsqueeze(-1)

