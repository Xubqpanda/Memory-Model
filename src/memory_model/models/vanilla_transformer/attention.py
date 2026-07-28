from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...config import ModelConfig
from .types import KVCache


class CausalSelfAttention(nn.Module):
    """Vectorized multi-head causal self-attention with an inference KV cache."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.n_head = config.n_head
        self.head_dim = config.d_model // config.n_head
        self.dropout = config.dropout
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model, bias=config.bias)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)
        self.resid_dropout = nn.Dropout(config.dropout)

    def _split_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = tensor.shape
        return tensor.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)

    def forward(
        self,
        x: torch.Tensor,
        past_key_value: KVCache | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, KVCache | None]:
        batch_size, seq_len, width = x.shape
        q, k, v = (self._split_heads(tensor) for tensor in self.qkv(x).chunk(3, dim=-1))

        past_len = 0
        if past_key_value is not None:
            past_k, past_v = past_key_value
            past_len = past_k.size(2)
            k = torch.cat((past_k, k), dim=2)
            v = torch.cat((past_v, v), dim=2)

        # A query at absolute position p may only see keys at positions <= p.
        # Without a cache, is_causal=True enables PyTorch's fastest available kernel.
        if past_len == 0:
            y = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            )
        else:
            query_positions = past_len + torch.arange(seq_len, device=x.device)
            key_positions = torch.arange(k.size(2), device=x.device)
            allowed = key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)
            y = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=allowed,
                dropout_p=self.dropout if self.training else 0.0,
            )

        y = y.transpose(1, 2).contiguous().view(batch_size, seq_len, width)
        y = self.resid_dropout(self.out_proj(y))
        return y, (k, v) if use_cache else None
