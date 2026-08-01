from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...config import ModelConfig
from ..embedding import build_rotary_position_embedding
from ..types import KVCache


class MultiHeadCausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with an inference KV cache.

    Reference:
        Vaswani et al., "Attention Is All You Need" (2017).
        https://arxiv.org/abs/1706.03762
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.n_head = config.n_head
        self.head_dim = config.d_model // config.n_head
        self.dropout = config.dropout
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model, bias=config.bias)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.rotary_embedding = build_rotary_position_embedding(config)

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

        past_len = 0 if past_key_value is None else past_key_value[0].size(2)
        if self.rotary_embedding is not None:
            position_ids = torch.arange(past_len, past_len + seq_len, device=x.device)
            q, k = self.rotary_embedding(q, k, position_ids)

        if past_key_value is not None:
            past_k, past_v = past_key_value
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


# Preserve the original public class name while making the concrete MHA
# implementation explicit for future GQA, MLA, and KDA variants.
CausalSelfAttention = MultiHeadCausalSelfAttention
