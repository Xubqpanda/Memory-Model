from __future__ import annotations

import torch
import torch.nn as nn

from ...config import ModelConfig
from ..attention import build_attention
from ..ffn import FeedForward
from ..norm import TransformerLayerNorm
from ..residual import ResidualConnection
from ..types import KVCache


class TransformerBlock(nn.Module):
    """Pre-Norm block: x + Attention(LN(x)), then x + FFN(LN(x)).

    References:
        Vaswani et al., "Attention Is All You Need" (Transformer, 2017).
        https://arxiv.org/abs/1706.03762
        Xiong et al., "On Layer Normalization in the Transformer Architecture"
        (Pre-LN analysis, 2020). https://arxiv.org/abs/2002.04745
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.attn_norm = TransformerLayerNorm(config)
        self.attn = build_attention(config)
        self.attn_residual = ResidualConnection()
        self.ffn_norm = TransformerLayerNorm(config)
        self.ffn = FeedForward(config)
        self.ffn_residual = ResidualConnection()

    def forward(
        self,
        x: torch.Tensor,
        past_key_value: KVCache | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, KVCache | None]:
        attn_update, new_cache = self.attn(self.attn_norm(x), past_key_value, use_cache)
        x = self.attn_residual(x, attn_update)
        ffn_update = self.ffn(self.ffn_norm(x))
        x = self.ffn_residual(x, ffn_update)
        return x, new_cache
