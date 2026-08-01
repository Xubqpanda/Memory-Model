from ...config import ModelConfig
from .mha import CausalSelfAttention, MultiHeadCausalSelfAttention


def build_attention(config: ModelConfig) -> MultiHeadCausalSelfAttention:
    if config.attention_type == "mha":
        return MultiHeadCausalSelfAttention(config)
    raise ValueError(f"unsupported attention_type: {config.attention_type}")


__all__ = ["CausalSelfAttention", "MultiHeadCausalSelfAttention", "build_attention"]
