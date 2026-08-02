from ...config import ModelConfig
from .gelu import FeedForward, GELUFeedForward
from .swiglu import SwiGLUFeedForward


def build_ffn(config: ModelConfig) -> GELUFeedForward | SwiGLUFeedForward:
    if config.ffn_type == "gelu":
        return GELUFeedForward(config)
    if config.ffn_type == "swiglu":
        return SwiGLUFeedForward(config)
    raise ValueError(f"unsupported ffn_type: {config.ffn_type}")


__all__ = ["FeedForward", "GELUFeedForward", "SwiGLUFeedForward", "build_ffn"]
