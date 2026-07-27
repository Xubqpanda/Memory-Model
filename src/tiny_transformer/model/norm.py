import torch.nn as nn

from ..config import ModelConfig


class TransformerLayerNorm(nn.LayerNorm):
    """Current normalization boundary, isolated for future RMSNorm experiments."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config.d_model, bias=config.bias)
