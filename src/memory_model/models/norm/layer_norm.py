import torch.nn as nn

from ...config import ModelConfig


class TransformerLayerNorm(nn.LayerNorm):
    """LayerNorm isolated behind a component boundary for future alternatives.

    Reference:
        Ba, Kiros, and Hinton, "Layer Normalization" (2016).
        https://arxiv.org/abs/1607.06450
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config.d_model, bias=config.bias)
