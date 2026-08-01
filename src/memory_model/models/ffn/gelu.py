import torch
import torch.nn as nn
import torch.nn.functional as F

from ...config import ModelConfig


class FeedForward(nn.Module):
    """Position-wise feed-forward network using GELU.

    References:
        Vaswani et al., "Attention Is All You Need" (position-wise FFN, 2017).
        https://arxiv.org/abs/1706.03762
        Hendrycks and Gimpel, "Gaussian Error Linear Units (GELUs)" (2016).
        https://arxiv.org/abs/1606.08415
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.up_proj = nn.Linear(config.d_model, config.d_ff, bias=config.bias)
        self.down_proj = nn.Linear(config.d_ff, config.d_model, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.down_proj(F.gelu(self.up_proj(x), approximate="tanh")))
