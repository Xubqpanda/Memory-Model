import torch
import torch.nn as nn
import torch.nn.functional as F

from ...config import ModelConfig


class SwiGLUFeedForward(nn.Module):
    """Position-wise SwiGLU feed-forward network.

    The two input projections produce a smooth, input-dependent gate and a
    candidate value. Their elementwise product is projected back to d_model.

    Reference:
        Shazeer, "GLU Variants Improve Transformer" (2020).
        https://arxiv.org/abs/2002.05202
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.d_model, config.d_ff, bias=config.bias)
        self.up_proj = nn.Linear(config.d_model, config.d_ff, bias=config.bias)
        self.down_proj = nn.Linear(config.d_ff, config.d_model, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gated_value = F.silu(self.gate_proj(x)) * self.up_proj(x)
        return self.dropout(self.down_proj(gated_value))
