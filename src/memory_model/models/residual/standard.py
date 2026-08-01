import torch
import torch.nn as nn


class ResidualConnection(nn.Module):
    """Add a sublayer update to the residual stream.

    Reference:
        He et al., "Deep Residual Learning for Image Recognition" (2015).
        https://arxiv.org/abs/1512.03385
    """

    def forward(self, residual: torch.Tensor, update: torch.Tensor) -> torch.Tensor:
        return residual + update
