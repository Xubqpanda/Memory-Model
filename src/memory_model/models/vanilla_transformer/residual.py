import torch
import torch.nn as nn


class ResidualConnection(nn.Module):
    """Combine the residual stream with a sublayer update.

    Keeping this operation behind a module makes gated, scaled, or memory-aware
    residual connections independently replaceable later.
    """

    def forward(self, residual: torch.Tensor, update: torch.Tensor) -> torch.Tensor:
        return residual + update
