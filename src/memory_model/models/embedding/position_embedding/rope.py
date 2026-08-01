from __future__ import annotations

import torch
import torch.nn as nn

from ....config import ModelConfig


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate pairs represented by the two halves of the head dimension."""

    first_half, second_half = x.chunk(2, dim=-1)
    return torch.cat((-second_half, first_half), dim=-1)


class RotaryPositionEmbedding(nn.Module):
    """Apply RoPE rotations to query and key vectors.

    Unlike learned absolute embeddings, RoPE is not added to the residual
    stream. It rotates Q and K inside every attention layer so their dot product
    contains relative-position information.

    Reference:
        Su et al., "RoFormer: Enhanced Transformer with Rotary Position
        Embedding" (2021). https://arxiv.org/abs/2104.09864
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        head_dim = config.d_model // config.n_head
        if head_dim % 2:
            raise ValueError("RoPE requires an even attention head dimension")
        inv_freq = 1.0 / (
            config.rope_theta
            ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if position_ids.ndim != 1:
            raise ValueError("position_ids must be a one-dimensional tensor")
        if query.size(-2) != position_ids.numel() or key.size(-2) != position_ids.numel():
            raise ValueError("position_ids length must match the current query/key sequence")

        frequencies = torch.outer(position_ids.float(), self.inv_freq)
        angles = torch.cat((frequencies, frequencies), dim=-1)
        cosine = angles.cos()[None, None, :, :].to(dtype=query.dtype)
        sine = angles.sin()[None, None, :, :].to(dtype=query.dtype)
        return (
            query * cosine + rotate_half(query) * sine,
            key * cosine + rotate_half(key) * sine,
        )
