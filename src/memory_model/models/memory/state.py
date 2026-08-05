from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class MemoryState:
    """Per-batch dynamic state; this is not a trainable Backbone parameter."""

    matrix: torch.Tensor
    normalizer: torch.Tensor

    @classmethod
    def zeros(
        cls,
        batch_size: int,
        key_dim: int,
        value_dim: int,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> "MemoryState":
        return cls(
            matrix=torch.zeros(batch_size, key_dim, value_dim, device=device, dtype=dtype),
            normalizer=torch.zeros(batch_size, key_dim, device=device, dtype=dtype),
        )

    def validate(self, batch_size: int, key_dim: int, value_dim: int) -> None:
        if self.matrix.shape != (batch_size, key_dim, value_dim):
            raise ValueError(
                "memory matrix must have shape "
                f"({batch_size}, {key_dim}, {value_dim}), got {tuple(self.matrix.shape)}"
            )
        if self.normalizer.shape != (batch_size, key_dim):
            raise ValueError(
                "memory normalizer must have shape "
                f"({batch_size}, {key_dim}), got {tuple(self.normalizer.shape)}"
            )

    def detach(self) -> "MemoryState":
        """Stop gradients across interactions while preserving the state values."""

        return MemoryState(self.matrix.detach(), self.normalizer.detach())

