from __future__ import annotations

import torch
import torch.nn as nn


class MemoryProjections(nn.Module):
    """Project Backbone hidden states into memory query, key and value spaces."""

    def __init__(self, hidden_size: int, key_dim: int, value_dim: int) -> None:
        super().__init__()
        self.query = nn.Linear(hidden_size, key_dim, bias=False)
        self.key = nn.Linear(hidden_size, key_dim, bias=False)
        self.value = nn.Linear(hidden_size, value_dim, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.query(hidden_states), self.key(hidden_states), self.value(hidden_states)

