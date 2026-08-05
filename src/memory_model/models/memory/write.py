from __future__ import annotations

import torch
import torch.nn as nn

from .selector import SelectionOutput
from .state import MemoryState


class GatedDeltaWriter(nn.Module):
    """Write selected key-value pairs with a gated delta rule."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.gate = nn.Linear(hidden_size, 1, bias=True)

    def forward(
        self,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        selection: SelectionOutput,
        state: MemoryState,
    ) -> tuple[MemoryState, torch.Tensor]:
        if hidden_states.ndim != 3 or keys.ndim != 3 or values.ndim != 3:
            raise ValueError("hidden_states, keys and values must be rank-3 tensors")
        batch_size, sequence_length, _ = hidden_states.shape
        key_dim = keys.size(-1)
        value_dim = values.size(-1)
        state.validate(batch_size, key_dim, value_dim)
        if selection.weights.shape != (batch_size, sequence_length):
            raise ValueError("selection weights must match hidden_states sequence shape")

        predicted = torch.einsum("bkv,btk->btv", state.matrix, keys)
        error = values - predicted
        weighted_error = selection.weights.unsqueeze(-1) * error
        delta = torch.einsum("btv,btk->bkv", weighted_error, keys)

        token_gates = torch.sigmoid(self.gate(hidden_states).squeeze(-1))
        update_gate = (selection.weights * token_gates).sum(dim=-1).view(batch_size, 1, 1)
        new_matrix = state.matrix + update_gate * delta
        new_normalizer = state.normalizer + torch.einsum(
            "bt,btk->bk", selection.weights, keys.abs()
        )
        return MemoryState(new_matrix, new_normalizer), update_gate.squeeze(-1).squeeze(-1)

