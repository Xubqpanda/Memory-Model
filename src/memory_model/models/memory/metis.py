from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .fusion import MemoryFusion
from .projection import MemoryProjections
from .read import MemoryReader
from .selector import AlphaTopPSelector, SelectionOutput
from .state import MemoryState
from .write import GatedDeltaWriter


@dataclass
class MemoryOutput:
    fused: torch.Tensor
    retrieved: torch.Tensor
    state: MemoryState
    selection: SelectionOutput
    update_gate: torch.Tensor


class MetisLiteMemory(nn.Module):
    """A small, standalone approximation of Metis Hyper Memory.

    It reads the previous state first, then writes the current interaction into
    a new state. The Backbone is external and can remain completely frozen.
    """

    def __init__(
        self,
        hidden_size: int,
        key_dim: int,
        value_dim: int,
        *,
        alpha_top_p: float = 0.9,
        straight_through: bool = True,
        fusion_weight: float = 0.1,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.projections = MemoryProjections(hidden_size, key_dim, value_dim)
        self.selector = AlphaTopPSelector(
            hidden_size, alpha_top_p, straight_through=straight_through
        )
        self.reader = MemoryReader()
        self.writer = GatedDeltaWriter(hidden_size)
        self.fusion = MemoryFusion(
            fusion_weight, memory_size=value_dim, backbone_size=hidden_size
        )

    def initial_state(
        self,
        batch_size: int,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> MemoryState:
        return MemoryState.zeros(
            batch_size,
            self.key_dim,
            self.value_dim,
            device=device,
            dtype=dtype,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        state: MemoryState | None = None,
        *,
        valid_mask: torch.Tensor | None = None,
        update_state: bool = True,
    ) -> MemoryOutput:
        if hidden_states.ndim != 3:
            raise ValueError("hidden_states must have shape [batch, sequence, hidden]")
        batch_size = hidden_states.size(0)
        if state is None:
            state = self.initial_state(
                batch_size, device=hidden_states.device, dtype=hidden_states.dtype
            )
        state.validate(batch_size, self.key_dim, self.value_dim)

        query, keys, values = self.projections(hidden_states)
        retrieved = self.reader(query, state)
        selection = self.selector(hidden_states, valid_mask)
        if update_state:
            new_state, update_gate = self.writer(
                hidden_states, keys, values, selection, state
            )
        else:
            new_state = state
            update_gate = torch.zeros(batch_size, device=hidden_states.device, dtype=hidden_states.dtype)
        fused = self.fusion(hidden_states, retrieved)
        return MemoryOutput(fused, retrieved, new_state, selection, update_gate)
