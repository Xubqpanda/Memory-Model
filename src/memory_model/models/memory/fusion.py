from __future__ import annotations

import torch
import torch.nn as nn


class MemoryFusion(nn.Module):
    """Add a learnable, initially small memory branch to the residual stream."""

    def __init__(
        self,
        initial_weight: float = 0.1,
        *,
        memory_size: int | None = None,
        backbone_size: int | None = None,
    ) -> None:
        super().__init__()
        if not 0.0 < initial_weight < 1.0:
            raise ValueError("initial_weight must be in (0, 1)")
        logit = torch.logit(torch.tensor(initial_weight))
        self.gate_logit = nn.Parameter(logit)
        if (memory_size is None) != (backbone_size is None):
            raise ValueError("memory_size and backbone_size must be specified together")
        if memory_size is None or memory_size == backbone_size:
            self.memory_projection: nn.Module = nn.Identity()
        else:
            self.memory_projection = nn.Linear(memory_size, backbone_size, bias=False)

    def forward(self, backbone_output: torch.Tensor, memory_output: torch.Tensor) -> torch.Tensor:
        memory_output = self.memory_projection(memory_output)
        if backbone_output.shape != memory_output.shape:
            raise ValueError("backbone_output and memory_output must have the same shape")
        return backbone_output + torch.sigmoid(self.gate_logit) * memory_output
