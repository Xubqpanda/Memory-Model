from dataclasses import dataclass

import torch


KVCache = tuple[torch.Tensor, torch.Tensor]


@dataclass
class ModelOutput:
    logits: torch.Tensor
    loss: torch.Tensor | None = None
    past_key_values: list[KVCache] | None = None
