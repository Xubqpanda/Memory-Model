from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class SelectionOutput:
    scores: torch.Tensor
    probabilities: torch.Tensor
    selected: torch.Tensor
    weights: torch.Tensor


class AlphaTopPSelector(nn.Module):
    """Learned token importance scorer with straight-through Alpha Top-P.

    The hard forward pass writes the smallest set whose probability mass reaches
    ``alpha_top_p``. During training, the backward pass follows the dense
    softmax distribution.

    Reference: Metis, MemTensor (2026), Alpha Top-P token selection.
    """

    def __init__(
        self,
        hidden_size: int,
        alpha_top_p: float = 0.9,
        *,
        straight_through: bool = True,
    ) -> None:
        super().__init__()
        if not 0.0 < alpha_top_p <= 1.0:
            raise ValueError("alpha_top_p must be in (0, 1]")
        self.alpha_top_p = alpha_top_p
        self.straight_through = straight_through
        self.pool_score = nn.Linear(hidden_size, 1, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> SelectionOutput:
        if hidden_states.ndim != 3:
            raise ValueError("hidden_states must have shape [batch, sequence, hidden]")
        scores = self.pool_score(hidden_states).squeeze(-1)
        if valid_mask is not None:
            if valid_mask.shape != scores.shape or valid_mask.dtype != torch.bool:
                raise ValueError("valid_mask must be a boolean tensor with shape [batch, sequence]")
            if (~valid_mask).all(dim=-1).any():
                raise ValueError("each sequence must contain at least one valid token")
            scores = scores.masked_fill(~valid_mask, float("-inf"))

        probabilities = F.softmax(scores, dim=-1)
        sorted_probabilities, sorted_indices = torch.sort(
            probabilities, dim=-1, descending=True
        )
        cumulative = sorted_probabilities.cumsum(dim=-1)
        # Include the token that crosses the threshold.
        keep_sorted = (cumulative - sorted_probabilities) < self.alpha_top_p
        selected = torch.zeros_like(keep_sorted).scatter(1, sorted_indices, keep_sorted)
        selected = selected & probabilities.gt(0)

        hard_weights = probabilities * selected.to(probabilities.dtype)
        hard_weights = hard_weights / hard_weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        if self.straight_through:
            weights = hard_weights.detach() - probabilities.detach() + probabilities
        else:
            weights = hard_weights
        return SelectionOutput(
            scores=scores,
            probabilities=probabilities,
            selected=selected,
            weights=weights,
        )
