from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class DPOLossOutput:
    loss: torch.Tensor
    losses: torch.Tensor
    chosen_rewards: torch.Tensor
    rejected_rewards: torch.Tensor
    reward_margins: torch.Tensor
    preference_accuracy: torch.Tensor
    policy_chosen_logps: torch.Tensor
    policy_rejected_logps: torch.Tensor


def disable_model_dropout(model: nn.Module) -> None:
    """Disable module and functional dropout paths for deterministic preferences."""

    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.p = 0.0
        # CausalSelfAttention stores the SDPA dropout probability as a float.
        dropout = getattr(module, "dropout", None)
        if isinstance(dropout, float):
            module.dropout = 0.0


def sequence_log_probs(
    logits: torch.Tensor,
    labels: torch.Tensor,
    loss_mask: torch.Tensor,
    *,
    average: bool = False,
) -> torch.Tensor:
    """Return assistant-response log probability for every sequence."""

    if logits.shape[:2] != labels.shape or labels.shape != loss_mask.shape:
        raise ValueError("logits, labels, and loss_mask shapes are inconsistent")
    token_logps = -F.cross_entropy(
        logits.transpose(1, 2),
        labels,
        reduction="none",
    )
    mask = loss_mask.to(token_logps.dtype)
    sequence_logps = (token_logps * mask).sum(dim=-1)
    if average:
        sequence_logps = sequence_logps / mask.sum(dim=-1).clamp_min(1)
    return sequence_logps


def dpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    reference_chosen_logps: torch.Tensor,
    reference_rejected_logps: torch.Tensor,
    *,
    beta: float = 0.1,
    label_smoothing: float = 0.0,
) -> DPOLossOutput:
    """Compute the reference-anchored Direct Preference Optimization loss."""

    if beta <= 0:
        raise ValueError("beta must be positive")
    if not 0.0 <= label_smoothing < 0.5:
        raise ValueError("label_smoothing must be in [0, 0.5)")
    shapes = {
        policy_chosen_logps.shape,
        policy_rejected_logps.shape,
        reference_chosen_logps.shape,
        reference_rejected_logps.shape,
    }
    if len(shapes) != 1:
        raise ValueError("all chosen/rejected log-probability tensors must have the same shape")

    policy_logratios = policy_chosen_logps - policy_rejected_logps
    reference_logratios = reference_chosen_logps - reference_rejected_logps
    preference_logits = beta * (policy_logratios - reference_logratios)
    losses = -(
        (1.0 - label_smoothing) * F.logsigmoid(preference_logits)
        + label_smoothing * F.logsigmoid(-preference_logits)
    )

    chosen_rewards = beta * (policy_chosen_logps - reference_chosen_logps).detach()
    rejected_rewards = beta * (policy_rejected_logps - reference_rejected_logps).detach()
    reward_margins = chosen_rewards - rejected_rewards
    preference_accuracy = (reward_margins > 0).to(torch.float32).mean()
    return DPOLossOutput(
        loss=losses.mean(),
        losses=losses,
        chosen_rewards=chosen_rewards,
        rejected_rewards=rejected_rewards,
        reward_margins=reward_margins,
        preference_accuracy=preference_accuracy,
        policy_chosen_logps=policy_chosen_logps.detach(),
        policy_rejected_logps=policy_rejected_logps.detach(),
    )
