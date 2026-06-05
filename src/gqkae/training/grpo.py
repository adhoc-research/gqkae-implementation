"""GRPO utilities for sequence-level QSCI rewards."""

from __future__ import annotations

import torch


def normalize_advantages(rewards: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    rewards = rewards.float()
    if rewards.numel() == 1:
        return torch.zeros_like(rewards)
    return (rewards - rewards.mean()) / rewards.std(unbiased=False).clamp_min(eps)


def grpo_loss(
    new_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    clip_epsilon: float = 0.2,
) -> torch.Tensor:
    """Token-wise clipped GRPO objective from the paper.

    Args:
        new_log_probs: shape (batch, sequence_length)
        old_log_probs: shape (batch, sequence_length), detached policy probabilities from sampling
        advantages: shape (batch,), sequence-level relative advantages
    """
    ratio = torch.exp(new_log_probs - old_log_probs.detach())
    adv = advantages.detach().unsqueeze(-1)
    unclipped = ratio * adv
    clipped = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * adv
    return -torch.minimum(unclipped, clipped).mean()
