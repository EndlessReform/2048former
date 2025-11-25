from __future__ import annotations

from typing import Tuple

import torch


def muzero_transform(x: torch.Tensor, epsilon: float) -> torch.Tensor:
    """
    MuZero value transform: h(x) = sign(x) * (sqrt(|x| + 1) - 1 + eps * |x|).

    Args:
        x: input tensor
        epsilon: small positive constant from MuZero (e.g., 0.001)
    """
    eps = float(epsilon)
    abs_x = torch.abs(x)
    return torch.sign(x) * (torch.sqrt(abs_x + 1.0) - 1.0 + eps * abs_x)


def muzero_inverse(x: torch.Tensor, epsilon: float) -> torch.Tensor:
    """
    Inverse of the MuZero value transform.

    Handles the eps=0 edge case separately to avoid divide-by-zero.
    """
    eps = float(epsilon)
    abs_x = torch.abs(x)
    sign = torch.sign(x)
    if eps == 0.0:
        return sign * ((abs_x + 1.0) ** 2 - 1.0)
    # y = sqrt(a + 1) - 1 + eps * a  => solve for a >= 0
    # a = abs(x)
    tmp = torch.sqrt(1.0 + 4.0 * eps * (abs_x + 1.0 + eps))
    inner = (tmp - 1.0) / (2.0 * eps)
    return sign * (inner * inner - 1.0)


def scalar_to_two_hot(
    values: torch.Tensor,
    *,
    support_min: float,
    support_max: float,
    vocab_size: int,
    epsilon: float = 0.001,
    apply_transform: bool = True,
) -> torch.Tensor:
    """
    Project scalars onto adjacent integer supports (two-hot) for cross-entropy.

    Args:
        values: tensor of shape (batch,) containing raw targets
        support_min: minimum support value (inclusive)
        support_max: maximum support value (inclusive)
        vocab_size: number of support points; must equal (support_max - support_min) + 1
        epsilon: MuZero transform epsilon
        apply_transform: when True, apply MuZero transform before projection
    """
    if vocab_size <= 0:
        raise ValueError("vocab_size must be positive")
    step = (support_max - support_min) / max(1, vocab_size - 1)
    if step <= 0:
        raise ValueError("support_max must be greater than support_min")

    x = values.float()
    if apply_transform:
        x = muzero_transform(x, epsilon)

    x = x.clamp(min=support_min, max=support_max)
    pos = (x - support_min) / step
    low_idx = torch.floor(pos).long()
    high_idx = torch.clamp(low_idx + 1, max=vocab_size - 1)
    high_w = (pos - low_idx.float()).clamp(min=0.0, max=1.0)
    low_w = 1.0 - high_w

    target = torch.zeros((*x.shape, vocab_size), device=x.device, dtype=torch.float32)
    target.scatter_(-1, low_idx.unsqueeze(-1), low_w.unsqueeze(-1))
    target.scatter_(-1, high_idx.unsqueeze(-1), high_w.unsqueeze(-1))
    return target


def two_hot_to_scalar(
    probs: torch.Tensor,
    *,
    support_min: float,
    support_max: float,
    epsilon: float = 0.001,
    apply_inverse_transform: bool = True,
) -> torch.Tensor:
    """
    Recover a scalar estimate from a value distribution via expectation.

    Args:
        probs: tensor of shape (..., vocab_size) representing value logits/probs
        support_min: minimum support value (inclusive)
        support_max: maximum support value (inclusive)
        epsilon: MuZero transform epsilon
        apply_inverse_transform: when True, apply the inverse MuZero transform
    """
    vocab_size = probs.shape[-1]
    step = (support_max - support_min) / max(1, vocab_size - 1)
    supports = torch.linspace(
        support_min,
        support_max,
        vocab_size,
        device=probs.device,
        dtype=probs.dtype,
    )
    expected = (probs * supports).sum(dim=-1)
    if apply_inverse_transform:
        expected = muzero_inverse(expected, epsilon)
    return expected


__all__: Tuple[str, ...] = (
    "muzero_transform",
    "muzero_inverse",
    "scalar_to_two_hot",
    "two_hot_to_scalar",
)
