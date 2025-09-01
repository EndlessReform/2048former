from typing import Literal, Optional

import torch


@torch.inference_mode()
def select_move(
    head_probs: list[torch.Tensor],
    *,
    legal_mask: Optional[torch.Tensor] = None,
    strategy: Literal["max-p1"] = "max-p1",
    one_bin_index: Optional[int] = None,
) -> torch.Tensor:
    """
    Select a move index per batch according to a simple strategy.

    Current default strategy "max-p1": choose the direction whose head assigns
    the highest probability to the bin corresponding to value==1.

    Args:
      head_probs: list of 4 tensors (B, n_bins), softmax probabilities per head.
      legal_mask: optional (B, 4) bool tensor to mask illegal moves.
      strategy: currently only "max-p1"; reserved for future strategies.
      one_bin_index: override the index used for the "1" bin. Defaults to n_bins-1.

    Returns:
      Long tensor of shape (B,) with selected move indices in {0,1,2,3}.
    """
    if len(head_probs) != 4:
        raise ValueError("Expected 4 heads (up, down, left, right)")

    B = head_probs[0].size(0)
    n_bins = head_probs[0].size(1)
    if one_bin_index is None:
        one_bin_index = n_bins - 1  # default: dedicated '1' bin at the end

    # Stack p1 across heads -> (B, 4)
    p1 = torch.stack([hp[:, one_bin_index] for hp in head_probs], dim=1)

    if legal_mask is not None:
        # Ensure mask is on same device and boolean dtype
        if legal_mask.device != p1.device or legal_mask.dtype != torch.bool:
            legal_mask = legal_mask.to(device=p1.device, dtype=torch.bool)
        if legal_mask.shape != p1.shape:
            raise ValueError(
                f"legal_mask must have shape {p1.shape}, got {tuple(legal_mask.shape)}"
            )
        # Very negative for illegal so they won't be chosen; fallback if none legal
        masked = p1.masked_fill(~legal_mask, float("-inf"))
        no_legal = ~torch.isfinite(masked).any(dim=1)
        best_masked = masked.argmax(dim=1)
        if no_legal.any():
            best_unmasked = p1.argmax(dim=1)
            best_masked = torch.where(no_legal, best_unmasked, best_masked)
        return best_masked.long()

    # No mask: global argmax on p1 across heads
    return p1.argmax(dim=1).long()
