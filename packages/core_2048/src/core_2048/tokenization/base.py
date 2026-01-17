from __future__ import annotations

from typing import Dict, Protocol

import torch


class EVTokenizer(Protocol):
    """Protocol for EV tokenizers that produce training targets from EVs."""

    def build_targets(self, *, evs: torch.Tensor, legal_mask: torch.Tensor) -> Dict[str, object]:
        """Return targets for training given branch EVs [B,4] and legality mask [B,4]."""


__all__ = ["EVTokenizer"]
