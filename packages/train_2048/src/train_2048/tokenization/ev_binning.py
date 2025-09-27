from __future__ import annotations

from typing import Dict

import torch

from ..binning import Binner, BinningConfig
from .base import EVTokenizer


class EVBinnerTokenizer(EVTokenizer):
    """Wrap Binner as an EVTokenizer to standardize EV → class targets.

    Produces per-branch integer bin targets in [0, n_bins-1].
    """

    def __init__(self, config: BinningConfig) -> None:
        self.binner = Binner.from_config(config)

    @property
    def n_bins(self) -> int:
        return self.binner.n_bins

    def to(self, device: torch.device) -> "EVBinnerTokenizer":
        self.binner.to_device(device)
        return self

    def build_targets(self, *, evs: torch.Tensor, legal_mask: torch.Tensor) -> Dict[str, object]:
        # legal_mask is not applied here; downstream losses will mask
        tgt = self.binner.bin_values(evs).long()
        return {"branch_bin_targets": tgt, "n_bins": int(self.n_bins)}


__all__ = ["EVBinnerTokenizer"]

