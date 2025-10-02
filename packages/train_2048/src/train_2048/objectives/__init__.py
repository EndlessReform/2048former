from __future__ import annotations

from typing import Optional

from .base import Objective
from .binned_ev import BinnedEV
from .hard_move import HardMove
from .macroxue_tokens import MacroxueTokens
from .value_head import ValueOrdinal, ValueCategorical


def make_objective(mode: str, *, tokenizer_path: Optional[str] = None) -> Objective:
    mode = mode.strip().lower()
    if mode == "binned_ev":
        return BinnedEV()
    if mode == "hard_move":
        return HardMove()
    if mode == "macroxue_tokens":
        return MacroxueTokens(tokenizer_path=tokenizer_path)
    if mode == "value_ordinal":
        return ValueOrdinal()
    if mode == "value_categorical":
        return ValueCategorical()
    raise ValueError(f"Unknown objective mode: {mode}")


__all__ = [
    "Objective",
    "make_objective",
    "BinnedEV",
    "HardMove",
    "MacroxueTokens",
    "ValueOrdinal",
    "ValueCategorical",
]
