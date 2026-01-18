from __future__ import annotations

from typing import Optional

from .base import Objective
from .branch_ce import BranchCE
from .hard_move import HardMove


def make_objective(mode: str, *, tokenizer_path: Optional[str] = None) -> Objective:
    mode = mode.strip().lower()
    if mode == "binned_ev":
        return BranchCE(target_mode=mode, tokenizer_path=tokenizer_path)
    if mode == "hard_move":
        return HardMove()
    if mode == "macroxue_tokens":
        return BranchCE(target_mode=mode, tokenizer_path=tokenizer_path)
    raise ValueError(f"Unknown objective mode: {mode}")


__all__ = ["Objective", "make_objective", "BranchCE", "HardMove"]
