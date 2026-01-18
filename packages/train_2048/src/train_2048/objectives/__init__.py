from __future__ import annotations

from typing import Optional

from .base import Objective
from .branch_ce import BranchCE
from .hard_move import HardMove
from .policy_value import PolicyValueCoral


def make_objective(
    mode: str,
    *,
    tokenizer_path: Optional[str] = None,
    cfg: Optional[object] = None,
) -> Objective:
    mode = mode.strip().lower()
    value_cfg = getattr(getattr(cfg, "target", None), "value_head", None)
    if value_cfg is not None and bool(getattr(value_cfg, "enabled", False)):
        tiles = list(getattr(value_cfg, "tiles", []) or [])
        if not tiles:
            raise ValueError("value_head.enabled requires non-empty value_head.tiles")
        return PolicyValueCoral(
            policy_mode=mode,
            tokenizer_path=tokenizer_path,
            tiles=tiles,
            include_underflow=bool(getattr(value_cfg, "include_underflow", True)),
            policy_weight=float(getattr(value_cfg, "policy_weight", 1.0)),
            value_weight=float(getattr(value_cfg, "value_weight", 1.0)),
            value_pooling=str(getattr(value_cfg, "pooling", "mean")),
            value_proj_dim=getattr(value_cfg, "proj_dim", None),
        )
    if mode == "binned_ev":
        return BranchCE(target_mode=mode, tokenizer_path=tokenizer_path)
    if mode == "hard_move":
        return HardMove()
    if mode == "macroxue_tokens":
        return BranchCE(target_mode=mode, tokenizer_path=tokenizer_path)
    raise ValueError(f"Unknown objective mode: {mode}")


__all__ = ["Objective", "make_objective", "BranchCE", "HardMove", "PolicyValueCoral"]
