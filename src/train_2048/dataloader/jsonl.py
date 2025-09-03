from __future__ import annotations

import torch
from typing import Callable, Optional

from ..binning import Binner


def make_collate_hf_steps(binner: Optional[Binner]) -> Callable:
    """Collate raw HF JSONL step items into tensors and optional bin targets.

    Each item should be a dict with keys:
    - pre_board: list[int] of length 16 (preferred) or raw board (int/str)
    - branches: list of {legal: bool, value: float}
    """

    def _collate(items: list[dict]):
        from ai_2048 import Board  # lazy import

        def _tokens_from_item(it: dict) -> list[int]:
            pb = it.get("pre_board")
            if isinstance(pb, (list, tuple)) and len(pb) == 16:
                return [int(x) for x in pb]
            return list(Board.from_raw(int(pb)).to_exponents())

        tokens = torch.tensor([_tokens_from_item(it) for it in items], dtype=torch.int64)

        mask_rows, val_rows = [], []
        for it in items:
            brs = it.get("branches") or []
            row_m, row_v = [], []
            for i in range(4):
                if i < len(brs):
                    bi = brs[i]
                    legal = bool(bi.get("legal", False))
                    val = float(bi.get("value", 0.0))
                else:
                    legal, val = False, 0.0
                row_m.append(legal)
                row_v.append(val if legal else 0.0)
            mask_rows.append(row_m)
            val_rows.append(row_v)

        branch_mask = torch.tensor(mask_rows, dtype=torch.bool)
        branch_vals = torch.tensor(val_rows, dtype=torch.float32)

        out = {"tokens": tokens, "branch_mask": branch_mask, "branch_values": branch_vals}
        if binner is not None:
            binner.to_device(branch_vals.device)
            out["branch_bin_targets"] = binner.bin_values(branch_vals).long()
            out["n_bins"] = binner.n_bins
        return out

    return _collate


__all__ = ["make_collate_hf_steps"]

