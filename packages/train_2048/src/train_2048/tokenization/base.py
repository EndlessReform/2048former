from __future__ import annotations

from typing import Dict, Protocol, Optional

import numpy as np
import torch


# Branch order for macroxue tokens: URDL (Up, Right, Down, Left)
BRANCH_ORDER_URDL = (0, 1, 2, 3)
# Canonical project order (requested): UDLR
BRANCH_ORDER_UDLR = (0, 2, 3, 1)


def remap_labels_urdl_to_udlr(labels: np.ndarray) -> np.ndarray:
    """Map class labels in URDL order to UDLR order for hard_move.

    Mapping: 0->0 (U), 1->3 (R), 2->1 (D), 3->2 (L)
    """
    perm = np.array([0, 3, 1, 2], dtype=np.int64)
    if labels.size:
        return perm[labels]
    return labels


def reorder_cols_urdl_to_udlr(arr: np.ndarray) -> np.ndarray:
    """Reorder last-dimension columns from URDL to UDLR order.

    Expects shape (..., 4). Returns a view when possible.
    """
    return arr[..., [0, 2, 3, 1]]


def reorder_cols_urdl_to_udlr_t(arr: torch.Tensor) -> torch.Tensor:
    """Torch variant of URDL→UDLR column reorder for (...,4) tensors."""
    return arr.index_select(-1, torch.tensor([0, 2, 3, 1], device=arr.device))


class BoardCodec:
    """Canonical helpers for decoding packed boards and legal bits."""

    @staticmethod
    def decode_packed_board_to_exps_u8(packed: np.ndarray, *, mask65536: Optional[np.ndarray] = None) -> np.ndarray:
        arr = packed.astype(np.uint64, copy=False)
        n = int(arr.shape[0])
        out = np.empty((n, 16), dtype=np.uint8)
        for i in range(16):
            out[:, i] = ((arr >> (4 * i)) & np.uint64(0xF)).astype(np.uint8, copy=False)
        if mask65536 is not None:
            m = mask65536.astype(np.uint16, copy=False)
            for i in range(16):
                sel = ((m >> i) & np.uint16(1)) != 0
                if np.any(sel):
                    out[sel, i] = np.uint8(16)
        return out

    @staticmethod
    def legal_mask_from_bits_urdl(bits: np.ndarray) -> np.ndarray:
        b = bits.astype(np.uint8, copy=False)
        return np.stack([(b & 1) != 0, (b & 2) != 0, (b & 4) != 0, (b & 8) != 0], axis=1)


class EVTokenizer(Protocol):
    """Protocol for EV tokenizers that produce training targets from EVs."""

    def build_targets(self, *, evs: torch.Tensor, legal_mask: torch.Tensor) -> Dict[str, object]:
        """Return targets for training given branch EVs [B,4] and legality mask [B,4]."""


__all__ = [
    "BRANCH_ORDER_URDL",
    "BRANCH_ORDER_UDLR",
    "remap_labels_urdl_to_udlr",
    "reorder_cols_urdl_to_udlr",
    "reorder_cols_urdl_to_udlr_t",
    "BoardCodec",
    "EVTokenizer",
]
