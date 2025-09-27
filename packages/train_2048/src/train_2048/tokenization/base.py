from __future__ import annotations

from typing import Dict, Protocol, Optional

import numpy as np
import torch


# Canonical branch order everywhere: UDLR (Up, Down, Left, Right)
BRANCH_ORDER_UDLR = (0, 1, 2, 3)


class BoardCodec:
    """Canonical helpers for decoding packed boards and legal bits."""

    @staticmethod
    def decode_packed_board_to_exps_u8(packed: np.ndarray, *, mask65536: Optional[np.ndarray] = None) -> np.ndarray:
        # MSB-first nibble order: cell i = bits [63-4i .. 60-4i]
        arr = packed.astype(np.uint64, copy=False)
        n = int(arr.shape[0])
        out = np.empty((n, 16), dtype=np.uint8)
        for i in range(16):
            shift = (15 - i) * 4
            out[:, i] = ((arr >> np.uint64(shift)) & np.uint64(0xF)).astype(np.uint8, copy=False)
        if mask65536 is not None:
            m = mask65536.astype(np.uint16, copy=False)
            for i in range(16):
                sel = ((m >> i) & np.uint16(1)) != 0
                if np.any(sel):
                    out[sel, i] = np.uint8(16)
        return out

    @staticmethod
    def legal_mask_from_bits_udlr(bits: np.ndarray) -> np.ndarray:
        # UDLR bit order: Up=1, Down=2, Left=4, Right=8
        b = bits.astype(np.uint8, copy=False)
        return np.stack([(b & 1) != 0, (b & 2) != 0, (b & 4) != 0, (b & 8) != 0], axis=1)


class EVTokenizer(Protocol):
    """Protocol for EV tokenizers that produce training targets from EVs."""

    def build_targets(self, *, evs: torch.Tensor, legal_mask: torch.Tensor) -> Dict[str, object]:
        """Return targets for training given branch EVs [B,4] and legality mask [B,4]."""


__all__ = [
    "BRANCH_ORDER_UDLR",
    "BoardCodec",
    "EVTokenizer",
]
