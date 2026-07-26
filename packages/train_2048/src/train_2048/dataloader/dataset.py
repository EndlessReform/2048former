"""Dataset wrapper for shard-based loading."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import numpy as np
from torch.utils.data import Dataset

from .shard_loader import ShardLoader


@dataclass(frozen=True)
class SamplerCursor:
    """Position of the next sample in a deterministic shard permutation."""

    seed: int
    epoch: int
    shard: int
    position: int

    def as_dict(self) -> dict[str, int]:
        return {
            "version": 1,
            "seed": int(self.seed),
            "epoch": int(self.epoch),
            "shard": int(self.shard),
            "position": int(self.position),
        }

    @classmethod
    def from_dict(cls, state: object) -> Optional["SamplerCursor"]:
        if not isinstance(state, dict) or int(state.get("version", 0)) != 1:
            return None
        try:
            return cls(
                seed=int(state["seed"]),
                epoch=max(0, int(state["epoch"])),
                shard=max(0, int(state["shard"])),
                position=max(0, int(state["position"])),
            )
        except (KeyError, TypeError, ValueError):
            return None


@dataclass(frozen=True)
class SampleRef:
    """Physical row index plus the cursor after consuming that row."""

    global_index: int
    next_cursor: SamplerCursor


class ShardDataset(Dataset):
    """Lightweight dataset that delegates to ShardLoader.

    Only returns global indices - actual data fetching happens in collate_fn.
    This keeps the dataset stateless and simple.
    """

    def __init__(self, shard_loader: ShardLoader, length: int):
        """
        Args:
            shard_loader: Manages shard loading
            length: Logical length of dataset (for sampler sizing)
        """
        self.shard_loader = shard_loader
        self._length = length

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, idx: int | SampleRef) -> int | SampleRef:
        """Return global index - collate will fetch actual data."""
        if isinstance(idx, SampleRef):
            return idx
        return int(idx)

    def get_rows(self, global_indices: np.ndarray) -> np.ndarray:
        """Fetch actual row data for given indices."""
        return self.shard_loader.get_rows(global_indices)
