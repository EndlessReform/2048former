"""Dataset wrapper for shard-based loading."""
from __future__ import annotations

from typing import Optional
import numpy as np
from torch.utils.data import Dataset

from .shard_loader import ShardLoader


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

    def __getitem__(self, idx: int) -> int:
        """Return global index - collate will fetch actual data."""
        return int(idx)

    def get_rows(self, global_indices: np.ndarray) -> np.ndarray:
        """Fetch actual row data for given indices."""
        return self.shard_loader.get_rows(global_indices)
