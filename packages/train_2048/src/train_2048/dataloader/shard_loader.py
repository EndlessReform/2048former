"""Shard-based data loading for efficient random sampling."""
from __future__ import annotations

from pathlib import Path
from typing import Optional
import numpy as np


class ShardInfo:
    """Metadata about a single shard."""

    def __init__(self, path: Path, index: int, num_steps: int):
        self.path = path
        self.index = index
        self.num_steps = num_steps
        self.offset = 0  # Will be set when building cumulative offsets

    def __repr__(self) -> str:
        return f"ShardInfo(idx={self.index}, steps={self.num_steps}, path={self.path.name})"


class ShardLoader:
    """Loads and manages dataset shards.

    Supports two modes:
    - Lazy loading with mmap for low memory usage
    - Eager loading entire shards into RAM for fast random access
    """

    def __init__(self, dataset_dir: str, mmap_mode: bool = False):
        self.dataset_dir = Path(dataset_dir)
        self.mmap_mode = mmap_mode
        self.shards: list[ShardInfo] = []
        self._loaded_shards: dict[int, np.ndarray] = {}

        # Discover shards
        shard_paths = sorted(self.dataset_dir.glob("steps-*.npy"))
        if not shard_paths:
            # Fallback to single steps.npy
            steps_path = self.dataset_dir / "steps.npy"
            if not steps_path.is_file():
                raise FileNotFoundError(f"No steps.npy or steps-*.npy in {self.dataset_dir}")
            shard_paths = [steps_path]

        # Build shard info with cumulative offsets
        offset = 0
        for idx, path in enumerate(shard_paths):
            # Quick shape check without loading full array
            arr = np.load(str(path), mmap_mode='r')
            num_steps = arr.shape[0]
            shard_info = ShardInfo(path, idx, num_steps)
            shard_info.offset = offset
            self.shards.append(shard_info)
            offset += num_steps

        self.total_steps = offset

    def load_shard(self, shard_idx: int) -> np.ndarray:
        """Load a shard into memory (or return mmap view)."""
        if shard_idx in self._loaded_shards:
            return self._loaded_shards[shard_idx]

        shard = self.shards[shard_idx]
        mode = 'r' if self.mmap_mode else None
        arr = np.load(str(shard.path), mmap_mode=mode)

        # Cache if not mmap (mmap arrays are already "cached" by OS)
        if not self.mmap_mode:
            self._loaded_shards[shard_idx] = arr

        return arr

    def unload_shard(self, shard_idx: int) -> None:
        """Release a shard from memory cache."""
        self._loaded_shards.pop(shard_idx, None)

    def get_rows(self, global_indices: np.ndarray) -> np.ndarray:
        """Fetch rows by global index (legacy interface for compatibility)."""
        # Sort indices by shard for efficient access
        sorted_idx = np.argsort(global_indices)
        sorted_global = global_indices[sorted_idx]

        # Find which shard each index belongs to
        shard_boundaries = np.array([s.offset for s in self.shards] + [self.total_steps])
        shard_idx = np.searchsorted(shard_boundaries[:-1], sorted_global, side='right') - 1

        # Allocate output
        first_shard = self.load_shard(0)
        out = np.empty(len(global_indices), dtype=first_shard.dtype)

        # Gather from each shard
        pos = 0
        for sid in np.unique(shard_idx):
            mask = shard_idx == sid
            count = mask.sum()
            shard = self.load_shard(sid)
            shard_offset = self.shards[sid].offset
            local_idx = sorted_global[mask] - shard_offset
            out[pos:pos + count] = shard[local_idx]
            pos += count

        # Unsort to match original order
        unsort_idx = np.empty_like(sorted_idx)
        unsort_idx[sorted_idx] = np.arange(len(global_indices))
        return out[unsort_idx]

    def __repr__(self) -> str:
        mode = "mmap" if self.mmap_mode else "eager"
        return f"ShardLoader({len(self.shards)} shards, {self.total_steps:,} steps, mode={mode})"


class InMemoryShardPool:
    """Loads entire shard into RAM and provides random sampling from it.

    This is the key optimization: load one shard fully, sample from it randomly,
    then move to the next shard. No index materialization needed.
    """

    def __init__(self, shard_loader: ShardLoader):
        self.loader = shard_loader
        self.current_shard_idx: Optional[int] = None
        self.current_shard: Optional[np.ndarray] = None

    def load_shard_for_sampling(self, shard_idx: int) -> None:
        """Load a specific shard into memory for random sampling."""
        if self.current_shard_idx == shard_idx and self.current_shard is not None:
            return  # Already loaded

        # Unload previous shard to free memory
        if self.current_shard_idx is not None:
            self.loader.unload_shard(self.current_shard_idx)

        # Load new shard fully into memory (not mmap)
        arr = np.load(str(self.loader.shards[shard_idx].path), mmap_mode=None)
        self.current_shard = arr
        self.current_shard_idx = shard_idx

    def sample_from_current_shard(self, n_samples: int, rng: np.random.Generator) -> np.ndarray:
        """Sample n_samples random steps from currently loaded shard."""
        if self.current_shard is None:
            raise RuntimeError("No shard loaded. Call load_shard_for_sampling first.")

        indices = rng.integers(0, len(self.current_shard), size=n_samples)
        return self.current_shard[indices]

    def get_current_shard_size(self) -> int:
        """Return number of steps in current shard."""
        if self.current_shard is None:
            raise RuntimeError("No shard loaded")
        return len(self.current_shard)
