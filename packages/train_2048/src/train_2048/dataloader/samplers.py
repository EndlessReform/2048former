"""Sampling strategies for training data."""
from __future__ import annotations

from typing import Iterator, Optional
import numpy as np
from torch.utils.data import Sampler

from .shard_loader import ShardLoader, InMemoryShardPool


class ShardPoolSampler(Sampler[int]):
    """Sample random steps from shards loaded sequentially into RAM.

    Strategy:
    1. Load entire shard into RAM
    2. Randomly sample all steps from that shard (shuffled, without replacement)
    3. Move to next shard
    4. Repeat for desired number of epochs

    This avoids:
    - Full dataset scans
    - Index materialization for billions of rows
    - Random disk seeks across shards

    Trust the metadata DB completely for run/shard info.
    """

    def __init__(
        self,
        shard_loader: ShardLoader,
        num_epochs: int = 1,
        *,
        seed: int = 42,
        run_ids: Optional[np.ndarray] = None,
        total_steps: Optional[int] = None,
        skip: int = 0,
    ):
        """
        Args:
            shard_loader: ShardLoader instance
            num_epochs: Number of times to iterate through all shards
            seed: Random seed
            run_ids: Optional filter - only sample from these run_ids
                Note: This still requires filtering, but done in-memory per shard
            total_steps: Total steps (from metadata) - avoids expensive counting
        """
        self.loader = shard_loader
        self.num_epochs = num_epochs
        self.seed = seed
        self.run_ids = run_ids
        self.pool = InMemoryShardPool(shard_loader)
        per_epoch = int(total_steps) if total_steps is not None else int(self.loader.total_steps)
        self._total_per_epoch = max(per_epoch, 0)
        self.num_epochs = max(1, int(num_epochs))
        self._total_length = self._total_per_epoch * self.num_epochs
        self.skip = max(0, int(skip))
        if self.skip > self._total_length:
            self.skip = self.skip % self._total_length if self._total_length > 0 else 0
        self._skip_remaining = self.skip

    def __iter__(self) -> Iterator[int]:
        rng = np.random.default_rng(self.seed)
        skip_remaining = self._skip_remaining
        self._skip_remaining = 0

        for epoch in range(self.num_epochs):
            # Iterate through all shards sequentially
            for shard_idx in range(len(self.loader.shards)):
                # Load shard fully into RAM
                self.pool.load_shard_for_sampling(shard_idx)
                shard_data = self.pool.current_shard
                shard_offset = self.loader.shards[shard_idx].offset

                # Get eligible indices within this shard
                if self.run_ids is not None:
                    mask = np.isin(shard_data['run_id'], self.run_ids)
                    eligible_indices = np.flatnonzero(mask)
                    if len(eligible_indices) == 0:
                        continue  # No eligible steps in this shard
                else:
                    eligible_indices = np.arange(len(shard_data))

                # Shuffle the eligible indices for this shard
                rng.shuffle(eligible_indices)

                # Yield all shuffled indices from this shard
                for local_idx in eligible_indices:
                    if skip_remaining > 0:
                        skip_remaining -= 1
                        continue
                    yield int(local_idx + shard_offset)

    def __len__(self) -> int:
        if self._total_length == 0:
            return 0
        effective = self._total_length - min(self.skip, self._total_length)
        return max(0, effective)


class BufferedShuffleSampler(Sampler[int]):
    """Memory-efficient shuffle using a ring buffer.

    One-pass shuffle without materializing full permutation.
    Good for when you want to iterate the full dataset once, shuffled.
    """

    def __init__(
        self,
        dataset_len: int,
        buffer_size: int = 1_000_000,
        seed: int = 42,
        skip: int = 0,
    ):
        self.dataset_len = int(dataset_len)
        self.buffer_size = min(buffer_size, self.dataset_len)
        self.seed = seed
        self.skip = max(0, int(skip))
        if self.skip > self.dataset_len:
            self.skip = self.skip % self.dataset_len if self.dataset_len > 0 else 0
        self._skip_consumed = False

    def __iter__(self) -> Iterator[int]:
        rng = np.random.default_rng(self.seed)
        buffer = np.arange(self.buffer_size, dtype=np.int64)
        next_idx = self.buffer_size
        skip_remaining = 0 if self._skip_consumed else self.skip
        self._skip_consumed = True

        # Reservoir sampling with immediate yield
        while next_idx < self.dataset_len:
            j = rng.integers(0, self.buffer_size)
            if skip_remaining > 0:
                skip_remaining -= 1
            else:
                yield int(buffer[j])
            buffer[j] = next_idx
            next_idx += 1

        # Drain buffer
        rng.shuffle(buffer)
        for idx in buffer:
            if skip_remaining > 0:
                skip_remaining -= 1
                continue
            yield int(idx)

    def __len__(self) -> int:
        return max(0, self.dataset_len - min(self.skip, self.dataset_len))


class SequentialSampler(Sampler[int]):
    """Simple sequential sampler for validation."""

    def __init__(self, dataset_len: int):
        self.dataset_len = dataset_len

    def __iter__(self) -> Iterator[int]:
        for i in range(self.dataset_len):
            yield i

    def __len__(self) -> int:
        return self.dataset_len
