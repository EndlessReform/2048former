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

    def __init__(
        self,
        dataset_dir: str,
        mmap_mode: bool = False,
        *,
        value_sidecar: bool = False,
        expected_total_steps: Optional[int] = None,
    ):
        self.dataset_dir = Path(dataset_dir)
        self.mmap_mode = mmap_mode
        self.shards: list[ShardInfo] = []
        self._loaded_shards: dict[int, np.ndarray] = {}
        # Optional value sidecar shards (values.npy / values-*.npy)
        self._value_paths: list[Path] | None = None
        self._value_offsets: list[int] | None = None
        self._loaded_value_shards: dict[int, np.ndarray] = {}
        self.value_total_steps: Optional[int] = None

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
        if value_sidecar:
            self._init_value_sidecar(expected_total_steps)

    def _init_value_sidecar(self, expected_total_steps: Optional[int]) -> None:
        """Load metadata for the value sidecar and validate alignment."""
        value_paths = sorted(self.dataset_dir.glob("values-*.npy"))
        if not value_paths:
            single = self.dataset_dir / "values.npy"
            if single.is_file():
                value_paths = [single]
        if not value_paths:
            raise FileNotFoundError(f"value_sidecar requested but no values.npy/values-*.npy found in {self.dataset_dir}")
        if len(value_paths) != len(self.shards):
            raise ValueError(
                f"value sidecar shard count ({len(value_paths)}) does not match steps shards ({len(self.shards)})"
            )

        offsets: list[int] = []
        total = 0
        # Lightweight alignment check on run_id/step_index without materializing full arrays
        sample_positions = (0, 1, -1)
        for idx, (val_path, step_info) in enumerate(zip(value_paths, self.shards)):
            val_arr = np.load(str(val_path), mmap_mode="r")
            rows = val_arr.shape[0]
            if rows != step_info.num_steps:
                raise ValueError(
                    f"value shard {val_path.name} has {rows} rows but steps shard {step_info.path.name} has {step_info.num_steps}"
                )
            for required in ("run_id", "step_index"):
                if required not in val_arr.dtype.names:
                    raise ValueError(f"value shard {val_path.name} missing '{required}' column")
            # Spot-check alignment on a few positions (first/second/last) to catch obvious drift.
            if rows > 0:
                step_arr = np.load(str(step_info.path), mmap_mode="r")
                for required in ("run_id", "step_index"):
                    if required not in step_arr.dtype.names:
                        raise ValueError(f"steps shard {step_info.path.name} missing '{required}' column")
                for pos in {p if p >= 0 else rows + p for p in sample_positions if abs(p) < rows}:
                    s_row = step_arr[int(pos)]
                    v_row = val_arr[int(pos)]
                    if s_row["run_id"] != v_row["run_id"] or s_row["step_index"] != v_row["step_index"]:
                        raise ValueError(
                            f"value shard {val_path.name} mismatch at position {pos}: "
                            f"steps run_id={s_row['run_id']} step_index={s_row['step_index']} "
                            f"vs values run_id={v_row['run_id']} step_index={v_row['step_index']}"
                        )
            offsets.append(total)
            total += rows

        if expected_total_steps is not None and total != int(expected_total_steps):
            raise ValueError(
                f"value sidecar rows ({total}) do not match metadata steps ({expected_total_steps})"
            )
        if total != self.total_steps:
            raise ValueError(
                f"value sidecar rows ({total}) do not match steps rows ({self.total_steps})"
            )
        self._value_paths = value_paths
        self._value_offsets = offsets
        self.value_total_steps = total

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

    def load_value_shard(self, shard_idx: int) -> np.ndarray:
        """Load a value shard into memory (or return mmap view)."""
        if self._value_paths is None or self._value_offsets is None:
            raise RuntimeError("value sidecar not initialised for this ShardLoader")
        if shard_idx in self._loaded_value_shards:
            return self._loaded_value_shards[shard_idx]

        path = self._value_paths[shard_idx]
        mode = "r" if self.mmap_mode else None
        arr = np.load(str(path), mmap_mode=mode)
        if not self.mmap_mode:
            self._loaded_value_shards[shard_idx] = arr
        return arr

    def unload_shard(self, shard_idx: int) -> None:
        """Release a shard from memory cache."""
        self._loaded_shards.pop(shard_idx, None)
        self._loaded_value_shards.pop(shard_idx, None)

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

    def get_value_rows(self, global_indices: np.ndarray) -> np.ndarray:
        """Fetch value sidecar rows aligned with the given global indices."""
        if self._value_paths is None or self._value_offsets is None:
            raise RuntimeError("value sidecar not initialised for this ShardLoader")

        sorted_idx = np.argsort(global_indices)
        sorted_global = global_indices[sorted_idx]
        shard_boundaries = np.array(list(self._value_offsets) + [self.total_steps])  # type: ignore[arg-type]
        shard_idx = np.searchsorted(shard_boundaries[:-1], sorted_global, side="right") - 1

        first_value = self.load_value_shard(0)
        out = np.empty(len(global_indices), dtype=first_value.dtype)

        pos = 0
        for sid in np.unique(shard_idx):
            mask = shard_idx == sid
            count = mask.sum()
            shard = self.load_value_shard(sid)
            shard_offset = self._value_offsets[sid]
            local_idx = sorted_global[mask] - shard_offset
            out[pos:pos + count] = shard[local_idx]
            pos += count

        unsort_idx = np.empty_like(sorted_idx)
        unsort_idx[sorted_idx] = np.arange(len(global_indices))
        return out[unsort_idx]

    def __repr__(self) -> str:
        mode = "mmap" if self.mmap_mode else "eager"
        value = " +values" if self._value_paths else ""
        return f"ShardLoader({len(self.shards)} shards, {self.total_steps:,} steps, mode={mode}{value})"

    def has_value_sidecar(self) -> bool:
        return self._value_paths is not None


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
