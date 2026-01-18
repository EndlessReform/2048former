"""Shard-based data loading for efficient random sampling."""
from __future__ import annotations

from pathlib import Path
from typing import Optional
import hashlib
import fcntl
import io
import os
import shutil
import numpy as np
import zstandard


def _is_zst(path: Path) -> bool:
    return path.suffix == ".zst"


def _shard_key(path: Path) -> str:
    name = path.name
    return name[:-4] if name.endswith(".zst") else name


def _read_npy_header(path: Path) -> tuple:
    """Read just the shape and dtype from a .npy file without loading the full array.

    For compressed files, decompresses only the header portion.
    Returns (shape, dtype).
    """
    if _is_zst(path):
        # Decompress just enough to read the .npy header (typically < 1KB)
        # The .npy header format is: magic (6 bytes) + version (2) + header_len (2 or 4) + header dict
        dctx = zstandard.ZstdDecompressor()
        try:
            with path.open("rb") as handle:
                with dctx.stream_reader(handle) as reader:
                    # Read first 64KB which is more than enough for any .npy header
                    header_data = reader.read(65536)
        except zstandard.ZstdError as exc:
            raise RuntimeError(f"Corrupt shard {path}: {exc}") from exc

        # Parse the header using numpy
        with io.BytesIO(header_data) as bio:
            import numpy.lib.format as npy_format
            bio.seek(0)
            version = npy_format.read_magic(bio)
            # Use version-specific header reader
            if version == (1, 0):
                shape, fortran, dtype = npy_format.read_array_header_1_0(bio)
            elif version == (2, 0):
                shape, fortran, dtype = npy_format.read_array_header_2_0(bio)
            else:
                raise ValueError(f"Unsupported .npy version: {version}")
            return shape, dtype
    else:
        # For uncompressed files, use mmap to avoid loading into memory
        arr = np.load(str(path), mmap_mode='r')
        return arr.shape, arr.dtype


def _load_npy(path: Path, *, mmap_mode: Optional[str]) -> np.ndarray:
    if _is_zst(path):
        if mmap_mode is not None:
            raise ValueError(f"mmap_mode is not supported for compressed shard {path}")
        dctx = zstandard.ZstdDecompressor()
        try:
            with path.open("rb") as handle:
                with dctx.stream_reader(handle) as reader:
                    data = reader.read()
        except zstandard.ZstdError as exc:
            raise RuntimeError(f"Corrupt shard {path}: {exc}") from exc
        return np.load(io.BytesIO(data))
    return np.load(str(path), mmap_mode=mmap_mode)


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
        cache_shards: bool = True,
        *,
        cache_keep_shards: int = 1,
        decompress_dir: Optional[str] = None,
        decompress_cleanup: bool = True,
    ):
        self.dataset_dir = Path(dataset_dir)
        self.mmap_mode = mmap_mode
        self.cache_shards = cache_shards
        self.cache_keep_shards = max(1, int(cache_keep_shards))
        self.decompress_cleanup = bool(decompress_cleanup)
        self.shards: list[ShardInfo] = []
        self._loaded_shards: dict[int, np.ndarray] = {}
        self._loaded_order: list[int] = []
        self._dtype: Optional[np.dtype] = None
        self._decompress_cache_dir: Optional[Path] = None
        self._decompressed_paths: dict[int, Path] = {}
        self._leases: dict[int, Path] = {}

        if decompress_dir is not None:
            cache_root = Path(decompress_dir)
            cache_root.mkdir(parents=True, exist_ok=True)
            # Namespace by dataset path to avoid collisions across runs.
            digest = hashlib.sha1(str(self.dataset_dir.resolve()).encode("utf-8")).hexdigest()[:16]
            self._decompress_cache_dir = cache_root / f"train_2048_shards_{digest}"
            self._decompress_cache_dir.mkdir(parents=True, exist_ok=True)

        # Discover shards
        shard_paths = sorted(
            list(self.dataset_dir.glob("steps-*.npy"))
            + list(self.dataset_dir.glob("steps-*.npy.zst"))
        )
        if not shard_paths:
            # Fallback to single steps.npy
            steps_path = self.dataset_dir / "steps.npy"
            steps_path_zst = self.dataset_dir / "steps.npy.zst"
            if steps_path.is_file() and steps_path_zst.is_file():
                raise FileNotFoundError(
                    f"Both steps.npy and steps.npy.zst exist in {self.dataset_dir}"
                )
            if steps_path.is_file():
                shard_paths = [steps_path]
            elif steps_path_zst.is_file():
                shard_paths = [steps_path_zst]
            else:
                raise FileNotFoundError(
                    f"No steps.npy[.zst] or steps-*.npy[.zst] in {self.dataset_dir}"
                )

        if self.mmap_mode and any(_is_zst(p) for p in shard_paths):
            if self._decompress_cache_dir is None:
                raise ValueError("mmap_mode is not supported for compressed shards without a tmpfs cache")
        shard_keys = [_shard_key(p) for p in shard_paths]
        if len(shard_keys) != len(set(shard_keys)):
            raise ValueError("Found both .npy and .npy.zst for the same shard")

        # Build shard info with cumulative offsets
        offset = 0
        for idx, path in enumerate(shard_paths):
            # Quick shape check without loading full array - just read header
            shape, dtype = _read_npy_header(path)
            if self._dtype is None:
                self._dtype = dtype
            elif dtype != self._dtype:
                raise ValueError(
                    f"Shard dtype mismatch: expected {self._dtype}, got {dtype} for {path}"
                )
            num_steps = shape[0]
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
        if _is_zst(shard.path) and mode is not None and self._decompress_cache_dir is not None:
            decompressed_path = self._ensure_decompressed(shard.path)
            arr = np.load(str(decompressed_path), mmap_mode=mode)
            self._acquire_lease(shard_idx, decompressed_path)
        else:
            arr = _load_npy(shard.path, mmap_mode=mode)

        # Cache if not mmap AND caching is enabled
        # (mmap arrays are already "cached" by OS, so no need to cache them)
        if not self.mmap_mode and self.cache_shards:
            self._loaded_shards[shard_idx] = arr
            self._loaded_order.append(shard_idx)
            self._evict_if_needed(keep_idx=shard_idx)
        elif self.mmap_mode and self.cache_shards:
            # Still cache mmap handles to avoid re-opening for every batch.
            self._loaded_shards[shard_idx] = arr
            self._loaded_order.append(shard_idx)
            self._evict_if_needed(keep_idx=shard_idx)

        return arr

    def unload_shard(self, shard_idx: int) -> None:
        """Release a shard from memory cache."""
        self._loaded_shards.pop(shard_idx, None)
        if shard_idx in self._loaded_order:
            self._loaded_order = [idx for idx in self._loaded_order if idx != shard_idx]
        self._release_lease(shard_idx)

    def get_rows(self, global_indices: np.ndarray) -> np.ndarray:
        """Fetch rows by global index (legacy interface for compatibility)."""
        # Sort indices by shard for efficient access
        sorted_idx = np.argsort(global_indices)
        sorted_global = global_indices[sorted_idx]

        # Find which shard each index belongs to
        shard_boundaries = np.array([s.offset for s in self.shards] + [self.total_steps])
        shard_idx = np.searchsorted(shard_boundaries[:-1], sorted_global, side='right') - 1

        # Allocate output without forcing a shard load
        dtype = self._dtype or self.load_shard(0).dtype
        out = np.empty(len(global_indices), dtype=dtype)

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

    def _ensure_decompressed(self, path: Path) -> Path:
        if self._decompress_cache_dir is None:
            raise RuntimeError("decompress cache dir not configured")
        target = self._decompress_cache_dir / _shard_key(path)
        lock_path = target.with_suffix(target.suffix + ".lock")
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with open(lock_path, "a", encoding="utf-8") as lock_handle:
            fcntl.flock(lock_handle, fcntl.LOCK_EX)
            if target.exists() and target.stat().st_size > 0:
                return target
            tmp_path = target.with_suffix(target.suffix + f".tmp.{os.getpid()}")
            dctx = zstandard.ZstdDecompressor()
            try:
                with path.open("rb") as handle:
                    with dctx.stream_reader(handle) as reader:
                        with open(tmp_path, "wb") as out_handle:
                            shutil.copyfileobj(reader, out_handle, length=16 * 1024 * 1024)
            except zstandard.ZstdError as exc:
                raise RuntimeError(f"Corrupt shard {path}: {exc}") from exc
            os.replace(tmp_path, target)
        return target

    def _acquire_lease(self, shard_idx: int, decompressed_path: Path) -> None:
        if shard_idx in self._leases:
            return
        lease_path = decompressed_path.with_suffix(decompressed_path.suffix + f".lease.{os.getpid()}")
        try:
            lease_path.touch(exist_ok=True)
        except OSError:
            return
        self._leases[shard_idx] = lease_path
        self._decompressed_paths[shard_idx] = decompressed_path

    def _release_lease(self, shard_idx: int) -> None:
        lease_path = self._leases.pop(shard_idx, None)
        decompressed_path = self._decompressed_paths.get(shard_idx)
        if lease_path is not None:
            try:
                lease_path.unlink(missing_ok=True)
            except Exception:
                pass
        if self.decompress_cleanup and decompressed_path is not None:
            try:
                self._cleanup_stale_leases(decompressed_path)
                if not list(decompressed_path.parent.glob(decompressed_path.name + ".lease.*")):
                    decompressed_path.unlink(missing_ok=True)
            except Exception:
                pass

    def _evict_if_needed(self, *, keep_idx: int) -> None:
        if not self.cache_shards:
            return
        while len(self._loaded_order) > self.cache_keep_shards:
            evict_idx = self._loaded_order[0]
            if evict_idx == keep_idx and len(self._loaded_order) > 1:
                evict_idx = self._loaded_order[1]
            self.unload_shard(evict_idx)

    def _cleanup_stale_leases(self, decompressed_path: Path) -> None:
        for lease in decompressed_path.parent.glob(decompressed_path.name + ".lease.*"):
            try:
                pid_str = lease.name.rsplit(".lease.", 1)[-1]
                pid = int(pid_str)
            except ValueError:
                continue
            if not _pid_alive(pid):
                try:
                    lease.unlink(missing_ok=True)
                except Exception:
                    pass


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


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

        # Load new shard through the shared loader to avoid duplicate copies
        arr = self.loader.load_shard(shard_idx)
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
