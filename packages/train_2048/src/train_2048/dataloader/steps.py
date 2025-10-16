from __future__ import annotations

from math import ceil
from pathlib import Path
from typing import Callable, Optional, Sequence, Tuple

import numpy as np
import sqlite3
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from .collate import (
    make_collate_macroxue,
    make_collate_steps,
)

from ..binning import Binner
from ..tokenization.macroxue import MacroxueTokenizerSpec
from ..tokenization.base import BoardCodec


class StepsDataset(Dataset):
    """Map-style dataset over steps.npy returning global indices.

    The dataset lazily loads shards as memmap arrays to avoid copying the
    entire dataset into RAM. It exposes ``get_rows`` to fetch rows given
    global indices. When ``indices`` is None, iterates all rows.
    """

    def __init__(
        self,
        dataset_dir: str,
        indices: Optional[np.ndarray] = None,
        mmap_mode: bool = False,
    ) -> None:
        import numpy as _np

        self.dataset_dir = str(dataset_dir)
        root = Path(self.dataset_dir)
        shard_paths = sorted(root.glob("steps-*.npy"))

        # Load shards lazily (memmap when requested). Build cumulative counts.
        if shard_paths:
            self.shards = [
                _np.load(str(p), mmap_mode="r" if mmap_mode else None)
                for p in shard_paths
            ]
            self.cum_counts = np.cumsum([0] + [s.shape[0] for s in self.shards])
        else:
            steps_path = root / "steps.npy"
            if not steps_path.is_file():
                raise FileNotFoundError(f"Missing steps.npy or steps-*.npy at {root}")
            self.shards = [_np.load(str(steps_path), mmap_mode="r" if mmap_mode else None)]
            self.cum_counts = np.array([0, self.shards[0].shape[0]])

        # Resolve the indices to iterate over. Avoid constructing a giant
        # arange by default; None sentinel means "all rows".
        if indices is None:
            self.indices = None  # type: Optional[np.ndarray]
        else:
            self.indices = _np.asarray(indices, dtype=_np.int64)

        total_rows = int(self.cum_counts[-1])
        print(
            f"[data] StepsDataset: {total_rows} rows across {len(self.shards)} shard(s) "
            f"from {root} (mmap={'on' if mmap_mode else 'off'})"
        )

    def __len__(self) -> int:  # type: ignore[override]
        # Report full length when indices=None so Samplers/DataLoader can size
        if self.indices is None:
            return int(self.cum_counts[-1])
        return int(self.indices.size)

    def __getitem__(self, idx: int) -> int:  # type: ignore[override]
        # Return the global step index; slicing happens in collate
        if self.indices is None:
            return int(idx)
        return int(self.indices[int(idx)])

    # ------------------------------------------------------------------
    # Helper to fetch a batch of rows from the lazy shards.
    # ------------------------------------------------------------------
    def get_rows(self, global_indices: np.ndarray) -> np.ndarray:
        """Return a structured array of rows for the given global indices."""
        import numpy as _np

        # Sort indices to read from shards sequentially. Use a stable indirect
        # sort that operates on int64 and avoid duplicating large arrays.
        global_indices = _np.asarray(global_indices, dtype=_np.int64)
        sorter = _np.argsort(global_indices, kind="mergesort")
        sorted_global_indices = global_indices[sorter]

        # Determine which shard each index belongs to
        shard_idx = _np.searchsorted(self.cum_counts, sorted_global_indices, side="right") - 1
        local_idx = sorted_global_indices - self.cum_counts[shard_idx]

        # Gather rows from each shard
        # Allocate output array once and fill per shard slice to avoid multiple
        # large temporaries and concatenations.
        out = _np.empty((sorted_global_indices.shape[0],), dtype=self.shards[0].dtype)
        write_pos = 0
        for s in _np.unique(shard_idx):
            mask = shard_idx == s
            cnt = int(mask.sum())
            if cnt == 0:
                continue
            out[write_pos : write_pos + cnt] = self.shards[s][local_idx[mask]]
            write_pos += cnt
        sorted_rows = out

        # Unsort the rows to match original order of global_indices
        unsort_sorter = _np.empty_like(sorter)
        unsort_sorter[sorter] = _np.arange(global_indices.size)
        return sorted_rows[unsort_sorter]

    def get_run_ids(self) -> np.ndarray:
        """Return an array of run_id values for all steps (shard-wise)."""
        import numpy as _np
        return _np.concatenate([s["run_id"] for s in self.shards])


class StreamingRandomSampler(Sampler[int]):
    """Sample random indices with replacement to avoid giant permutations.

    Yields exactly ``total_samples`` indices per epoch, uniformly from
    ``[0, dataset_len)``. This avoids allocating ``randperm(N)`` for large N.
    """

    def __init__(self, dataset_len: int, total_samples: int, seed: int = 42, skip: int = 0) -> None:
        if dataset_len <= 0 or total_samples <= 0:
            raise ValueError("dataset_len and total_samples must be > 0")
        self.dataset_len = int(dataset_len)
        self.total_samples = int(total_samples)
        self.seed = int(seed)
        self.skip = max(0, int(skip))
        if self.skip > self.total_samples:
            self.skip = self.skip % self.total_samples if self.total_samples > 0 else 0
        self._skip_consumed = False

    def __iter__(self):
        rng = np.random.default_rng(self.seed)
        skip_remaining = 0 if self._skip_consumed else self.skip
        self._skip_consumed = True
        # Use int64 for indexing; DataLoader will cast as needed
        emitted = 0
        while emitted < self.total_samples:
            sample = int(rng.integers(0, self.dataset_len))
            if skip_remaining > 0:
                skip_remaining -= 1
                emitted += 1
                continue
            yield sample
            emitted += 1

    def __len__(self) -> int:
        return max(0, self.total_samples - min(self.skip, self.total_samples))


class BufferedShuffleSampler(Sampler[int]):
    """One-pass shuffle with bounded memory (no full randperm).

    Maintains a ring buffer of ``buffer_size`` indices. Yields a uniform
    permutation without materializing the full list in memory.
    """

    def __init__(self, dataset_len: int, buffer_size: int = 1_000_000, seed: int = 42) -> None:
        if dataset_len <= 0:
            raise ValueError("dataset_len must be > 0")
        self.dataset_len = int(dataset_len)
        self.buffer_size = max(1, int(min(buffer_size, dataset_len)))
        self.seed = int(seed)

    def __iter__(self):
        rng = np.random.default_rng(self.seed)
        B = self.buffer_size
        # Initialize buffer with first B indices
        buffer = np.arange(B, dtype=np.int64)
        next_idx = B
        while next_idx < self.dataset_len:
            j = int(rng.integers(0, B))
            yield int(buffer[j])
            buffer[j] = next_idx
            next_idx += 1
        # Emit the remaining buffer in random order
        rng.shuffle(buffer)
        for idx in buffer.tolist():
            yield int(idx)

    def __len__(self) -> int:
        return self.dataset_len


class FilteredBufferedShuffleSampler(Sampler[int]):
    """Shuffle indices with a ring buffer, keeping only eligible step_index.

    Iterates the dataset once per epoch, streaming through shards and
    enqueuing indices whose ``step_index`` falls within the inclusive window
    [step_index_min, step_index_max].
    """

    def __init__(
        self,
        dataset: "StepsDataset",
        *,
        buffer_size: int = 1_000_000,
        seed: int = 42,
        step_index_min: Optional[int] = None,
        step_index_max: Optional[int] = None,
        chunk_size: int = 1_000_000,
    ) -> None:
        if buffer_size <= 0:
            raise ValueError("buffer_size must be > 0")
        self.dataset = dataset
        self.buffer_size = int(min(max(1, buffer_size), max(1, len(dataset))))
        self.seed = int(seed)
        self.smin = step_index_min
        self.smax = step_index_max
        self.chunk_size = int(max(1, chunk_size))
        # Cache eligible count lazily
        self._eligible_count: Optional[int] = None

    def _eligible_index_generator(self):
        import numpy as _np
        ds = self.dataset
        for s_idx, shard in enumerate(ds.shards):
            offset = int(ds.cum_counts[s_idx])
            N = shard.shape[0]
            # Iterate in chunks to cap memory bandwidth
            for lo in range(0, N, self.chunk_size):
                hi = min(N, lo + self.chunk_size)
                step_idx = shard['step_index'][lo:hi].astype(_np.int64, copy=False)
                keep = _np.ones(step_idx.shape[0], dtype=bool)
                if self.smin is not None:
                    keep &= step_idx >= int(self.smin)
                if self.smax is not None:
                    keep &= step_idx <= int(self.smax)
                if not keep.any():
                    continue
                local = _np.flatnonzero(keep)
                global_idx = local + (offset + lo)
                for g in global_idx.tolist():
                    yield int(g)

    def __iter__(self):
        import numpy as _np
        rng = _np.random.default_rng(self.seed)
        gen = self._eligible_index_generator()
        B = self.buffer_size
        # Fill buffer
        buffer = []
        try:
            for _ in range(B):
                buffer.append(next(gen))
        except StopIteration:
            pass
        if not buffer:
            return iter(())
        buffer = _np.asarray(buffer, dtype=_np.int64)
        # Replace-yield loop
        for nxt in gen:
            j = int(rng.integers(0, buffer.shape[0]))
            yield int(buffer[j])
            buffer[j] = int(nxt)
        # Drain remaining in random order
        rng.shuffle(buffer)
        for g in buffer.tolist():
            yield int(g)

    def __len__(self) -> int:
        if self._eligible_count is not None:
            return int(self._eligible_count)
        # Compute and cache
        total = 0
        for s_idx, shard in enumerate(self.dataset.shards):
            step_idx = shard['step_index']
            if self.smin is not None and self.smax is not None:
                mask = (step_idx >= int(self.smin)) & (step_idx <= int(self.smax))
                total += int(np.count_nonzero(mask))
            elif self.smin is not None:
                total += int(np.count_nonzero(step_idx >= int(self.smin)))
            elif self.smax is not None:
                total += int(np.count_nonzero(step_idx <= int(self.smax)))
            else:
                total += int(step_idx.shape[0])
        self._eligible_count = int(total)
        return int(total)


class FilteredStreamingSampler(Sampler[int]):
    """Random sampling with replacement over eligible indices.

    Maintains a ring buffer of eligible indices sourced by scanning shards
    with a step_index window, and yields exactly ``total_samples`` draws.
    """

    def __init__(
        self,
        dataset: "StepsDataset",
        total_samples: int,
        *,
        seed: int = 42,
        step_index_min: Optional[int] = None,
        step_index_max: Optional[int] = None,
        buffer_size: int = 1_000_000,
        chunk_size: int = 1_000_000,
    ) -> None:
        if total_samples <= 0:
            raise ValueError("total_samples must be > 0")
        self.dataset = dataset
        self.total_samples = int(total_samples)
        self.seed = int(seed)
        self.smin = step_index_min
        self.smax = step_index_max
        self.buffer_size = int(max(1, buffer_size))
        self.chunk_size = int(max(1, chunk_size))

    def _eligible_index_generator(self):
        import numpy as _np
        ds = self.dataset
        for s_idx, shard in enumerate(ds.shards):
            offset = int(ds.cum_counts[s_idx])
            N = shard.shape[0]
            for lo in range(0, N, self.chunk_size):
                hi = min(N, lo + self.chunk_size)
                step_idx = shard['step_index'][lo:hi].astype(_np.int64, copy=False)
                keep = _np.ones(step_idx.shape[0], dtype=bool)
                if self.smin is not None:
                    keep &= step_idx >= int(self.smin)
                if self.smax is not None:
                    keep &= step_idx <= int(self.smax)
                if not keep.any():
                    continue
                local = _np.flatnonzero(keep)
                global_idx = local + (offset + lo)
                for g in global_idx.tolist():
                    yield int(g)

    def __iter__(self):
        import numpy as _np
        rng = _np.random.default_rng(self.seed)
        gen = self._eligible_index_generator()
        # Fill buffer
        buf_list = []
        try:
            for _ in range(self.buffer_size):
                buf_list.append(next(gen))
        except StopIteration:
            pass
        if not buf_list:
            return iter(())
        buffer = _np.asarray(buf_list, dtype=_np.int64)
        # Replacement sampling with ongoing refresh
        for _ in range(self.total_samples):
            j = int(rng.integers(0, buffer.shape[0]))
            yield int(buffer[j])
            try:
                nxt = next(gen)
                buffer[j] = int(nxt)
            except StopIteration:
                # No more new eligible indices; keep sampling from buffer
                pass

    def __len__(self) -> int:
        return self.total_samples


def _select_run_ids(
    conn: sqlite3.Connection,
    dataset: StepsDataset,
    *,
    run_sql: Optional[str],
    sql_params: Sequence | None,
    val_run_sql: Optional[str],
    val_sql_params: Sequence | None,
    val_run_pct: float,
    val_split_seed: int,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Return (train_run_ids, val_run_ids_or_None) as numpy arrays.

    The function now operates on a :class:`StepsDataset` instance instead of a
    pre‑loaded ``steps`` array.  It uses :meth:`StepsDataset.get_run_ids` to
    obtain the universe of run IDs without materialising the full dataset.
    """

    # Determine dtype for run_id column without loading whole column
    # Inspect dtype from the first shard
    try:
        run_dtype = dataset.shards[0]['run_id'].dtype
    except Exception:
        run_dtype = np.uint32

    def _fetch_ids(sql: str, params: Sequence | None) -> np.ndarray:
        rows = conn.execute(sql, tuple(params or ())).fetchall()
        return np.asarray([r[0] for r in rows], dtype=run_dtype)

    # Universe of runs
    if run_sql:
        uni = _fetch_ids(run_sql, sql_params)
    else:
        # Fallback: infer from metadata DB to avoid scanning steps.npy
        rows = conn.execute("SELECT id FROM runs").fetchall()
        uni = np.asarray([r[0] for r in rows], dtype=run_dtype)

    if val_run_sql:
        val_ids = _fetch_ids(val_run_sql, val_sql_params)
        # Intersect to ensure val is within universe
        val_ids = np.intersect1d(val_ids, uni, assume_unique=False)
        train_ids = np.setdiff1d(uni, val_ids, assume_unique=False)
        return train_ids, val_ids

    if val_run_pct > 0.0:
        uni_unique = np.unique(uni)
        rng = np.random.default_rng(int(val_split_seed))
        n_val = max(1, int(np.ceil(val_run_pct * len(uni_unique))))
        perm = rng.permutation(len(uni_unique))
        val_sel = np.sort(perm[:n_val])
        val_ids = uni_unique[val_sel]
        train_ids = np.setdiff1d(uni_unique, val_ids, assume_unique=True)
        return train_ids, val_ids

    return uni, None


def _sum_steps_for_ids(conn: sqlite3.Connection, run_ids: Optional[np.ndarray]) -> int:
    """Return sum of step counts for the given run ids using metadata.db only.

    Back-compat note: historical schemas used different column names:
      - steps (new)
      - num_steps (older)
      - num_moves (legacy)
    We probe in that order.
    """
    rows: list[tuple[int, int]] = []
    for col in ("steps", "num_steps", "num_moves"):
        try:
            cur = conn.execute(f"SELECT id, {col} FROM runs")
            rows = [(int(rid), int(cnt or 0)) for (rid, cnt) in cur]
            if rows:
                break
        except Exception:
            continue
    if not rows:
        return 0
    if run_ids is None:
        return int(sum(cnt for (_id, cnt) in rows))
    sel = set(int(x) for x in run_ids.tolist())
    return int(sum(cnt for (_id, cnt) in rows if _id in sel))


def _indices_from_run_ids(dataset: StepsDataset, run_ids: np.ndarray) -> np.ndarray:
    """Return global indices for steps belonging to the given run_ids.

    Uses vectorized np.isin per shard (C impl) to avoid Python-level loops.
    Processes shards incrementally to cap peak memory.
    """
    if run_ids is None or run_ids.size == 0:
        return np.empty((0,), dtype=np.int64)
    run_ids = np.asarray(run_ids)
    total_rows = int(dataset.cum_counts[-1])
    out_dtype = np.int32 if total_rows <= np.iinfo(np.int32).max else np.int64
    parts: list[np.ndarray] = []
    for s_idx, shard in enumerate(dataset.shards):
        offset = int(dataset.cum_counts[s_idx])
        rids = shard['run_id']  # memmap view
        mask = np.isin(rids, run_ids, assume_unique=False)
        if not mask.any():
            continue
        local_idx = np.flatnonzero(mask)
        global_idx = (local_idx + offset).astype(out_dtype, copy=False)
        parts.append(global_idx)
    if not parts:
        return np.empty((0,), dtype=np.int64)
    out = np.concatenate(parts)
    return out.astype(np.int64, copy=False)


def _indices_excluding_run_ids(dataset: StepsDataset, exclude_run_ids: np.ndarray) -> np.ndarray:
    """Return global indices for steps whose run_id is NOT in exclude_run_ids.

    Vectorized per shard for speed; avoids a Python loop per element.
    """
    exclude_run_ids = np.asarray(exclude_run_ids) if exclude_run_ids is not None else None
    total_rows = int(dataset.cum_counts[-1])
    out_dtype = np.int32 if total_rows <= np.iinfo(np.int32).max else np.int64
    parts: list[np.ndarray] = []
    for s_idx, shard in enumerate(dataset.shards):
        offset = int(dataset.cum_counts[s_idx])
        rids = shard['run_id']
        if exclude_run_ids is None or exclude_run_ids.size == 0:
            local_idx = np.arange(rids.shape[0], dtype=out_dtype)
        else:
            mask = ~np.isin(rids, exclude_run_ids, assume_unique=False)
            if not mask.any():
                continue
            local_idx = np.flatnonzero(mask).astype(out_dtype, copy=False)
        parts.append(local_idx + offset)
    if not parts:
        return np.empty((0,), dtype=np.int64)
    out = np.concatenate(parts)
    return out.astype(np.int64, copy=False)





def build_steps_dataloaders(
    dataset_dir: str,
    binner: Optional[Binner],
    target_mode: str,
    batch_size: int,
    *,
    physical_batch_size: Optional[int] = None,
    tokenizer_path: Optional[str] = None,
    ev_tokenizer: Optional[object] = None,
    train_num_steps: Optional[int] = None,
    seed: int = 42,
    shuffle: bool = False,
    shuffle_buffer_size: int = 1_000_000,
    val_num_steps: Optional[int] = None,
    val_steps_pct: float = 0.0,
    run_sql: Optional[str] = None,
    sql_params: Sequence | None = None,
    val_run_sql: Optional[str] = None,
    val_sql_params: Sequence | None = None,
    val_run_pct: float = 0.0,
    val_split_seed: int = 42,
    num_workers_train: int = 12,
    mmap_mode: bool = False,
    step_index_min: Optional[int] = None,
    step_index_max: Optional[int] = None,
) -> Tuple[DataLoader, Optional[DataLoader], int]:
    """Create train/val DataLoaders from steps.npy + metadata.db.

    Returns (dl_train, dl_val_or_None, per_epoch_steps).
    """

    meta_path = Path(dataset_dir) / "metadata.db"
    if not meta_path.is_file():
        raise FileNotFoundError(f"Missing metadata.db at {meta_path}")

    ds_train = StepsDataset(dataset_dir, mmap_mode=mmap_mode)
    conn = sqlite3.connect(str(meta_path))
    try:
        train_rids, val_rids = _select_run_ids(
            conn,
            ds_train,
            run_sql=run_sql,
            sql_params=sql_params or (),
            val_run_sql=val_run_sql,
            val_sql_params=val_sql_params or (),
            val_run_pct=float(val_run_pct or 0.0),
            val_split_seed=int(val_split_seed or 42),
        )
        meta_train_steps = _sum_steps_for_ids(conn, train_rids)
        meta_val_steps = _sum_steps_for_ids(conn, val_rids) if val_rids is not None else 0
    finally:
        conn.close()

    print(f"[data] Selected runs: train={train_rids.size} val={(0 if val_rids is None else val_rids.size)}")
    # If no explicit validation split and train==all runs, avoid materialising indices
    # and iterate the full dataset directly.
    if val_rids is None and run_sql is None and (val_run_pct == 0.0) and (val_run_sql is None):
        train_indices = None
    else:
        train_indices = _indices_from_run_ids(ds_train, train_rids)
    val_indices = _indices_from_run_ids(ds_train, val_rids) if val_rids is not None else None

    # Optional: apply step-index window filtering (inclusive) if provided in config
    if (step_index_min is not None) or (step_index_max is not None):
        import numpy as _np
        smin_i = int(step_index_min) if step_index_min is not None else None
        smax_i = int(step_index_max) if step_index_max is not None else None
        def _filter_by_step_index(idx: _np.ndarray) -> _np.ndarray:
            if idx is None or idx.size == 0:
                return idx
            parts: list[_np.ndarray] = []
            for s_idx, shard in enumerate(ds_train.shards):
                offset = int(ds_train.cum_counts[s_idx])
                lo = offset
                hi = int(ds_train.cum_counts[s_idx + 1])
                mask = (idx >= lo) & (idx < hi)
                if not mask.any():
                    continue
                local = (idx[mask] - offset).astype(_np.int64, copy=False)
                step_idx = shard['step_index'][local].astype(_np.int64, copy=False)
                keep = _np.ones_like(local, dtype=bool)
                if smin_i is not None:
                    keep &= step_idx >= smin_i
                if smax_i is not None:
                    keep &= step_idx <= smax_i
                parts.append((local[keep] + offset).astype(_np.int64, copy=False))
            if not parts:
                return _np.empty((0,), dtype=_np.int64)
            return _np.concatenate(parts)
        if train_indices is not None:
            train_indices = _filter_by_step_index(train_indices)
        if val_indices is not None:
            val_indices = _filter_by_step_index(val_indices)

    ds_train.indices = train_indices
    if target_mode == "macroxue_tokens":
        if tokenizer_path is None:
            raise ValueError("tokenizer_path is required for macroxue_tokens mode")
        collate = make_collate_macroxue(ds_train, tokenizer_path)
    else:
        collate = make_collate_steps(target_mode, ds_train, binner, ev_tokenizer=ev_tokenizer)

    effective_batch_size = int(batch_size)
    loader_batch_size = int(physical_batch_size or batch_size)
    if effective_batch_size <= 0 or loader_batch_size <= 0:
        raise ValueError("batch sizes must be > 0")

    print(
        f"[data] Building DataLoader(train): workers={num_workers_train} batch_size={loader_batch_size}"
        f" (effective={effective_batch_size})"
    )
    prefetch_train = 8 if num_workers_train > 0 else None
    train_sampler = None
    is_streaming = False
    if train_num_steps is not None and int(train_num_steps) > 0:
        total_len = len(ds_train)
        total_samples = int(train_num_steps) * effective_batch_size
        if step_index_min is not None or step_index_max is not None:
            print(
                f"[data] Using filtered streaming sampler: samples={total_samples} buffer={shuffle_buffer_size} step_index in [{step_index_min},{step_index_max}]"
            )
            train_sampler = FilteredStreamingSampler(
                ds_train,
                total_samples,
                seed=seed,
                step_index_min=step_index_min,
                step_index_max=step_index_max,
                buffer_size=int(shuffle_buffer_size),
            )
        else:
            print(f"[data] Using streaming sampler: total_samples={total_samples} over dataset_len={total_len}")
            train_sampler = StreamingRandomSampler(total_len, total_samples, seed=seed)
        is_streaming = True
    elif train_indices is None and (step_index_min is not None or step_index_max is not None):
        print(
            f"[data] Using filtered buffered shuffle: buffer={shuffle_buffer_size} step_index in [{step_index_min},{step_index_max}]"
        )
        train_sampler = FilteredBufferedShuffleSampler(
            ds_train,
            buffer_size=int(shuffle_buffer_size),
            seed=seed,
            step_index_min=step_index_min,
            step_index_max=step_index_max,
        )
    elif train_indices is None:
        if shuffle:
            total_len = len(ds_train)
            print(f"[data] Using buffered shuffle: dataset_len={total_len} buffer_size={shuffle_buffer_size}")
            train_sampler = BufferedShuffleSampler(total_len, buffer_size=int(shuffle_buffer_size), seed=seed)
        else:
            train_sampler = None
    dl_train = DataLoader(
        ds_train,
        batch_size=loader_batch_size,
        shuffle=False,
        sampler=train_sampler,
        num_workers=num_workers_train,
        collate_fn=collate,
        pin_memory=True,
        persistent_workers=True if num_workers_train > 0 else False,
        prefetch_factor=prefetch_train,
    )

    dl_val: Optional[DataLoader] = None
    if val_indices is not None and val_indices.size > 0:
        num_workers_val = max(2, num_workers_train // 3)
        print(
            f"[data] Building DataLoader(val): workers={num_workers_val} batch_size={loader_batch_size}"
        )
        prefetch_val = 4 if num_workers_val > 0 else None
        ds_val = StepsDataset(dataset_dir, indices=val_indices, mmap_mode=mmap_mode)
        if target_mode == "macroxue_tokens":
            if tokenizer_path is None:
                raise ValueError("tokenizer_path is required for macroxue_tokens mode")
            collate_v = make_collate_macroxue(ds_val, tokenizer_path)
        else:
            collate_v = make_collate_steps(target_mode, ds_val, binner, ev_tokenizer=ev_tokenizer)
        # Optionally cap validation steps via a sampler
        val_sampler = None
        max_val_steps = None
        if val_num_steps is not None and int(val_num_steps) > 0:
            max_val_steps = int(val_num_steps)
        elif val_steps_pct and val_steps_pct > 0.0:
            # Derive from actual planned train steps this epoch
            planned_train_steps = int(train_num_steps) if (train_num_steps is not None and int(train_num_steps) > 0) else int(ceil(int(meta_train_steps) / max(1, int(batch_size))))
            max_val_steps = int(max(1, round(float(planned_train_steps) * float(val_steps_pct))))
        if max_val_steps is not None:
            total_val = len(ds_val)
            total_samples = int(max_val_steps) * loader_batch_size
            print(f"[data] Capping validation: steps={max_val_steps} samples={total_samples} (of {total_val})")
            val_sampler = StreamingRandomSampler(total_val, total_samples, seed=seed + 1)
        dl_val = DataLoader(
            ds_val,
            batch_size=loader_batch_size,
            shuffle=(val_sampler is None),
            sampler=val_sampler,
            num_workers=num_workers_val,
            collate_fn=collate_v,
            pin_memory=True,
            persistent_workers=True if num_workers_val > 0 else False,
            prefetch_factor=prefetch_val,
        )

    if is_streaming:
        per_epoch_steps = int(train_num_steps)
    else:
        # Prefer metadata count to avoid scanning steps.npy
        per_epoch_steps = ceil(int(meta_train_steps) / max(1, effective_batch_size))
    print(
        f"[data] Epoch steps (train/val): {per_epoch_steps} (meta rows: train={meta_train_steps} val={meta_val_steps})"
    )
    return dl_train, dl_val, per_epoch_steps


__all__ = [
    "StepsDataset",
    "make_collate_steps",
    "build_steps_dataloaders",
]
