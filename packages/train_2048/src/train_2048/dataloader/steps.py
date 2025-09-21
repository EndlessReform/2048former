from __future__ import annotations

from math import ceil
from pathlib import Path
from typing import Callable, Optional, Sequence, Tuple

import numpy as np
import sqlite3
import torch
from torch.utils.data import Dataset, DataLoader, Sampler

from ..binning import Binner
from ..tokenization.macroxue import MacroxueTokenizerSpec


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

    def __init__(self, dataset_len: int, total_samples: int, seed: int = 42) -> None:
        if dataset_len <= 0 or total_samples <= 0:
            raise ValueError("dataset_len and total_samples must be > 0")
        self.dataset_len = int(dataset_len)
        self.total_samples = int(total_samples)
        self.seed = int(seed)

    def __iter__(self):
        rng = np.random.default_rng(self.seed)
        # Use int64 for indexing; DataLoader will cast as needed
        for _ in range(self.total_samples):
            yield int(rng.integers(0, self.dataset_len))

    def __len__(self) -> int:
        return self.total_samples


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

        # Resolve the indices to iterate over. Avoid constructing a giant
        # arange by default; defer to a None sentinel which means "all rows".
        if indices is None:
            self.indices = None  # meaning: 0..total_rows-1
        else:
            self.indices = _np.asarray(indices, dtype=_np.int64)

    def __len__(self) -> int:  # type: ignore[override]
        # Report full length even when indices=None so Samplers/DataLoader can size
        total = int(self.cum_counts[-1])
        if self.indices is None:
            return total
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
        """Return a structured array of rows for the given global indices.

        Parameters
        ----------
        global_indices:
            1‑D array of global step indices.
        """
        import numpy as _np

        # Sort indices to read from shards sequentially
        sorter = np.argsort(global_indices)
        sorted_global_indices = global_indices[sorter]

        # Determine which shard each index belongs to
        shard_idx = _np.searchsorted(self.cum_counts, sorted_global_indices, side="right") - 1
        local_idx = sorted_global_indices - self.cum_counts[shard_idx]

        # Gather rows from each shard
        rows = []
        for s in np.unique(shard_idx):
            mask = shard_idx == s
            rows.append(self.shards[s][local_idx[mask]])

        sorted_rows = _np.concatenate(rows)

        # Unsort the rows to match original order of global_indices
        unsort_sorter = np.empty_like(sorter)
        unsort_sorter[sorter] = np.arange(global_indices.size)

        return sorted_rows[unsort_sorter]


    def get_run_ids(self) -> np.ndarray:
        """Return an array of run_id values for all steps.

        This loads only the ``run_id`` column from each shard, avoiding a full
        materialisation of the dataset.
        """
        return np.concatenate([s['run_id'] for s in self.shards])


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
    """Return sum(steps) for the given run ids using metadata.db only.

    When ``run_ids`` is None, sums over all runs.
    """
    cur = conn.execute("SELECT id, steps FROM runs")
    if run_ids is None:
        return int(sum(int(steps or 0) for (_id, steps) in cur))
    sel = set(int(x) for x in run_ids.tolist())
    return int(sum(int(steps or 0) for (_id, steps) in cur if _id in sel))


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


def make_collate_macroxue(
    dataset: StepsDataset,
    tokenizer_path: str,
) -> Callable:
    """Collate function for macroxue tokenization.

    Returns a dictionary with winner and margin targets.
    """
    spec = MacroxueTokenizerSpec.from_json(Path(tokenizer_path))
    knots = {vt: np.array(k) for vt, k in spec.ecdf_knots.items()}
    delta_edges = np.array(spec.delta_edges)
    n_bins = len(delta_edges) - 1
    illegal_token = n_bins
    winner_token = n_bins + 1

    # Precompute valuation type mapping (dataset id -> spec index).
    # We try to read dataset_dir/valuation_types.json to align enums.
    # Fallback to identity mapping if not found.
    vt_name_to_spec_idx = {name: i for i, name in enumerate(spec.valuation_types)}
    ds_vt_mapping: Optional[dict[int, int]] = None
    try:
        import json as _json  # lazy
        vt_path = Path(dataset.dataset_dir) / "valuation_types.json"
        if vt_path.is_file():
            payload = _json.loads(vt_path.read_text())
            # Support either list[str] or dict[str,int->str]
            if isinstance(payload, list):
                ds_id_to_name = {int(i): str(name) for i, name in enumerate(payload)}
            elif isinstance(payload, dict):
                ds_id_to_name = {int(k): str(v) for k, v in payload.items()}
            else:
                raise TypeError("valuation_types.json must be list or dict")
            tmp_map: dict[int, int] = {}
            for ds_id, name in ds_id_to_name.items():
                if name not in vt_name_to_spec_idx:
                    # Spec and dataset disagree — fail early
                    raise KeyError(
                        f"Valuation type '{name}' (id {ds_id}) missing from tokenizer spec"
                    )
                tmp_map[ds_id] = vt_name_to_spec_idx[name]
            ds_vt_mapping = tmp_map
    except Exception:
        # Silent fallback; identity will be used below
        ds_vt_mapping = None

    def _unpack_board_to_exps_u8(packed: np.ndarray, *, mask65536: Optional[np.ndarray] = None) -> np.ndarray:
        # packed: (N,) uint64 with 16 packed 4-bit tiles, LSB = cell 0
        arr = packed.astype(np.uint64, copy=False)
        n = int(arr.shape[0])
        out = np.empty((n, 16), dtype=np.uint8)
        for i in range(16):
            out[:, i] = ((arr >> (4 * i)) & np.uint64(0xF)).astype(np.uint8, copy=False)
        # Correct exponents for 2**16 tiles if mask present. Keep exponent 16
        # so callers can configure the model vocab to 17 (0..16).
        if mask65536 is not None:
            m = mask65536.astype(np.uint16, copy=False)
            for i in range(16):
                sel = ((m >> i) & np.uint16(1)) != 0
                if np.any(sel):
                    out[sel, i] = np.uint8(16)
        return out

    def _collate(batch_indices: Sequence[int]):
        import numpy as _np

        idxs = _np.asarray(batch_indices, dtype=_np.int64)
        batch = dataset.get_rows(idxs)

        # Decode packed boards → (N,16) exponents without ai_2048 dependency
        if 'board' not in batch.dtype.names:
            raise KeyError("Expected 'board' field in steps.npy for macroxue dataset")
        mask65536 = batch['tile_65536_mask'] if 'tile_65536_mask' in batch.dtype.names else None
        exps = _unpack_board_to_exps_u8(batch['board'], mask65536=mask65536)
        tokens = torch.from_numpy(exps.copy()).to(dtype=torch.int64)

        branch_evs = batch["branch_evs"]
        valuation_types = batch["valuation_type"].astype(_np.int64, copy=False)
        ev_legal = batch["ev_legal"]

        percentiles = np.zeros_like(branch_evs, dtype=np.float32)

        # If dataset->spec mapping is available, remap ids; else assume identity
        if ds_vt_mapping is not None:
            vt_spec_ids = _np.empty_like(valuation_types)
            if valuation_types.size:
                uniq = _np.unique(valuation_types)
                for ds_id in uniq.tolist():
                    if ds_id not in ds_vt_mapping:
                        raise KeyError(f"Dataset valuation_type id {ds_id} missing from mapping")
                    vt_spec_ids[valuation_types == ds_id] = int(ds_vt_mapping[ds_id])
        else:
            vt_spec_ids = valuation_types  # assume consistent enum ordering

        for vt_name, vt_id in vt_name_to_spec_idx.items():
            mask = vt_spec_ids == vt_id
            if not np.any(mask):
                continue

            vt_knots = knots[vt_name]
            evs = branch_evs[mask]

            # Vectorized replication of tokenizer._percentile_from_knots
            idx = np.searchsorted(vt_knots, evs, side="right")  # in [0, len]
            n = len(vt_knots) - 1
            p = np.zeros_like(evs, dtype=np.float32)
            # In-range where 0 < idx < len(knots)
            valid = (idx > 0) & (idx < len(vt_knots))
            if np.any(valid):
                lo = vt_knots[idx[valid] - 1]
                hi = vt_knots[idx[valid]]
                ratio = np.divide(
                    evs[valid] - lo,
                    hi - lo,
                    out=np.zeros_like(evs[valid]),
                    where=(hi > lo),
                )
                p[valid] = (idx[valid] - 1 + ratio) / n
            # Below/above bounds
            p[evs <= vt_knots[0]] = 0.0
            p[evs >= vt_knots[-1]] = 1.0
            percentiles[mask] = np.clip(p, 0.0, 1.0)

        legal_mask = np.stack([
            (ev_legal & 1) != 0,
            (ev_legal & 8) != 0, # right
            (ev_legal & 2) != 0, # down
            (ev_legal & 4) != 0, # left
        ], axis=1)

        # Mark illegal as -inf so they never win
        percentiles = percentiles.astype(np.float32, copy=False)
        percentiles[~legal_mask] = -np.inf

        winner_indices = np.argmax(percentiles, axis=1)
        winner_percentiles = np.max(percentiles, axis=1, keepdims=True)

        # For illegal moves, set delta to 1 (max bin) so they map to ILLEGAL later
        deltas = np.clip(winner_percentiles - percentiles, 0, 1)
        # Digitize using right-inclusive edges, consistent with tokenizer._digitize
        # idx = bisect_right(edges, delta) - 1, then clamp to [0, len(edges)-2]
        margin_bins = np.searchsorted(delta_edges, deltas, side="right") - 1
        margin_bins = np.clip(margin_bins, 0, n_bins - 1)

        targets = margin_bins.astype(np.int64, copy=False)
        # Assign ILLEGAL class for illegal actions
        targets[~legal_mask] = illegal_token
        # Winner action gets WINNER class
        rows = np.arange(len(idxs), dtype=np.int64)
        targets[rows, winner_indices] = winner_token

        return {
            "tokens": tokens,
            "targets": torch.from_numpy(targets.copy()).long(),
            "n_classes": n_bins + 2,
        }

    return _collate


def make_collate_steps(
    target_mode: str,
    dataset: StepsDataset,
    binner: Optional[Binner],
) -> Callable:
    """Collate function that gathers tokens/mask/values and builds training targets."""

    if target_mode not in {"binned_ev", "hard_move"}:
        raise ValueError(f"Unknown target mode: {target_mode}")
    if target_mode == "binned_ev" and binner is None:
        raise ValueError("Binner is required when target_mode='binned_ev'")

    import ai_2048 as a2  # lazy import

    def _collate(batch_indices: Sequence[int]):
        import numpy as _np

        idxs = _np.asarray(batch_indices, dtype=_np.int64)
        exps_buf, dirs, evs = a2.batch_from_steps(dataset.get_rows, idxs, parallel=True)

        # Tokens from exponents buffer
        exps = _np.frombuffer(exps_buf, dtype=_np.uint8).reshape(-1, 16)
        tokens = torch.from_numpy(exps.copy()).to(dtype=torch.int64)

        # EVs and legality mask from ev_legal bitfield (Up=1, Right=8, Down=2, Left=4)
        if not isinstance(evs, _np.ndarray):
            evs = _np.asarray(evs, dtype=_np.float32)
        evs = evs.astype(_np.float32, copy=False)

        batch = dataset.get_rows(idxs)
        if 'ev_legal' in batch.dtype.names:
            bits = batch['ev_legal'].astype(_np.uint8, copy=False)
            legal = _np.stack([
                (bits & 1) != 0,   # Up -> col 0
                (bits & 8) != 0,   # Right -> col 1
                (bits & 2) != 0,   # Down -> col 2
                (bits & 4) != 0,   # Left -> col 3
            ], axis=1)
        else:
            # Fallback: treat finite EVs as legal if mask not present
            legal = _np.isfinite(evs)

        evs_clean = (evs * legal.astype(_np.float32, copy=False)).astype(_np.float32, copy=False)

        branch_values = torch.from_numpy(evs_clean.copy())  # (N,4) float32
        branch_mask = torch.from_numpy(legal.astype(_np.bool_, copy=False))  # (N,4) bool

        out = {
            "tokens": tokens,
            "branch_mask": branch_mask,
            "branch_values": branch_values,
        }
        if target_mode == "binned_ev":
            assert binner is not None  # for type checkers
            binner.to_device(branch_values.device)
            out["branch_bin_targets"] = binner.bin_values(branch_values).long()
            out["n_bins"] = int(binner.n_bins)
        else:  # hard_move
            dirs_arr = _np.asarray(dirs, dtype=_np.int64)
            out["move_targets"] = torch.from_numpy(dirs_arr.copy()).to(dtype=torch.long)
        return out

    return _collate


def build_steps_dataloaders(
    dataset_dir: str,
    binner: Optional[Binner],
    target_mode: str,
    batch_size: int,
    *,
    tokenizer_path: Optional[str] = None,
    train_num_steps: Optional[int] = None,
    seed: int = 42,
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

    ds_train.indices = train_indices
    if target_mode == "macroxue_tokens":
        if tokenizer_path is None:
            raise ValueError("tokenizer_path is required for macroxue_tokens mode")
        collate = make_collate_macroxue(ds_train, tokenizer_path)
    else:
        collate = make_collate_steps(target_mode, ds_train, binner)

    print(f"[data] Building DataLoader(train): workers={num_workers_train} batch_size={batch_size}")
    prefetch_train = 8 if num_workers_train > 0 else None
    train_sampler = None
    if train_num_steps is not None and int(train_num_steps) > 0:
        total_len = len(ds_train)
        total_samples = int(train_num_steps) * int(batch_size)
        print(f"[data] Using streaming sampler: total_samples={total_samples} over dataset_len={total_len}")
        train_sampler = StreamingRandomSampler(total_len, total_samples, seed=seed)
    elif train_indices is None:
        # Full-dataset epoch, avoid massive randperm by using a bounded-memory shuffler
        total_len = len(ds_train)
        print(f"[data] Using buffered shuffle: dataset_len={total_len} buffer_size={shuffle_buffer_size}")
        train_sampler = BufferedShuffleSampler(total_len, buffer_size=int(shuffle_buffer_size), seed=seed)
    dl_train = DataLoader(
        ds_train,
        batch_size=batch_size,
        shuffle=False,
        sampler=train_sampler,
        num_workers=num_workers_train,
        collate_fn=collate,
        pin_memory=True,
        persistent_workers=True if num_workers_train > 0 else False,
        prefetch_factor=prefetch_train if prefetch_train is not None else 2,
    )

    dl_val: Optional[DataLoader] = None
    if val_indices is not None and val_indices.size > 0:
        num_workers_val = max(2, num_workers_train // 3)
        print(f"[data] Building DataLoader(val): workers={num_workers_val} batch_size={batch_size}")
        prefetch_val = 4 if num_workers_val > 0 else None
        ds_val = StepsDataset(dataset_dir, indices=val_indices, mmap_mode=mmap_mode)
        if target_mode == "macroxue_tokens":
            if tokenizer_path is None:
                raise ValueError("tokenizer_path is required for macroxue_tokens mode")
            collate_v = make_collate_macroxue(ds_val, tokenizer_path)
        else:
            collate_v = make_collate_steps(target_mode, ds_val, binner)
        # Optionally cap validation steps via a sampler
        val_sampler = None
        max_val_steps = None
        if val_num_steps is not None and int(val_num_steps) > 0:
            max_val_steps = int(val_num_steps)
        elif val_steps_pct and val_steps_pct > 0.0 and train_num_steps:
            max_val_steps = int(max(1, round(float(train_num_steps) * float(val_steps_pct))))
        if max_val_steps is not None:
            total_val = len(ds_val)
            total_samples = int(max_val_steps) * int(batch_size)
            print(f"[data] Capping validation: steps={max_val_steps} samples={total_samples} (of {total_val})")
            val_sampler = StreamingRandomSampler(total_val, total_samples, seed=seed + 1)
        dl_val = DataLoader(
            ds_val,
            batch_size=batch_size,
            shuffle=(val_sampler is None),
            sampler=val_sampler,
            num_workers=num_workers_val,
            collate_fn=collate_v,
            pin_memory=True,
            persistent_workers=True if num_workers_val > 0 else False,
            prefetch_factor=prefetch_val if prefetch_val is not None else 2,
        )

    if train_sampler is not None:
        per_epoch_steps = int(train_num_steps)
    else:
        # Prefer metadata count to avoid scanning steps.npy
        per_epoch_steps = ceil(int(meta_train_steps) / max(1, int(batch_size)))
    print(
        f"[data] Epoch steps (train/val): {per_epoch_steps} (meta rows: train={meta_train_steps} val={meta_val_steps})"
    )
    return dl_train, dl_val, per_epoch_steps


__all__ = [
    "StepsDataset",
    "make_collate_steps",
    "build_steps_dataloaders",
]
