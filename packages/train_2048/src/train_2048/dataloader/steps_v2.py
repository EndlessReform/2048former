"""Streamlined dataloader builder using shard-based loading.

Key improvements:
- No index materialization for billions of rows
- No full dataset scans
- Trust metadata DB completely
- Load shards sequentially into RAM, sample randomly
"""
from __future__ import annotations

from math import ceil
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence, Tuple
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from .shard_loader import ShardLoader, _read_npy_header
from .dataset import SampleRef, ShardDataset
from .metadata import MetadataDB
from .samplers import ShardPoolSampler, BufferedShuffleSampler, SequentialSampler
from .collate import make_collate_macroxue, make_collate_steps

from ..tokenization.base import BoardCodec


class _ArrayLoader:
    """Serve a fixed in-memory row set through the collate loader interface."""

    def __init__(self, rows: np.ndarray):
        self.rows = rows

    def get_rows(self, global_indices: np.ndarray) -> np.ndarray:
        return self.rows[global_indices]


def _with_sampler_cursor(collate_fn: Callable) -> Callable:
    """Attach the cursor returned with the last consumed item in a batch."""

    def _collate(batch_items):
        cursor = None
        if batch_items and isinstance(batch_items[0], SampleRef):
            if not all(isinstance(item, SampleRef) for item in batch_items):
                raise TypeError("Mixed cursor and integer sample references")
            cursor = batch_items[-1].next_cursor.as_dict()
            augmentation_keys = np.array(
                [
                    (
                        item.next_cursor.epoch,
                        item.next_cursor.shard,
                        item.next_cursor.position,
                    )
                    for item in batch_items
                ],
                dtype=np.uint64,
            )
            batch_items = [item.global_index for item in batch_items]
            batch = collate_fn(batch_items, augmentation_keys=augmentation_keys)
        else:
            batch = collate_fn(batch_items)
        if cursor is not None:
            batch["_data_cursor"] = cursor
        return batch

    return _collate


def _materialize_validation_rows(
    shard_loader: ShardLoader,
    val_run_ids: np.ndarray,
    *,
    total_samples: int,
    seed: int,
) -> tuple[np.ndarray, int, int]:
    """Build a deterministic validation set from one held-out-bearing shard.

    Shards in a source pool share the same generation algorithm and board depth,
    so using the first shard containing held-out runs avoids a dataset-wide scan.
    Sampling with replacement is used when the requested fixed validation set is
    larger than the eligible rows in that shard.
    """
    if total_samples <= 0:
        raise ValueError("total_samples must be positive")
    if len(val_run_ids) == 0:
        raise ValueError("validation run IDs must not be empty")

    rng = np.random.default_rng(seed)
    for shard_idx in range(len(shard_loader.shards)):
        shard = shard_loader.load_shard(shard_idx)
        try:
            eligible_indices = np.flatnonzero(np.isin(shard["run_id"], val_run_ids))
            if len(eligible_indices) == 0:
                continue
            replace = total_samples > len(eligible_indices)
            selected = rng.choice(
                eligible_indices,
                size=total_samples,
                replace=replace,
            )
            return shard[selected].copy(), shard_idx, len(eligible_indices)
        finally:
            shard_loader.unload_shard(shard_idx)

    raise ValueError(
        "No physical shard contains rows for the selected validation run IDs; "
        "metadata and packed rows disagree"
    )


def build_steps_dataloaders(
    dataset_dir: str,
    target_mode: str,
    batch_size: int,
    *,
    physical_batch_size: Optional[int] = None,
    tokenizer_path: Optional[str] = None,
    ev_tokenizer: Optional[object] = None,
    train_num_steps: Optional[int] = None,
    num_epochs: Optional[int] = None,
    resume_skip_samples: int = 0,
    resume_data_cursor: Optional[dict] = None,
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
    # New shard-based params
    shard_locality: bool = False,
    shard_locality_block_size: Optional[int] = None,
    shard_cache_in_memory: bool = True,
    shard_cache_keep_shards: int = 1,
    rotation_augment: Optional[object] = None,
    flip_augment: Optional[object] = None,
) -> Tuple[DataLoader, Optional[DataLoader], int, Dict[str, Any]]:
    """Build train/val dataloaders using efficient shard-based loading.

    Returns (dl_train, dl_val_or_None, per_epoch_steps).

    New strategy:
    - Load entire shard into RAM sequentially
    - Sample random steps from shard without regard to run boundaries
    - Trust metadata.db completely for counts
    - No index materialization or full scans
    """

    # Initialize metadata DB and shard loader
    metadata = MetadataDB(dataset_dir)

    # Use mmap_mode=False when shard_cache_in_memory=True for better performance
    # (we're loading shards fully anyway)
    use_mmap = mmap_mode and not shard_cache_in_memory

    # Detect if shards are compressed (before creating loader)
    ds_path = Path(dataset_dir)
    has_compressed_shards = (
        any(ds_path.glob("steps-*.npy.zst"))
        or (ds_path / "steps.npy.zst").exists()
    )

    decompress_dir: Optional[str] = None
    has_decompress_cache = False
    if has_compressed_shards and mmap_mode:
        max_shard_bytes = 0
        shard_paths = sorted(
            list(ds_path.glob("steps-*.npy.zst")) or ([ds_path / "steps.npy.zst"] if (ds_path / "steps.npy.zst").exists() else [])
        )
        for shard_path in shard_paths:
            shape, dtype = _read_npy_header(shard_path)
            num_bytes = int(np.prod(shape)) * int(dtype.itemsize)
            max_shard_bytes = max(max_shard_bytes, num_bytes)

        def _is_tmpfs(path: str) -> bool:
            try:
                with open("/proc/mounts", "r", encoding="utf-8") as handle:
                    for line in handle:
                        parts = line.split()
                        if len(parts) >= 3 and parts[1] == path:
                            return parts[2] == "tmpfs"
            except OSError:
                return False
            return False

        def _free_bytes(path: str) -> int:
            try:
                stat = os.statvfs(path)
                return stat.f_bavail * stat.f_frsize
            except OSError:
                return 0

        for candidate in ("/dev/shm", "/tmp"):
            if _is_tmpfs(candidate) and _free_bytes(candidate) >= max_shard_bytes:
                decompress_dir = candidate
                has_decompress_cache = True
                break

        if has_decompress_cache:
            use_mmap = True
            if shard_cache_in_memory:
                print(
                    f"[data] INFO: Using tmpfs mmap cache for compressed shards; "
                    f"overriding shard_cache_in_memory=True to mmap"
                )
        else:
            if use_mmap:
                print(
                    f"[data] WARNING: Compressed shards without tmpfs cache; "
                    f"disabling mmap_mode and falling back to in-memory loading"
                )
            use_mmap = False

    # CRITICAL: Force num_workers=0 when shard-local in-memory caching is active.
    # Each worker is a separate process and would load/copy full shards independently.
    if num_workers_train > 0 and (
        (has_compressed_shards and not has_decompress_cache) or (shard_locality and not use_mmap)
    ):
        if has_compressed_shards and not has_decompress_cache:
            print(f"[data] WARNING: Compressed shards detected (.zst files) without tmpfs cache")
        else:
            print(f"[data] WARNING: shard_locality with in-memory shards requires num_workers=0")
        print(f"[data] Forcing num_workers=0 to prevent shard duplication across workers")
        num_workers_train = 0

    # Cache shards when shard-locality is enabled so the sampler and collate share
    # a single in-memory shard (evicted on shard rotation).
    cache_shards = bool(shard_locality or shard_cache_in_memory or use_mmap)
    shard_loader = ShardLoader(
        dataset_dir,
        mmap_mode=use_mmap,
        cache_shards=cache_shards,
        cache_keep_shards=shard_cache_keep_shards,
        decompress_dir=decompress_dir,
        decompress_cleanup=True,
    )

    print(f"[data] Dataset: {shard_loader}")
    print(f"[data] Metadata: {metadata.get_run_count()} runs")

    # Split runs into train/val using metadata
    train_run_ids, val_run_ids = metadata.split_runs_train_val(
        run_sql=run_sql,
        sql_params=sql_params,
        val_run_sql=val_run_sql,
        val_sql_params=val_sql_params,
        val_run_pct=val_run_pct,
        val_split_seed=val_split_seed,
    )

    print(
        f"[data] Run split: train={len(train_run_ids)} "
        f"val={0 if val_run_ids is None else len(val_run_ids)}"
    )

    split_requested = run_sql is not None or (
        val_run_ids is not None and len(val_run_ids) > 0
    )
    if split_requested and not shard_locality:
        raise ValueError(
            "Run-filtered training requires shard_locality=true; the streaming "
            "and buffered samplers cannot enforce train_run_ids"
        )

    # Get step counts from metadata (no scanning!)
    meta_train_steps = metadata.get_total_steps_for_runs(train_run_ids)
    meta_val_steps = metadata.get_total_steps_for_runs(val_run_ids) if val_run_ids is not None else 0

    print(f"[data] Steps (from metadata): train={meta_train_steps:,} val={meta_val_steps:,}")

    # Setup collate function
    # CRITICAL: For compressed shards with num_workers=0, pass shard_loader to avoid double-loading
    # The collate would otherwise create its own loader and decompress shards again!
    pass_loader = shard_loader if num_workers_train == 0 else None
    loader_kwargs = None if pass_loader is not None else {
        "mmap_mode": use_mmap,
        "cache_shards": cache_shards,
        "cache_keep_shards": shard_cache_keep_shards,
        "decompress_dir": decompress_dir,
        "decompress_cleanup": True,
    }

    if target_mode == "macroxue_tokens":
        if tokenizer_path is None:
            raise ValueError("tokenizer_path required for macroxue_tokens mode")
        from .collate import make_collate_macroxue_worker_safe
        collate_fn = make_collate_macroxue_worker_safe(
            dataset_dir,
            tokenizer_path,
            rotation_augment=rotation_augment,
            flip_augment=flip_augment,
            shard_loader=pass_loader,
            shard_loader_kwargs=loader_kwargs,
            augmentation_seed=seed,
        )
    else:
        from .collate import make_collate_steps_worker_safe
        collate_fn = make_collate_steps_worker_safe(
            dataset_dir,
            target_mode,
            ev_tokenizer=ev_tokenizer,
            rotation_augment=rotation_augment,
            flip_augment=flip_augment,
            shard_loader=pass_loader,
            shard_loader_kwargs=loader_kwargs,
            augmentation_seed=seed,
        )

    # Build training dataloader
    effective_batch_size = int(batch_size)
    loader_batch_size = int(physical_batch_size or batch_size)
    skip_samples = max(0, int(resume_skip_samples or 0))
    if resume_data_cursor is not None:
        if not shard_locality:
            raise ValueError("A committed data cursor requires shard_locality=true")
        skip_samples = 0
    if skip_samples > 0:
        skipped_steps = skip_samples / max(1, loader_batch_size)
        print(
            f"[data] Resume skip: dropping the first {skip_samples:,} samples (~{skipped_steps:,.0f} micro-batches)"
        )

    # TODO: Handle step_index_min/max filtering if needed
    # For now, ignoring these filters in the new implementation
    if step_index_min is not None or step_index_max is not None:
        print(
            f"[data] WARNING: step_index filtering not yet implemented in v2 loader "
            f"(step_index_min={step_index_min}, step_index_max={step_index_max})"
        )

    # For legacy samplers that don't use ShardDataset
    total_dataset_len = shard_loader.total_steps
    samples_per_epoch = max(1, int(meta_train_steps))

    sampler_info: Dict[str, Any] = {}

    # SAFETY CHECK: NEVER ALLOW SEQUENTIAL TRAINING
    # Sequential training on large datasets is catastrophically bad:
    # - No randomization between epochs
    # - Train/val contamination (run filtering not applied)
    # - Temporal/spatial correlation in consecutive samples
    # This should NEVER happen in production training.
    if not shard_locality and not shuffle and train_num_steps is None:
        raise ValueError(
            "CRITICAL ERROR: Configuration would result in SEQUENTIAL (non-shuffled) training!\n"
            "This is catastrophically bad for model convergence and will cause:\n"
            "  - No randomization between batches/epochs\n"
            "  - Train/validation contamination\n"
            "  - Severe overfitting to sample order\n"
            "\n"
            "You MUST enable one of:\n"
            "  - shard_locality=true (recommended for large datasets with run splits)\n"
            "  - shuffle=true with shuffle_buffer_size (for smaller datasets)\n"
            "  - train_num_steps with streaming random sampler\n"
            "\n"
            "Current config: shard_locality={}, shuffle={}, train_num_steps={}\n".format(
                shard_locality, shuffle, train_num_steps
            )
        )

    # Training sampler strategy
    if shard_locality:
        # New shard-based sampling: load shard, iterate through all steps randomly, move to next
        # Support both num_epochs and num_steps
        if num_epochs is not None and num_epochs > 0:
            base_epochs = max(1, int(num_epochs))
            extra_epochs = ceil(skip_samples / samples_per_epoch) if skip_samples > 0 else 0
            effective_epochs = base_epochs + extra_epochs
            if extra_epochs > 0:
                print(
                    f"[data] Using ShardPoolSampler: {effective_epochs} epoch(s) "
                    f"(requested={base_epochs}, +{extra_epochs} to cover resume skip)"
                )
            else:
                print(f"[data] Using ShardPoolSampler: {effective_epochs} epoch(s), shard-by-shard iteration")
            train_sampler = ShardPoolSampler(
                shard_loader,
                num_epochs=effective_epochs,
                seed=seed,
                run_ids=train_run_ids,
                total_steps=meta_train_steps,  # Trust metadata!
                skip=skip_samples,
                resume_cursor=resume_data_cursor,
            )
            # Dataset length is determined by sampler
            train_dataset = ShardDataset(shard_loader, len(train_sampler))
            sampler_info.update({
                "kind": "ShardPoolSampler",
                "epochs": effective_epochs,
                "requested_epochs": base_epochs,
            })
        elif train_num_steps is not None:
            # If num_steps specified, we need to calculate how many epochs that is
            # and potentially stop mid-epoch
            requested_steps = int(train_num_steps)
            requested_samples = requested_steps * effective_batch_size
            # For now, use a simple approach: create enough epochs, dataloader will stop
            estimated_steps_per_epoch = ceil(meta_train_steps / effective_batch_size)
            estimated_epochs = max(1, ceil(train_num_steps / estimated_steps_per_epoch))
            min_epochs_for_plan = max(1, ceil((skip_samples + requested_samples) / samples_per_epoch))
            effective_epochs = max(estimated_epochs, min_epochs_for_plan)

            print(
                f"[data] Using ShardPoolSampler: ~{effective_epochs} epoch(s) for {train_num_steps} steps"
            )
            train_sampler = ShardPoolSampler(
                shard_loader,
                num_epochs=effective_epochs,
                seed=seed,
                run_ids=train_run_ids,
                total_steps=meta_train_steps,  # Trust metadata!
                skip=skip_samples,
                resume_cursor=resume_data_cursor,
            )
            train_dataset = ShardDataset(shard_loader, len(train_sampler))
            sampler_info.update({
                "kind": "ShardPoolSampler",
                "epochs": effective_epochs,
                "requested_steps": requested_steps,
            })
        else:
            # Default: 1 epoch
            base_epochs = 1
            extra_epochs = ceil(skip_samples / samples_per_epoch) if skip_samples > 0 else 0
            effective_epochs = base_epochs + extra_epochs
            if extra_epochs > 0:
                print(
                    f"[data] Using ShardPoolSampler: {effective_epochs} epoch(s) "
                    f"(default 1 +{extra_epochs} for resume)"
                )
            else:
                print(f"[data] Using ShardPoolSampler: {effective_epochs} epoch(s) (default)")
            train_sampler = ShardPoolSampler(
                shard_loader,
                num_epochs=effective_epochs,
                seed=seed,
                run_ids=train_run_ids,
                total_steps=meta_train_steps,  # Trust metadata!
                skip=skip_samples,
                resume_cursor=resume_data_cursor,
            )
            train_dataset = ShardDataset(shard_loader, len(train_sampler))
            sampler_info.update({
                "kind": "ShardPoolSampler",
                "epochs": effective_epochs,
            })
    elif train_num_steps is not None:
        # Legacy streaming sampler (still useful for some cases)
        requested_steps = int(train_num_steps)
        requested_samples = requested_steps * effective_batch_size
        total_samples = skip_samples + requested_samples
        print(f"[data] Using streaming sampler: {requested_samples:,} samples (+{skip_samples:,} skip)")

        # Create a sampler that just yields indices
        from .steps import StreamingRandomSampler
        train_sampler = StreamingRandomSampler(
            dataset_len=total_dataset_len,
            total_samples=total_samples,
            seed=seed,
            skip=skip_samples,
        )
        train_dataset = ShardDataset(shard_loader, total_samples)
        sampler_info.update({
            "kind": "StreamingRandomSampler",
            "requested_steps": requested_steps,
        })
    elif shuffle:
        # Buffered shuffle for full dataset iteration
        print(f"[data] Using buffered shuffle: buffer_size={shuffle_buffer_size:,}")
        train_sampler = BufferedShuffleSampler(
            dataset_len=total_dataset_len,
            buffer_size=shuffle_buffer_size,
            seed=seed,
            skip=skip_samples,
        )
        train_dataset = ShardDataset(shard_loader, total_dataset_len)
        sampler_info.update({
            "kind": "BufferedShuffleSampler",
        })
    else:
        # This should never be reached due to safety check above
        raise RuntimeError("Unreachable: sequential training path should be blocked by safety check")

    collate_fn = _with_sampler_cursor(collate_fn)
    prefetch_train = 8 if num_workers_train > 0 else None
    train_generator = torch.Generator().manual_seed(seed)
    dl_train = DataLoader(
        train_dataset,
        batch_size=loader_batch_size,
        shuffle=False,
        sampler=train_sampler,
        num_workers=num_workers_train,
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=True if num_workers_train > 0 else False,
        prefetch_factor=prefetch_train,
        generator=train_generator,
    )

    print(
        f"[data] Train DataLoader: batch_size={loader_batch_size} "
        f"(effective={effective_batch_size}), workers={num_workers_train}"
    )

    skip_applied = int(getattr(train_sampler, "skip", skip_samples)) if hasattr(train_sampler, "skip") else skip_samples
    sampler_info.setdefault("kind", type(train_sampler).__name__)
    sampler_info.update(
        {
            "skip_samples": skip_applied,
            "output_samples": len(train_sampler),
            "stream_samples": len(train_sampler) + skip_applied,
            "seed": int(seed),
            "shard_locality": bool(shard_locality),
        }
    )
    if train_num_steps is not None:
        sampler_info.setdefault("target_steps", int(train_num_steps))
    if num_epochs is not None:
        sampler_info.setdefault("requested_epochs", int(num_epochs))

    # Validation dataloader
    dl_val: Optional[DataLoader] = None
    val_info: Dict[str, Any] = {}
    if val_run_ids is not None and len(val_run_ids) > 0:
        # Determine validation sample count
        max_val_steps = None
        if val_num_steps is not None and val_num_steps > 0:
            max_val_steps = val_num_steps
        elif val_steps_pct > 0.0:
            planned_train_steps = train_num_steps if train_num_steps else ceil(meta_train_steps / effective_batch_size)
            max_val_steps = max(1, int(round(planned_train_steps * val_steps_pct)))

        if max_val_steps is None:
            raise ValueError(
                "A validation run split requires val_num_steps or val_steps_pct; "
                "uncapped validation cannot be materialized safely"
            )

        total_val_samples = max_val_steps * loader_batch_size
        print(f"[data] Validation: {max_val_steps} steps ({total_val_samples:,} samples)")
        val_rows, val_shard_idx, eligible_val_rows = _materialize_validation_rows(
            shard_loader,
            val_run_ids,
            total_samples=total_val_samples,
            seed=seed + 1,
        )
        print(
            f"[data] Validation: fixed held-out set from shard {val_shard_idx} "
            f"({eligible_val_rows:,} eligible rows)"
        )
        val_loader = _ArrayLoader(val_rows)
        val_dataset = ShardDataset(val_loader, len(val_rows))
        val_sampler = SequentialSampler(len(val_rows))

        # Validation is fixed and unaugmented. Keeping it in-process also avoids
        # copying the materialized rows into worker processes.
        if target_mode == "macroxue_tokens":
            from .collate import make_collate_macroxue_worker_safe
            val_collate = make_collate_macroxue_worker_safe(
                dataset_dir,
                tokenizer_path,
                rotation_augment=None,
                flip_augment=None,
                shard_loader=val_loader,
            )
        else:
            from .collate import make_collate_steps_worker_safe
            val_collate = make_collate_steps_worker_safe(
                dataset_dir,
                target_mode,
                ev_tokenizer=ev_tokenizer,
                rotation_augment=None,
                flip_augment=None,
                shard_loader=val_loader,
            )
        val_generator = torch.Generator().manual_seed(seed + 1)
        dl_val = DataLoader(
            val_dataset,
            batch_size=loader_batch_size,
            shuffle=(val_sampler is None),
            sampler=val_sampler,
            num_workers=0,
            collate_fn=val_collate,
            pin_memory=True,
            persistent_workers=False,
            generator=val_generator,
            prefetch_factor=None,
        )
        val_info = {
            "samples": int(total_val_samples),
            "source_shard": int(val_shard_idx),
            "eligible_rows_in_source_shard": int(eligible_val_rows),
            "seed": int(seed + 1),
            "augmentation": False,
        }

    # Calculate per-epoch steps
    if train_num_steps is not None:
        per_epoch_steps = int(train_num_steps)
    else:
        per_epoch_steps = ceil(meta_train_steps / effective_batch_size)

    print(f"[data] Per-epoch steps: {per_epoch_steps:,}")

    metadata: Dict[str, Any] = {
        "dataset_dir": str(Path(dataset_dir).resolve()),
        "train_run_ids": train_run_ids,
        "val_run_ids": val_run_ids,
        "meta_train_steps": int(meta_train_steps),
        "meta_val_steps": int(meta_val_steps),
        "sampler": sampler_info,
        "effective_batch_size": effective_batch_size,
        "loader_batch_size": loader_batch_size,
        "train_dataset_len": len(train_sampler),
        "total_dataset_len": int(total_dataset_len),
        "resume_skip_samples": int(skip_applied),
        "train_num_steps": int(train_num_steps) if train_num_steps is not None else None,
        "num_epochs": int(num_epochs) if num_epochs is not None else None,
        "validation": val_info,
    }

    return dl_train, dl_val, per_epoch_steps, metadata
