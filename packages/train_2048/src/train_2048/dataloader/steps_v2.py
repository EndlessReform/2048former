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
from typing import Optional, Sequence, Tuple

import numpy as np
from torch.utils.data import DataLoader

from .shard_loader import ShardLoader
from .dataset import ShardDataset
from .metadata import MetadataDB
from .samplers import ShardPoolSampler, BufferedShuffleSampler, SequentialSampler
from .collate import make_collate_macroxue, make_collate_steps

from ..binning import Binner
from ..tokenization.base import BoardCodec


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
    num_epochs: Optional[int] = None,
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
) -> Tuple[DataLoader, Optional[DataLoader], int]:
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
    shard_loader = ShardLoader(dataset_dir, mmap_mode=use_mmap)

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

    # Get step counts from metadata (no scanning!)
    meta_train_steps = metadata.get_total_steps_for_runs(train_run_ids)
    meta_val_steps = metadata.get_total_steps_for_runs(val_run_ids) if val_run_ids is not None else 0

    print(f"[data] Steps (from metadata): train={meta_train_steps:,} val={meta_val_steps:,}")

    # Setup collate function - pass shard_loader directly to avoid pickling issues
    if target_mode == "macroxue_tokens":
        if tokenizer_path is None:
            raise ValueError("tokenizer_path required for macroxue_tokens mode")
        # Create a worker-safe collate that doesn't capture dataset in closure
        from .collate import make_collate_macroxue_worker_safe
        collate_fn = make_collate_macroxue_worker_safe(dataset_dir, tokenizer_path)
    else:
        from .collate import make_collate_steps_worker_safe
        collate_fn = make_collate_steps_worker_safe(dataset_dir, target_mode, binner, ev_tokenizer=ev_tokenizer)

    # Build training dataloader
    effective_batch_size = int(batch_size)
    loader_batch_size = int(physical_batch_size or batch_size)

    # TODO: Handle step_index_min/max filtering if needed
    # For now, ignoring these filters in the new implementation
    if step_index_min is not None or step_index_max is not None:
        print(
            f"[data] WARNING: step_index filtering not yet implemented in v2 loader "
            f"(step_index_min={step_index_min}, step_index_max={step_index_max})"
        )

    # For legacy samplers that don't use ShardDataset
    total_dataset_len = shard_loader.total_steps

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
            print(f"[data] Using ShardPoolSampler: {num_epochs} epoch(s), shard-by-shard iteration")
            train_sampler = ShardPoolSampler(
                shard_loader,
                num_epochs=num_epochs,
                seed=seed,
                run_ids=train_run_ids,
                total_steps=meta_train_steps,  # Trust metadata!
            )
            # Dataset length is determined by sampler
            train_dataset = ShardDataset(shard_loader, len(train_sampler))
        elif train_num_steps is not None:
            # If num_steps specified, we need to calculate how many epochs that is
            # and potentially stop mid-epoch
            total_samples = train_num_steps * effective_batch_size
            # For now, use a simple approach: create enough epochs, dataloader will stop
            estimated_steps_per_epoch = ceil(meta_train_steps / effective_batch_size)
            estimated_epochs = max(1, ceil(train_num_steps / estimated_steps_per_epoch))

            print(f"[data] Using ShardPoolSampler: ~{estimated_epochs} epoch(s) for {train_num_steps} steps")
            train_sampler = ShardPoolSampler(
                shard_loader,
                num_epochs=estimated_epochs,
                seed=seed,
                run_ids=train_run_ids,
                total_steps=meta_train_steps,  # Trust metadata!
            )
            train_dataset = ShardDataset(shard_loader, len(train_sampler))
        else:
            # Default: 1 epoch
            print(f"[data] Using ShardPoolSampler: 1 epoch (default)")
            train_sampler = ShardPoolSampler(
                shard_loader,
                num_epochs=1,
                seed=seed,
                run_ids=train_run_ids,
                total_steps=meta_train_steps,  # Trust metadata!
            )
            train_dataset = ShardDataset(shard_loader, len(train_sampler))
    elif train_num_steps is not None:
        # Legacy streaming sampler (still useful for some cases)
        total_samples = train_num_steps * effective_batch_size
        print(f"[data] Using streaming sampler: {total_samples:,} samples")

        # Create a sampler that just yields indices
        from .steps import StreamingRandomSampler
        train_sampler = StreamingRandomSampler(
            dataset_len=total_dataset_len,
            total_samples=total_samples,
            seed=seed,
        )
        train_dataset = ShardDataset(shard_loader, total_samples)
    elif shuffle:
        # Buffered shuffle for full dataset iteration
        print(f"[data] Using buffered shuffle: buffer_size={shuffle_buffer_size:,}")
        train_sampler = BufferedShuffleSampler(
            dataset_len=total_dataset_len,
            buffer_size=shuffle_buffer_size,
            seed=seed,
        )
        train_dataset = ShardDataset(shard_loader, total_dataset_len)
    else:
        # This should never be reached due to safety check above
        raise RuntimeError("Unreachable: sequential training path should be blocked by safety check")

    prefetch_train = 8 if num_workers_train > 0 else None
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
    )

    print(
        f"[data] Train DataLoader: batch_size={loader_batch_size} "
        f"(effective={effective_batch_size}), workers={num_workers_train}"
    )

    # Validation dataloader
    dl_val: Optional[DataLoader] = None
    if val_run_ids is not None and len(val_run_ids) > 0:
        num_workers_val = max(2, num_workers_train // 3)

        # Determine validation sample count
        max_val_steps = None
        if val_num_steps is not None and val_num_steps > 0:
            max_val_steps = val_num_steps
        elif val_steps_pct > 0.0:
            planned_train_steps = train_num_steps if train_num_steps else ceil(meta_train_steps / effective_batch_size)
            max_val_steps = max(1, int(round(planned_train_steps * val_steps_pct)))

        if max_val_steps:
            total_val_samples = max_val_steps * loader_batch_size
            print(f"[data] Validation: {max_val_steps} steps ({total_val_samples:,} samples)")

            # Simple random sampling for validation
            from .steps import StreamingRandomSampler
            val_sampler = StreamingRandomSampler(
                dataset_len=total_dataset_len,
                total_samples=total_val_samples,
                seed=seed + 1,
            )
            val_dataset = ShardDataset(shard_loader, total_val_samples)
        else:
            # Use all val data
            val_sampler = None
            val_dataset = ShardDataset(shard_loader, total_dataset_len)

        # Use same collate function
        prefetch_val = 4 if num_workers_val > 0 else None
        dl_val = DataLoader(
            val_dataset,
            batch_size=loader_batch_size,
            shuffle=(val_sampler is None),
            sampler=val_sampler,
            num_workers=num_workers_val,
            collate_fn=collate_fn,
            pin_memory=True,
            persistent_workers=True if num_workers_val > 0 else False,
            prefetch_factor=prefetch_val,
        )

    # Calculate per-epoch steps
    if train_num_steps is not None:
        per_epoch_steps = train_num_steps
    else:
        per_epoch_steps = ceil(meta_train_steps / effective_batch_size)

    print(f"[data] Per-epoch steps: {per_epoch_steps:,}")

    return dl_train, dl_val, per_epoch_steps
