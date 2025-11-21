from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

from torch.utils.data import DataLoader

from ..config import RejectionConfig, TrainingConfig
from ..rejection import build_rejection_dataloaders
from .steps_v2 import build_steps_dataloaders as _build_steps_dataloaders_v2


def build_dataloaders(
    cfg: TrainingConfig,
    *,
    num_workers_train: int = 12,
) -> Tuple[DataLoader, Optional[DataLoader], int, Dict[str, Any]]:
    """Construct train/val DataLoaders from steps.npy + metadata.db.

    Splits are disjoint by runs, configured via SQL or random run split.
    """

    target_mode = getattr(cfg.target, "mode", "binned_ev")
    binner = None
    if target_mode == "binned_ev":
        from ..binning import Binner

        binner = Binner.from_config(cfg.binning)

    ds_cfg = cfg.dataset
    return build_steps_dataloaders(
        dataset_dir=ds_cfg.resolved_dataset_dir(),
        binner=binner,
        target_mode=target_mode,
        batch_size=cfg.batch.batch_size,
        physical_batch_size=cfg.batch.physical_batch_size(),
        tokenizer_path=ds_cfg.resolved_tokenizer_path(),
        run_sql=getattr(ds_cfg, "run_sql", None),
        sql_params=getattr(ds_cfg, "sql_params", None),
        val_run_sql=getattr(ds_cfg, "val_run_sql", None),
        val_sql_params=getattr(ds_cfg, "val_sql_params", None),
        val_run_pct=float(getattr(ds_cfg, "val_run_pct", 0.0) or 0.0),
        val_split_seed=int(getattr(ds_cfg, "val_split_seed", 42) or 42),
        num_workers_train=num_workers_train,
        mmap_mode=getattr(ds_cfg, "mmap_mode", False),
        step_index_min=getattr(ds_cfg, "step_index_min", None),
        step_index_max=getattr(ds_cfg, "step_index_max", None),
        shuffle=bool(getattr(ds_cfg, "shuffle", False)),
        shuffle_buffer_size=int(getattr(ds_cfg, "shuffle_buffer_size", 1_000_000) or 1_000_000),
        shard_locality=bool(getattr(ds_cfg, "shard_locality", False)),
        shard_locality_block_size=getattr(ds_cfg, "shard_locality_block_size", None),
        shard_cache_in_memory=bool(getattr(ds_cfg, "shard_cache_in_memory", False)),
        shard_cache_keep_shards=int(getattr(ds_cfg, "shard_cache_keep_shards", 1) or 1),
        train_num_steps=getattr(ds_cfg, "num_steps", None),
        num_epochs=getattr(ds_cfg, "num_epochs", None),
        val_num_steps=getattr(ds_cfg, "val_num_steps", None),
        val_steps_pct=float(getattr(ds_cfg, "val_steps_pct", 0.0) or 0.0),
        rejection=ds_cfg.rejection,
        seed=cfg.seed,
    )


def build_steps_dataloaders(
    dataset_dir: str,
    binner: Optional[object],
    target_mode: str,
    batch_size: int,
    *,
    physical_batch_size: Optional[int] = None,
    tokenizer_path: Optional[str] = None,
    ev_tokenizer: Optional[object] = None,
    train_num_steps: Optional[int] = None,
    num_epochs: Optional[int] = None,
    resume_skip_samples: int = 0,
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
    shard_locality: bool = False,
    shard_locality_block_size: Optional[int] = None,
    shard_cache_in_memory: bool = True,
    shard_cache_keep_shards: int = 1,
    rejection: Optional[RejectionConfig] = None,
) -> Tuple[DataLoader, Optional[DataLoader], int, Dict[str, Any]]:
    if rejection is not None:
        return build_rejection_dataloaders(
            dataset_dir=dataset_dir,
            binner=binner,
            target_mode=target_mode,
            batch_size=batch_size,
            physical_batch_size=physical_batch_size,
            tokenizer_path=tokenizer_path,
            ev_tokenizer=ev_tokenizer,
            train_num_steps=train_num_steps,
            num_epochs=num_epochs,
            resume_skip_samples=resume_skip_samples,
            seed=seed,
            shuffle=shuffle,
            shuffle_buffer_size=shuffle_buffer_size,
            val_num_steps=val_num_steps,
            val_steps_pct=val_steps_pct,
            run_sql=run_sql,
            sql_params=sql_params,
            val_run_sql=val_run_sql,
            val_sql_params=val_sql_params,
            val_run_pct=val_run_pct,
            val_split_seed=val_split_seed,
            num_workers_train=num_workers_train,
            mmap_mode=mmap_mode,
            step_index_min=step_index_min,
            step_index_max=step_index_max,
            shard_locality=shard_locality,
            shard_locality_block_size=shard_locality_block_size,
            shard_cache_in_memory=shard_cache_in_memory,
            shard_cache_keep_shards=shard_cache_keep_shards,
            rejection_cfg=rejection,
        )

    return _build_steps_dataloaders_v2(
        dataset_dir=dataset_dir,
        binner=binner,
        target_mode=target_mode,
        batch_size=batch_size,
        physical_batch_size=physical_batch_size,
        tokenizer_path=tokenizer_path,
        ev_tokenizer=ev_tokenizer,
        train_num_steps=train_num_steps,
        num_epochs=num_epochs,
        resume_skip_samples=resume_skip_samples,
        seed=seed,
        shuffle=shuffle,
        shuffle_buffer_size=shuffle_buffer_size,
        val_num_steps=val_num_steps,
        val_steps_pct=val_steps_pct,
        run_sql=run_sql,
        sql_params=sql_params,
        val_run_sql=val_run_sql,
        val_sql_params=val_sql_params,
        val_run_pct=val_run_pct,
        val_split_seed=val_split_seed,
        num_workers_train=num_workers_train,
        mmap_mode=mmap_mode,
        step_index_min=step_index_min,
        step_index_max=step_index_max,
        shard_locality=shard_locality,
        shard_locality_block_size=shard_locality_block_size,
        shard_cache_in_memory=shard_cache_in_memory,
        shard_cache_keep_shards=shard_cache_keep_shards,
    )


__all__ = ["build_dataloaders", "build_steps_dataloaders"]
