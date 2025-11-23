from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from torch.utils.data import DataLoader

from ..config import TrainingConfig

# Use new shard-based implementation by default
from .steps_v2 import build_steps_dataloaders


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
        value_cfg=getattr(cfg, "value_training", None),
    )


__all__ = ["build_dataloaders"]
