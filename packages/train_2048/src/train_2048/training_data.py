from __future__ import annotations

from typing import Any, Optional
from pathlib import Path
import hashlib

import numpy as np
import torch
from torch.utils.data import DataLoader

from .binning import BinningConfig
from .config import TrainingConfig
from .dataloader import build_steps_dataloaders
from .dataloader.metadata import MetadataDB
from .tokenization.ev_binning import EVBinnerTokenizer


def _extract_batch_dim(value: Any) -> Optional[int]:
    if isinstance(value, torch.Tensor):
        return int(value.shape[0])
    if hasattr(value, "shape") and getattr(value, "shape", None) is not None:
        shape = value.shape
        if isinstance(shape, (tuple, list)) and len(shape) > 0:
            return int(shape[0])
    if isinstance(value, (list, tuple)):
        return len(value)
    return None


def infer_batch_size(batch: Any) -> int:
    """Infer batch size from a dataloader payload."""
    if isinstance(batch, dict):
        for val in batch.values():
            size = _extract_batch_dim(val)
            if size is not None:
                return size
    elif isinstance(batch, (list, tuple)):
        for val in batch:
            size = _extract_batch_dim(val)
            if size is not None:
                return size
    else:
        size = _extract_batch_dim(batch)
        if size is not None:
            return size
    raise ValueError("Unable to infer batch size from training batch payload")


def _hash_run_ids(run_ids: Optional[np.ndarray]) -> str:
    if run_ids is None:
        return "none"
    arr = np.asarray(run_ids, dtype=np.int64)
    return hashlib.sha1(arr.tobytes()).hexdigest()


def collect_dataset_signature(cfg: TrainingConfig) -> dict:
    """Collect dataset metadata needed to validate resume compatibility."""
    ds_cfg = cfg.dataset
    metadata = MetadataDB(ds_cfg.resolved_dataset_dir())
    train_run_ids, val_run_ids = metadata.split_runs_train_val(
        run_sql=ds_cfg.run_sql,
        sql_params=ds_cfg.sql_params,
        val_run_sql=ds_cfg.val_run_sql,
        val_sql_params=ds_cfg.val_sql_params,
        val_run_pct=ds_cfg.val_run_pct,
        val_split_seed=ds_cfg.val_split_seed,
    )
    meta_train_steps = metadata.get_total_steps_for_runs(train_run_ids)
    meta_val_steps = metadata.get_total_steps_for_runs(val_run_ids) if val_run_ids is not None else 0
    return {
        "dataset_dir": str(Path(ds_cfg.resolved_dataset_dir()).resolve()),
        "train_run_ids": train_run_ids,
        "val_run_ids": val_run_ids,
        "meta_train_steps": int(meta_train_steps),
        "meta_val_steps": int(meta_val_steps),
    }


def compute_dataset_fingerprint(signature: dict) -> str:
    """Compute a stable fingerprint for a dataset signature payload."""
    digest = hashlib.sha1()
    digest.update(signature["dataset_dir"].encode("utf-8"))
    digest.update(str(signature["meta_train_steps"]).encode("utf-8"))
    digest.update(str(signature["meta_val_steps"]).encode("utf-8"))
    digest.update(_hash_run_ids(signature.get("train_run_ids")).encode("utf-8"))
    digest.update(_hash_run_ids(signature.get("val_run_ids")).encode("utf-8"))
    return digest.hexdigest()


def _to_serializable(obj: Any) -> Any:
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(v) for v in obj]
    return obj


def build_dataset_checkpoint_metadata(signature: dict, dataloader_meta: dict, fingerprint: str) -> dict:
    """Build dataset metadata stored inside checkpoint bundles."""
    sampler_info = _to_serializable(dataloader_meta.get("sampler", {}))
    return {
        "version": 1,
        "fingerprint": fingerprint,
        "dataset_dir": signature["dataset_dir"],
        "meta_train_steps": int(signature["meta_train_steps"]),
        "meta_val_steps": int(signature["meta_val_steps"]),
        "train_run_count": int(len(signature["train_run_ids"])),
        "train_runs_hash": _hash_run_ids(signature["train_run_ids"]),
        "val_run_count": int(len(signature["val_run_ids"])) if signature.get("val_run_ids") is not None else 0,
        "val_runs_hash": _hash_run_ids(signature.get("val_run_ids")),
        "sampler": sampler_info,
        "train_num_steps": dataloader_meta.get("train_num_steps"),
        "num_epochs": dataloader_meta.get("num_epochs"),
        "total_dataset_len": int(dataloader_meta.get("total_dataset_len", 0) or 0),
        "train_dataset_len": int(dataloader_meta.get("train_dataset_len", 0) or 0),
        "resume_skip_samples": int(dataloader_meta.get("resume_skip_samples", 0) or 0),
    }


def init_datasets(
    cfg: TrainingConfig,
    target_mode: str,
    *,
    train_num_steps_override: Optional[int] = None,
    resume_skip_samples: int = 0,
) -> tuple[DataLoader, Optional[DataLoader], int, dict]:
    """Construct train/val dataloaders plus step metadata."""
    ev_tok = None
    if target_mode == "binned_ev":
        # Standardize EV tokenization via EVTokenizer wrapper.
        ev_tok = EVBinnerTokenizer(BinningConfig(**cfg.binning.model_dump()))
    return build_steps_dataloaders(
        dataset_dir=cfg.dataset.resolved_dataset_dir(),
        binner=None,
        target_mode=target_mode,
        batch_size=cfg.batch.batch_size,
        physical_batch_size=cfg.batch.physical_batch_size(),
        train_num_steps=(cfg.dataset.num_steps if train_num_steps_override is None else train_num_steps_override),
        resume_skip_samples=resume_skip_samples,
        seed=cfg.seed,
        shuffle_buffer_size=getattr(cfg.dataset, "shuffle_buffer_size", 1_000_000),
        shard_locality=getattr(cfg.dataset, "shard_locality", False),
        shard_locality_block_size=getattr(cfg.dataset, "shard_locality_block_size", None),
        shard_cache_in_memory=getattr(cfg.dataset, "shard_cache_in_memory", False),
        shard_cache_keep_shards=getattr(cfg.dataset, "shard_cache_keep_shards", 1),
        val_num_steps=getattr(cfg.dataset, "val_num_steps", None),
        val_steps_pct=getattr(cfg.dataset, "val_steps_pct", 0.0),
        tokenizer_path=cfg.dataset.resolved_tokenizer_path(),
        ev_tokenizer=ev_tok,
        run_sql=cfg.dataset.run_sql,
        sql_params=cfg.dataset.sql_params,
        val_run_sql=cfg.dataset.val_run_sql,
        val_sql_params=cfg.dataset.val_sql_params,
        val_run_pct=cfg.dataset.val_run_pct,
        val_split_seed=cfg.dataset.val_split_seed,
        num_workers_train=12,
        mmap_mode=cfg.dataset.mmap_mode,
        rotation_augment=getattr(cfg.dataset, "rotation_augment", None),
        flip_augment=getattr(cfg.dataset, "flip_augment", None),
    )


__all__ = [
    "infer_batch_size",
    "collect_dataset_signature",
    "compute_dataset_fingerprint",
    "build_dataset_checkpoint_metadata",
    "init_datasets",
]
