from __future__ import annotations

import logging
from dataclasses import dataclass
from math import ceil
import math
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence

import numpy as np
from torch.utils.data import DataLoader, Sampler

from ..binning import Binner
from ..config import RejectionConfig, RejectionFilterConfig
from ..dataloader.metadata import MetadataDB
from ..dataloader.steps import (
    StepsDataset,
    StreamingRandomSampler,
    make_collate_macroxue,
    make_collate_steps,
)

LOGGER = logging.getLogger("train_2048.rejection")

# Annotation bit flags (keep in sync with Rust schema)
POLICY_P1_BIT = 1 << 0


@dataclass(frozen=True)
class FilterRuntime:
    """Resolved runtime information for a rejection filter."""

    config: RejectionFilterConfig
    positions: Optional[np.ndarray]  # local indices (within training subset)
    capacity: int

    @property
    def id(self) -> str:
        return self.config.id

    @property
    def name(self) -> str:
        return self.config.name

    @property
    def base_weight(self) -> float:
        return float(self.config.weight)

    @property
    def anneal_until_epoch(self) -> Optional[int]:
        return self.config.anneal_until_epoch


def build_rejection_dataloaders(
    dataset_dir: str,
    binner: Optional[Binner],
    target_mode: str,
    batch_size: int,
    *,
    rejection_cfg: RejectionConfig,
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
) -> tuple[DataLoader, Optional[DataLoader], int, Dict[str, object]]:
    """Build dataloaders that sample by rejection mix instead of shard locality."""

    if step_index_min is not None or step_index_max is not None:
        raise NotImplementedError("step_index_min/step_index_max unsupported with rejection sampling")
    if shuffle or shard_locality:
        LOGGER.warning(
            "[rejection] Ignoring shuffle/shard_locality flags; sampler handles ordering internally."
        )
    if resume_skip_samples:
        LOGGER.warning(
            "[rejection] resume_skip_samples=%s is not yet supported; continuing without skipping.",
            resume_skip_samples,
        )
    annotation_dir = rejection_cfg.resolved_annotation_dir()
    dataset_path = Path(dataset_dir)
    ds_train = StepsDataset(dataset_dir, mmap_mode=mmap_mode)
    total_rows = len(ds_train)

    metadata = MetadataDB(dataset_dir)
    train_run_ids, val_run_ids = metadata.split_runs_train_val(
        run_sql=run_sql,
        sql_params=sql_params,
        val_run_sql=val_run_sql,
        val_sql_params=val_sql_params,
        val_run_pct=val_run_pct,
        val_split_seed=val_split_seed,
    )
    run_ids_all = ds_train.get_run_ids()
    train_mask = np.isin(run_ids_all, train_run_ids)
    train_indices = np.nonzero(train_mask)[0].astype(np.int64, copy=False)
    if train_indices.size == 0:
        raise ValueError("rejection sampler: training split produced zero rows")
    train_indices.sort(kind="mergesort")
    train_total = int(train_indices.size)
    ds_train.indices = train_indices

    val_indices: Optional[np.ndarray] = None
    if val_run_ids is not None and val_run_ids.size > 0:
        val_mask = np.isin(run_ids_all, val_run_ids)
        indices = np.nonzero(val_mask)[0].astype(np.int64, copy=False)
        if indices.size == 0:
            LOGGER.warning("[rejection] validation split requested but matched zero rows; skipping val loader.")
        else:
            val_indices = indices

    meta_train_steps = metadata.get_total_steps_for_runs(train_run_ids)
    meta_val_steps = metadata.get_total_steps_for_runs(val_run_ids) if val_run_ids is not None else 0

    LOGGER.info(
        "[rejection] dataset=%s rows=%s train_rows=%s val_rows=%s annotation_dir=%s seed=%s",
        dataset_path,
        f"{total_rows:,}",
        f"{train_total:,}",
        ("0" if val_indices is None else f"{val_indices.size:,}"),
        annotation_dir,
        rejection_cfg.seed,
    )

    annotation_total, filter_matches = _collect_filter_indices(annotation_dir, rejection_cfg.filters)

    if annotation_total != total_rows:
        raise ValueError(
            f"annotation rows ({annotation_total}) do not match dataset rows ({total_rows}); "
            "ensure the dataset and annotation directory are aligned"
        )

    used_mask = np.zeros(train_total, dtype=bool)
    mapped_positions: Dict[str, np.ndarray] = {}

    for cfg in rejection_cfg.filters:
        if cfg.name == "passthrough":
            continue
        positions = filter_matches[cfg.id]
        if positions.size:
            locs = np.searchsorted(train_indices, positions)
            valid = (locs < train_total)
            if valid.any():
                locs = locs[valid]
                pos_valid = positions[valid]
                match = train_indices[locs] == pos_valid
                locs = locs[match]
                if locs.size:
                    locs = np.unique(locs.astype(np.int64, copy=False))
                else:
                    locs = np.empty((0,), dtype=np.int64)
            else:
                locs = np.empty((0,), dtype=np.int64)
        else:
            locs = np.empty((0,), dtype=np.int64)
        used_mask[locs] = True
        mapped_positions[cfg.id] = locs

    base_mask = used_mask.copy()

    filters_runtime: List[FilterRuntime] = []
    for cfg in rejection_cfg.filters:
        if cfg.name == "passthrough":
            remaining_capacity = int(train_total - int(base_mask.sum()))
            filters_runtime.append(
                FilterRuntime(
                    config=cfg,
                    positions=None,
                    capacity=remaining_capacity,
                )
            )
        else:
            locs = mapped_positions.get(cfg.id, np.empty((0,), dtype=np.int64))
            filters_runtime.append(
                FilterRuntime(
                    config=cfg,
                    positions=locs,
                    capacity=int(locs.size),
                )
            )

    total_weight = sum(runtime.base_weight for runtime in filters_runtime)
    if total_weight <= 0.0:
        raise ValueError("rejection mix weights must be positive")
    ratios = [runtime.base_weight / total_weight for runtime in filters_runtime]
    capacities = [runtime.capacity for runtime in filters_runtime]

    limits: List[float] = []
    for ratio, cap in zip(ratios, capacities):
        if ratio <= 0.0:
            continue
        if cap <= 0:
            limits.append(0.0)
        else:
            limits.append(cap / ratio)
    target_total = int(math.floor(min(limits))) if limits else 0
    target_total = max(0, min(target_total, train_total))

    if target_total == 0:
        raise ValueError("rejection mix resolved to zero available samples; check filter capacities.")

    draw_counts: List[int] = []
    fractional: List[float] = []
    for ratio, cap in zip(ratios, capacities):
        desired = ratio * target_total
        draw = min(cap, int(math.floor(desired)))
        draw_counts.append(draw)
        fractional.append(desired - draw)

    remaining = int(target_total - sum(draw_counts))
    if remaining > 0:
        order = sorted(range(len(filters_runtime)), key=lambda i: fractional[i], reverse=True)
        for idx in order:
            if remaining <= 0:
                break
            cap = capacities[idx]
            if draw_counts[idx] >= cap:
                continue
            inc = min(remaining, cap - draw_counts[idx])
            draw_counts[idx] += inc
            remaining -= inc

    if remaining > 0:
        LOGGER.warning(
            "[rejection] Unable to fully allocate target_total=%s due to capacity limits; reducing by %s",
            target_total,
            remaining,
        )
        target_total -= remaining
        for idx in reversed(range(len(draw_counts))):
            if remaining <= 0:
                break
            dec = min(draw_counts[idx], remaining)
            draw_counts[idx] -= dec
            remaining -= dec
        if target_total <= 0:
            raise ValueError("rejection mix exhausted after capacity adjustment")

    for runtime, count in zip(filters_runtime, draw_counts):
        LOGGER.info(
            "[rejection] filter id=%s assign=%s capacity=%s weight=%.4f",
            runtime.id,
            f"{int(count):,}",
            f"{int(runtime.capacity):,}",
            runtime.base_weight,
        )

    effective_batch_size = int(batch_size)
    loader_batch_size = int(physical_batch_size or batch_size)
    if effective_batch_size <= 0 or loader_batch_size <= 0:
        raise ValueError("batch sizes must be positive")

    if target_mode == "macroxue_tokens":
        if tokenizer_path is None:
            raise ValueError("tokenizer_path required for macroxue_tokens mode")
        collate_fn = make_collate_macroxue(ds_train, tokenizer_path)
    else:
        collate_fn = make_collate_steps(target_mode, ds_train, binner, ev_tokenizer=ev_tokenizer)

    per_epoch_samples = target_total
    per_epoch_steps = ceil(per_epoch_samples / effective_batch_size)

    sampler_seed = int(seed) ^ int(rejection_cfg.seed)
    sampler = _RejectionSampler(
        filters_runtime,
        draw_counts=draw_counts,
        base_mask=base_mask,
        dataset_len=train_total,
        seed=sampler_seed,
    )

    train_loader = DataLoader(
        ds_train,
        batch_size=loader_batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=num_workers_train,
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=True if num_workers_train > 0 else False,
        prefetch_factor=8 if num_workers_train > 0 else None,
    )

    metadata: Dict[str, object] = {
        "dataset_dir": str(dataset_path.resolve()),
        "train_run_ids": train_run_ids,
        "val_run_ids": val_run_ids,
        "meta_train_steps": int(meta_train_steps),
        "meta_val_steps": int(meta_val_steps),
        "sampler": {
            "kind": "RejectionSampler",
            "total_samples": per_epoch_samples,
            "filters": {
                runtime.id: {
                    "name": runtime.name,
                    "weight": runtime.base_weight,
                    "anneal_until_epoch": runtime.anneal_until_epoch,
                    "capacity": int(runtime.capacity),
                    "draw": int(draw),
                }
                for runtime, draw in zip(filters_runtime, draw_counts)
            },
        },
        "effective_batch_size": effective_batch_size,
        "loader_batch_size": loader_batch_size,
        "train_dataset_len": int(train_total),
        "total_dataset_len": int(total_rows),
        "val_dataset_len": int(val_indices.size) if val_indices is not None else 0,
        "resume_skip_samples": 0,
        "train_num_steps": int(train_num_steps) if train_num_steps is not None else None,
        "num_epochs": int(num_epochs) if num_epochs is not None else None,
    }

    dl_val: Optional[DataLoader] = None
    if val_indices is not None and val_indices.size > 0:
        num_workers_val = max(2, num_workers_train // 3)
        ds_val = StepsDataset(dataset_dir, indices=val_indices, mmap_mode=mmap_mode)
        if target_mode == "macroxue_tokens":
            if tokenizer_path is None:
                raise ValueError("tokenizer_path required for macroxue_tokens mode")
            collate_val = make_collate_macroxue(ds_val, tokenizer_path)
        else:
            collate_val = make_collate_steps(target_mode, ds_val, binner, ev_tokenizer=ev_tokenizer)

        val_sampler = None
        if val_num_steps is not None and int(val_num_steps) > 0:
            total_val = len(ds_val)
            total_samples = int(val_num_steps) * loader_batch_size
            val_sampler = StreamingRandomSampler(total_val, total_samples, seed=sampler_seed + 1)
        elif val_steps_pct and val_steps_pct > 0.0:
            planned_train_steps = (
                int(train_num_steps)
                if train_num_steps is not None and int(train_num_steps) > 0
                else per_epoch_steps
            )
            derived_steps = int(max(1, round(float(planned_train_steps) * float(val_steps_pct))))
            total_val = len(ds_val)
            total_samples = derived_steps * loader_batch_size
            val_sampler = StreamingRandomSampler(total_val, total_samples, seed=sampler_seed + 1)

        dl_val = DataLoader(
            ds_val,
            batch_size=loader_batch_size,
            shuffle=(val_sampler is None),
            sampler=val_sampler,
            num_workers=num_workers_val,
            collate_fn=collate_val,
            pin_memory=True,
            persistent_workers=True if num_workers_val > 0 else False,
            prefetch_factor=4 if num_workers_val > 0 else None,
        )

    return train_loader, dl_val, per_epoch_steps, metadata


class _RejectionSampler(Sampler[int]):
    """Sample indices according to a fixed draw plan."""

    def __init__(
        self,
        filters: Iterable[FilterRuntime],
        draw_counts: Iterable[int],
        base_mask: np.ndarray,
        dataset_len: int,
        seed: int,
    ) -> None:
        self.filters: List[FilterRuntime] = list(filters)
        if not self.filters:
            raise ValueError("rejection sampler requires at least one filter")
        self.draw_counts: List[int] = [int(max(0, c)) for c in draw_counts]
        self.total_samples = int(sum(self.draw_counts))
        self.base_mask = base_mask.astype(bool, copy=True)
        self.dataset_len = int(dataset_len)
        self.seed = int(seed)
        self._epoch = 0

    def __len__(self) -> int:
        return self.total_samples

    def __iter__(self) -> Iterator[int]:
        rng = np.random.default_rng(self.seed + self._epoch)
        epoch = self._epoch
        self._epoch += 1

        mask = self.base_mask.copy()
        selections: List[np.ndarray] = []
        log_parts: List[str] = []

        for runtime, desired in zip(self.filters, self.draw_counts):
            if desired <= 0:
                continue
            if runtime.positions is not None:
                pool = runtime.positions
                if desired >= pool.size:
                    chosen = pool.astype(np.int64, copy=True)
                else:
                    chosen = rng.choice(pool, size=desired, replace=False)
                mask[chosen] = True
                selections.append(chosen)
                log_parts.append(
                    f"{runtime.id}: take={desired:,} unique={chosen.size:,} dup=0"
                )
            else:
                chosen = _sample_unseen(mask, desired, rng, self.dataset_len)
                selections.append(chosen)
                log_parts.append(
                    f"{runtime.id}: take={desired:,} unique={chosen.size:,} dup=0"
                )

        if log_parts:
            LOGGER.info("[rejection] epoch %s distribution: %s", epoch, "; ".join(log_parts))

        if not selections:
            return iter(())

        merged = np.concatenate(selections).astype(np.int64, copy=False)
        rng.shuffle(merged)

        def _generator() -> Iterator[int]:
            for value in merged:
                yield int(value)

        return _generator()


def _collect_filter_indices(
    annotation_dir: str, filters: Sequence[RejectionFilterConfig]
) -> tuple[int, Dict[str, np.ndarray]]:
    shard_paths = _sorted_annotation_shards(annotation_dir)
    if not shard_paths:
        raise FileNotFoundError(f"no annotation shards found in {annotation_dir}")

    offsets: Dict[str, List[np.ndarray]] = {cfg.id: [] for cfg in filters}
    total_rows = 0

    for shard_path in shard_paths:
        shard = np.load(shard_path, mmap_mode="r", allow_pickle=False)
        shard_len = shard.shape[0]
        for cfg in filters:
            local_idx = _apply_filter(cfg.name, shard)
            if local_idx.size:
                offsets[cfg.id].append(local_idx + total_rows)
        total_rows += shard_len

    resolved = {
        cfg.id: (
            np.concatenate(offsets[cfg.id]).astype(np.int64, copy=False)
            if offsets[cfg.id]
            else np.empty((0,), dtype=np.int64)
        )
        for cfg in filters
    }
    return total_rows, resolved


def _sorted_annotation_shards(annotation_dir: str) -> List[Path]:
    root = Path(annotation_dir)
    shards = sorted(root.glob("annotations-*.npy"))
    if not shards:
        single = root / "annotations.npy"
        if single.is_file():
            shards = [single]
    return shards


def _apply_filter(name: str, shard: np.ndarray) -> np.ndarray:
    if name == "passthrough":
        return np.arange(shard.shape[0], dtype=np.int64)
    if name == "student_wrong_p1":
        teacher_move = shard["teacher_move"].astype(np.int64, copy=False)
        argmax_head = shard["argmax_head"].astype(np.int64, copy=False)
        policy_mask = shard["policy_kind_mask"].astype(np.int32, copy=False)
        valid = (teacher_move >= 0) & (teacher_move < 4) & ((policy_mask & POLICY_P1_BIT) != 0)
        mismatched = valid & (argmax_head != teacher_move)
        return np.nonzero(mismatched)[0].astype(np.int64, copy=False)
    raise ValueError(f"unsupported rejection filter name '{name}'")


def _sample_unseen(mask: np.ndarray, count: int, rng: np.random.Generator, dataset_len: int) -> np.ndarray:
    if count <= 0:
        return np.empty((0,), dtype=np.int64)
    result = np.empty(count, dtype=np.int64)
    filled = 0
    while filled < count:
        batch = max(count - filled, 1024)
        candidates = rng.integers(0, dataset_len, size=batch * 2, dtype=np.int64)
        available = candidates[~mask[candidates]]
        if available.size == 0:
            continue
        unique = np.unique(available)
        take = min(unique.size, count - filled)
        chosen = unique[:take]
        mask[chosen] = True
        result[filled : filled + take] = chosen
        filled += take
    return result
