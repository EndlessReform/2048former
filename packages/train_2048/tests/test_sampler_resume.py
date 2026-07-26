from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from train_2048.dataloader.dataset import SampleRef
from train_2048.dataloader.samplers import ShardPoolSampler
from train_2048.dataloader.shard_loader import ShardLoader


def _write_shards(dataset_dir: Path) -> None:
    dataset_dir.mkdir()
    dtype = np.dtype([("run_id", "<u4"), ("value", "<u4")])
    for shard_idx in range(3):
        rows = np.zeros(8, dtype=dtype)
        rows["run_id"] = np.repeat(
            np.array([shard_idx * 2, shard_idx * 2 + 1], dtype=np.uint32),
            4,
        )
        rows["value"] = np.arange(shard_idx * 8, shard_idx * 8 + 8)
        np.save(dataset_dir / f"steps-{shard_idx:05d}.npy", rows)


def _make_sampler(
    dataset_dir: Path,
    *,
    resume_cursor: dict | None = None,
) -> ShardPoolSampler:
    loader = ShardLoader(str(dataset_dir), cache_shards=True)
    selected_runs = np.array([0, 2, 4], dtype=np.uint32)
    return ShardPoolSampler(
        loader,
        num_epochs=2,
        seed=123,
        run_ids=selected_runs,
        total_steps=12,
        resume_cursor=resume_cursor,
    )


def test_resume_uses_consumed_cursor_not_prefetched_producer_position(
    tmp_path: Path,
) -> None:
    dataset_dir = tmp_path / "pool"
    _write_shards(dataset_dir)

    uninterrupted = list(_make_sampler(dataset_dir))
    assert all(isinstance(ref, SampleRef) for ref in uninterrupted)

    producer = iter(_make_sampler(dataset_dir))
    prefetched = [next(producer) for _ in range(10)]
    consumed_count = 3
    committed_cursor = prefetched[consumed_count - 1].next_cursor.as_dict()

    resumed = list(_make_sampler(dataset_dir, resume_cursor=committed_cursor))
    expected_indices = [ref.global_index for ref in uninterrupted[consumed_count:]]
    resumed_indices = [ref.global_index for ref in resumed]
    assert resumed_indices == expected_indices


def test_resume_cursor_crosses_shard_and_epoch_boundaries(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "pool"
    _write_shards(dataset_dir)
    uninterrupted = list(_make_sampler(dataset_dir))

    for consumed_count in (4, 8, 12, 16, 23):
        cursor = uninterrupted[consumed_count - 1].next_cursor.as_dict()
        resumed = list(_make_sampler(dataset_dir, resume_cursor=cursor))
        assert [ref.global_index for ref in resumed] == [
            ref.global_index for ref in uninterrupted[consumed_count:]
        ]


def test_resume_cursor_rejects_a_different_training_seed(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "pool"
    _write_shards(dataset_dir)
    cursor = list(_make_sampler(dataset_dir))[0].next_cursor.as_dict()
    cursor["seed"] = 999

    with pytest.raises(ValueError, match="seed does not match"):
        _make_sampler(dataset_dir, resume_cursor=cursor)
