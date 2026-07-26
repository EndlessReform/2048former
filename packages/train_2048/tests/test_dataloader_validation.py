from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pytest
import torch

from train_2048.config import DatasetConfig
from train_2048.dataloader.steps_v2 import build_steps_dataloaders


STEP_DTYPE = np.dtype(
    [
        ("run_id", "<u4"),
        ("step_index", "<u4"),
        ("board", "<u8"),
        ("branch_evs", "<f4", (4,)),
        ("ev_legal", "u1"),
        ("move_dir", "u1"),
    ],
    align=True,
)


def _write_tiny_pool(dataset_dir: Path) -> np.ndarray:
    dataset_dir.mkdir()
    all_rows = []
    with sqlite3.connect(dataset_dir / "metadata.db") as conn:
        conn.execute("CREATE TABLE runs (id INTEGER PRIMARY KEY, steps INTEGER NOT NULL)")
        for run_id in range(4):
            conn.execute("INSERT INTO runs (id, steps) VALUES (?, ?)", (run_id, 3))

    for shard_idx, run_ids in enumerate(((0, 1), (2, 3))):
        rows = np.zeros(6, dtype=STEP_DTYPE)
        for row_idx in range(6):
            run_id = run_ids[row_idx // 3]
            rows[row_idx]["run_id"] = run_id
            rows[row_idx]["step_index"] = row_idx % 3
            rows[row_idx]["board"] = shard_idx * 100 + row_idx + 1
            rows[row_idx]["branch_evs"] = np.arange(4, dtype=np.float32)
            rows[row_idx]["ev_legal"] = 0b1111
            rows[row_idx]["move_dir"] = run_id
        np.save(dataset_dir / f"steps-{shard_idx:05d}.npy", rows)
        all_rows.append(rows)
    return np.concatenate(all_rows)


def _augmentation_config(mode: str, seed: int | None) -> object:
    return type(
        "AugmentationConfig",
        (),
        {"mode": mode, "seed": seed, "allow_noop": False},
    )()


def _build(dataset_dir: Path):
    return build_steps_dataloaders(
        str(dataset_dir),
        "hard_move",
        batch_size=2,
        num_epochs=1,
        val_num_steps=2,
        val_run_sql="SELECT id FROM runs WHERE id IN (?, ?)",
        val_sql_params=(1, 3),
        num_workers_train=0,
        shard_locality=True,
        rotation_augment=_augmentation_config("random_k", 11),
        flip_augment=_augmentation_config("random_axis", 12),
        seed=17,
    )


def test_validation_is_disjoint_filtered_and_deterministic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_dir = tmp_path / "pool"
    physical_rows = _write_tiny_pool(dataset_dir)

    # Validation must not consult either random augmentation path.
    def _unexpected_augmentation(*args, **kwargs):
        raise AssertionError("validation augmentation was invoked")

    monkeypatch.setattr(
        "train_2048.dataloader.collate.sample_rotation_k",
        _unexpected_augmentation,
    )
    monkeypatch.setattr(
        "train_2048.dataloader.collate.sample_flip_axis",
        _unexpected_augmentation,
    )

    dl_train, dl_val, _, metadata = _build(dataset_dir)
    assert dl_val is not None
    train_ids = set(metadata["train_run_ids"].tolist())
    val_ids = set(metadata["val_run_ids"].tolist())
    assert train_ids == {0, 2}
    assert val_ids == {1, 3}
    assert train_ids.isdisjoint(val_ids)

    sampled_train_indices = np.array(
        [ref.global_index for ref in dl_train.sampler],
        dtype=np.int64,
    )
    assert set(physical_rows["run_id"][sampled_train_indices].tolist()) == train_ids

    first_pass = list(dl_val)
    first_targets = np.concatenate([batch["move_targets"].numpy() for batch in first_pass])
    first_tokens = np.concatenate([batch["tokens"].numpy() for batch in first_pass])
    assert set(first_targets.tolist()).issubset(val_ids)
    assert metadata["validation"] == {
        "samples": 4,
        "source_shard": 0,
        "eligible_rows_in_source_shard": 3,
        "seed": 18,
        "augmentation": False,
    }

    _, dl_val_again, _, _ = _build(dataset_dir)
    assert dl_val_again is not None
    second_pass = list(dl_val_again)
    second_targets = np.concatenate(
        [batch["move_targets"].numpy() for batch in second_pass]
    )
    second_tokens = np.concatenate([batch["tokens"].numpy() for batch in second_pass])
    assert np.array_equal(first_targets, second_targets)
    assert np.array_equal(first_tokens, second_tokens)


def test_validation_split_requires_a_bounded_sample(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "pool"
    _write_tiny_pool(dataset_dir)

    with pytest.raises(ValueError, match="requires val_num_steps or val_steps_pct"):
        build_steps_dataloaders(
            str(dataset_dir),
            "hard_move",
            batch_size=2,
            num_epochs=1,
            val_run_pct=0.25,
            num_workers_train=0,
            shard_locality=True,
        )


def test_run_split_rejects_sampler_that_cannot_filter(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "pool"
    _write_tiny_pool(dataset_dir)

    with pytest.raises(ValueError, match="requires shard_locality=true"):
        build_steps_dataloaders(
            str(dataset_dir),
            "hard_move",
            batch_size=2,
            train_num_steps=2,
            val_num_steps=1,
            val_run_pct=0.25,
            num_workers_train=0,
            shard_locality=False,
        )


def test_loader_iteration_does_not_advance_model_rng(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "pool"
    _write_tiny_pool(dataset_dir)

    torch.manual_seed(1234)
    expected = torch.rand(8)
    torch.manual_seed(1234)
    dl_train, dl_val, _, _ = _build(dataset_dir)
    next(iter(dl_train))
    assert dl_val is not None
    next(iter(dl_val))
    actual = torch.rand(8)

    assert torch.equal(actual, expected)


def test_dataset_workers_must_be_non_negative() -> None:
    assert DatasetConfig(num_workers_train=0).num_workers_train == 0
    with pytest.raises(ValueError, match="num_workers_train must be >= 0"):
        DatasetConfig(num_workers_train=-1)


def _build_prefetch_loader(dataset_dir: Path, resume_cursor: dict | None = None):
    dl_train, _, _, _ = build_steps_dataloaders(
        str(dataset_dir),
        "hard_move",
        batch_size=2,
        num_epochs=1,
        num_workers_train=2,
        mmap_mode=True,
        shard_locality=True,
        shard_cache_in_memory=False,
        resume_data_cursor=resume_cursor,
        rotation_augment=_augmentation_config("random_k", None),
        flip_augment=_augmentation_config("random_axis", None),
        seed=29,
    )
    return dl_train


def test_committed_cursor_ignores_worker_prefetch(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "pool"
    _write_tiny_pool(dataset_dir)

    uninterrupted_loader = _build_prefetch_loader(dataset_dir)
    uninterrupted = list(uninterrupted_loader)
    expected_tokens = np.concatenate(
        [batch["tokens"].numpy() for batch in uninterrupted],
    )

    interrupted_loader = _build_prefetch_loader(dataset_dir)
    interrupted_iter = iter(interrupted_loader)
    next(interrupted_iter)
    committed_batch = next(interrupted_iter)
    committed_cursor = committed_batch["_data_cursor"]

    resumed_loader = _build_prefetch_loader(dataset_dir, committed_cursor)
    resumed = list(resumed_loader)
    resumed_tokens = np.concatenate([batch["tokens"].numpy() for batch in resumed])
    assert np.array_equal(resumed_tokens, expected_tokens[4:])
