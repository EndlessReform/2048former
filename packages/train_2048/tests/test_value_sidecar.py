import sqlite3
from pathlib import Path

import numpy as np
import pytest

from train_2048.dataloader.shard_loader import ShardLoader


STEP_DTYPE = np.dtype([("run_id", "<u4"), ("step_index", "<u4")])
VALUE_DTYPE = np.dtype(
    [
        ("run_id", "<u4"),
        ("step_index", "<u4"),
        ("reward", "<f4"),
        ("reward_scaled", "<f4"),
        ("return_raw", "<f4"),
        ("return_scaled", "<f4"),
    ]
)


def _write_metadata(db_path: Path, runs: list[tuple[int, int]]) -> None:
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("CREATE TABLE runs (id INTEGER PRIMARY KEY, steps INTEGER)")
        conn.executemany("INSERT INTO runs (id, steps) VALUES (?, ?)", runs)
        conn.commit()
    finally:
        conn.close()


def test_value_sidecar_alignment_sharded(tmp_path: Path) -> None:
    """Ensure sharded values-*.npy aligns with steps-*.npy and can be fetched by index."""
    ds_dir = tmp_path / "aligned"
    ds_dir.mkdir()
    _write_metadata(ds_dir / "metadata.db", [(0, 2), (1, 2)])

    steps0 = np.array([(0, 0), (0, 1)], dtype=STEP_DTYPE)
    steps1 = np.array([(1, 0), (1, 1)], dtype=STEP_DTYPE)
    np.save(ds_dir / "steps-000.npy", steps0)
    np.save(ds_dir / "steps-001.npy", steps1)

    vals0 = np.array(
        [
            (0, 0, 0.0, 0.0, 1.0, 1.0),
            (0, 1, 0.5, 0.5, 2.0, 2.0),
        ],
        dtype=VALUE_DTYPE,
    )
    vals1 = np.array(
        [
            (1, 0, 1.0, 1.0, 3.0, 3.0),
            (1, 1, 2.0, 2.0, 4.0, 4.0),
        ],
        dtype=VALUE_DTYPE,
    )
    np.save(ds_dir / "values-000.npy", vals0)
    np.save(ds_dir / "values-001.npy", vals1)

    loader = ShardLoader(str(ds_dir), mmap_mode=True, value_sidecar=True, expected_total_steps=4)
    idxs = np.array([0, 3], dtype=np.int64)
    step_rows = loader.get_rows(idxs)
    value_rows = loader.get_value_rows(idxs)

    assert np.array_equal(step_rows["run_id"], value_rows["run_id"])
    assert np.array_equal(step_rows["step_index"], value_rows["step_index"])
    assert value_rows["return_scaled"].tolist() == [1.0, 4.0]


def test_value_sidecar_row_mismatch_raises(tmp_path: Path) -> None:
    """Mismatch between steps and values shards should fail fast."""
    ds_dir = tmp_path / "mismatch"
    ds_dir.mkdir()
    _write_metadata(ds_dir / "metadata.db", [(0, 2)])

    steps0 = np.array([(0, 0), (0, 1)], dtype=STEP_DTYPE)
    np.save(ds_dir / "steps-000.npy", steps0)

    # Only one value row instead of two -> should raise during loader init.
    vals0 = np.array([(0, 0, 0.0, 0.0, 1.0, 1.0)], dtype=VALUE_DTYPE)
    np.save(ds_dir / "values-000.npy", vals0)

    with pytest.raises(ValueError, match="has 1 rows"):
        ShardLoader(str(ds_dir), mmap_mode=True, value_sidecar=True, expected_total_steps=2)


def test_value_sidecar_metadata_mismatch_raises(tmp_path: Path) -> None:
    """Metadata total steps must match the paired sidecar rows."""
    ds_dir = tmp_path / "meta_mismatch"
    ds_dir.mkdir()
    # Metadata says 3 steps but we will only write 2.
    _write_metadata(ds_dir / "metadata.db", [(0, 3)])

    steps0 = np.array([(0, 0), (0, 1)], dtype=STEP_DTYPE)
    np.save(ds_dir / "steps-000.npy", steps0)

    vals0 = np.array(
        [
            (0, 0, 0.0, 0.0, 1.0, 1.0),
            (0, 1, 0.5, 0.5, 2.0, 2.0),
        ],
        dtype=VALUE_DTYPE,
    )
    np.save(ds_dir / "values-000.npy", vals0)

    with pytest.raises(ValueError, match="metadata steps"):
        ShardLoader(str(ds_dir), mmap_mode=True, value_sidecar=True, expected_total_steps=3)
