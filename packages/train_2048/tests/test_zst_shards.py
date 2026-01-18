from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest
import zstandard

from train_2048.dataloader.shard_loader import ShardLoader
from train_2048.dataloader.steps import StepsDataset


def _write_compressed_npy(path: Path, array: np.ndarray) -> None:
    buffer = io.BytesIO()
    np.save(buffer, array)
    cctx = zstandard.ZstdCompressor(level=3)
    path.write_bytes(cctx.compress(buffer.getvalue()))


def test_shard_loader_reads_zst(tmp_path: Path) -> None:
    data = np.arange(12, dtype=np.int32)
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    shard_path = dataset_dir / "steps-00000.npy.zst"
    _write_compressed_npy(shard_path, data)

    loader = ShardLoader(str(dataset_dir))
    assert loader.total_steps == data.shape[0]

    loaded = loader.load_shard(0)
    assert np.array_equal(loaded, data)


def test_steps_dataset_rejects_zst(tmp_path: Path) -> None:
    data = np.arange(10, dtype=np.int16)
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    shard_path = dataset_dir / "steps-00000.npy.zst"
    _write_compressed_npy(shard_path, data)

    with pytest.raises(ValueError, match="Compressed shards are only supported"):
        StepsDataset(str(dataset_dir))


def test_mmap_rejected_for_zst(tmp_path: Path) -> None:
    data = np.arange(4, dtype=np.int32)
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    shard_path = dataset_dir / "steps.npy.zst"
    _write_compressed_npy(shard_path, data)

    with pytest.raises(ValueError, match="mmap_mode is not supported"):
        ShardLoader(str(dataset_dir), mmap_mode=True)

    with pytest.raises(ValueError, match="Compressed steps.npy.zst is only supported"):
        StepsDataset(str(dataset_dir), mmap_mode=True)
