from __future__ import annotations

import io
import sqlite3
from pathlib import Path

import numpy as np
import zstandard

from train_2048.dataloader.collate import make_collate_steps_worker_safe
from train_2048.dataloader.shard_loader import ShardLoader


def _write_zst_npy(path: Path, array: np.ndarray) -> None:
    buf = io.BytesIO()
    np.save(buf, array)
    cctx = zstandard.ZstdCompressor()
    path.write_bytes(cctx.compress(buf.getvalue()))


def _pack_exps_to_u64(exps: np.ndarray) -> np.uint64:
    packed = np.uint64(0)
    for i, value in enumerate(exps.tolist()):
        shift = np.uint64((15 - i) * 4)
        packed |= np.uint64(int(value) & 0xF) << shift
    return packed


def _create_metadata(db_path: Path, run_tiles: dict[int, int]) -> None:
    with sqlite3.connect(str(db_path)) as conn:
        conn.execute(
            "CREATE TABLE runs (id INTEGER PRIMARY KEY, steps INT, highest_tile INT)"
        )
        for run_id, tile in run_tiles.items():
            conn.execute(
                "INSERT INTO runs (id, steps, highest_tile) VALUES (?, ?, ?)",
                (int(run_id), 1, int(tile)),
            )


def test_highest_tile_targets_with_sharded_mmap(tmp_path: Path) -> None:
    dtype = np.dtype(
        [
            ("run_id", "<u4"),
            ("board", "<u8"),
            ("branch_evs", "<f4", (4,)),
            ("ev_legal", "<u1"),
            ("move_dir", "<u1"),
        ],
        align=True,
    )
    exps = np.zeros(16, dtype=np.uint8)
    packed = _pack_exps_to_u64(exps)

    shard0 = np.zeros(3, dtype=dtype)
    shard0["run_id"] = np.array([0, 1, 1], dtype=np.uint32)
    shard0["board"] = packed
    shard0["branch_evs"] = np.zeros((3, 4), dtype=np.float32)
    shard0["ev_legal"] = np.uint8(0b1111)
    shard0["move_dir"] = np.uint8(0)

    shard1 = np.zeros(2, dtype=dtype)
    shard1["run_id"] = np.array([2, 3], dtype=np.uint32)
    shard1["board"] = packed
    shard1["branch_evs"] = np.zeros((2, 4), dtype=np.float32)
    shard1["ev_legal"] = np.uint8(0b1111)
    shard1["move_dir"] = np.uint8(0)

    _write_zst_npy(tmp_path / "steps-00000.npy.zst", shard0)
    _write_zst_npy(tmp_path / "steps-00001.npy.zst", shard1)

    _create_metadata(
        tmp_path / "metadata.db",
        {0: 1024, 1: 2048, 2: 4096, 3: 8192},
    )

    loader = ShardLoader(
        str(tmp_path),
        mmap_mode=True,
        cache_shards=True,
        decompress_dir=str(tmp_path),
    )
    arr0 = loader.load_shard(0)
    assert isinstance(arr0, np.memmap)

    collate = make_collate_steps_worker_safe(
        str(tmp_path),
        "hard_move",
        ev_tokenizer=None,
        shard_loader=loader,
        include_highest_tile=True,
    )

    out = collate([0, 2, 4])
    assert out["highest_tile"].tolist() == [1024, 2048, 8192]
