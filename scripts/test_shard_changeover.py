#!/usr/bin/env python3
"""Test shard-to-shard changeover using tmpfs mmap cache."""

from __future__ import annotations

import argparse
import os
import sys
import time
from multiprocessing import get_context
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "packages/train_2048/src"))

from train_2048.config import load_config
from train_2048.dataloader.shard_loader import ShardLoader, InMemoryShardPool, _read_npy_header


def _is_tmpfs(path: str) -> bool:
    try:
        with open("/proc/mounts", "r", encoding="utf-8") as handle:
            for line in handle:
                parts = line.split()
                if len(parts) >= 3 and parts[1] == path:
                    return parts[2] == "tmpfs"
    except OSError:
        return False
    return False


def _free_bytes(path: str) -> int:
    try:
        stat = os.statvfs(path)
        return stat.f_bavail * stat.f_frsize
    except OSError:
        return 0


def _pick_tmpfs(min_bytes: int) -> Optional[str]:
    for candidate in ("/dev/shm", "/tmp"):
        if _is_tmpfs(candidate) and _free_bytes(candidate) >= min_bytes:
            return candidate
    return None


def _list_cache_files(decompress_dir: str) -> list[str]:
    root = Path(decompress_dir)
    matches = []
    for cache_dir in root.glob("train_2048_shards_*"):
        for entry in cache_dir.iterdir():
            if entry.name.endswith(".lock"):
                continue
            matches.append(entry.name)
    return sorted(matches)


def _worker(
    rank: int,
    dataset_dir: str,
    decompress_dir: str,
    barrier_load0,
    barrier_load1,
    barrier_done,
) -> None:
    loader = ShardLoader(
        dataset_dir,
        mmap_mode=True,
        cache_shards=True,
        cache_keep_shards=1,
        decompress_dir=decompress_dir,
        decompress_cleanup=True,
    )
    pool = InMemoryShardPool(loader)
    rng = np.random.default_rng(1234 + rank)

    # Load shard 0 and fetch a small batch via loader (collate-style path).
    pool.load_shard_for_sampling(0)
    shard0 = pool.current_shard
    idx0 = rng.integers(0, len(shard0), size=256, dtype=np.int64)
    loader.get_rows(idx0 + loader.shards[0].offset)
    barrier_load0.wait()

    # Rotate to shard 1 and fetch another small batch.
    pool.load_shard_for_sampling(1)
    shard1 = pool.current_shard
    idx1 = rng.integers(0, len(shard1), size=256, dtype=np.int64)
    loader.get_rows(idx1 + loader.shards[1].offset)
    barrier_load1.wait()

    # Drop the current shard to release leases.
    loader.unload_shard(1)
    barrier_done.wait()


def main() -> None:
    parser = argparse.ArgumentParser(description="Test shard changeover with tmpfs mmap cache.")
    parser.add_argument(
        "--config",
        default="config/pretraining/v2/ablation/50m-100k-attn-sink-expt.toml",
        help="Path to training config TOML",
    )
    parser.add_argument("--workers", type=int, default=2, help="Number of worker processes")
    args = parser.parse_args()

    cfg = load_config(args.config)
    ds_dir = cfg.dataset.resolved_dataset_dir()
    if not cfg.dataset.mmap_mode:
        raise RuntimeError("Config has dataset.mmap_mode=false; set true to test tmpfs mmap path.")
    ds_path = Path(ds_dir)
    shard_paths = sorted(list(ds_path.glob("steps-*.npy.zst")))
    if not shard_paths:
        raise FileNotFoundError(f"No .zst shards found in {ds_dir}")

    # Estimate largest shard to ensure tmpfs has room.
    max_shard_bytes = 0
    for shard in shard_paths[:2]:
        shape, dtype = _read_npy_header(shard)
        num_bytes = int(np.prod(shape)) * int(dtype.itemsize)
        max_shard_bytes = max(max_shard_bytes, num_bytes)

    decompress_dir = _pick_tmpfs(max_shard_bytes)
    if decompress_dir is None:
        raise RuntimeError("No tmpfs with enough free space found (/dev/shm or /tmp)")

    print(f"[test] Dataset: {ds_dir}")
    print(f"[test] Using tmpfs cache: {decompress_dir}")
    print(f"[test] Workers: {args.workers}")

    ctx = get_context("spawn")
    barrier_load0 = ctx.Barrier(args.workers + 1)
    barrier_load1 = ctx.Barrier(args.workers + 1)
    barrier_done = ctx.Barrier(args.workers + 1)

    procs = []
    for rank in range(args.workers):
        proc = ctx.Process(
            target=_worker,
            args=(rank, ds_dir, decompress_dir, barrier_load0, barrier_load1, barrier_done),
        )
        proc.start()
        procs.append(proc)

    barrier_load0.wait()
    print("[test] After shard 0 load:", _list_cache_files(decompress_dir))

    barrier_load1.wait()
    print("[test] After shard 1 load:", _list_cache_files(decompress_dir))

    barrier_done.wait()
    time.sleep(0.2)
    print("[test] After unload:", _list_cache_files(decompress_dir))

    for proc in procs:
        proc.join(timeout=5)
        if proc.exitcode != 0:
            raise RuntimeError(f"Worker {proc.pid} exited with {proc.exitcode}")

    print("[test] OK")


if __name__ == "__main__":
    main()
