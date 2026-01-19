"""Wipe tmpfs shard-decompression caches left by killed runs."""
from __future__ import annotations

import argparse
import hashlib
import os
import shutil
from pathlib import Path
from typing import Iterable


CACHE_PREFIX = "train_2048_shards_"
DEFAULT_ROOTS = ("/dev/shm", "/tmp")


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _dataset_digest(dataset_dir: str) -> str:
    resolved = Path(dataset_dir).resolve()
    return hashlib.sha1(str(resolved).encode("utf-8")).hexdigest()[:16]


def _iter_cache_dirs(root: Path) -> Iterable[Path]:
    if not root.exists():
        return []
    return [p for p in root.glob(f"{CACHE_PREFIX}*") if p.is_dir()]


def _has_live_leases(cache_dir: Path) -> bool:
    for lease in cache_dir.glob("*.lease.*"):
        try:
            pid = int(lease.name.rsplit(".lease.", 1)[-1])
        except ValueError:
            continue
        if _pid_alive(pid):
            return True
    return False


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Remove tmpfs shard-decompression caches created by train_2048.",
    )
    parser.add_argument(
        "--dataset-dir",
        help="Dataset directory used for shard loading (targets its tmpfs cache).",
    )
    parser.add_argument(
        "--roots",
        nargs="*",
        default=list(DEFAULT_ROOTS),
        help="Tmpfs roots to search (default: /dev/shm /tmp).",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help=f"Remove all {CACHE_PREFIX}* directories in the roots.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete even if live lease files are detected.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Actually delete (otherwise dry-run).",
    )
    args = parser.parse_args()

    if not args.all and not args.dataset_dir:
        parser.error("Provide --dataset-dir or --all.")

    roots = [Path(p) for p in args.roots]
    targets: list[Path] = []

    if args.dataset_dir:
        digest = _dataset_digest(args.dataset_dir)
        name = f"{CACHE_PREFIX}{digest}"
        for root in roots:
            candidate = root / name
            if candidate.is_dir():
                targets.append(candidate)

    if args.all:
        for root in roots:
            targets.extend(_iter_cache_dirs(root))

    if not targets:
        print("[wipe_tmpfs_shards] No cache directories found.")
        return 0

    seen: set[Path] = set()
    for target in targets:
        if target in seen:
            continue
        seen.add(target)
        live = _has_live_leases(target)
        if live and not args.force:
            print(f"[wipe_tmpfs_shards] SKIP (live leases): {target}")
            continue
        if args.yes:
            shutil.rmtree(target, ignore_errors=True)
            print(f"[wipe_tmpfs_shards] deleted {target}")
        else:
            print(f"[wipe_tmpfs_shards] dry-run: would delete {target}")

    if not args.yes:
        print("[wipe_tmpfs_shards] dry-run only; pass --yes to delete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
