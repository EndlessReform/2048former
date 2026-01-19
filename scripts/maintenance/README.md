# Maintenance Scripts

Small utilities for keeping runs and artifacts tidy.

## Cleanup empty checkpoints

Remove checkpoint run folders that contain no `.safetensors` or `.pt` files.

```sh
uv run --locked scripts/maintenance/cleanup_empty_checkpoints.py --dry-run
uv run --locked scripts/maintenance/cleanup_empty_checkpoints.py checkpoints/ --yes
```

## Wipe tmpfs shard caches

Remove tmpfs decompression caches for compressed shards (useful after killed runs).

```sh
uv run --locked scripts/maintenance/wipe_tmpfs_shards.py --dataset-dir /path/to/dataset --yes
uv run --locked scripts/maintenance/wipe_tmpfs_shards.py --all
```
