# train_2048

This package contains tools for training a 2048 AI model.

## Data Loading

Large datasets (100GB+) can cause OS page thrashing when randomising across shards. The training config exposes a shard-aware path that keeps one shard resident in RAM at a time:

- Set `dataset.shard_locality = true` to traverse shards sequentially while still sampling randomly within each shard.
- Optionally cap per-shard draws via `dataset.shard_locality_block_size` (default is the full shard).
- Enable `dataset.shard_cache_in_memory = true` (with `dataset.shard_cache_keep_shards`) to materialise the active shard into RAM while keeping only a small number cached.
- For compressed `.npy.zst` shards with `dataset.mmap_mode = true`, the loader will decompress into tmpfs (`/dev/shm` or `/tmp`) and mmap the decompressed file so multiple workers share the same pages.
- When shard-locality + tmpfs-mmap are active and `dataset.val_num_steps` is set, validation samples are taken from the first shard only and cached in RAM. This avoids tmpfs overcommit during val (no extra shard decompressions).

## Augmentation

Training-time board augmentation is configured under `dataset.rotation_augment` and `dataset.flip_augment`. It applies rotations and/or flips to boards and permutes targets (UDLR) to match.

- **Order:** Rotation is applied first, then flip.
- **Scope:** Applies to training collate only (no dataset expansion on disk).
- **Macroxue Support:** Supported for v2 (recomputes `board_eval` for rotated/flipped boards). Not supported for v1 (will raise assertion).

```toml
[dataset.rotation_augment]
mode = "random_k"     # Options: "none", "random_k" (0, 90, 180, 270 deg)
allow_noop = true

[dataset.flip_augment]
mode = "random_axis"  # Options: "none", "random_axis" (UD, LR)
allow_noop = true
```

### UDLR Permutation Reference

UDLR indices: Up=0, Down=1, Left=2, Right=3.

**Rotation:**
- **90° CW:** `perm=[2, 3, 1, 0]`, `move_dir=[3, 2, 0, 1]`
- **180°:** `perm=[1, 0, 3, 2]`, `move_dir=[1, 0, 3, 2]`
- **270° CW:** `perm=[3, 2, 0, 1]`, `move_dir=[2, 3, 1, 0]`

**Flip:**
- **Left-Right:** `perm=[0, 1, 3, 2]`, `move_dir=[0, 1, 3, 2]`
- **Up-Down:** `perm=[1, 0, 2, 3]`, `move_dir=[1, 0, 2, 3]`

## AMP / Precision

Autocast configuration lives under `[amp]` in the training config:

- `amp.autocast_type = "bf16"` (default), `"fp32"` (disable autocast), or `"mxfp8"` for TransformerEngine FP8.
- `mxfp8` requires CUDA + TransformerEngine and only resumes from TE-tagged checkpoints.

## Tokenization

Macroxue tokenization lives in `core_2048.tokenization.macroxue` and is shared across training and inference. It converts expectimax-derived action values into compact categorical targets suitable for transformer training.

### Tokenization Scheme (Advantage v2)

The current scheme is advantage-based and operates directly on winner-relative deltas:

1. **Compute advantages**: For each valuation type, subtract a baseline (board_eval for search; winner EV for tuple tables) to get per-branch advantages.
2. **Token generation**:
   - **ILLEGAL**: moves that do not change the board state.
   - **FAILURE**: search branches below the configured cutoff (search only).
   - **BIN_k**: winner-relative disadvantages bucketed into `num_bins` quantile edges (most negative → near-zero).
   - **WINNER**: the selected move.

### Tokenizer Specification

The tokenizer uses a JSON spec with per-valuation `bin_edges`, a `failure_cutoff` for search, and a fixed `vocab_order`. The default location is `out/tokenizer.json`.

### Generating the Tokenizer

To generate a tokenizer spec from a packed dataset:

```bash
uv run tokenizer-macroxue --help
```

For example:

```bash
uv run tokenizer-macroxue datasets/packed/macroxue --output out/tokenizer.json
```

## Output Head Format (Macroxue)

When training with `target.mode = "macroxue_tokens"`, each of the four per‑move heads (UDLR) predicts a categorical distribution over classes derived from the tokenizer:

- Head order: UDLR everywhere (data, training, inference).
- Number of classes per head: `n_classes = len(vocab_order)` from `tokenizer.json`.
- Class indices per head:
  - `0` — ILLEGAL
  - `1` — FAILURE (search only; tuple tables can still emit FAILURE for non-positive EVs)
  - `2 .. (1 + num_bins)` — disadvantage bins (more negative → lower index)
  - `2 + num_bins` — WINNER

Notes:

- Each head uses cross-entropy against these per-branch class targets.
- Inference expects the WINNER class to be the final index; this matches server/client conventions.

Quick usage:

- Train (example): `uv run python main.py --config config/pretraining/v2/10m-100k-ablation.toml`
- Profile steps 2-10 with torch profiler (trace saved under the run `profiles/` dir): `uv run python main.py --config config/pretraining/v2/10m-100k-ablation.toml --device cuda --profile --profile-start 2 --profile-end 10`
- The tokenizer path is configured at `dataset.tokenizer_path` and must point to a `tokenizer.json` generated as above.
