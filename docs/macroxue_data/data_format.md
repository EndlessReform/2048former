# Macroxue self-play data

For pretraining we use self-play board states from Macroxue's hybrid expectimax 2048 implementation (from a [personal fork](https://github.com/EndlessReform/macroxue-expectimax-2048) instrumented for data capture). Since the original repo is GPL-v3, we keep it at arm's-length and only train on outputs, as shown here.

## Core algorithm

TODO: fill in how this differs from vanilla expectimax (not the focus of this project).

## Raw output (as captured from the Macroxue fork)

A folder of games might look like this:

```
example-folder
├── d6_10g_v1 # Folder names and file naming conventions are arbitrary; don't rely on them.
│   ├── depth06_worker00_seed0272350805_game000000.jsonl.gz
│   ├── depth06_worker00_seed0272350805_game000000.meta.json
│   ├── ...
│   ├── depth06_worker09_seed0272350814_game000000.jsonl.gz
│   └── depth06_worker09_seed0272350814_game000000.meta.json
└── d7_10g_v1
    ├── depth07_worker00_seed1273930896_game000000.jsonl.gz
    ├── depth07_worker00_seed1273930896_game000000.meta.json
    ├── ...
    ├── depth07_worker09_seed1273930905_game000000.jsonl.gz
    └── depth07_worker09_seed1273930905_game000000.meta.json
```

Each run (full game) produces two files:

- `*.meta.json` or `*.meta.json.gz`: per-run metadata (uncompressed or gzipped)
- `*.jsonl.gz`: per-step records (always gzipped JSONL)

### Metadata sidecar (`*.meta.json[.gz]`)

The packer only reads these fields (others are ignored):

```json
{
  "seed": 272350805,
  "num_moves": 27885,
  "score": 795564,
  "max_tile": 32768,
  "max_rank": 15
}
```

Notes:
- `max_rank` is optional; if absent, steps fall back to `0` unless the step JSON supplies it.
- Additional fields (`depth`, `seconds`, etc.) may exist in the source logs but are ignored by the packer.

### Step records (`*.jsonl.gz`)

The packer reads the following fields per line; extra fields are ignored:

```json
{
  "seed": 272350805,
  "step_index": 50,
  "max_rank": 6,
  "move": "right",
  "valuation_type": "search",
  "board": [6, 5, 3, 1, 2, 2, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0],
  "branch_evs": {"up": null, "left": 2.511, "right": 2.536, "down": -5.262}
}
```

Notes:
- `step_index` is optional; if missing, the packer assigns a running index starting at 0.
- `seed` and `max_rank` are optional per-step; the packer falls back to the `.meta.json` values.
- `valuation_type` is optional; missing values default to `"search"`.
- `branch_evs` is a map from move name (`"up"`, `"down"`, `"left"`, `"right"`) to a float or `null`. `null` indicates an illegal move for that step.

## Packed dataset format (what training consumes)

A packed dataset directory contains:

- `steps.npy` or `steps-00000.npy`, `steps-00001.npy`, ...
- `metadata.db`
- `valuation_types.json`
- (optional) `values-*.npy` sidecars in some datasets (see below)

### Row-level dtype (`steps-*.npy`)

The Macroxue pack uses an unaligned NumPy record dtype (no padding). The on-disk dtype from the Rust packer matches this layout exactly. When `--include-cumulative-reward` is enabled, an extra `cumulative_reward` field is inserted (see below).

```python
import numpy as np

STEP_ROW_DTYPE = np.dtype(
    [
        ("run_id", "<u4"),           # sequential run id assigned by the packer
        ("step_index", "<u4"),       # index within the run
        ("board", "<u8"),            # 16 packed 4-bit tiles (MSB nibble = cell 0)
        ("board_eval", "<i4"),       # Macroxue heuristic score for the current board
        ("cumulative_reward", "<i4"),# sum of merge rewards from this step to game end (optional)
        ("tile_65536_mask", "<u2"),  # bit i set when tile exponent >= 16 (>= 2**16)
        ("move_dir", "<u1"),         # UDLR: 0=up, 1=down, 2=left, 3=right
        ("valuation_type", "<u1"),   # enum id; see valuation_types.json
        ("ev_legal", "<u1"),         # UDLR bitfield (bit0=up, bit1=down, bit2=left, bit3=right)
        ("max_rank", "<u1"),         # max tile exponent for the step
        ("seed", "<u4"),             # PRNG seed (truncated to u32)
        ("branch_evs", "<f4", (4,)), # EVs [up, down, left, right]
    ],
    align=False,
)

assert STEP_ROW_DTYPE.itemsize == 50  # with cumulative_reward; 46 without it
```

Key details:

- `board` is packed MSB-first: cell index 0 is the highest-order nibble of the uint64. The packer preserves the raw `board` list order from the JSON.
- `tile_65536_mask` stores overflow flags. If an exponent >= 16 appears in the source, the stored nibble is clamped to 15 and the corresponding mask bit is set. **The exact exponent > 16 is not preserved**.
- `board_eval` is computed during packing using the Rust port of the Macroxue heuristic (see `crates/dataset-packer/src/macroxue/board_eval.rs`). It is stored as an `i32`.
  - The training collate path will recompute `board_eval` only if the field is missing or if data augmentation (rotation/flip) changes the board.
  - If you recompute `board_eval` from the packed board, tiles > 65536 are treated as exactly 65536 due to the mask, so values can differ from the stored `board_eval`.
- `cumulative_reward` (if present) stores the undiscounted merge reward-to-go from this step to game end (including the current move). Compute immediate rewards as `cumulative_reward[i] - cumulative_reward[i+1]` (or `cumulative_reward[i]` for the last step).
- `branch_evs` uses UDLR order. Illegal moves are stored as `0.0` but **must** be masked using `ev_legal`.
- `move_dir` uses UDLR order (`0=up, 1=down, 2=left, 3=right`).
- `valuation_type` is a `u8` enum; the mapping lives in `valuation_types.json`.

### Run ordering and `run_id`

`run_id` is assigned by the Rust packer by enumerating `.meta.json[.gz]` files in **lexicographic path order**. The packer processes in parallel but uses the enumerated index as the stable `run_id`.

Within each run, rows are appended in the order they appear in the JSONL. Always use `(run_id, step_index)` to reconstruct a trajectory; do not assume any global contiguity across runs.

### `metadata.db`

The packer writes a SQLite database with:

- `runs(id INTEGER PRIMARY KEY, seed BIGINT, steps INT, max_score INT, highest_tile INT)`
- `session(meta_key TEXT PRIMARY KEY, meta_value TEXT)`

The training dataloader selects runs from `runs` and tolerates historical column names (`steps`, `num_steps`, or `num_moves`). It does not require `session` for loading.

If the packer is invoked with `--include-cumulative-reward`, the session table includes:

- `meta_key = "cumulative_reward"`
- `meta_value = "true"` or `"false"`

### `valuation_types.json`

This file maps **name -> id** (u8). Example from `datasets/d6_3b_v0`:

```json
{
  "tuple11": 1,
  "search": 0,
  "tuple10": 2
}
```

Important:
- Keep `valuation_types.json` alongside the shards. The loader in `packages/train_2048` inverts this map and uses it to decode `valuation_type`.
- The fallback mapping in the loader is `{"search": 0, "tuple10": 1, "tuple11": 2}` which **does not match** the packer's default ordering (`search`, `tuple11`, `tuple10`). Do not rely on the fallback.

### Optional sidecars (`values-*.npy`)

Some datasets (e.g. `datasets/d6_3b_v0`) include `values-*.npy` shards. These are **not** consumed by the training dataloader today. The observed dtype in `d6_3b_v0` is:

```python
VALUE_ROW_DTYPE = np.dtype(
    [
        ("run_id", "<u4"),
        ("step_index", "<u4"),
        ("reward", "<f4"),
        ("reward_scaled", "<f4"),
        ("return_raw", "<f4"),
        ("return_scaled", "<f4"),
    ],
    align=False,
)
```

## Packing utility (`crates/dataset-packer`)

The packer consumes raw Macroxue logs and produces the packed layout:

```bash
cargo run -p dataset-packer -- pack \
  --input /path/to/raw \
  --output /tmp/macroxue-pack \
  --workers 8 \
  --shard-rows 10000000 \
  --overwrite
```

Behavior summary:
- Discovers `*.meta.json` or `*.meta.json.gz` under `--input` and pairs them with `*.jsonl.gz` of the same basename.
- Assigns `run_id` in lexicographic order of the meta file paths.
- Parses `branch_evs` and sets `ev_legal` bits for moves with a non-null EV.
- Computes `board_eval` from the **original** board exponents before packing/clamping.
- Computes `cumulative_reward` only when `--include-cumulative-reward` is set.
- Emits `steps-*.npy`, `metadata.db`, and `valuation_types.json`.

Merging two packed datasets (reindexing `run_id`s and unifying valuation enums) uses the same binary:

```bash
cargo run -p dataset-packer -- merge \
  --left datasets/macroxue/d6 \
  --right datasets/macroxue/d7 \
  --output datasets/macroxue/merged \
  --shard-rows 10000000 \
  --overwrite
```

## Loader expectations (train_2048)

The Macroxue collate path requires these fields in `steps-*.npy`:

- `board` (packed u64)
- `branch_evs` (float32[4])
- `valuation_type` (u8)
- `ev_legal` (u8)
- `move_dir` (u8)
- `tile_65536_mask` and `board_eval` are optional, but missing `board_eval` will trigger on-the-fly evaluation (slower; also see overflow note above).

For non-Macroxue datasets (`target.mode` = `binned_ev` or `hard_move`), the loader expects:

- `board` (packed u64)
- `branch_evs` **or** legacy `ev_values`
- Optional `ev_legal` (if missing, legal moves are inferred from finite EVs)
- `move_dir` or legacy `move` if using `hard_move`

## Compatibility notes / footguns

- Always ship `valuation_types.json` with the shards. The loader's fallback ordering is not consistent with the packer default.
- `board_eval` is required for search tokenization; missing values are recomputed from the packed board and will differ when tiles exceed 65536.
- `branch_evs` uses UDLR order and must be masked by `ev_legal`; illegal moves may have `0.0` EVs.
- `tile_65536_mask` only records whether exponent >= 16; the exact exponent above 16 is not stored.
- If you stitch datasets with different board encodings (packed `board` vs `exps`), normalize them before batching; the Macroxue loader expects packed boards.

## Proposal: zstd-compressed shards

### Motivation

Current disk usage for 3B steps is ~120GB. With zstd compression at level 3, this drops to ~53GB (44% of original). Given the shard-local loading pattern (load one 4GB shard, sample exhaustively, move on), decompression overhead is negligible: ~4 seconds per shard on a Ryzen 9 9900X, amortized over minutes of training per shard.

### Does this make sense?

**Yes**, given:

- Shard-local sampling means we load each shard once per epoch anyway
- 60GB RAM >> 4GB shard, so decompressing into RAM is fine
- mmap mode is unused (and incompatible with compression, but we don't need it)
- Disk savings of ~67GB on the full dataset is meaningful for keeping multiple dataset versions

**Tradeoffs**:

- Can't `hexdump`/inspect shards directly (use `zstd -d -c file.npy.zst | hexdump`)
- Minor implementation work (~50-70 lines total)
- Slightly slower shard discovery (we read the full shard to get shape; acceptable with shard-local loading)

### Recommended libraries

**Rust (dataset-packer)**:
```toml
zstd = "0.13"
```

The `zstd` crate wraps libzstd and is the de facto standard. Use `zstd::stream::Encoder` around the existing `BufWriter<File>`.

**Python (train_2048)**:
```toml
zstandard = ">=0.23"
```

The `zstandard` package (by Gregory Szorc) is the canonical Python binding. Avoid `zstd` (different package, less maintained).

### Implementation approach

#### Packer changes (`crates/dataset-packer/src/writer.rs`)

1. Add `--compress` flag (default off for backward compat during transition)
2. Write `.npy.tmp` shards as usual, then zstd-compress to `.npy.zst` in `finish()`
3. Output extension: `.npy.zst`
4. Delete the uncompressed `.npy.tmp` after successful compression/rename

#### Loader changes (`packages/train_2048/.../shard_loader.py`)

1. Update shard discovery glob: `glob("steps-*.npy") + glob("steps-*.npy.zst")`
2. In `load_shard()` / `load_shard_for_sampling()`:

```python
import zstandard
import io

def _load_npy(path: Path) -> np.ndarray:
    if path.suffix == '.zst':
        dctx = zstandard.ZstdDecompressor()
        with open(path, 'rb') as f:
            data = dctx.decompress(f.read())
        return np.load(io.BytesIO(data))
    return np.load(str(path))
```

3. Compressed shards are only supported in shard-local loading. Legacy loaders should reject `.npy.zst` with a clear error.
4. Shape discovery currently loads the full shard (same as shard-local sampling). This is acceptable for now.

### Safety checks

Before deploying compressed shards:

1. **Round-trip verification**: After packing, decompress and compare checksums:
   ```bash
   # In packer, emit sha256 of uncompressed data to metadata.db or sidecar
   # On load, verify: zstd -d -c shard.npy.zst | sha256sum
   ```

2. **Row count check**: Verify `steps-*.npy.zst` row counts sum to `metadata.db` total:
   ```python
   assert sum(shard.num_steps for shard in shards) == db_total_steps
   ```

3. **Decompression error handling**: Catch `zstandard.ZstdError` and fail loud:
   ```python
   try:
       data = dctx.decompress(f.read())
   except zstandard.ZstdError as e:
       raise RuntimeError(f"Corrupt shard {path}: {e}") from e
   ```

4. **First-batch sanity check**: On first batch of training, verify a few rows against known values or dtype expectations (board is u64, move_dir < 4, etc.). Already done in collate, but worth keeping.

Don't bother with:
- Per-row checksums (overkill, zstd has internal integrity)
- Compression ratio assertions (varies by data)
- Gradual rollout / feature flags (just convert and ship)

### Migration path

1. Implement with `--compress` flag defaulting to off
2. Re-pack one dataset (e.g., `d6_1b_v2`) with compression
3. Run training, verify loss curves match uncompressed
4. Convert remaining datasets, delete uncompressed originals
5. Flip default to `--compress` on

---

## Optional cumulative future merge reward field

### Motivation

For value-based or return-conditioned training, we may want the cumulative future merge reward (reward-to-go) from each step to the end of the game. This must be computed during packing because game boundaries are lost after steps are written to shards. The packer only includes it when `--include-cumulative-reward` is set.

### Field specification

Add to `MacroxueStepRow` in `schema.rs` and enable in the packer with `--include-cumulative-reward`:

```rust
pub cumulative_reward: i32,  // sum of merge rewards from this step to game end
```

**Semantics**: At step `i`, `cumulative_reward` = Σ(merge rewards for steps i, i+1, ..., T) where T is the terminal step.

### Implementation

In `parse_steps_file()` (`crates/dataset-packer/src/macroxue/parse.rs`), after collecting all rows for a game:

```rust
use twenty48_utils::engine::{merge_reward_exps, Move};

// Compute exact merge reward per step, then reverse cumsum
let mut cumulative = 0i64;
for i in (0..rows.len()).rev() {
    let move_dir = Move::from_udlr(rows[i].move_dir);
    let step_reward = merge_reward_exps(&raw_board_exps[i], move_dir) as i64;
    cumulative += step_reward;
    rows[i].cumulative_reward = cumulative as i32;
}
```

Here `raw_board_exps` is a `Vec<[u8; 16]>` captured from the JSONL `board` arrays for each step.

**Why this is exact**: `merge_reward_exps()` computes the merge reward directly from the pre-move exponents without relying on the packed 4-bit board, so it stays correct even when exponents exceed 15. The spawned tile (2 or 4) only appears in the *next* step's `board` field, so the merge reward is unaffected by spawns.

### Additional changes

1. **`schema.rs`**: Add `cumulative_reward: i32` to `MacroxueStepRow` (kept optional via `--include-cumulative-reward`)
2. **`engine/state.rs`**: Add `Move::from_udlr(u8) -> Move` if not present
3. **`macroxue` packer**: compute the cumulative reward during pack, not merge

### Discounted returns (punted)

γ-discounted returns can be computed as a post-hoc sidecar if needed:

```python
# Assuming steps are ordered by (run_id, step_index) within each shard
for run in runs:
    steps = shard[shard['run_id'] == run]
    discounted = 0.0
    for i in reversed(range(len(steps))):
        step_reward = steps[i].cumulative_reward - steps[i+1].cumulative_reward if i+1 < len(steps) else steps[i].cumulative_reward
        discounted = step_reward + gamma * discounted
        # write to sidecar
```

This preserves ordering invariant (`step_index` monotonic per `run_id`) and doesn't require re-packing. Punt until actually needed.

### Verification

After packing, spot-check a few games:
1. When present, the first step's `cumulative_reward` should equal the game's final score (from `metadata.db`)
2. `cumulative_reward[i] - cumulative_reward[i+1]` should equal the merge reward for move i (last step uses `cumulative_reward[i]`)
