# Macroxue self-play data

For our pretraining data, we use self-play board states from Macroxue's hybrid expectimax 2048 implementation. (more precisely, from a [personal fork](https://github.com/EndlessReform/macroxue-expectimax-2048) instrumented for data capture). Note: since the original repo is GPL-v3, we keep it at arms-length and only train on outputs, as shown here.

## Core algorithm

TODO fill in how this differs from vanilla expectimax - not the focus of this project

### Raw output

A folder of games might look like this:

```
example-folder
├── d6_10g_v1 # Note: these folder names and file naming conventions are arbitrary, don't rely on them
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

For a 2048 _run_ (aka a full game) through board _states_ (also referred to as steps), the process saves two files.

**Metadata sidecar:** `.meta.json`, uncompressed.

```json
// Will not actually be prettyprinted
{
  "seed": 272350805,
  "depth": 6,
  "game_index": 0,
  "steps_file": "selfplay_logs/d6_test_v1/depth06_worker00_seed0272350805_game000000.jsonl.gz",
  "num_moves": 27885,
  "score": 795564,
  "max_tile": 32768,
  "max_rank": 15,
  "sum_tile": 61434,
  "seconds": 50.247026
}
```

In the `.jsonl.gz` corresponding, we see rows like:

`valuation_type: search`:

```json
// Earlygame: valuations can go MUCH higher
{'seed': 272350805,
 'step_index': 50,
 'max_rank': 6,
 'move': 'right',
 'valuation_type': 'search',
 'valuation': 2.536,
 'board': [6, 5, 3, 1, 2, 2, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0],
 'branch_evs': {'up': None, 'left': 2.511, 'right': 2.536, 'down': -5.262}}
```

`tuple11`:

```json
{
  "seed": 272350805,
  "step_index": 20000,
  "max_rank": 15,
  "move": "down",
  "valuation_type": "tuple11",
  "valuation": 0.817625,
  "board": [15, 11, 8, 1, 13, 9, 7, 1, 6, 5, 3, 3, 1, 2, 4, 1],
  "branch_evs": {
    "up": 0.735862,
    "left": 0.817631,
    "right": 0.209081,
    "down": 0.817681
  }
}
```

For a full explanation of valuation types, see `docs/valuations.md`.

NOTE: earlier runs might have extra fields on the `.jsonl.gz`. Only assume the fields here are present, but ensure whatever reader parses the jsonl.gz (e.g. serde-json) is capable of reading these files.


## Pretraining-ready format

Similar to the earlier MattKennedy pure expectimax, we keep two files: a `metadata.db` and a `steps.npy` (or sharded `steps-00000.npy`, …) pool. The intent is to keep data in-RAM, for highly efficient random read access over rows at bsz 1024-4096.

TODO: add sharding

### Row-level data format

We use a custom npy compound dtype for rows, aligned and padded to keep SIMD loads straightforward:

```python
import numpy as np

STEP_ROW_DTYPE = np.dtype(
    [
        ("run_id", np.uint32),           # sequential game id
        ("step_index", np.uint32),       # index within the run
        ("board", np.uint64),            # 16 packed 4-bit tiles (MSB nibble = cell 0)
        ("board_eval", np.int32),        # macroxue heuristic score for the current board
        ("tile_65536_mask", np.uint16),  # bit i set when tile exponent >= 16 (>= 2**16)
        ("move_dir", np.uint8),          # UDLR: 0=up, 1=down, 2=left, 3=right
        ("valuation_type", np.uint8),    # enum (see valuation_types.json)
        ("ev_legal", np.uint8),          # UDLR bitfield (bit0=up, bit1=down, bit2=left, bit3=right)
        ("max_rank", np.uint8),          # denormalised for filtering
        ("seed", np.uint32),             # PRNG seed that generated the run
        ("branch_evs", np.float32, (4,)),# EVs [up, down, left, right]
    ],
    align=True,
)

assert STEP_ROW_DTYPE.itemsize == 48
```

Dataset pack shards preserve this layout. `crates/dataset-packer` walks `.meta.json` files lexicographically, assigns sequential `run_id`s (matching `runs.id`), and appends each run's rows in the order they appear in the source JSONL. Readers should rely on `run_id` + `step_index` when reconstructing trajectories rather than assuming global contiguity beyond per-run grouping.

`board_eval` is re-computed during packing using the Rust port of the original Macroxue heuristic (see `crates/dataset-packer/src/macroxue/board_eval.rs`). Values match `packages/train_2048/tokenization/macroxue/board_eval.py` and remain within 32-bit signed range. Downstream tokenization code can use this field directly when constructing advantage bins without re-running the heuristic in Python.

### Runs metadata

We keep the existing SQLite metadata schema so the training dataloader can keep issuing SQL over `runs(id INTEGER PRIMARY KEY, seed BIGINT, steps INT, max_score INT, highest_tile INT)` with optional `session(meta_key TEXT PRIMARY KEY, meta_value TEXT)`. All fields are present in the Macroxue sidecars, so the build is mechanical:

1. Enumerate every `.meta.json` (one per game) and assign a stable integer `id` that matches the `run_id` values written into `steps.npy` (we typically use the incrementing run counter from the ingest loop).
2. Parse `seed`, `num_moves`, `score`, and `max_tile` from the JSON and insert them as `seed`, `steps`, `max_score`, and `highest_tile`. Other fields (`depth`, `seconds`, etc.) can be stored as JSON under `session` if you want to preserve them, but the loader does not require them.
3. Create `metadata.db`, run the schema shown above, and bulk insert the rows. The Rust packer described below performs the join and ensures `steps['run_id']` stays aligned.

This preserves compatibility with `packages/train_2048.dataloader`, which expects to select run IDs via SQL and intersect them with `steps['run_id']`.

### Packing utility

`crates/dataset-packer` bundles a CLI that walks the raw Macroxue drops, pairs each `*.meta.json` **or** `*.meta.json.gz` with the matching `*.jsonl.gz`, and emits the aligned artifacts:

```bash
cargo run -p dataset-packer -- pack --input /path/to/raw --output /tmp/macroxue-pack --workers 8 --shard-rows 10000000 --overwrite
```

Outputs:
- `steps.npy` (or sharded `steps-00000.npy`, …) with the dtype above.
- `metadata.db` populated from the sidecars (`runs` + `session` tables).
- `valuation_types.json`: index → valuation-name lookup used by `valuation_type`.

### Dataset comparison

| Aspect | Macroxue pack (`steps-*.npy`) | Self-play v1 (`steps-*.npy`) | Legacy expectimax (`steps.npy`) | Footguns / Incompatibilities |
| --- | --- | --- | --- | --- |
| Step dtype | 48-byte struct with packed board, valuation metadata, branch EVs, legal mask | Minimal struct: `run_id<u64>`, `step_idx<u32>`, `exps<(16,)u8>` | Packed board only (`board<u64>` + sparse extras depending on drop) | Mixing loaders: self-play reader expects `exps`, Macroxue loader needs `board` + masks; selecting wrong collate path will KeyError.
| Board encoding | Packed MSB-first 4-bit nibbles in `board`, with `tile_65536_mask` overflow bits | Plain exponents array `[16]` | Packed MSB-first (no overflow mask) | Converting between pools requires `BoardCodec` conversions; forgetting to apply `tile_65536_mask` loses 65536+ tiles.
| Move/legality info | `move_dir<u8>` + `ev_legal` bitmask + `branch_evs[f32;4]` | No move or EV fields (pure board snapshots) | Sometimes includes `move` string but no legality mask | Downstream tasks that assume EVs will break on self-play v1/legacy; guard on dtype fields before indexing.
| Valuation metadata | `valuation_type<u8>`; lookup in `valuation_types.json` | None (engine logits only) | None | Readers must ignore `valuation_type` if lookup file missing; ensure merge scripts keep enums consistent.
| Run id width | `run_id<u32>` (packer remaps into contiguous range) | `run_id<u64>` (matches engine game id) | Varies; often `u32` | Treat as upcastable when stitching pools; do not assume 32-bit to avoid overflow when merging.
| Sharding | Optional `steps-00000.npy` shards (size via `--shard-rows`) | Always shards when buffer hits `report.shard_max_steps` (default 1M) | Usually single `steps.npy` | Older tooling that hardcodes single-file paths must glob for `steps-*.npy` before loading.
| Metadata db | `runs` + `session` tables | Same schema | `runs` table only in some drops | Consumers should fallback gracefully if `session` missing for legacy pools.
| Embeddings alignment | No inline embeddings; offline jobs produce separate features | Optional inline `embeddings-*.npy` per shard (float32) | None | Training code must handle embeddings absent (Macroxue/legacy) vs present (self-play) without assuming shapes.

#### Reader footguns

- Pick the dataset-specific collate path (`make_collate_macroxue` vs `make_collate_steps`). Each validates for expected fields; use the wrong one and you will either crash or silently drop columns.
- When stitching Macroxue with self-play v1, normalise board representations early (e.g., decode `board` into exponents) before batching; mixed representations inside one batch lead to incorrect tokenisation.
- `run_id` ordering differs (packer lexicographic vs engine game IDs). Always join on `(run_id, step_index)` rather than relying on contiguous global indices.
- Legacy expectimax pools lack the `tile_65536_mask`; if you need to detect >65536 tiles, recompute from the raw JSON source or discard that signal for those runs.
- `valuation_type` ids are per-pack. Keep the accompanying `valuation_types.json` alongside `steps-*.npy` or regenerate enums before training.

`--shard-rows` is optional today but provides the extension point for splitting the dataset without rewriting the packer. Progress is reported via `indicatif`, logging is standard `env_logger`, and the heavy lifting (JSON parsing, board packing) runs under `rayon`.

Merging two packed datasets (reindexing `run_id`s, unifying valuation enums, and optionally deleting the sources once the merge succeeds) uses the same binary:

```bash
cargo run -p dataset-packer -- merge --left datasets/macroxue/d6 --right datasets/macroxue/d7 \
    --output datasets/macroxue/merged --shard-rows 10000000 --overwrite --delete-inputs
```

#### On Rust

- `.npy` writers live in `crates/dataset-packer` (library) and `crates/game-engine/src/ds_writer.rs`, both using `npyz` with the `derive` feature to keep structured dtypes in sync with NumPy. Always express the dtype explicitly (see `StepRow::dtype()` / `step_row_dtype()`) and ensure `align=True` on the Python side matches the Rust struct layout.
- Metadata writers use `rusqlite` (bundled) via `SessionRecorder` in `crates/game-engine/src/recorder.rs`. Stick to WAL + `synchronous=NORMAL`, and keep inserts idempotent (`INSERT … ON CONFLICT DO UPDATE`).
- When adding new fields, update both the Rust schema and the docs first, then regenerate the helper scripts; previous breakages came from mismatching dtype definitions.
