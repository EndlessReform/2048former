# Dataset packer

For use with self-play logs from my [Macroxue 2048 expectimax fork](https://github.com/EndlessReform/macroxue-expectimax-2048/tree/data-collection).

## Usage examples

Packing a directory of files:

```bash
cargo run -p dataset-packer --release -- pack --input ../macroxue-2048-ai/selfplay_logs/d7_test_v1 --output datasets/raws/d7_test_v1 --overwrite --workers 24
cargo run -p dataset-packer --release -- pack --input ../macroxue-2048-ai/selfplay_logs/d7_a --input ../macroxue-2048-ai/selfplay_logs/d7_b --output datasets/raws/d7_combo --overwrite --workers 24
# Include cumulative reward in the packed schema:
cargo run -p dataset-packer --release -- pack --input ../macroxue-2048-ai/selfplay_logs/d7_test_v1 --output datasets/raws/d7_test_v1 --overwrite --workers 24 --include-cumulative-reward
# Compressed shards (steps-*.npy.zst):
cargo run -p dataset-packer --release -- pack --input ../macroxue-2048-ai/selfplay_logs/d7_test_v1 --output datasets/raws/d7_test_v1 --overwrite --workers 24 --compress
```

Merging two packs:

```bash
cargo run --release -p dataset-packer -- merge --left ./datasets/raws/d6_19200_v1 --right ./datasets/raws/d6_24000g_v1 --output ./datasets/macroxue/d6_1b_v2 --shard-rows 100000000
# Compressed output shards:
cargo run --release -p dataset-packer -- merge --left ./datasets/raws/d6_19200_v1 --right ./datasets/raws/d6_24000g_v1 --output ./datasets/macroxue/d6_1b_v2 --shard-rows 100000000 --compress
# Delete input directories after a successful merge:
cargo run --release -p dataset-packer -- merge --left ./datasets/raws/d6_19200_v1 --right ./datasets/raws/d6_24000g_v1 --output ./datasets/macroxue/d6_1b_v2 --shard-rows 100000000 --delete-inputs
```

## Output layout (Macroxue packs)

Packed datasets include:

- `steps.npy` or `steps-*.npy` (`.zst` when `--compress`)
- `metadata.db`
- `valuation_types.json`

### Step row fields

The Macroxue step rows are stored as an unaligned NumPy record dtype:

```python
STEP_ROW_DTYPE = np.dtype(
    [
        ("run_id", "<u4"),
        ("step_index", "<u4"),
        ("board", "<u8"),
        ("board_eval", "<i4"),
        ("cumulative_reward", "<i4"),  # optional; only with --include-cumulative-reward
        ("tile_65536_mask", "<u2"),
        ("move_dir", "<u1"),
        ("valuation_type", "<u1"),
        ("ev_legal", "<u1"),
        ("max_rank", "<u1"),
        ("seed", "<u4"),
        ("branch_evs", "<f4", (4,)),
    ],
    align=False,
)
```

Notes:
- `board` is packed MSB-first 4-bit exponents (cell 0 is the high nibble). Tiles >= 2^16 are clamped to 15 with `tile_65536_mask` tracking overflow.
- `board_eval` is computed during packing using the Macroxue heuristic.
- `cumulative_reward` is optional (opt-in); it stores the undiscounted merge reward-to-go from this step to game end.

### metadata.db session flag

When packing, `metadata.db` includes a `session` table entry:

- `meta_key = "cumulative_reward"`
- `meta_value = "true"` or `"false"`

Merges require both inputs to agree on this flag (and the shard dtype).

For full details, see `docs/macroxue_data/data_format.md`.
