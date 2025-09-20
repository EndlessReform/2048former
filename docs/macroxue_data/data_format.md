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

Similar to the earlier MattKennedy pure expectimax, we keep two files: a `metadata.db` and a `steps.npy` file as a pool. The intent is to keep data in-RAM, for highly efficient random read access over rows at bsz 1024-4096.

TODO: add sharding

### Row-level data format

We use a custom npy compound dtype for rows, 32-byte aligned for SIMD:

```python
import numpy as np

STEP_ROW_DTYPE = np.dtype(
    [
        ("board", np.uint64),            # 16 packed 4-bit tiles (LSB = cell 0)
        ("tile_65536_mask", np.uint16),  # bit i set when tile i stores 2**16
        ("move", np.uint8),              # 0=up, 1=right, 2=down, 3=left
        ("valuation_type", np.uint8),    # enum in docs/macroxue_data/valuation.md
        ("seed", np.uint32),             # PRNG seed that generated the run
        ("branch_evs", np.float32, (4,)),# EVs for [up, right, down, left]
    ],
    align=True,
)

assert STEP_ROW_DTYPE.itemsize == 32
```

### Runs metadata

We keep the existing SQLite metadata schema so the training dataloader can keep issuing SQL over `runs(id INTEGER PRIMARY KEY, seed BIGINT, steps INT, max_score INT, highest_tile INT)` with optional `session(meta_key TEXT PRIMARY KEY, meta_value TEXT)`. All fields are present in the Macroxue sidecars, so the build is mechanical:

1. Enumerate every `.meta.json` (one per game) and assign a stable integer `id` that matches the `run_id` values written into `steps.npy` (we typically use the incrementing run counter from the ingest loop).
2. Parse `seed`, `num_moves`, `score`, and `max_tile` from the JSON and insert them as `seed`, `steps`, `max_score`, and `highest_tile`. Other fields (`depth`, `seconds`, etc.) can be stored as JSON under `session` if you want to preserve them, but the loader does not require them.
3. Create `metadata.db`, run the schema shown above, and bulk insert the rows. A tiny helper script does the job:

```python
import json, sqlite3
from pathlib import Path

root = Path("macroxue_ingest")
db = sqlite3.connect(root / "metadata.db")
db.executescript(
    """
    CREATE TABLE IF NOT EXISTS runs (
        id INTEGER PRIMARY KEY,
        seed BIGINT NOT NULL,
        steps INT NOT NULL,
        max_score INT NOT NULL,
        highest_tile INT NOT NULL
    );
    DELETE FROM runs;
    """
)

rows = []
for idx, meta_path in enumerate(sorted(root.glob("**/*.meta.json"))):
    meta = json.loads(meta_path.read_text())
    rows.append(
        (idx, meta["seed"], meta["num_moves"], meta["score"], meta["max_tile"])
    )

db.executemany("INSERT INTO runs VALUES (?,?,?,?,?)", rows)
db.commit()
```

This preserves compatibility with `packages/train_2048.dataloader`, which expects to select run IDs via SQL and intersect them with `steps['run_id']`.

#### On Rust

- `.npy` writers live in `crates/game-engine/src/ds_writer.rs` and rely on `npyz` with the `derive` feature to keep structured dtypes in sync with NumPy. Always express the dtype explicitly (see `step_row_dtype()`) and ensure `align=True` on the Python side matches the Rust struct layout.
- Metadata writers use `rusqlite` (bundled) via `SessionRecorder` in `crates/game-engine/src/recorder.rs`. Stick to WAL + `synchronous=NORMAL`, and keep inserts idempotent (`INSERT … ON CONFLICT DO UPDATE`).
- When adding new fields, update both the Rust schema and the docs first, then regenerate the helper scripts; previous breakages came from mismatching dtype definitions.
