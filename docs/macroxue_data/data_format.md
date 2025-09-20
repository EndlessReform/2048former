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
# TODO GPT fill this in!
```

### Runs metadata

TODO GPT fill this in!