# train_2048

This package contains tools for training a 2048 AI model.

## Data Loading

Large datasets (100GB+) can cause OS page thrashing when randomising across shards. The training config exposes a shard-aware path that keeps one shard resident in RAM at a time:

- Set `dataset.shard_locality = true` to traverse shards sequentially while still sampling randomly within each shard.
- Optionally cap per-shard draws via `dataset.shard_locality_block_size` (default is the full shard).
- Enable `dataset.shard_cache_in_memory = true` (with `dataset.shard_cache_keep_shards`) to materialise the active shard into RAM while keeping only a small number cached.

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

## Tokenization

The `train_2048.tokenization.macroxue` module provides a tokenizer for "Macroxue" game states. This tokenizer converts game states with expectimax-derived action values into a sequence of tokens that can be used to train a transformer model.

### Tokenization Scheme

The tokenization scheme is called "Winner+Margins". It is designed to convey the structure of action-values to the model without exposing the raw expectimax values. The process is as follows:

1.  **ECDF-based Percentile Mapping**: For each valuation type (e.g., different search depths or heuristics), an empirical cumulative distribution function (ECDF) is fit to the distribution of all action-values. This ECDF is then used to map each action-value to a percentile in the range `[0, 1]`.

2.  **Token Generation**:
    *   **Winner Token**: The action with the highest percentile is identified as the "winner", and a `[WINNER]_{action}` token is generated (e.g., `[WINNER]_up`).
    *   **Illegal Move Tokens**: Actions that do not change the board state are considered illegal and generate an `[ILLEGAL]_{action}` token (e.g., `[ILLEGAL]_down`).
    *   **Margin Tokens**: For all other legal actions, the "regret margin" is calculated as the difference between the winner's percentile and the action's percentile (`Δ_a = p* - p_a`). This margin is then binned into one of 32 quantile-based bins, and a margin token `a#d{k}` is generated, where `a` is the action and `k` is the bin index (e.g., `left#d3`).

### Tokenizer Specification

The tokenizer is configured using a JSON specification file. This file contains the ECDF knots for each valuation type and the edges of the margin bins. The default location for this file is `out/tokenizer.json`.

### Generating the Tokenizer

To generate the tokenizer spec, run the following command from the project root:

```bash
uv run python -m train_2048.tokenization.macroxue
```

This will process the raw game state data (by default, from `datasets/raws/macroxue_d6_240g_tokenization/**/*.jsonl.gz`) and generate the `out/tokenizer.json` file.

## Output Head Format (Macroxue)

When training with `target.mode = "macroxue_tokens"`, each of the four per‑move heads (UDLR) predicts a categorical distribution over classes derived from the tokenizer. The class layout is standardized to match existing v1 semantics (worst → best):

- Head order: UDLR everywhere (data, training, inference).
- Number of classes per head: `n_classes = loser_bins + 2`, where `loser_bins = len(delta_edges) - 1` from `tokenizer.json`.
- Class indices per head:
  - `0` — ILLEGAL (move does not change the board)
  - `1 .. loser_bins` — margin bins ordered from worst → best
  - `loser_bins + 1` — WINNER (top action; “p1” bin)

Notes and rationale:

- The tokenizer computes ECDF‑based percentiles per valuation type and quantile‑based `delta_edges` for loser margins only. Collation maps those margins to ascending class indices so that higher indices always mean better options (consistent with v1 binned EV heads).
- During training, each head uses cross‑entropy against these per‑branch class targets. Winner bins typically concentrate probability mass near the last class.
- During inference, selection logic reads the last class (`p1`) per head for argmax decisions and optional tail aggregation; this layout matches the server/client expectations.

Quick usage:

- Train (example): `uv run python main.py --config config/pretraining/v2/10m-100k-ablation.toml`
- Profile steps 2-10 with torch profiler (trace saved under the run `profiles/` dir): `uv run python main.py --config config/pretraining/v2/10m-100k-ablation.toml --device cuda --profile --profile-start 2 --profile-end 10`
- The tokenizer path is configured at `dataset.tokenizer_path` and must point to a `tokenizer.json` generated as above.
