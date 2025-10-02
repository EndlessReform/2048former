# train_2048

This package contains tools for training a 2048 AI model.

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
- The tokenizer path is configured at `dataset.tokenizer_path` and must point to a `tokenizer.json` generated as above.

## Value Head Fine-Tuning

The value-head flow reuses the frozen encoder and learns either a linear probe or a shallow MLP on top of the mean-pooled board representation. Key pieces live in:

- Collate + dataset plumbing: `packages/train_2048/src/train_2048/dataloader/collate.py` / `steps.py`
- Objectives: `packages/train_2048/src/train_2048/objectives/value_head.py`
- Config knobs: `[target]` and `[value_head]` in `packages/train_2048/src/train_2048/config.py`

### Train a Value Head

1. Update `config/value_head/quantile_stratified.toml` with the desired init/checkpoint directories, dataset path, and (optionally) SQL filters for train/val runs.
2. Kick off training (ordinal/BCE reach-rate by default):
   ```bash
   uv run python packages/train_2048/src/train_2048/main.py \
     --config config/value_head/quantile_stratified.toml
   ```
3. Adjust `[value_head]` to switch between a linear probe (`head_type = "probe"`) and the 1-hidden-layer MLP (`head_type = "mlp"`), override dropout/hidden size, or unfreeze the transformer trunk (`freeze_trunk = false`).

### Evaluate on a Holdout Split

Use `benchmarks/value_head_eval.py` to score the trained ordinal head on a disjoint pool of game states. The script streams states directly from `steps.npy` shards, computes per-threshold AUROC/Brier/ECE, and optionally stratifies metrics by step quantiles (early/mid/late by default).

```bash
uv run python benchmarks/value_head_eval.py \
  --config config/value_head/quantile_stratified.toml \
  --init inits/v1_50m \
  --checkpoint checkpoints/value-head/model-stable.safetensors \
  --dataset selfplay/holdout_v2 \
  --run-sql "SELECT id FROM runs WHERE split = 'val'" \
  --output-csv out/value_head_metrics.csv
```

Flags of note:

- `--stage-boundaries` (default: `0 0.25 0.75 1.0`) controls the progress slices for early/mid/late reporting.
- `--threshold` can override the tile milestones evaluated (defaults come from `[value_head].tile_thresholds`).
- `--ece-bins` sets the calibration bin count (default 20).

The script prints a table of metrics and can emit JSON/CSV for downstream tracking.
