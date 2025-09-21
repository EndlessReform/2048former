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
