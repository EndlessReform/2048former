# Macroxue tokenization v2

Overall strategy for tokenizing expected value (EV) of the four branches up/down/left/right: "top-1, illegal, clearly harmful, or binned disadvantage".

Problem statement:
1. Raw board evaluations increase with the base valuation of the board
2. One or two moves (which move the high tiles out of the corners, breaking monotonicity) dominate metrics
3. Scale differences mean strategies like HL-Gauss weighting score across bins are not useful

## Strategy

Solution: Learning to rank by advantage. For both Search valuations and tuple10/tuple11 valuations,
- Selected move by teacher goes to WINNER (last class index) token
- Illegal moves go to ILLEGAL (first class index) token

For Search nodes:
- Board value is re-estimated using the macroxue valuation heuristic in `packages/train_2048/tokenization/macroxue/board_eval.py`. We multiply back the branch EV valuation (moves are now all legal) by 1000 and cast to int, to undo transform from original codebase. We then calculate the _advantage_ of each move (the values shown in the dataset are from terminal depth, e.g. d6) vs the current board value. Then:
    - for non-selected values, any value with _negative_ absolute advantage (vs current node) below a cutoff (e.g. -1500) is bucketed to the FAILURE token (second class index). This cutoff is a hyperparameter chosen when fitting the tokenizer and must be serialized with the resulting tokenizer artifact so that inference uses the same threshold.
    - For the other non-selected values, we take the absolute _disadvantage_ of that move vs the selected (e.g. -8, -150, etc). 
        - When creating the tokenizer, we use `pd.qcut` to distribute this into `n` bin edges (e.g. 32 bins) of roughly equal # of items. Search nodes learn their own set of bin edges.
        - When tokenizing, we then slot that branch to its bin.

For legal non-winning tuple10/tuple11 nodes, we follow a similar process:
- Branches with value 0 (aka 0 probability of reaching rate) are binned to the FAILURE token
- Remaining branches are checked for advantage against winning branch.
    - Branches with disadvantage 0 (a surprisingly large number) are binned to `num_bins - 1` index (second-to-last)
    - We fit separate `pd.qcut` bin edges for tuple10 and tuple11 disadvantages (e.g. 31 remaining bins when `n=32`) so each tuple family has quantiles matched to its own distribution.

`WINNER` is whichever branch the teacher expectimax selected (see `docs/macroxue_data/valuation.md` for details). Even when multiple branches tie for best score, the expectimax choice is the single branch assigned to `WINNER`; other tied branches are handled through the zero-disadvantage bin policy above.

Across all node types the vocabulary order is fixed and must be respected by any tokenizer:
1. `ILLEGAL`
2. `FAILURE`
3. Bin edges in monotonically decreasing disadvantage order (higher indices are closer to the winner). **The count of these disadvantage bins must be identical for Search, tuple10, and tuple11 so that a single shared vocabulary works everywhere.**
4. `WINNER`

## Implementation goals

> NOTE TO LLMs: feel free to extend, check off, or modify this section as long as the core todo items are still covered

- [x] Create script for fitting tokenizer on dataset; document usage
- [x] Port branch evaluation logic from python to Rust (if not already done)
- [x] add board valuation in dataset (if not already implemented)
- [x] Implement tokenization strategy in packages/train_2048
- [ ] Verify existing inference rails can use this successfully
- [ ] Visualization and sanity checks
    - [ ] Wire tokenization (optionally) to crates/annotation-server and ui/annotation-viewer, to allow humans to sanity-check strategy before release
    - [ ] Get predictions using this strategy during annotation; visualize predicted bin vs actual

### Tokenizer artifact format

Emit a `tokenizer.json` with:
- `tokenizer_type`: string identifier for this scheme (e.g. `"macroxue_ev_advantage_v2"`)
- `search`: object containing `bin_edges` (ordered list), `failure_cutoff` (integer threshold), and any other search-specific hyperparameters. Ensure `len(bin_edges)` matches the tuple families.
- `tuple10` / `tuple11`: each an object containing its own `bin_edges`, again with identical counts so token indices are shared
- `vocab_order`: explicit list of token names in the enforced order (`ILLEGAL`, `FAILURE`, …, `WINNER`)

This JSON must be sufficient for both data preprocessing and runtime inference to replicate the fitted tokenizer behavior without relying on training-time codepaths.

### CLI fitting workflow

Run `uv run tokenizer-macroxue DATASET_DIR --output tokenizer.json --num-bins 32 --search-failure-cutoff -1500`. The script expects a packed dataset directory (with `steps.npy`/`steps-*.npy` and `valuation_types.json`). It will error out if the pack is missing `board_eval`, so re-pack with the latest Rust tooling first. All valuation families share the `--num-bins` setting, and the saved `tokenizer.json` follows the schema above.
