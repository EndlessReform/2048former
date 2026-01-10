# compute_perplexity.py

Compute perplexity for 2048 checkpoints on steps datasets.

The script auto-detects `macroxue_tokens` when `tokenizer.json` is present in
the checkpoint directory; otherwise it uses EV binning from
`training-config.json`.

## Usage

```bash
uv run --locked scripts/compute_perplexity.py \
  --init checkpoints/20251011_171039 \
  --dataset datasets/raws/d7_test_v1 \
  --batch-size 1024
```

## Optional flags

- `--n-samples N`: sample N boards instead of scanning the full dataset.
- `--indices path.npy`: evaluate the given index array (int64).
- `--per-board-out out/ppl.npz`: write per-board loss/perplexity sidecar.
- `--show-hardest`: print the hardest boards table.
- `--top-k K`: number of hardest boards (with `--show-hardest`).

## Output

The summary includes overall perplexity, per-head perplexity, and the IQR
of per-branch perplexity. When `--show-hardest` is set, the script prints the
K hardest boards by per-board perplexity.

---
# Appendix: Proposal for TUI sampling

Once we have a dataset:perplexity pairing, a human annotator needs to be able to scrub through
the boards to:
- Page through random boards
- Page through hardest boards / easiest boards to search for outliers
- Navigate through _games_: a screen (possible) to look through games and do the above scrubbing
- Third screen for summary statistics, with features to 'drill down' by features like:
  - Head direction
  - Importantly: weight of the board / presence of tiles / 'building up' to tiles

UX philosophy:
- The 2048 board should visually look like a 2048 board (in the center), with breadcrumbs to show whether 
- This should have keyboard navigation that's Vim-first (for core movement idiom), but uses either F-keys (classic terminal style)
  or mnemonic binds. No mouse should be required.
- UX should be maximalist: for a board, try to expose details from both teacher and student. Assume at least a MBP style full screen, possibly making use of a fullscreen desktop worth of chrome.

Tech:
- Use existing `core_2048` / `twenty48` PyO3 ops when possible
- Ensure not all boards are loaded into memory at once (dataset)
