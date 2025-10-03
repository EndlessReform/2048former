# bin utilities

## visualize-runs.py
- usage: `uv run bin/visualize-runs.py -i <dataset_dir> -o <out_dir>`
- reads `metadata.db` to count runs by `max_score`
- writes `<dataset>_max_score_hist.png` with bars for small score sets and histogram otherwise
- writes `<dataset>_run_steps_hist.png` as a run-length histogram and overlays x-axis tile reach steps (colored by tile 512–32768; requires `steps*.npy`)
- optional `--tile-progress` emits `<dataset>_tile_presence.png`, charting per-step-bin counts of board states containing tiles 8192–65536 (bin size via `--progress-bin-size`, default 500)
- optional `--survivorship` emits `<dataset>_survivorship.png`, plotting how many runs remain alive per step bin (uses `--progress-bin-size`)
- optional `--max-runs` limits the number of runs loaded from metadata (first N by id)
- auto-detects packed-board (`board`/`tile_65536_mask`) and structured `exps` step formats
- `--bins` controls histogram resolution when scores are dense (default 50)
