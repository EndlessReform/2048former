# bin utilities

## visualize-runs.py
- usage: `uv run bin/visualize-runs.py -i <dataset_dir> -o <out_dir>`
- reads `metadata.db` to count runs by `max_score`
- writes `<dataset>_max_score_hist.png` with bars for small score sets and histogram otherwise
- writes `<dataset>_run_steps_hist.png` as a run-length histogram and overlays x-axis tile reach steps (colored by tile 512–32768; requires `steps*.npy`)
- `--bins` controls histogram resolution when scores are dense (default 50)
