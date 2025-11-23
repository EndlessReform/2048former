# Dataset packer

For use with self-play logs from my [Macroxue 2048 expectimax fork](https://github.com/EndlessReform/macroxue-expectimax-2048/tree/data-collection).

## Usage examples

Packing a directory of files:

```bash
cargo run -p dataset-packer --release -- pack --input ../macroxue-2048-ai/selfplay_logs/d7_test_v1 --output datasets/raws/d7_test_v1 --overwrite --workers 24
```

Merging two packs:

```bash
cargo run --release -p dataset-packer -- merge --left ./datasets/raws/d6_19200_v1 --right ./datasets/raws/d6_24000g_v1 --output ./datasets/macroxue/d6_1b_v2 --shard-rows 100000000
```

Adding a value sidecar to an existing packed dataset (per-step reward and discounted returns):

```bash
cargo run -p dataset-packer --release --bin value-sidecar -- \
  --dataset ./datasets/macroxue/d7_test_v1 \
  --gamma 0.997 \
  --reward-scale 1.0 \
  --overwrite
```

Outputs a `values.npy` (or `values-*.npy` matching the input sharding) aligned 1:1 with `steps-*.npy` rows. Fields: `run_id`, `step_index`, `reward`, `reward_scaled`, `return_raw`, `return_scaled`.
Uses all available cores by default; pass `--workers N` to reduce parallelism. Reward calculation is parallelized even for single long runs.
