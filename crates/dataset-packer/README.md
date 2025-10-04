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