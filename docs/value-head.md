# Adding value head

## Background

The 2048 score is the sum of all merges made throughout the game (episode). In addition to the supervised _policy_ head trained from Macroxue game states, we want to add a _value head_ that predicts discounted return per state.

With full episodes available, the target should start as full Monte Carlo discounted reward,
$$G_t = \sum_{k=0}^{T-t-1} \gamma^k r_{t+k},$$
where the raw reward is the step-wise merge gain. Later we can layer on N-step TD for self-play/distillation. Following MuZero (Schrittwieser et al. 2019) and Stochastic MuZero (Antonoglou et al. 2022):
- Apply the invertible transform $h(x) = \mathrm{sign}(x)(\sqrt{x + 1} - 1 + \epsilon x)$ with $\epsilon = 0.001$ to stabilise large targets.
- Experiment with **MSE** vs **two-hot** (or one-hot) discrete cross-entropy over $n$ bins (e.g., 601).

## Where things stand (2025-xx)

Status snapshot (Apr 2025):
- ✅ Value sidecar CLI (`value-sidecar`) computes per-step rewards via `twenty48-utils` merge scores and writes discounted returns (`reward`, `reward_scaled`, `return_raw`, `return_scaled`) aligned to `steps-*.npy`. Flags: `--gamma`, `--reward-scale`, `--workers`, `--overwrite`.
- ⏳ Value head + training/inference plumbing still pending.

- Macroxue packs only store policy metadata: see `crates/dataset-packer/src/schema.rs` and `docs/macroxue_data/data_format.md`. There is no reward/return field yet; `runs.max_score` in SQLite is the final score only.
- The transformer only exposes policy heads: `packages/core_2048/src/core_2048/model.py` implements four binned EV heads or a single 4-way policy head. No value head or shared/value config exists.
- Training plumbs policy-only batches: `packages/train_2048/src/train_2048/dataloader/collate.py` emits branch EV bins/moves; objectives are policy-only (`objectives/binned_ev.py`, `macroxue_tokens.py`).
- Inference and annotations are policy-only: the protobuf (`proto/train_2048/inference/v1/inference.proto`) and server (`packages/infer_2048/src/infer_2048/server.py`) return policy bins/logits and optional embeddings; the annotation writer (`crates/dataset-packer/src/schema.rs`) has no value slots.

## Data pipeline plan

**Value sidecar for packed datasets** (new Rust CLI):
- [x] Compute per-step merge reward from packed boards + move (`crates/twenty48-utils` shift/score tables give deterministic merge score per move).
- [x] For each run (ordered by `run_id`, `step_index`), compute discounted returns backward. Expose `--gamma` and `--reward_scale` to manage the 1k–1.5m score range.
- [x] Write a NumPy sidecar (aligned with `steps-*.npy`) that keeps `run_id`, `step_index`, `reward`, and `return` (raw and transformed). Keep binning/transform in Python to allow objective swaps.

Run the sidecar generation on an existing packed dataset:

```bash
cargo run -p dataset-packer --bin value-sidecar -- \
  --dataset datasets/macroxue/d7_10g_v1 \
  --gamma 0.997 \
  --reward-scale 1.0 \
  --overwrite
```

The sidecar uses all cores by default; use `--workers N` if you need to cap threads.
Per-step reward computation is parallelized even for a single long run to avoid single-core bottlenecks.
- [ ] Consider sharding alongside annotations (`annotations-*.npy`) and copy `metadata.db` to keep joins simple.

## Model and training plan

- [ ] Add a value head atop the pooled board representation (same mean-pooled hidden state as policy heads). Keep the option to share trunk weights with policy.
- [ ] Config knobs to enable value head, choose objective (MSE vs discrete/two-hot), and freeze/unfreeze the encoder.
- [ ] Data loader joins: load value sidecar and align on `(run_id, step_index)` to batch `return` targets alongside existing policy fields.
- [ ] Training modes: (a) train value head only on frozen trunk; (b) fine-tune both heads jointly with weighted losses; (c) from-scratch co-training.
- [ ] Optional: experiment with Perceiver-style aggregation instead of mean-pooling for the value readout.

## Inference and evaluation plan

- [ ] Extend the protobuf + server to stream value predictions (raw and inverse-transformed) optionally, without forcing 1-ply inference to pay for it.
- [ ] Add value fields to annotation sidecars and viewer rails so per-step values can be inspected (match `annotation_manifest.json` shape checks).
- [ ] Add a smoke-test script mirroring `bin/play_2048.py` that logs predicted value vs realised returns over a short rollout.

## Contradictions vs current stack (review)

- There is **no value head implemented** today in `core_2048` or `train_2048`; all heads are policy-only.
- Datasets **do not carry reward/return columns**; only `max_score` per run exists, so a new sidecar/pack step is required before value training is possible.
- Inference/annotation rails **only move policy logits/bins**; the gRPC schema and Rust/Python servers would need extensions before value outputs can flow end-to-end.
