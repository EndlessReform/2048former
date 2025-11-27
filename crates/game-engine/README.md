# Game Engine Orchestrator

## Overview
- Tokio-based runner that reuses the self-play feeder to batch boards, call the gRPC inference service, and apply moves with `ai-2048`.
- Ships two binaries:
  - `game-engine` &mdash; drives on-policy self-play and records new datasets.
  - `annotation-engine` &mdash; re-scores existing Macroxue or self-play shards and writes annotation sidecars (including per-bin log-probabilities).
- Inference responses carry per-head bin distributions (`InferenceOutput::Bins`) and, when present, value predictions. The annotator persists both the legacy summary (`annotations-*.npy`) and a log-probability sidecar (`annotations-logp-<dtype>-*.npy`, `dtype = f32` by default, `f16` via CLI), plus value sidecars (`annotations-value-*.npy` and `annotations-value-bins-*.npy`) when the server exposes a value head.

## Prerequisites
- A running inference server that implements `train_2048.inference.v1.Infer` (see `packages/infer_2048`).
- Rust toolchain (1.80+) with Cargo.
- Optional: tokenizer JSON for Macroxue tokenization previews.

## Building

```bash
cargo build -p game-engine
```

## Running Self-Play
1. Start the inference server (example):

   ```bash
   uv run infer-2048 \
     --init inits/v1_pretrained_50m \
     --uds unix:/tmp/2048_infer.sock \
     --device cuda
   ```

2. Launch the orchestrator with a TOML config:

   ```bash
   cargo run -p game-engine -- \
     --config config/inference/top-score.toml
   ```

### Sample Config (copy/paste)

```toml
# save as config/selfplay/example.toml
max_concurrent_games = 256
max_steps = 10_000_000

[orchestrator.connection]
# Prefer UDS locally; comment this section out and set tcp_addr for remote.
uds_path = "/tmp/2048_infer.sock"

[orchestrator.batch]
target_batch = 32
max_batch = 256
inflight_batches = 4

[orchestrator.report]
session_dir = "runs/example"
shard_max_steps = 1_000_000
max_gb = 16.0

[sampling]
strategy = "TailAggConf"
temperature = 0.8
top_p = 0.9
top_k = 2
```

## Running Annotation
1. Ensure the inference server above is running.
2. Annotate an existing dataset:

   ```bash
   cargo run -p game-engine --release --bin annotation-engine -- \
     --config config/inference/orchestrator.toml \
     --dataset datasets/macroxue/d6_10g_v1 \
     --output annotations/d6_10g_v1/new-model \
     --student-bins-dtype f16 \
     --overwrite
   ```

   - `--student-bins-dtype` selects the dtype for the log-probability sidecar (`f32` default, `f16` optional).
   - Use `--limit N` for smoke tests and `--overwrite` to replace previous shards.
   - The inference server now advertises tokenizer/value metadata via `Describe`; the annotator sizes sidecars from that handshake. Pass `--no-value-sidecar` to skip writing value outputs even when the server exposes a value head.

## Output Artifacts

| Binary             | Artifacts                                                                                                      |
|--------------------|---------------------------------------------------------------------------------------------------------------|
| `game-engine`      | `steps-*.npy`, optional `embeddings-*.npy`, and `metadata.db` inside `session_dir`.                             |
| `annotation-engine`| `annotations-*.npy` (summary row), `annotations-logp-<dtype>-*.npy` (per-bin log-probs), optional `annotations-value-*.npy` (scalar value/value_xform), optional `annotations-value-bins-*.npy` (value distribution), copied `metadata.db`, `valuation_types.json`, and `annotation_manifest.json` capturing bitmasks, dtype, and max bin counts. |

Manifest `policy_kind_mask` now exposes `POLICY_P1`, `POLICY_LOGPROBS`, `POLICY_HARD`, and `POLICY_STUDENT_BINS` bits so downstream tools can detect available annotations without scanning shards. When present, `value_kind_mask` records `VALUE` and `VALUE_PROBS` coverage per run alongside value metadata.

## Kendall tau evaluator (value head vs teacher branch_evs)
- Binary: `kendall-tau`
- Purpose: For each Macroxue step, expand legal moves into new boards via `twenty48_utils::engine::Board::shift`, run value-only inference on those boards (static batches, default `--batch-size 2048`), compute Kendall’s tau against teacher `branch_evs`, and write a sidecar `kendall-*.npy` aligned to `steps-*.npy`.
- Row fields: `run_id`, `step_index`, `legal_mask` (source), `scored_mask` (post-shift non-noop moves), `teacher_best`, `predicted_best`, `kendall_tau`, `value_scores[4]`.
- CLI example:

  ```bash
  cargo run -p game-engine --bin kendall-tau -- \
    --dataset datasets/macroxue/d7_test_v1 \
    --uds /tmp/2048_infer.sock \
    --batch-size 2048 \
    --output annotations/d7_test_v1/kendall \
    --overwrite
  ```

  Use `--limit N` for smoke tests. Requires a value head; Describe must report `has_value_head=true`.

## Notes & Safety Knobs
- `max_concurrent_games`, `max_steps`, and `[orchestrator.batch]` control throughput/backpressure.
- `orchestrator.report.*` sets shard sizing (`shard_max_steps`), output directory, and optional size caps (`max_gb`).
- `inline_embeddings = true` requests embeddings from the server; they are paired against steps before writing `embeddings-*.npy`.
- Annotation requires `argmax_only = false` in the orchestrator config so full distributions are returned.
