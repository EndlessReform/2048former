# Board Annotation Rails

## Background
Existing self-play pools come in two primary layouts:
- **Macroxue packs** captured from expectimax rollouts with rich metadata (`board`, `branch_evs`, `valuation_type`). They are materialised as `steps-*.npy` shards plus `metadata.db`. Their dtype is defined in `crates/dataset-packer/src/lib.rs` and consumed through the `StepsDataset`/`BoardCodec` helpers in `packages/train_2048/src/train_2048/dataloader/steps.py` and `packages/train_2048/src/train_2048/tokenization/base.py`.
- **Self-play v1 pools** recorded from our lean engine (`docs/self-play-v1.md`). They reuse the same shard/SQLite pattern but store exponent boards only (`run_id`, `step_idx`, `exps`).

Inference already runs over these boards via the gRPC service in `packages/infer_2048/src/infer_2048/server.py` backed by the protobuf contract in `proto/train_2048/inference/v1/inference.proto`. Today the service only emits per-move probabilities (possibly binned) and optional embeddings. There is no tooling to persist those predictions, combine them with teacher data, or surface them to the React/Vite viewer under `ui/annotation-viewer`.

## Goals
- Support offline annotation of step pools (Macroxue or self-play v1 format) with model predictions, storing results alongside source metadata.
- Allow multiple prediction heads per move (policy bins, scalar value heads such as `p_die` or `p_reach_16384`, raw expectimax scores) through a schema that can grow without breaking readers.
- Enable high-throughput annotation jobs (batching 10k–50k boards, concurrent inference clients) that reuse the existing inference stack.
- Provide an optional thin HTTP service exposing annotated runs and derived metrics to the front-end scrubber.
- Keep the storage + indexing story aligned with existing rails (NumPy structured shards + SQLite indices) for ease of reuse in training, benchmarking, or notebook workflows.

## Non-Goals
- Building the front-end UI; only the backend contract and data layout are defined here.
- Training new heads or changing model architecture (only how we consume its outputs).
- Replacing the existing inference gRPC API; changes must be backward-compatible.
- Streaming live games; the pipeline is batch-oriented over saved sessions.

## Requirements & Constraints
- Input pools may come from multiple sessions and must cope with 50M+ steps.
- Annotation should tolerate missing teacher metadata (self-play v1 lacks `branch_evs`). Metrics that depend on them must degrade gracefully.
- Storage needs to support multiple annotated models against the same pool without duplicating source boards.
- Schema must let us surface arbitrary scalar heads with friendly names (e.g. `p_die`, `reach_32768`) and, when available, multi-bin outputs (expectimax deltas, log advantages).
- Thin server should handle pagination/filtering without loading entire pools into RAM and must be optional (offline users can read NumPy/SQLite directly).

## Proposed Architecture

### Overview
```
+---------------------+      +-----------------------+      +---------------------+      +-----------------------+
| steps pool (macroxue| ---> | Annotation job runner | ---> | annotation session  | ---> | optional HTTP server  |
| or self-play v1)    |      | (local or gRPC model) |      | (npy + sqlite)      |      | (FastAPI/Axum)        |
+---------------------+      +-----------------------+      +---------------------+      +-----------------------+
                                                                      |                               |
                                                                      v                               v
                                                              React viewer (UI)                notebooks/CLI
```

### Source pool loader
- Reuse `StepsDataset` (`packages/train_2048/src/train_2048/dataloader/steps.py`) to stream shards as memmaps.
- Detect pool flavour by inspecting the first shard dtype:
  - Macroxue: contains fields `board`, `tile_65536_mask`, `branch_evs`, `move_dir`.
  - Self-play v1: contains `exps` (uint8[16]); also check metadata.db for `runs`.
- Introduce `AnnotationPool` helper:
  - Abstracts iteration into batches of `{run_id, step_index, exps[16], legal_mask?, teacher_policy?}`.
  - Uses `BoardCodec.decode_packed_board_to_exps_u8` for packed boards + masks, falling back to existing exponent arrays when available.
  - Optionally attaches teacher EVs if `branch_evs` present; otherwise leaves them `None` so downstream metrics can check capability.

### Annotation job runner
- New module `packages/train_2048/src/train_2048/annotation/job.py` wrapping the lifecycle:
  1. Read source pool metadata (copy `metadata.db` into the session, store provenance in `session_meta`).
  2. Chunk steps into configurable batches (default 16k). Provide asynchronous prefetch to keep inference saturated.
  3. Produce model tokens (`torch.uint8` exponents) and dispatch to either:
     - **Local client**: load init with `core_2048.load_encoder_from_init` and call `forward_distributions` (`packages/core_2048/src/core_2048/infer.py`).
     - **gRPC client**: call `Inference.Infer` with extended request (see next section).
  4. Collect predictions, derive metrics, and hand rows to the session writer.
- Metrics computed per step:
  - `model_top_action`, `model_top_prob` (from policy head).
  - `teacher_action` (from dataset `move_dir`) and optional `teacher_best_branch` vs model.
  - `policy_logit_gap` / `log_advantage` = `log p(model_top) - log p(teacher)`.
  - `teacher_branch_evs` copy when available for UI overlays.
  - `value_head_*` scalars pulled directly from model outputs (names declared by the model or config).
- Runner stores partial progress (processed shard, row offsets) so jobs can resume.

### Inference contract changes
- Extend `proto/train_2048/inference/v1/inference.proto` with backward-compatible fields:
  - `InferRequest.return_logits` (bool, default `false`) to request unsmoothed per-head logits.
  - `InferRequest.return_value_heads` (bool, default `false`) to request scalar heads.
  - `InferResponse` adds:
    - `repeated HeadLogits head_logits = 10;` inside `Output` where `HeadLogits { repeated float logits = 1; }`.
    - `repeated ValueHeadSpec value_specs = 11;` at response level, each `ValueHeadSpec { string name = 1; uint32 dim = 2; }`.
    - `enum ValueDType { VALUE_DTYPE_UNSPECIFIED = 0; FP32 = 1; FP16 = 2; BF16 = 3; }` and `value_dtype = 12`.
    - `bytes value_concat = 13;` storing concatenated value head outputs shaped `[batch, sum(dim_i)]`.
  - Keep defaults so existing clients (that ignore new fields) continue to work.
- Update `packages/infer_2048/src/infer_2048/server.py` to honour new flags:
  - Materialise logits before softmax and return when requested.
  - Surface model-defined scalar heads (e.g., value tower) via a new hook on the model: expect `(hidden_states, (policy_logits, value_heads_dict))` where `value_heads_dict` maps names → tensors.
  - Pack scalars/short vectors into the concatenated buffer with the declared dtype.
- Update local inference helper `packages/core_2048/src/core_2048/infer.py` with utilities mirroring the gRPC packing (so the annotation job shares decoding code regardless of transport).

### Annotation session storage
Organise output under `annotations/<session_id>/`:
```
annotations/<session_id>/
├── source.json                     # pointer to source pool, job config, model info
├── metadata.db                     # copy of source runs + extra tables
├── schema.json                     # describes stored heads and metrics
├── progress.json                   # optional resume markers
├── predictions-000000.npy          # structured rows aligned to source order
├── predictions-000001.npy
├── metrics-000000.npy              # derived metrics (small dtype)
└── value_heads-000000.npy          # optional; one per dtype when large
```
- `predictions-*.npy` dtype v1:
  - `run_id: <u4`, `step_index: <u4`, `legal_mask: |u1[4]` (UDLR),
  - `policy: <f2[4]` (float16 softmax probs for UDLR),
  - `logits: <f4[4]` when requested, else zeros. (Store dtype info in schema.)
- `metrics-*.npy` dtype v1:
  - `run_id`, `step_index`,
  - `teacher_action: |u1` (`0..3` with `255` sentinel for unknown),
  - `model_top_action: |u1`,
  - `model_top_prob: <f2`,
  - `logit_gap: <f4`,
  - `has_branch_evs: |u1`, optional `teacher_best_action: |u1` when available.
- `value_heads-*.npy`: pack `[N, num_slots]` float16 arrays where `schema.json` maps slot index → `{name, dim, dtype, offset}`. For scalar heads we simply reserve one slot each; multi-dimensional heads (e.g., expectimax per-branch scores) reserve contiguous slots and include shape info.
- `metadata.db` extends the copied `runs` table with:
  - `annotation_sessions(id INTEGER PRIMARY KEY, model_id TEXT, created_at TEXT, source_path TEXT, config_json TEXT)`.
  - `annotation_stats(run_id INTEGER, total_steps INTEGER, total_disagreements INTEGER, PRIMARY KEY(run_id))` for quick filtering.
- `schema.json` documents arrays:
```json
{
  "version": 1,
  "policy": {"dtype": "float16", "order": "udlr"},
  "logits": {"dtype": "float32", "present": true},
  "value_heads": [
    {"name": "p_die", "dtype": "float16", "offset": 0, "dim": 1},
    {"name": "reach_32768", "dtype": "float16", "offset": 1, "dim": 1}
  ],
  "metrics": {
    "fields": ["teacher_action", "model_top_action", "model_top_prob", "logit_gap"],
    "teacher_action_missing": 255
  }
}
```

### Derived metrics and extensibility
- Metrics module `packages/train_2048/src/train_2048/annotation/metrics.py` computes disagreement, calibration stats, and optional expectimax comparisons. Additional metrics can append columns to `metrics-*.npy` so long as schema is updated.
- Annotator stores per-run aggregates (counts, mean gaps) in SQLite for fast filtering (`WHERE model_top_action != teacher_action AND logit_gap > τ`).
- Value head storage uses slot-based schema so new heads can be appended without rewriting historic shards: new session simply lists extra slots; consumers check `schema.json` to see availability.

### Thin server
- New optional service `packages/annotate_2048/src/annotate_2048/server.py` (FastAPI) exposes REST endpoints over annotation sessions:
  - `GET /sessions` → list sessions with model metadata.
  - `GET /sessions/{session}/runs` → run-level stats (score, steps, disagreement rate) pulled from SQLite.
  - `GET /sessions/{session}/runs/{run_id}/steps?offset=&limit=&metrics=` → paginated step slices returning board exps, policy probs, teacher metadata, selected value heads.
  - `GET /sessions/{session}/schema` → returned verbatim `schema.json`.
  - `GET /sessions/{session}/steps/search?metric=logit_gap&min=0.5` → uses SQLite aggregates + memmap slicing to return matching step ids; client resolves to boards.
- Server shares NumPy memmaps across requests; it should load shards lazily and cache `AnnotationsDataset` handles per session. 
- UI (Vite app) fetches schema first to render available panels, then hits the steps endpoint while scrubbing. Client-side filtering for fine-grained metrics remains possible by fetching slices and applying heuristics locally.

### Concurrency & scaling notes
- Batching defaults: 16k boards per inference call; configurable via job config.
- Runner can use multiple async inference workers when gRPC (round-robin across channels) or run multi-gpu local inference via `--device` override.
- Shard writers flush to disk every ~5M rows (matching packer defaults) to amortise I/O.
- Resume support: `progress.json` records last completed shard + row offset; reruns skip existing shards unless `--overwrite` set.

### Multi-model support
- Session directory keyed by `{source_pool}/{model_tag}/{timestamp}` to avoid collisions.
- `metadata.db.annotation_sessions` stores `model_revision` (hash of init) so we can compare multiple sessions for the same pool.
- Server enumerates sessions so UI can toggle between them.

## Implementation Plan
1. **Proto & server plumbing**
   - Extend `proto/train_2048/inference/v1/inference.proto` with logits/value head fields; regenerate stubs (`uv run --project packages/infer_2048 ...`).
   - Update `packages/infer_2048/src/infer_2048/server.py` to populate new outputs and expose model-defined heads.
   - Add decoding helpers in `packages/core_2048/src/core_2048/infer.py` for local paths.
2. **Annotation library**
   - Create `packages/train_2048/src/train_2048/annotation/` with pool detection (`pool.py`), job runner (`job.py`), metrics (`metrics.py`), and writer (`writer.py`).
   - Implement CLI `bin/annotate_pool.py` (or `uv run python -m train_2048.annotation.cli`) that accepts `--pool`, `--session-out`, `--init` or `--grpc` address, `--batch-size`, `--value-heads` selection.
   - Ensure writer creates shards + schema + SQLite updates incrementally and supports resume.
3. **Server**
   - Scaffold FastAPI service in `packages/annotate_2048` (lightweight dependency set) with pagination endpoints.
   - Provide Docker/dev config for local run (`uv run --project packages/annotate_2048 python -m annotate_2048.server --session annotations/...`).
4. **UI integration hooks**
   - Document REST contract; stub out fetch layer inside `ui/annotation-viewer` (actual UI later).
5. **Testing & validation**
   - Unit tests for pool detection + dtype conversions (macroxue vs self-play) using synthetic shards.
   - Round-trip tests for value head packing/decoding.
   - Integration smoke: run annotator over a tiny pool, verify server endpoints match expected JSON.

## Open Questions / Risks
- **Model interface for value heads**: today encoder returns `(hidden, head_out)` with either logits tuple or single tensor. We need a consistent way for models to expose auxiliary heads (maybe through config or `model.value_heads()` hook). Decide before implementation.
- **Storage bloat**: duplicating logits and probabilities doubles size. We can make logits optional (`--store-logits`). Evaluate actual need before defaulting to `true`.
- **Self-play v1 teacher metadata**: without branch EVs we cannot compute disagreement vs teacher EV; only actual action is known. UI should differentiate between "action disagreement" and "EV disagreement".
- **Value head schemas**: scalars are easy; multi-bin outputs (e.g., expectimax per branch) may require nested storage. Option: dedicate separate `value_<name>-*.npy` shards when `dim > 8` to keep structured dtype manageable.
- **Server tech**: FastAPI is quick to ship but brings async overhead; Axum (Rust) may be preferable for binary streaming. We can start with FastAPI (Python ecosystem) and revisit if performance is insufficient.
- **Concurrency on network inference**: gRPC server currently synchronous per batch. If we need overlapping batches, consider opening multiple channels or adding streaming API later.

## Appendix: Example session layout
```
annotations/macroxue_d6_v1/model=v1_50m/2025-02-10T04-12-33Z/
├── source.json
├── metadata.db
├── schema.json
├── progress.json
├── predictions-000000.npy
├── predictions-000001.npy
├── metrics-000000.npy
└── value_heads-000000.npy
```
`source.json` captures `{ "pool": "datasets/macroxue/d6_v1", "pool_type": "macroxue", "init": "inits/v1_50m", "device": "cuda", "batch_size": 16384 }`.
