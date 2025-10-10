# Board Annotation Rails

## Background

Existing self-play pools come in two primary layouts:

- **Macroxue packs** captured from expectimax rollouts with rich metadata (`board`, `branch_evs`, `valuation_type`). They are materialised as `steps-*.npy` shards plus `metadata.db`. Their dtype is defined in `crates/dataset-packer/src/lib.rs` and consumed through the `StepsDataset`/`BoardCodec` helpers in `packages/train_2048/src/train_2048/dataloader/steps.py` and `packages/train_2048/src/train_2048/tokenization/base.py`.
- **Self-play v1 pools** recorded from our lean engine (`docs/self-play-v1.md`). They reuse the same shard/SQLite pattern but store exponent boards only (`run_id`, `step_idx`, `exps`).

Inference already runs over these boards via the gRPC service in `packages/infer_2048/src/infer_2048/server.py` backed by the protobuf contract in `proto/train_2048/inference/v1/inference.proto`. Historically we only emitted per-move probabilities (possibly binned) and optional embeddings; there was no tooling to persist those predictions, combine them with teacher data, or surface them to the React/Vite viewer under `ui/annotation-viewer`.

## Goals

- Support offline annotation of step pools (Macroxue or self-play v1 format) with model predictions, storing results alongside source metadata.
- Keep batching/backpressure identical to the live self-play orchestrator by reusing the existing feeder pipeline.
- Produce artifacts that are mmap-friendly (NumPy structured arrays + SQLite) so downstream notebooks, CLI tools, or a future viewer can consume them easily.

## Non-Goals

- Training new heads or changing the model architecture.
- Replacing the existing inference gRPC API (all changes stay backward-compatible).
- Streaming live games; the pipeline is batch-oriented over saved sessions.

## Requirements & Constraints

- Input pools may come from multiple sessions and must cope with tens of millions of steps.
- Annotation should tolerate missing teacher metadata (self-play v1 lacks `branch_evs`).
- Storage must support multiple annotated models against the same pool without duplicating source boards.
- The schema should be extensible so we can surface additional value heads later.

## Implementation Overview

### Rust-first annotation pipeline

- `crates/game-engine` now exposes a reusable library surface (`src/lib.rs`) with a shared `pipeline` module for connection and feeder setup.
- A new binary, `annotation-engine`, reuses the gRPC feeder to score previously recorded boards instead of launching fresh games.
- CLI flags:
  - `--config <file>`: orchestrator TOML providing connection/batch settings (same format as the self-play runner).
  - `--dataset <dir>`: Macroxue pack or self-play v1 pool to annotate.
  - `--output <dir>`: destination directory for annotation shards.
  - `--overwrite`, `--limit N`: optional restart/smoke-test knobs.

### Input handling

- Dataset type is auto detected: we attempt to decode the first shard as `MacroxueStepRow`; on failure we fall back to `SelfplayStepRow`.
- Macroxue boards are unpacked from the packed nibble + overflow mask format back to `[16]` exponents; teacher move + legality bits are preserved. Self-play v1 pools already provide the exponent array but lack teacher metadata, so we emit sentinel values (`teacher_move = 255`, `legal_mask = 0`).

### Inference + batching reuse

- `pipeline::connect_inference` mirrors the self-play runner’s UDS/TCP connection logic, keeping all client configuration in one place.
- `pipeline::build_feeder` forwards to `Feeder::new`; both binaries now construct feeders through this helper so batching/backpressure and monitoring remain identical.
- Concurrency is capped at `inflight_batches * max_batch`. Optional inline embeddings are plumbed through the feeder but currently drained and dropped pending a storage format.
- Inference responses now carry `model_metadata.policy` (bin count, tokenizer identifier, vocab order when available) so clients can size their writers without hard-coding vocabularies.

### Output layout

- `dataset_packer::writer::StepsWriter` gained `with_prefix`, allowing the annotation job to reuse the shard writer while emitting files named `annotations-000000.npy`, …
- `AnnotationRow` schema:
  - `run_id: u32`
  - `step_index: u32`
  - `teacher_move: u8` (`0=Up..3=Right`, `255` unknown)
  - `legal_mask: u8` (UDLR bitmask)
  - `policy_kind_mask: u8` (bitset: `policy_p1`, `policy_logprobs`, `policy_hard` when present)
  - `argmax_head: u8`
  - `argmax_prob: f32`
  - `policy_p1: [f32; 4]` (probability mass of the final bin per branch)
  - `policy_logp: [f32; 4]` (natural log probabilities over branches)
  - `policy_hard: [f32; 4]` (teacher one-hot when available)
- `metadata.db` is copied into the output directory; `valuation_types.json` is preserved for Macroxue pools to keep enum alignment.
- `annotation_manifest.json` records the per-run `policy_kind_mask` so downstream tools can feature-detect available annotations without scanning shards.

### Reality check (2025-02)

- The annotation CLI ships as the `annotation-engine` binary (`crates/game-engine/src/bin/annotate.rs`) and delegates orchestration to `annotation::run` for dataset detection, metadata fan-out, shard writing, and manifest updates.
- `annotation::run` handles both Macroxue and self-play shards; Macroxue steps retain teacher metadata while self-play shards synthesize `teacher_move = 255` and `legal_mask = 0` before writing `AnnotationRow` via `dataset_packer::writer::StepsWriter::with_prefix`.
- gRPC integration already returns full bin distributions (`InferenceOutput::Bins`), but `annotation.rs` distills them to the per-branch final-bin mass (`policy_p1`/`policy_logp`) while ignoring embeddings unless explicitly handled elsewhere.
- `annotation-engine` now writes a matching `annotations-student-*.npy` shard for each `annotations-*.npy` file. The sidecar is a float32 tensor shaped `[steps_in_shard, 4, num_bins]` containing raw student probabilities for every branch/bin pair. `annotation_manifest.json` records `student_bins.num_bins` so downstream tooling can assert shapes without loading the shard.
- `crates/annotation-server/src/main.rs` eagerly loads Macroxue runs plus annotations into memory, exposes `/runs`, `/runs/{run_id}`, `/runs/{run_id}/disagreements`, and `/health`, and optionally hydrates `MacroxueTokenizerV2` when `--tokenizer` is provided; self-play v1 packs are not yet served through this rail.
- `ui/annotation-viewer` is live today: `App.tsx` wires TanStack Query-powered run browsing, a timeline scrubber, disagreement heat map, keyboard navigation, the `MoveInsights` table, and `TokenizationControls` that probe `/health` to toggle teacher token previews.
- `infer_2048` now advertises tokenizer metadata (head type, bin count, tokenizer name, vocab order when present) in every response so the Rust annotation path can size the student sidecar without guessing.

### Usage quick start

Start inference server:

```bash
uv run infer-2048 --init checkpoints/20251002_002432 --uds unix:/tmp/2048_infer.sock --device cuda
```

Then run the client:

```bash
cargo run -p game-engine --release --bin annotation-engine -- \
  --config config/inference/orchestrator.toml \
  --dataset datasets/raws/d7_test_v1 \
  --output annotations/d7_test_v1 \
  --overwrite
```

Generated artifacts:

- `annotations-*.npy`
- `metadata.db` (copy of source)
- `valuation_types.json` (Macroxue only)
- `annotation_manifest.json` (policy kind bitmasks per run)

### Serving annotations over HTTP

Run the Axum server against a base dataset plus its annotation directory:

```bash
cargo run -p annotation-server --release -- \
  --dataset datasets/macroxue/d6_10g_v1 \
  --annotations annotations/d6_10g_v1/model=v1_50m \
  --tokenizer tokenizer.json \
  --host 127.0.0.1 \
  --port 8080
```

The `--tokenizer` flag is optional and enables tokenization features for the UI.

- `GET /health` returns server status including tokenizer information if loaded and echoes `student_bins.num_bins` for quick capability detection.
- `GET /runs?page=1&page_size=25&min_score=500_000` returns paginated run summaries with optional filters on score, highest tile, and step counts.
- Each run summary now carries a `policy_kind_mask`, and both list/detail responses export a `policy_kind_legend` object so clients can decode bit assignments without hard-coding constants.
- List and detail responses also piggy-back a `student_bins` metadata block (`{ "num_bins": <usize> }`) whenever a student sidecar is present, so callers can assert tensor shapes up front.
- `GET /runs/{run_id}?offset=0&limit=200&tokenize=true` streams step slices including packed boards, teacher metadata, Macroxue branch EVs, the annotated policy payload, and (when present) `student_bins`. Each `student_bins` payload exposes `{ "num_bins": <usize>, "heads": [[f32; num_bins]; 4] }`, mirroring the on-disk tensor. When `tokenize=true` is specified and a tokenizer is loaded, each step also includes a `tokens` array with the tokenized branch values.

## Annotation Viewer (`ui/annotation-viewer`)

### Shipped functionality

- React/Vite SPA under `ui/annotation-viewer` uses TanStack Query for `/runs`, `/runs/:run_id`, and `/runs/:run_id/disagreements`, caching a sliding window (`WINDOW_SIZE = 257`) per run while coordinating keyboard and scrubber navigation (`App.tsx`).
- `RunsSidebar.tsx` renders run metadata (score, steps, highest tile, disagreement percentage) with collapse/expand toggles and selection state; it presently consumes the backend defaults (`page_size = 25`).
- The scrubber timeline paints a disagreement density strip, supports arrow-key plus `[`/`]` shortcuts, and lazily re-centers its fetch window as the operator pans forward/back.
- `MoveInsights.tsx` displays teacher EVs, normalized advantages, severity heuristics for disagreements, student `policy_p1`, and optional tokenizer chips when preview mode is active.
- `TokenizationControls.tsx` calls `/health`, surfaces tokenizer metadata when available, and toggles the tokenizer column without breaking when the backend reports “tokenizer not loaded.”

### Tokenization preview behaviour today

- Teacher tokens appear only when the annotation server loads `MacroxueTokenizerV2` and the client opts into `tokenize=true`; otherwise the tokenizer column remains hidden and the toggle reverts to “Annotations only.”
- Student annotations only contribute `policy_p1`/`policy_logp`, so the UI reconstructs a scalar probability column (and its log) but cannot show student bin ids, entropy, or vocab labels.
- Missing annotations degrade gracefully: steps without policy rows simply dim the Move Insights columns while preserving teacher EV context.

### Known gaps and backlog

- There is no deep linking or React Router integration; all viewer state is local, which limits shareability.
- `RunsSidebar` lacks virtualization and richer filter controls, so very large pools become unwieldy and rely solely on backend query params.
- Tokenization toggles do not differentiate “teacher token unavailable” from “student bins unavailable,” which will matter once we ingest student-only experiments.
- Student distribution visualizations (bin histograms, entropy chips, KL metrics) are absent, preventing comparisons envisioned in `tokenization_v2.md`.
- Developer ergonomics items (proxying via `vite.config.ts`, Storybook-style fixtures, modularized CSS) are still open from the original roadmap.
- Accessibility and polish features—focus outlines, keyboard tutorial overlays, data export helpers—remain future enhancements.

## Current Limitations & Future Work

- The API now ships full student distributions, but the viewer still renders only aggregate policy columns; we need UI hooks for bin charts, entropy, and deltas before annotators benefit.
- Sidecar metadata stops at `student_bins.num_bins`; provenance (model init, tokenizer hash, vocab ordering) remains manual and should move into the manifest or `/health`.
- The annotator now reorders concurrent results via an in-memory buffer so `annotations-*.npy` and `annotations-student-*.npy` stay aligned; further tuning may still be needed for very large inflight windows.
- `annotation-server` still targets Macroxue packs, loads everything eagerly, and lacks streaming pagination or self-play v1 ingestion.
- Inline embeddings are drained in `annotation::run` but never persisted, blocking downstream embedding experiments.

## Codex' Proposal (student bins follow-up)

### Objectives

- Treat the new `[step, head, bin]` sidecar as first-class data: surface it cleanly in the API and viewer without reintroducing the old bookkeeping overhead.
- Recover some throughput by reintroducing bounded parallelism while keeping deterministic ordering between `annotations-*.npy` and `annotations-student-*.npy`.
- Capture lightweight provenance (model init, tokenizer hash) alongside `num_bins` so operators can diff runs without manual notes.

### Backend adjustments

- Extend `/runs/:id` with optional knobs (`student_topk`, `student_mode=full|topk`) if payload trimming becomes necessary, but keep the on-disk tensor untouched.
- Stamp inference/model provenance into `annotation_manifest.json` and echo it through `/health` so the UI can label runs without extra calls.
- Tune the ordered buffer (batch flush size, inflight limits) to balance memory usage and throughput once larger pools are back under sustained load.

### UI work

- Teach `TokenizationControls`/`MoveInsights` to plot the four per-branch distributions (stacked bars or sparklines), expose entropy and argmax deltas, and handle student-only vs. teacher-only permutations gracefully.
- Highlight disagreements where the student argmax differs from the teacher move or policy head, and surface quick filters for “entropy spikes” or “low-confidence wins.”

### Nice-to-haves

- Persist inline embeddings (opt-in) so later tools can join similarity search with the existing annotation flow.
- Add streaming pagination to `/runs/:id` plus list virtualization in the viewer so multi-million-step packs stay responsive.
