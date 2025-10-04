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
- Building the front-end UI; the current work focuses on backend rails.
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
  --host 127.0.0.1 \
  --port 8080
```
- `GET /runs?page=1&page_size=25&min_score=500_000` returns paginated run summaries with optional filters on score, highest tile, and step counts.
- Each run summary now carries a `policy_kind_mask`, and both list/detail responses export a `policy_kind_legend` object so clients can decode bit assignments without hard-coding constants.
- `GET /runs/{run_id}?offset=0&limit=200` streams step slices including packed boards, teacher metadata, Macroxue branch EVs, and the annotated policy payload (mask + `policy_p1`, `policy_logp`, `policy_hard`). The response shape leaves room for future value heads.

## Proposed Annotation Viewer

### Experience goals
- Keep the interaction rhythm true to the original 2048: a single focus board with butter-smooth tile slides, warm amber tiles, and playful easing while still foregrounding analytics.
- Help annotators compare teacher trajectories and model beliefs at a glance without falling back to notebooks.
- Scale to tens of thousands of runs by leaning on the existing REST slices (`/runs`, `/runs/:run_id`) with pagination and streaming fetches.

### Layout concept
- **Left rail** — run browser and filters. Mirrors `/runs` query knobs (`min_score`, `min_highest_tile`, `sort`) with Tailwind form controls and emits debounced requests.
- **Center stage** — the 4×4 2048 board rendered with reusable `Tile` components. Tile colors follow the classic palette (`#eee4da`, `#ede0c8`, `#f2b179`, …) with Tailwind CSS variables (`tile-2`, `tile-4`, …) defined in `tailwind.config.js` so the grid feels instantly familiar.
- **Right rail** — move insights: branch EV sparklines, teacher vs. model callouts, and annotation metadata.
- **Bottom timeline** — step scrubber with keyboard shortcuts (←/→) and momentum scrolling. Prefetches the next window of `/runs/:run_id?offset=…&limit=…` so anims stay snappy.

### Component sketch
1. `RunsSidebar` — infinite list backed by `react-query` (or TanStack Query) with virtualization. Displays score, highest tile, step count badges, and a thumbnail sparkline of branch EV spread.
2. `RunSummaryHeader` — seeds the main pane with run-level stats (score, highest tile, step count, seed) directly from `RunDetailResponse.run`.
3. `BoardView` — 16-tile grid using CSS grid + Tailwind utility classes. Animates value changes by transitioning transform/opacity; ghost tiles show impending spawn (if we derive it later).
4. `MoveCallouts` — compares `teacher_move` against `annotation.argmax_head`. When they disagree, highlight the board background in the move’s direction color (e.g., amber for teacher, teal for model) and render a small arrow overlay.
5. `BranchEVChart` — horizontal mini-bars for each legal move. Uses `branch_evs` if present; fades disabled moves (mask bit set to 0). Tooltips surface EV, `policy_p1`, and `hard_target` weight so annotators can see both model softness and teacher hard targets side-by-side.
6. `StepControls` — pagination state machine that rewrites the `offset`/`limit` query while retaining a small client-side cache (`Map<runId, StepBuffer>`). Includes “jump to turn” input and “auto-play” toggle that advances every N ms.
7. `ThemeProvider` — wraps Tailwind with custom fonts (`'Clear Sans', 'Helvetica Neue', sans-serif`) and adds a subtle linen background gradient so the UI evokes the original 2048 vibe without copying it pixel-for-pixel.

### Data + networking plan
- Use a light API layer (`src/api/client.ts`) that exposes `listRuns(query: RunsQuery)` and `getRun(runId, params: StepsQuery)`, mirroring `RunsQuery` / `StepsQuery` from the Axum handlers. Encode filters in the URL search params to keep the UI shareable.
- Cache run slices in memory; when the user scrubs past the loaded window, issue another `getRun` call with the new offset. Because responses include pagination totals, we can render progress bars and disable overflow navigation instantly.
- Extend `AnnotationRow` and `AnnotationPayload` so the server emits **both** policy traces: the soft head probabilities (`policy_p1`) and a four-branch hard target vector derived from the teacher move (one-hot for known teachers, `null` when unavailable). Include a `policy_kind` enum so the client can badge rows when only one representation is present.
- Handle missing annotations gracefully: when `annotation` is `null`, show a “Pending model prediction” badge and dim the policy columns.
- Surface server errors (404 for missing run) inline in the right rail with a retry button that replays the request.

### Visual language
- Tiles pop out via drop shadows and rounded corners, echoing the New York Times 2048 launch aesthetics. Tailwind utilities (`shadow-tile`, `rounded-xl`) keep it consistent.
- Use a restrained neutral palette for chrome (`#faf8ef` background, `#776e65` text) so high-value moves (e.g., `policy_p1 > 0.7`) can use accent gradients (`from-amber-400 to-orange-500`).
- Animate state transitions with Tailwind’s `transition` utilities; for the board, apply a short (120 ms) ease-out to tile shifts and a slightly longer (200 ms) fade for annotation overlays to mimic tile merges.

### Implementation notes
- Scaffold routing with React Router so `/runs/:runId/:stepIndex` deep-links reproduce the precise board state and filters.
- Add a reusable `useAnnotationFeed` hook that wraps TanStack Query + context for keyboard navigation.
- Extend `vite.config.ts` for proxying `/runs` to the Axum backend in dev (`/api` prefix). In production, serve the static bundle next to the Rust binary and point requests at the same origin.
- Ship a Storybook-like `MockProvider` that feeds snapshot JSONs (captured via `curl`) into components for visual regression without hitting the backend.
- Co-locate Tailwind component classes via `@apply` in `App.css` until the UI graduates into dedicated feature folders (`features/runs`, `features/board`, `features/insights`).
- Update the schema writer and server DTOs to persist branch-wise log probabilities (`policy_logp: [f32; 4]`) and teacher one-hot annotations (`policy_hard: [f32; 4]`). The UI should prefer log space for numerical stability, deriving `policy_p1 = logp.exp()` only when rendering percentages.

### Future polish
- Layer in keyboard tutorials (press `?` to show shortcuts) and quick filters (“Show disagreements”, “High EV swings”).
- Use Web Workers to precompute diff stats (e.g., teacher EV delta) so the main thread stays smooth while stepping fast.
- Offer export actions that download the active run slice as JSON/CSV for cross-tool analysis.

## Current Limitations & Future Work
- Only the top-bin probability (`p1`) per move is stored today. Capturing the full per-bin distribution/logits remains on the roadmap.
- Auxiliary value heads are not exposed yet; once the inference server returns them we can extend the row schema or add sibling shards using the same `StructuredRow` plumbing.
- The Axum server keeps everything in-memory and serves JSON only; the React/Vite viewer is still pending.
- Provenance files (`source.json`, schema manifests) are not emitted yet and will be added when the viewer/REST work begins.
