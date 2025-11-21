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

## Rejection Sampling (proposal)

### Current implementation snapshot

The training pipeline now accepts a minimal rejection sampler defined directly in the training
config. Enable it by adding a `[dataset.rejection]` stanza; no external cache building is required.

```toml
[dataset.rejection]
annotation_dir = "annotations/d6_1440_v2_50m_20251015"
seed = 20251015

[[dataset.rejection.filters]]
id = "student_errors"
name = "student_wrong_p1"
weight = 0.9

[[dataset.rejection.filters]]
id = "anneal_random"
name = "passthrough"
weight = 0.1
anneal_until_epoch = 6
```

On every epoch the sampler logs how many samples were drawn per filter and whether duplication was
necessary (e.g., when the requested proportion exceeds available student errors).

Offline rejection sampling will piggyback on the annotation sidecars and produce training-ready indices without rewriting the dataset shards. The immediate target is teacher-supervised training; we only need the summary rows and `policy_p1`, but we leave enough structure to support richer disagreement heuristics later.

### Filter surface

We will introduce a small `FilterSpec` abstraction in Python (`train_2048.annotation.filters`) that converts a named filter plus optional parameters into a boolean mask over annotation rows. Initial support covers `student_wrong_p1`, defined as `teacher_move` being legal and not matching the `argmax_head` inferred from `policy_p1`. The dispatcher accepts a dataclass payload (`FilterSpec(name: str, params: dict[str, Any] | None = None)`) so future filters can reuse thresholds or combine features without changing the call-site shape. Filters return an `np.ndarray` of indices compatible with `StepsDataset`, and we cache results under `annotations/<run>/filters/<filter_id>.npy` to avoid repeated scans.

### Mixing and annealing

Data mixes are described in TOML under a new `[rejection]` section consumed by the training CLI. Each entry names the dataset, the paired annotation directory, the filter spec, and a target fraction. Optional `phase.until_epoch` keys allow annealing: when the current epoch exceeds the boundary we advance to the next mix definition. Fractions are normalized per phase, then converted into target sample counts by multiplying against the desired batch budget.

```toml
[rejection]
cache_dir = "datasets/rejection_cache"

[[rejection.phases]]
until_epoch = 10
[[rejection.phases.mix]]
dataset = "datasets/teacher_depth8"
annotation = "annotations/teacher_depth8"
filter = { name = "student_wrong_p1" }
weight = 0.85
[[rejection.phases.mix]]
dataset = "datasets/teacher_depth8"
annotation = "annotations/teacher_depth8"
filter = { name = "all" }
weight = 0.10
[[rejection.phases.mix]]
dataset = "datasets/teacher_depth6"
annotation = "annotations/teacher_depth6"
filter = { name = "all" }
weight = 0.05

[[rejection.phases]]
# Falls back to the last phase when `until_epoch` is omitted.
[[rejection.phases.mix]]
dataset = "datasets/teacher_depth8"
annotation = "annotations/teacher_depth8"
filter = { name = "student_wrong_p1" }
weight = 0.25
[[rejection.phases.mix]]
dataset = "datasets/teacher_depth8"
annotation = "annotations/teacher_depth8"
filter = { name = "all" }
weight = 0.50
[[rejection.phases.mix]]
dataset = "datasets/teacher_depth6"
annotation = "annotations/teacher_depth6"
filter = { name = "all" }
weight = 0.25
```

At runtime we materialise per-mix index arrays (sampling with replacement when the filter set is smaller than the requested draw) and wrap them in a lightweight `WeightedBatchSampler`. Annealing phases swap the active sampler at epoch boundaries, so schedulers only need to reinitialise once per phase.

### Runtime workflow

1. Resolve the mix phase from the training epoch; lazily load (or reuse) cached filter indices.
2. Join the step indices back to `StepsDataset` through `run_id`/`step_index` pairs pulled from the annotation summary rows.
3. Materialise a concatenated index buffer per mix weight, shuffling in place before every epoch to preserve randomness.
4. Deliver balanced batches by chunking the mixed buffer in training order while keeping a record of the originating mix for logging and metrics.

### Open questions

- How should we expose per-source sampling stats to the trainer logs so we can monitor annealing drift without adding W&B?
- Do we need a guardrail when the requested fraction exceeds the number of disagreement steps, or is sampling with replacement acceptable for early experiments?
- Should we extend the `annotation_manifest.json` to advertise available filter inputs (e.g., `policy_logp`, `argmax_prob`) before the Python side assumes they exist?
- Are depth-specific mixes better referenced by a short alias in the config (e.g., `source = "depth8.recent"`) instead of raw paths for long-term maintainability?

## Counter-Proposal: Rejection Sampling Design

After reviewing the original proposal, the codebase, and the existing training pipeline, here's my counter-proposal for rejection sampling implementation. I largely agree with the original direction but suggest some refinements for cleaner integration and better extensibility.

### What the Original Got Right

1. **Filters as cached index arrays**: Precomputing filter results and caching them as `.npy` files is smart—avoids repeated scans and integrates naturally with the existing shard/mmap infrastructure.

2. **TOML-based mix specification**: Using `[[rejection.phases]]` to describe data mixes aligns perfectly with the existing config patterns in `config.py` and keeps experiment definitions declarative.

3. **Annealing via phases**: Epoch-based phase transitions are simple and fit naturally into the existing training loop structure.

4. **FilterSpec abstraction**: A dataclass-based filter spec (`FilterSpec(name: str, params: dict[str, Any])`) provides the right balance of simplicity and extensibility.

### Suggested Changes

#### 1. **Simpler mix identity and caching**

Instead of computing a filter ID from hashing the filter spec, use a **user-provided mix ID** in the TOML. This makes debugging easier and gives operators explicit control over cache invalidation:

```toml
[[rejection.phases.mix]]
id = "d8_wrong_p1"  # explicit cache key
dataset = "datasets/teacher_depth8"
annotation = "annotations/teacher_depth8"
filter = { name = "student_wrong_p1" }
weight = 0.85
```

Cache path becomes: `{cache_dir}/{mix_id}.npy` instead of `{annotation_dir}/filters/{filter_hash}.npy`. This also centralizes all rejection artifacts under one `cache_dir` rather than scattering them across annotation directories.

#### 2. **Decouple filter implementation from data loading**

The original proposes `train_2048.annotation.filters` as a Python module. I suggest a slightly different structure:

- **`train_2048/dataloader/filters.py`**: Core filter registry and dispatcher
- **`train_2048/dataloader/rejection.py`**: Mix resolution, phase management, and sampler construction
- **CLI tool: `build-rejection-cache`**: Standalone script to precompute and cache filter indices before training

This separation keeps the dataloader module cohesive and makes it trivial to dry-run filter logic without launching a full training job.

**Example CLI usage:**

```bash
uv run build-rejection-cache \
  --config config/rejection/depth8-mix.toml \
  --validate  # Check all filters work, report stats, but don't cache
```

#### 3. **Filter return value: structured metadata**

Instead of returning just an index array, filters should return a small dataclass:

```python
@dataclass
class FilterResult:
    indices: np.ndarray      # global step indices
    metadata: dict[str, Any] # optional stats (count, coverage, etc.)
```

This lets us log per-filter statistics (e.g., "student_wrong_p1 matched 12.3% of steps") directly into the training logs or W&B without re-scanning data. The metadata can include:

- Match count
- Coverage percentage
- Min/max step indices (for sanity checks)

#### 4. **Join annotations to steps via a lightweight index, not run_id/step_index pairs**

The original proposes joining back to `StepsDataset` through `(run_id, step_index)` pairs. This requires either:

- Loading metadata.db to build a lookup map, or
- Assuming annotations and steps are perfectly aligned (which they are!)

Since the annotation engine already guarantees alignment (see `annotation.rs:106-370`—the ordered buffer ensures deterministic write order), we can **directly use annotation shard indices as global step indices**. No joins needed.

**Concretely:**

1. Filter scans `annotations-*.npy` shards and collects matching row indices
2. Returns global annotation indices (0-based across all annotation shards)
3. Training dataloader treats annotation indices == step indices (since they're 1:1 aligned)

This assumes we're only filtering annotated datasets, which is correct for rejection sampling (you need student predictions to determine disagreements).

#### 5. **Explicit "passthrough" filter instead of special-casing `all`**

The original uses `filter = { name = "all" }` to sample randomly from an entire dataset. I suggest being more explicit:

```toml
[[rejection.phases.mix]]
id = "d8_random"
dataset = "datasets/teacher_depth8"
annotation = "annotations/teacher_depth8"
filter = { name = "passthrough" }  # or "random_all"
weight = 0.10
```

This makes it clear that you're not applying a disagreement filter. Under the hood, `passthrough` just returns `np.arange(dataset_length)`.

#### 6. **Sampling with replacement: make it configurable per mix**

When a filter matches fewer steps than the requested draw size, the original proposal mentions "sampling with replacement." I suggest making this **opt-in per mix**:

```toml
[[rejection.phases.mix]]
id = "d8_wrong_p1_upsampled"
dataset = "datasets/teacher_depth8"
annotation = "annotations/teacher_depth8"
filter = { name = "student_wrong_p1" }
weight = 0.25
upsample = true  # sample with replacement if filter size < requested draw
```

Default behavior (`upsample = false`) should **warn and clip** instead of silently duplicating data. This catches config errors early.

#### 7. **Config validation and dry-run mode**

Add a `--validate-config` flag to the training CLI that:

1. Resolves all phases and mixes
2. Runs all filters and reports match counts
3. Computes expected samples per epoch per mix
4. Prints warnings about upsampling or empty filters
5. Exits without starting training

This helps catch TOML typos and filter logic bugs before burning GPU hours.

#### 8. **Per-source batch logging**

To monitor annealing and data balance, emit per-mix sample counts as part of the standard training logs (not just W&B). For example:

```
[Epoch 1, Step 500] loss=0.234 acc=0.821 mix_samples={d8_wrong_p1: 425, d8_random: 50, d6_anneal: 25}
```

This can be done cheaply by maintaining a counter dict in the sampler and flushing it every `report_every` steps.

### Proposed TOML Schema (Refined)

```toml
[rejection]
# Centralized cache for all filter indices
cache_dir = "datasets/rejection_cache"
# Validation mode: compute stats but don't create mix samplers
validate_only = false

[[rejection.phases]]
# Optional: specify until which epoch this phase runs (omit for final phase)
until_epoch = 10

[[rejection.phases.mix]]
# Unique identifier for caching and logging
id = "d8_errors"
# Source dataset and annotation directory
dataset = "datasets/teacher_depth8"
annotation = "annotations/teacher_depth8"
# Filter specification
filter = { name = "student_wrong_p1" }
# Mix weight (normalized within phase)
weight = 0.85
# Optional: allow sampling with replacement if filter size < draw size
upsample = false

[[rejection.phases.mix]]
id = "d8_baseline"
dataset = "datasets/teacher_depth8"
annotation = "annotations/teacher_depth8"
filter = { name = "passthrough" }
weight = 0.10

[[rejection.phases.mix]]
id = "d6_anneal"
dataset = "datasets/teacher_depth6"
annotation = "annotations/teacher_depth6"
filter = { name = "passthrough" }
weight = 0.05

[[rejection.phases]]
# Second phase (runs from epoch 11 onward)
[[rejection.phases.mix]]
id = "d8_errors"  # reuse cached filter
weight = 0.25
upsample = true

[[rejection.phases.mix]]
id = "d8_baseline"
weight = 0.50

[[rejection.phases.mix]]
id = "d6_anneal"
weight = 0.25
```

### Implementation Sketch

**`train_2048/dataloader/filters.py`:**

```python
from dataclasses import dataclass
import numpy as np

@dataclass
class FilterResult:
    indices: np.ndarray
    metadata: dict[str, Any]

class FilterRegistry:
    _filters = {}

    @classmethod
    def register(cls, name: str):
        def decorator(fn):
            cls._filters[name] = fn
            return fn
        return decorator

    @classmethod
    def apply(cls, name: str, annotation_rows: np.ndarray, params: dict) -> FilterResult:
        if name not in cls._filters:
            raise ValueError(f"Unknown filter: {name}")
        return cls._filters[name](annotation_rows, params)

@FilterRegistry.register("student_wrong_p1")
def filter_student_wrong_p1(rows: np.ndarray, params: dict) -> FilterResult:
    """Select steps where teacher move is legal but student argmax disagrees."""
    teacher_move = rows["teacher_move"]
    legal_mask = rows["legal_mask"]
    argmax_head = rows["argmax_head"]

    # Teacher move must be valid (<4) and legal
    valid = teacher_move < 4
    legal = ((legal_mask >> teacher_move) & 1) == 1
    # Student argmax differs from teacher
    disagrees = argmax_head != teacher_move

    mask = valid & legal & disagrees
    indices = np.where(mask)[0]

    return FilterResult(
        indices=indices,
        metadata={
            "count": len(indices),
            "coverage": len(indices) / len(rows) if len(rows) > 0 else 0.0,
        }
    )

@FilterRegistry.register("passthrough")
def filter_passthrough(rows: np.ndarray, params: dict) -> FilterResult:
    """Return all indices (no filtering)."""
    indices = np.arange(len(rows))
    return FilterResult(
        indices=indices,
        metadata={"count": len(indices), "coverage": 1.0}
    )
```

**`train_2048/dataloader/rejection.py`:**

```python
from pathlib import Path
import numpy as np
from .filters import FilterRegistry, FilterResult

def load_or_build_filter_cache(
    cache_dir: Path,
    mix_id: str,
    annotation_dir: Path,
    filter_name: str,
    filter_params: dict,
) -> FilterResult:
    """Load cached filter indices or compute and cache them."""
    cache_path = cache_dir / f"{mix_id}.npy"
    meta_path = cache_dir / f"{mix_id}.json"

    if cache_path.exists() and meta_path.exists():
        # Load from cache
        indices = np.load(cache_path)
        with open(meta_path) as f:
            metadata = json.load(f)
        return FilterResult(indices=indices, metadata=metadata)

    # Load annotation shards and apply filter
    annotation_rows = load_annotation_shards(annotation_dir)
    result = FilterRegistry.apply(filter_name, annotation_rows, filter_params)

    # Cache results
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, result.indices)
    with open(meta_path, "w") as f:
        json.dump(result.metadata, f)

    return result
```

### Summary of Changes from Original Proposal

| Aspect             | Original                      | Counter-Proposal                              |
| ------------------ | ----------------------------- | --------------------------------------------- |
| Cache location     | `annotations/<run>/filters/`  | `{cache_dir}/{mix_id}.npy`                    |
| Mix identification | Implicit filter hash          | Explicit `id` field                           |
| Filter return type | `np.ndarray`                  | `FilterResult(indices, metadata)`             |
| "All" filter       | `filter = { name = "all" }`   | `filter = { name = "passthrough" }`           |
| Upsampling         | Implicit when needed          | Explicit `upsample` flag per mix              |
| Join strategy      | `(run_id, step_index)` lookup | Direct index alignment (annotations == steps) |
| Validation         | Runtime errors                | CLI `--validate-config` mode                  |
| Logging            | W&B-only (question)           | Per-mix counters in standard logs             |
| CLI tooling        | None                          | `build-rejection-cache` for pre-caching       |

### Open Questions (Updated)

1. **Should we allow filtering on non-annotated datasets?** Current design assumes rejection sampling only makes sense with annotations. If we want to support "random sample from depth-6 without annotations," we'd need to adjust the filter to work directly on step shards.

2. **How to handle annotation/step length mismatches?** Should we validate that `len(annotations) == len(steps)` at cache build time and error loudly if they don't match?

3. **Per-mix shuffle behavior?** Should each mix's indices be shuffled before sampling, or should we shuffle the concatenated mix buffer? Original proposal mentions "shuffling in place before every epoch"—where exactly?

4. **Should filter params support templating?** For example, `filter = { name = "top_k_disagreements", params = { k = 10000 } }` could be useful for selecting only the N worst mistakes. Worth including now or defer?

5. **Backwards compatibility**: If `[rejection]` section is absent, training should fall back to existing `[dataset]` behavior. Should we warn if both are present, or allow `rejection` to override `dataset`?
