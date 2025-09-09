# Self-Play v1 (Lean): Single NPY + Embeddings + Index

Goal: a minimal, non-baroque design that reuses our existing NPY+SQLite rails, produces exactly one steps file, one embeddings file, and one index per session, and avoids extra fields/features until we need them.

What we will not do in v1: no rewards/done/legal, no q/int8 quant, no column-splitting into many NPYs, no time-based frequent flushes.

----------------------------------------

Principles
- Reuse rails: mirror the existing “steps.npy + metadata.db” pattern already used by training and examples.
- Minimal schema: only what critic training needs (tokens and run mapping). Labels derive from run-level highest_tile.
- Simple files: exactly one `steps.npy` (structured dtype), one `embeddings.bf16.npy` (optional), one `metadata.db` per recording session.
- High-throughput friendly: accumulate in memory and write in big chunks; rotate by step-count, not by aggressive timers.

----------------------------------------

Recording (engine)
- Config (Rust):
  - limits: `max_games?`, `max_steps?`, `max_ram_mb?`, `max_wall_ms?` (optional), `max_disk_mb?` (optional)
  - sampling: `sample_rate` (keep 1 of N steps)
  - output dir: `selfplay/<TS>_model=<tag>/`
- Buffering:
  - Per-game buffer holds its steps until terminal. On game end, append to the session buffer.
  - Session buffer is a single growable vec of step records (contiguous, row order = write order).
- Flush/rotation policy (40k steps/s):
  - Default: rotate when `session_steps >= 10_000_000` (≈250 s at 40k/s) or when `max_ram_mb` is near the cap.
  - No 500 ms timer. Optional safety timer (e.g., 30–60 s) only to bound RAM if step-rate dips; off by default.
  - On rotation or shutdown, write files once.
- Files written (per session):
  - `steps.npy` (structured array)
  - `metadata.db` (SQLite index with runs and minimal session meta)
  - `embeddings.bf16.npy` is produced later by a Python job; not written by the engine.

----------------------------------------

steps.npy (structured dtype)
- Schema (exact fields; fixed-width, no extras):
  - `run_id: <u64 or i64>` — stable per-game ID
  - `step_idx: <u32>` — 0-based index within run
  - `exps: (16,)u1` — board exponents (0..15), row-major
  - Optional (off by default): `action: u1` — 0..3 if we choose to keep it
- Notes:
  - This mirrors the spirit of the existing dataset rails but strips to the minimum. No `reward`, `done`, or `legal` fields.
  - Keeping exps in-place avoids recomputing tokens later and is compatible with our tokenizer.
  - Row order is the canonical index for alignment with `embeddings.bf16.npy`.

metadata.db (SQLite)
- Tables:
  - `runs(id INTEGER PRIMARY KEY, seed BIGINT, steps INT, max_score INT, highest_tile INT)`
  - `session(meta_key TEXT PRIMARY KEY, meta_value TEXT)` — JSON of engine/model/config hashes, timestamps
- Purpose:
  - Only run-level facts live here. Critic labels are derived from `highest_tile` at load time (per step via its `run_id`).
  - Provides the single “index” the loader needs and keeps the step file clean.

embeddings.bf16.npy (optional, offline)
- Shape: `[N, D]`, row-aligned to `steps.npy` (N rows, D = pooled dim).
- Dtype: `bfloat16` (native model dtype). No quantization in v1.
- Meta sidecar: `embeddings.json` with `{ model_sha, pooling, D, dtype: "bf16" }`.
- Produced by a separate Python job that loads `steps.npy`, runs the tower in eval, mean-pools hidden states (or uses CLS/pool if available), and saves the array.

----------------------------------------

Labels and targets (derived, not stored)
- For each run: compute `highest_tile` once (already available from engine end-state).
- Per step labels (for thresholds T ∈ {8192, 16384, 32768}): `1{highest_tile >= T}` broadcast across that run’s steps.
- The critic dataloader derives these booleans using `run_id -> highest_tile` lookup; nothing extra is serialized.

----------------------------------------

Critic dataloader (Python)
- Path: `src/train_2048/dataloader_critic.py`.
- Inputs: `root_dir` (session), `batch_size`, `num_workers`, optional `use_embeddings` flag.
- Loader behavior:
  - Load `steps.npy` (mmap) and `metadata.db`.
  - Build an in-memory map `run_id -> highest_tile` from SQLite (small).
  - If `use_embeddings` and `embeddings.bf16.npy` exists: mmap it and yield `(embedding, labels)`.
  - Else: yield `(tokens_from_exps, labels)` (slow path, but fine for sanity/development).
- Labels: vectorized thresholding of `highest_tile` to produce a 3-bit boolean per sample; optional class weights computed once per epoch.

----------------------------------------

Training the critic head
- Head: small MLP or linear probe `in_dim=D, out_dim=3`, logits per threshold.
- Loss: `BCEWithLogitsLoss` with optional `pos_weight` from dataset stats.
- Dtype: inputs can remain bf16 on device; cast to fp32 inside the head if needed for numerics (implementation detail, no file format impact).
- CLI: reuse `main.py` with a new config (e.g., `config/critic.example.toml`) that points to a session directory and sets `use_embeddings=true`.

----------------------------------------

Engine implementation notes (v1)
- One recorder thread per process that receives per-game step vectors and appends into a single session buffer (Vec<Step>).
- On rotation/end:
  - Write `steps.npy` once (structured dtype with the three fields).
  - Write `metadata.db` with `runs` (id, seed, steps, max_score, highest_tile) and `session` meta.
  - Atomically rename temp files into place.
- Defaults tuned for 40k steps/s:
  - `rotate_steps = 10_000_000` (≈250 s)
  - `sample_rate = 1` (keep all) — downsample if disk becomes a concern
  - No periodic time flush by default; optional `rotate_secs` can be set (e.g., 300 s) if desired.

----------------------------------------

Why this is simpler
- Exactly three artifacts per session: `steps.npy`, `embeddings.bf16.npy` (optional), `metadata.db`.
- No derived step fields (rewards/done/legal) and no quantization.
- No large matrix of tiny `.npy` columns, no new loader complexity.
- Reuses our existing SQLite filtering story and aligns with current config patterns.

----------------------------------------

Open edges (kept small on purpose)
- If NumPy bfloat16 serialization is a concern, we can store embeddings as `uint16` bit-patterns with a declared `dtype_bits = "bf16"` in `embeddings.json`; the loader views them as bf16 tensors on read. This is a fallback, not the default.
- If a single session grows too large, start a new session directory (session rotation) rather than introducing intra-session shard files.
- If later we need actions or legality for auxiliary tasks, we can add an optional `action: u1` field to `steps.npy` without disturbing existing readers.

----------------------------------------

Checklist
- Engine
  - [ ] Add minimal recorder (per-game buffer → session buffer).
  - [ ] Session rotation by step-count; final single-write `steps.npy` + `metadata.db`.
  - [ ] Store `highest_tile` per run only in SQLite.
- Python
  - [ ] `dataloader_critic.py` (mmap `steps.npy`, derive labels via SQLite, optional embeddings path).
  - [ ] Offline embedder that writes `embeddings.bf16.npy` row-aligned + `embeddings.json`.
  - [ ] `CriticHead` and training entry (config in `config/critic.example.toml`).
