Game Engine Orchestrator (Rust)

Overview
- Tokio-based orchestrator that runs many 2048 games concurrently, batches boards via a Feeder, and calls a Python inference server over gRPC.
- The gRPC API returns per-head probabilities over bins (float32); the Rust side selects moves (max-p1) and applies them.

Prereqs
- A running inference server that implements proto `train_2048.inference.v1.Infer`.
- Connection options (choose one in your TOML under `[orchestrator.connection]`):
  - UDS (recommended locally): `uds_path = "/tmp/2048_infer.sock"`
  - TCP: `tcp_addr = "http://127.0.0.1:50051"`

Build
- cargo build -p game-engine

Run
- Start the server (example):
  - `uv run infer-2048 --init inits/v1_pretrained_50m --uds unix:/tmp/2048_infer.sock --device cpu --compile-mode none`
- Start the client (example):
  - `cargo run -p game-engine -- --config config/selfplay/collect-10m.toml`

Config
- See CONFIG.md for all keys. Minimal required fields:
  - `[orchestrator.connection]` — set exactly one of:
    - `uds_path = "/tmp/2048_infer.sock"`
    - `tcp_addr = "http://127.0.0.1:50051"`
  - `[orchestrator.batch]` (defaults are sensible; tune later)

Recording (optional)
- To enable saving, set `[orchestrator.report].session_dir`. If omitted, no files are written.
- Files written when buffers flush (on size/teardown):
  - `steps-000001.npy`, `steps-000002.npy`, …: structured dtype rows `[('run_id','<u8'), ('step_idx','<u4'), ('exps','|u1',(16,))]`. Default shard size is 1_000_000 rows (`report.shard_max_steps`).
  - `embeddings-000001.npy`, … (optional): float32 `[N, D]` shards when `orchestrator.inline_embeddings = true` and embeddings are paired.
  - `metadata.db`: SQLite with tables:
    - `runs(id INTEGER PRIMARY KEY, seed BIGINT, steps INT, max_score INT, highest_tile INT)`
    - `session(meta_key TEXT PRIMARY KEY, meta_value TEXT)`

Safety knobs (optional)
- `[orchestrator.report]`
  - `max_ram_mb` (approx.): stop collecting new rows if in-memory buffer estimate exceeds this.
  - `max_gb` (approx.): skip writing an artifact if its estimated size exceeds this many GB.
  - `shard_max_steps` (default 1_000_000): flush steps/embeddings once the buffer reaches this many paired rows.
- `[orchestrator]`
  - `inline_embeddings`: request embeddings inline and write a shard when complete.
  - `fixed_seed`: if set, seeds are deterministic; otherwise full randomness is used.

What it does
- Starts a Feeder with micro-batching and a sliding in-flight window.
- Spawns up to `max_concurrent_games` per-game actors (GameActor) and runs until `num_seeds` games and/or `max_steps` total steps (whichever comes first).
- Each actor:
  - Submits current board (16 exponents) via Feeder and awaits probabilities.
  - Selects a move using max-p1 (probability of the 1-bin) with a legality mask.
  - Applies the move via `ai_2048` and continues until the game ends.

Notes
- UDS and TCP are both supported. Prefer UDS for local single-node runs (lower overhead, no TCP stack).
- The Feeder integrates the response router (per-item oneshots) to keep the design simple.
