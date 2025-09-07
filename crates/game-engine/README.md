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
- cargo run -p game-engine -- --config config/inference/top-score.toml

Config
- See CONFIG.md for all keys. Minimal required fields:
  - `[orchestrator.connection]` — set exactly one of:
    - `uds_path = "/tmp/2048_infer.sock"`
    - `tcp_addr = "http://127.0.0.1:50051"`
  - `[orchestrator.batch]` (defaults are sensible; tune later)

What it does
- Starts a Feeder with micro-batching and a sliding in-flight window.
- Spawns up to `max_concurrent_games` per-game actors (GameActor) and runs `num_seeds` total games.
- Each actor:
  - Submits current board (16 exponents) via Feeder and awaits probabilities.
  - Selects a move using max-p1 (probability of the 1-bin) with a legality mask.
  - Applies the move via `ai_2048` and continues until the game ends.

Notes
- UDS and TCP are both supported. Prefer UDS for local single-node runs (lower overhead, no TCP stack).
- The Feeder integrates the response router (per-item oneshots) to keep the design simple.
