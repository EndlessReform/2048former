Rust orchestrator configuration (inference)

Overview
- This config controls the orchestrator’s connection to the inference server and the micro-batching feeder behavior. It does not expose backend- or dtype-specific knobs; the wire/API remain backend-agnostic.

Top-level keys
- num_seeds (u32): Number of games to run in total (for benchmarks).
- max_concurrent_games (u32): Max per-actor games scheduled concurrently.
- max_retries (u32): Transient RPC retry attempts (batch-level).
- sampling.strategy (string): Sampling strategy for bins (e.g., "argmax").
- orchestrator (table): Group for connection and batching settings.

orchestrator.connection
- uds_path (string, optional): Path to Unix domain socket (e.g., "/tmp/2048_infer.sock").
- tcp_addr (string, optional): TCP endpoint for the server (e.g., "http://127.0.0.1:50051").
  Exactly one of uds_path or tcp_addr should be set.

orchestrator.batch
- flush_us (u64, default 250): Time-boxed flush interval for micro-batches (microseconds).
- target_batch (usize, default 512): Target batch size when building micro-batches.
- max_batch (usize, default 1024): Upper bound for burst filling under load.
- inflight_batches (usize, default 2): Sliding window of batches in flight (credits).
- per_game_inflight (usize, default 16): Per-game cap on in-flight items for fairness.
- queue_cap (usize, default 65536): Capacity of the bounded global request queue.

Notes
- The server API returns per-head probabilities over bins. Sampling and tree/strategy logic stay on the Rust side.
- No dtype/backend/cuda-graphs hints are present in this config or in the gRPC contract.
