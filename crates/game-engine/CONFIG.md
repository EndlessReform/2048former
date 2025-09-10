Rust orchestrator configuration (inference)

Overview
- This config controls the orchestrator’s connection to the inference server and the micro-batching feeder behavior. It does not expose backend- or dtype-specific knobs; the wire/API remain backend-agnostic.

Top-level keys
- num_seeds (u32, optional): Number of games to run in total.
- max_steps (u64, optional): Total step budget across all games. If both are set, stop when either is reached.
- max_concurrent_games (u32): Max per-actor games scheduled concurrently.
- max_retries (u32): Transient RPC retry attempts (batch-level).
- sampling (table): Strategy and parameters for move selection.
  - strategy (string): One of "Argmax", "Softmax", "TopPTopK", "TailAgg", "TailAggConf".
  - temperature (f64, optional): For Softmax/TopPTopK; default 1.0.
  - top_p (f64, optional): For TopPTopK; default 0.8.
  - top_k (usize, optional): For TopPTopK; default 2.
  - alpha_p2, beta_p3 (f64, optional): For TailAgg simple; defaults 0.02 and 0.0.
  - tail_bins (usize, optional): Advanced TailAgg extra bins; default 0 (disabled).
  - tail_decay (f64, optional): Advanced TailAgg decay; default 0.5.
  - conf_alpha (f64, optional): For TailAggConf; max p2 weight at zero margin; default 0.20.
  - conf_beta (f64, optional): For TailAggConf; slope for margin-to-weight mapping; default 10.0.
  - conf_gamma (f64, optional): For TailAggConf; exponent shaping for decay; default 1.0.
  - start_gate (u64, optional): Step to start applying non-argmax sampling; default 0.
  - stop_gate (u64, optional): Step to stop applying non-argmax sampling; default none.
- orchestrator (table): Group for connection, batching, and recording settings.

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
orchestrator.inline_embeddings (bool, default false)
- If true, request embeddings inline and write a single shard `embeddings-000001.npy` if complete.

orchestrator.fixed_seed (u64, optional)
- Base seed for deterministic runs (`base + i`). If omitted, a default constant base is used for reproducibility unless `random_seeds=true`.

orchestrator.random_seeds (bool, default false)
- Opt-out: when true, ignore fixed/default base seed and use full randomness (recommended for dataset collection).

orchestrator.report
- session_dir (string, optional): Directory for saving artifacts. If omitted, nothing is written.
- results_file (string, optional): JSONL of per-game summaries.
- max_ram_mb (usize, optional): Approximate in-memory buffer cap; stop collecting new rows if exceeded.
- max_gb (float, optional): Approx disk failsafe per artifact; skip writing if estimated size exceeds this many GB.
