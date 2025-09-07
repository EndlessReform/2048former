# High‑Throughput 2048 Inference Engine

*A compact, two‑day, GPU‑saturating design for a transformer EV predictor. Written for an 8pm‑zonked grad student who just needs it to work.*

---

## 0) TL;DR Architecture

Two processes with a **single, narrow RPC** between them. Rust owns concurrency, batching, search, and backpressure. Python owns a **dumb** forward pass.

```
┌──────────────────────────┐      UDS gRPC (one unary call)      ┌─────────────────────────┐
│ Rust Orchestrator        │  Forward([B×16 exponents])  ───────▶ │ Python Inference Server │
│  • Game actors (Tokio)   │  ◀──────  EVs [B×4 floats]          │  (Torch CUDA or MLX)     │
│  • Strategy scheduler    │                                      │  • Prealloc I/O, 1 loop  │
│  • Micro‑batcher (feeder)│                                      │                         │
└──────────────────────────┘                                      └─────────────────────────┘
```

Key invariants:

- The **feeder never blocks** on a submitted batch. Sampling/trees run on a different lane.
- Maintain a **sliding window** of ≥2 in‑flight batches to overlap copy/compute.
- Batch formation and backpressure happen **in Rust**, independent of the backend.

---

## 1) Libraries & Build Choices

**Rust (orchestrator)**

- gRPC: [`tonic`] with Unix Domain Sockets (UDS)
- Concurrency: [`tokio`] for async runtime; [`rayon`] for CPU‑bound sampling/tree work
- Channels: `tokio::sync::{mpsc, oneshot}` (bounded mpsc for backpressure)
- Metrics/logs: `tracing`, `tracing-subscriber`
- Config: `serde` + `toml`

**Python (inference server)**

- gRPC: `grpcio` (use `grpc.aio`), address `unix:/path/to.sock`
- Backend A (CUDA): PyTorch (`torch`); optional `torch.compile`
- Backend B (Metal): MLX (`mlx`), mirror same API
- Perf toggles: `torch.inference_mode()`, autocast fp16/bf16, pinned host buffers

> Keep it boring. One RPC. No Python thread pools. No ONNX/Triton.

---

## 2) RPC Surface (single method)

**Service**: `Inference.ForwardBatch(ForwardRequest) -> ForwardResponse`

**ForwardRequest**

- `batch_id: u64`
- `request_ids: Vec<u64>` — stable per item for routing
- `boards: bytes` — flat `[B,16]` `u8` exponents (0 = empty)
- `dtype_hint: enum { FP16, BF16, FP32 }`
- `padding_to: u32` (0 = none; otherwise padded to fixed MAX\_BATCH for CUDA Graphs)

**ForwardResponse**

- `batch_id: u64`
- `request_ids: Vec<u64>` (same order)
- `evs: bytes` — flat `[B,4]` floats (fp16/bf16/fp32 per hint)
- `aux_json: string` (optional, small; for debug/telemetry)

Transport: **UDS** (`/tmp/2048_infer.sock`) in development; TCP only if you later remote it.

---

## 3) Rust Orchestrator Layout

### 3.1 Tasks/Threads at a Glance

- **Tokio runtime (current\_thread)** hosts:
  - **Feeder** (micro‑batcher) — *one task only*. Builds/submits batches, manages credits.
  - **Response router** — completes oneshots; returns global credits.
  - **gRPC client** — called by feeder via `tonic` async stub.
- **Rayon thread‑pool** hosts CPU‑bound continuations:
  - Game logic, MCTS/expectimax expansion, reward calcs, etc.
- **Per‑game actors** are lightweight Tokio tasks that *delegate heavy work to Rayon*.

> Separation rule: **Feeder never executes sampling**, and **sampling never runs on the feeder’s executor thread**.

### 3.2 Channels & Queues

- `GLOBAL_REQ_MPSC<B>` (bounded): producers = games/strategies; consumer = feeder.
- `RESP_ONESHOTS`: per‑request oneshot senders stored in a slab; router completes them.
- `COMPLETION_MPSC`: feeder waits here when in‑flight window is full.

Bounded queues are your backpressure. If `GLOBAL_REQ_MPSC` is full, noisy games briefly block or drop per policy.

### 3.3 Backpressure and Credits

Parameters (tune later):

- `TARGET_BATCH = 512 (CUDA) | 256 (MLX)`
- `MAX_BATCH = 1024`
- `FLUSH_US = 200..300` (µs)
- `W = 2..3` (batches in flight)
- `C_global_items = W * MAX_BATCH`
- `C_game_items = 8..32` (per‑game in‑flight cap)
- `QUEUE_CAP = 65_536`

### 3.4 Feeder Loop (pseudocode)

```
loop {
  // 1) Build a batch quickly
  buf = drain_round_robin(GLOBAL_REQ_MPSC, TARGET_BATCH, MAX_BATCH, C_game_items)
  if buf.is_empty() {
    sleep(FLUSH_US) ; continue
  }

  // 2) Respect global in‑flight item credits
  while inflight_items + buf.len() > C_global_items {
    wait_for_any_completion(COMPLETION_MPSC)
  }

  // 3) Submit without awaiting
  submit_async(buf) ; inflight_items += buf.len()
  // Feeder immediately returns to draining; it does nothing CPU‑heavy
}
```

``

- Packs `[B,16] u8` into a **preallocated pinned host slab** (double buffered)
- Calls gRPC; attaches a completion callback that:
  - Routes results via oneshots
  - Decrements `inflight_items`
  - Sends a tiny message on `COMPLETION_MPSC`

### 3.5 Response Routing

- `ForwardResponse` includes `request_ids` aligned with outputs.
- Router looks up the oneshot sender for each `request_id` and completes it.
- The awaiting game/continuation resumes (on **Rayon**, not on feeder thread).

### 3.6 Per‑Game Actor Pattern

State per game: `{tokens: usize (<= C_game_items), rr_cursor, local_queue}`

- To request inference: if `tokens>0`, decrement and push an `InferenceItem` to `GLOBAL_REQ_MPSC`. Else, park briefly or schedule later.
- On EV arrival: return a token; schedule next expansion/step via Rayon.
- Optional: Deduplicate within a search iteration using a 64‑bit hash of the 16 exponents.

---

## 4) Python Inference Server (Torch or MLX)

### 4.1 Process Setup (common)

- Load model once; move to device; `model.eval()`.
- Preallocate **device I/O tensors** sized for `MAX_BATCH`.
- Preallocate **pinned host** staging buffers for input/output (CUDA only).
- Keep a single async gRPC server with **one handler** that does:
  1. H2D copy (or MLX array assign)
  2. Forward pass under inference mode + autocast
  3. D2H copy (as needed)

### 4.2 Torch/CUDA Handler (pseudocode)

```
@inference_mode
async def ForwardBatch(req):
    B = len(req.request_ids)

    # (Optional) Pad to MAX_BATCH when using CUDA Graphs
    N = req.padding_to or B

    # Copy into pinned host input (reused) then async H2D
    host_in[:B,:].copy_(from_bytes(req.boards))
    d_in[:N,:].copy_(host_in[:N,:], non_blocking=True)

    with autocast("cuda", dtype=hint_to_torch(req.dtype_hint)):
        if use_cuda_graph:
            d_out = graph.replay(d_in)   # fixed shape [MAX_BATCH,16]
        else:
            d_out[:N,:] = model(d_in[:N,:])

    host_out[:B,:].copy_(d_out[:B,:], non_blocking=True)
    await torch.cuda.current_stream().synchronize()  # simple correctness first

    return pack_response(req.batch_id, req.request_ids, host_out[:B,:])
```

Notes:

- Start **without** CUDA Graphs; add them later behind a flag.
- Use **two CUDA streams** only if you have appetite (H2D for N+1 overlapping compute for N). Not required for day‑1.

### 4.3 MLX Handler

- Same shape logic; no CUDA specifics. Keep preallocated arrays on device; assign slices; run forward; copy out.

### 4.4 Dtypes

- Prefer `fp16` or `bf16` for speed; emit fp16 outputs and let Rust upcast if needed.

---

## 5) Batching & Fairness Details

- **Round‑robin drain**: while building a batch, iterate per‑game subqueues to avoid one game dominating.
- **Per‑game tokens**: simple, predictable fairness. A game cannot hold >`C_game_items` items in flight.
- **Global credits**: enforce `C_global_items`; keeps memory bounded and maintains the sliding window.
- **Time‑boxed flush**: even with small queues, flush every `FLUSH_US` if credits available; this prevents “wait for 1024 then stall.”
- **Target/Max split**: build toward `TARGET_BATCH`; allow spikes up to `MAX_BATCH` under heavy load.

---

## 6) Data Representation

- **Input**: `[B,16] u8` tile exponents (0 empty). Encode in Rust; no PyO3 in the hot path.
- **Model**: perform embedding of exponents inside the model; the server receives only raw exponents.
- **Output**: `[B,4]` EVs (fp16/bf16/fp32). Rust may upcast to fp32 for numerics.

---

## 7) Observability (minimal but sufficient)

**Rust** (print once/sec):

- `items/sec`, `batch_size_avg`, `p50/p95 RPC latency`, `inflight_items`, queue depth
- Use `tracing` with a compact fmt; CSV line is fine if you want to plot later.

**Python** (every N batches):

- H2D ms, forward ms, D2H ms, total ms, B
- Optional NVTX ranges if you profile with Nsight later.

Sanity goals:

- CUDA: GPU active time ≥ 80% under 256+ active games
- MLX: Device busy and power draw consistent; no choppy spikes

---

## 8) Configuration (shared TOML)

```toml
backend = "cuda"            # or "mlx"
uds_path = "/tmp/2048_infer.sock"
dtype = "fp16"               # fp16|bf16|fp32
use_cuda_graph = false

# batching/scheduling
flush_us = 250
target_batch = 512
max_batch = 1024
inflight_batches = 2
per_game_inflight = 16
queue_cap = 65536
```

Load this from Rust; pass the same to Python via env or a small sidecar file.

---

## 9) Implementation Plan (2 focused days)

**Day 1**

1. Define protobuf once (service + two messages).
2. Python server (CUDA path first): load weights; preallocate; implement single handler; return fixed‑shape buffer.
3. Rust skeleton: start Tokio (current\_thread), wire `tonic` client over UDS; implement **feeder** with `GLOBAL_REQ_MPSC`, `COMPLETION_MPSC`, credits; stub **router**.
4. Synthetic load: spawn 256 fake games that push random boards; tune `flush_us`, `target_batch` until the GPU charts look smooth.

**Day 2**

1. Replace fakes with real game actors; integrate strategy/MCTS via Rayon tasks.
2. Add per‑game token buckets and simple round‑robin drain.
3. Optional: padding + CUDA Graph flag; keep a clean off‑switch.
4. Clone server for MLX; confirm API parity.
5. Add tiny on‑iteration dedup (hash map) if you see repeated boards in MCTS.

---

## 10) Pseudocode Reference (all non‑binding snippets)

### Feeder drain (round‑robin)

```
fn drain_round_robin(q, target, max, per_game_cap) -> Vec<Item> {
  let mut out = Vec::with_capacity(target);
  while out.len() < target {
    if let Some(item) = q.try_recv_rr() {   // per‑game subqueues preferred; else tag items with game_id
       if inflight_per_game[item.game_id] < per_game_cap {
          inflight_per_game[item.game_id] += 1; out.push(item);
       }
    } else { break }
  }
  if out.is_empty() { out } else if out.len() < target { try_burst_fill(out, max) } else { out }
}
```

### Submit path (credits + completion)

```
fn submit_async(batch) {
  pack_into_pinned_slab(batch.inputs);
  let fut = grpc.forward(batch);
  tokio::spawn(async move {
     let resp = fut.await?;
     route(resp.request_ids, resp.evs);
     COMPLETION_MPSC.send(1);
  });
}
```

### Game actor (tokens)

```
loop {
 
```
