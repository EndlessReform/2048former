Nsight Systems Profiling (Server + Client)
=========================================

This guide shows how to capture CPU and GPU traces for both the Python inference server and the Rust orchestrator using Nsight Systems (`nsys`). It uses a small wrapper script to launch both processes under `nsys` and produce two .qdrep files you can open in the macOS GUI.

Prerequisites
- Nsight Systems CLI installed (`nsys` in PATH).
- Workspace installed via `uv sync` (server) and Rust toolchain (client).

Quick Start

1) Run the profiler wrapper (UDS recommended locally):

  bin/nsys_profile.sh \
    --init inits/v1_50m \
    --uds /tmp/2048_infer.sock \
    --device cuda \
    --client-config config/inference/top-score.toml \
    --release

2) The script produces two files under `profiles/<timestamp>/`:

- `server.qdrep` — Python server trace (CUDA kernels + CPU + Python events)
- `client.qdrep` — Rust client trace (CPU/OS threads; no CUDA on client)

3) Open traces in Nsight Systems (macOS GUI)

- Copy the two `.qdrep` files to your macOS machine if needed.
- Open each in Nsight Systems. The timelines share wall-clock time, so you can correlate client batch formation with server H2D/forward/D2H.

Script Arguments

- `--init <dir>`: Path to model init (config.json + model.safetensors)
- `--uds <path>`: Unix domain socket path (preferred locally). The script will prefix `unix:` as needed.
- `--tcp <host:port>`: TCP endpoint if not using UDS.
- `--device <cuda|mps|cpu>`: Server device (default: cuda).
- `--client-config <toml>`: Rust orchestrator config (default: `config/inference/top-score.toml`).
- `--release`: Run the Rust client in release mode for realistic performance.
- `--outdir <dir>`: Custom output directory for traces (defaults to `profiles/<timestamp>`).

What the script captures

- Server: `--sample=cpu --trace=cuda,osrt,nvtx,python`
  - CPU sampling and threads
  - CUDA kernels and memcpy
  - NVTX ranges (add them in server code to annotate regions)
  - Python runtime events (helps attribute CPU burn in Python)

- Client: `--sample=cpu --trace=osrt,nvtx`
  - CPU sampling and OS runtime (threads, syscalls)
  - NVTX if you add marks in Rust (optional)

Tips

- UDS vs TCP: Prefer UDS locally (`--uds /tmp/2048_infer.sock`) to avoid TCP costs.
- Batch knobs (in your TOML) to drive higher GPU util while profiling:
  - `[orchestrator.batch] target_batch=1024, inflight_batches=3, per_game_inflight=32, flush_us=150..250`
- Add NVTX ranges in server for clarity (optional quick edits):
  - Use `import torch.cuda.nvtx as nvtx` and wrap `with nvtx.range("forward"):` around the forward pass; similarly `"pack_pb"`, `"h2d"`, `"d2h"`.
  - Nsight Systems will show these regions alongside kernels.

Troubleshooting

- If the server trace is empty: ensure `nsys` found CUDA (`--trace=cuda` shows kernels only if a CUDA context is created). Also double-check that the server terminated (the script sends SIGTERM once the client finishes).
- If the client cannot connect: the script waits for the UDS file to appear; increase `--sleep` or check the UDS path.
- If you need a single combined trace: `nsys` cannot merge two processes into one file; capture them separately (as above) and line up by wall-clock in the GUI.

