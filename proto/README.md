Proto API for the 2048 inference orchestrator.

Layout
- Package: `train_2048.inference.v1`
- File: `train_2048/inference/v1/inference.proto`

Notes
- Inputs are compact boards encoded as 16 exponents (0=empty, 1=2, 2=4, ...),
  carried as `bytes` of length 16 per item. Each item has a caller-supplied `id`.
- `Inference.Infer` is unary and batched; outputs are aligned to `item_ids` order.
- Response contains per-head probabilities (float32) over bins. Head order is
  [Up, Down, Left, Right]. Sampling stays on the orchestrator.
- No dtype/backend/cuda-graph hints appear in the wire contract. Servers handle
  their own padding/windowing internally.
- Fields are additive; do not reuse or renumber existing tags.

Codegen (sketch)
- Rust: use `tonic-build` in a `build.rs` to compile from `proto/`.
- Python: run `grpc_tools.protoc` to generate `*_pb2.py` and `*_pb2_grpc.py` under `src/train_2048/proto/` (keep generated code out of VCS).

Buf (optional)
- Lint and breaking-change checks can be enabled with `buf` using `buf.yaml` here.
