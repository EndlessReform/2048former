Proto API for the 2048 inference orchestrator.

Layout
- Package: `train_2048.inference.v1`
- File: `train_2048/inference/v1/inference.proto`

Notes
- Inputs are flattened token sequences matching the model encoder input. For 2048 that is
  typically 16 exponents in row-major order (0=empty, 1=2, 2=4, ...).
- `Inference.Infer` is unary and batched; the N outputs match input order.
- Response contains per-head log-probs over bins. Head order is [Up, Down, Left, Right].
- Binning: with default config, the '1' value uses the last bin (n_bins-1).
- Fields are additive; do not reuse or renumber existing tags.

Codegen (sketch)
- Rust: use `tonic-build` in a `build.rs` to compile from `proto/`.
- Python: run `grpc_tools.protoc` to generate `*_pb2.py` and `*_pb2_grpc.py` under `src/train_2048/proto/` (keep generated code out of VCS).

Buf (optional)
- Lint and breaking-change checks can be enabled with `buf` using `buf.yaml` here.
