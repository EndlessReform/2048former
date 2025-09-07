Inference server for the 2048 EV model (Torch). Thin gRPC wrapper around a single forward pass.

Quick start

- Install deps:
  - uv sync
  - If not already present, generate protobuf stubs into src/ so that
    Python can import `train_2048.inference.v1` directly:
    uv run --project packages/infer_2048 \
      python -m grpc_tools.protoc -I proto \
      --python_out=packages/infer_2048/src \
      --grpc_python_out=packages/infer_2048/src \
      proto/train_2048/inference/v1/inference.proto

- Run server (module mode; works without installing console script):
  - UDS: uv run --project packages/infer_2048 python -m infer_2048.main --init inits/v1_50m --uds unix:/tmp/2048_infer.sock --device cuda
  - TCP: uv run --project packages/infer_2048 python -m infer_2048.main --init inits/v1_50m --tcp 127.0.0.1:50051 --device cpu

- Or install console script for this subproject, then run the CLI name:
  - uv sync --project packages/infer_2048
  - uv run --project packages/infer_2048 infer-2048 --init inits/v1_50m --uds unix:/tmp/2048_infer.sock --device cuda

Notes

- Service is defined in proto/train_2048/inference/v1/inference.proto (rpc Infer).
- Input is u8 exponents of the board; output is per-head probability distributions over bins.
- The Python server keeps the model resident on device and uses core_2048 helpers.
 - If you run a file directly (e.g., `uv run main.py`), Python won’t see the `src/` layout. Use `python -m infer_2048.main` or the console script.
