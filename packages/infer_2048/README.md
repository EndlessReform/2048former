Inference server for the 2048 EV model (Torch). Thin gRPC wrapper around a single forward pass.

Quick start

- Install deps:
  - uv sync
  - If not already present, generate protobuf stubs:
    uv run python -m grpc_tools.protoc -I proto \
      --python_out=packages/infer_2048/src/infer_2048/proto \
      --grpc_python_out=packages/infer_2048/src/infer_2048/proto \
      proto/train_2048/inference/v1/inference.proto

- Run server (UDS):
  uv run infer-2048 --init inits/v1_50m --uds unix:/tmp/2048_infer.sock --device cuda

- Run server (TCP):
  uv run infer-2048 --init inits/v1_50m --tcp 127.0.0.1:50051 --device cpu

Notes

- Service is defined in proto/train_2048/inference/v1/inference.proto (rpc Infer).
- Input is u8 exponents of the board; output is per-head probability distributions over bins.
- The Python server keeps the model resident on device and uses core_2048 helpers.

