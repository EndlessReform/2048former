from __future__ import annotations

import asyncio
import time
from typing import Optional

import grpc

import torch

from core_2048 import load_encoder_from_init, prepare_model_for_inference, forward_distributions


# Expect stubs under package-local path: src/infer_2048/proto/train_2048/inference/v1
import importlib
import sys

# Resolve generated stubs from the package-local path, and alias them to the
# absolute import names that grpc_tools emits (train_2048.inference.v1.*).
# This avoids collisions with the real training package and keeps console
# scripts working reliably.
try:
    # Import local package modules
    _pkg_v1 = importlib.import_module("infer_2048.proto.train_2048.inference.v1")
    _pb2 = importlib.import_module("infer_2048.proto.train_2048.inference.v1.inference_pb2")
    # Preinstall alias chain so importing *_pb2_grpc (which imports absolute) succeeds
    sys.modules.setdefault("train_2048", importlib.import_module("infer_2048.proto.train_2048"))
    sys.modules.setdefault(
        "train_2048.inference", importlib.import_module("infer_2048.proto.train_2048.inference")
    )
    sys.modules.setdefault("train_2048.inference.v1", _pkg_v1)
    sys.modules["train_2048.inference.v1.inference_pb2"] = _pb2
    # Now import grpc stubs (this will import train_2048.inference.v1.inference_pb2 via our alias)
    _pb2_grpc = importlib.import_module(
        "infer_2048.proto.train_2048.inference.v1.inference_pb2_grpc"
    )
    sys.modules["train_2048.inference.v1.inference_pb2_grpc"] = _pb2_grpc
    # Finally, import using absolute names consistently
    from train_2048.inference.v1 import inference_pb2, inference_pb2_grpc  # type: ignore
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "Protobuf stubs not found under infer_2048/proto/train_2048/inference/v1.\n"
        "Generate with:\n"
        "  uv run --project packages/infer_2048 python -m grpc_tools.protoc -I proto --python_out=packages/infer_2048/src/infer_2048/proto --grpc_python_out=packages/infer_2048/src/infer_2048/proto proto/train_2048/inference/v1/inference.proto\n"
    ) from e


class InferenceService(inference_pb2_grpc.InferenceServicer):
    def __init__(self, model: torch.nn.Module) -> None:
        self.model = model

    async def Infer(self, request, context):  # type: ignore[override]
        t0 = time.perf_counter()
        items = request.items
        B = len(items)
        if B == 0:
            return inference_pb2.InferResponse(batch_id=request.batch_id, item_ids=[], outputs=[], latency_ms=0)

        # Build tokens tensor (B,16) from bytes
        tokens_list = []
        item_ids = []
        for it in items:
            b = it.board
            if len(b) != 16:
                await context.abort(grpc.StatusCode.INVALID_ARGUMENT, f"Item {it.id} has invalid board length {len(b)} (expected 16)")
            tokens_list.append([x for x in b])
            item_ids.append(int(it.id))
        tokens = torch.tensor(tokens_list, dtype=torch.long)

        # Forward: per-head (B, n_bins)
        head_probs = forward_distributions(self.model, tokens, set_eval=True)

        # Convert to response
        outputs = []
        for i in range(B):
            heads = []
            for h in range(4):
                probs = head_probs[h][i].detach().to("cpu", copy=False).tolist()
                heads.append(inference_pb2.HeadProbs(probs=probs))
            outputs.append(inference_pb2.Output(heads=heads))

        t1 = time.perf_counter()
        latency_ms = int((t1 - t0) * 1000.0)
        return inference_pb2.InferResponse(
            batch_id=request.batch_id,
            item_ids=item_ids,
            outputs=outputs,
            latency_ms=latency_ms,
        )


async def serve_async(*, init_dir: str, bind: str, device: Optional[str]) -> None:

    model = load_encoder_from_init(init_dir)
    model, _ = prepare_model_for_inference(model, device=device, prefer_bf16=True, compile_mode="reduce-overhead")
    svc = InferenceService(model)

    server = grpc.aio.server()
    inference_pb2_grpc.add_InferenceServicer_to_server(svc, server)
    server.add_insecure_port(bind)
    await server.start()
    print(f"Inference server listening on {bind} (device={device})")
    try:
        await server.wait_for_termination()
    except asyncio.CancelledError:
        pass
    finally:
        await server.stop(0)
