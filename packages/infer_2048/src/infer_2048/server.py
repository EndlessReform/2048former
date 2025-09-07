from __future__ import annotations

import asyncio
import time
from typing import Optional, List
import os

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


VERBOSE = os.environ.get("INFER_2048_LOG", "0") == "1"


def vlog(msg: str) -> None:
    if VERBOSE:
        print(msg, flush=True)


class InferenceService(inference_pb2_grpc.InferenceServicer):
    def __init__(self, model: torch.nn.Module) -> None:
        self.model = model

    async def Infer(self, request, context):  # type: ignore[override]
        t0 = time.perf_counter()
        items = request.items
        B = len(items)
        vlog(f"[server] Infer start: B={B}")
        if B == 0:
            return inference_pb2.InferResponse(batch_id=request.batch_id, item_ids=[], outputs=[], latency_ms=0)

        try:
            # Build tokens tensor (B,16) from bytes
            tokens_list: list[list[int]] = []
            item_ids: list[int] = []
            for it in items:
                b = it.board
                if len(b) != 16:
                    await context.abort(
                        grpc.StatusCode.INVALID_ARGUMENT,
                        f"Item {it.id} has invalid board length {len(b)} (expected 16)",
                    )
                # Copy once from bytes -> Python ints; we'll upload in one shot below
                tokens_list.append([x for x in b])
                item_ids.append(int(it.id))
            vlog("[server] tokens built")

            # Single async HtoD copy by using pinned host memory
            model_device = next(self.model.parameters()).device
            # Use a single, synchronous copy to device to avoid
            # potential stream/capture issues with compile/profilers.
            tokens = torch.as_tensor(tokens_list, dtype=torch.long, device=model_device)
            vlog("[server] tokens moved to device")

            # Forward: per-head (B, n_bins)
            head_probs = forward_distributions(self.model, tokens, set_eval=True)
            vlog("[server] forward done")

            # Convert to response with a SINGLE DtoH copy: stack heads -> (B,4,n_bins) on CPU
            probs_all = torch.stack(head_probs, dim=1)  # (B, 4, n_bins)
            probs_cpu = probs_all.to("cpu", non_blocking=False)
            probs_list: list[list[list[float]]] = probs_cpu.tolist()  # B x 4 x n_bins
            outputs = [
                inference_pb2.Output(
                    heads=[inference_pb2.HeadProbs(probs=head) for head in probs_list[i]]
                )
                for i in range(B)
            ]
            vlog("[server] response materialized")
        except Exception as e:
            print(f"[server] Infer exception: {e}", flush=True)
            await context.abort(grpc.StatusCode.INTERNAL, f"server exception: {e}")
            raise

        t1 = time.perf_counter()
        latency_ms = int((t1 - t0) * 1000.0)
        resp = inference_pb2.InferResponse(
            batch_id=request.batch_id,
            item_ids=item_ids,
            outputs=outputs,
            latency_ms=latency_ms,
        )
        vlog(f"[server] Infer end: B={B} latency_ms={latency_ms}")
        return resp


async def serve_async(
    *,
    init_dir: str,
    bind: str,
    device: Optional[str],
    compile_mode: Optional[str] = "reduce-overhead",
    warmup_sizes: Optional[List[int]] = None,
    dynamic_batch: bool = False,
) -> None:

    if VERBOSE:
        print(f"[server] loading init from: {init_dir}", flush=True)
    model = load_encoder_from_init(init_dir)
    if VERBOSE:
        print("[server] preparing model for inference...", flush=True)

    # If requested via env, disable CUDA Graphs inside Inductor to avoid profiler issues.
    if os.environ.get("INFER_2048_NO_CUDAGRAPHS", "0") == "1":
        try:
            import torch._inductor.config as inductor_config

            # Best-effort toggles; ignore if keys not present in this version.
            if hasattr(inductor_config, "triton") and hasattr(inductor_config.triton, "cudagraphs"):
                inductor_config.triton.cudagraphs = False  # type: ignore[attr-defined]
            if hasattr(inductor_config, "cudagraphs"):
                inductor_config.cudagraphs = False  # type: ignore[attr-defined]
        except Exception:
            pass
    model, _ = prepare_model_for_inference(
        model,
        device=device,
        prefer_bf16=True,
        compile_mode=compile_mode,
    )

    # Optional warmup to establish dynamic batch behavior before serving
    if warmup_sizes:
        vlog(f"[server] warmup start: sizes={warmup_sizes} dynamic_batch={dynamic_batch}")
        dev = next(model.parameters()).device
        for i, bs in enumerate(warmup_sizes):
            ex = torch.empty((int(bs), 16), dtype=torch.long, device=dev)
            if dynamic_batch and i == 0:
                try:
                    torch._dynamo.mark_dynamic(ex, 0)  # type: ignore[attr-defined]
                except Exception:
                    pass
            _ = forward_distributions(model, ex, set_eval=True)
        vlog("[server] warmup complete")

    svc = InferenceService(model)

    vlog("[server] creating grpc server...")
    server = grpc.aio.server()
    inference_pb2_grpc.add_InferenceServicer_to_server(svc, server)
    server.add_insecure_port(bind)
    vlog("[server] starting grpc server...")
    await server.start()
    print(
        f"[server] ready on {bind} (device={device}, compile_mode={compile_mode}, "
        f"warmup_sizes={warmup_sizes}, dynamic_batch={dynamic_batch})",
        flush=True,
    )
    try:
        await server.wait_for_termination()
    except asyncio.CancelledError:
        pass
    finally:
        await server.stop(0)
