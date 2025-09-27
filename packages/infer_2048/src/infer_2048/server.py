from __future__ import annotations

import asyncio
import time
from typing import Optional, List
import os

import grpc

import torch
import numpy as np

from core_2048 import load_encoder_from_init, prepare_model_for_inference
from core_2048.infer import forward_distributions
import torch.nn.functional as F


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
        # Reusable pinned host buffer for tokens to enable async H2D copies.
        # Allocated lazily and grown geometrically to avoid realloc thrash.
        self._tokens_cpu = None  # type: Optional[torch.Tensor]
        # Optional one-shot dumps for debugging input/outputs parity
        self._dump_tokens_path = os.environ.get("INFER_2048_DUMP_TOKENS_PATH")
        self._dump_logits_path = os.environ.get("INFER_2048_DUMP_LOGITS_PATH")
        self._dump_done = False
        # Canonical branch/head order is UDLR everywhere. No per-model reordering.

    def _ensure_pinned_tokens(self, batch: int) -> torch.Tensor:
        cap = 0 if self._tokens_cpu is None else int(self._tokens_cpu.shape[0])
        if self._tokens_cpu is None or cap < batch:
            new_cap = max(1, cap)
            while new_cap < batch:
                new_cap *= 2
            self._tokens_cpu = torch.empty((new_cap, 16), dtype=torch.uint8, pin_memory=True)
        assert self._tokens_cpu is not None
        return self._tokens_cpu[:batch]

    @staticmethod
    def _tokens_from_bytes_mps(boards_bytes: bytes, batch: int, device: torch.device) -> torch.Tensor:
        # Create a tensor view directly over the request bytes to avoid redundant host copies.
        raw = torch.frombuffer(memoryview(boards_bytes), dtype=torch.uint8)
        boards = raw.as_strided((batch, 16), (16, 1))
        return boards.to(device=device, dtype=torch.long)

    async def Infer(self, request, context):  # type: ignore[override]
        t0 = time.perf_counter()
        items = request.items
        B = len(items)
        vlog(f"[server] Infer start: B={B}")
        if B == 0:
            return inference_pb2.InferResponse(batch_id=request.batch_id, item_ids=[], outputs=[], latency_ms=0)

        try:
            argmax_only: bool = bool(getattr(request, "argmax_only", False))
            return_embedding: bool = bool(getattr(request, "return_embedding", False))
            if argmax_only and return_embedding:
                return_embedding = False

            # Build item_ids and a single contiguous bytes buffer of boards
            item_ids: list[int] = []
            boards_parts: list[bytes] = []
            for it in items:
                b = it.board
                if len(b) != 16:
                    await context.abort(
                        grpc.StatusCode.INVALID_ARGUMENT,
                        f"Item {it.id} has invalid board length {len(b)} (expected 16)",
                    )
                boards_parts.append(b if isinstance(b, (bytes, bytearray)) else bytes(b))
                item_ids.append(int(it.id))
            boards_bytes = b"".join(boards_parts)
            # Zero-copy NumPy view -> (B,16) uint8
            np_view = np.frombuffer(boards_bytes, dtype=np.uint8)
            try:
                np_view = np_view.reshape(B, 16)
            except ValueError:
                await context.abort(
                    grpc.StatusCode.INVALID_ARGUMENT,
                    f"Invalid boards buffer length {np_view.size}, expected {B*16}",
                )
            try:
                model_device = next(self.model.parameters()).device
            except StopIteration:
                model_device = torch.device("cpu")
            model_device = torch.device(model_device)

            if model_device.type == "mps":
                tokens = self._tokens_from_bytes_mps(boards_bytes, B, model_device)
                vlog("[server] tokens staged (mps direct) and moved to device")
            else:
                # Ensure reusable pinned host buffer has sufficient capacity
                tokens_cpu = self._ensure_pinned_tokens(B)

                # Copy NumPy view into pinned buffer (NumPy -> pinned Tensor via ndarray view),
                # then async H2D to device with cast to long. Avoid torch.from_numpy on a
                # non-writable view to prevent warnings and undefined behavior.
                dst_np = tokens_cpu.numpy()  # writable ndarray sharing pinned memory
                dst_np[...] = np_view  # memcpy from read-only view into pinned buffer
                tokens = tokens_cpu.to(device=model_device, dtype=torch.long, non_blocking=True)
                vlog("[server] tokens staged (pinned) and moved to device")

            # Optional: dump the first batch's input tokens (exponents) exactly once
            if (self._dump_tokens_path or self._dump_logits_path) and not self._dump_done:
                try:
                    if self._dump_tokens_path:
                        # Dump uint8 exponents and item ids for alignment
                        np.save(self._dump_tokens_path, np_view.copy())
                        np.save(self._dump_tokens_path + ".ids.npy", np.asarray(item_ids, dtype=np.int64))
                except Exception as _e:
                    pass

            # Forward once to obtain hidden states and logits/policy
            prev_mode = self.model.training
            self.model.eval()
            with torch.inference_mode():
                hidden_states, head_out = self.model(tokens)
                # Outputs are already in canonical UDLR order
                if isinstance(head_out, (list, tuple)):
                    # Binned heads: list length 4
                    is_binned = True
                    head_probs_t = [F.softmax(logits.float(), dim=-1) for logits in head_out]
                else:
                    # Single 4-way policy head (UDLR)
                    policy_logits = head_out.float()  # (B,4)
                    is_binned = False
                    policy_probs = F.softmax(policy_logits, dim=-1)
                if return_embedding:
                    board_repr = hidden_states.mean(dim=1)  # (B, H)

                # Optional: dump the first batch's raw logits once
                if (self._dump_logits_path or self._dump_tokens_path) and not self._dump_done:
                    try:
                        if not is_binned and self._dump_logits_path:
                            np.save(self._dump_logits_path, policy_logits.to("cpu").numpy())
                        elif is_binned and self._dump_logits_path:
                            arr = np.stack([t.to("cpu").numpy() for t in head_out], axis=1)
                            np.save(self._dump_logits_path, arr)
                    except Exception:
                        pass
                    self._dump_done = True

            outputs: list[inference_pb2.Output] = []
            emb_dim: int = 0
            embeddings_concat: Optional[bytes] = None
            argmax_heads: list[int] = []
            argmax_p1: list[float] = []

            if argmax_only:
                if is_binned:
                    # Use p1 (last bin) by default; permit tail2 scoring via env override.
                    use_tail2 = os.environ.get("INFER_2048_TAIL2", "0") != "0"
                    if use_tail2:
                        tails = []
                        for hp in head_probs_t:
                            n = int(hp.shape[-1])
                            if n >= 2:
                                tails.append((hp[:, -1] + hp[:, -2]).float())
                            else:
                                tails.append(hp[:, -1].float())
                        score_tensor = torch.stack(tails, dim=1)
                    else:
                        score_tensor = torch.stack([hp[:, -1] for hp in head_probs_t], dim=1)
                    p_cpu = score_tensor.to(dtype=torch.float32, device="cpu", non_blocking=False)
                    best_p, best_head = torch.max(p_cpu, dim=1)
                    argmax_heads = [int(h) for h in best_head.tolist()]
                    argmax_p1 = [float(v) for v in best_p.tolist()]
                    if os.environ.get("INFER_2048_DUMP_P1", "0") != "0":
                        try:
                            n_bins = int(head_probs_t[0].shape[-1]) if head_probs_t else -1
                            first = p_cpu[0].tolist() if p_cpu.shape[0] > 0 else []
                            vlog(f"[server] DEBUG score (argmax_only) n_bins={n_bins} score[0]={first} best={argmax_heads[0] if argmax_heads else None}")
                        except Exception:
                            pass
                else:
                    # Single policy head (Up, Down, Left, Right)
                    pol_cpu = policy_probs.to(dtype=torch.float32, device="cpu", non_blocking=False)
                    best_p, best_head = torch.max(pol_cpu, dim=1)
                    argmax_heads = [int(h) for h in best_head.tolist()]
                    argmax_p1 = [float(v) for v in best_p.tolist()]
                    if os.environ.get("INFER_2048_DUMP_POLICY", "0") != "0":
                        try:
                            first = pol_cpu[0].tolist() if pol_cpu.shape[0] > 0 else []
                            vlog(f"[server] DEBUG policy (argmax_only) probs[0]={first} best_idx={argmax_heads[0] if argmax_heads else None}")
                        except Exception:
                            pass
            else:
                if not is_binned:
                    # Support non-argmax for action_policy by returning per-move probabilities
                    # encoded as 4 heads with a single-bin probability each. The Rust client
                    # treats n_bins==1 as p1-only and can apply Softmax/TopP over heads.
                    pol_cpu = policy_probs.to(dtype=torch.float32, device="cpu", non_blocking=False)
                    probs_list: list[list[float]] = pol_cpu.tolist()  # shape (B,4)
                    if os.environ.get("INFER_2048_DUMP_POLICY", "0") != "0":
                        try:
                            first = probs_list[0] if len(probs_list) > 0 else []
                            vlog(f"[server] DEBUG policy (full) probs[0]={first}")
                        except Exception:
                            pass

                    if return_embedding:
                        br_cpu = board_repr.to(dtype=torch.float32, device="cpu", non_blocking=False).contiguous()
                        emb_dim = int(br_cpu.shape[1])
                        embeddings_concat = br_cpu.numpy().tobytes()
                    outputs = [
                        inference_pb2.Output(
                            heads=[inference_pb2.HeadProbs(probs=[p]) for p in probs_list[i]]
                        )
                        for i in range(B)
                    ]
                else:
                    probs_all = torch.stack(head_probs_t, dim=1)  # (B, 4, n_bins)
                    probs_cpu = probs_all.to("cpu", non_blocking=False)
                    probs_list: list[list[list[float]]] = probs_cpu.tolist()
                    if os.environ.get("INFER_2048_DUMP_P1", "0") != "0":
                        try:
                            p1_first = [float(probs_list[0][h][-1]) for h in range(len(probs_list[0]))]
                            vlog(f"[server] DEBUG p1 (bins) n_bins={len(probs_list[0][0])} p1[0]={p1_first} best={int(np.argmax(np.array(p1_first)))}")
                        except Exception:
                            pass

                    if return_embedding:
                        br_cpu = board_repr.to(dtype=torch.float32, device="cpu", non_blocking=False).contiguous()
                        emb_dim = int(br_cpu.shape[1])
                        embeddings_concat = br_cpu.numpy().tobytes()
                        outputs = [
                            inference_pb2.Output(
                                heads=[inference_pb2.HeadProbs(probs=head) for head in probs_list[i]],
                            )
                            for i in range(B)
                        ]
                    else:
                        outputs = [
                            inference_pb2.Output(
                                heads=[inference_pb2.HeadProbs(probs=head) for head in probs_list[i]]
                            )
                            for i in range(B)
                        ]
            # Restore training state
            self.model.train(prev_mode)
            vlog("[server] response materialized")
        except Exception as e:
            print(f"[server] Infer exception: {e}", flush=True)
            await context.abort(grpc.StatusCode.INTERNAL, f"server exception: {e}")
            raise

        t1 = time.perf_counter()
        latency_ms = int((t1 - t0) * 1000.0)
        # Build response; include embedding metadata if present
        if argmax_only:
            resp = inference_pb2.InferResponse(
                batch_id=request.batch_id,
                item_ids=item_ids,
                outputs=[],
                latency_ms=latency_ms,
                argmax_heads=argmax_heads,
                argmax_p1=argmax_p1,
            )
        elif return_embedding:
            resp = inference_pb2.InferResponse(
                batch_id=request.batch_id,
                item_ids=item_ids,
                outputs=outputs,
                latency_ms=latency_ms,
                embed_dim=int(emb_dim),
                embed_dtype=inference_pb2.InferResponse.FP32,
                embeddings_concat=embeddings_concat or b"",
            )
        else:
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
    force_fp32: bool = False,
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
        prefer_bf16=(False if force_fp32 else True),
        compile_mode=(None if force_fp32 else compile_mode),
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
    # Allow large batch payloads by increasing gRPC message limits (both directions).
    # Default is ~4MB which can cap batch size when returning full distributions.
    max_mb = int(os.environ.get("INFER_2048_GRPC_MAX_MB", "64"))
    max_bytes = max(4, max_mb) * 1024 * 1024
    server = grpc.aio.server(
        options=[
            ("grpc.max_send_message_length", max_bytes),
            ("grpc.max_receive_message_length", max_bytes),
        ]
    )
    inference_pb2_grpc.add_InferenceServicer_to_server(svc, server)
    server.add_insecure_port(bind)
    vlog("[server] starting grpc server...")
    await server.start()
    print(
        f"[server] ready on {bind} (device={device}, compile_mode={compile_mode}, "
        f"warmup_sizes={warmup_sizes}, dynamic_batch={dynamic_batch}, max_msg_mb={max_mb})",
        flush=True,
    )
    try:
        await server.wait_for_termination()
    except asyncio.CancelledError:
        pass
    finally:
        await server.stop(0)
