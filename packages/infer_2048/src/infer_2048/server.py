from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

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


DEFAULT_VALUE_EPS = 0.001


@dataclass
class ValueHeadMeta:
    objective: Optional[str] = None
    vocab_size: Optional[int] = None
    vocab_type: Optional[str] = None
    support_min: Optional[float] = None
    support_max: Optional[float] = None
    transform_epsilon: Optional[float] = None
    apply_transform: Optional[bool] = None
    target: Optional[str] = None

    @property
    def has_head(self) -> bool:
        return bool(self.objective or self.vocab_size or self.vocab_type)


@dataclass
class ValueOutputs:
    probs: Optional[list[list[float]]] = None
    value: Optional[list[float]] = None
    value_xform: Optional[list[float]] = None


def muzero_inverse(x: torch.Tensor, epsilon: float) -> torch.Tensor:
    eps = float(epsilon)
    abs_x = torch.abs(x)
    sign = torch.sign(x)
    if eps == 0.0:
        return sign * ((abs_x + 1.0) ** 2 - 1.0)
    tmp = torch.sqrt(1.0 + 4.0 * eps * (abs_x + 1.0 + eps))
    inner = (tmp - 1.0) / (2.0 * eps)
    return sign * (inner * inner - 1.0)


def value_expectation_from_probs(
    probs: torch.Tensor, *, support_min: float, support_max: float
) -> torch.Tensor:
    vocab_size = probs.shape[-1]
    step = (support_max - support_min) / max(1, vocab_size - 1)
    if step < 0:
        raise ValueError("support_max must be greater than support_min")
    supports = torch.linspace(
        support_min, support_max, vocab_size, device=probs.device, dtype=probs.dtype
    )
    return (probs * supports).sum(dim=-1)


def should_apply_inverse(meta: ValueHeadMeta) -> bool:
    target = (meta.target or "").lower()
    if meta.apply_transform:
        return True
    if target == "return_scaled":
        return True
    return False


def extract_value_metadata(model: torch.nn.Module) -> ValueHeadMeta:
    meta = ValueHeadMeta()
    cfg = getattr(model, "config", None)
    if cfg is not None:
        vcfg = getattr(cfg, "value_head", None)
        if vcfg is not None:
            obj = getattr(vcfg, "objective", None)
            meta.objective = getattr(obj, "type", None) or meta.objective
            vs = getattr(obj, "vocab_size", None)
            meta.vocab_size = int(vs) if vs is not None else meta.vocab_size
            meta.vocab_type = getattr(obj, "vocab_type", None) or meta.vocab_type

    init_info = getattr(model, "_init_load_info", None)
    if isinstance(init_info, dict):
        bundle_meta = init_info.get("bundle_metadata") or {}
        training_cfg = bundle_meta.get("training_config")
        if isinstance(training_cfg, dict):
            vt = training_cfg.get("value_training") or {}
            if isinstance(vt, dict):
                meta.target = vt.get("target") or meta.target
                if "ce_apply_transform" in vt:
                    try:
                        meta.apply_transform = bool(vt.get("ce_apply_transform"))
                    except Exception:
                        pass
                if "ce_transform_epsilon" in vt:
                    try:
                        meta.transform_epsilon = float(vt.get("ce_transform_epsilon"))
                    except Exception:
                        pass
                if "ce_support_min" in vt:
                    try:
                        meta.support_min = float(vt.get("ce_support_min"))
                    except Exception:
                        pass
                if "ce_support_max" in vt:
                    try:
                        meta.support_max = float(vt.get("ce_support_max"))
                    except Exception:
                        pass
                if meta.vocab_size is None and "ce_vocab_size" in vt:
                    try:
                        meta.vocab_size = int(vt.get("ce_vocab_size"))
                    except Exception:
                        pass
    return meta


def compute_value_outputs(
    value_out: Optional[torch.Tensor], meta: ValueHeadMeta, include_probs: bool
) -> Optional[ValueOutputs]:
    if value_out is None:
        return None

    objective = (meta.objective or "").lower()
    if objective == "cross_entropy" or (value_out.dim() > 1 and value_out.shape[-1] > 1):
        probs = F.softmax(value_out.float(), dim=-1)
        vocab_size = int(meta.vocab_size or probs.shape[-1])
        support_min = meta.support_min if meta.support_min is not None else 0.0
        support_max = (
            meta.support_max if meta.support_max is not None else float(max(0, vocab_size - 1))
        )

        value_xform = value_expectation_from_probs(
            probs, support_min=float(support_min), support_max=float(support_max)
        )
        value_xform_list = value_xform.to("cpu", non_blocking=False).tolist()
        value_list = value_xform_list
        if should_apply_inverse(meta):
            eps = meta.transform_epsilon if meta.transform_epsilon is not None else DEFAULT_VALUE_EPS
            value_list = muzero_inverse(value_xform, float(eps)).to("cpu", non_blocking=False).tolist()
        probs_list = probs.to("cpu", non_blocking=False).tolist() if include_probs else None
        return ValueOutputs(probs=probs_list, value=value_list, value_xform=value_xform_list)

    # Regression / scalar value head
    pred = value_out.float().view(-1)
    value_xform_list = pred.to("cpu", non_blocking=False).tolist()
    value_list = value_xform_list
    if should_apply_inverse(meta) and meta.transform_epsilon is not None:
        value_list = muzero_inverse(pred, float(meta.transform_epsilon)).to("cpu", non_blocking=False).tolist()
    return ValueOutputs(probs=None, value=value_list, value_xform=value_xform_list)


def resolve_output_mode(mode: int) -> tuple[bool, bool, bool]:
    if mode == inference_pb2.InferRequest.OUTPUT_MODE_POLICY_ONLY:
        return True, False, False
    if mode == inference_pb2.InferRequest.OUTPUT_MODE_VALUE_ONLY:
        return False, True, True
    if mode == inference_pb2.InferRequest.OUTPUT_MODE_POLICY_AND_VALUE:
        return True, True, True
    # Default: policy always, value when present (no error if absent)
    return True, True, False


def build_model_metadata(
    model: torch.nn.Module, init_dir: str, *, value_meta: Optional[ValueHeadMeta] = None
) -> inference_pb2.ModelMetadata:
    metadata = inference_pb2.ModelMetadata()
    policy_meta = inference_pb2.PolicyMetadata()

    config = getattr(model, "config", None)
    if config is not None:
        head_type = getattr(config, "head_type", None)
        if head_type:
            policy_meta.head_type = str(head_type)
        bin_count = getattr(config, "output_n_bins", None)
        if bin_count is not None:
            try:
                policy_meta.bin_count = int(bin_count)
            except (TypeError, ValueError):
                pass

    spec_path = Path(init_dir) / "tokenizer.json"
    if spec_path.is_file():
        try:
            with spec_path.open("r", encoding="utf-8") as f:
                spec = json.load(f)
            tokenizer_type = spec.get("tokenizer_type") or spec.get("tokenizerType")
            if tokenizer_type:
                policy_meta.tokenizer_type = str(tokenizer_type)
            vocab_order = spec.get("vocab_order") or spec.get("vocabOrder")
            if isinstance(vocab_order, list):
                policy_meta.vocab_labels.extend([str(label) for label in vocab_order])
        except Exception:
            pass

    if policy_meta.head_type or policy_meta.bin_count or policy_meta.tokenizer_type or len(policy_meta.vocab_labels):
        metadata.policy.CopyFrom(policy_meta)

    vm_info = value_meta or extract_value_metadata(model)
    vm_pb = inference_pb2.ValueMetadata()
    if vm_info.objective:
        vm_pb.objective = str(vm_info.objective)
    if vm_info.vocab_size is not None:
        try:
            vm_pb.vocab_size = int(vm_info.vocab_size)
        except (TypeError, ValueError):
            pass
    if vm_info.vocab_type:
        vm_pb.vocab_type = str(vm_info.vocab_type)
    if vm_info.support_min is not None:
        vm_pb.support_min = float(vm_info.support_min)
    if vm_info.support_max is not None:
        vm_pb.support_max = float(vm_info.support_max)
    if vm_info.transform_epsilon is not None:
        vm_pb.transform_epsilon = float(vm_info.transform_epsilon)
    if vm_info.apply_transform is not None:
        vm_pb.apply_transform = bool(vm_info.apply_transform)
    if vm_info.target:
        vm_pb.target = str(vm_info.target)

    if (
        vm_info.has_head
        or vm_info.target
        or vm_info.support_min is not None
        or vm_info.support_max is not None
        or vm_info.transform_epsilon is not None
        or vm_info.apply_transform is not None
    ):
        metadata.value.CopyFrom(vm_pb)
    return metadata


class InferenceService(inference_pb2_grpc.InferenceServicer):
    def __init__(
        self,
        model: torch.nn.Module,
        metadata: Optional[inference_pb2.ModelMetadata] = None,
        value_meta: Optional[ValueHeadMeta] = None,
        init_dir: Optional[str] = None,
    ) -> None:
        self.model = model
        self._value_meta = value_meta or extract_value_metadata(model)
        self._init_dir = init_dir
        # Reusable pinned host buffer for tokens to enable async H2D copies.
        # Allocated lazily and grown geometrically to avoid realloc thrash.
        self._tokens_cpu = None  # type: Optional[torch.Tensor]
        # Optional one-shot dumps for debugging input/outputs parity
        self._dump_tokens_path = os.environ.get("INFER_2048_DUMP_TOKENS_PATH")
        self._dump_logits_path = os.environ.get("INFER_2048_DUMP_LOGITS_PATH")
        self._dump_done = False
        # Canonical branch/head order is UDLR everywhere. No per-model reordering.
        self._model_metadata = metadata

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
            resp = inference_pb2.InferResponse(
                batch_id=request.batch_id, item_ids=[], outputs=[], latency_ms=0
            )
            if self._model_metadata is not None:
                resp.model_metadata.CopyFrom(self._model_metadata)
            return resp

        try:
            argmax_only: bool = bool(getattr(request, "argmax_only", False))
            return_embedding: bool = bool(getattr(request, "return_embedding", False))
            if argmax_only and return_embedding:
                return_embedding = False
            output_mode_raw = getattr(
                request,
                "output_mode",
                inference_pb2.InferRequest.OUTPUT_MODE_UNSPECIFIED,
            )
            include_policy, include_value, require_value = resolve_output_mode(int(output_mode_raw))
            include_value_probs: bool = bool(getattr(request, "include_value_probs", False))
            # Preserve legacy behavior: when policy is requested, also return value probs.
            include_value_probs = include_value_probs or include_policy

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
            head_probs_t: list[torch.Tensor] = []
            policy_probs: Optional[torch.Tensor]
            policy_probs = None
            policy_logits: Optional[torch.Tensor]
            policy_logits = None
            value_batch: Optional[ValueOutputs] = None
            is_binned: bool = False
            board_repr: Optional[torch.Tensor]
            board_repr = None
            with torch.inference_mode():
                forward_out = self.model(tokens)
                if isinstance(forward_out, tuple) and len(forward_out) == 3:
                    hidden_states, head_out, value_out = forward_out
                else:
                    hidden_states, head_out = forward_out
                    value_out = None
                policy_outputs_needed = include_policy or argmax_only
                # Outputs are already in canonical UDLR order
                if policy_outputs_needed:
                    if isinstance(head_out, (list, tuple)):
                        # Binned heads: list length 4
                        is_binned = True
                        head_probs_t = [F.softmax(logits.float(), dim=-1) for logits in head_out]
                    else:
                        # Single 4-way policy head (UDLR)
                        policy_logits = head_out.float()  # (B,4)
                        is_binned = False
                        policy_probs = F.softmax(policy_logits, dim=-1)
                else:
                    is_binned = isinstance(head_out, (list, tuple))

                if include_value and value_out is None:
                    if require_value:
                        await context.abort(
                            grpc.StatusCode.INVALID_ARGUMENT,
                            "value head requested but model did not return one",
                        )
                    include_value = False
                if include_value:
                    value_batch = compute_value_outputs(value_out, self._value_meta, include_value_probs)
                    if value_batch is None and require_value:
                        await context.abort(
                            grpc.StatusCode.INVALID_ARGUMENT,
                            "value head requested but model did not return outputs",
                        )
                    if value_batch is None:
                        include_value = False

                if return_embedding:
                    board_repr = hidden_states.mean(dim=1)  # (B, H)

                # Optional: dump the first batch's raw logits once
                if (self._dump_logits_path or self._dump_tokens_path) and not self._dump_done and policy_outputs_needed:
                    try:
                        if not is_binned and self._dump_logits_path and policy_logits is not None:
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

            if argmax_only and include_policy:
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
            elif include_policy:
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

                    if return_embedding and board_repr is not None:
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

                    if return_embedding and board_repr is not None:
                        br_cpu = board_repr.to(dtype=torch.float32, device="cpu", non_blocking=False).contiguous()
                        emb_dim = int(br_cpu.shape[1])
                        embeddings_concat = br_cpu.numpy().tobytes()
                    outputs = [
                        inference_pb2.Output(
                            heads=[inference_pb2.HeadProbs(probs=head) for head in probs_list[i]],
                        )
                        for i in range(B)
                    ]

            if return_embedding and embeddings_concat is None and board_repr is not None:
                br_cpu = board_repr.to(dtype=torch.float32, device="cpu", non_blocking=False).contiguous()
                emb_dim = int(br_cpu.shape[1])
                embeddings_concat = br_cpu.numpy().tobytes()

            if include_value and value_batch is not None:
                if not outputs:
                    outputs = [inference_pb2.Output() for _ in range(B)]
                for i in range(B):
                    v_msg = inference_pb2.ValueOutput()
                    if value_batch.value is not None:
                        v_msg.value = float(value_batch.value[i])
                    if value_batch.value_xform is not None:
                        v_msg.value_xform = float(value_batch.value_xform[i])
                    if include_value_probs and value_batch.probs is not None:
                        v_msg.probs.extend(value_batch.probs[i])
                    outputs[i].value.CopyFrom(v_msg)
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
        resp_kwargs: Dict[str, object] = {
            "batch_id": request.batch_id,
            "item_ids": item_ids,
            "outputs": outputs,
            "latency_ms": latency_ms,
        }
        if argmax_only and include_policy:
            resp_kwargs["argmax_heads"] = argmax_heads
            resp_kwargs["argmax_p1"] = argmax_p1
        if return_embedding:
            resp_kwargs["embed_dim"] = int(emb_dim)
            resp_kwargs["embed_dtype"] = inference_pb2.InferResponse.FP32
            resp_kwargs["embeddings_concat"] = embeddings_concat or b""
        resp = inference_pb2.InferResponse(**resp_kwargs)
        if self._model_metadata is not None:
            resp.model_metadata.CopyFrom(self._model_metadata)
        vlog(f"[server] Infer end: B={B} latency_ms={latency_ms}")
        return resp

    async def Describe(self, request, context):  # type: ignore[override]
        try:
            resp = inference_pb2.DescribeResponse(
                has_policy_head=True, has_value_head=self._value_meta.has_head
            )
            if self._model_metadata is None and self._init_dir:
                try:
                    self._model_metadata = build_model_metadata(
                        self.model, self._init_dir, value_meta=self._value_meta
                    )
                except Exception:
                    pass
            if self._model_metadata is not None:
                resp.model_metadata.CopyFrom(self._model_metadata)
            return resp
        except Exception as e:
            print(f"[server] Describe exception: {e}", flush=True)
            await context.abort(grpc.StatusCode.INTERNAL, f"server exception: {e}")
            raise


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

    value_meta = extract_value_metadata(model)
    model_metadata = build_model_metadata(model, init_dir, value_meta=value_meta)
    svc = InferenceService(
        model, metadata=model_metadata, value_meta=value_meta, init_dir=init_dir
    )

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
