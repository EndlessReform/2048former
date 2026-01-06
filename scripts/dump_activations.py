#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import save_file as safe_save_file

from core_2048 import load_encoder_from_init
from train_2048.dataloader.steps import StepsDataset
from train_2048.tokenization.base import BoardCodec


VALID_OUTPUTS = {
    "final_hidden",
    "pooled",
    "head_logits",
    "head_probs",
    "layer_outputs",
    "attn_norm",
    "mlp_input",
    "attn_weights",
}

DTYPE_MAP = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


class ActivationCapture:
    """Capture per-layer outputs from encoder blocks."""

    def __init__(self, model, outputs: set[str]) -> None:
        self.model = model
        self.outputs = outputs
        self.layer_outputs: list[torch.Tensor] = []
        self.attn_norm: list[torch.Tensor] = []
        self.mlp_input: list[torch.Tensor] = []
        self._handles: list[torch.utils.hooks.RemovableHandle] = []

    def __enter__(self):
        for blk in self.model.blocks:
            if "layer_outputs" in self.outputs:
                h = blk.register_forward_hook(self._capture_layer)
                self._handles.append(h)
            if "attn_norm" in self.outputs:
                h = blk.attn_norm.register_forward_hook(self._capture_attn_norm)
                self._handles.append(h)
            if "mlp_input" in self.outputs:
                h = blk.mlp_norm.register_forward_hook(self._capture_mlp_input)
                self._handles.append(h)
        return self

    def __exit__(self, *_):
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def _capture_layer(self, _module, _inp, out):
        self.layer_outputs.append(out.detach())

    def _capture_attn_norm(self, _module, _inp, out):
        self.attn_norm.append(out.detach())

    def _capture_mlp_input(self, _module, _inp, out):
        self.mlp_input.append(out.detach())

    def reset(self) -> None:
        self.layer_outputs.clear()
        self.attn_norm.clear()
        self.mlp_input.clear()


def parse_outputs(raw: str | None) -> set[str]:
    if raw is None or raw.strip() == "":
        return {"final_hidden"}
    outputs = {item.strip() for item in raw.split(",") if item.strip()}
    unknown = outputs.difference(VALID_OUTPUTS)
    if unknown:
        raise ValueError(f"Unknown outputs: {sorted(unknown)}")
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dump transformer activations from a checkpoint.")
    parser.add_argument("--init", required=True, help="Checkpoint dir, .pt bundle, or hf:// path.")
    parser.add_argument("--dataset", required=True, help="Steps dataset directory.")
    parser.add_argument("--indices", help="Optional .npy int64 array of indices to sample.")
    parser.add_argument("--n-samples", type=int, default=1024, help="Number of samples to draw.")
    parser.add_argument("--batch-size", type=int, default=1024, help="Forward-pass batch size.")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for sampling.")
    parser.add_argument(
        "--outputs",
        default="final_hidden",
        help="Comma-separated list of activation keys.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="activations.safetensors",
        help="Output path (.safetensors, .pt, or .npz).",
    )
    parser.add_argument(
        "--dtype",
        choices=sorted(DTYPE_MAP.keys()),
        default="fp32",
        help="Storage dtype for activations.",
    )
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="Run inference and discard activations without writing outputs.",
    )
    parser.add_argument("--device", default=None, help="Compute device (CUDA only).")
    return parser.parse_args()


def _chunked(indices: np.ndarray, batch_size: int) -> Iterable[np.ndarray]:
    for start in range(0, int(indices.size), batch_size):
        yield indices[start : start + batch_size]


def _sample_indices(dataset_len: int, n_samples: int, seed: int) -> np.ndarray:
    if dataset_len <= 0:
        raise ValueError("Dataset length must be > 0")
    rng = np.random.default_rng(seed)
    if n_samples <= dataset_len:
        return rng.choice(dataset_len, size=n_samples, replace=False).astype(np.int64, copy=False)
    print(
        f"[dump] Warning: n_samples ({n_samples}) > dataset_len ({dataset_len}); sampling with replacement."
    )
    return rng.integers(0, dataset_len, size=n_samples, dtype=np.int64)


def _load_indices(path: str) -> np.ndarray:
    arr = np.load(path)
    arr = np.asarray(arr, dtype=np.int64).reshape(-1)
    if arr.size == 0:
        raise ValueError("Indices array is empty")
    return arr


def _decode_boards(rows: np.ndarray) -> np.ndarray:
    if "board" not in rows.dtype.names:
        raise KeyError("Expected 'board' field in steps.npy")
    mask65536 = rows["tile_65536_mask"] if "tile_65536_mask" in rows.dtype.names else None
    return BoardCodec.decode_packed_board_to_exps_u8(rows["board"], mask65536=mask65536)


def _estimate_layer_bytes(n_samples: int, num_layers: int, hidden_size: int, dtype: torch.dtype) -> int:
    bytes_per = torch.tensor([], dtype=dtype).element_size()
    return int(n_samples) * int(num_layers) * 16 * int(hidden_size) * bytes_per


def _storage_dtype_for_npz(dtype: torch.dtype) -> np.dtype:
    if dtype == torch.float32:
        return np.float32
    if dtype == torch.float16:
        return np.float16
    raise ValueError("npz output does not support bf16; use .safetensors or .pt")


def _maybe_rich_console():
    try:
        from rich.console import Console
        from rich.table import Table

        return Console(), Table
    except Exception:
        return None, None


def _print_speed(
    *,
    n_samples: int,
    batch_size: int,
    elapsed_s: float,
    device: torch.device,
) -> None:
    batches = (n_samples + batch_size - 1) // batch_size
    samples_per_s = n_samples / elapsed_s if elapsed_s > 0 else float("inf")
    tokens_per_s = (n_samples * 16) / elapsed_s if elapsed_s > 0 else float("inf")
    console, table_cls = _maybe_rich_console()
    if console and table_cls:
        table = table_cls(title="Activation Dump Speed")
        table.add_column("Metric")
        table.add_column("Value", justify="right")
        table.add_row("Device", str(device))
        table.add_row("Samples", f"{n_samples}")
        table.add_row("Batches", f"{batches}")
        table.add_row("Elapsed (s)", f"{elapsed_s:.3f}")
        table.add_row("Samples/s", f"{samples_per_s:,.1f}")
        table.add_row("Tokens/s", f"{tokens_per_s:,.1f}")
        console.print(table)
    else:
        print(
            "[dump] Speed:",
            f"samples={n_samples}",
            f"batches={batches}",
            f"elapsed={elapsed_s:.3f}s",
            f"samples/s={samples_per_s:,.1f}",
            f"tokens/s={tokens_per_s:,.1f}",
            f"device={device}",
        )


def main() -> None:
    args = parse_args()
    outputs = parse_outputs(args.outputs)
    if args.batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    if not args.indices and args.n_samples <= 0:
        raise ValueError("n_samples must be > 0")

    if "attn_weights" in outputs:
        raise NotImplementedError("attn_weights capture is not implemented yet.")

    device_str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)
    if device.type != "cuda":
        raise RuntimeError("This script only supports CUDA devices.")

    storage_dtype = DTYPE_MAP[args.dtype]
    out_path = Path(args.output)
    out_suffix = out_path.suffix.lower()
    if not args.no_write:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if out_suffix not in {".safetensors", ".pt", ".npz"}:
            raise ValueError("Output path must end with .safetensors, .pt, or .npz")

    model = load_encoder_from_init(args.init)
    model.to(device)
    model.eval()

    ds = StepsDataset(args.dataset, mmap_mode=True)
    if args.indices:
        indices = _load_indices(args.indices)
    else:
        indices = _sample_indices(len(ds), args.n_samples, args.seed)

    n_samples = int(indices.size)

    if "layer_outputs" in outputs:
        est = _estimate_layer_bytes(
            n_samples=n_samples,
            num_layers=model.config.num_hidden_layers,
            hidden_size=model.config.hidden_size,
            dtype=storage_dtype,
        )
        if est >= 8 * (1024**3):
            gib = est / (1024**3)
            print(f"[dump] Warning: layer_outputs will be ~{gib:.1f} GiB on disk.")

    collect_outputs = not args.no_write
    results: dict[str, list[torch.Tensor]] = {k: [] for k in outputs} if collect_outputs else {}
    indices_batches: list[np.ndarray] = []
    boards_batches: list[np.ndarray] = []

    def _capture_tensor(t: torch.Tensor) -> torch.Tensor:
        if t.is_floating_point():
            return t.to(device="cpu", dtype=storage_dtype)
        return t.to(device="cpu")

    if device.type == "cuda":
        torch.cuda.synchronize()
    start_time = time.perf_counter()
    with torch.inference_mode(), ActivationCapture(model, outputs) as cap:
        for batch_indices in _chunked(indices, args.batch_size):
            rows = ds.get_rows(batch_indices)
            boards = _decode_boards(rows)
            tokens = torch.from_numpy(boards).to(device=device, dtype=torch.long)

            hidden, head_out = model(tokens)

            if "final_hidden" in outputs:
                if collect_outputs:
                    results["final_hidden"].append(_capture_tensor(hidden))
            if "pooled" in outputs:
                pooled = hidden.mean(dim=1)
                if collect_outputs:
                    results["pooled"].append(_capture_tensor(pooled))
            if "head_logits" in outputs:
                if model.head_type == "binned_ev":
                    logits = torch.stack(head_out, dim=1)
                else:
                    logits = head_out
                if collect_outputs:
                    results["head_logits"].append(_capture_tensor(logits))
            if "head_probs" in outputs:
                if model.head_type == "binned_ev":
                    probs = [F.softmax(h.float(), dim=-1) for h in head_out]
                    probs_t = torch.stack(probs, dim=1)
                else:
                    probs_t = F.softmax(head_out.float(), dim=-1)
                if collect_outputs:
                    results["head_probs"].append(_capture_tensor(probs_t))
            if "layer_outputs" in outputs:
                layer_t = torch.stack(cap.layer_outputs, dim=1)
                if collect_outputs:
                    results["layer_outputs"].append(_capture_tensor(layer_t))
            if "attn_norm" in outputs:
                attn_t = torch.stack(cap.attn_norm, dim=1)
                if collect_outputs:
                    results["attn_norm"].append(_capture_tensor(attn_t))
            if "mlp_input" in outputs:
                mlp_t = torch.stack(cap.mlp_input, dim=1)
                if collect_outputs:
                    results["mlp_input"].append(_capture_tensor(mlp_t))
            if any(k in outputs for k in ("layer_outputs", "attn_norm", "mlp_input")):
                cap.reset()

            if collect_outputs:
                indices_batches.append(np.asarray(batch_indices, dtype=np.int64))
                boards_batches.append(boards.astype(np.uint8, copy=False))

    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start_time
    _print_speed(n_samples=n_samples, batch_size=args.batch_size, elapsed_s=elapsed, device=device)

    if args.no_write:
        print("[dump] Skipped writing outputs (--no-write).")
        return

    indices_all = np.concatenate(indices_batches, axis=0)
    boards_all = np.concatenate(boards_batches, axis=0)

    payload: dict[str, torch.Tensor] = {}
    for key, chunks in results.items():
        if chunks:
            payload[key] = torch.cat(chunks, dim=0)

    payload["indices"] = torch.from_numpy(indices_all)
    payload["boards"] = torch.from_numpy(boards_all)

    init_info = getattr(model, "_init_load_info", {})
    head_type = getattr(model, "head_type", "binned_ev")
    n_bins = model.config.output_n_bins if head_type == "binned_ev" else 4
    metadata = {
        "checkpoint": str(init_info.get("init_path", args.init)),
        "hidden_size": str(model.config.hidden_size),
        "num_layers": str(model.config.num_hidden_layers),
        "head_type": str(head_type),
        "n_bins": str(n_bins),
        "n_samples": str(n_samples),
        "seed": str(args.seed),
        "dtype": str(args.dtype),
        "outputs": ",".join(sorted(outputs)),
    }

    if out_suffix == ".safetensors":
        safe_save_file(payload, str(out_path), metadata=metadata)
    elif out_suffix == ".pt":
        torch.save({"metadata": metadata, **payload}, str(out_path))
    else:
        np_dtype = _storage_dtype_for_npz(storage_dtype)
        np_payload = {
            key: tensor.cpu().numpy().astype(np_dtype, copy=False)
            if tensor.is_floating_point()
            else tensor.cpu().numpy()
            for key, tensor in payload.items()
        }
        np_payload["metadata"] = np.array(json.dumps(metadata))
        np.savez(str(out_path), **np_payload)

    print(f"[dump] Wrote {out_path}")


if __name__ == "__main__":
    main()
