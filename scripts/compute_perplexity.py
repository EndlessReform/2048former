#!/usr/bin/env python3
from __future__ import annotations

import argparse
import heapq
import json
import math
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F

from core_2048 import load_encoder_from_init
from train_2048.binning import BinningConfig
from train_2048.dataloader.steps import StepsDataset
from train_2048.tokenization.base import BoardCodec
from train_2048.tokenization.ev_binning import EVBinnerTokenizer
from train_2048.tokenization.macroxue import MacroxueTokenizerV2, MacroxueTokenizerV2Spec


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute perplexity for EV-binned or macroxue targets.")
    parser.add_argument("--init", required=True, help="Checkpoint dir, .pt bundle, or hf:// path.")
    parser.add_argument("--dataset", required=True, help="Steps dataset directory.")
    parser.add_argument("--indices", help="Optional .npy int64 array of indices to sample.")
    parser.add_argument(
        "--n-samples",
        type=int,
        default=None,
        help="Number of samples to draw (default: all rows).",
    )
    parser.add_argument("--batch-size", type=int, default=1024, help="Forward-pass batch size.")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for sampling.")
    parser.add_argument("--device", default=None, help="Compute device (CUDA only).")
    parser.add_argument(
        "--per-board-out",
        help="Optional .npz to save per-board perplexity sidecar.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Top-K hardest boards to show when --show-hardest is set.",
    )
    parser.add_argument(
        "--show-hardest",
        action="store_true",
        help="Show the hardest boards table (uses --top-k).",
    )
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
        f"[ppl] Warning: n_samples ({n_samples}) > dataset_len ({dataset_len}); sampling with replacement."
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


def _load_binning_config(init_path: str) -> BinningConfig:
    init = Path(init_path)
    if init.is_file():
        init = init.parent
    if init.is_dir():
        cfg_path = init / "training-config.json"
        if cfg_path.is_file():
            with cfg_path.open("r", encoding="utf-8") as f:
                payload = json.load(f)
            binning = payload.get("binning")
            if isinstance(binning, dict):
                return BinningConfig(**binning)
    return BinningConfig()


def _load_training_target_mode(init_path: str) -> str | None:
    init = Path(init_path)
    if init.is_file():
        init = init.parent
    cfg_path = init / "training-config.json"
    if cfg_path.is_file():
        with cfg_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        target = payload.get("target", {})
        mode = target.get("mode")
        if isinstance(mode, str):
            return mode
    return None


def _load_macroxue_tokenizer(init_path: str) -> tuple[MacroxueTokenizerV2, int] | None:
    init = Path(init_path)
    if init.is_file():
        init = init.parent
    tok_path = init / "tokenizer.json"
    if not tok_path.is_file():
        return None
    with tok_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if payload.get("tokenizer_type") != "macroxue_ev_advantage_v2":
        return None
    spec = MacroxueTokenizerV2Spec.from_dict(payload)
    tokenizer = MacroxueTokenizerV2(spec)
    n_classes = len(spec.vocab_order)
    return tokenizer, n_classes


def _load_valuation_type_mapping(dataset_dir: str) -> dict[int, str]:
    vt_path = Path(dataset_dir) / "valuation_types.json"
    if vt_path.is_file():
        with vt_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        if isinstance(payload, list):
            return {i: name for i, name in enumerate(payload)}
        if isinstance(payload, dict):
            keys_are_ints = all(str(k).isdigit() for k in payload.keys())
            if keys_are_ints:
                return {int(k): str(v) for k, v in payload.items()}
            return {int(v): str(k) for k, v in payload.items()}
        raise TypeError("Unsupported valuation_types.json format")
    return {0: "search", 1: "tuple10", 2: "tuple11"}


def _maybe_rich_console():
    try:
        from rich.console import Console
        from rich.table import Table

        return Console(), Table
    except Exception:
        return None, None


def _print_summary(
    *,
    title: str,
    device: torch.device,
    n_samples: int,
    n_bins: int,
    count_label: str,
    head_count_label: str,
    elapsed_s: float,
    total_nll: float,
    total_count: int,
    head_nll: np.ndarray,
    head_count: np.ndarray,
    head_display_count: np.ndarray,
    ppl_q1: float,
    ppl_q3: float,
    ppl_iqr: float,
    top_rows: list[tuple[int, float, int]],
) -> None:
    console, table_cls = _maybe_rich_console()
    ppl = math.exp(total_nll / max(1, total_count))
    samples_per_s = n_samples / elapsed_s if elapsed_s > 0 else float("inf")
    if console and table_cls:
        summary = table_cls(title=title)
        summary.add_column("Metric")
        summary.add_column("Value", justify="right")
        summary.add_row("Device", str(device))
        summary.add_row("Samples", f"{n_samples}")
        summary.add_row(count_label, f"{total_count}")
        summary.add_row("Classes", f"{n_bins}")
        summary.add_row("Mean NLL", f"{total_nll / max(1, total_count):.6f}")
        summary.add_row("Perplexity", f"{ppl:.6f}")
        summary.add_row("Perplexity IQR", f"{ppl_iqr:.6f} (Q1={ppl_q1:.6f}, Q3={ppl_q3:.6f})")
        summary.add_row("Samples/s", f"{samples_per_s:,.1f}")
        summary.add_row("Elapsed (s)", f"{elapsed_s:.3f}")
        console.print(summary)

        heads = table_cls(title="Per-Head Perplexity (UDLR)")
        heads.add_column("Head")
        heads.add_column(head_count_label, justify="right")
        heads.add_column("Mean NLL", justify="right")
        heads.add_column("Perplexity", justify="right")
        for i in range(4):
            mean_nll = float(head_nll[i] / max(1, head_count[i]))
            heads.add_row(
                ["Up", "Down", "Left", "Right"][i],
                f"{int(head_display_count[i])}",
                f"{mean_nll:.6f}",
                f"{math.exp(mean_nll):.6f}",
            )
        console.print(heads)

        if top_rows:
            hardest = table_cls(title="Hardest Boards (by perplexity)")
            hardest.add_column("Index", justify="right")
            hardest.add_column("Perplexity", justify="right")
            hardest.add_column(count_label, justify="right")
            for idx, ppl_val, legal in top_rows:
                hardest.add_row(str(idx), f"{ppl_val:.6f}", str(legal))
            console.print(hardest)
    else:
        print(
            "[ppl] device=", device,
            "samples=", n_samples,
            f"{count_label.lower().replace(' ', '_')}=", total_count,
            "classes=", n_bins,
            "mean_nll=", f"{total_nll / max(1, total_count):.6f}",
            "perplexity=", f"{ppl:.6f}",
            "ppl_iqr=", f"{ppl_iqr:.6f}",
            "elapsed_s=", f"{elapsed_s:.3f}",
            "samples/s=", f"{samples_per_s:,.1f}",
        )


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    if args.n_samples is not None and args.n_samples <= 0:
        raise ValueError("n_samples must be > 0")

    device_str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)
    if device.type != "cuda":
        raise RuntimeError("This script only supports CUDA devices.")

    model = load_encoder_from_init(args.init)
    if getattr(model, "head_type", None) != "binned_ev":
        raise RuntimeError(f"Unsupported head_type={getattr(model, 'head_type', None)!r} (expected binned_ev).")
    model.to(device)
    model.eval()

    target_mode = _load_training_target_mode(args.init)
    macroxue = _load_macroxue_tokenizer(args.init)
    if target_mode == "macroxue_tokens" or (target_mode is None and macroxue is not None):
        if macroxue is None:
            raise RuntimeError("macroxue_tokens target but tokenizer.json missing or invalid.")
        mode = "macroxue_tokens"
        macroxue_tokenizer, n_classes = macroxue
        vt_map = _load_valuation_type_mapping(args.dataset)
    else:
        mode = "binned_ev"
        binning_cfg = _load_binning_config(args.init)
        ev_tokenizer = EVBinnerTokenizer(binning_cfg).to(device)
        n_classes = int(ev_tokenizer.n_bins)

    ds = StepsDataset(args.dataset, mmap_mode=True)
    if args.indices:
        indices = _load_indices(args.indices)
    elif args.n_samples is None:
        indices = np.arange(len(ds), dtype=np.int64)
    else:
        indices = _sample_indices(len(ds), args.n_samples, args.seed)
    n_samples = int(indices.size)

    total_nll = 0.0
    total_count = 0
    head_nll = np.zeros(4, dtype=np.float64)
    head_count = np.zeros(4, dtype=np.int64)
    branch_nll_chunks: list[np.ndarray] = []

    sidecar_indices: list[np.ndarray] = []
    sidecar_loss: list[np.ndarray] = []
    sidecar_ppl: list[np.ndarray] = []
    sidecar_legal: list[np.ndarray] = []
    sidecar_board: list[np.ndarray] = []
    sidecar_mask: list[np.ndarray] = []

    if device.type == "cuda":
        torch.cuda.synchronize()
    start_time = time.perf_counter()
    top_heap: list[tuple[float, int, int]] = []
    with torch.inference_mode():
        if device.type == "cuda":
            autocast = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        else:
            class _Null:
                def __enter__(self):
                    return None
                def __exit__(self, *args):
                    return False
            autocast = _Null()

        for batch_indices in _chunked(indices, args.batch_size):
            rows = ds.get_rows(batch_indices)
            boards = _decode_boards(rows)
            tokens = torch.from_numpy(boards).to(device=device, dtype=torch.long)

            if mode == "macroxue_tokens":
                branch_evs = rows["branch_evs"].astype(np.float32, copy=False)
                valuation_types = rows["valuation_type"].astype(np.int64, copy=False)
                move_dirs = rows["move_dir"].astype(np.int64, copy=False)
                legal_mask = BoardCodec.legal_mask_from_bits_udlr(rows["ev_legal"])

                if "board_eval" in rows.dtype.names:
                    board_evals = rows["board_eval"].astype(np.int32, copy=False)
                else:
                    from train_2048.tokenization.macroxue.board_eval import evaluate_board_batch

                    board_evals = evaluate_board_batch(boards)

                targets_np = np.zeros((len(batch_indices), 4), dtype=np.int64)
                for i in range(len(batch_indices)):
                    vt_name = vt_map.get(int(valuation_types[i]))
                    if vt_name is None:
                        raise KeyError(f"Unrecognized valuation_type ID from dataset: {valuation_types[i]}")
                    targets_np[i, :] = macroxue_tokenizer.encode_row(
                        valuation_type=vt_name,
                        branch_evs=branch_evs[i],
                        move_dir=int(move_dirs[i]),
                        legal_mask=legal_mask[i],
                        board_eval=int(board_evals[i]),
                    )

                targets = torch.from_numpy(targets_np).to(device=device, dtype=torch.long)
                per_board_nll = torch.zeros(tokens.size(0), device=device, dtype=torch.float32)
                per_board_count = torch.full(
                    (tokens.size(0),),
                    4,
                    device=device,
                    dtype=torch.int64,
                )

                with autocast:
                    _hs, head_out = model(tokens)
                    for h in range(4):
                        logits_h = head_out[h].float()
                        if logits_h.shape[-1] != n_classes:
                            raise RuntimeError(
                                f"Head {h} output dim {logits_h.shape[-1]} != tokenizer classes {n_classes}"
                            )
                        tgt_h = targets[:, h]
                        nll_h = -F.log_softmax(logits_h, dim=-1).gather(-1, tgt_h.unsqueeze(-1)).squeeze(-1)
                        head_nll[h] += float(nll_h.sum().item())
                        head_count[h] += int(nll_h.numel())
                        branch_nll_chunks.append(nll_h.cpu().numpy().astype(np.float32, copy=False))
                        per_board_nll = per_board_nll + nll_h
            else:
                if "branch_evs" in rows.dtype.names:
                    evs = rows["branch_evs"].astype(np.float32, copy=False)
                elif "ev_values" in rows.dtype.names:
                    evs = rows["ev_values"].astype(np.float32, copy=False)
                else:
                    raise KeyError("'branch_evs' or 'ev_values' missing from steps.npy")

                if "ev_legal" in rows.dtype.names:
                    legal = BoardCodec.legal_mask_from_bits_udlr(rows["ev_legal"])
                else:
                    legal = np.isfinite(evs)

                branch_values = torch.from_numpy(evs.copy()).to(device=device, dtype=torch.float32)
                branch_mask = torch.from_numpy(legal.astype(np.bool_, copy=False)).to(device=device)
                targets_bins = ev_tokenizer.build_targets(
                    evs=branch_values,
                    legal_mask=branch_mask,
                )["branch_bin_targets"]

                per_board_nll = torch.zeros(tokens.size(0), device=device, dtype=torch.float32)
                per_board_count = torch.zeros(tokens.size(0), device=device, dtype=torch.int64)

                with autocast:
                    _hs, head_out = model(tokens)
                    for h in range(4):
                        logits_h = head_out[h].float()
                        if logits_h.shape[-1] != n_classes:
                            raise RuntimeError(
                                f"Head {h} output dim {logits_h.shape[-1]} != binner bins {n_classes}"
                            )
                        tgt_h = targets_bins[:, h]
                        mask_h = branch_mask[:, h]
                        nll_h = -F.log_softmax(logits_h, dim=-1).gather(-1, tgt_h.unsqueeze(-1)).squeeze(-1)
                        if mask_h.any():
                            masked = nll_h[mask_h]
                            head_nll[h] += float(masked.sum().item())
                            head_count[h] += int(mask_h.sum().item())
                            branch_nll_chunks.append(masked.cpu().numpy().astype(np.float32, copy=False))
                            per_board_nll = per_board_nll + torch.where(mask_h, nll_h, torch.zeros_like(nll_h))
                            per_board_count = per_board_count + mask_h.to(torch.int64)

            total_nll += float(per_board_nll.sum().item())
            total_count += int(per_board_count.sum().item())

            if args.per_board_out or args.show_hardest:
                per_board_count_safe = per_board_count.clamp_min(1)
                per_board_loss = (per_board_nll / per_board_count_safe).cpu().numpy()
                per_board_ppl = np.exp(per_board_loss)
                per_board_legal = per_board_count.cpu().numpy().astype(np.int64, copy=False)
                if args.per_board_out:
                    sidecar_indices.append(np.asarray(batch_indices, dtype=np.int64))
                    sidecar_loss.append(per_board_loss.astype(np.float32, copy=False))
                    sidecar_ppl.append(per_board_ppl.astype(np.float32, copy=False))
                    sidecar_legal.append(per_board_legal)
                    sidecar_board.append(rows["board"].astype(np.uint64, copy=False))
                    if "tile_65536_mask" in rows.dtype.names:
                        sidecar_mask.append(rows["tile_65536_mask"].astype(np.uint16, copy=False))

                if args.show_hardest and args.top_k > 0:
                    for idx, ppl_val, legal_cnt in zip(batch_indices, per_board_ppl, per_board_legal):
                        if legal_cnt == 0:
                            continue
                        entry = (float(ppl_val), int(idx), int(legal_cnt))
                        if len(top_heap) < args.top_k:
                            heapq.heappush(top_heap, entry)
                        else:
                            if entry[0] > top_heap[0][0]:
                                heapq.heapreplace(top_heap, entry)
            # per-branch NLLs are collected during head loops

    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start_time

    if args.show_hardest and args.top_k > 0 and top_heap:
        top_rows = [(idx, ppl_val, legal) for ppl_val, idx, legal in sorted(top_heap, reverse=True)]
    else:
        top_rows = []

    if branch_nll_chunks:
        branch_nll = np.concatenate(branch_nll_chunks, axis=0)
        ppl_vals = np.exp(branch_nll.astype(np.float64, copy=False))
        ppl_q1 = float(np.percentile(ppl_vals, 25.0))
        ppl_q3 = float(np.percentile(ppl_vals, 75.0))
        ppl_iqr = float(ppl_q3 - ppl_q1)
    else:
        ppl_q1 = float("nan")
        ppl_q3 = float("nan")
        ppl_iqr = float("nan")

    head_display_count = np.full(4, n_samples, dtype=np.int64)
    count_label = "Branches" if mode == "macroxue_tokens" else "Legal branches"
    _print_summary(
        title="Perplexity (macroxue tokens)" if mode == "macroxue_tokens" else "Perplexity (EV bins)",
        device=device,
        n_samples=n_samples,
        n_bins=int(n_classes),
        count_label=count_label,
        head_count_label="Samples",
        elapsed_s=elapsed,
        total_nll=total_nll,
        total_count=total_count,
        head_nll=head_nll,
        head_count=head_count,
        head_display_count=head_display_count,
        ppl_q1=ppl_q1,
        ppl_q3=ppl_q3,
        ppl_iqr=ppl_iqr,
        top_rows=top_rows,
    )

    if args.per_board_out:
        out_path = Path(args.per_board_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "indices": np.concatenate(sidecar_indices, axis=0),
            "loss": np.concatenate(sidecar_loss, axis=0),
            "perplexity": np.concatenate(sidecar_ppl, axis=0),
            "count": np.concatenate(sidecar_legal, axis=0),
            "board": np.concatenate(sidecar_board, axis=0),
        }
        if sidecar_mask:
            payload["tile_65536_mask"] = np.concatenate(sidecar_mask, axis=0)
        payload["metadata"] = np.array(
            json.dumps(
                {
                    "checkpoint": str(args.init),
                    "dataset": str(args.dataset),
                    "mode": mode,
                    "classes": int(n_classes),
                    "count_label": count_label,
                    "samples": n_samples,
                }
            )
        )
        np.savez(str(out_path), **payload)
        print(f"[ppl] Wrote sidecar: {out_path}")


if __name__ == "__main__":
    main()
