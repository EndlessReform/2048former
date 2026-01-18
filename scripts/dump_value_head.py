#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch

from core_2048 import load_encoder_from_init, prepare_model_for_inference
from train_2048.tokenization.base import BoardCodec

try:  # pragma: no cover - tqdm optional
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None  # type: ignore[assignment]

VALUE_ROW_DTYPE = np.dtype(
    [
        ("run_id", "<u4"),
        ("step_index", "<u4"),
        ("max_rank", "<u1"),
        ("target_tile", "<u4"),
        ("target_index", "<u2"),
        ("p_next", "<f4"),
    ],
    align=False,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Dump value-head probabilities for each step as values-*.npy sidecars. "
            "Uses the step's max_rank to select p(next_tile)."
        )
    )
    parser.add_argument("--init", required=True, help="Checkpoint dir, .pt bundle, or hf:// path.")
    parser.add_argument("--dataset", required=True, help="Steps dataset directory.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for values-*.npy (default: dataset dir).",
    )
    parser.add_argument("--batch-size", type=int, default=2048, help="Forward-pass batch size.")
    parser.add_argument("--device", default="cuda", help="Compute device (CUDA only).")
    parser.add_argument(
        "--compile-mode",
        default="reduce-overhead",
        help="torch.compile mode or 'none' to disable.",
    )
    parser.add_argument(
        "--tiles",
        default=None,
        help="Comma-separated tile list (fallback if training-config.json missing).",
    )
    parser.add_argument(
        "--include-underflow",
        action="store_true",
        help="Include underflow class when using --tiles (default: false).",
    )
    return parser.parse_args()


def _chunked(n_rows: int, batch_size: int) -> Iterable[tuple[int, int]]:
    for start in range(0, int(n_rows), batch_size):
        end = min(int(n_rows), start + batch_size)
        yield start, end


def _resolve_tiles(init_path: str, tiles_arg: str | None, include_underflow_arg: bool) -> tuple[list[int], bool]:
    init = Path(init_path)
    if init.is_file():
        init = init.parent
    cfg_path = init / "training-config.json"
    if cfg_path.is_file():
        with cfg_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        value_cfg = payload.get("target", {}).get("value_head", {})
        tiles = value_cfg.get("tiles")
        if tiles:
            tiles_list = [int(x) for x in tiles]
            include_underflow = bool(value_cfg.get("include_underflow", True))
            return sorted(set(tiles_list)), include_underflow
    if tiles_arg is None:
        raise FileNotFoundError("Missing training-config.json; pass --tiles explicitly.")
    tiles_list = [int(x.strip()) for x in tiles_arg.split(",") if x.strip()]
    if not tiles_list:
        raise ValueError("--tiles must be a non-empty comma-separated list")
    return sorted(set(tiles_list)), bool(include_underflow_arg)


def _list_step_shards(dataset_dir: Path) -> list[Path]:
    shards = sorted(dataset_dir.glob("steps-*.npy"))
    if shards:
        return shards
    steps = dataset_dir / "steps.npy"
    if steps.is_file():
        return [steps]
    raise FileNotFoundError(f"Missing steps.npy or steps-*.npy under {dataset_dir}")


def _decode_boards(rows: np.ndarray) -> np.ndarray:
    if "board" not in rows.dtype.names:
        raise KeyError("Expected 'board' field in steps.npy")
    mask65536 = rows["tile_65536_mask"] if "tile_65536_mask" in rows.dtype.names else None
    return BoardCodec.decode_packed_board_to_exps_u8(rows["board"], mask65536=mask65536)


def _coral_logits_to_probs(logits: torch.Tensor) -> torch.Tensor:
    probs_gt = torch.sigmoid(logits)
    ones = torch.ones((logits.shape[0], 1), device=logits.device, dtype=probs_gt.dtype)
    zeros = torch.zeros((logits.shape[0], 1), device=logits.device, dtype=probs_gt.dtype)
    probs_prev = torch.cat([ones, probs_gt], dim=1)
    probs_next = torch.cat([probs_gt, zeros], dim=1)
    return probs_prev - probs_next


def _choose_target_indices(
    max_ranks: np.ndarray,
    tiles: Sequence[int],
    include_underflow: bool,
) -> tuple[np.ndarray, np.ndarray]:
    tiles_arr = np.asarray(tiles, dtype=np.int64)
    if tiles_arr.ndim != 1 or tiles_arr.size == 0:
        raise ValueError("tiles must be a non-empty 1D array")
    ranks = np.asarray(max_ranks, dtype=np.int64)
    next_exp = np.clip(ranks + 1, 0, 60)
    desired_tiles = np.left_shift(np.int64(1), next_exp)
    counts = (desired_tiles[:, None] >= tiles_arr[None, :]).sum(axis=1)
    if include_underflow:
        class_idx = counts
    else:
        max_idx = max(0, tiles_arr.size - 1)
        class_idx = np.minimum(counts, max_idx)
    return desired_tiles.astype(np.uint32, copy=False), class_idx.astype(np.int64, copy=False)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type != "cuda":
        raise RuntimeError("This script only supports CUDA devices.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    tiles, include_underflow = _resolve_tiles(args.init, args.tiles, args.include_underflow)

    model = load_encoder_from_init(args.init)
    compile_mode = None if str(args.compile_mode).lower() == "none" else args.compile_mode
    model, _dtype = prepare_model_for_inference(
        model,
        device=device,
        prefer_bf16=True,
        compile_mode=compile_mode,
    )

    num_classes = int(getattr(getattr(model, "config", None), "value_head_num_classes", 0))
    if num_classes <= 1:
        raise RuntimeError("Model does not appear to have a value head configured.")
    expected_classes = len(tiles) + (1 if include_underflow else 0)
    if num_classes != expected_classes:
        raise RuntimeError(
            f"Value head class mismatch: model has {num_classes}, tiles imply {expected_classes}."
        )

    dataset_dir = Path(args.dataset)
    output_dir = Path(args.output_dir) if args.output_dir else dataset_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    shard_paths = _list_step_shards(dataset_dir)
    shard_iter = shard_paths
    if tqdm is not None:
        shard_iter = tqdm(shard_paths, desc="Shards", unit="shard")
    for steps_path in shard_iter:
        steps = np.load(str(steps_path), mmap_mode="r")
        n_rows = int(steps.shape[0])
        out_name = steps_path.name.replace("steps", "values", 1)
        out_path = output_dir / out_name
        out = np.lib.format.open_memmap(out_path, mode="w+", dtype=VALUE_ROW_DTYPE, shape=(n_rows,))

        batch_iter = _chunked(n_rows, args.batch_size)
        if tqdm is not None:
            total_batches = (n_rows + args.batch_size - 1) // args.batch_size
            batch_iter = tqdm(
                batch_iter,
                desc=f"{steps_path.name}",
                unit="batch",
                total=total_batches,
                leave=False,
            )
        for start, end in batch_iter:
            rows = steps[start:end]
            boards = _decode_boards(rows)
            tokens = torch.from_numpy(boards).to(device=device, dtype=torch.long)

            if "max_rank" in rows.dtype.names:
                max_ranks = rows["max_rank"].astype(np.int64, copy=False)
            else:
                max_ranks = boards.max(axis=1).astype(np.int64, copy=False)

            target_tiles, target_idx = _choose_target_indices(max_ranks, tiles, include_underflow)

            with torch.inference_mode():
                _hs, _heads, value_logits = model(tokens, return_value=True)
                if value_logits.shape[-1] != num_classes - 1:
                    raise RuntimeError(
                        f"Value head logits width mismatch: got {value_logits.shape[-1]}, "
                        f"expected {num_classes - 1}."
                    )
                probs = _coral_logits_to_probs(value_logits.float())
                sel = probs.gather(1, torch.from_numpy(target_idx).to(device=device)[:, None])
                p_next = sel.squeeze(1).to(device="cpu", dtype=torch.float32).numpy()

            out["run_id"][start:end] = rows["run_id"].astype(np.uint32, copy=False)
            out["step_index"][start:end] = rows["step_index"].astype(np.uint32, copy=False)
            out["max_rank"][start:end] = max_ranks.astype(np.uint8, copy=False)
            out["target_tile"][start:end] = target_tiles
            out["target_index"][start:end] = target_idx.astype(np.uint16, copy=False)
            out["p_next"][start:end] = p_next.astype(np.float32, copy=False)

        print(f"[value] wrote {out_path} ({n_rows} rows)")


if __name__ == "__main__":
    main()
