#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from safetensors.torch import safe_open


DEFAULT_PREFIX = "rmsnorm_gammas/"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Quick check for RMSNorm gamma starvation in a safetensors dump.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("activations.safetensors"),
        help="Path to activations.safetensors.",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default=DEFAULT_PREFIX,
        help="Key prefix used for RMSNorm gammas.",
    )
    return parser.parse_args()


def _format_pct(num: int, denom: int) -> str:
    if denom == 0:
        return "0.00%"
    return f"{100.0 * (num / denom):.2f}%"


def main() -> None:
    args = _parse_args()
    path = args.input
    prefix = args.prefix

    with safe_open(path, framework="pt") as f:
        keys = sorted([key for key in f.keys() if key.startswith(prefix)])
        if not keys:
            available = ", ".join(sorted(f.keys()))
            msg = f"No RMSNorm gamma keys with prefix '{prefix}'. Available keys: {available}"
            raise KeyError(msg)

        total_elements = 0
        total_ones = 0
        all_starved = True

        print(f"[ln] Found {len(keys)} RMSNorm gamma tensors in {path}")
        for key in keys:
            tensor = f.get_tensor(key)
            if not tensor.is_floating_point():
                raise ValueError(f"{key} is not floating point (dtype={tensor.dtype}).")
            numel = int(tensor.numel())
            ones = int((tensor == 1.0).sum().item())
            total_elements += numel
            total_ones += ones
            if ones != numel:
                all_starved = False

            stats_tensor = tensor.float()
            min_val = float(stats_tensor.min().item()) if numel > 0 else 0.0
            max_val = float(stats_tensor.max().item()) if numel > 0 else 0.0
            mean_val = float(stats_tensor.mean().item()) if numel > 0 else 0.0
            display = key.removeprefix(prefix)
            print(
                f"[ln] {display}: shape={tuple(tensor.shape)} dtype={tensor.dtype} "
                f"ones={ones}/{numel} ({_format_pct(ones, numel)}) "
                f"min={min_val:.6f} max={max_val:.6f} mean={mean_val:.6f}"
            )

    summary = "yes" if all_starved else "no"
    print(
        f"[ln] All RMSNorm gammas exactly 1.0: {summary} "
        f"({total_ones}/{total_elements} overall)"
    )


if __name__ == "__main__":
    main()
