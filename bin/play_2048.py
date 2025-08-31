#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
import time
from typing import Optional

import torch

from train_2048.config import load_encoder_from_init
from train_2048.inference import ModelPolicy


def _auto_device_name() -> str:
    try:
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Play one 2048 game with the model policy")
    parser.add_argument(
        "--init",
        type=str,
        required=True,
        help="Path to init directory containing config.json (and optional model.safetensors)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device override: cuda|mps|cpu. Defaults: mps>cuda>cpu",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Game RNG seed. If omitted, a random seed is chosen",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=50,
        help="Print progress every N moves (default: 50)",
    )
    args = parser.parse_args(argv)

    from ai_2048 import Board, Rng  # import lazily

    device_str = args.device or _auto_device_name()
    seed = args.seed if args.seed is not None else random.randrange(2**31 - 1)

    print(f"Loading model from: {args.init}")
    model = load_encoder_from_init(args.init)
    policy = ModelPolicy(model, device=device_str)

    rng = Rng(int(seed))
    board = Board.empty().with_random_tile(rng=rng).with_random_tile(rng=rng)

    log_every = max(1, int(args.log_interval))
    start_t = time.perf_counter()
    last_t = start_t
    last_moves = 0
    moves = 0

    dtype_str = str(getattr(policy, "used_dtype", None))
    print(
        f"Starting game: device={device_str} dtype={dtype_str} seed={seed} initial score={board.score()} highest={board.highest_tile()}"
    )

    while not board.is_game_over():
        mv = policy.best_move(board)
        if mv is None:
            break
        board = board.make_move(mv, rng=rng)
        moves += 1

        if moves % log_every == 0:
            now = time.perf_counter()
            dt = now - last_t
            total_dt = now - start_t
            window_mps = (moves - last_moves) / dt if dt > 0 else float("inf")
            avg_mps = moves / total_dt if total_dt > 0 else float("inf")
            print(
                f"[{moves:6d}] score={board.score():6d} highest={board.highest_tile():4d} "
                f"mps(window)={window_mps:8.2f} mps(avg)={avg_mps:8.2f}",
                flush=True,
            )
            last_t = now
            last_moves = moves

    end_t = time.perf_counter()
    total_dt = end_t - start_t
    avg_mps = moves / total_dt if total_dt > 0 else float("inf")

    print(
        f"Done: device={device_str} seed={seed} moves={moves} time={total_dt:.2f}s avg_mps={avg_mps:.2f} "
        f"final score={board.score()} highest={board.highest_tile()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
