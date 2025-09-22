from __future__ import annotations

import argparse
import os
import random
import grpc  # type: ignore

# Try to import stubs, falling back to aliasing absolute names to package-local path
try:
    from infer_2048.proto.train_2048.inference.v1 import (
        inference_pb2 as _pb2,  # type: ignore
    )
    # Mirror server's alias trick so *_pb2_grpc can import absolute names
    import importlib, sys
    sys.modules.setdefault("train_2048", importlib.import_module("infer_2048.proto.train_2048"))
    sys.modules.setdefault("train_2048.inference", importlib.import_module("infer_2048.proto.train_2048.inference"))
    sys.modules.setdefault("train_2048.inference.v1", importlib.import_module("infer_2048.proto.train_2048.inference.v1"))
    sys.modules["train_2048.inference.v1.inference_pb2"] = _pb2
    from infer_2048.proto.train_2048.inference.v1 import (
        inference_pb2,  # type: ignore
        inference_pb2_grpc,  # type: ignore
    )
except Exception:
    from train_2048.inference.v1 import inference_pb2, inference_pb2_grpc  # type: ignore


def board_to_exps_py(b) -> bytes:
    # ai_2048 Board exposes to_exponents() -> iterable of 16 u8
    vals = list(b.to_exponents())
    return bytes(vals)


def legal_mask_udlr_py(b) -> list[bool]:
    from ai_2048 import Move, Rng  # lazy import

    base_vals = list(b.to_values())
    order = [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]
    dummy = Rng(0)
    mask: list[bool] = []
    for mv in order:
        nb = b.make_move(mv, rng=dummy)
        mask.append(list(nb.to_values()) != base_vals)
    return mask


def main() -> int:
    ap = argparse.ArgumentParser(description="Probe server argmax legality rate")
    ap.add_argument("--uds", help="UDS path like /tmp/2048_infer.sock")
    ap.add_argument("--n", type=int, default=512, help="Number of random boards to probe")
    args = ap.parse_args()

    # Server channel
    target = args.uds if args.uds.startswith("unix:") else f"unix:{args.uds}"
    ch = grpc.insecure_channel(target)
    stub = inference_pb2_grpc.InferenceStub(ch)

    from ai_2048 import Board, Rng  # lazy import

    rng = Rng(123456)
    total = 0
    illegal = 0
    for gid in range(args.n):
        b = Board.empty().with_random_tile(rng=rng).with_random_tile(rng=rng)
        # random walk a bit
        for _ in range(random.randint(0, 50)):
            # Apply a random legal move to decorrelate boards
            mask = legal_mask_udlr_py(b)
            moves = [i for i, ok in enumerate(mask) if ok]
            if not moves:
                break
            mv = random.choice(moves)
            from ai_2048 import Move
            mv_map = [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]
            b = b.make_move(mv_map[mv], rng=rng)
        mask = legal_mask_udlr_py(b)
        req = inference_pb2.InferRequest(items=[inference_pb2.Item(id=gid, board=board_to_exps_py(b))], argmax_only=True)
        resp = stub.Infer(req, timeout=3.0)
        head = int(resp.argmax_heads[0]) if resp.argmax_heads else -1
        total += 1
        if head < 0 or head >= 4 or not mask[head]:
            illegal += 1
    rate = (illegal / total) if total else 0.0
    print(f"probed={total} illegal_rate={rate*100:.2f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
