from __future__ import annotations

"""
Server vs. offline policy argmax agreement (bounded, streaming, no spam).

Sends random boards to the running inference server (argmax_only) and compares
the returned head index with offline model argmax on the same boards. This
isolates mismatches in the inference stack (client/server/orchestrator order).

Run (server must be running on UDS):
  uv run python benchmarks/server_client_agreement.py \
    --init checkpoints/<run_dir> \
    --dataset datasets/macroxue/d6_1b \
    --uds /tmp/2048_infer.sock \
    --num 5000 --batch 512 --device cuda
"""

import argparse
import os
import random
from pathlib import Path
from typing import Optional, List

import numpy as np
import torch
import torch.nn.functional as F
import grpc

from core_2048 import load_encoder_from_init
from train_2048.dataloader.steps import StepsDataset


def _import_stubs():
    import importlib, sys
    _pkg_v1 = importlib.import_module("infer_2048.proto.train_2048.inference.v1")
    _pb2 = importlib.import_module("infer_2048.proto.train_2048.inference.v1.inference_pb2")
    sys.modules.setdefault("train_2048", importlib.import_module("infer_2048.proto.train_2048"))
    sys.modules.setdefault("train_2048.inference", importlib.import_module("infer_2048.proto.train_2048.inference"))
    sys.modules.setdefault("train_2048.inference.v1", _pkg_v1)
    sys.modules["train_2048.inference.v1.inference_pb2"] = _pb2
    _pb2_grpc = importlib.import_module("infer_2048.proto.train_2048.inference.v1.inference_pb2_grpc")
    sys.modules["train_2048.inference.v1.inference_pb2_grpc"] = _pb2_grpc
    from train_2048.inference.v1 import inference_pb2, inference_pb2_grpc  # type: ignore
    return inference_pb2, inference_pb2_grpc


def _unpack_board_to_exps_u8(packed: np.ndarray, *, mask65536: Optional[np.ndarray] = None) -> np.ndarray:
    arr = packed.astype(np.uint64, copy=False)
    n = int(arr.shape[0])
    out = np.empty((n, 16), dtype=np.uint8)
    for i in range(16):
        out[:, i] = ((arr >> (4 * i)) & np.uint64(0xF)).astype(np.uint8, copy=False)
    if mask65536 is not None:
        m = mask65536.astype(np.uint16, copy=False)
        for i in range(16):
            sel = ((m >> i) & np.uint16(1)) != 0
            if np.any(sel):
                out[sel, i] = np.uint8(16)
    return out


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--init", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--uds", required=False)
    ap.add_argument("--compile-mode", type=str, default="reduce-overhead")
    ap.add_argument("--server-device", type=str, default=None)
    ap.add_argument("--num", type=int, default=5000)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--device", type=str, default=None)
    args = ap.parse_args(argv)

    dev = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    # Offline model
    model = load_encoder_from_init(args.init).to(dev)
    model.eval()
    head_type = getattr(getattr(model, "config", None), "head_type", "binned_ev")
    if head_type != "action_policy":
        print(f"WARNING: checkpoint head_type={head_type}, this probe targets policy head.")

    # Dataset (mmap), random indices
    ds = StepsDataset(args.dataset, mmap_mode=True)
    total = len(ds)
    n = min(int(args.num), total)
    idxs = np.random.default_rng(123).integers(0, total, size=n, dtype=np.int64)

    # gRPC stubs
    inference_pb2, inference_pb2_grpc = _import_stubs()

    # Start local server like top-score bench does
    uds_path = args.uds or "/tmp/2048_infer.sock"
    bind = f"unix:{uds_path}" if not str(uds_path).startswith("unix:") else uds_path
    device_for_server = args.server_device or ("cuda" if torch.cuda.is_available() else "cpu")
    env = os.environ.copy()
    env.setdefault("INFER_2048_LOG", "0")
    cmd = [
        "uv", "run", "infer-2048",
        "--init", args.init,
        "--device", device_for_server,
        "--compile-mode", args.compile_mode,
        "--uds", bind,
    ]
    proc = None
    try:
        import subprocess, time
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=None, env=env)
        # Wait for socket and gRPC ready
        sock_path = uds_path[5:] if str(uds_path).startswith("unix:") else uds_path
        t0 = time.time()
        while not Path(sock_path).exists():
            if time.time() - t0 > 30.0:
                print("timeout waiting for UDS", sock_path)
                return 2
            if proc.poll() is not None:
                print("server exited early")
                return 2
            time.sleep(0.1)
        ch = grpc.insecure_channel(f"unix://{sock_path}")
        fut = grpc.channel_ready_future(ch)
        fut.result(timeout=30.0)
        stub = inference_pb2_grpc.InferenceStub(ch)

        mismatches = 0
        conf = np.zeros((4,4), dtype=np.int64)
        B = max(1, int(args.batch))
        for off in range(0, n, B):
            sel = idxs[off : off + B]
            batch = ds.get_rows(sel)
            exps = _unpack_board_to_exps_u8(
                batch['board'],
                mask65536=(batch['tile_65536_mask'] if 'tile_65536_mask' in batch.dtype.names else None),
            )
            tokens = torch.from_numpy(exps).to(device=dev, dtype=torch.long)

            # Offline policy argmax (UDLR index)
            with torch.inference_mode():
                _, head_out = model(tokens)
                if isinstance(head_out, (list, tuple)):
                    head_probs = [F.softmax(h.float(), dim=-1) for h in head_out]
                    p1 = torch.stack([hp[:, -1] for hp in head_probs], dim=1)
                    offline_idx = torch.argmax(p1, dim=1).cpu().numpy().astype(np.int64)
                else:
                    probs = F.softmax(head_out.float(), dim=-1)
                    offline_idx = torch.argmax(probs, dim=1).cpu().numpy().astype(np.int64)

            # Build server request (argmax_only)
            items = [
                inference_pb2.Item(id=int(sel[i]), board=bytes(exps[i].tolist()))
                for i in range(exps.shape[0])
            ]
            req = inference_pb2.InferRequest(
                batch_id=random.randrange(2**31 - 1), items=items, argmax_only=True
            )
            resp = stub.Infer(req, timeout=5.0)

            if len(resp.argmax_heads) != len(items):
                print(
                    f"ERROR: server returned {len(resp.argmax_heads)} heads for {len(items)} items"
                )
                return 2
            server_idx = np.array([int(x) for x in resp.argmax_heads], dtype=np.int64)

            # Compare raw indices (for policy this is UDLR); for binned_ev this is undefined.
            mismatches += int((server_idx != offline_idx).sum())
            for t, p in zip(offline_idx.tolist(), server_idx.tolist()):
                conf[int(t), int(p)] += 1

        print(
            f"Checked {n} boards; server vs offline argmax mismatches: {mismatches} ({mismatches / max(1,n):.4f})"
        )
        print("Confusion (rows=offline UDLR, cols=server UDLR):")
        for i, name in enumerate(["U","D","L","R"]):
            row = ", ".join(str(int(conf[i,j])) for j in range(4))
            print(f"  {name}: {row}")
        return 0
    finally:
        if proc is not None:
            try:
                proc.terminate()
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
