from __future__ import annotations

import argparse
import asyncio
from typing import Optional

from infer_2048.server import serve_async


def main(argv: Optional[list[str]] = None) -> None:
    p = argparse.ArgumentParser(description="2048 EV gRPC inference server (Torch)")
    p.add_argument("--init", required=True, help="Path to init directory (config.json + model.safetensors)")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--uds", help="UDS address, e.g., unix:/tmp/2048_infer.sock")
    g.add_argument("--tcp", help="TCP host:port, e.g., 127.0.0.1:50051")
    p.add_argument("--device", default=None, help="Device override (cuda|mps|cpu)")
    p.add_argument(
        "--compile-mode",
        default="reduce-overhead",
        help="Torch compile mode (e.g., reduce-overhead, max-autotune, or none to disable)",
    )
    p.add_argument(
        "--force-fp32",
        action="store_true",
        help="Force FP32 (disable bf16 preference and compilation)",
    )
    p.add_argument(
        "--warmup-sizes",
        default="",
        help="Comma-separated batch sizes to warm up (e.g., 256,1024)",
    )
    p.add_argument(
        "--dynamic-batch",
        action="store_true",
        help="Mark batch dimension dynamic during warmup to avoid later recompiles",
    )
    args = p.parse_args(argv)

    bind: str
    if args.uds:
        bind = args.uds
        if not bind.startswith("unix:"):
            bind = "unix:" + bind
    else:
        bind = args.tcp or "127.0.0.1:50051"

    compile_mode: str | None
    if str(args.compile_mode).lower() in {"none", "off", "disable", "disabled"}:
        compile_mode = None
    else:
        compile_mode = str(args.compile_mode)

    warmups: list[int] = []
    if args.warmup_sizes:
        try:
            warmups = [int(x) for x in str(args.warmup_sizes).split(",") if x.strip()]
        except Exception:
            raise SystemExit("--warmup-sizes must be a comma-separated list of integers")

    try:
        asyncio.run(
            serve_async(
                init_dir=args.init,
                bind=bind,
                device=args.device,
                compile_mode=(None if args.force_fp32 else compile_mode),
                force_fp32=bool(args.force_fp32),
                warmup_sizes=warmups,
                dynamic_batch=args.dynamic_batch,
            )
        )
    except KeyboardInterrupt:
        # Graceful shutdown path without noisy tracebacks
        print("[server] Shutting down (KeyboardInterrupt)", flush=True)


if __name__ == "__main__":
    main()
