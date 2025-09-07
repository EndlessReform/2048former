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
    args = p.parse_args(argv)

    bind: str
    if args.uds:
        bind = args.uds
        if not bind.startswith("unix:"):
            bind = "unix:" + bind
    else:
        bind = args.tcp or "127.0.0.1:50051"

    asyncio.run(serve_async(init_dir=args.init, bind=bind, device=args.device))


if __name__ == "__main__":
    main()
