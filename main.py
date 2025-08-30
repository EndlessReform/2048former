import argparse
from typing import Optional

import torch

from train_2048.config import load_config, load_encoder_from_init


def main(argv: Optional[list[str]] = None):
    parser = argparse.ArgumentParser(description="Train 2048 transformer scaffold")
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.example.toml",
        help="Path to a TOML config file",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device override (e.g., cuda, cpu). Defaults to CUDA if available.",
    )
    args = parser.parse_args(argv)

    cfg = load_config(args.config)

    device_str = (
        args.device
        if args.device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    device = torch.device(device_str)

    model = load_encoder_from_init(cfg.init_dir).to(device)

    # Print a brief summary
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Loaded Encoder on {device_str} with {n_params:,} params")
    print(f"Config loaded from: {args.config}")


if __name__ == "__main__":
    main()
