#!/usr/bin/env python3
"""Convert a training .pt bundle into a safetensors checkpoint for inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import torch
from safetensors.torch import save_file as safe_save_file

from train_2048.config import normalize_state_dict_keys


def _load_bundle(path: Path) -> Dict[str, object]:
    bundle = torch.load(str(path), map_location="cpu")
    if not isinstance(bundle, dict):
        raise ValueError(f"Expected dict payload in {path}, found {type(bundle)!r}")
    if "model" not in bundle:
        raise ValueError(f"Bundle at {path} lacks a 'model' state_dict")
    return bundle


def _prepare_state_dict(raw: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    state = normalize_state_dict_keys(raw)
    return {k: v.detach().to("cpu") for k, v in state.items()}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Convert .pt training bundle to safetensors")
    parser.add_argument("pt_path", type=Path, help="Path to the .pt bundle produced during training")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Destination .safetensors path (defaults to replacing the extension)",
    )
    parser.add_argument(
        "--write-config",
        action="store_true",
        help="Also dump encoder_config JSON next to the output (config.json by default)",
    )
    parser.add_argument(
        "--config-out",
        type=Path,
        default=None,
        help="Optional override for the config JSON path (implies --write-config)",
    )
    args = parser.parse_args(argv)

    if not args.pt_path.is_file():
        raise FileNotFoundError(f"No .pt file found at {args.pt_path}")

    bundle = _load_bundle(args.pt_path)
    model_state = bundle.get("model")
    if not isinstance(model_state, dict):
        raise ValueError("Bundle 'model' entry is not a state_dict")

    out_path = args.out
    if out_path is None:
        out_path = args.pt_path.with_suffix(".safetensors")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    safe_save_file(_prepare_state_dict(model_state), str(out_path), metadata={"format": "pt"})
    print(f"[convert] Wrote {out_path}")

    config_payload = bundle.get("encoder_config")
    if args.config_out is not None:
        config_path = args.config_out
        write_config = True
    else:
        config_path = out_path.with_name("config.json")
        write_config = args.write_config

    if write_config:
        if not isinstance(config_payload, dict):
            raise ValueError("Bundle lacks encoder_config dict; cannot write config.json")
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(json.dumps(config_payload, indent=2), encoding="utf-8")
        print(f"[convert] Wrote {config_path}")


if __name__ == "__main__":
    main()
