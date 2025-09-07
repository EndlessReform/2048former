from __future__ import annotations

import json
from pathlib import Path

from safetensors.torch import load_file as safe_load_file

from .model import Encoder, EncoderConfig


def load_encoder_from_init(init_dir: str) -> Encoder:
    """
    Construct an Encoder from an init folder.

    Expects:
    - `<init_dir>/config.json`: JSON matching EncoderConfig fields
    - Optional `<init_dir>/model.safetensors`: weights to load

    Returns a model on CPU with random weights if the safetensors file is absent.
    """
    init_path = Path(init_dir)
    cfg_path = init_path / "config.json"
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Missing init config: {cfg_path}")

    with cfg_path.open("r", encoding="utf-8") as f:
        enc_cfg_dict = json.load(f)

    enc_cfg = EncoderConfig.model_validate(enc_cfg_dict)
    model = Encoder(enc_cfg)

    weights_path = init_path / "model.safetensors"
    if weights_path.is_file():
        state = safe_load_file(str(weights_path))
        state = normalize_state_dict_keys(state)
        try:
            model.load_state_dict(state, strict=True)
        except Exception:
            # Fallback to non-strict if keys still mismatch after normalization
            model.load_state_dict(state, strict=False)

    return model


def normalize_state_dict_keys(state: dict) -> dict:
    """
    Strip known wrapper prefixes from state_dict keys (e.g., `_orig_mod.`, `module.`).

    Returns a new dict with normalized keys.
    """
    prefixes = ("_orig_mod.", "module.")
    out = {}
    for k, v in state.items():
        nk = k
        changed = True
        while changed:
            changed = False
            for p in prefixes:
                if nk.startswith(p):
                    nk = nk[len(p) :]
                    changed = True
        out[nk] = v
    return out

