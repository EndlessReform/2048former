from __future__ import annotations

import json
from pathlib import Path

from safetensors.torch import load_file as safe_load_file
import torch

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

    # If weights exist, peek shapes to adjust config before constructing model.
    # Prefer best checkpoint when present.
    weight_candidates = [
        init_path / "model-best.safetensors",
        init_path / "model-stable.safetensors",
        init_path / "model.safetensors",
        init_path / "model-final.safetensors",
    ]
    pt_candidates = [
        init_path / "model-stable.pt",
        init_path / "model.pt",
    ]
    weights_path = next((p for p in weight_candidates if p.is_file()), None)
    state = None
    if weights_path is not None and weights_path.is_file():
        s = safe_load_file(str(weights_path))
        state = normalize_state_dict_keys(s)
    else:
        # Try PT bundle with {'model': state_dict, 'optimizer': ..., 'encoder_config': ...}
        pt_path = next((p for p in pt_candidates if p.is_file()), None)
        if pt_path is not None and pt_path.is_file():
            try:
                bundle = torch.load(str(pt_path), map_location="cpu")
                maybe_state = bundle.get("model") if isinstance(bundle, dict) else None
                if isinstance(maybe_state, dict):
                    state = normalize_state_dict_keys(maybe_state)
                    # If encoder config present, prefer it
                    if isinstance(bundle.get("encoder_config"), dict):
                        enc_cfg_dict.update(bundle["encoder_config"])  # type: ignore[arg-type]
            except Exception:
                pass

    # If we loaded any state, infer additional config from it
    if isinstance(state, dict):
        # Infer head type and dimensions
        head_type = enc_cfg_dict.get("head_type", "binned_ev")
        if any(k.startswith("policy_head.") for k in state.keys()):
            head_type = "action_policy"
        elif any(k.startswith("ev_heads.0.") for k in state.keys()):
            head_type = "binned_ev"
        enc_cfg_dict["head_type"] = head_type

        # Infer output_n_bins for binned_ev from first head weight if present
        if head_type == "binned_ev":
            w0 = state.get("ev_heads.0.weight")
            if w0 is not None and hasattr(w0, "shape") and len(w0.shape) == 2:
                out_dim = int(w0.shape[0])
                enc_cfg_dict["output_n_bins"] = out_dim

        # Infer vocab size from embedding if present (ensure >= current)
        tok = state.get("tok_emb.weight")
        if tok is not None and hasattr(tok, "shape") and len(tok.shape) == 2:
            vocab = int(tok.shape[0])
            enc_cfg_dict["input_vocab_size"] = max(int(enc_cfg_dict.get("input_vocab_size", 16)), vocab)

    enc_cfg = EncoderConfig.model_validate(enc_cfg_dict)
    model = Encoder(enc_cfg)

    if state is not None:
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
