from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from safetensors.torch import load_file as safe_load_file
import torch

from .model import Encoder, EncoderConfig


def _load_pt_bundle(path: Path) -> Tuple[Optional[Dict[str, torch.Tensor]], Dict[str, Any]]:
    """Return (state_dict, encoder_config) extracted from a saved .pt bundle."""

    bundle = torch.load(str(path), map_location="cpu")
    if not isinstance(bundle, dict):
        return None, {}

    state = bundle.get("model")
    state_dict: Optional[Dict[str, torch.Tensor]] = None
    if isinstance(state, dict):
        state_dict = normalize_state_dict_keys(state)

    enc_cfg = bundle.get("encoder_config")
    if isinstance(enc_cfg, dict):
        enc_cfg_dict: Dict[str, Any] = dict(enc_cfg)
    else:
        enc_cfg_dict = {}

    # Some historical checkpoints store the encoder config nested inside the
    # training config. If present, surface it to align with newer bundles.
    if not enc_cfg_dict:
        training_cfg = bundle.get("training_config")
        if isinstance(training_cfg, dict):
            maybe_encoder = training_cfg.get("encoder") or training_cfg.get("model")
            if isinstance(maybe_encoder, dict):
                enc_cfg_dict = dict(maybe_encoder)

    return state_dict, enc_cfg_dict


def load_encoder_from_init(init_dir: str) -> Encoder:
    """
    Construct an Encoder from an init directory or checkpoint bundle.

    Accepts either:
    - A directory that contains `config.json` and optionally model weights.
    - A `.pt` bundle produced by this project (state + encoder_config).

    Returns a model on CPU with random weights if no weights are found.
    """
    init_path = Path(init_dir)
    enc_cfg_dict: Optional[Dict[str, Any]] = None

    # Load encoder config either from config.json (directory case) or from the
    # checkpoint payload (file case / fallback).
    if init_path.is_dir():
        cfg_path = init_path / "config.json"
        if cfg_path.is_file():
            with cfg_path.open("r", encoding="utf-8") as f:
                enc_cfg_dict = json.load(f)
    else:
        # If the path points directly to a bundle, attempt to read the config
        # from the file. This mirrors the directory flow and allows users to
        # pass a single .pt checkpoint as the init.
        if init_path.suffix.lower() in {".pt", ".pth"} and init_path.is_file():
            _, enc_cfg_payload = _load_pt_bundle(init_path)
            if enc_cfg_payload:
                enc_cfg_dict = enc_cfg_payload

    # Discover weights (prefer safetensors when available).
    weight_candidates = []
    pt_candidates = []
    if init_path.is_dir():
        # Prefer final or best checkpoints by default. "model-stable" is a
        # pre-decay snapshot saved early in training for some schedules; keep it
        # as a fallback so inference does not accidentally load an undertrained
        # model when a final checkpoint is available.
        weight_candidates = [
            init_path / "model-best.safetensors",
            init_path / "model.safetensors",
            init_path / "model-final.safetensors",
            init_path / "model-stable.safetensors",
        ]
        pt_candidates = [
            init_path / "model-best.pt",
            init_path / "model.pt",
            init_path / "model-stable.pt",
        ]
    else:
        if init_path.suffix.lower() in {".safetensors", ".bin"}:
            weight_candidates = [init_path]
        elif init_path.suffix.lower() in {".pt", ".pth"}:
            pt_candidates = [init_path]

    weights_path = next((p for p in weight_candidates if p.is_file()), None)
    state: Optional[Dict[str, torch.Tensor]] = None
    if weights_path is not None:
        s = safe_load_file(str(weights_path))
        state = normalize_state_dict_keys(s)
    else:
        pt_path = next((p for p in pt_candidates if p.is_file()), None)
        if pt_path is not None:
            try:
                state, enc_cfg_payload = _load_pt_bundle(pt_path)
                if enc_cfg_payload:
                    if enc_cfg_dict is None:
                        enc_cfg_dict = enc_cfg_payload
                    else:
                        enc_cfg_dict.update(enc_cfg_payload)
            except Exception:
                state = None

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

    if enc_cfg_dict is None:
        raise FileNotFoundError(
            f"Missing encoder config: expected config.json next to '{init_dir}' or embedded in the checkpoint."
        )

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
