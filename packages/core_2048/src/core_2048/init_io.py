from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from safetensors.torch import load_file as safe_load_file
import torch
from huggingface_hub import snapshot_download

from .model import Encoder, EncoderConfig


_HF_PREFIX = "hf://"


def _resolve_init_path(init_dir: str, init_info: Dict[str, Any]) -> Path:
    """Return a local filesystem path, downloading from HuggingFace if needed."""
    if not init_dir.startswith(_HF_PREFIX):
        path = Path(init_dir)
        init_info.setdefault("resolved_init_path", str(path))
        return path

    raw = init_dir[len(_HF_PREFIX) :].strip("/")
    revision = None
    if "@" in raw:
        raw, revision = raw.split("@", 1)

    parts = raw.split("/")
    if len(parts) < 2:
        raise ValueError("HuggingFace init path must be hf://<org>/<repo>[/subdir][@revision]")
    repo_id = "/".join(parts[:2])
    subpath = "/".join(parts[2:]) if len(parts) > 2 else ""

    cache_path = Path(
        snapshot_download(
            repo_id=repo_id,
            revision=revision or None,
        )
    )
    resolved = cache_path / subpath if subpath else cache_path

    init_info.update(
        {
            "source": "huggingface",
            "hf_repo_id": repo_id,
            "hf_revision": revision,
            "hf_subpath": subpath or None,
            "hf_cache_path": str(cache_path),
            "resolved_init_path": str(resolved),
        }
    )
    return resolved


def _load_pt_bundle(path: Path) -> Tuple[Optional[Dict[str, torch.Tensor]], Dict[str, Any], Dict[str, Any]]:
    """Return (state_dict, encoder_config, bundle_meta) extracted from a saved .pt bundle."""

    bundle = torch.load(str(path), map_location="cpu")
    if not isinstance(bundle, dict):
        return None, {}, {}

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

    bundle_meta = {k: v for k, v in bundle.items() if k not in ("model", "optimizer")}

    return state_dict, enc_cfg_dict, bundle_meta


def load_encoder_from_init(init_dir: str) -> Encoder:
    """
    Construct an Encoder from an init directory or checkpoint bundle.

    Accepts either:
    - A directory that contains `config.json` and optionally model weights.
    - A `.pt` bundle produced by this project (state + encoder_config).
    - A Hugging Face repo path prefixed with ``hf://`` (optionally with a subdir).

    Returns a model on CPU with random weights if no weights are found.
    """
    init_info: Dict[str, Any] = {"init_path": str(init_dir)}
    init_path = _resolve_init_path(init_dir, init_info)
    enc_cfg_dict: Optional[Dict[str, Any]] = None
    bundle_meta: Optional[Dict[str, Any]] = None

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
            _, enc_cfg_payload, bundle_meta = _load_pt_bundle(init_path)
            if enc_cfg_payload:
                enc_cfg_dict = enc_cfg_payload
            init_info["weights_type"] = "pt"
            init_info["weights_path"] = str(init_path)
            init_info["available_pt"] = [str(init_path)]

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
        init_info["available_pt"] = [str(p) for p in pt_candidates if p.is_file()]
    else:
        if init_path.suffix.lower() in {".safetensors", ".bin"}:
            weight_candidates = [init_path]
        elif init_path.suffix.lower() in {".pt", ".pth"}:
            pt_candidates = [init_path]
            init_info.setdefault("available_pt", []).append(str(init_path))

    weights_path = next((p for p in weight_candidates if p.is_file()), None)
    state: Optional[Dict[str, torch.Tensor]] = None
    if weights_path is not None:
        s = safe_load_file(str(weights_path))
        state = normalize_state_dict_keys(s)
        init_info["weights_path"] = str(weights_path)
        init_info["weights_type"] = "safetensors"
    else:
        pt_path = next((p for p in pt_candidates if p.is_file()), None)
        if pt_path is not None:
            try:
                state, enc_cfg_payload, bundle_meta = _load_pt_bundle(pt_path)
                if enc_cfg_payload:
                    if enc_cfg_dict is None:
                        enc_cfg_dict = enc_cfg_payload
                    else:
                        enc_cfg_dict.update(enc_cfg_payload)
                init_info["weights_path"] = str(pt_path)
                init_info["weights_type"] = "pt"
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

        # Infer value head config if present in weights
        v_w = state.get("value_head.weight")
        if v_w is not None and hasattr(v_w, "shape") and len(v_w.shape) == 2:
            vh_cfg = dict(enc_cfg_dict.get("value_head", {}) or {})
            vh_cfg.setdefault("enabled", True)
            obj_cfg = dict(vh_cfg.get("objective", {}) or {})
            # Choose objective based on output dimension if unspecified
            if "type" not in obj_cfg:
                obj_cfg["type"] = "cross_entropy" if int(v_w.shape[0]) > 1 else "mse"
            if obj_cfg.get("type") == "cross_entropy":
                obj_cfg.setdefault("vocab_size", int(v_w.shape[0]))
                obj_cfg.setdefault("vocab_type", str(obj_cfg.get("vocab_type") or "unknown"))
            vh_cfg["objective"] = obj_cfg
            if any(k.startswith("value_pre_pool_mlp.") for k in state.keys()):
                vh_cfg.setdefault("pre_pool_mlp", True)
            enc_cfg_dict["value_head"] = vh_cfg

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

    if bundle_meta is not None:
        init_info["bundle_metadata"] = bundle_meta
    if "weights_type" not in init_info:
        init_info["weights_type"] = "random"
    setattr(model, "_init_load_info", init_info)

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
