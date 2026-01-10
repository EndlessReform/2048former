from __future__ import annotations

from typing import Any, Dict, Optional
import json
from pathlib import Path

import transformer_engine.pytorch as te
import torch
import torch.nn as nn
from safetensors import safe_open
from safetensors.torch import load_file as safe_load_file

from core_2048 import EncoderConfig
from core_2048.init_io import _load_pt_bundle, _resolve_init_path, normalize_state_dict_keys
from core_2048.model import AbsolutePositionalEmbedding


class TEEncoderBlock(nn.Module):
    def __init__(self, config: EncoderConfig):
        """
        TODO: Post-processing script to change key names
        """
        super().__init__()
        self.config = config

        self.attn = te.MultiheadAttention(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_gqa_groups=(config.num_key_value_heads or config.num_attention_heads),
            attention_dropout=config.attention_dropout_prob,
            # norm
            layernorm_epsilon=config.layer_norm_eps,
            attn_mask_type="no_mask", # Bidirectional full attention,
            # TODO: experiment with this, seems to be one-click
            # zero_centered_gamma=True,
            # Pre-norm
            input_layernorm=True,
            normalization="RMSNorm",
            qk_norm_type="RMSNorm" if config.use_qk_norm else None,
            qk_norm_eps=config.layer_norm_eps,
            softmax_type="learnable" if config.use_attention_sinks else "vanilla",
            bias=False,
        )
        self.mlp = te.LayerNormMLP(
            hidden_size=config.hidden_size,
            ffn_hidden_size=config.intermediate_size,
            eps=config.layer_norm_eps,
            activation="swiglu",
            normalization="RMSNorm",
            bias=False,
        )

    def forward(self, x: torch.Tensor):
        # Norms are fused
        x = x + self.attn(x)
        x = x + self.mlp(x)
        return x


class TEEncoder(nn.Module):
    def __init__(self, config: EncoderConfig):
        super().__init__()
        self.config = config
        self.tok_emb = nn.Embedding(config.input_vocab_size, config.hidden_size)
        self.pos_emb = AbsolutePositionalEmbedding(
            max_position_embeddings=config.max_position_embeddings,
            hidden_size=config.hidden_size
        )

        # TODO: does TE do bookkeeping for keeping last N blocks in BF16 for NVFP4 or do we have to manage it here?
        self.blocks = nn.ModuleList(
            [TEEncoderBlock(config) for _ in range(config.num_hidden_layers)]
        )
        self.final_ln = te.RMSNorm(config.hidden_size, eps=config.layer_norm_eps)

        # Out proj still has to be from torch due to precision issues
        self.head_type = config.head_type
        if self.head_type == "binned_ev":
            assert config.output_n_bins is not None, "output_n_bins must be specified for binned_ev head"
            self.ev_heads = nn.ModuleList(
                [nn.Linear(config.hidden_size, config.output_n_bins) for _ in range(4)]
            )
            self.policy_head = None
        else:
            self.policy_head = nn.Linear(config.hidden_size, 4)
            self.ev_heads = None

    def forward(self, input_ids: torch.Tensor):
        _b, S = input_ids.shape
        device = input_ids.device

        # Clone to avoid CUDA graph issues
        x = self.tok_emb(input_ids).clone()
        pe = self.pos_emb(S, device)
        x = x + pe.unsqueeze(0)

        # No pre-norm: handled by blocks
        for block in self.blocks:
            x = block(x)

        x = self.final_ln(x)

        # Final mean pool
        board_repr = x.mean(dim=1) # (bsz, hidden_dim)
        if self.head_type == "binned_ev":
            assert self.ev_heads is not None, "ev_heads must be specified for binned_ev head"
            ev_preds = [head(board_repr) for head in self.ev_heads]
            return x, ev_preds
        elif self.head_type == "action_policy":
            assert self.policy_head is not None, "policy_head must be specified for policy head"
            policy_pred = self.policy_head(board_repr)
            return x, policy_pred
        else:
            raise ValueError(f"Unknown head type: {self.head_type}")


def _infer_backend_from_bundle(bundle_meta: Dict[str, Any]) -> Optional[str]:
    backend = bundle_meta.get("model_backend")
    if isinstance(backend, str):
        return backend
    autocast_type = bundle_meta.get("autocast_type")
    if isinstance(autocast_type, str):
        return "transformer_engine" if autocast_type == "mxfp8" else "torch"
    training_cfg = bundle_meta.get("training_config")
    if isinstance(training_cfg, dict):
        amp_cfg = training_cfg.get("amp")
        if isinstance(amp_cfg, dict):
            autocast_type = amp_cfg.get("autocast_type")
            if isinstance(autocast_type, str):
                return "transformer_engine" if autocast_type == "mxfp8" else "torch"
    return None


def _infer_backend_from_safetensors(path: Path) -> Optional[str]:
    try:
        with safe_open(str(path), framework="pt", device="cpu") as f:
            metadata = f.metadata() or {}
    except Exception:
        return None
    backend = metadata.get("model_backend")
    if backend:
        return backend
    autocast_type = metadata.get("autocast_type")
    if autocast_type:
        return "transformer_engine" if autocast_type == "mxfp8" else "torch"
    return None


def _assert_te_backend(backend: Optional[str], source: str) -> None:
    if backend is None:
        print(f"[te] WARNING: checkpoint metadata missing for {source}.")
        raise RuntimeError(
            "TransformerEngine training requires TE checkpoints. "
            "Re-save weights with a TE-enabled trainer to resume."
        )
    if backend != "transformer_engine":
        print(f"[te] WARNING: checkpoint backend '{backend}' is incompatible with TransformerEngine.")
        raise RuntimeError("Incompatible checkpoint backend for TransformerEngine training.")


def load_te_encoder_from_init(init_dir: str) -> TEEncoder:
    """Load a TransformerEngine-backed encoder from an init directory or bundle."""
    init_info: Dict[str, Any] = {"init_path": str(init_dir)}
    init_path = _resolve_init_path(init_dir, init_info)
    enc_cfg_dict: Optional[Dict[str, Any]] = None
    bundle_meta: Optional[Dict[str, Any]] = None

    if init_path.is_dir():
        cfg_path = init_path / "config.json"
        if cfg_path.is_file():
            with cfg_path.open("r", encoding="utf-8") as f:
                enc_cfg_dict = json.load(f)
    else:
        if init_path.suffix.lower() in {".pt", ".pth"} and init_path.is_file():
            _, enc_cfg_payload, bundle_meta = _load_pt_bundle(init_path)
            if enc_cfg_payload:
                enc_cfg_dict = enc_cfg_payload
            init_info["weights_type"] = "pt"
            init_info["weights_path"] = str(init_path)
            init_info["available_pt"] = [str(init_path)]

    weight_candidates = []
    pt_candidates = []
    if init_path.is_dir():
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
        backend = _infer_backend_from_safetensors(weights_path)
        _assert_te_backend(backend, str(weights_path))
        state = normalize_state_dict_keys(safe_load_file(str(weights_path)))
        init_info["weights_path"] = str(weights_path)
        init_info["weights_type"] = "safetensors"
    else:
        pt_path = next((p for p in pt_candidates if p.is_file()), None)
        if pt_path is not None:
            state, enc_cfg_payload, bundle_meta = _load_pt_bundle(pt_path)
            if enc_cfg_payload:
                if enc_cfg_dict is None:
                    enc_cfg_dict = enc_cfg_payload
                else:
                    enc_cfg_dict.update(enc_cfg_payload)
            backend = _infer_backend_from_bundle(bundle_meta or {})
            _assert_te_backend(backend, str(pt_path))
            init_info["weights_path"] = str(pt_path)
            init_info["weights_type"] = "pt"

    if isinstance(state, dict) and enc_cfg_dict is not None:
        head_type = enc_cfg_dict.get("head_type", "binned_ev")
        if any(k.startswith("policy_head.") for k in state.keys()):
            head_type = "action_policy"
        elif any(k.startswith("ev_heads.0.") for k in state.keys()):
            head_type = "binned_ev"
        enc_cfg_dict["head_type"] = head_type

        if head_type == "binned_ev":
            w0 = state.get("ev_heads.0.weight")
            if w0 is not None and hasattr(w0, "shape") and len(w0.shape) == 2:
                enc_cfg_dict["output_n_bins"] = int(w0.shape[0])

        tok = state.get("tok_emb.weight")
        if tok is not None and hasattr(tok, "shape") and len(tok.shape) == 2:
            vocab = int(tok.shape[0])
            enc_cfg_dict["input_vocab_size"] = max(int(enc_cfg_dict.get("input_vocab_size", 16)), vocab)

    if enc_cfg_dict is None:
        raise FileNotFoundError(
            f"Missing encoder config: expected config.json next to '{init_dir}' or embedded in the checkpoint."
        )

    enc_cfg = EncoderConfig.model_validate(enc_cfg_dict)
    model = TEEncoder(enc_cfg)

    if state is not None:
        try:
            model.load_state_dict(state, strict=True)
        except Exception as exc:
            raise RuntimeError(f"Failed to load TransformerEngine weights from '{init_dir}': {exc}") from exc

    if bundle_meta is not None:
        init_info["bundle_metadata"] = bundle_meta
    if "weights_type" not in init_info:
        init_info["weights_type"] = "random"
    init_info["model_backend"] = "transformer_engine"
    setattr(model, "_init_load_info", init_info)

    return model


__all__ = ["TEEncoder", "load_te_encoder_from_init"]
