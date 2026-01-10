from __future__ import annotations

from typing import Callable, Optional
import math

import torch
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from safetensors import safe_open

from .amp import (
    require_transformer_engine,
    resolve_autocast_type,
    use_transformer_engine,
)
from .config import DropoutConfig, TrainingConfig, load_encoder_from_init
from .objectives import Objective, make_objective


def apply_dropout_from_config(
    model: torch.nn.Module, dropout_cfg: DropoutConfig
) -> None:
    """Overwrite model dropout settings using the training config as source of truth."""
    dropout_p = float(dropout_cfg.dropout_prob)
    attn_dropout_p = float(dropout_cfg.attention_dropout_prob)

    enc_cfg = getattr(model, "config", None)
    if enc_cfg is not None:
        enc_cfg.dropout_prob = dropout_p
        enc_cfg.attention_dropout_prob = attn_dropout_p

    for block in getattr(model, "blocks", []):
        attn = getattr(block, "attn", None)
        if attn is not None:
            attn.attn_dropout_p = attn_dropout_p
            resid_dropout = getattr(attn, "resid_dropout", None)
            if resid_dropout is not None:
                resid_dropout.p = dropout_p

        mlp = getattr(block, "mlp", None)
        if mlp is not None:
            mlp_dropout = getattr(mlp, "dropout", None)
            if mlp_dropout is not None:
                mlp_dropout.p = dropout_p


def _copy_model_metadata(dst: torch.nn.Module, src: torch.nn.Module) -> None:
    for attr in ("_training_backend", "_autocast_type", "_init_load_info"):
        if hasattr(src, attr):
            setattr(dst, attr, getattr(src, attr))


def _infer_backend_from_bundle(bundle_meta: dict) -> Optional[str]:
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


def _read_safetensors_metadata(path: str) -> dict[str, str]:
    try:
        with safe_open(path, framework="pt", device="cpu") as f:
            return f.metadata() or {}
    except Exception:
        return {}


def _infer_backend_from_safetensors(metadata: dict[str, str]) -> Optional[str]:
    backend = metadata.get("model_backend")
    if backend:
        return backend
    autocast_type = metadata.get("autocast_type")
    if autocast_type:
        return "transformer_engine" if autocast_type == "mxfp8" else "torch"
    return None


def _validate_checkpoint_backend(
    model: torch.nn.Module, cfg: TrainingConfig, device: torch.device
) -> None:
    init_info = getattr(model, "_init_load_info", {}) or {}
    weights_type = init_info.get("weights_type")
    if weights_type in (None, "random"):
        return
    expected_backend = (
        "transformer_engine" if use_transformer_engine(cfg, device) else "torch"
    )
    backend: Optional[str] = None
    weights_path = init_info.get("weights_path")
    if weights_type == "pt":
        bundle_meta = init_info.get("bundle_metadata")
        if isinstance(bundle_meta, dict):
            backend = _infer_backend_from_bundle(bundle_meta)
        if expected_backend == "transformer_engine" and backend is None:
            print(
                f"[amp] WARNING: missing checkpoint backend metadata for {weights_path or 'pt bundle'}."
            )
            raise RuntimeError("TransformerEngine training requires TE checkpoints.")
    elif weights_type == "safetensors":
        if weights_path:
            backend = _infer_backend_from_safetensors(
                _read_safetensors_metadata(weights_path)
            )
        if expected_backend == "transformer_engine" and backend is None:
            print(
                f"[amp] WARNING: missing checkpoint backend metadata for {weights_path or 'safetensors'}."
            )
            raise RuntimeError("TransformerEngine training requires TE checkpoints.")
    if backend is not None and backend != expected_backend:
        print(
            f"[amp] WARNING: checkpoint backend '{backend}' is incompatible with {expected_backend} training."
        )
        raise RuntimeError("Checkpoint backend mismatch.")


def load_training_encoder(cfg: TrainingConfig, device: torch.device) -> torch.nn.Module:
    autocast_type = resolve_autocast_type(cfg)
    if autocast_type == "mxfp8":
        require_transformer_engine(cfg, device)
        from .modeling.te_encoder import load_te_encoder_from_init

        model = load_te_encoder_from_init(cfg.init_dir)
        backend = "transformer_engine"
    else:
        model = load_encoder_from_init(cfg.init_dir)
        backend = "torch"
    _validate_checkpoint_backend(model, cfg, device)
    setattr(model, "_training_backend", backend)
    setattr(model, "_autocast_type", autocast_type)
    return model


def maybe_compile_model(
    model: torch.nn.Module, cfg: TrainingConfig, device: torch.device
) -> torch.nn.Module:
    compile_enabled = bool(getattr(cfg, "compile_enabled", True))
    if use_transformer_engine(cfg, device) and compile_enabled:
        print("[amp] WARNING: torch.compile disabled for TransformerEngine.")
        return model
    if compile_enabled:
        compiled = torch.compile(model, mode="reduce-overhead")
        _copy_model_metadata(compiled, model)
        return compiled
    return model


def use_fp32_master_weights(cfg: TrainingConfig, device: torch.device) -> bool:
    """Return True when master weights should stay in fp32 on CUDA."""
    amp_cfg = getattr(cfg, "amp", None)
    if device.type != "cuda":
        return False
    if use_transformer_engine(cfg, device):
        if bool(getattr(amp_cfg, "master_weights_fp32", False)):
            print(
                "[amp] WARNING: master_weights_fp32 ignored for mxfp8; using bf16 weights."
            )
        return False
    return bool(getattr(amp_cfg, "master_weights_fp32", False))


def init_grad_scaler(
    cfg: TrainingConfig,
    device: torch.device,
    *,
    use_fp32_master_weights: bool,
) -> Optional[GradScaler]:
    """Create a GradScaler when explicitly enabled for CUDA training."""
    amp_cfg = getattr(cfg, "amp", None)
    if device.type != "cuda":
        return None
    if resolve_autocast_type(cfg) == "mxfp8":
        if bool(getattr(amp_cfg, "grad_scaler_enabled", False)):
            print("[amp] WARNING: GradScaler disabled for TransformerEngine mxfp8.")
        return None
    if not bool(getattr(amp_cfg, "grad_scaler_enabled", False)):
        return None
    if not use_fp32_master_weights:
        return None
    return GradScaler()


def move_model_to_device(
    model: torch.nn.Module,
    device: torch.device,
    *,
    use_fp32_master_weights: bool,
) -> torch.nn.Module:
    if device.type == "cuda":
        if use_fp32_master_weights:
            model = model.to(device=device, dtype=torch.float32)
            # Workaround for PyTorch bug: RMSNorm doesn't dispatch to fused kernel
            # when input dtype != weight dtype under autocast. Unlike LayerNorm,
            # RMSNorm is not in autocast's eligible op list, so it can't auto-cast.
            # This causes torch.compile to hang with fp32 master weights + bf16 autocast.
            # Cast norm layers AND embeddings to bf16 so dtypes stay consistent.
            # These small params don't benefit from fp32 master weights anyway.
            # See: https://github.com/pytorch/pytorch/issues/167308
            for module in model.modules():
                if isinstance(
                    module, (torch.nn.RMSNorm, torch.nn.LayerNorm, torch.nn.Embedding)
                ):
                    module.to(dtype=torch.bfloat16)
            return model
        return model.to(device=device, dtype=torch.bfloat16)
    return model.to(device=device)


def init_model(
    cfg: TrainingConfig,
    device: torch.device,
    *,
    target_mode: str,
    dl_train: Optional[DataLoader],
) -> tuple[torch.nn.Module, Objective]:
    use_fp32_master = use_fp32_master_weights(cfg, device)
    model = load_training_encoder(cfg, device)
    apply_dropout_from_config(model, cfg.dropout)
    objective = make_objective(
        target_mode, tokenizer_path=cfg.dataset.resolved_tokenizer_path()
    )
    model = objective.prepare_model(model, device, cfg=cfg, dl_train=dl_train)
    apply_dropout_from_config(model, cfg.dropout)
    model = move_model_to_device(model, device, use_fp32_master_weights=use_fp32_master)
    model = maybe_compile_model(model, cfg, device)
    return model, objective


def init_optimizer(
    model: torch.nn.Module, cfg: TrainingConfig
) -> torch.optim.Optimizer:
    """Create the optimizer configured for the run."""
    opt_cfg = cfg.hyperparameters.optimizer
    if opt_cfg.name == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=cfg.hyperparameters.learning_rate,
            betas=(opt_cfg.beta1, opt_cfg.beta2),
            eps=opt_cfg.eps,
            weight_decay=opt_cfg.weight_decay,
        )
    if opt_cfg.name == "muon":
        from muon import SingleDeviceMuonWithAuxAdam  # type: ignore

        def _is_norm_param(name: str, p: torch.nn.Parameter) -> bool:
            lname = name.lower()
            return (
                "layernorm" in lname or "ln" in lname or ".norm" in lname
            ) and p.ndim == 1

        def _is_embedding_or_head(name: str) -> bool:
            lname = name.lower()
            return (
                "embed" in lname
                or "embedding" in lname
                or "token_emb" in lname
                or "lm_head" in lname
                or "classifier" in lname
                or "out_proj.weight" in lname
            )

        linear_weight_ids = set()
        for module in model.modules():
            if isinstance(module, torch.nn.Linear):
                weight = getattr(module, "weight", None)
                if weight is not None:
                    linear_weight_ids.add(id(weight))

        muon_params, adam_decay, adam_no_decay = [], [], []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if (
                id(p) in linear_weight_ids
                and p.ndim >= 2
                and not _is_embedding_or_head(name)
            ):
                muon_params.append(p)
            else:
                if _is_norm_param(name, p) or p.ndim == 1 or name.endswith(".bias"):
                    adam_no_decay.append(p)
                else:
                    adam_decay.append(p)
        if len(muon_params) == 0:
            raise ValueError(
                "No Muon-eligible parameters found (expected 2-D hidden weights)."
            )
        muon_lr = cfg.hyperparameters.muon_lr or 2e-2
        adam_lr = cfg.hyperparameters.learning_rate
        param_groups = [
            dict(
                params=muon_params,
                use_muon=True,
                lr=muon_lr,
                weight_decay=opt_cfg.weight_decay,
            ),
            dict(
                params=adam_decay,
                use_muon=False,
                lr=adam_lr,
                betas=(opt_cfg.beta1, opt_cfg.beta2),
                eps=opt_cfg.eps,
                weight_decay=opt_cfg.weight_decay,
            ),
            dict(
                params=adam_no_decay,
                use_muon=False,
                lr=adam_lr,
                betas=(opt_cfg.beta1, opt_cfg.beta2),
                eps=opt_cfg.eps,
                weight_decay=0.0,
            ),
        ]
        return SingleDeviceMuonWithAuxAdam(param_groups)
    raise ValueError(f"Unknown optimizer: {opt_cfg.name}")


def make_scheduler(
    cfg: TrainingConfig,
    optimizer: torch.optim.Optimizer,
    total_steps: int,
) -> tuple[Callable[[int], float], Callable[[float], float], dict]:
    """Return schedule scaling and apply functions plus schedule metadata."""
    lr_cfg = cfg.hyperparameters.lr_schedule
    base_lrs = [
        pg.get("lr", cfg.hyperparameters.learning_rate) for pg in optimizer.param_groups
    ]

    total_steps = max(total_steps, 0)
    warmup_steps = int(lr_cfg.warmup_steps or 0)
    warmup_steps = max(
        0, min(warmup_steps, total_steps if total_steps > 0 else warmup_steps)
    )

    decay_steps = 0
    stable_steps = 0
    decay_start_step = warmup_steps

    if lr_cfg.name == "warmup-stable-decay":
        if getattr(lr_cfg, "cooldown_pct", None):
            decay_steps = (
                int(math.ceil(total_steps * float(lr_cfg.cooldown_pct)))
                if total_steps > 0
                else int(lr_cfg.decay_steps or 0)
            )
        else:
            decay_steps = int(lr_cfg.decay_steps or 0)
        decay_steps = max(0, min(decay_steps, max(total_steps - warmup_steps, 0)))
        stable_steps = max(0, total_steps - warmup_steps - decay_steps)
        decay_start_step = warmup_steps + stable_steps
    elif lr_cfg.name == "cosine":
        decay_steps = max(0, total_steps - warmup_steps)
        decay_start_step = warmup_steps  # cosine begins immediately after warmup

    def scale_for_step(step_idx: int) -> float:
        if lr_cfg.name == "constant" or total_steps <= 0:
            return 1.0
        if warmup_steps > 0 and step_idx < warmup_steps:
            return float(step_idx + 1) / float(warmup_steps)
        if lr_cfg.name == "cosine":
            if decay_steps <= 0:
                return 1.0
            progress = max(0, min(step_idx - warmup_steps, decay_steps))
            if decay_steps == 0:
                return lr_cfg.min_lr_ratio
            frac = float(progress) / float(decay_steps)
            cosine = 0.5 * (1.0 + math.cos(math.pi * frac))
            return lr_cfg.min_lr_ratio + (1.0 - lr_cfg.min_lr_ratio) * cosine
        # warmup-stable-decay path
        if step_idx < (warmup_steps + stable_steps):
            return 1.0
        if decay_steps <= 0:
            return 1.0
        pos = step_idx - decay_start_step + 1
        pos = max(1, min(pos, decay_steps))
        frac = float(pos) / float(decay_steps)
        return 1.0 - (1.0 - lr_cfg.min_lr_ratio) * frac

    def apply(scale: float) -> float:
        for pg, base_lr in zip(optimizer.param_groups, base_lrs):
            pg["lr"] = float(base_lr) * float(scale)
        return optimizer.param_groups[0]["lr"]

    meta = dict(
        decay_steps=decay_steps,
        warmup_steps=warmup_steps,
        stable_steps=stable_steps,
        decay_start_step=decay_start_step,
    )
    return scale_for_step, apply, meta


__all__ = [
    "apply_dropout_from_config",
    "load_training_encoder",
    "maybe_compile_model",
    "use_fp32_master_weights",
    "init_grad_scaler",
    "move_model_to_device",
    "init_model",
    "init_optimizer",
    "make_scheduler",
]
