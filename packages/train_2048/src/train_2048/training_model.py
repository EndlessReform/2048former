from __future__ import annotations

from typing import Callable, Optional
import math

import torch
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader

from .config import DropoutConfig, TrainingConfig, load_encoder_from_init
from .objectives import Objective, make_objective


def apply_dropout_from_config(model: torch.nn.Module, dropout_cfg: DropoutConfig) -> None:
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


def use_fp32_master_weights(cfg: TrainingConfig, device: torch.device) -> bool:
    """Return True when master weights should stay in fp32 on CUDA."""
    amp_cfg = getattr(cfg, "amp", None)
    return device.type == "cuda" and bool(getattr(amp_cfg, "master_weights_fp32", False))


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
            return model.to(device=device, dtype=torch.float32)
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
    model = load_encoder_from_init(cfg.init_dir)
    apply_dropout_from_config(model, cfg.dropout)
    objective = make_objective(target_mode, tokenizer_path=cfg.dataset.resolved_tokenizer_path())
    model = objective.prepare_model(model, device, cfg=cfg, dl_train=dl_train)
    apply_dropout_from_config(model, cfg.dropout)
    model = move_model_to_device(model, device, use_fp32_master_weights=use_fp32_master)
    if getattr(cfg, "compile_enabled", True):
        model = torch.compile(model, mode="reduce-overhead")
    return model, objective


def init_optimizer(model: torch.nn.Module, cfg: TrainingConfig) -> torch.optim.Optimizer:
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
        from muon import MuonWithAuxAdam  # type: ignore

        def _is_norm_param(name: str, p: torch.nn.Parameter) -> bool:
            lname = name.lower()
            return ("layernorm" in lname or "ln" in lname or ".norm" in lname) and p.ndim == 1

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
            if id(p) in linear_weight_ids and p.ndim >= 2 and not _is_embedding_or_head(name):
                muon_params.append(p)
            else:
                if _is_norm_param(name, p) or p.ndim == 1 or name.endswith(".bias"):
                    adam_no_decay.append(p)
                else:
                    adam_decay.append(p)
        if len(muon_params) == 0:
            raise ValueError("No Muon-eligible parameters found (expected 2-D hidden weights).")
        muon_lr = cfg.hyperparameters.muon_lr or 2e-2
        adam_lr = cfg.hyperparameters.learning_rate
        param_groups = [
            dict(params=muon_params, use_muon=True, lr=muon_lr, weight_decay=opt_cfg.weight_decay),
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
        return MuonWithAuxAdam(param_groups)
    raise ValueError(f"Unknown optimizer: {opt_cfg.name}")


def make_scheduler(
    cfg: TrainingConfig,
    optimizer: torch.optim.Optimizer,
    total_steps: int,
) -> tuple[Callable[[int], float], Callable[[float], float], dict]:
    """Return schedule scaling and apply functions plus schedule metadata."""
    lr_cfg = cfg.hyperparameters.lr_schedule
    base_lrs = [pg.get("lr", cfg.hyperparameters.learning_rate) for pg in optimizer.param_groups]

    total_steps = max(total_steps, 0)
    warmup_steps = int(lr_cfg.warmup_steps or 0)
    warmup_steps = max(0, min(warmup_steps, total_steps if total_steps > 0 else warmup_steps))

    decay_steps = 0
    stable_steps = 0
    decay_start_step = warmup_steps

    if lr_cfg.name == "warmup-stable-decay":
        if getattr(lr_cfg, "cooldown_pct", None):
            decay_steps = int(math.ceil(total_steps * float(lr_cfg.cooldown_pct))) if total_steps > 0 else int(lr_cfg.decay_steps or 0)
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

    meta = dict(decay_steps=decay_steps, warmup_steps=warmup_steps, stable_steps=stable_steps, decay_start_step=decay_start_step)
    return scale_for_step, apply, meta


__all__ = [
    "apply_dropout_from_config",
    "use_fp32_master_weights",
    "init_grad_scaler",
    "move_model_to_device",
    "init_model",
    "init_optimizer",
    "make_scheduler",
]
