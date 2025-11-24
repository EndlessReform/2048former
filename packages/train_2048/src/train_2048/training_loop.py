from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Callable
from pathlib import Path
import math
import time
import hashlib

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import numpy as np

from .config import DropoutConfig, TrainingConfig, load_encoder_from_init
from .dataloader import build_steps_dataloaders
from .dataloader.metadata import MetadataDB
from .tokenization.ev_binning import EVBinnerTokenizer
from .binning import BinningConfig
from .objectives import make_objective
from .checkpointing import (
    create_run_dir,
    dump_training_and_model_config,
    save_safetensors,
    maybe_save_stable,
    maybe_save_best,
    dangerous_dump_pt,
    maybe_resume_optimizer_from_init,
    maybe_save_pt_interval,
)


def _format_postfix(
    metrics: Dict[str, float | list[float] | None],
    lr: float,
    target_mode: str,
    *,
    global_step: Optional[int] = None,
    accum_steps: Optional[int] = None,
    micro_batch_size: Optional[int] = None,
    dt_data_ms: float | None = None,
    dt_comp_ms: float | None = None,
) -> str:
    loss = float(metrics.get("loss", 0.0))
    parts: list[str] = []
    if global_step is not None:
        parts.append(f"step={global_step}")
    info_parts: list[str] = [f"loss={loss:.4f}"]
    if target_mode in ("binned_ev", "macroxue_tokens"):
        if target_mode == "macroxue_tokens":
            pa = metrics.get("policy_agreement")
            if pa is None:
                pa = metrics.get("policy_agree")
            if pa is not None:
                info_parts.append(f"agree={float(pa) * 100:.1f}%")
    value_loss = metrics.get("value_loss")
    policy_loss = metrics.get("policy_loss")
    if value_loss is not None:
        if policy_loss is not None:
            info_parts.append(f"policy_ce={float(policy_loss):.4f}")
        info_parts.append(f"value_mse={float(value_loss):.4f}")
    else:
        acc = metrics.get("policy_accuracy")
        if acc is None:
            acc = metrics.get("policy_acc")
        if acc is not None:
            info_parts.append(f"policy_acc={float(acc):.3f}")
        elif policy_loss is not None:
            info_parts.append(f"policy_ce={float(policy_loss):.4f}")
    info_parts.append(f"lr={lr:.2e}")
    if accum_steps is not None and micro_batch_size is not None and accum_steps > 1:
        effective = int(accum_steps) * int(micro_batch_size)
        parts.append(f"mb={int(micro_batch_size)} eff={effective}")
    if dt_data_ms is not None and dt_comp_ms is not None:
        info_parts.append(f"data={dt_data_ms:.1f}ms")
        info_parts.append(f"comp={dt_comp_ms:.1f}ms")
    base = "  ".join(info_parts)
    if parts:
        return f"{' '.join(parts)}  {base}"
    return base


def _extract_batch_dim(value: Any) -> Optional[int]:
    if isinstance(value, torch.Tensor):
        return int(value.shape[0])
    if hasattr(value, "shape") and getattr(value, "shape", None) is not None:
        shape = value.shape
        if isinstance(shape, (tuple, list)) and len(shape) > 0:
            return int(shape[0])
    if isinstance(value, (list, tuple)):
        return len(value)
    return None


def _infer_batch_size(batch: Any) -> int:
    if isinstance(batch, dict):
        for val in batch.values():
            size = _extract_batch_dim(val)
            if size is not None:
                return size
    elif isinstance(batch, (list, tuple)):
        for val in batch:
            size = _extract_batch_dim(val)
            if size is not None:
                return size
    else:
        size = _extract_batch_dim(batch)
        if size is not None:
            return size
    raise ValueError("Unable to infer batch size from training batch payload")


def _hash_run_ids(run_ids: Optional[np.ndarray]) -> str:
    if run_ids is None:
        return "none"
    arr = np.asarray(run_ids, dtype=np.int64)
    return hashlib.sha1(arr.tobytes()).hexdigest()


def _collect_dataset_signature(cfg: TrainingConfig) -> dict:
    ds_cfg = cfg.dataset
    metadata = MetadataDB(ds_cfg.resolved_dataset_dir())
    train_run_ids, val_run_ids = metadata.split_runs_train_val(
        run_sql=ds_cfg.run_sql,
        sql_params=ds_cfg.sql_params,
        val_run_sql=ds_cfg.val_run_sql,
        val_sql_params=ds_cfg.val_sql_params,
        val_run_pct=ds_cfg.val_run_pct,
        val_split_seed=ds_cfg.val_split_seed,
    )
    meta_train_steps = metadata.get_total_steps_for_runs(train_run_ids)
    meta_val_steps = metadata.get_total_steps_for_runs(val_run_ids) if val_run_ids is not None else 0
    return {
        "dataset_dir": str(Path(ds_cfg.resolved_dataset_dir()).resolve()),
        "train_run_ids": train_run_ids,
        "val_run_ids": val_run_ids,
        "meta_train_steps": int(meta_train_steps),
        "meta_val_steps": int(meta_val_steps),
    }


def _compute_dataset_fingerprint(signature: dict) -> str:
    digest = hashlib.sha1()
    digest.update(signature["dataset_dir"].encode("utf-8"))
    digest.update(str(signature["meta_train_steps"]).encode("utf-8"))
    digest.update(str(signature["meta_val_steps"]).encode("utf-8"))
    digest.update(_hash_run_ids(signature.get("train_run_ids")).encode("utf-8"))
    digest.update(_hash_run_ids(signature.get("val_run_ids")).encode("utf-8"))
    return digest.hexdigest()


def _to_serializable(obj: Any) -> Any:
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(v) for v in obj]
    return obj


def _build_dataset_checkpoint_metadata(signature: dict, dataloader_meta: dict, fingerprint: str) -> dict:
    sampler_info = _to_serializable(dataloader_meta.get("sampler", {}))
    return {
        "version": 1,
        "fingerprint": fingerprint,
        "dataset_dir": signature["dataset_dir"],
        "meta_train_steps": int(signature["meta_train_steps"]),
        "meta_val_steps": int(signature["meta_val_steps"]),
        "train_run_count": int(len(signature["train_run_ids"])),
        "train_runs_hash": _hash_run_ids(signature["train_run_ids"]),
        "val_run_count": int(len(signature["val_run_ids"])) if signature.get("val_run_ids") is not None else 0,
        "val_runs_hash": _hash_run_ids(signature.get("val_run_ids")),
        "sampler": sampler_info,
        "train_num_steps": dataloader_meta.get("train_num_steps"),
        "num_epochs": dataloader_meta.get("num_epochs"),
        "total_dataset_len": int(dataloader_meta.get("total_dataset_len", 0) or 0),
        "train_dataset_len": int(dataloader_meta.get("train_dataset_len", 0) or 0),
        "resume_skip_samples": int(dataloader_meta.get("resume_skip_samples", 0) or 0),
    }


def _build_resume_state(global_step: int, samples_consumed: int, skip_samples: int, cfg: TrainingConfig) -> dict:
    return {
        "version": 1,
        "global_step": int(global_step),
        "samples_consumed": int(samples_consumed),
        "skip_samples": int(skip_samples),
        "effective_batch_size": int(cfg.batch.batch_size),
        "micro_batch_size": int(cfg.batch.physical_batch_size()),
        "grad_accum_steps": int(cfg.batch.grad_accum_steps()),
    }


def init_datasets(
    cfg: TrainingConfig,
    target_mode: str,
    *,
    train_num_steps_override: Optional[int] = None,
    resume_skip_samples: int = 0,
) -> tuple[DataLoader, Optional[DataLoader], int, dict]:
    ev_tok = None
    if target_mode == "binned_ev":
        # Standardize EV tokenization via EVTokenizer wrapper
        ev_tok = EVBinnerTokenizer(BinningConfig(**cfg.binning.model_dump()))
    return build_steps_dataloaders(
        dataset_dir=cfg.dataset.resolved_dataset_dir(),
        binner=None,
        target_mode=target_mode,
        batch_size=cfg.batch.batch_size,
        physical_batch_size=cfg.batch.physical_batch_size(),
        train_num_steps=(cfg.dataset.num_steps if train_num_steps_override is None else train_num_steps_override),
        resume_skip_samples=resume_skip_samples,
        seed=cfg.seed,
        shuffle_buffer_size=getattr(cfg.dataset, "shuffle_buffer_size", 1_000_000),
        shard_locality=getattr(cfg.dataset, "shard_locality", False),
        shard_locality_block_size=getattr(cfg.dataset, "shard_locality_block_size", None),
        shard_cache_in_memory=getattr(cfg.dataset, "shard_cache_in_memory", False),
        shard_cache_keep_shards=getattr(cfg.dataset, "shard_cache_keep_shards", 1),
        val_num_steps=getattr(cfg.dataset, "val_num_steps", None),
        val_steps_pct=getattr(cfg.dataset, "val_steps_pct", 0.0),
        tokenizer_path=cfg.dataset.resolved_tokenizer_path(),
        ev_tokenizer=ev_tok,
        run_sql=cfg.dataset.run_sql,
        sql_params=cfg.dataset.sql_params,
        val_run_sql=cfg.dataset.val_run_sql,
        val_sql_params=cfg.dataset.val_sql_params,
        val_run_pct=cfg.dataset.val_run_pct,
        val_split_seed=cfg.dataset.val_split_seed,
        num_workers_train=12,
        mmap_mode=cfg.dataset.mmap_mode,
        value_cfg=getattr(cfg, "value_training", None),
    )


def _apply_dropout_from_config(model: torch.nn.Module, dropout_cfg: DropoutConfig) -> None:
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


def init_model(cfg: TrainingConfig, device: torch.device, *, target_mode: str, dl_train: Optional[DataLoader]):
    model = load_encoder_from_init(cfg.init_dir)
    _apply_dropout_from_config(model, cfg.dropout)
    model = model.to(device=(device if device.type != "cuda" else device), dtype=(torch.bfloat16 if device.type == "cuda" else None))
    objective = make_objective(target_mode, tokenizer_path=cfg.dataset.resolved_tokenizer_path())
    model = objective.prepare_model(model, device, cfg=cfg, dl_train=dl_train)
    _apply_dropout_from_config(model, cfg.dropout)
    if getattr(cfg, "compile_enabled", True):
        model = torch.compile(model, mode="reduce-overhead")
    return model, objective


def init_optimizer(model: torch.nn.Module, cfg: TrainingConfig) -> torch.optim.Optimizer:
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
                "embed" in lname or "embedding" in lname or "token_emb" in lname or "lm_head" in lname or "classifier" in lname or "out_proj.weight" in lname
            )

        muon_params, adam_decay, adam_no_decay = [], [], []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if p.ndim >= 2 and not _is_embedding_or_head(name):
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
            dict(params=adam_decay, use_muon=False, lr=adam_lr, betas=(opt_cfg.beta1, opt_cfg.beta2), eps=opt_cfg.eps, weight_decay=opt_cfg.weight_decay),
            dict(params=adam_no_decay, use_muon=False, lr=adam_lr, betas=(opt_cfg.beta1, opt_cfg.beta2), eps=opt_cfg.eps, weight_decay=0.0),
        ]
        return MuonWithAuxAdam(param_groups)
    raise ValueError(f"Unknown optimizer: {opt_cfg.name}")


def make_scheduler(cfg: TrainingConfig, optimizer: torch.optim.Optimizer, total_steps: int) -> tuple[Callable[[int], float], Callable[[float], float], dict]:
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


def _wandb_log(data: Dict[str, float | int], step: int) -> None:
    try:
        import wandb  # type: ignore
        wandb.log(data, step=step)
    except Exception:
        pass


def _build_train_payload(
    metrics: Dict[str, float | list[float] | None],
    lr: float,
    target_mode: str,
    *,
    epoch: Optional[int],
    dt_data_ms: float,
    dt_comp_ms: float,
    effective_batch_size: Optional[int],
    accum_steps: Optional[int],
) -> Dict[str, float | int]:
    payload: Dict[str, float | int] = {
        "train/loss": float(metrics["loss"]),
        "train/lr": float(lr),
        "train/data_time_ms": float(dt_data_ms),
        "train/compute_time_ms": float(dt_comp_ms),
    }
    if epoch is not None:
        payload["train/epoch"] = int(epoch)
    if target_mode in ("binned_ev", "macroxue_tokens"):
        hl = metrics["head_losses"]
        payload.update({"train/loss_u": float(hl[0]), "train/loss_d": float(hl[1]), "train/loss_l": float(hl[2]), "train/loss_r": float(hl[3])})
    else:
        # Hard target path (e.g., hard_move): log canonical policy accuracy only.
        acc = metrics.get("policy_accuracy")
        if acc is None:
            acc = metrics.get("policy_acc")
        if acc is not None:
            payload["train/policy_accuracy"] = float(acc)
    if metrics.get("policy_loss") is not None:
        payload["train/policy_loss"] = float(metrics["policy_loss"])
        payload["train/policy_ce"] = float(metrics["policy_loss"])
    if metrics.get("value_loss") is not None:
        payload["train/value_loss"] = float(metrics["value_loss"])
        payload["train/value_mse"] = float(metrics["value_loss"])
    if effective_batch_size is not None:
        payload["train/effective_batch_size"] = int(effective_batch_size)
    if accum_steps is not None:
        payload["train/accum_steps"] = int(accum_steps)
    return payload


def _build_val_payload(metrics: Dict[str, float | list[float] | None], target_mode: str, *, epoch: Optional[int]) -> Dict[str, float | int]:
    payload: Dict[str, float | int] = {"val/loss": float(metrics["loss"])}
    if epoch is not None:
        payload["train/epoch"] = int(epoch)
    if target_mode in ("binned_ev", "macroxue_tokens"):
        hl = metrics["head_losses"]
        payload.update({"val/loss_u": float(hl[0]), "val/loss_d": float(hl[1]), "val/loss_l": float(hl[2]), "val/loss_r": float(hl[3])})
    else:
        # Hard target path (e.g., hard_move): log canonical policy accuracy only.
        acc = metrics.get("policy_accuracy")
        if acc is None:
            acc = metrics.get("policy_acc")
        if acc is not None:
            payload["val/policy_accuracy"] = float(acc)
    if metrics.get("policy_loss") is not None:
        payload["val/policy_loss"] = float(metrics["policy_loss"])
        payload["val/policy_ce"] = float(metrics["policy_loss"])
    if metrics.get("value_loss") is not None:
        payload["val/value_loss"] = float(metrics["value_loss"])
        payload["val/value_mse"] = float(metrics["value_loss"])
    return payload


def _maybe_log_val(objective, model, dl_val, device, *, cfg: TrainingConfig, target_mode: str, step: int, wandb_run: Optional[object], epoch: Optional[int]) -> Optional[Dict[str, float | list[float] | None]]:
    if dl_val is None:
        return None
    if (cfg.dataset.val_every or 0) <= 0:
        return None
    if step <= 0 or (step % int(cfg.dataset.val_every)) != 0:
        return None
    val_metrics = objective.evaluate(model, dl_val, device)
    if wandb_run is not None:
        _wandb_log(_build_val_payload(val_metrics, target_mode, epoch=epoch), step=step)
    return val_metrics


def _accumulate_metric_sums(
    sums: Dict[str, list[float] | float],
    counts: Dict[str, int],
    metrics: Dict[str, float | list[float] | None],
) -> None:
    for key, value in metrics.items():
        if value is None:
            continue
        if isinstance(value, list):
            arr = [float(x) for x in value]
            prev = sums.get(key)
            if prev is None or not isinstance(prev, list):
                sums[key] = arr
            else:
                if len(prev) != len(arr):
                    raise ValueError(f"Metric list length mismatch for '{key}'")
                sums[key] = [p + a for p, a in zip(prev, arr)]
            counts[key] = counts.get(key, 0) + 1
        else:
            sums[key] = float(sums.get(key, 0.0)) + float(value)
            counts[key] = counts.get(key, 0) + 1


def _finalize_metric_sums(
    sums: Dict[str, list[float] | float],
    counts: Dict[str, int],
    last_metrics: Dict[str, float | list[float] | None],
) -> Dict[str, float | list[float] | None]:
    result: Dict[str, float | list[float] | None] = {}
    keys = set(last_metrics.keys()) | set(sums.keys())
    for key in keys:
        count = counts.get(key, 0)
        if key in sums and count > 0:
            total = sums[key]
            if isinstance(total, list):
                result[key] = [v / count for v in total]
            else:
                result[key] = total / count
        else:
            result[key] = last_metrics.get(key)
    return result


def run_training(cfg: TrainingConfig, device_str: str, wandb_run: Optional[object] = None, *, show_timing_in_bar: bool = False) -> Tuple[Path, int]:
    device = torch.device(device_str)
    target_mode = getattr(cfg.target, "mode", "binned_ev")

    dataset_signature = _collect_dataset_signature(cfg)
    dataset_fingerprint = _compute_dataset_fingerprint(dataset_signature)

    model = load_encoder_from_init(cfg.init_dir)
    _apply_dropout_from_config(model, cfg.dropout)
    init_info = getattr(model, "_init_load_info", {})
    weight_type = init_info.get("weights_type", "unknown")
    bundle_meta = init_info.get("bundle_metadata") if weight_type == "pt" else {}
    available_pt = init_info.get("available_pt", [])
    bundle_path_from_weights = init_info.get("weights_path") if weight_type == "pt" else None

    resume_payload_meta = bundle_meta.get("resume") if isinstance(bundle_meta, dict) else {}
    resume_dataset_meta = bundle_meta.get("dataset") if isinstance(bundle_meta, dict) else {}
    resume_global_step_meta = bundle_meta.get("global_step") if isinstance(bundle_meta, dict) else None

    effective_batch_size_cfg = int(cfg.batch.batch_size)
    resume_samples_consumed = None
    if isinstance(resume_payload_meta, dict) and "samples_consumed" in resume_payload_meta:
        try:
            resume_samples_consumed = int(resume_payload_meta["samples_consumed"])
        except Exception:
            resume_samples_consumed = None
    if resume_samples_consumed is None and resume_global_step_meta is not None:
        try:
            resume_samples_consumed = int(resume_global_step_meta) * effective_batch_size_cfg
        except Exception:
            resume_samples_consumed = None

    fingerprint_loaded = resume_dataset_meta.get("fingerprint") if isinstance(resume_dataset_meta, dict) else None
    dataset_dirs_match = (
        isinstance(resume_dataset_meta, dict)
        and resume_dataset_meta.get("dataset_dir")
        and resume_dataset_meta.get("dataset_dir") == dataset_signature["dataset_dir"]
    )
    dataset_match = bool(fingerprint_loaded) and fingerprint_loaded == dataset_fingerprint
    if not dataset_match and fingerprint_loaded is None:
        dataset_match = dataset_dirs_match

    resume_skip_samples = int(resume_samples_consumed or 0) if dataset_match else 0
    if resume_skip_samples > 0 and dataset_match:
        approx_steps = resume_skip_samples / max(1, effective_batch_size_cfg)
        print(f"[resume] Resuming data stream after {resume_skip_samples:,} samples (~{approx_steps:,.0f} steps)")
    elif (resume_samples_consumed or 0) > 0 and not dataset_match:
        print(
            "[resume] Dataset metadata mismatch between checkpoint and current configuration; "
            "training data will restart from the beginning."
        )

    dl_train, dl_val, per_epoch_steps, dataloader_meta = init_datasets(
        cfg,
        target_mode,
        train_num_steps_override=cfg.dataset.num_steps,
        resume_skip_samples=resume_skip_samples,
    )

    model = model.to(
        device=(device if device.type != "cuda" else device),
        dtype=(torch.bfloat16 if device.type == "cuda" else None),
    )
    objective = make_objective(target_mode, tokenizer_path=cfg.dataset.resolved_tokenizer_path())
    model = objective.prepare_model(model, device, cfg=cfg, dl_train=dl_train)
    _apply_dropout_from_config(model, cfg.dropout)
    if getattr(cfg, "compile_enabled", True):
        model = torch.compile(model, mode="reduce-overhead")

    try:
        n_params = sum(int(p.numel()) for p in model.parameters())
        print(f"Model parameters: {n_params:,}")
    except Exception:
        pass

    optimizer = init_optimizer(model, cfg)

    resumed_step: Optional[int] = None
    resume_state = None
    bundle_path_for_resume = None
    resolved_init_path = init_info.get("resolved_init_path", cfg.init_dir)
    init_dir_path = Path(resolved_init_path)
    if bundle_path_from_weights:
        bundle_path_for_resume = Path(bundle_path_from_weights)
    elif init_dir_path.is_file() and init_dir_path.suffix.lower() in {".pt", ".pth"}:
        bundle_path_for_resume = init_dir_path
    else:
        if available_pt:
            print(
                "[resume] WARNING: PT bundle(s) found alongside safetensors "
                f"{available_pt}, but weights were loaded from safetensors. "
                "Optimizer state will NOT be resumed to avoid desync."
            )

    try:
        if bundle_path_for_resume is not None:
            resume_state = maybe_resume_optimizer_from_init(
                resolved_init_path, optimizer, bundle_path=str(bundle_path_for_resume)
            )
            if resume_state is not None:
                resumed_step = resume_state.global_step
        else:
            resume_state = None
    except Exception as exc:
        print(f"[resume] Optimizer resume attempt failed: {exc}")
        resume_state = None

    if resume_state and resume_state.global_step is not None:
        print(f"[resume] Optimizer resumed; starting from global_step={resume_state.global_step}")
    elif resume_state is None and bundle_path_for_resume is not None:
        print("[resume] No optimizer state found in checkpoint; continuing with fresh optimizer.")

    model.train()

    global_step = 0
    if resume_state and resume_state.global_step is not None:
        global_step = int(resume_state.global_step)
    elif resume_global_step_meta is not None:
        try:
            global_step = int(resume_global_step_meta)
        except Exception:
            global_step = 0

    samples_consumed = int(resume_skip_samples)

    fixed_steps = int(cfg.dataset.num_steps or 0)
    if fixed_steps > 0:
        total_steps = fixed_steps
        remaining_steps = max(0, fixed_steps - global_step)
        steps_this_epoch = remaining_steps
        epochs = 1
    else:
        epochs = int(cfg.dataset.num_epochs or 1)
        steps_this_epoch = per_epoch_steps
        total_steps = per_epoch_steps * max(epochs, 1)
        remaining_steps = steps_this_epoch

    if remaining_steps <= 0 and fixed_steps > 0:
        print(f"[resume] Target steps ({fixed_steps}) already reached at global_step={global_step}; no training steps remain.")

    scale_for_step, apply_lr, sched_meta = make_scheduler(cfg, optimizer, total_steps)

    run_ckpt_dir = create_run_dir(cfg.checkpoint_dir)
    print(f"Checkpoint directory: {run_ckpt_dir}")
    dump_training_and_model_config(run_ckpt_dir, cfg, model)

    dataset_checkpoint_meta = _build_dataset_checkpoint_metadata(dataset_signature, dataloader_meta, dataset_fingerprint)

    print(f"[resume] Starting training loop from global_step={global_step} with {samples_consumed:,} samples consumed.")

    pre_decay_flag = {"saved": False}
    best_tracker: Dict[str, float] = {}
    wandb_report_every = max(1, int(getattr(getattr(cfg, "wandb", None), "report_every", 1)))
    base_grad_accum_steps = max(1, cfg.batch.grad_accum_steps())
    adaptive_cfg = getattr(cfg.batch, "adaptive", None)
    lr_schedule_name = getattr(cfg.hyperparameters.lr_schedule, "name", "constant")
    peak_lr = 0.0
    micro_batch_size = cfg.batch.physical_batch_size()

    total_planned_epoch_steps = steps_this_epoch if fixed_steps > 0 else per_epoch_steps

    for epoch in range(epochs):
        it = iter(dl_train)
        current_epoch_steps = total_planned_epoch_steps
        if fixed_steps > 0:
            current_epoch_steps = steps_this_epoch
        pbar = tqdm(
            range(current_epoch_steps),
            desc=("Train" if fixed_steps > 0 else f"Epoch {epoch + 1}/{epochs}"),
            dynamic_ncols=True,
            total=current_epoch_steps,
        )
        for _ in pbar:
            maybe_save_stable(
                model,
                run_ckpt_dir,
                optimizer=optimizer,
                training_cfg=cfg,
                global_step=global_step,
                decay_steps=sched_meta["decay_steps"],
                decay_start=sched_meta["decay_start_step"],
                preflag=pre_decay_flag,
                resume_state=_build_resume_state(global_step, samples_consumed, resume_skip_samples, cfg),
                dataset_metadata=dataset_checkpoint_meta,
            )
            lr_scale = scale_for_step(global_step)
            lr_now = apply_lr(lr_scale)
            peak_lr = max(peak_lr, lr_now)
            accum_multiplier = 1
            if adaptive_cfg and adaptive_cfg.enabled and lr_schedule_name == "cosine" and peak_lr > 0.0:
                lr_ratio = lr_now / peak_lr if peak_lr > 0.0 else 1.0
                accum_multiplier = adaptive_cfg.multiplier_for_ratio(lr_ratio)
            accum_steps = max(1, base_grad_accum_steps * accum_multiplier)
            loss_scale = 1.0 / float(accum_steps)

            metric_sums: Dict[str, list[float] | float] = {}
            metric_counts: Dict[str, int] = {}
            last_metrics: Dict[str, float | list[float] | None] = {}
            total_data_time = 0.0
            total_comp_time = 0.0

            for accum_idx in range(accum_steps):
                zero_grad = accum_idx == 0
                optimizer_step = accum_idx == (accum_steps - 1)
                t0 = time.perf_counter()
                try:
                    batch = next(it)
                except StopIteration:
                    it = iter(dl_train)
                    batch = next(it)
                t1 = time.perf_counter()
                metrics = objective.train_step(
                    model,
                    batch,
                    optimizer,
                    device,
                    cfg=cfg,
                    zero_grad=zero_grad,
                    optimizer_step=optimizer_step,
                    loss_scale=loss_scale,
                )
                try:
                    samples_consumed += _infer_batch_size(batch)
                except Exception:
                    samples_consumed += cfg.batch.physical_batch_size()
                last_metrics = metrics
                _accumulate_metric_sums(metric_sums, metric_counts, metrics)
                t2 = time.perf_counter()
                total_data_time += (t1 - t0)
                total_comp_time += (t2 - t1)

            aggregated_metrics = _finalize_metric_sums(metric_sums, metric_counts, last_metrics)
            agreement = aggregated_metrics.get("policy_agreement")
            if agreement is None:
                agreement = aggregated_metrics.get("policy_agree")
            if agreement is not None:
                if not hasattr(run_training, "_pa_ema"):
                    run_training._pa_ema = float(agreement)  # type: ignore[attr-defined]
                decay = 0.95
                run_training._pa_ema = float(decay * run_training._pa_ema + (1 - decay) * float(agreement))  # type: ignore[attr-defined]
                aggregated_metrics["policy_agreement"] = run_training._pa_ema  # type: ignore[attr-defined]
            dt_data_ms = total_data_time * 1e3
            dt_comp_ms = total_comp_time * 1e3
            effective_batch_now = int(micro_batch_size) * int(accum_steps)
            pbar.set_postfix_str(
                _format_postfix(
                    aggregated_metrics,
                    lr_now,
                    target_mode,
                    global_step=global_step,
                    accum_steps=accum_steps,
                    micro_batch_size=micro_batch_size,
                    dt_data_ms=(dt_data_ms if show_timing_in_bar else None),
                    dt_comp_ms=(dt_comp_ms if show_timing_in_bar else None),
                )
            )
            if wandb_run is not None and (global_step % wandb_report_every == 0):
                _wandb_log(
                    _build_train_payload(
                        aggregated_metrics,
                        lr_now,
                        target_mode,
                        epoch=(None if fixed_steps > 0 else epoch),
                        dt_data_ms=dt_data_ms,
                        dt_comp_ms=dt_comp_ms,
                        effective_batch_size=effective_batch_now,
                        accum_steps=accum_steps,
                    ),
                    step=global_step,
                )
            _maybe_log_val(objective, model, dl_val, device, cfg=cfg, target_mode=target_mode, step=global_step, wandb_run=wandb_run, epoch=(None if fixed_steps > 0 else epoch))
            global_step += 1
            resume_state_dict = _build_resume_state(global_step, samples_consumed, resume_skip_samples, cfg)
            maybe_save_pt_interval(
                model=model,
                run_dir=run_ckpt_dir,
                optimizer=optimizer,
                training_cfg=cfg,
                step=global_step,
                interval=getattr(cfg.checkpoint, "save_pt_every_steps", None),
                resume_state=resume_state_dict,
                dataset_metadata=dataset_checkpoint_meta,
            )
            maybe_save_best(
                model=model,
                run_dir=run_ckpt_dir,
                evaluate_fn=objective.evaluate,
                dl_val=dl_val,
                device=device,
                cfg_checkpoint=cfg.checkpoint,
                step=global_step,
                epoch=(None if fixed_steps > 0 else epoch),
                best_tracker=best_tracker,
                optimizer=optimizer,
                training_cfg=cfg,
                wandb_run=wandb_run,
                resume_state=resume_state_dict,
                dataset_metadata=dataset_checkpoint_meta,
            )
            if fixed_steps == 0:
                dangerous_dump_pt(cfg=cfg, run_dir=run_ckpt_dir, model=model, optimizer=optimizer, step=global_step)
        if fixed_steps == 0 and getattr(cfg, "checkpoint", None) is not None and cfg.checkpoint.every_epochs is not None and ((epoch + 1) % int(cfg.checkpoint.every_epochs) == 0):
            save_safetensors(model, run_ckpt_dir / f"model-epoch-{epoch + 1:04d}.safetensors")

    ckpt_path = save_safetensors(model, run_ckpt_dir / "model.safetensors")
    print(f"Final checkpoint saved: {ckpt_path}")

    if wandb_run is not None:
        try:
            import wandb  # type: ignore
            wandb.summary["final/global_step"] = global_step
            wandb.summary["final/checkpoint_path"] = str(ckpt_path)
            if pre_decay_flag.get("saved", False):
                wandb.summary["stable/checkpoint_path"] = str(run_ckpt_dir / "model-stable.safetensors")
            wandb.finish()
        except Exception:
            pass

    return ckpt_path, global_step


__all__ = ["run_training"]
