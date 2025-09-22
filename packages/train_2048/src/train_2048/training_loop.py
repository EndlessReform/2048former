from __future__ import annotations

from typing import Dict, Optional, Tuple, Callable
from pathlib import Path
import math
import time

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import TrainingConfig, load_encoder_from_init
from .dataloader import build_steps_dataloaders
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
)


def _format_postfix(metrics: Dict[str, float | list[float] | None], lr: float, target_mode: str, dt_data_ms: float | None = None, dt_comp_ms: float | None = None) -> str:
    loss = float(metrics.get("loss", 0.0))
    base = f"loss={loss:.4f}"
    if target_mode in ("binned_ev", "macroxue_tokens"):
        head_losses = metrics.get("head_losses") or [0.0, 0.0, 0.0, 0.0]
        u, d, l, r = [float(x) for x in head_losses]
        base += f"  u/d/l/r={u:.3f}/{d:.3f}/{l:.3f}/{r:.3f}"
        if target_mode == "macroxue_tokens":
            pa = metrics.get("policy_agree")
            if pa is not None:
                base += f"  agree={float(pa) * 100:.1f}%"
    else:
        acc = metrics.get("policy_acc")
        if acc is not None:
            base += f"  policy_acc={float(acc):.3f}"
    base += f"  lr={lr:.2e}"
    if dt_data_ms is not None and dt_comp_ms is not None:
        base += f"  data={dt_data_ms:.1f}ms  comp={dt_comp_ms:.1f}ms"
    return base


def init_datasets(cfg: TrainingConfig, target_mode: str) -> tuple[DataLoader, Optional[DataLoader], int]:
    ev_tok = None
    if target_mode == "binned_ev":
        # Standardize EV tokenization via EVTokenizer wrapper
        ev_tok = EVBinnerTokenizer(BinningConfig(**cfg.binning.model_dump()))
    return build_steps_dataloaders(
        dataset_dir=cfg.dataset.resolved_dataset_dir(),
        binner=None,
        target_mode=target_mode,
        batch_size=cfg.batch.batch_size,
        train_num_steps=cfg.dataset.num_steps,
        seed=cfg.seed,
        shuffle_buffer_size=getattr(cfg.dataset, "shuffle_buffer_size", 1_000_000),
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
    )


def init_model(cfg: TrainingConfig, device: torch.device, *, target_mode: str, dl_train: Optional[DataLoader]):
    model = load_encoder_from_init(cfg.init_dir)
    model = model.to(device=(device if device.type != "cuda" else device), dtype=(torch.bfloat16 if device.type == "cuda" else None))
    objective = make_objective(target_mode, tokenizer_path=cfg.dataset.resolved_tokenizer_path())
    model = objective.prepare_model(model, device, cfg=cfg, dl_train=dl_train)
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

    def _compute_decay_steps(total: int) -> int:
        if lr_cfg.name != "warmup-stable-decay":
            return 0
        if getattr(lr_cfg, "cooldown_pct", None):
            return int(math.ceil(total * float(lr_cfg.cooldown_pct)))
        return int(lr_cfg.decay_steps or 0)

    warmup_steps = int(lr_cfg.warmup_steps or 0)
    warmup_steps = max(0, min(warmup_steps, max(total_steps, 0)))
    decay_steps = _compute_decay_steps(total_steps)
    decay_steps = max(0, min(decay_steps, max(total_steps - warmup_steps, 0)))
    stable_steps = max(0, total_steps - warmup_steps - decay_steps)
    decay_start_step = warmup_steps + stable_steps

    def scale_for_step(step_idx: int) -> float:
        if lr_cfg.name == "constant" or total_steps <= 0:
            return 1.0
        if warmup_steps > 0 and step_idx < warmup_steps:
            return float(step_idx + 1) / float(warmup_steps)
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


def _build_train_payload(metrics: Dict[str, float | list[float] | None], lr: float, target_mode: str, *, epoch: Optional[int], dt_data_ms: float, dt_comp_ms: float) -> Dict[str, float | int]:
    payload: Dict[str, float | int] = {"train/loss": float(metrics["loss"]), "train/lr": float(lr), "train/data_time_ms": float(dt_data_ms), "train/compute_time_ms": float(dt_comp_ms)}
    if epoch is not None:
        payload["train/epoch"] = int(epoch)
    if target_mode in ("binned_ev", "macroxue_tokens"):
        hl = metrics["head_losses"]
        payload.update({"train/loss_u": float(hl[0]), "train/loss_d": float(hl[1]), "train/loss_l": float(hl[2]), "train/loss_r": float(hl[3])})
    else:
        if metrics.get("policy_acc") is not None:
            payload["train/policy_acc"] = float(metrics["policy_acc"])
        if metrics.get("policy_agree") is not None:
            payload["train/policy_agree"] = float(metrics["policy_agree"])  # type: ignore[arg-type]
    return payload


def _build_val_payload(metrics: Dict[str, float | list[float] | None], target_mode: str, *, epoch: Optional[int]) -> Dict[str, float | int]:
    payload: Dict[str, float | int] = {"val/loss": float(metrics["loss"])}
    if epoch is not None:
        payload["train/epoch"] = int(epoch)
    if target_mode in ("binned_ev", "macroxue_tokens"):
        hl = metrics["head_losses"]
        payload.update({"val/loss_u": float(hl[0]), "val/loss_d": float(hl[1]), "val/loss_l": float(hl[2]), "val/loss_r": float(hl[3])})
    else:
        if metrics.get("policy_acc") is not None:
            payload["val/policy_acc"] = float(metrics["policy_acc"])
        if metrics.get("policy_agree") is not None:
            payload["val/policy_agree"] = float(metrics["policy_agree"])  # type: ignore[arg-type]
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


def run_training(cfg: TrainingConfig, device_str: str, wandb_run: Optional[object] = None) -> Tuple[Path, int]:
    device = torch.device(device_str)
    target_mode = getattr(cfg.target, "mode", "binned_ev")

    dl_train, dl_val, per_epoch_steps = init_datasets(cfg, target_mode)
    model, objective = init_model(cfg, device, target_mode=target_mode, dl_train=dl_train)
    optimizer = init_optimizer(model, cfg)

    model.train()

    fixed_steps = cfg.dataset.num_steps or 0
    epochs = (cfg.dataset.num_epochs or 1) if fixed_steps == 0 else 1
    steps_this_epoch = fixed_steps if fixed_steps > 0 else per_epoch_steps
    total_steps = fixed_steps if fixed_steps > 0 else (per_epoch_steps * max(epochs, 1))

    scale_for_step, apply_lr, sched_meta = make_scheduler(cfg, optimizer, total_steps)

    run_ckpt_dir = create_run_dir(cfg.checkpoint_dir)
    dump_training_and_model_config(run_ckpt_dir, cfg, model)

    global_step = 0
    pre_decay_flag = {"saved": False}
    best_tracker: Dict[str, float] = {}
    wandb_report_every = max(1, int(getattr(getattr(cfg, "wandb", None), "report_every", 1)))

    for epoch in range(epochs):
        it = iter(dl_train)
        pbar = tqdm(range(steps_this_epoch), desc=("Train" if fixed_steps > 0 else f"Epoch {epoch + 1}/{epochs}"), dynamic_ncols=True, total=steps_this_epoch)
        for _ in pbar:
            maybe_save_stable(model, run_ckpt_dir, global_step=global_step, decay_steps=sched_meta["decay_steps"], decay_start=sched_meta["decay_start_step"], preflag=pre_decay_flag)
            lr_now = apply_lr(scale_for_step(global_step))
            t0 = time.perf_counter()
            try:
                batch = next(it)
            except StopIteration:
                it = iter(dl_train)
                batch = next(it)
            t1 = time.perf_counter()
            metrics = objective.train_step(model, batch, optimizer, device)
            if metrics.get("policy_agree") is not None:
                if not hasattr(run_training, "_pa_ema"):
                    run_training._pa_ema = float(metrics["policy_agree"])  # type: ignore[attr-defined]
                decay = 0.95
                run_training._pa_ema = float(decay * run_training._pa_ema + (1 - decay) * float(metrics["policy_agree"]))  # type: ignore[attr-defined]
                metrics["policy_agree"] = run_training._pa_ema  # type: ignore[attr-defined]
            t2 = time.perf_counter()
            pbar.set_postfix_str(_format_postfix(metrics, lr_now, target_mode, (t1 - t0) * 1e3, (t2 - t1) * 1e3))
            if wandb_run is not None and (global_step % wandb_report_every == 0):
                _wandb_log(_build_train_payload(metrics, lr_now, target_mode, epoch=(None if fixed_steps > 0 else epoch), dt_data_ms=(t1 - t0) * 1e3, dt_comp_ms=(t2 - t1) * 1e3), step=global_step)
            _maybe_log_val(objective, model, dl_val, device, cfg=cfg, target_mode=target_mode, step=global_step, wandb_run=wandb_run, epoch=(None if fixed_steps > 0 else epoch))
            global_step += 1
            maybe_save_best(model=model, run_dir=run_ckpt_dir, evaluate_fn=objective.evaluate, dl_val=dl_val, device=device, cfg_checkpoint=cfg.checkpoint, step=global_step, epoch=(None if fixed_steps > 0 else epoch), best_tracker=best_tracker, wandb_run=wandb_run)
            if fixed_steps == 0:
                dangerous_dump_pt(cfg=cfg, run_dir=run_ckpt_dir, model=model, optimizer=optimizer, step=global_step)
        if fixed_steps == 0 and getattr(cfg, "checkpoint", None) is not None and cfg.checkpoint.every_epochs is not None and ((epoch + 1) % int(cfg.checkpoint.every_epochs) == 0):
            save_safetensors(model, run_ckpt_dir / f"model-epoch-{epoch + 1:04d}.safetensors")

    ckpt_path = save_safetensors(model, run_ckpt_dir / "model.safetensors")

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
