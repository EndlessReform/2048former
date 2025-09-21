from __future__ import annotations

from typing import Dict, Optional, Tuple
from pathlib import Path
import math

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import TrainingConfig, load_encoder_from_init
from .dataloader import build_steps_dataloaders
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


def run_training(cfg: TrainingConfig, device_str: str, wandb_run: Optional[object] = None) -> Tuple[Path, int]:
    device = torch.device(device_str)

    # Init model
    model = load_encoder_from_init(cfg.init_dir)
    model = model.to(device=(device if device.type != "cuda" else device), dtype=(torch.bfloat16 if device.type == "cuda" else None))

    # Data
    target_mode = getattr(cfg.target, "mode", "binned_ev")
    dl_train, dl_val, per_epoch_steps = build_steps_dataloaders(
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
        run_sql=cfg.dataset.run_sql,
        sql_params=cfg.dataset.sql_params,
        val_run_sql=cfg.dataset.val_run_sql,
        val_sql_params=cfg.dataset.val_sql_params,
        val_run_pct=cfg.dataset.val_run_pct,
        val_split_seed=cfg.dataset.val_split_seed,
        num_workers_train=12,
        mmap_mode=cfg.dataset.mmap_mode,
    )

    # Objective
    objective = make_objective(target_mode, tokenizer_path=cfg.dataset.resolved_tokenizer_path())
    model = objective.prepare_model(model, device, cfg=cfg, dl_train=dl_train)

    # Compile
    if getattr(cfg, "compile_enabled", True):
        model = torch.compile(model, mode="reduce-overhead")

    # Optimizer
    opt_cfg = cfg.hyperparameters.optimizer
    if opt_cfg.name == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg.hyperparameters.learning_rate,
            betas=(opt_cfg.beta1, opt_cfg.beta2),
            eps=opt_cfg.eps,
            weight_decay=opt_cfg.weight_decay,
        )
    elif opt_cfg.name == "muon":
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
        optimizer = MuonWithAuxAdam(param_groups)
    else:
        raise ValueError(f"Unknown optimizer: {opt_cfg.name}")

    model.train()

    # Steps/epochs
    fixed_steps = cfg.dataset.num_steps or 0
    epochs = cfg.dataset.num_epochs or 1
    total_steps = fixed_steps if fixed_steps > 0 else (per_epoch_steps * max(epochs, 1))

    lr_cfg = cfg.hyperparameters.lr_schedule
    base_lrs = [pg.get("lr", cfg.hyperparameters.learning_rate) for pg in optimizer.param_groups]

    def _compute_decay_steps(total: int) -> int:
        if lr_cfg.name != "warmup-stable-decay":
            return 0
        if getattr(lr_cfg, "cooldown_pct", None):
            return int(math.ceil(total * float(lr_cfg.cooldown_pct)))
        return int(lr_cfg.decay_steps or 0)

    decay_steps = _compute_decay_steps(total_steps)
    warmup_steps = int(lr_cfg.warmup_steps or 0)
    warmup_steps = max(0, min(warmup_steps, max(total_steps, 0)))
    decay_steps = max(0, min(decay_steps, max(total_steps - warmup_steps, 0)))
    stable_steps = max(0, total_steps - warmup_steps - decay_steps)
    decay_start_step = warmup_steps + stable_steps

    def _lr_scale_for_step(step_idx: int) -> float:
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

    def _apply_lr(scale: float) -> float:
        for pg, base_lr in zip(optimizer.param_groups, base_lrs):
            pg["lr"] = float(base_lr) * float(scale)
        return optimizer.param_groups[0]["lr"]

    # Checkpoint dir and configs
    run_ckpt_dir = create_run_dir(cfg.checkpoint_dir)
    dump_training_and_model_config(run_ckpt_dir, cfg, model)

    global_step = 0
    pre_decay_flag = {"saved": False}
    best_tracker: Dict[str, float] = {}

    wandb_report_every = max(1, int(getattr(getattr(cfg, "wandb", None), "report_every", 1)))

    def _safe_wandb_log(data: Dict[str, float | int], step: int) -> None:
        try:
            import wandb  # type: ignore
            wandb.log(data, step=step)
        except Exception:
            pass

    # Fixed steps mode
    if fixed_steps > 0:
        it = iter(dl_train)
        pbar = tqdm(range(fixed_steps), desc="Train", dynamic_ncols=True, total=fixed_steps)
        for _ in pbar:
            maybe_save_stable(model, run_ckpt_dir, global_step=global_step, decay_steps=decay_steps, decay_start=decay_start_step, preflag=pre_decay_flag)
            lr_now = _apply_lr(_lr_scale_for_step(global_step))
            import time
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
                log_payload = {"train/loss": metrics["loss"], "train/lr": lr_now, "train/data_time_ms": (t1 - t0) * 1e3, "train/compute_time_ms": (t2 - t1) * 1e3}
                if target_mode in ("binned_ev", "macroxue_tokens"):
                    hl = metrics["head_losses"]
                    log_payload.update({"train/loss_u": hl[0], "train/loss_d": hl[1], "train/loss_l": hl[2], "train/loss_r": hl[3]})
                else:
                    if metrics.get("policy_acc") is not None:
                        log_payload["train/policy_acc"] = metrics["policy_acc"]
                    if metrics.get("policy_agree") is not None:
                        log_payload["train/policy_agree"] = metrics["policy_agree"]
                _safe_wandb_log(log_payload, step=global_step)
            if dl_val is not None and (cfg.dataset.val_every or 0) > 0 and (global_step > 0 and (global_step % int(cfg.dataset.val_every) == 0)):
                val_metrics = objective.evaluate(model, dl_val, device)
                if wandb_run is not None:
                    payload = {"val/loss": val_metrics["loss"]}
                    if target_mode in ("binned_ev", "macroxue_tokens"):
                        hl = val_metrics["head_losses"]
                        payload.update({"val/loss_u": hl[0], "val/loss_d": hl[1], "val/loss_l": hl[2], "val/loss_r": hl[3]})
                    else:
                        if val_metrics.get("policy_acc") is not None:
                            payload["val/policy_acc"] = val_metrics["policy_acc"]
                        if val_metrics.get("policy_agree") is not None:
                            payload["val/policy_agree"] = val_metrics["policy_agree"]
                    _safe_wandb_log(payload, step=global_step)
            global_step += 1
            maybe_save_best(model=model, run_dir=run_ckpt_dir, evaluate_fn=objective.evaluate, dl_val=dl_val, device=device, target_mode=target_mode, cfg_checkpoint=cfg.checkpoint, step=global_step, epoch=None, best_tracker=best_tracker, wandb_run=wandb_run)
    else:
        for epoch in range(epochs):
            if per_epoch_steps > 0:
                import time
                pbar = tqdm(range(per_epoch_steps), desc=f"Epoch {epoch + 1}/{epochs}", dynamic_ncols=True, total=per_epoch_steps)
                it = iter(dl_train)
                for _ in pbar:
                    t0 = time.perf_counter()
                    try:
                        batch = next(it)
                    except StopIteration:
                        break
                    t1 = time.perf_counter()
                    lr_now = _apply_lr(_lr_scale_for_step(global_step))
                    metrics = objective.train_step(model, batch, optimizer, device)
                    t2 = time.perf_counter()
                    maybe_save_stable(model, run_ckpt_dir, global_step=global_step, decay_steps=decay_steps, decay_start=decay_start_step, preflag=pre_decay_flag)
                    pbar.set_postfix_str(_format_postfix(metrics, lr_now, target_mode, (t1 - t0) * 1e3, (t2 - t1) * 1e3))
                    if wandb_run is not None and (global_step % wandb_report_every == 0):
                        payload = {"train/loss": metrics["loss"], "train/lr": lr_now, "train/epoch": epoch, "train/data_time_ms": (t1 - t0) * 1e3, "train/compute_time_ms": (t2 - t1) * 1e3}
                        if target_mode in ("binned_ev", "macroxue_tokens"):
                            hl = metrics["head_losses"]
                            payload.update({"train/loss_u": hl[0], "train/loss_d": hl[1], "train/loss_l": hl[2], "train/loss_r": hl[3]})
                        else:
                            if metrics.get("policy_acc") is not None:
                                payload["train/policy_acc"] = metrics["policy_acc"]
                        _safe_wandb_log(payload, step=global_step)
                    if dl_val is not None and (cfg.dataset.val_every or 0) > 0 and (global_step > 0 and (global_step % int(cfg.dataset.val_every) == 0)):
                        val_metrics = objective.evaluate(model, dl_val, device)
                        if wandb_run is not None:
                            payload = {"val/loss": val_metrics["loss"], "train/epoch": epoch}
                            if target_mode in ("binned_ev", "macroxue_tokens"):
                                hl = val_metrics["head_losses"]
                                payload.update({"val/loss_u": hl[0], "val/loss_d": hl[1], "val/loss_l": hl[2], "val/loss_r": hl[3]})
                            else:
                                if val_metrics.get("policy_acc") is not None:
                                    payload["val/policy_acc"] = val_metrics["policy_acc"]
                            _safe_wandb_log(payload, step=global_step)
                    global_step += 1
                    maybe_save_best(model=model, run_dir=run_ckpt_dir, evaluate_fn=objective.evaluate, dl_val=dl_val, device=device, target_mode=target_mode, cfg_checkpoint=cfg.checkpoint, step=global_step, epoch=epoch, best_tracker=best_tracker, wandb_run=wandb_run)
                    dangerous_dump_pt(cfg=cfg, run_dir=run_ckpt_dir, model=model, optimizer=optimizer, step=global_step)
                if getattr(cfg, "checkpoint", None) is not None and cfg.checkpoint.every_epochs is not None and ((epoch + 1) % int(cfg.checkpoint.every_epochs) == 0):
                    save_safetensors(model, run_ckpt_dir / f"model-epoch-{epoch + 1:04d}.safetensors")
            else:
                import time
                pbar = tqdm(desc=f"Epoch {epoch + 1}/{epochs}", dynamic_ncols=True)
                it = iter(dl_train)
                while True:
                    t0 = time.perf_counter()
                    try:
                        batch = next(it)
                    except StopIteration:
                        break
                    t1 = time.perf_counter()
                    lr_now = _apply_lr(_lr_scale_for_step(global_step))
                    metrics = objective.train_step(model, batch, optimizer, device)
                    t2 = time.perf_counter()
                    maybe_save_stable(model, run_ckpt_dir, global_step=global_step, decay_steps=decay_steps, decay_start=decay_start_step, preflag=pre_decay_flag)
                    pbar.set_postfix_str(_format_postfix(metrics, lr_now, target_mode, (t1 - t0) * 1e3, (t2 - t1) * 1e3))
                    if wandb_run is not None and (global_step % wandb_report_every == 0):
                        payload = {"train/loss": metrics["loss"], "train/lr": lr_now, "train/epoch": epoch, "train/data_time_ms": (t1 - t0) * 1e3, "train/compute_time_ms": (t2 - t1) * 1e3}
                        if target_mode in ("binned_ev", "macroxue_tokens"):
                            hl = metrics["head_losses"]
                            payload.update({"train/loss_u": hl[0], "train/loss_d": hl[1], "train/loss_l": hl[2], "train/loss_r": hl[3]})
                        else:
                            if metrics.get("policy_acc") is not None:
                                payload["train/policy_acc"] = metrics["policy_acc"]
                        _safe_wandb_log(payload, step=global_step)
                    if dl_val is not None and (cfg.dataset.val_every or 0) > 0 and (global_step > 0 and (global_step % int(cfg.dataset.val_every) == 0)):
                        val_metrics = objective.evaluate(model, dl_val, device)
                        if wandb_run is not None:
                            payload = {"val/loss": val_metrics["loss"], "train/epoch": epoch}
                            if target_mode in ("binned_ev", "macroxue_tokens"):
                                hl = val_metrics["head_losses"]
                                payload.update({"val/loss_u": hl[0], "val/loss_d": hl[1], "val/loss_l": hl[2], "val/loss_r": hl[3]})
                            else:
                                if val_metrics.get("policy_acc") is not None:
                                    payload["val/policy_acc"] = val_metrics["policy_acc"]
                            _safe_wandb_log(payload, step=global_step)
                    global_step += 1
                    maybe_save_best(model=model, run_dir=run_ckpt_dir, evaluate_fn=objective.evaluate, dl_val=dl_val, device=device, target_mode=target_mode, cfg_checkpoint=cfg.checkpoint, step=global_step, epoch=epoch, best_tracker=best_tracker, wandb_run=wandb_run)
                if getattr(cfg, "checkpoint", None) is not None and cfg.checkpoint.every_epochs is not None and ((epoch + 1) % int(cfg.checkpoint.every_epochs) == 0):
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

