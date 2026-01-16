from __future__ import annotations

from typing import Optional
from pathlib import Path
import contextlib
import time

import torch
from tqdm import tqdm

from .checkpointing import (
    create_run_dir,
    dump_training_and_model_config,
    save_safetensors,
    maybe_save_stable,
    maybe_save_best,
    dangerous_dump_pt,
    maybe_save_pt_interval,
)
from .config import TrainingConfig
from .objectives import make_objective
from .training_data import (
    collect_dataset_signature,
    compute_dataset_fingerprint,
    init_datasets,
    infer_batch_size,
    build_dataset_checkpoint_metadata,
)
from .grad_logging import compute_grad_norm_stats, dump_named_grads
from .training_metrics import (
    format_postfix,
    wandb_log,
    build_train_payload,
    maybe_log_val,
    accumulate_metric_sums,
    finalize_metric_sums,
)
from .training_model import (
    apply_dropout_from_config,
    init_grad_scaler,
    init_optimizer,
    make_scheduler,
    load_training_encoder,
    maybe_compile_model,
    move_model_to_device,
    use_fp32_master_weights,
)
from .training_resume import (
    build_resume_state,
    extract_resume_metadata,
    resolve_resume_bundle_path,
    resolve_resume_skip_samples,
    maybe_resume_optimizer_state,
)


def run_training(
    cfg: TrainingConfig,
    device_str: str,
    wandb_run: Optional[object] = None,
    *,
    profile: bool = False,
    profile_start: int = 2,
    profile_end: int = 10,
) -> tuple[Path, int]:
    """Run a training loop for the given configuration."""
    device = torch.device(device_str)
    use_fp32_master = use_fp32_master_weights(cfg, device)
    grad_scaler = init_grad_scaler(cfg, device, use_fp32_master_weights=use_fp32_master)
    target_mode = getattr(cfg.target, "mode", "binned_ev")
    profile_start = max(0, int(profile_start))
    profile_end = max(profile_start, int(profile_end))
    profile_enabled = bool(profile) and device.type == "cuda"
    profiler_active = False
    profiler: Optional[torch.profiler.profile] = None
    profile_trace_path: Optional[Path] = None
    if profile and device.type != "cuda":
        print("[profile] Torch profiling requested but device is not CUDA; ignoring.")

    dataset_signature = collect_dataset_signature(cfg)
    dataset_fingerprint = compute_dataset_fingerprint(dataset_signature)

    model = load_training_encoder(cfg, device)
    apply_dropout_from_config(model, cfg.dropout)
    init_info = getattr(model, "_init_load_info", {})
    resume_payload_meta, resume_dataset_meta, resume_global_step_meta = extract_resume_metadata(init_info)

    resume_skip_samples = resolve_resume_skip_samples(
        cfg,
        dataset_signature,
        dataset_fingerprint,
        resume_payload_meta=resume_payload_meta,
        resume_dataset_meta=resume_dataset_meta,
        resume_global_step_meta=resume_global_step_meta,
    )

    dl_train, dl_val, per_epoch_steps, dataloader_meta = init_datasets(
        cfg,
        target_mode,
        train_num_steps_override=cfg.dataset.num_steps,
        resume_skip_samples=resume_skip_samples,
    )

    objective = make_objective(target_mode, tokenizer_path=cfg.dataset.resolved_tokenizer_path())
    model = objective.prepare_model(model, device, cfg=cfg, dl_train=dl_train)
    apply_dropout_from_config(model, cfg.dropout)
    model = move_model_to_device(model, device, use_fp32_master_weights=use_fp32_master)
    model = maybe_compile_model(model, cfg, device)

    try:
        n_params = sum(int(p.numel()) for p in model.parameters())
        print(f"Model parameters: {n_params:,}")
    except Exception:
        pass

    optimizer = init_optimizer(model, cfg)

    bundle_path_for_resume, resolved_init_path = resolve_resume_bundle_path(init_info, cfg)
    resume_state = maybe_resume_optimizer_state(resolved_init_path, optimizer, bundle_path_for_resume)

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
    if profile_enabled:
        profile_dir = run_ckpt_dir / "profiles"
        profile_dir.mkdir(parents=True, exist_ok=True)
        profile_trace_path = profile_dir / f"torch_profile_step_{profile_start:06d}_{profile_end:06d}.json"

    dataset_checkpoint_meta = build_dataset_checkpoint_metadata(dataset_signature, dataloader_meta, dataset_fingerprint)

    print(f"[resume] Starting training loop from global_step={global_step} with {samples_consumed:,} samples consumed.")

    pre_decay_flag = {"saved": False}
    best_tracker: dict[str, float] = {}
    wandb_report_every = max(1, int(getattr(getattr(cfg, "wandb", None), "report_every", 1)))
    grad_norm_every = int(getattr(getattr(cfg, "grad_logging", None), "norm_every_steps", 0) or 0)
    grad_dump_every = int(getattr(getattr(cfg, "grad_logging", None), "dump_every_steps", 0) or 0)
    grad_dump_names = list(getattr(getattr(cfg, "grad_logging", None), "dump_param_names", []) or [])
    grad_dump_dir_cfg = getattr(getattr(cfg, "grad_logging", None), "dump_dir", None)
    grad_dump_dir: Optional[Path] = None
    if grad_dump_dir_cfg:
        dump_path = Path(str(grad_dump_dir_cfg))
        grad_dump_dir = dump_path if dump_path.is_absolute() else (run_ckpt_dir / dump_path)
    elif grad_dump_every > 0 and grad_dump_names:
        grad_dump_dir = run_ckpt_dir / "grads"
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
            step_id = global_step
            in_profile_step = profile_enabled and profile_start <= step_id <= profile_end
            if profile_enabled and step_id == profile_start and not profiler_active:
                profiler = torch.profiler.profile(
                    activities=[
                        torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA,
                    ],
                    record_shapes=True,
                    profile_memory=True,
                    with_stack=False,
                )
                profiler.start()
                profiler_active = True
                if profile_trace_path is not None:
                    print(
                        "[profile] Torch profiler active "
                        f"(steps {profile_start}-{profile_end} -> {profile_trace_path})"
                    )
            maybe_save_stable(
                model,
                run_ckpt_dir,
                optimizer=optimizer,
                training_cfg=cfg,
                global_step=global_step,
                decay_steps=sched_meta["decay_steps"],
                decay_start=sched_meta["decay_start_step"],
                preflag=pre_decay_flag,
                resume_state=build_resume_state(global_step, samples_consumed, resume_skip_samples, cfg),
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

            metric_sums: dict[str, list[float] | float] = {}
            metric_counts: dict[str, int] = {}
            last_metrics: dict[str, float | list[float] | None] = {}
            total_data_time = 0.0
            total_comp_time = 0.0

            profile_context: contextlib.AbstractContextManager[None]
            if in_profile_step and profiler_active and profiler is not None:
                profile_context = torch.profiler.record_function(f"train_step_{step_id}")
            else:
                profile_context = contextlib.nullcontext()
            with profile_context:
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
                        grad_scaler=grad_scaler,
                        zero_grad=zero_grad,
                        optimizer_step=optimizer_step,
                        loss_scale=loss_scale,
                    )
                    try:
                        samples_consumed += infer_batch_size(batch)
                    except Exception:
                        samples_consumed += cfg.batch.physical_batch_size()
                    last_metrics = metrics
                    accumulate_metric_sums(metric_sums, metric_counts, metrics)
                    t2 = time.perf_counter()
                    total_data_time += (t1 - t0)
                    total_comp_time += (t2 - t1)
            if in_profile_step and profiler_active and profiler is not None:
                profiler.step()

            aggregated_metrics = finalize_metric_sums(metric_sums, metric_counts, last_metrics)
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
                format_postfix(
                    aggregated_metrics,
                    lr_now,
                    target_mode,
                    global_step=global_step,
                    accum_steps=accum_steps,
                    micro_batch_size=micro_batch_size,
                    dt_data_ms=dt_data_ms,
                    dt_comp_ms=dt_comp_ms,
                )
            )
            if wandb_run is not None and (global_step % wandb_report_every == 0):
                wandb_log(
                    build_train_payload(
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
            if grad_norm_every > 0 and global_step > 0 and (global_step % grad_norm_every) == 0:
                grad_stats = compute_grad_norm_stats(model)
                if grad_stats is not None:
                    wandb_log(
                        {
                            "grad/norm_mean": grad_stats.mean,
                            "grad/norm_std": grad_stats.std,
                            "grad/norm_p95": grad_stats.p95,
                            "grad/norm_max": grad_stats.max,
                            "grad/norm_global": grad_stats.global_norm,
                            "grad/norm_count": grad_stats.count,
                        },
                        step=global_step,
                    )
            if (
                grad_dump_every > 0
                and grad_dump_dir is not None
                and grad_dump_names
                and global_step > 0
                and (global_step % grad_dump_every) == 0
            ):
                dump_path = dump_named_grads(
                    model,
                    grad_dump_names,
                    step=global_step,
                    out_dir=grad_dump_dir,
                )
                if dump_path is not None:
                    print(f"[grad] Saved gradient dump: {dump_path}")
            maybe_log_val(
                objective,
                model,
                dl_val,
                device,
                cfg=cfg,
                target_mode=target_mode,
                step=global_step,
                wandb_run=wandb_run,
                epoch=(None if fixed_steps > 0 else epoch),
            )
            global_step += 1
            resume_state_dict = build_resume_state(global_step, samples_consumed, resume_skip_samples, cfg)
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
            if profile_enabled and profiler_active and step_id == profile_end:
                if profiler is not None:
                    profiler.stop()
                    if profile_trace_path is not None:
                        profiler.export_chrome_trace(str(profile_trace_path))
                        print(f"[profile] Trace written: {profile_trace_path}")
                    profiler = None
                profiler_active = False
            if fixed_steps == 0:
                dangerous_dump_pt(cfg=cfg, run_dir=run_ckpt_dir, model=model, optimizer=optimizer, step=global_step)
        if (
            fixed_steps == 0
            and getattr(cfg, "checkpoint", None) is not None
            and cfg.checkpoint.every_epochs is not None
            and ((epoch + 1) % int(cfg.checkpoint.every_epochs) == 0)
        ):
            save_safetensors(model, run_ckpt_dir / f"model-epoch-{epoch + 1:04d}.safetensors")

    if profile_enabled and profiler_active and profiler is not None:
        profiler.stop()
        if profile_trace_path is not None:
            profiler.export_chrome_trace(str(profile_trace_path))
            print(f"[profile] Trace written: {profile_trace_path}")
        profiler = None
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
