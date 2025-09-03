from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

from safetensors.torch import save_file as safe_save_file
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import os, tempfile, torch
import torch.distributed as dist
import time, json, math

from .config import TrainingConfig, load_encoder_from_init, normalize_state_dict_keys
from .binning import Binner
from .dataloader import StepBatchDataset, make_collate_step_batches


def _format_postfix(loss: float, head_losses: list[float], lr: float) -> str:
    u, d, l, r = head_losses
    return f"loss={loss:.4f}  u/d/l/r={u:.3f}/{d:.3f}/{l:.3f}/{r:.3f}  lr={lr:.2e}"


def train(
    cfg: TrainingConfig,
    device_str: str,
    wandb_run: Optional[object] = None,
) -> Tuple[Path, int]:
    """
    Run training according to the provided config.

    Returns (checkpoint_path, global_step).
    """

    device = torch.device(device_str)

    # Model
    model = load_encoder_from_init(cfg.init_dir)
    if device.type == "cuda":
        model = model.to(device=device, dtype=torch.bfloat16)
    else:
        model = model.to(device)
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
        try:
            from muon import MuonWithAuxAdam
        except ImportError as e:
            raise RuntimeError(
                "Muon not available. Install with `pip install muon-optimizer`."
            ) from e

        if (
            torch.cuda.is_available()
            and dist.is_available()
            and not dist.is_initialized()
        ):
            dist.init_process_group(
                backend="nccl",  # CUDA path
                init_method=f"file://{tempfile.gettempdir()}/pg",  # no rendezvous server
                rank=0,
                world_size=1,
            )

        def _is_norm_param(name: str, p: torch.nn.Parameter) -> bool:
            # Common LayerNorm name patterns; adjust if your code uses different ones
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

        muon_params, adam_decay, adam_no_decay = [], [], []

        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue

            if p.ndim >= 2 and not _is_embedding_or_head(name):
                # internal 2-D matrices (hidden layers) -> Muon
                muon_params.append(p)
            else:
                # everything else -> AdamW; split decay/no-decay as usual
                if _is_norm_param(name, p) or p.ndim == 1 or name.endswith(".bias"):
                    adam_no_decay.append(p)
                else:
                    adam_decay.append(p)

        if len(muon_params) == 0:
            raise ValueError(
                "No Muon-eligible parameters found (expected 2-D hidden weights)."
            )

        muon_lr = cfg.hyperparameters.muon_lr or 2e-2  # typical starting point
        adam_lr = cfg.hyperparameters.learning_rate

        param_groups = [
            # Muon group (hidden 2-D matrices)
            dict(
                params=muon_params,
                use_muon=True,
                lr=muon_lr,
                weight_decay=opt_cfg.weight_decay,
            ),
            # AdamW groups (aux optimizer inside MuonWithAuxAdam)
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
        print(
            "Muon params:",
            [name for name, p in model.named_parameters() if p in muon_params],
        )
        print(
            "Adam decay params:",
            [name for name, p in model.named_parameters() if p in adam_decay],
        )
        print(
            "Adam no decay params:",
            [name for name, p in model.named_parameters() if p in adam_no_decay],
        )

        optimizer = MuonWithAuxAdam(param_groups)
    else:
        raise ValueError(f"Unknown optimizer: {opt_cfg.name}")

    # Data: build train (and optional val) views using ai_2048 split
    binner = Binner.from_config(cfg.binning)
    split_enabled = (getattr(cfg.dataset, "val_steps", 0) or 0) > 0 or (
        getattr(cfg.dataset, "val_pct", 0.0) or 0.0
    ) > 0.0
    # Split settings
    split_unit = "step" if (cfg.dataset.val_steps or 0) > 0 else "run"
    split_test_size = int(cfg.dataset.val_steps or 0) if split_unit == "step" else None
    split_test_pct = float(cfg.dataset.val_pct or 0.0) if split_unit == "run" else None

    collate_fn = make_collate_step_batches(binner)

    # Train dataset/dataloader
    ds_train = StepBatchDataset(
        pack_path=cfg.dataset.resolved_packfile(),
        batch_size=cfg.batch.batch_size,
        shuffle=True,
        seed=cfg.seed,
        split_role=("train" if split_enabled else "none"),
        split_unit=split_unit,
        split_test_size=split_test_size,
        split_test_pct=split_test_pct,
        split_seed=42,
    )
    dl_train = DataLoader(
        ds_train,
        batch_size=None,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # Optional validation dataset/dataloader
    dl_val = None
    if split_enabled:
        ds_val = StepBatchDataset(
            pack_path=cfg.dataset.resolved_packfile(),
            batch_size=cfg.batch.batch_size,
            shuffle=False,
            seed=cfg.seed,
            split_role="val",
            split_unit=split_unit,
            split_test_size=split_test_size,
            split_test_pct=split_test_pct,
            split_seed=42,
        )
        dl_val = DataLoader(
            ds_val,
            batch_size=None,
            num_workers=2,
            collate_fn=collate_fn,
            pin_memory=True,
        )

    # Training
    model.train()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Loaded Encoder on {device_str} with {n_params:,} params")

    # Determine total steps for scheduler and progress bars
    fixed_steps = cfg.dataset.num_steps or 0
    epochs = cfg.dataset.num_epochs or 1
    per_epoch_steps = getattr(ds_train, "total_steps", 0) or 0
    total_steps = fixed_steps if fixed_steps > 0 else (per_epoch_steps * max(epochs, 1))

    # LR scheduler setup (supports constant and warmup-stable-decay)
    lr_cfg = cfg.hyperparameters.lr_schedule
    base_lrs = [
        pg.get("lr", cfg.hyperparameters.learning_rate) for pg in optimizer.param_groups
    ]

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

    # Create timestamped checkpoint directory and store configs
    ckpt_root = Path(cfg.checkpoint_dir)
    ckpt_root.mkdir(parents=True, exist_ok=True)
    run_id = time.strftime("%Y%m%d_%H%M%S")
    run_ckpt_dir = ckpt_root / run_id
    run_ckpt_dir.mkdir(parents=True, exist_ok=True)
    # - training-config.json: serialized training config used for the run
    try:
        with (run_ckpt_dir / "training-config.json").open("w", encoding="utf-8") as f:
            json.dump(cfg.model_dump(), f, indent=2)
    except Exception:
        pass
    # - config.json: serialize the model's EncoderConfig so the checkpoint is a valid init folder
    try:
        enc_cfg = getattr(model, "config", None)
        if enc_cfg is not None:
            with (run_ckpt_dir / "config.json").open("w", encoding="utf-8") as f:
                json.dump(enc_cfg.model_dump(), f, indent=2)
    except Exception:
        pass

    global_step = 0
    pre_decay_ckpt_saved = False

    # W&B logging cadence
    wandb_report_every = 1
    try:
        if getattr(cfg, "wandb", None) is not None:
            wandb_report_every = max(1, int(getattr(cfg.wandb, "report_every", 1)))
    except Exception:
        wandb_report_every = 1

    if fixed_steps > 0:
        it = iter(dl_train)
        pbar = tqdm(
            range(fixed_steps), desc="Train", dynamic_ncols=True, total=fixed_steps
        )
        for _ in pbar:
            # Save stable checkpoint right before decay starts
            if (
                not pre_decay_ckpt_saved
                and decay_steps > 0
                and global_step == decay_start_step
            ):
                _save_checkpoint(model, run_ckpt_dir / "model-stable.safetensors")
                pre_decay_ckpt_saved = True

            lr_now = _apply_lr(_lr_scale_for_step(global_step))
            # Rewind the dataloader if we exhaust it before reaching fixed_steps
            try:
                batch = next(it)
            except StopIteration:
                it = iter(dl_train)
                batch = next(it)
            metrics = train_step(model, batch, optimizer, device)
            pbar.set_postfix_str(
                _format_postfix(metrics["loss"], metrics["head_losses"], lr_now)
            )
            if wandb_run is not None and (global_step % wandb_report_every == 0):
                _safe_wandb_log(
                    {
                        "train/loss": metrics["loss"],
                        "train/loss_u": metrics["head_losses"][0],
                        "train/loss_d": metrics["head_losses"][1],
                        "train/loss_l": metrics["head_losses"][2],
                        "train/loss_r": metrics["head_losses"][3],
                        "train/lr": lr_now,
                    },
                    step=global_step,
                )
            # Periodic validation
            if dl_val is not None and (cfg.dataset.val_every or 0) > 0 and (
                global_step > 0 and (global_step % int(cfg.dataset.val_every) == 0)
            ):
                val_metrics = evaluate(model, dl_val, device)
                if wandb_run is not None:
                    _safe_wandb_log(
                        {
                            "val/loss": val_metrics["loss"],
                            "val/loss_u": val_metrics["head_losses"][0],
                            "val/loss_d": val_metrics["head_losses"][1],
                            "val/loss_l": val_metrics["head_losses"][2],
                            "val/loss_r": val_metrics["head_losses"][3],
                        },
                        step=global_step,
                    )
            global_step += 1
    else:
        for epoch in range(epochs):
            if per_epoch_steps > 0:
                pbar = tqdm(
                    dl_train,
                    desc=f"Epoch {epoch + 1}/{epochs}",
                    dynamic_ncols=True,
                    total=per_epoch_steps,
                )
            else:
                pbar = tqdm(
                    dl_train, desc=f"Epoch {epoch + 1}/{epochs}", dynamic_ncols=True
                )
            for batch in pbar:
                if (
                    not pre_decay_ckpt_saved
                    and decay_steps > 0
                    and global_step == decay_start_step
                ):
                    _save_checkpoint(model, run_ckpt_dir / "model-stable.safetensors")
                    pre_decay_ckpt_saved = True

                lr_now = _apply_lr(_lr_scale_for_step(global_step))
                metrics = train_step(model, batch, optimizer, device)
                pbar.set_postfix_str(
                    _format_postfix(metrics["loss"], metrics["head_losses"], lr_now)
                )
                if wandb_run is not None and (global_step % wandb_report_every == 0):
                    _safe_wandb_log(
                        {
                            "train/loss": metrics["loss"],
                            "train/loss_u": metrics["head_losses"][0],
                            "train/loss_d": metrics["head_losses"][1],
                            "train/loss_l": metrics["head_losses"][2],
                            "train/loss_r": metrics["head_losses"][3],
                            "train/lr": lr_now,
                            "train/epoch": epoch,
                        },
                        step=global_step,
                    )
                # Periodic validation
                if dl_val is not None and (cfg.dataset.val_every or 0) > 0 and (
                    global_step > 0 and (global_step % int(cfg.dataset.val_every) == 0)
                ):
                    val_metrics = evaluate(model, dl_val, device)
                    if wandb_run is not None:
                        _safe_wandb_log(
                            {
                                "val/loss": val_metrics["loss"],
                                "val/loss_u": val_metrics["head_losses"][0],
                                "val/loss_d": val_metrics["head_losses"][1],
                                "val/loss_l": val_metrics["head_losses"][2],
                                "val/loss_r": val_metrics["head_losses"][3],
                                "train/epoch": epoch,
                            },
                            step=global_step,
                        )
                global_step += 1

    # Save final checkpoint (canonical name expected by loaders)
    ckpt_path = _save_checkpoint(model, run_ckpt_dir / "model.safetensors")
    print(f"Saved final checkpoint: {ckpt_path}")

    # Wrap up W&B
    if wandb_run is not None:
        try:
            import wandb  # type: ignore

            wandb.summary["final/global_step"] = global_step
            wandb.summary["final/checkpoint_path"] = str(ckpt_path)
            if pre_decay_ckpt_saved:
                wandb.summary["stable/checkpoint_path"] = str(
                    run_ckpt_dir / "model-stable.safetensors"
                )
            wandb.finish()
        except Exception:
            pass

    return ckpt_path, global_step


def _safe_wandb_log(data: Dict[str, float], step: int) -> None:
    try:
        import wandb  # type: ignore

        wandb.log(data, step=step)
    except Exception:
        # Silent failure to avoid interrupting training
        pass


def train_step(
    model,
    batch: Dict[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
):
    """Single optimization step over one DataLoader batch (no micro-batching)."""

    tokens = batch["tokens"].to(device, non_blocking=True)
    branch_mask = batch["branch_mask"].to(device, non_blocking=True)
    targets_bins = batch["branch_bin_targets"].to(device, non_blocking=True)

    optimizer.zero_grad(set_to_none=True)

    if device.type == "cuda":
        autocast = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    else:

        class _NullCtx:
            def __enter__(self):
                return None

            def __exit__(self, exc_type, exc, tb):
                return False

        autocast = _NullCtx()

    with autocast:
        _hs, ev_logits = model(tokens)

        per_head_losses = []
        for h in range(4):
            logits_h = ev_logits[h].float()  # compute loss in fp32 for stability
            tgt_h = targets_bins[:, h]
            mask_h = branch_mask[:, h]

            # Cross-entropy per-sample
            loss_h = F.cross_entropy(logits_h, tgt_h, reduction="none")
            # Mask illegal moves
            if mask_h.any():
                loss_h = loss_h[mask_h].mean()
            else:
                loss_h = torch.zeros((), device=logits_h.device, dtype=torch.float32)
            per_head_losses.append(loss_h)

        loss = sum(per_head_losses)

    loss.backward()
    optimizer.step()

    head_losses = [lh.detach().item() for lh in per_head_losses]
    total_loss = loss.detach().item()

    return {"loss": total_loss, "head_losses": head_losses}


@torch.no_grad()
def evaluate(model: torch.nn.Module, dl_val: DataLoader, device: torch.device):
    was_training = model.training
    model.eval()

    total_loss = 0.0
    total_heads = torch.zeros(4, dtype=torch.float64)
    n_batches = 0

    if device.type == "cuda":
        autocast = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    else:
        class _NullCtx:
            def __enter__(self):
                return None

            def __exit__(self, exc_type, exc, tb):
                return False

        autocast = _NullCtx()

    for batch in dl_val:
        tokens = batch["tokens"].to(device, non_blocking=True)
        branch_mask = batch["branch_mask"].to(device, non_blocking=True)
        targets_bins = batch["branch_bin_targets"].to(device, non_blocking=True)

        with autocast:
            _hs, ev_logits = model(tokens)
            per_head_losses = []
            for h in range(4):
                logits_h = ev_logits[h].float()
                tgt_h = targets_bins[:, h]
                mask_h = branch_mask[:, h]
                loss_h = F.cross_entropy(logits_h, tgt_h, reduction="none")
                if mask_h.any():
                    loss_h = loss_h[mask_h].mean()
                else:
                    loss_h = torch.zeros((), device=logits_h.device, dtype=torch.float32)
                per_head_losses.append(loss_h)

            loss = sum(per_head_losses)

        total_loss += float(loss.detach().item())
        total_heads += torch.tensor(
            [lh.detach().item() for lh in per_head_losses], dtype=torch.float64
        )
        n_batches += 1

    if was_training:
        model.train()

    if n_batches == 0:
        return {"loss": 0.0, "head_losses": [0.0, 0.0, 0.0, 0.0]}

    avg_loss = float(total_loss / n_batches)
    avg_heads = (total_heads / n_batches).tolist()
    return {"loss": avg_loss, "head_losses": avg_heads}


def _save_checkpoint(model: torch.nn.Module, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Normalize keys to avoid wrappers like _orig_mod. or module.
    raw_state = {k: v.detach().to("cpu") for k, v in model.state_dict().items()}
    state_cpu = normalize_state_dict_keys(raw_state)
    safe_save_file(state_cpu, str(path), metadata={"format": "pt"})
    return path


__all__ = ["train", "train_step"]
