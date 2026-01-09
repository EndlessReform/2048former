from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler

from .base import Objective


class BinnedEV(Objective):
    name = "binned_ev"

    def prepare_model(self, model: torch.nn.Module, device: torch.device, *, cfg: object, dl_train=None) -> torch.nn.Module:
        return model.to(device)

    def train_step(
        self,
        model: torch.nn.Module,
        batch: Dict[str, torch.Tensor],
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        *,
        cfg: object,
        grad_scaler: Optional[GradScaler] = None,
        zero_grad: bool = True,
        optimizer_step: bool = True,
        loss_scale: float = 1.0,
    ) -> Dict[str, float | list[float] | None]:
        tokens = batch["tokens"].to(device, non_blocking=True)
        branch_mask = batch["branch_mask"].to(device, non_blocking=True)
        targets_bins = batch["branch_bin_targets"].to(device, non_blocking=True)

        if zero_grad:
            optimizer.zero_grad(set_to_none=True)

        if device.type == "cuda":
            autocast = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        else:
            class _Null:
                def __enter__(self):
                    return None
                def __exit__(self, *args):
                    return False
            autocast = _Null()

        with autocast:
            _hs, head_out = model(tokens)
            per_head_losses: list[torch.Tensor] = []
            for h in range(4):
                logits_h = head_out[h].float()
                tgt_h = targets_bins[:, h]
                mask_h = branch_mask[:, h]
                loss_h = F.cross_entropy(logits_h, tgt_h, reduction="none")
                loss_h = loss_h[mask_h].mean() if mask_h.any() else torch.zeros((), device=logits_h.device, dtype=torch.float32)
                per_head_losses.append(loss_h)
            loss = sum(per_head_losses)

        scaled_loss = loss * float(loss_scale)
        if grad_scaler is not None:
            grad_scaler.scale(scaled_loss).backward()
            if optimizer_step:
                if cfg.hyperparameters.grad_clip_norm is not None:
                    grad_scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.hyperparameters.grad_clip_norm)
                grad_scaler.step(optimizer)
                grad_scaler.update()
        else:
            scaled_loss.backward()
            if optimizer_step:
                if cfg.hyperparameters.grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.hyperparameters.grad_clip_norm)
                optimizer.step()

        head_losses = [float(l.detach().item()) for l in per_head_losses]
        return {"loss": float(loss.detach().item()), "head_losses": head_losses, "policy_accuracy": None, "policy_agreement": None}

    @torch.no_grad()
    def evaluate(self, model: torch.nn.Module, dl_val: DataLoader, device: torch.device) -> Dict[str, float | list[float] | None]:
        was_training = model.training
        model.eval()

        total_loss = 0.0
        total_heads = torch.zeros(4, dtype=torch.float64)
        n_batches = 0

        if device.type == "cuda":
            autocast = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        else:
            class _Null:
                def __enter__(self):
                    return None
                def __exit__(self, *args):
                    return False
            autocast = _Null()

        for batch in dl_val:
            tokens = batch["tokens"].to(device, non_blocking=True)
            branch_mask = batch["branch_mask"].to(device, non_blocking=True)
            targets_bins = batch["branch_bin_targets"].to(device, non_blocking=True)

            with autocast:
                _hs, head_out = model(tokens)
                per_head_losses: list[torch.Tensor] = []
                for h in range(4):
                    logits_h = head_out[h].float()
                    tgt_h = targets_bins[:, h]
                    mask_h = branch_mask[:, h]
                    loss_h = F.cross_entropy(logits_h, tgt_h, reduction="none")
                    loss_h = loss_h[mask_h].mean() if mask_h.any() else torch.zeros((), device=logits_h.device, dtype=torch.float32)
                    per_head_losses.append(loss_h)
                loss = sum(per_head_losses)

            total_loss += float(loss.detach().item())
            total_heads += torch.tensor([lh.detach().item() for lh in per_head_losses], dtype=torch.float64)
            n_batches += 1

        if was_training:
            model.train()

        if n_batches == 0:
            return {"loss": 0.0, "head_losses": [0.0, 0.0, 0.0, 0.0], "policy_accuracy": None, "policy_agreement": None}

        avg_loss = float(total_loss / n_batches)
        avg_heads = (total_heads / n_batches).tolist()
        return {"loss": avg_loss, "head_losses": avg_heads, "policy_accuracy": None, "policy_agreement": None}


__all__ = ["BinnedEV"]
