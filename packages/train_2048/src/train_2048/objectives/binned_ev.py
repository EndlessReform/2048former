from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .base import Objective
from .utils import unpack_model_outputs
from ..config import ValueTrainingConfig


class BinnedEV(Objective):
    name = "binned_ev"

    def __init__(self) -> None:
        self.value_cfg: Optional[ValueTrainingConfig] = None

    def _freeze_module_params(self, module: Optional[torch.nn.Module]) -> None:
        if module is None:
            return
        if isinstance(module, torch.nn.ModuleList):
            for m in module:
                for p in m.parameters():
                    p.requires_grad = False
        else:
            for p in module.parameters():
                p.requires_grad = False

    def prepare_model(self, model: torch.nn.Module, device: torch.device, *, cfg: object, dl_train=None) -> torch.nn.Module:
        self.value_cfg = getattr(cfg, "value_training", None)
        if self.value_cfg and getattr(self.value_cfg, "enabled", False):
            if getattr(self.value_cfg, "objective", "mse") != "mse":
                raise ValueError(f"Unsupported value objective {self.value_cfg.objective}; only 'mse' is wired for training")
            if getattr(model, "value_head", None) is None:
                raise ValueError("value_training.enabled but model has no value head")
            if getattr(self.value_cfg, "freeze_trunk", False):
                self._freeze_module_params(getattr(model, "tok_emb", None))
                self._freeze_module_params(getattr(model, "pos_emb", None))
                self._freeze_module_params(getattr(model, "blocks", None))
                self._freeze_module_params(getattr(model, "final_ln", None))
            if getattr(self.value_cfg, "effective_policy_weight", None):
                if float(self.value_cfg.effective_policy_weight()) == 0.0:
                    self._freeze_module_params(getattr(model, "ev_heads", None))
                    self._freeze_module_params(getattr(model, "policy_head", None))
        return model.to(device)

    def train_step(
        self,
        model: torch.nn.Module,
        batch: Dict[str, torch.Tensor],
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        *,
        cfg: object,
        zero_grad: bool = True,
        optimizer_step: bool = True,
        loss_scale: float = 1.0,
    ) -> Dict[str, float | list[float] | None]:
        tokens = batch["tokens"].to(device, non_blocking=True)
        branch_mask = batch["branch_mask"].to(device, non_blocking=True)
        targets_bins = batch["branch_bin_targets"].to(device, non_blocking=True)
        value_cfg = self.value_cfg or getattr(cfg, "value_training", None)
        value_enabled = bool(value_cfg and getattr(value_cfg, "enabled", False))
        policy_weight = float(value_cfg.effective_policy_weight()) if value_cfg else 1.0
        value_weight = float(getattr(value_cfg, "loss_weight", 0.0)) if value_cfg else 0.0
        value_scale = float(getattr(value_cfg, "value_loss_policy_scale", 1.0)) if value_cfg else 1.0

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
            _hs, head_out, value_out = unpack_model_outputs(model(tokens))
            per_head_losses: list[torch.Tensor] = []
            for h in range(4):
                logits_h = head_out[h].float()
                tgt_h = targets_bins[:, h]
                mask_h = branch_mask[:, h]
                loss_h = F.cross_entropy(logits_h, tgt_h, reduction="none")
                loss_h = loss_h[mask_h].mean() if mask_h.any() else torch.zeros((), device=logits_h.device, dtype=torch.float32)
                per_head_losses.append(loss_h)
            policy_loss = sum(per_head_losses)
            value_loss_tensor: torch.Tensor | None = None
            if value_enabled:
                if "value_targets" not in batch:
                    raise KeyError("value_targets missing from batch despite value training being enabled")
                if value_out is None:
                    raise ValueError("Model did not return a value head output while value_training.enabled")
                value_targets = batch["value_targets"].to(device, non_blocking=True).float()
                value_pred = value_out.float()
                if value_pred.shape != value_targets.shape:
                    raise ValueError(
                        f"value target shape {tuple(value_targets.shape)} does not match predictions {tuple(value_pred.shape)}"
                    )
                value_loss_tensor = F.mse_loss(value_pred, value_targets, reduction="mean")
            loss = policy_loss * float(policy_weight)
            if value_loss_tensor is not None:
                loss = loss + value_loss_tensor * float(value_weight * value_scale)

        scaled_loss = loss * float(loss_scale)
        scaled_loss.backward()
        if optimizer_step:
            if cfg.hyperparameters.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.hyperparameters.grad_clip_norm)
            optimizer.step()

        head_losses = [float(l.detach().item()) for l in per_head_losses]
        metrics: Dict[str, float | list[float] | None] = {
            "loss": float(loss.detach().item()),
            "head_losses": head_losses,
            "policy_loss": float(policy_loss.detach().item()),
            "policy_accuracy": None,
            "policy_agreement": None,
            "value_loss": float(value_loss_tensor.detach().item()) if value_loss_tensor is not None else None,
        }
        return metrics

    @torch.no_grad()
    def evaluate(self, model: torch.nn.Module, dl_val: DataLoader, device: torch.device) -> Dict[str, float | list[float] | None]:
        was_training = model.training
        model.eval()

        total_loss = 0.0
        total_heads = torch.zeros(4, dtype=torch.float64)
        total_policy_loss = 0.0
        total_value_loss = 0.0
        n_batches = 0
        value_batches = 0
        value_cfg = self.value_cfg
        value_enabled = bool(value_cfg and getattr(value_cfg, "enabled", False))
        policy_weight = float(value_cfg.effective_policy_weight()) if value_cfg else 1.0
        value_weight = float(getattr(value_cfg, "loss_weight", 0.0)) if value_cfg else 0.0
        value_scale = float(getattr(value_cfg, "value_loss_policy_scale", 1.0)) if value_cfg else 1.0

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
                _hs, head_out, value_out = unpack_model_outputs(model(tokens))
                per_head_losses: list[torch.Tensor] = []
                for h in range(4):
                    logits_h = head_out[h].float()
                    tgt_h = targets_bins[:, h]
                    mask_h = branch_mask[:, h]
                    loss_h = F.cross_entropy(logits_h, tgt_h, reduction="none")
                    loss_h = loss_h[mask_h].mean() if mask_h.any() else torch.zeros((), device=logits_h.device, dtype=torch.float32)
                    per_head_losses.append(loss_h)
                policy_loss = sum(per_head_losses)
                value_loss_tensor: torch.Tensor | None = None
                if value_enabled:
                    if "value_targets" not in batch:
                        raise KeyError("value_targets missing from batch despite value training being enabled")
                    if value_out is None:
                        raise ValueError("Model did not return a value head output while value_training.enabled")
                    value_targets = batch["value_targets"].to(device, non_blocking=True).float()
                    value_pred = value_out.float()
                    if value_pred.shape != value_targets.shape:
                        raise ValueError(
                            f"value target shape {tuple(value_targets.shape)} does not match predictions {tuple(value_pred.shape)}"
                        )
                    value_loss_tensor = F.mse_loss(value_pred, value_targets, reduction="mean")
                loss = policy_loss * float(policy_weight)
                if value_loss_tensor is not None:
                    loss = loss + value_loss_tensor * float(value_weight * value_scale)

            total_loss += float(loss.detach().item())
            total_heads += torch.tensor([lh.detach().item() for lh in per_head_losses], dtype=torch.float64)
            total_policy_loss += float(policy_loss.detach().item())
            if value_loss_tensor is not None:
                total_value_loss += float(value_loss_tensor.detach().item())
                value_batches += 1
            n_batches += 1

        if was_training:
            model.train()

        if n_batches == 0:
            return {
                "loss": 0.0,
                "head_losses": [0.0, 0.0, 0.0, 0.0],
                "policy_accuracy": None,
                "policy_agreement": None,
                "value_loss": None,
                "policy_loss": None,
            }

        avg_loss = float(total_loss / n_batches)
        avg_heads = (total_heads / n_batches).tolist()
        avg_value_loss = None
        if value_enabled and value_batches > 0:
            avg_value_loss = float(total_value_loss / value_batches)
        return {
            "loss": avg_loss,
            "head_losses": avg_heads,
            "policy_accuracy": None,
            "policy_agreement": None,
            "value_loss": avg_value_loss,
            "policy_loss": float(total_policy_loss / n_batches),
        }


__all__ = ["BinnedEV"]
