from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .base import Objective


class HardMove(Objective):
    name = "hard_move"

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
        zero_grad: bool = True,
        optimizer_step: bool = True,
        loss_scale: float = 1.0,
    ) -> Dict[str, float | list[float] | None]:
        tokens = batch["tokens"].to(device, non_blocking=True)
        move_targets = batch["move_targets"].to(device, non_blocking=True)
        branch_mask = batch.get("branch_mask")
        if branch_mask is not None:
            branch_mask = branch_mask.to(device, non_blocking=True)

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
            if isinstance(head_out, (list, tuple)):
                if not all(t.shape[-1] == 1 for t in head_out):
                    raise RuntimeError("hard_move expects single policy head or 4x1 logits list")
                logits = torch.stack([t.float().squeeze(-1) for t in head_out], dim=1)
            else:
                logits = head_out.float()

            loss_per_sample = F.cross_entropy(logits, move_targets, reduction="none")
            if branch_mask is not None and branch_mask.numel() == move_targets.numel() * 4:
                # Collate already standardizes legality to UDLR; use as-is
                chosen_legal = branch_mask[torch.arange(move_targets.size(0), device=device), move_targets]
                loss = loss_per_sample[chosen_legal].mean() if bool(chosen_legal.any()) else torch.zeros((), device=logits.device, dtype=torch.float32)
            else:
                loss = loss_per_sample.mean()

        scaled_loss = loss * float(loss_scale)
        scaled_loss.backward()
        if optimizer_step:
            if cfg.hyperparameters.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.hyperparameters.grad_clip_norm)
            optimizer.step()

        preds = logits.argmax(dim=1)
        if branch_mask is not None and branch_mask.numel() == move_targets.numel() * 4:
            acc_mask = branch_mask[torch.arange(move_targets.size(0), device=device), move_targets]
            if bool(acc_mask.any()):
                policy_accuracy = ((preds == move_targets) & acc_mask).float()[acc_mask].mean()
            else:
                policy_accuracy = torch.zeros((), device=logits.device, dtype=torch.float32)
        else:
            policy_accuracy = (preds == move_targets).float().mean()

        probs = F.softmax(logits, dim=-1)
        p_t = probs[torch.arange(move_targets.size(0), device=device), move_targets]
        if branch_mask is not None and branch_mask.numel() == move_targets.numel() * 4:
            if 'acc_mask' in locals() and bool(acc_mask.any()):
                policy_agreement = float(p_t[acc_mask].mean().detach().item())
            else:
                policy_agreement = None
        else:
            policy_agreement = float(p_t.mean().detach().item())

        # Per-head decomposition of loss for logging
        head_losses: list[torch.Tensor] = []
        for h in range(4):
            sel = move_targets == h
            if sel.any():
                head_losses.append(loss_per_sample[sel].mean())
            else:
                head_losses.append(torch.zeros((), device=logits.device, dtype=torch.float32))

        return {
            "loss": float(loss.detach().item()),
            "head_losses": [float(x.detach().item()) for x in head_losses],
            "policy_accuracy": float(policy_accuracy.detach().item()),
            "policy_agreement": policy_agreement,
        }

    @torch.no_grad()
    def evaluate(self, model: torch.nn.Module, dl_val: DataLoader, device: torch.device) -> Dict[str, float | list[float] | None]:
        was_training = model.training
        model.eval()

        total_loss = 0.0
        total_heads = torch.zeros(4, dtype=torch.float64)
        n_batches = 0
        total_correct = 0.0
        total_examples = 0
        agree_sum = 0.0
        agree_cnt = 0

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
            move_targets = batch["move_targets"].to(device, non_blocking=True)
            branch_mask = batch.get("branch_mask")
            if branch_mask is not None:
                branch_mask = branch_mask.to(device, non_blocking=True)

            with autocast:
                _hs, head_out = model(tokens)
                if isinstance(head_out, (list, tuple)):
                    if not all(t.shape[-1] == 1 for t in head_out):
                        raise RuntimeError("hard_move expects single policy head or 4x1 logits list")
                    logits = torch.stack([t.float().squeeze(-1) for t in head_out], dim=1)
                else:
                    logits = head_out.float()
                loss_per_sample = F.cross_entropy(logits, move_targets, reduction="none")

                if branch_mask is not None and branch_mask.numel() == move_targets.numel() * 4:
                    chosen_legal = branch_mask[torch.arange(move_targets.size(0), device=device), move_targets]
                    loss = loss_per_sample[chosen_legal].mean() if bool(chosen_legal.any()) else torch.zeros((), device=logits.device, dtype=torch.float32)
                else:
                    loss = loss_per_sample.mean()

                preds = logits.argmax(dim=1)
                if branch_mask is not None and branch_mask.numel() == move_targets.numel() * 4:
                    acc_mask = branch_mask[torch.arange(move_targets.size(0), device=device), move_targets]
                    total_correct += float(((preds == move_targets) & acc_mask).sum().item())
                    total_examples += int(acc_mask.sum().item())
                else:
                    total_correct += float((preds == move_targets).sum().item())
                    total_examples += int(move_targets.numel())

                probs = F.softmax(logits, dim=-1)
                p_t = probs[torch.arange(move_targets.size(0), device=device), move_targets]
                if branch_mask is not None and branch_mask.numel() == move_targets.numel() * 4:
                    if bool(acc_mask.any()):
                        agree_sum += float(p_t[acc_mask].sum().item())
                        agree_cnt += int(acc_mask.sum().item())
                else:
                    agree_sum += float(p_t.sum().item())
                    agree_cnt += int(move_targets.numel())

                head_losses: list[torch.Tensor] = []
                for h in range(4):
                    sel = move_targets == h
                    if sel.any():
                        head_losses.append(loss_per_sample[sel].mean())
                    else:
                        head_losses.append(torch.zeros((), device=logits.device, dtype=torch.float32))

            total_loss += float(loss.detach().item())
            total_heads += torch.tensor([lh.detach().item() for lh in head_losses], dtype=torch.float64)
            n_batches += 1

        if was_training:
            model.train()

        if n_batches == 0:
            return {"loss": 0.0, "head_losses": [0.0, 0.0, 0.0, 0.0], "policy_accuracy": None, "policy_agreement": None}

        avg_loss = float(total_loss / n_batches)
        avg_heads = (total_heads / n_batches).tolist()
        policy_accuracy = float(total_correct / total_examples) if total_examples > 0 else None
        policy_agreement = (agree_sum / agree_cnt) if (agree_cnt > 0) else None
        return {"loss": avg_loss, "head_losses": avg_heads, "policy_accuracy": policy_accuracy, "policy_agreement": policy_agreement}


__all__ = ["HardMove"]
