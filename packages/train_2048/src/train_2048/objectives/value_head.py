from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .base import Objective
from ..config import ValueHeadConfig


@dataclass
class _HeadOutputs:
    logits: torch.Tensor
    board_repr: torch.Tensor


def _build_value_head(hidden_size: int, out_dim: int, cfg: ValueHeadConfig) -> nn.Module:
    head_type = cfg.head_type
    if head_type == "probe":
        return nn.Linear(hidden_size, out_dim)
    if head_type == "mlp":
        layers: list[nn.Module] = [
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, cfg.mlp_hidden_dim),
            nn.GELU(),
        ]
        if cfg.mlp_dropout > 0.0:
            layers.append(nn.Dropout(cfg.mlp_dropout))
        layers.append(nn.Linear(cfg.mlp_hidden_dim, out_dim))
        return nn.Sequential(*layers)
    raise ValueError(f"Unsupported value head type: {head_type}")


def _forward_value_head(
    model: torch.nn.Module,
    head: nn.Module,
    tokens: torch.Tensor,
) -> _HeadOutputs:
    hidden_states, _ = model(tokens)
    board_repr = hidden_states.mean(dim=1)
    logits = head(board_repr)
    return _HeadOutputs(logits=logits, board_repr=board_repr)


class _ValueHeadBase(Objective):
    name = "value_head"

    def __init__(self, mode: str) -> None:
        self.mode = mode
        self.thresholds: Optional[torch.Tensor] = None

    def prepare_model(
        self,
        model: torch.nn.Module,
        device: torch.device,
        *,
        cfg: object,
        dl_train=None,
    ) -> torch.nn.Module:
        vh_cfg: ValueHeadConfig
        if hasattr(cfg, "value_head"):
            vh_cfg = getattr(cfg, "value_head")  # type: ignore[assignment]
        else:
            vh_cfg = ValueHeadConfig()

        hidden_size = int(getattr(model.config, "hidden_size"))  # type: ignore[attr-defined]
        if self.mode == "value_ordinal":
            out_dim = len(vh_cfg.tile_thresholds)
        else:
            out_dim = len(vh_cfg.tile_thresholds) + 1

        if vh_cfg.freeze_trunk:
            for param in model.parameters():
                param.requires_grad = False

        head = _build_value_head(hidden_size, out_dim, vh_cfg)
        head.to(device=device)
        model.value_head = head  # type: ignore[attr-defined]

        for param in head.parameters():
            param.requires_grad = True

        self.thresholds = torch.tensor(vh_cfg.tile_thresholds, dtype=torch.float32, device=device)
        self._value_cfg = vh_cfg
        return model

    def _autocast(self, device: torch.device):
        if device.type == "cuda":
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)

        class _Null:
            def __enter__(self):
                return None

            def __exit__(self, *args):
                return False

        return _Null()

    def _require_head(self, model: torch.nn.Module) -> nn.Module:
        head = getattr(model, "value_head", None)
        if head is None:
            raise AttributeError("Model is missing value_head; call prepare_model first")
        return head


class ValueOrdinal(_ValueHeadBase):
    name = "value_ordinal"

    def __init__(self) -> None:
        super().__init__(mode="value_ordinal")

    def train_step(
        self,
        model: torch.nn.Module,
        batch: Dict[str, torch.Tensor],
        optimizer: torch.optim.Optimizer,
        device: torch.device,
    ) -> Dict[str, float | list[float] | None]:
        tokens = batch["tokens"].to(device, non_blocking=True)
        targets = batch["value_targets_bce"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        head = self._require_head(model)
        with self._autocast(device):
            outputs = _forward_value_head(model, head, tokens)
            logits = outputs.logits.float()
            loss_matrix = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
            loss = loss_matrix.mean()
            per_threshold = loss_matrix.mean(dim=0)

        loss.backward()
        optimizer.step()

        metrics = {
            "loss": float(loss.detach().item()),
            "value_accuracy": None,
            "value_bce_per_threshold": [float(x.detach().item()) for x in per_threshold],
            "value_bce_high": float(per_threshold[-1].detach().item()) if per_threshold.numel() else float(loss.detach().item()),
            "policy_accuracy": None,
            "policy_agreement": None,
        }
        return metrics

    @torch.no_grad()
    def evaluate(
        self,
        model: torch.nn.Module,
        dl_val: DataLoader,
        device: torch.device,
    ) -> Dict[str, float | list[float] | None]:
        was_training = model.training
        model.eval()

        head = self._require_head(model)
        total_loss = 0.0
        total_threshold = None
        n_batches = 0

        for batch in dl_val:
            tokens = batch["tokens"].to(device, non_blocking=True)
            targets = batch["value_targets_bce"].to(device, non_blocking=True)

            with self._autocast(device):
                outputs = _forward_value_head(model, head, tokens)
                logits = outputs.logits.float()
                loss_matrix = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
                loss = loss_matrix.mean()
                per_threshold = loss_matrix.mean(dim=0)

            total_loss += float(loss.detach().item())
            if total_threshold is None:
                total_threshold = per_threshold.detach().double()
            else:
                total_threshold += per_threshold.detach().double()
            n_batches += 1

        if was_training:
            model.train()

        if n_batches == 0 or total_threshold is None:
            return {
                "loss": 0.0,
                "value_accuracy": None,
                "value_bce_per_threshold": [],
                "value_bce_high": 0.0,
                "policy_accuracy": None,
                "policy_agreement": None,
            }

        avg_loss = total_loss / n_batches
        avg_threshold = (total_threshold / n_batches).cpu().tolist()
        value_bce_high = avg_threshold[-1] if avg_threshold else avg_loss
        return {
            "loss": float(avg_loss),
            "value_accuracy": None,
            "value_bce_per_threshold": avg_threshold,
            "value_bce_high": float(value_bce_high),
            "policy_accuracy": None,
            "policy_agreement": None,
        }


class ValueCategorical(_ValueHeadBase):
    name = "value_categorical"

    def __init__(self) -> None:
        super().__init__(mode="value_categorical")

    def train_step(
        self,
        model: torch.nn.Module,
        batch: Dict[str, torch.Tensor],
        optimizer: torch.optim.Optimizer,
        device: torch.device,
    ) -> Dict[str, float | list[float] | None]:
        tokens = batch["tokens"].to(device, non_blocking=True)
        targets = batch["value_targets_ce"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        head = self._require_head(model)
        with self._autocast(device):
            outputs = _forward_value_head(model, head, tokens)
            logits = outputs.logits.float()
            loss = F.cross_entropy(logits, targets, reduction="mean")

        loss.backward()
        optimizer.step()

        preds = logits.argmax(dim=1)
        accuracy = (preds == targets).float().mean()

        metrics = {
            "loss": float(loss.detach().item()),
            "value_accuracy": float(accuracy.detach().item()),
            "value_bce_per_threshold": None,
            "value_bce_high": None,
            "policy_accuracy": None,
            "policy_agreement": None,
        }
        return metrics

    @torch.no_grad()
    def evaluate(
        self,
        model: torch.nn.Module,
        dl_val: DataLoader,
        device: torch.device,
    ) -> Dict[str, float | list[float] | None]:
        was_training = model.training
        model.eval()

        head = self._require_head(model)
        total_loss = 0.0
        total_correct = 0.0
        total_examples = 0
        n_batches = 0

        for batch in dl_val:
            tokens = batch["tokens"].to(device, non_blocking=True)
            targets = batch["value_targets_ce"].to(device, non_blocking=True)

            with self._autocast(device):
                outputs = _forward_value_head(model, head, tokens)
                logits = outputs.logits.float()
                loss = F.cross_entropy(logits, targets, reduction="mean")

            preds = logits.argmax(dim=1)
            total_loss += float(loss.detach().item())
            total_correct += float((preds == targets).sum().detach().item())
            total_examples += int(targets.numel())
            n_batches += 1

        if was_training:
            model.train()

        if n_batches == 0 or total_examples == 0:
            return {
                "loss": 0.0,
                "value_accuracy": None,
                "value_bce_per_threshold": None,
                "value_bce_high": None,
                "policy_accuracy": None,
                "policy_agreement": None,
            }

        avg_loss = total_loss / n_batches
        accuracy = total_correct / total_examples
        return {
            "loss": float(avg_loss),
            "value_accuracy": float(accuracy),
            "value_bce_per_threshold": None,
            "value_bce_high": None,
            "policy_accuracy": None,
            "policy_agreement": None,
        }


__all__ = ["ValueOrdinal", "ValueCategorical"]
