from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler

from .training_loop import run_training as train
from .objectives import make_objective
from .checkpointing import save_safetensors as _save_checkpoint
from .config import TrainingConfig


def train_step(
    model: torch.nn.Module,
    batch: Dict[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    target_mode: str,
    *,
    cfg: TrainingConfig,
    grad_scaler: Optional[GradScaler] = None,
):
    """Dispatcher that forwards to the concrete objective's train_step."""
    obj = make_objective(target_mode)
    return obj.train_step(model, batch, optimizer, device, cfg=cfg, grad_scaler=grad_scaler)


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    dl_val: DataLoader,
    device: torch.device,
    target_mode: str,
):
    obj = make_objective(target_mode)
    return obj.evaluate(model, dl_val, device)


__all__ = ["train", "train_step", "evaluate"]
