from __future__ import annotations

from typing import Dict, Optional, Protocol

import torch
from torch.utils.data import DataLoader


class Objective(Protocol):
    """Protocol for a training objective.

    Implementations encapsulate loss/metrics for train and eval, and may
    optionally prepare the model (e.g., resize heads).
    """

    name: str

    def prepare_model(
        self, model: torch.nn.Module, device: torch.device, *, cfg: object, dl_train: Optional[DataLoader]
    ) -> torch.nn.Module:
        """Optionally mutate/resize the model for the objective and return it."""

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
        """Perform one optimization step and return metrics."""

    @torch.no_grad()
    def evaluate(
        self,
        model: torch.nn.Module,
        dl_val: DataLoader,
        device: torch.device,
    ) -> Dict[str, float | list[float] | None]:
        """Evaluate the model over a validation loader and return metrics."""


__all__ = ["Objective"]
