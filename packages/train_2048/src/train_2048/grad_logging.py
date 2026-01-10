from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from pathlib import Path
from typing import Iterable

import torch


@dataclass
class GradNormStats:
    mean: float
    std: float
    p95: float
    max: float
    global_norm: float
    count: int


@torch.no_grad()
def compute_grad_norm_stats(model: torch.nn.Module) -> GradNormStats | None:
    norms: list[float] = []
    total_sq = 0.0
    for _name, param in model.named_parameters():
        grad = param.grad
        if grad is None:
            continue
        grad = grad.detach()
        if grad.is_sparse:
            grad = grad.coalesce()._values()
        grad_f = grad.float()
        norms.append(float(torch.linalg.vector_norm(grad_f).item()))
        total_sq += float((grad_f * grad_f).sum().item())
    if not norms:
        return None
    norms_t = torch.tensor(norms, dtype=torch.float32)
    mean = float(norms_t.mean().item())
    std = float(norms_t.std(unbiased=False).item()) if len(norms) > 1 else 0.0
    p95 = float(torch.quantile(norms_t, 0.95).item())
    max_val = float(norms_t.max().item())
    global_norm = sqrt(total_sq)
    return GradNormStats(
        mean=mean,
        std=std,
        p95=p95,
        max=max_val,
        global_norm=global_norm,
        count=len(norms),
    )


def dump_named_grads(
    model: torch.nn.Module,
    names: Iterable[str],
    *,
    step: int,
    out_dir: Path,
) -> Path | None:
    name_set = set(names)
    if not name_set:
        return None
    grads: dict[str, torch.Tensor] = {}
    for name, param in model.named_parameters():
        if name not in name_set:
            continue
        grad = param.grad
        if grad is None:
            continue
        grads[name] = grad.detach().cpu()
    if not grads:
        return None
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"grads-step-{step:08d}.pt"
    payload = {"step": int(step), "grads": grads}
    torch.save(payload, str(path))
    return path


__all__ = ["GradNormStats", "compute_grad_norm_stats", "dump_named_grads"]
